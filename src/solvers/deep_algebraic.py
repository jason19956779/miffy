"""Tier 3 deep algebraic RSPC+EDC solver plugins."""

import random
from collections import Counter, defaultdict

from src.core import rspc
from src.solvers import SolverRegistry
from src.solvers.base import SectorSolverPlugin
from src.solvers.portfolio import clean_single_pass, subgrid_peel_decode


def assemble_bipartite_core(
    lba: int,
    raw_passes: list[tuple[bytes, bytes]],
    prefer_p: bool = True,
) -> tuple[bytearray, set[int]]:
    """Construct maximum zero-conflict P/Q core across passes + core-injected hybrid harvesting."""
    hdr4 = rspc.lba_header(lba)
    passes = [bytearray(rp[0]) for rp in raw_passes]
    c2_sets = [rspc.c2_set(rp[1]) for rp in raw_passes]

    sec_out = bytearray(2352)
    rspc.init_sector_header(lba, sec_out)
    known_all: set[int] = set(rspc.FIXED_POS)

    for plane in (0, 1):
        p_cands: dict[int, Counter] = defaultdict(Counter)
        q_cands: dict[int, Counter] = defaultdict(Counter)

        for (raw_sec, c2_b), c2 in zip(raw_passes, c2_sets):
            _, s_clean, _ = clean_single_pass(lba, raw_sec, c2_b)
            for w_mode, s_base in ((4, bytearray(raw_sec)), (1, s_clean)):
                s = bytearray(s_base)
                s[0:12] = rspc.SYNC_12
                s[12:16] = hdr4
                s[2068:2076] = b"\x00" * 8
                for c in range(43):
                    cw = rspc.P_IDX[2 * c + plane]
                    if rspc.syndromes_p(s, cw) == (0, 0):
                        hits = sum(1 for pos in cw if pos in c2)
                        p_cands[c][bytes(s[p] for p in cw)] += (30 if hits == 0 else (6 if hits == 1 else 2)) * w_mode
                    else:
                        eras = [i for i, pos in enumerate(cw) if pos in c2]
                        if 1 <= len(eras) <= 2:
                            ok, corrs = rspc.solve_2eras(s, cw, eras)
                            if ok:
                                st = bytearray(s)
                                for pos, val in corrs:
                                    st[pos] = val
                                if rspc.syndromes_p(st, cw) == (0, 0):
                                    p_cands[c][bytes(st[p] for p in cw)] += 2
                for q in range(26):
                    cw = rspc.Q_IDX[2 * q + plane]
                    if rspc.syndromes_q(s, cw) == (0, 0):
                        hits = sum(1 for pos in cw if pos in c2)
                        q_cands[q][bytes(s[p] for p in cw)] += (30 if hits == 0 else (6 if hits == 1 else 2)) * w_mode
                    else:
                        eras = [i for i, pos in enumerate(cw) if pos in c2]
                        if 1 <= len(eras) <= 2:
                            ok, corrs = rspc.solve_2eras(s, cw, eras)
                            if ok:
                                st = bytearray(s)
                                for pos, val in corrs:
                                    st[pos] = val
                                if rspc.syndromes_q(st, cw) == (0, 0):
                                    q_cands[q][bytes(st[p] for p in cw)] += 2

        best_p = {c: p_cands[c].most_common(1)[0][0] for c in range(43) if p_cands[c]}
        best_q = {q: q_cands[q].most_common(1)[0][0] for q in range(26) if q_cands[q]}

        for _ in range(6):
            # Prune conflicting P/Q codewords
            for _ in range(60):
                for c in list(best_p.keys()):
                    if any(b_q[c] != best_p[c][(q + c) % 26] for q, b_q in best_q.items()):
                        for b_p, _ in p_cands[c].most_common():
                            if all(b_q[c] == b_p[(q + c) % 26] for q, b_q in best_q.items()):
                                best_p[c] = b_p
                                break
                for q in list(best_q.keys()):
                    if any(best_q[q][c] != b_p[(q + c) % 26] for c, b_p in best_p.items()):
                        for b_q, _ in q_cands[q].most_common():
                            if all(b_q[c] == b_p[(q + c) % 26] for c, b_p in best_p.items()):
                                best_q[q] = b_q
                                break
                p_conf = {c: sum(1 for q, b_q in best_q.items() if b_q[c] != b_p[(q + c) % 26]) for c, b_p in best_p.items()}
                q_conf = {q: sum(1 for c, b_p in best_p.items() if b_q[c] != b_p[(q + c) % 26]) for q, b_q in best_q.items()}
                mpc = max(p_conf.values(), default=0)
                mqc = max(q_conf.values(), default=0)
                if max(mpc, mqc) == 0:
                    break
                wp = max(p_conf, key=lambda c: (p_conf[c], -p_cands[c][best_p[c]])) if p_conf else None
                wq = max(q_conf, key=lambda q: (q_conf[q], -q_cands[q][best_q[q]])) if q_conf else None
                if mpc > mqc or (mpc == mqc and not prefer_p and wp is not None):
                    del best_p[wp]
                elif wq is not None:
                    del best_q[wq]
                elif wp is not None:
                    del best_p[wp]

            known_pl = set(rspc.FIXED_POS)
            core = bytearray(2352)
            rspc.init_sector_header(lba, core)
            for c, b_p in best_p.items():
                for i, pos in enumerate(rspc.P_IDX[2 * c + plane]):
                    core[pos] = b_p[i]
                    known_pl.add(pos)
            for q, b_q in best_q.items():
                for i, pos in enumerate(rspc.Q_IDX[2 * q + plane]):
                    core[pos] = b_q[i]
                    known_pl.add(pos)

            # Inject core into every raw pass to harvest newly solvable codewords
            added = False
            for s_raw, c2 in zip(passes, c2_sets):
                sh = bytearray(s_raw)
                for pos in known_pl:
                    sh[pos] = core[pos]
                for c in range(43):
                    if c in best_p:
                        continue
                    cw = rspc.P_IDX[2 * c + plane]
                    un = [i for i, pos in enumerate(cw) if pos not in known_pl]
                    if 1 <= len(un) <= 2:
                        ok, corrs = rspc.solve_2eras(sh, cw, un)
                        if ok:
                            st = bytearray(sh)
                            for pos, val in corrs:
                                st[pos] = val
                            if rspc.syndromes_p(st, cw) == (0, 0):
                                b_p = bytes(st[p] for p in cw)
                                if all(b_q[c] == b_p[(q + c) % 26] for q, b_q in best_q.items()):
                                    p_cands[c][b_p] += 60
                                    added = True
                for q in range(26):
                    if q in best_q:
                        continue
                    cw = rspc.Q_IDX[2 * q + plane]
                    un = [i for i, pos in enumerate(cw) if pos not in known_pl]
                    if 1 <= len(un) <= 2:
                        ok, corrs = rspc.solve_2eras(sh, cw, un)
                        if ok:
                            st = bytearray(sh)
                            for pos, val in corrs:
                                st[pos] = val
                            if rspc.syndromes_q(st, cw) == (0, 0):
                                b_q = bytes(st[p] for p in cw)
                                if all(b_q[c] == b_p[(q + c) % 26] for c, b_p in best_p.items()):
                                    q_cands[q][b_q] += 60
                                    added = True
            if not added:
                break

        for pos in range(16 + plane, 2352, 2):
            if pos in known_pl:
                sec_out[pos] = core[pos]
                known_all.add(pos)
            else:
                cnt = Counter()
                for s_raw, c2 in zip(passes, c2_sets):
                    cnt[s_raw[pos]] += 10 if pos not in c2 else 1
                sec_out[pos] = cnt.most_common(1)[0][0]

    return sec_out, known_all


def gf2_solve(equations: list[tuple[int, int]], n_vars: int) -> tuple[int, int, list[int]]:
    """Binary Gaussian elimination over GF(2) returning `(rank, sol0, null_basis)`."""
    basis: dict[int, tuple[int, int]] = {}
    for m, b in equations:
        while m:
            bit = (m & -m).bit_length() - 1
            if bit in basis:
                bm, br = basis[bit]
                m ^= bm
                b ^= br
            else:
                basis[bit] = (m, b)
                break
        if m == 0 and b != 0:
            return -1, 0, []
    sol = 0
    pivots = sorted(basis.keys(), reverse=True)
    pset = set(pivots)
    for bit in pivots:
        m, b = basis[bit]
        if b ^ ((bin((m ^ (1 << bit)) & sol).count("1")) & 1):
            sol |= 1 << bit
    null_basis = []
    for fv in range(n_vars):
        if fv in pset:
            continue
        vec = 1 << fv
        for bit in pivots:
            m, _ = basis[bit]
            if (bin((m ^ (1 << bit)) & vec).count("1")) & 1:
                vec |= 1 << bit
        null_basis.append(vec)
    return len(pivots), sol, null_basis


@SolverRegistry.register
class SubsetDiversitySubgridSolver(SectorSolverPlugin):
    name = "subset_diversity_subgrid"
    tier = 3
    description = "Speed/direction slices + deterministic random k-of-N subgrid sampling to escape weight-3 traps."

    def solve(self, lba, raw_passes, context=None):
        n = len(raw_passes)
        if n < 8:
            return False, -1, -1, bytearray(2352), "PARTIAL_SUBSET"
        subsets = [
            ("LAST32", raw_passes[-32:] if n > 32 else raw_passes),
            ("FWD_SLICE", raw_passes[::2]),
            ("REV_SLICE", raw_passes[1::2]),
        ]
        rng = random.Random(lba)
        for r_i in range(28):
            k = rng.choice([k0 for k0 in (8, 12, 16, 20, 24, 32) if k0 < n] or [n])
            subsets.append((f"RAND{r_i}_K{k}", rng.sample(raw_passes, k)))

        best = (False, -1, -1, bytearray(2352), "PARTIAL_SUBSET")
        for label, sub in subsets:
            if len(sub) < 2:
                continue
            ok, pz, qz, sec = subgrid_peel_decode(lba, sub, False, False)
            if ok:
                return True, 86, 52, sec, f"DEEP_{label}"
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, sec, f"PARTIAL_DEEP_{label}")
        return best


@SolverRegistry.register
class BipartiteCoreHybridSolver(SectorSolverPlugin):
    name = "bipartite_core_hybrid"
    tier = 3
    description = "K43,26 zero-conflict bipartite core + core-injected hybrid pass harvesting."

    def solve(self, lba, raw_passes, context=None):
        best = (False, -1, -1, bytearray(2352), "PARTIAL_CORE")
        best_known: set[int] | None = None
        for prefer_p in (True, False):
            sec, known = assemble_bipartite_core(lba, raw_passes, prefer_p=prefer_p)
            chk = bytearray(sec)
            if rspc.check_and_fix_q_parity(chk):
                if context is not None:
                    context["known_mask_2048"] = set(range(2048))
                return True, 86, 52, chk, f"BIPARTITE_CORE_P{int(prefer_p)}"
            pz, qz = rspc.count_pq(sec)
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, sec, f"PARTIAL_CORE_P{int(prefer_p)}")
                best_known = {p - 16 for p in known if 16 <= p < 2064}
        if context is not None and best_known is not None:
            context["known_mask_2048"] = best_known
        return best


@SolverRegistry.register
class JointGF2EdcSolver(SectorSolverPlugin):
    name = "joint_gf2_edc"
    tier = 3
    description = "Joint GF(2) linear RSPC P-syndrome + 32-bit EDC Gaussian solver over core residuals."

    def solve(self, lba, raw_passes, context=None):
        best = (False, -1, -1, bytearray(2352), "PARTIAL_GF2")
        c2_sets = [rspc.c2_set(rp[1]) for rp in raw_passes]
        qpar_clean = {}
        for pos in range(2248, 2352):
            vals = [rp[0][pos] for rp, c2 in zip(raw_passes, c2_sets) if pos not in c2] or [rp[0][pos] for rp in raw_passes]
            qpar_clean[pos] = {v for v, c in Counter(vals).items() if c >= 2}

        oracle_fn = (context or {}).get("oracle_fn")
        for prefer_p in (True, False):
            sec_core, known = assemble_bipartite_core(lba, raw_passes, prefer_p=prefer_p)
            un_all = sorted(p for p in range(16, 2248) if p not in known)
            if not un_all or len(un_all) > 70:
                continue

            pin_cands = []
            for pos in un_all:
                cvals = [rp[0][pos] for rp, c2 in zip(raw_passes, c2_sets) if pos not in c2]
                if cvals:
                    val, cnt = Counter(cvals).most_common(1)[0]
                    pin_cands.append((cnt / len(cvals), cnt, pos, val))
                else:
                    val, cnt = Counter(rp[0][pos] for rp in raw_passes).most_common(1)[0]
                    pin_cands.append((0.0, cnt, pos, val))
            pin_cands.sort(reverse=True)

            max_pins = max(0, len(un_all) - 2)
            pin_schedule = [0] + [p for p in range(1, max_pins + 1) if p <= 10 or p % 2 == 0 or len(un_all) - p <= 20]
            for n_pins in pin_schedule:
                sec_base = bytearray(sec_core)
                pinned = set()
                for ratio, _, pos, val in pin_cands[:n_pins]:
                    if n_pins > 0 and ratio < 0.8 and len(un_all) - n_pins > 16:
                        break
                    sec_base[pos] = val
                    pinned.add(pos)
                un_pos = [p for p in un_all if p not in pinned]
                if not un_pos or len(un_pos) > 36:
                    continue

                p_touch = sorted({rspc.BYTE_TO_P[p] for p in un_pos})
                sec0 = bytearray(sec_base)
                for p in un_pos:
                    sec0[p] = 0
                if any(rspc.syndromes_p(sec0, rspc.P_IDX[idx]) != (0, 0) for idx in range(86) if idx not in p_touch):
                    continue

                rhs_bits = []
                for idx in p_touch:
                    s0, s1 = rspc.syndromes_p(sec0, rspc.P_IDX[idx])
                    rhs_bits.extend((s0 >> b) & 1 for b in range(8))
                    rhs_bits.extend((s1 >> b) & 1 for b in range(8))
                edc0 = rspc.edc_residual(sec0[:2068])
                rhs_bits.extend((edc0 >> b) & 1 for b in range(32))

                n_eq = len(rhs_bits)
                col_vecs = [0] * n_eq
                zsec = bytearray(2352)
                for k, pos in enumerate(un_pos):
                    for bit_idx in range(8):
                        var_id = 8 * k + bit_idx
                        zsec[pos] = 1 << bit_idx
                        eq_off = 0
                        for idx in p_touch:
                            s0, s1 = rspc.syndromes_p(zsec, rspc.P_IDX[idx])
                            for b in range(8):
                                if (s0 >> b) & 1:
                                    col_vecs[eq_off + b] |= 1 << var_id
                                if (s1 >> b) & 1:
                                    col_vecs[eq_off + 8 + b] |= 1 << var_id
                            eq_off += 16
                        if pos < 2068:
                            dedc = rspc.edc_residual(zsec[:2068])
                            for b in range(32):
                                if (dedc >> b) & 1:
                                    col_vecs[eq_off + b] |= 1 << var_id
                        zsec[pos] = 0

                rank, sol0, null_basis = gf2_solve(list(zip(col_vecs, rhs_bits)), 8 * len(un_pos))
                if rank == -1 or len(null_basis) > 12:
                    continue

                candidates = []
                for mask in range(1 << len(null_basis)):
                    sol = sol0
                    for bi, nb in enumerate(null_basis):
                        if (mask >> bi) & 1:
                            sol ^= nb
                    cand = bytearray(sec0)
                    for k, pos in enumerate(un_pos):
                        cand[pos] = (sol >> (8 * k)) & 0xFF
                    if not rspc.check_and_fix_q_parity(cand):
                        continue
                    q_match = sum(1 for pos in range(2248, 2352) if cand[pos] in qpar_clean[pos])
                    raw_hits = sum(1 for pos in un_pos for rp in raw_passes if rp[0][pos] == cand[pos])
                    o_score = oracle_fn(bytes(cand[16:2064])) if oracle_fn is not None else 0.0
                    candidates.append((q_match, o_score, raw_hits, cand))

                if candidates:
                    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                    q_match, o_score, _, winner = candidates[0]
                    if q_match >= 98 or (oracle_fn is not None and o_score > -20.0):
                        if context is not None:
                            context["known_mask_2048"] = set(range(2048))
                        return True, 86, 52, winner, f"GF2_P_EDC_PIN{n_pins}_ND{len(null_basis)}"
        return best
