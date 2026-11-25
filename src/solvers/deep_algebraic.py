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

        for _ in range(8):
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

            for _ in range(3):
                for c in range(43):
                    if c not in best_p and p_cands[c]:
                        for b_p, _ in p_cands[c].most_common():
                            if all(b_q[c] == b_p[(q + c) % 26] for q, b_q in best_q.items()):
                                best_p[c] = b_p
                                break
                for q in range(26):
                    if q not in best_q and q_cands[q]:
                        for b_q, _ in q_cands[q].most_common():
                            if all(b_q[c] == b_p[(q + c) % 26] for c, b_p in best_p.items()):
                                best_q[q] = b_q
                                break

            known_pl = set(rspc.FIXED_POS)
            core = bytearray(2352)
            rspc.init_sector_header(lba, core)
            for c, b_p in best_p.items():
                for i, pos in enumerate(rspc.P_IDX[2 * c + plane]):
                    if pos not in rspc.FIXED_POS:
                        core[pos] = b_p[i]
                    known_pl.add(pos)
            for q, b_q in best_q.items():
                for i, pos in enumerate(rspc.Q_IDX[2 * q + plane]):
                    if pos not in rspc.FIXED_POS:
                        core[pos] = b_q[i]
                    known_pl.add(pos)

            # Inject core into every raw pass to harvest newly solvable codewords (including C2 & pair trials)
            added = False
            for s_raw, c2 in zip(passes, c2_sets):
                sh = bytearray(s_raw)
                rspc.init_sector_header(lba, sh)
                for pos in known_pl:
                    if pos not in rspc.FIXED_POS:
                        sh[pos] = core[pos]
                for c in range(43):
                    if c in best_p:
                        continue
                    cw = rspc.P_IDX[2 * c + plane]
                    un = [i for i, pos in enumerate(cw) if pos not in known_pl and pos not in rspc.FIXED_POS]
                    un_c2 = [i for i in un if cw[i] in c2]
                    trials = []
                    if rspc.syndromes_p(sh, cw) == (0, 0):
                        trials.append((bytes(sh[p] for p in cw), 25))
                    if 1 <= len(un) <= 2:
                        ok, corrs = rspc.solve_2eras(sh, cw, un)
                        if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                            st = bytearray(sh)
                            for pos, val in corrs:
                                st[pos] = val
                            if rspc.syndromes_p(st, cw) == (0, 0):
                                trials.append((bytes(st[p] for p in cw), 60))
                    if 1 <= len(un_c2) <= 2 and un_c2 != un:
                        ok, corrs = rspc.solve_2eras(sh, cw, un_c2)
                        if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                            st = bytearray(sh)
                            for pos, val in corrs:
                                st[pos] = val
                            if rspc.syndromes_p(st, cw) == (0, 0):
                                trials.append((bytes(st[p] for p in cw), 15))
                    if len(un) == 3:
                        for keep_i in un:
                            if cw[keep_i] not in c2:
                                ok, corrs = rspc.solve_2eras(sh, cw, [i for i in un if i != keep_i])
                                if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                                    st = bytearray(sh)
                                    for pos, val in corrs:
                                        st[pos] = val
                                    if rspc.syndromes_p(st, cw) == (0, 0):
                                        trials.append((bytes(st[p] for p in cw), 5))
                    for b_p, w in trials:
                        if all(b_q[c] == b_p[(q + c) % 26] for q, b_q in best_q.items()):
                            p_cands[c][b_p] += w
                            added = True
                for q in range(26):
                    if q in best_q:
                        continue
                    cw = rspc.Q_IDX[2 * q + plane]
                    un = [i for i, pos in enumerate(cw) if pos not in known_pl and pos not in rspc.FIXED_POS]
                    un_c2 = [i for i in un if cw[i] in c2]
                    trials = []
                    if rspc.syndromes_q(sh, cw) == (0, 0):
                        trials.append((bytes(sh[p] for p in cw), 25))
                    if 1 <= len(un) <= 2:
                        ok, corrs = rspc.solve_2eras(sh, cw, un)
                        if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                            st = bytearray(sh)
                            for pos, val in corrs:
                                st[pos] = val
                            if rspc.syndromes_q(st, cw) == (0, 0):
                                trials.append((bytes(st[p] for p in cw), 60))
                    if 1 <= len(un_c2) <= 2 and un_c2 != un:
                        ok, corrs = rspc.solve_2eras(sh, cw, un_c2)
                        if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                            st = bytearray(sh)
                            for pos, val in corrs:
                                st[pos] = val
                            if rspc.syndromes_q(st, cw) == (0, 0):
                                trials.append((bytes(st[p] for p in cw), 15))
                    if 3 <= len(un) <= 5:
                        for i1 in range(len(un)):
                            for i2 in range(i1 + 1, len(un)):
                                eras2 = [un[i1], un[i2]]
                                if all(cw[i] not in c2 for i in un if i not in eras2):
                                    ok, corrs = rspc.solve_2eras(sh, cw, eras2)
                                    if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                                        st = bytearray(sh)
                                        for pos, val in corrs:
                                            st[pos] = val
                                        if rspc.syndromes_q(st, cw) == (0, 0):
                                            trials.append((bytes(st[p] for p in cw), 4))
                    for b_q, w in trials:
                        if all(b_q[c] == b_p[(q + c) % 26] for c, b_p in best_p.items()):
                            q_cands[q][b_q] += w
                            added = True

            # Algebraic peeling on this plane
            for _ in range(25):
                peeled = False
                for c in range(43):
                    if c in best_p:
                        continue
                    cw = rspc.P_IDX[2 * c + plane]
                    un = [i for i, pos in enumerate(cw) if pos not in known_pl and pos not in rspc.FIXED_POS]
                    if 1 <= len(un) <= 2:
                        ok, corrs = rspc.solve_2eras(core, cw, un)
                        if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                            for pos, val in corrs:
                                core[pos] = val
                            if rspc.syndromes_p(core, cw) == (0, 0):
                                for pos in cw:
                                    known_pl.add(pos)
                                best_p[c] = bytes(core[p] for p in cw)
                                peeled = True
                for q in range(26):
                    if q in best_q:
                        continue
                    cw = rspc.Q_IDX[2 * q + plane]
                    un = [i for i, pos in enumerate(cw) if pos not in known_pl and pos not in rspc.FIXED_POS]
                    if 1 <= len(un) <= 2:
                        ok, corrs = rspc.solve_2eras(core, cw, un)
                        if ok and all(p not in rspc.FIXED_POS for p, _ in corrs):
                            for pos, val in corrs:
                                core[pos] = val
                            if rspc.syndromes_q(core, cw) == (0, 0):
                                for pos in cw:
                                    known_pl.add(pos)
                                best_q[q] = bytes(core[p] for p in cw)
                                peeled = True
                if not peeled:
                    break
            if len(best_p) == 43 and len(best_q) == 26:
                break
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
    description = "Speed/direction slices + deterministic random k-of-N subgrid sampling with nested top-k slicing."

    def solve(self, lba, raw_passes, context=None):
        filtered = [(r, c) for r, c in raw_passes if not rspc.is_frame_slip(lba, r)]
        n = len(filtered)
        if n < 8:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, -1, -1, sec, "PARTIAL_SUBSET"

        pass_cache = (
            context["pass_cache"]
            if (context is not None and "pass_cache" in context and len(context["pass_cache"]) == n)
            else rspc.build_pass_cache(lba, filtered)
        )
        all_idx = list(range(n))
        subsets = [
            ("LAST32", all_idx[-32:] if n > 32 else all_idx),
            ("FWD_SLICE", all_idx[::2]),
            ("REV_SLICE", all_idx[1::2]),
        ]
        rng = random.Random(lba)
        for r_i in range(28):
            k = rng.choice([k0 for k0 in (8, 12, 16, 20, 24, 32) if k0 < n] or [n])
            subsets.append((f"RAND{r_i}_K{k}", rng.sample(all_idx, k)))

        best = (False, -1, -1, bytearray(2352), "PARTIAL_SUBSET")
        for label, idxs in subsets:
            if len(idxs) < 2:
                continue
            scored_idxs = sorted(idxs, key=lambda i: pass_cache[i]["p_ok_count"], reverse=True)
            m = len(scored_idxs)
            k_subs = sorted({ks for ks in (m, max(2, m // 2), 6, 4, 3, 2) if 2 <= ks <= m}, reverse=True)
            for k_s in k_subs:
                inc_modes = (False, True) if k_s in (m, k_subs[0], 2) else (False,)
                for inc_q in inc_modes:
                    ok, pz, qz, sec = subgrid_peel_decode(
                        lba,
                        filtered,
                        include_q_parity_in_uncov=inc_q,
                        c2_strict=False,
                        pass_cache=pass_cache,
                        indices=scored_idxs[:k_s],
                    )
                    if ok:
                        return True, 86, 52, sec, f"DEEP_{label}_K{k_s}_Q{int(inc_q)}"
                    if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                        best = (False, pz, qz, sec, f"PARTIAL_DEEP_{label}_K{k_s}")
        return best


@SolverRegistry.register
class BipartiteCoreHybridSolver(SectorSolverPlugin):
    name = "bipartite_core_hybrid"
    tier = 3
    description = "K43,26 zero-conflict bipartite core + core-injected hybrid pass harvesting."

    def solve(self, lba, raw_passes, context=None):
        filtered = [(r, c) for r, c in raw_passes if not rspc.is_frame_slip(lba, r)]
        if not filtered:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, -1, -1, sec, "PARTIAL_CORE"
        best = (False, -1, -1, bytearray(2352), "PARTIAL_CORE")
        best_known: set[int] | None = None
        for prefer_p in (True, False):
            sec, known = assemble_bipartite_core(lba, filtered, prefer_p=prefer_p)
            chk = bytearray(sec)
            if rspc.check_and_fix_q_parity(chk, expected_lba=lba):
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


def _build_and_solve_gf2_system(
    sec_base: bytearray,
    un_pos: list[int],
    p_touch: list[int],
    q_eqs: list[int] | None = None,
) -> tuple[bytearray, int, int, list[int]]:
    """Build and solve the linear GF(2) system for RSPC P (+ optional Q) syndromes + 32-bit EDC."""
    q_list = q_eqs or []
    sec0 = bytearray(sec_base)
    for pos in un_pos:
        sec0[pos] = 0

    rhs_bits: list[int] = []
    for idx in p_touch:
        s0, s1 = rspc.syndromes_p(sec0, rspc.P_IDX[idx])
        rhs_bits.extend((s0 >> b) & 1 for b in range(8))
        rhs_bits.extend((s1 >> b) & 1 for b in range(8))
    for idx in q_list:
        s0, s1 = rspc.syndromes_q(sec0, rspc.Q_IDX[idx])
        rhs_bits.extend((s0 >> b) & 1 for b in range(8))
        rhs_bits.extend((s1 >> b) & 1 for b in range(8))
    edc0 = rspc.edc_residual(sec0[:2068])
    rhs_bits.extend((edc0 >> b) & 1 for b in range(32))

    col_vecs = [0] * len(rhs_bits)
    zsec = bytearray(2352)
    p_map = {idx: i * 16 for i, idx in enumerate(p_touch)}
    q_map = {idx: (len(p_touch) + i) * 16 for i, idx in enumerate(q_list)}
    edc_off = (len(p_touch) + len(q_list)) * 16

    for k, pos in enumerate(un_pos):
        p_idx = rspc.BYTE_TO_P.get(pos)
        q_idx = rspc.BYTE_TO_Q.get(pos)
        p_off = p_map.get(p_idx, -1) if p_idx is not None else -1
        q_off = q_map.get(q_idx, -1) if q_idx is not None else -1
        for bit_idx in range(8):
            var_id = 8 * k + bit_idx
            zsec[pos] = 1 << bit_idx
            if p_off >= 0 and p_idx is not None:
                s0, s1 = rspc.syndromes_p(zsec, rspc.P_IDX[p_idx])
                for b in range(8):
                    if (s0 >> b) & 1:
                        col_vecs[p_off + b] |= 1 << var_id
                    if (s1 >> b) & 1:
                        col_vecs[p_off + 8 + b] |= 1 << var_id
            if q_off >= 0 and q_idx is not None:
                s0, s1 = rspc.syndromes_q(zsec, rspc.Q_IDX[q_idx])
                for b in range(8):
                    if (s0 >> b) & 1:
                        col_vecs[q_off + b] |= 1 << var_id
                    if (s1 >> b) & 1:
                        col_vecs[q_off + 8 + b] |= 1 << var_id
            if pos < 2068:
                dedc = rspc.edc_residual(zsec[:2068])
                for b in range(32):
                    if (dedc >> b) & 1:
                        col_vecs[edc_off + b] |= 1 << var_id
            zsec[pos] = 0

    rank, sol0, null_basis = gf2_solve(list(zip(col_vecs, rhs_bits)), 8 * len(un_pos))
    return sec0, rank, sol0, null_basis


def _solve_gf2_subset(
    lba: int,
    sec_core: bytearray,
    un_all: list[int],
    raw_passes: list[tuple[bytes, bytes]],
    c2_sets: list[set[int]],
    qpar_clean: dict[int, set[int]],
    oracle_fn=None,
) -> tuple[bool, bytearray | None, str]:
    n_p = max(1, len(raw_passes))
    pin_cands = []
    for pos in un_all:
        cvals = [rp[0][pos] for rp, c2 in zip(raw_passes, c2_sets) if pos not in c2]
        if cvals:
            val, cnt = Counter(cvals).most_common(1)[0]
        else:
            val, _ = Counter(rp[0][pos] for rp in raw_passes).most_common(1)[0]
            cnt = 0
        clean_frac = cnt / n_p
        all_frac = sum(1 for rp in raw_passes if rp[0][pos] == val) / n_p
        pin_cands.append((clean_frac, all_frac, cnt, pos, val))
    pin_cands.sort(reverse=True)

    max_pins = max(0, len(un_all) - 1)
    pin_schedule = [0] + [p for p in range(1, max_pins + 1) if p <= 12 or p % 2 == 0 or len(un_all) - p <= 24]
    for n_pins in pin_schedule:
        sec_base = bytearray(sec_core)
        pinned = set()
        for clean_frac, _, _, pos, val in pin_cands[:n_pins]:
            if n_pins > 0 and clean_frac < 0.30 and len(un_all) - n_pins > 18:
                break
            sec_base[pos] = val
            pinned.add(pos)
        un_pos = [p for p in un_all if p not in pinned and p not in rspc.FIXED_POS]
        if not un_pos or len(un_pos) > 45:
            continue

        p_touch = sorted({rspc.BYTE_TO_P[p] for p in un_pos if p in rspc.BYTE_TO_P})
        if any(
            rspc.syndromes_p(sec_base, rspc.P_IDX[idx]) != (0, 0)
            and any(p not in un_pos for p in rspc.P_IDX[idx])
            and all(p not in un_pos for p in rspc.P_IDX[idx])
            for idx in range(86)
            if idx not in p_touch
        ):
            continue

        sec0, rank, sol0, null_basis = _build_and_solve_gf2_system(sec_base, un_pos, p_touch)
        if any(rspc.syndromes_p(sec0, rspc.P_IDX[idx]) != (0, 0) for idx in range(86) if idx not in p_touch):
            continue
        if rank == -1 or len(null_basis) > 13:
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
            if not rspc.check_and_fix_q_parity(cand, expected_lba=lba):
                continue
            q_match = sum(1 for pos in range(2248, 2352) if cand[pos] in qpar_clean[pos])
            raw_hits = sum(1 for pos in un_pos for rp in raw_passes if rp[0][pos] == cand[pos])
            o_score = oracle_fn(bytes(cand[16:2064])) if oracle_fn is not None else 0.0
            candidates.append((q_match, o_score, raw_hits, cand))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            q_match, o_score, _, winner = candidates[0]
            if (
                q_match >= 100
                or (len(null_basis) == 0 and q_match >= 98)
                or (oracle_fn is not None and o_score > -20.0)
            ):
                return True, winner, f"GF2_P_EDC_PIN{n_pins}_ND{len(null_basis)}"
    return False, None, ""


def _solve_gf2_plane_pq_edc(
    lba: int,
    sec_merged: bytearray,
    pl_bad: int,
    raw_passes: list[tuple[bytes, bytes]],
    c2_sets: list[set[int]],
    qpar_clean: dict[int, set[int]],
    qpar_maj: dict[int, int],
    qpar_conf: dict[int, float],
    oracle_fn=None,
) -> tuple[bool, bytearray | None, str]:
    """Solve one plane's bad_p x bad_q stopping set using joint GF(2) P + Q (+ single-clean Q-par) + EDC."""
    n_p = max(1, len(raw_passes))
    sec = bytearray(sec_merged)
    rspc.init_sector_header(lba, sec)
    for pos in range(2248 + pl_bad, 2352, 2):
        q_idx = rspc.BYTE_TO_Q[pos]
        if rspc.syndromes_q(sec, rspc.Q_IDX[q_idx]) != (0, 0):
            sec[pos] = qpar_maj[pos]

    bad_p = sorted(i for i in range(pl_bad, 86, 2) if rspc.syndromes_p(sec, rspc.P_IDX[i]) != (0, 0))
    bad_q = sorted(i for i in range(pl_bad, 52, 2) if rspc.syndromes_q(sec, rspc.Q_IDX[i]) != (0, 0))
    if not bad_p:
        chk = bytearray(sec)
        if rspc.check_and_fix_q_parity(chk, expected_lba=lba):
            return True, chk, f"PER_PLANE_QPAR_RESTORED_P{pl_bad}"
        return False, None, "no_bad_p_q_fail"

    bad_p_set, bad_q_set = set(bad_p), set(bad_q)
    uncov = sorted(
        j
        for j in range(16 + pl_bad, 2248, 2)
        if j not in rspc.FIXED_POS and rspc.BYTE_TO_P[j] in bad_p_set and rspc.BYTE_TO_Q[j] in bad_q_set
    )
    if not uncov or len(uncov) > 85:
        return False, None, f"uncov_{len(uncov)}"

    pin_cands = []
    for pos in uncov:
        cvals = [rp[0][pos] for rp, c2 in zip(raw_passes, c2_sets) if pos not in c2]
        if cvals:
            val, cnt = Counter(cvals).most_common(1)[0]
        else:
            val, _ = Counter(rp[0][pos] for rp in raw_passes).most_common(1)[0]
            cnt = 0
        all_cnt = sum(1 for rp in raw_passes if rp[0][pos] == val)
        pin_cands.append((cnt / n_p, all_cnt / n_p, pos, val))
    pin_cands.sort(reverse=True)

    touched_qpar = [rspc.Q_IDX[q][i] for q in bad_q for i in (43, 44)]
    conf_thresholds = (0.90, 0.75, 1.1) if len(uncov) <= 12 else (0.90, 0.75)
    best_nd_fallback: tuple[tuple[int, int, int, float, int], bytearray, str] | None = None

    for conf_thr in conf_thresholds:
        q_eqs: list[int] = []
        extra_qpar: list[int] = []
        if conf_thr <= 1.0:
            for q in bad_q:
                p43, p44 = rspc.Q_IDX[q][43], rspc.Q_IDX[q][44]
                ok43 = qpar_conf[p43] >= conf_thr
                ok44 = qpar_conf[p44] >= conf_thr
                if ok43 and ok44:
                    q_eqs.append(q)
                elif ok43:
                    q_eqs.append(q)
                    extra_qpar.append(p44)
                elif ok44:
                    q_eqs.append(q)
                    extra_qpar.append(p43)

        max_pins = min(14, max(0, len(uncov) - 1))
        for n_pins in range(0, max_pins + 1):
            if n_pins > 0 and pin_cands[n_pins - 1][0] < 0.30 and len(uncov) - n_pins > 18:
                break
            if best_nd_fallback is not None and (n_pins == 0 or pin_cands[n_pins - 1][0] < 0.35):
                return True, best_nd_fallback[1], best_nd_fallback[2]

            sec_b = bytearray(sec)
            pinned = set()
            for _, _, pos, val in pin_cands[:n_pins]:
                sec_b[pos] = val
                pinned.add(pos)
            un_pos = [p for p in uncov if p not in pinned] + extra_qpar
            if not un_pos or len(un_pos) > 55:
                continue

            sec0, rank, sol0, null_basis = _build_and_solve_gf2_system(sec_b, un_pos, bad_p, q_eqs)
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
                if not rspc.check_and_fix_q_parity(cand, expected_lba=lba):
                    continue
                tq_clean = sum(1 for pos in touched_qpar if cand[pos] in qpar_clean[pos])
                qm = sum(1 for pos in range(2248, 2352) if cand[pos] in qpar_clean[pos])
                tq_votes = sum(sum(1 for rp in raw_passes if rp[0][pos] == cand[pos]) for pos in touched_qpar)
                raw_hits = sum(1 for pos in un_pos if pos < 2248 for rp in raw_passes if rp[0][pos] == cand[pos])
                o_score = oracle_fn(bytes(cand[16:2064])) if oracle_fn is not None else 0.0
                candidates.append(((tq_clean, qm, tq_votes, o_score, raw_hits), cand))

            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            (tqc, qm, tqv, osc, rh), winner = candidates[0]

            if len(null_basis) == 0:
                if qm >= 100 and (not touched_qpar or tqc >= max(1, len(touched_qpar) - 3)):
                    return True, winner, f"GF2_PLANE_P{pl_bad}_PIN{n_pins}_ND0_QM{qm}"
            else:
                unique_best = len(candidates) == 1
                if not unique_best:
                    (tqc2, qm2, _, _, rh2) = candidates[1][0]
                    unique_best = (tqc, qm, rh) > (tqc2, qm2, rh2) or (oracle_fn is not None and osc > -20.0)
                if qm >= 102 and (not touched_qpar or tqc >= max(1, len(touched_qpar) - 2)) and unique_best:
                    strat = f"GF2_PLANE_P{pl_bad}_PIN{n_pins}_ND{len(null_basis)}_QM{qm}"
                    if best_nd_fallback is None or (tqc, qm, tqv, osc, rh) > best_nd_fallback[0]:
                        best_nd_fallback = ((tqc, qm, tqv, osc, rh), winner, strat)
                    # Check if next pin also has >=0.35 clean support to try collapsing nullity to 0
                    if n_pins + 1 > max_pins or pin_cands[n_pins][0] < 0.35:
                        return True, best_nd_fallback[1], best_nd_fallback[2]

    if best_nd_fallback is not None:
        return True, best_nd_fallback[1], best_nd_fallback[2]
    return False, None, "failed"


@SolverRegistry.register
class PerPlaneSubgridGF2Solver(SectorSolverPlugin):
    name = "per_plane_subgrid_gf2"
    tier = 3
    priority = 10
    description = "Exploit even/odd RSPC plane independence: solve/merge planes or bridge 1-plane stopping set via P+Q+EDC GF(2)."

    def solve(self, lba, raw_passes, context=None):
        filtered = [(r, c) for r, c in raw_passes if not rspc.is_frame_slip(lba, r)]
        n = len(filtered)
        if n < 2:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, -1, -1, sec, "PARTIAL_PER_PLANE"

        pass_cache = (
            context["pass_cache"]
            if (context is not None and "pass_cache" in context and len(context["pass_cache"]) == n)
            else rspc.build_pass_cache(lba, filtered)
        )
        all_idx = list(range(n))
        c2_pl_counts = [
            (
                sum(1 for pos in pass_cache[i]["c2"] if pos >= 16 and (pos & 1) == 0),
                sum(1 for pos in pass_cache[i]["c2"] if pos >= 16 and (pos & 1) == 1),
            )
            for i in all_idx
        ]
        all_q_order = sorted(
            all_idx,
            key=lambda i: (sum(pass_cache[i]["q_ok_raw"]), -len(pass_cache[i]["c2"])),
            reverse=True,
        )
        k_grid = sorted({k for k in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 28, n) if 2 <= k <= n})

        solved_by_plane: dict[int, list[bytearray]] = {0: [], 1: []}
        solved_keys: dict[int, set[bytes]] = {0: set(), 1: set()}
        partial_by_plane: dict[int, list[tuple[tuple[int, int, int], bytearray]]] = {0: [], 1: []}
        partial_keys: dict[int, set[bytes]] = {0: set(), 1: set()}
        best = (False, -1, -1, bytearray(2352), "PARTIAL_PER_PLANE")

        for pl in (0, 1):
            orders = [
                (
                    sorted(
                        all_idx,
                        key=lambda i: (
                            sum(pass_cache[i]["q_ok_raw"][pl::2]),
                            sum(pass_cache[i]["p_ok_raw"][pl::2]),
                            -c2_pl_counts[i][pl],
                        ),
                        reverse=True,
                    ),
                    ((False, False), (True, False), (False, True)),
                ),
                (
                    sorted(
                        all_idx,
                        key=lambda i: (
                            sum(pass_cache[i]["p_ok_raw"][pl::2]),
                            sum(pass_cache[i]["q_ok_raw"][pl::2]),
                            -c2_pl_counts[i][pl],
                        ),
                        reverse=True,
                    ),
                    ((False, False), (True, False), (False, True)),
                ),
                (
                    sorted(all_idx, key=lambda i: -c2_pl_counts[i][pl], reverse=True),
                    ((False, False), (True, False), (False, True)),
                ),
                (
                    all_q_order,
                    ((False, False), (False, True)),
                ),
            ]
            for order, modes in orders:
                for k in k_grid:
                    for inc_q, c2s in modes:
                        ok, pz, qz, sec = subgrid_peel_decode(
                            lba,
                            filtered,
                            include_q_parity_in_uncov=inc_q,
                            c2_strict=c2s,
                            pass_cache=pass_cache,
                            indices=order[:k],
                        )
                        if ok:
                            return True, 86, 52, sec, f"PER_PLANE_FULL_K{k}"
                        if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                            best = (False, pz, qz, sec, f"PARTIAL_PER_PLANE_K{k}")
                        pp = sum(1 for i in range(pl, 86, 2) if rspc.syndromes_p(sec, rspc.P_IDX[i]) == (0, 0))
                        qq = sum(1 for i in range(pl, 52, 2) if rspc.syndromes_q(sec, rspc.Q_IDX[i]) == (0, 0))
                        key = bytes(sec[16 + pl : 2352 : 2])
                        if pp == 43 and qq == 26:
                            if key not in solved_keys[pl] and len(solved_by_plane[pl]) < 4:
                                solved_keys[pl].add(key)
                                solved_by_plane[pl].append(sec)
                                # Check immediate merge with any already-solved opposite plane
                                for opp_sec in solved_by_plane[1 - pl]:
                                    merged = bytearray(2352)
                                    rspc.init_sector_header(lba, merged)
                                    s0 = sec if pl == 0 else opp_sec
                                    s1 = opp_sec if pl == 0 else sec
                                    for pos in range(16, 2352, 2):
                                        merged[pos] = s0[pos]
                                    for pos in range(17, 2352, 2):
                                        merged[pos] = s1[pos]
                                    if rspc.check_and_fix_q_parity(merged, expected_lba=lba):
                                        return True, 86, 52, merged, "PER_PLANE_MERGE"
                        elif key not in partial_keys[pl]:
                            bad_p = {i for i in range(pl, 86, 2) if rspc.syndromes_p(sec, rspc.P_IDX[i]) != (0, 0)}
                            bad_q = {i for i in range(pl, 52, 2) if rspc.syndromes_q(sec, rspc.Q_IDX[i]) != (0, 0)}
                            uncov_cnt = sum(
                                1
                                for j in range(16 + pl, 2248, 2)
                                if j not in rspc.FIXED_POS
                                and rspc.BYTE_TO_P[j] in bad_p
                                and rspc.BYTE_TO_Q[j] in bad_q
                            )
                            score = (pp + qq, qq, -uncov_cnt)
                            if len(partial_by_plane[pl]) < 4 or score > partial_by_plane[pl][-1][0]:
                                partial_keys[pl].add(key)
                                partial_by_plane[pl].append((score, sec))
                                partial_by_plane[pl].sort(key=lambda x: x[0], reverse=True)
                                partial_by_plane[pl] = partial_by_plane[pl][:4]

        # If one plane solved completely and the other stopped partially, bridge the stopping set via joint GF(2) P+Q+EDC
        oracle_fn = (context or {}).get("oracle_fn")
        c2_sets = [pass_cache[i]["c2"] for i in range(n)]
        qpar_clean: dict[int, set[int]] = {}
        qpar_maj: dict[int, int] = {}
        qpar_conf: dict[int, float] = {}
        min_clean = max(2, n // 10) if n >= 2 else 1
        for pos in range(2248, 2352):
            avals = [filtered[i][0][pos] for i in range(n)]
            cnt_a = Counter(avals)
            val_a, hit_a = cnt_a.most_common(1)[0]
            qpar_maj[pos] = val_a
            qpar_conf[pos] = hit_a / n
            qpar_clean[pos] = {v for v, c in cnt_a.items() if c >= min_clean} or {val_a}

        for pl_good in (0, 1):
            pl_bad = 1 - pl_good
            if not solved_by_plane[pl_good] or not partial_by_plane[pl_bad]:
                continue
            for good_sec in solved_by_plane[pl_good][:2]:
                for _, part_sec in partial_by_plane[pl_bad][:4]:
                    merged = bytearray(2352)
                    rspc.init_sector_header(lba, merged)
                    s0 = good_sec if pl_good == 0 else part_sec
                    s1 = part_sec if pl_good == 0 else good_sec
                    for pos in range(16, 2352, 2):
                        merged[pos] = s0[pos]
                    for pos in range(17, 2352, 2):
                        merged[pos] = s1[pos]
                    ok_g, win_g, strat_g = _solve_gf2_plane_pq_edc(
                        lba,
                        merged,
                        pl_bad,
                        filtered,
                        c2_sets,
                        qpar_clean,
                        qpar_maj,
                        qpar_conf,
                        oracle_fn=oracle_fn,
                    )
                    if ok_g and win_g is not None:
                        return True, 86, 52, win_g, strat_g
        return best


_EDC_BIT_TABLE: list[list[int]] | None = None


def _get_edc_bit_table() -> list[list[int]]:
    """Lazy-init 2068x8 linear GF(2) EDC basis vectors for bytes 16..2067."""
    global _EDC_BIT_TABLE
    if _EDC_BIT_TABLE is not None:
        return _EDC_BIT_TABLE
    tbl = [[0] * 8 for _ in range(2068)]
    z = bytearray(2068)
    for pos in range(16, 2068):
        for b in range(8):
            z[pos] = 1 << b
            tbl[pos][b] = rspc.edc_residual(z)
        z[pos] = 0
    _EDC_BIT_TABLE = tbl
    return tbl


def _edc_of_byte(tbl: list[list[int]], pos: int, val: int) -> int:
    if pos >= 2068 or val == 0:
        return 0
    eb = tbl[pos]
    r = 0
    for b in range(8):
        if (val >> b) & 1:
            r ^= eb[b]
    return r


def _solve_weight3_forged_p_edc(
    lba: int,
    sec_core: bytearray,
    raw_passes: list[tuple[bytes, bytes]],
    qpar_clean: dict[int, set[int]],
    oracle_fn=None,
) -> tuple[bool, bytearray | None, str]:
    """Break forged 86/86 P loops (EDC != 0) by solving 1-column weight-3 P+EDC equations."""
    e_res = rspc.edc_residual(sec_core[:2068])
    if e_res == 0:
        return False, None, ""
    tbl = _get_edc_bit_table()
    w = rspc.P_WEIGHTS
    mul = rspc.GF_MUL
    qpar_counts = [Counter(rp[0][pos] for rp in raw_passes) for pos in range(2248, 2352)]

    candidates = []
    for cw in rspc.P_IDX:
        for i2 in (24, 25):
            w2 = w[i2]
            p2 = cw[i2]
            for i0 in range(24):
                p0 = cw[i0]
                if p0 in rspc.FIXED_POS:
                    continue
                w0 = w[i0]
                for i1 in range(i0 + 1, 24):
                    p1 = cw[i1]
                    if p1 in rspc.FIXED_POS:
                        continue
                    w1 = w[i1]
                    e1_u = rspc.gf_div(w0 ^ w2, w1 ^ w2)
                    piv: dict[int, tuple[int, int]] = {}
                    for b in range(8):
                        v = _edc_of_byte(tbl, p0, 1 << b) ^ _edc_of_byte(tbl, p1, mul[e1_u][1 << b])
                        mask = 1 << b
                        while v:
                            lb = (v & -v).bit_length() - 1
                            if lb in piv:
                                pv, pm = piv[lb]
                                v ^= pv
                                mask ^= pm
                            else:
                                piv[lb] = (v, mask)
                                break
                    rem, alpha = e_res, 0
                    in_span = True
                    while rem:
                        lb = (rem & -rem).bit_length() - 1
                        if lb in piv:
                            pv, pm = piv[lb]
                            rem ^= pv
                            alpha ^= pm
                        else:
                            in_span = False
                            break
                    if not in_span or alpha == 0:
                        continue
                    e0 = alpha
                    e1 = mul[e1_u][alpha]
                    e2 = e0 ^ e1
                    if e1 == 0 or e2 == 0:
                        continue
                    cand = bytearray(sec_core)
                    cand[p0] ^= e0
                    cand[p1] ^= e1
                    cand[p2] ^= e2
                    if not rspc.check_and_fix_q_parity(cand, expected_lba=lba):
                        continue
                    q_match = sum(1 for pos in range(2248, 2352) if cand[pos] in qpar_clean[pos])
                    changed_qpar = [pos for pos in range(2248, 2352) if cand[pos] != sec_core[pos]]
                    changed_qpar_votes = sum(qpar_counts[pos - 2248][cand[pos]] for pos in changed_qpar)
                    changed_qpar_hits = sum(1 for pos in changed_qpar if cand[pos] in qpar_clean[pos])
                    o_score = oracle_fn(bytes(cand[16:2064])) if oracle_fn is not None else 0.0
                    candidates.append((q_match, changed_qpar_votes, o_score, changed_qpar_hits, cand))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        q_match, _, o_score, changed_qpar_hits, winner = candidates[0]
        if q_match >= 100 and (changed_qpar_hits >= 3 or (oracle_fn is not None and o_score > -20.0)):
            return True, winner, "GF2_W3_FORGED_P_EDC"
    return False, None, ""


@SolverRegistry.register
class JointGF2EdcSolver(SectorSolverPlugin):
    name = "joint_gf2_edc"
    tier = 3
    description = "Joint GF(2) linear RSPC P-syndrome + 32-bit EDC Gaussian solver over core residuals and weight-3 P traps."

    def solve(self, lba, raw_passes, context=None):
        filtered = [(r, c) for r, c in raw_passes if not rspc.is_frame_slip(lba, r)]
        if not filtered:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, -1, -1, sec, "PARTIAL_GF2"

        best = (False, -1, -1, bytearray(2352), "PARTIAL_GF2")
        c2_sets = [rspc.c2_set(rp[1]) for rp in filtered]
        qpar_clean = {}
        for pos in range(2248, 2352):
            vals = [rp[0][pos] for rp, c2 in zip(filtered, c2_sets) if pos not in c2] or [rp[0][pos] for rp in filtered]
            qpar_clean[pos] = {v for v, c in Counter(vals).items() if c >= 2}

        oracle_fn = (context or {}).get("oracle_fn")
        for prefer_p in (True, False):
            sec_core, known = assemble_bipartite_core(lba, filtered, prefer_p=prefer_p)
            chk = bytearray(sec_core)
            if rspc.check_and_fix_q_parity(chk, expected_lba=lba):
                q_match = sum(1 for pos in range(2248, 2352) if chk[pos] in qpar_clean[pos])
                o_score = oracle_fn(bytes(chk[16:2064])) if oracle_fn is not None else 0.0
                if q_match >= 98 or (oracle_fn is not None and o_score > -20.0):
                    if context is not None:
                        context["known_mask_2048"] = set(range(2048))
                    return True, 86, 52, chk, f"BIPARTITE_CORE_P{int(prefer_p)}"

            un_all = sorted(p for p in range(16, 2248) if p not in known and p not in rspc.FIXED_POS)
            if 1 <= len(un_all) <= 85:
                ok_g, win_g, strat_g = _solve_gf2_subset(
                    lba, sec_core, un_all, filtered, c2_sets, qpar_clean, oracle_fn=oracle_fn
                )
                if ok_g and win_g is not None:
                    if context is not None:
                        context["known_mask_2048"] = set(range(2048))
                    return True, 86, 52, win_g, strat_g

            # Weight-3 Q-trap recovery on 1..2 failing P-columns even when bipartite core marks un_all empty
            p_bad = [idx for idx, cw in enumerate(rspc.P_IDX) if rspc.syndromes_p(sec_core, cw) != (0, 0)]
            if len(p_bad) == 1:
                st1, c1 = rspc.solve_1err(sec_core, rspc.P_IDX[p_bad[0]])
                if st1 == 1 and c1[0][0] not in rspc.FIXED_POS:
                    cand1 = bytearray(sec_core)
                    cand1[c1[0][0]] = c1[0][1]
                    if rspc.check_and_fix_q_parity(cand1, expected_lba=lba):
                        q_match = sum(1 for pos in range(2248, 2352) if cand1[pos] in qpar_clean[pos])
                        o_score = oracle_fn(bytes(cand1[16:2064])) if oracle_fn is not None else 0.0
                        if q_match >= 98 or (oracle_fn is not None and o_score > -20.0):
                            if context is not None:
                                context["known_mask_2048"] = set(range(2048))
                            return True, 86, 52, cand1, "GF2_P_TRAP_1ERR"

            if 1 <= len(p_bad) <= 2:
                col_un = sorted({pos for idx in p_bad for pos in rspc.P_IDX[idx] if pos not in rspc.FIXED_POS})
                if col_un:
                    ok_t, win_t, strat_t = _solve_gf2_subset(
                        lba, sec_core, col_un, filtered, c2_sets, qpar_clean, oracle_fn=oracle_fn
                    )
                    if ok_t and win_t is not None:
                        if context is not None:
                            context["known_mask_2048"] = set(range(2048))
                        return True, 86, 52, win_t, f"{strat_t}_PTRAP"

            if not p_bad and rspc.edc_residual(sec_core[:2068]) != 0:
                ok_w3, win_w3, strat_w3 = _solve_weight3_forged_p_edc(
                    lba, sec_core, filtered, qpar_clean, oracle_fn=oracle_fn
                )
                if ok_w3 and win_w3 is not None:
                    if context is not None:
                        context["known_mask_2048"] = set(range(2048))
                    return True, 86, 52, win_w3, strat_w3
        return best
