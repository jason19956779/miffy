"""Tier 1 & Tier 2 portfolio sector solvers built over src.core.rspc."""

from collections import Counter
from itertools import combinations

from src.core import rspc
from src.solvers import SolverRegistry
from src.solvers.base import SectorSolverPlugin


def _orth_cw(is_p: bool, pos: int) -> tuple[int, ...] | None:
    if is_p and pos in rspc.BYTE_TO_Q:
        return rspc.Q_IDX[rspc.BYTE_TO_Q[pos]]
    if not is_p and pos in rspc.BYTE_TO_P:
        return rspc.P_IDX[rspc.BYTE_TO_P[pos]]
    return None


def clean_single_pass(
    lba: int, raw_sec: bytes, c2_bytes: bytes
) -> tuple[bool, bytearray, set[int]]:
    """Single-pass C2-erasure RSPC decoding + orthogonal 1-error polish."""
    sec = bytearray(raw_sec)
    rspc.init_sector_header(lba, sec)
    c2_eras = rspc.c2_set(c2_bytes)
    eras = set(c2_eras)

    for _ in range(15):
        if rspc.check_and_fix_q_parity(sec):
            return True, sec, c2_eras
        changed = 0
        for cw_list in (rspc.P_IDX, rspc.Q_IDX):
            for cw in cw_list:
                if rspc.syndromes(sec, cw) == (0, 0):
                    for pos in cw:
                        eras.discard(pos)
                    continue
                cw_eras = [i for i, pos in enumerate(cw) if pos in eras]
                if 1 <= len(cw_eras) <= 2:
                    ok, corrs = rspc.solve_2eras(sec, cw, cw_eras)
                    if ok:
                        for pos, val in corrs:
                            sec[pos] = val
                        for i in cw_eras:
                            eras.discard(cw[i])
                        changed += 1
                        continue
                st1, corr1 = rspc.solve_1err(sec, cw)
                if st1 == 1 and corr1[0][0] in eras:
                    pos, val = corr1[0]
                    sec[pos] = val
                    eras.discard(pos)
                    changed += 1
        if changed == 0:
            for is_p, cw_list in ((True, rspc.P_IDX), (False, rspc.Q_IDX)):
                for cw in cw_list:
                    st1, corr1 = rspc.solve_1err(sec, cw)
                    if st1 == 1:
                        pos, val = corr1[0]
                        if pos not in rspc.FIXED_POS:
                            old = sec[pos]
                            sec[pos] = val
                            orth = _orth_cw(is_p, pos)
                            if orth is not None and rspc.syndromes(sec, orth) == (0, 0):
                                eras.discard(pos)
                                changed += 1
                            else:
                                sec[pos] = old
        if changed == 0:
            break
    return rspc.check_and_fix_q_parity(sec), sec, c2_eras


def _try_plane_sweep(comp: bytearray, candidate_vals: list[set[int]]) -> bytearray | None:
    test = bytearray(comp)
    for plane in (0, 1):
        p_bad = [i for i in range(plane, 86, 2) if rspc.syndromes_p(test, rspc.P_IDX[i]) != (0, 0)]
        q_bad = [i for i in range(plane, 52, 2) if rspc.syndromes_q(test, rspc.Q_IDX[i]) != (0, 0)]
        if 1 <= len(p_bad) <= 2 and q_bad:
            p_set = set(p_bad)
            mod_q = set()
            for q_idx in q_bad:
                cw = rspc.Q_IDX[q_idx]
                eras = [i for i, pos in enumerate(cw[:43]) if rspc.BYTE_TO_P[pos] in p_set]
                ok, corrs = rspc.solve_2eras(test, cw, eras)
                if ok and corrs:
                    for pos, val in corrs:
                        test[pos] = val
                    mod_q.add(q_idx)
            for p_idx in p_bad:
                cw = rspc.P_IDX[p_idx]
                if rspc.syndromes_p(test, cw) == (0, 0):
                    continue
                st1, c1 = rspc.solve_1err(test, cw)
                if st1 == 1:
                    test[c1[0][0]] = c1[0][1]
                else:
                    cand_i = [i for i, pos in enumerate(cw) if rspc.BYTE_TO_Q.get(pos) in mod_q]
                    if 2 <= len(cand_i) <= 12:
                        for i0, i1 in combinations(cand_i, 2):
                            ok, corrs = rspc.solve_2eras(test, cw, [i0, i1])
                            if ok and len(corrs) == 2 and all(v in candidate_vals[p] for p, v in corrs):
                                for p, v in corrs:
                                    test[p] = v
                                break
        elif 1 <= len(q_bad) <= 2 and p_bad:
            q_set = set(q_bad)
            for p_idx in p_bad:
                cw = rspc.P_IDX[p_idx]
                eras = [i for i, pos in enumerate(cw) if pos in rspc.BYTE_TO_Q and rspc.BYTE_TO_Q[pos] in q_set]
                ok, corrs = rspc.solve_2eras(test, cw, eras)
                if ok and corrs:
                    for pos, val in corrs:
                        test[pos] = val
    return test if rspc.check_and_fix_q_parity(test) else None


def subgrid_peel_decode(
    lba: int,
    raw_passes: list[tuple[bytes, bytes]],
    include_q_parity_in_uncov: bool = True,
    c2_strict: bool = False,
) -> tuple[bool, int, int, bytearray]:
    """Codeword-weighted vote + K43,26 subgrid peeling + weight-3 unmasking."""
    passes: list[bytearray] = []
    c2_sets: list[set[int]] = []
    c2_union: set[int] = set()
    candidate_vals: list[set[int]] = [set() for _ in range(2352)]

    for raw_sec, c2_bytes in raw_passes:
        sec = bytearray(raw_sec)
        rspc.init_sector_header(lba, sec)
        c2 = rspc.c2_set(c2_bytes)
        passes.append(sec)
        c2_sets.append(c2)
        c2_union.update(c2)
        for j in range(2352):
            candidate_vals[j].add(sec[j])

    votes = [Counter() for _ in range(2352)]
    for p_idx, sec in enumerate(passes):
        c2 = c2_sets[p_idx]
        p_ok = [
            rspc.syndromes_p(sec, cw) == (0, 0)
            and (not c2_strict or sum(1 for pos in cw if pos in c2) < 2)
            for cw in rspc.P_IDX
        ]
        q_ok = [
            rspc.syndromes_q(sec, cw) == (0, 0)
            and (not c2_strict or sum(1 for pos in cw if pos in c2) < 2)
            for cw in rspc.Q_IDX
        ]
        p_1err = {}
        for idx, cw in enumerate(rspc.P_IDX):
            if not p_ok[idx]:
                st1, c1 = rspc.solve_1err(sec, cw)
                if st1 == 1:
                    p_1err[c1[0][0]] = c1[0][1]
        for idx, cw in enumerate(rspc.Q_IDX):
            if not q_ok[idx]:
                st1, c1 = rspc.solve_1err(sec, cw)
                if st1 == 1 and p_1err.get(c1[0][0]) == c1[0][1]:
                    votes[c1[0][0]][c1[0][1]] += 80
                    candidate_vals[c1[0][0]].add(c1[0][1])

        for j in range(16, 2352):
            if j in rspc.FIXED_POS:
                continue
            pok = j in rspc.BYTE_TO_P and p_ok[rspc.BYTE_TO_P[j]]
            qok = j in rspc.BYTE_TO_Q and q_ok[rspc.BYTE_TO_Q[j]]
            c2ok = j not in c2
            if c2_strict:
                w = 120 if (pok and qok and c2ok) else (60 if (pok and qok) else (35 if (qok and c2ok) else (30 if (pok and c2ok) else (12 if qok else (10 if pok else (5 if c2ok else 1))))))
            else:
                w = 120 if (pok and qok) else (35 if (qok and c2ok) else (25 if qok else (30 if (pok and c2ok) else (20 if pok else (4 if c2ok else 1)))))
            votes[j][sec[j]] += w

    comp = bytearray(2352)
    rspc.init_sector_header(lba, comp)
    for j in range(16, 2352):
        if j not in rspc.FIXED_POS:
            comp[j] = votes[j].most_common(1)[0][0]

    used_unmask: set[tuple[int, int]] = set()
    for _ in range(45):
        if rspc.check_and_fix_q_parity(comp):
            return True, 86, 52, comp
        swept = _try_plane_sweep(comp, candidate_vals)
        if swept is not None:
            return True, 86, 52, swept

        p_bad = {i for i, cw in enumerate(rspc.P_IDX) if rspc.syndromes_p(comp, cw) != (0, 0)}
        q_bad = {i for i, cw in enumerate(rspc.Q_IDX) if rspc.syndromes_q(comp, cw) != (0, 0)}

        q_par_fixed = 0
        for idx in list(q_bad):
            st1, c1 = rspc.solve_1err(comp, rspc.Q_IDX[idx])
            if st1 == 1 and c1[0][0] >= 2248:
                comp[c1[0][0]] = c1[0][1]
                q_bad.discard(idx)
                q_par_fixed += 1
        if q_par_fixed:
            continue

        if include_q_parity_in_uncov:
            uncov = {
                j
                for j in range(16, 2352)
                if j not in rspc.FIXED_POS
                and (
                    (j in rspc.BYTE_TO_P and rspc.BYTE_TO_P[j] in p_bad and j in rspc.BYTE_TO_Q and rspc.BYTE_TO_Q[j] in q_bad)
                    or (j >= 2248 and rspc.BYTE_TO_Q[j] in q_bad)
                )
            }
        else:
            uncov = {
                j
                for j in range(16, 2248)
                if j not in rspc.FIXED_POS and rspc.BYTE_TO_P[j] in p_bad and rspc.BYTE_TO_Q[j] in q_bad
            }

        changed = 0
        for bad_set, cw_list in ((p_bad, rspc.P_IDX), (q_bad, rspc.Q_IDX)):
            for idx in list(bad_set):
                cw = cw_list[idx]
                if rspc.syndromes(comp, cw) == (0, 0):
                    continue
                eras = [i for i, pos in enumerate(cw) if pos in uncov]
                if len(eras) == 1 or (include_q_parity_in_uncov and len(eras) == 2):
                    ok, corrs = rspc.solve_2eras(comp, cw, eras)
                    if ok and corrs:
                        for pos, val in corrs:
                            comp[pos] = val
                            uncov.discard(pos)
                        changed += 1
        if changed:
            continue

        for is_p, bad_set, cw_list in ((True, p_bad, rspc.P_IDX), (False, q_bad, rspc.Q_IDX)):
            for idx in list(bad_set):
                cw = cw_list[idx]
                st1, c1 = rspc.solve_1err(comp, cw)
                if st1 == 1 and c1[0][0] in uncov:
                    pos, val = c1[0]
                    old = comp[pos]
                    comp[pos] = val
                    orth = _orth_cw(is_p, pos)
                    orth_ok = (
                        orth is None
                        or rspc.syndromes(comp, orth) == (0, 0)
                        or (rspc.solve_1err(comp, orth)[0] == 1 and rspc.solve_1err(comp, orth)[1][0][0] in uncov)
                    )
                    if (val in candidate_vals[pos]) or orth_ok:
                        uncov.discard(pos)
                        changed += 1
                    else:
                        comp[pos] = old
        if changed:
            continue

        for is_p, bad_set, cw_list in ((True, p_bad, rspc.P_IDX), (False, q_bad, rspc.Q_IDX)):
            for idx in list(bad_set):
                cw = cw_list[idx]
                if rspc.syndromes(comp, cw) == (0, 0):
                    continue
                ec = [i for i, pos in enumerate(cw) if pos in uncov]
                if 2 <= len(ec) <= 10:
                    for i0, i1 in combinations(ec, 2):
                        ok, corrs = rspc.solve_2eras(comp, cw, [i0, i1])
                        if not ok or len(corrs) != 2:
                            continue
                        (pos0, val0), (pos1, val1) = corrs
                        old0, old1 = comp[pos0], comp[pos1]
                        comp[pos0], comp[pos1] = val0, val1
                        o0, o1 = _orth_cw(is_p, pos0), _orth_cw(is_p, pos1)
                        z0 = o0 is not None and rspc.syndromes(comp, o0) == (0, 0)
                        z1 = o1 is not None and rspc.syndromes(comp, o1) == (0, 0)
                        e0 = o0 is not None and rspc.solve_1err(comp, o0)[0] == 1 and rspc.solve_1err(comp, o0)[1][0][0] in uncov
                        e1 = o1 is not None and rspc.solve_1err(comp, o1)[0] == 1 and rspc.solve_1err(comp, o1)[1][0][0] in uncov
                        in0 = val0 in candidate_vals[pos0]
                        in1 = val1 in candidate_vals[pos1]
                        if (z0 and z1) or (z0 and (in1 or e1)) or (z1 and (in0 or e0)) or (in0 and in1 and (len(ec) == 2 or e0 or e1)):
                            changed += 1
                            break
                        comp[pos0], comp[pos1] = old0, old1
                if changed:
                    break
            if changed:
                break
        if changed:
            continue

        small_res = (len(p_bad) + len(q_bad)) <= 8
        for src_bad, src_cws, orth_map in (
            (q_bad, rspc.Q_IDX, rspc.BYTE_TO_P),
            (p_bad, rspc.P_IDX, rspc.BYTE_TO_Q),
        ):
            grouped: dict[int, list[tuple[int, int]]] = {}
            for idx in src_bad:
                st1, c1 = rspc.solve_1err(comp, src_cws[idx])
                if st1 == 1 and c1[0][0] not in rspc.FIXED_POS and c1[0][0] in orth_map:
                    pos, val = c1[0]
                    if (pos, val) not in used_unmask:
                        grouped.setdefault(orth_map[pos], []).append((pos, val))
            for _, corrs in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
                if len(corrs) >= 2 or (
                    len(corrs) == 1
                    and (corrs[0][1] in candidate_vals[corrs[0][0]] or (small_res and corrs[0][0] in c2_union))
                ):
                    for pos, val in corrs:
                        comp[pos] = val
                        used_unmask.add((pos, val))
                    changed += 1
            if changed:
                break
        if changed == 0:
            break

    if rspc.check_and_fix_q_parity(comp):
        return True, 86, 52, comp
    pz, qz = rspc.count_pq(comp)
    return False, pz, qz, comp


@SolverRegistry.register
class SinglePassC2Solver(SectorSolverPlugin):
    name = "single_c2"
    tier = 1
    description = "Single-pass C2 erasure decoding + orthogonal 1-error polish (latest passes first)."

    def solve(self, lba, raw_passes, context=None):
        best = (False, -1, -1, bytearray(2352), "PARTIAL_A")
        for raw_sec, c2_b in reversed(raw_passes):
            ok, sec, _ = clean_single_pass(lba, raw_sec, c2_b)
            if ok:
                return True, 86, 52, sec, "A_SINGLE_C2"
            pz, qz = rspc.count_pq(sec)
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, sec, "PARTIAL_A")
        return best


@SolverRegistry.register
class SubgridPeelSolver(SectorSolverPlugin):
    name = "subgrid_peel"
    tier = 1
    description = "Top-K pass codeword-weighted voting + K43,26 subgrid peeling + weight-3 unmasking."

    def solve(self, lba, raw_passes, context=None):
        scored = sorted(
            raw_passes,
            key=lambda x: sum(1 for cw in rspc.P_IDX if rspc.syndromes_p(x[0], cw) == (0, 0)),
            reverse=True,
        )
        n = len(scored)
        ks = sorted(
            {k for k in (n, n - 1, round(0.8 * n), round(0.6 * n), n // 2, 6, 4, 3, 2) if 1 <= k <= n},
            reverse=True,
        )
        if n >= 8:
            ks.extend(k for k in range(n - 2, 1, -1) if k not in ks)

        best = (False, -1, -1, bytearray(2352), "PARTIAL_B")
        for c2_strict in (False, True):
            for k in ks:
                sub = scored[:k]
                for inc_q in (False, True):
                    ok, pz, qz, v = subgrid_peel_decode(lba, sub, inc_q, c2_strict)
                    if ok:
                        return True, 86, 52, v, f"B_SUBGRID_K{k}_Q{int(inc_q)}_S{int(c2_strict)}"
                    if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                        best = (False, pz, qz, v, f"PARTIAL_B_K{k}_Q{int(inc_q)}_S{int(c2_strict)}")
        return best


@SolverRegistry.register
class CleanedConsensusSolver(SectorSolverPlugin):
    name = "cleaned_consensus"
    tier = 2
    description = "C2-cleaned pass consensus across top-K subsets + erasure polish."

    def solve(self, lba, raw_passes, context=None):
        if len(raw_passes) < 2:
            return False, -1, -1, bytearray(2352), "PARTIAL_C"
        cleaned = []
        cand_vals = [set() for _ in range(2352)]
        for raw_sec, c2_b in raw_passes:
            ok, c_sec, c2_eras = clean_single_pass(lba, raw_sec, c2_b)
            if ok:
                return True, 86, 52, c_sec, "C2_SINGLE_PASS"
            pz, qz = rspc.count_pq(c_sec)
            cleaned.append((pz + qz, c_sec, c2_eras))
            for j in range(2352):
                cand_vals[j].add(c_sec[j])

        cleaned.sort(key=lambda x: x[0], reverse=True)
        n = len(cleaned)
        ks = sorted({n, max(2, n - 1), min(6, n), min(4, n), min(3, n), min(2, n)}, reverse=True)
        best = (False, -1, -1, cleaned[0][1], "PARTIAL_C")
        for k in ks:
            if k < 2:
                continue
            votes = [Counter() for _ in range(2352)]
            for _, sec, c2 in cleaned[:k]:
                p_ok = [rspc.syndromes_p(sec, cw) == (0, 0) for cw in rspc.P_IDX]
                q_ok = [rspc.syndromes_q(sec, cw) == (0, 0) for cw in rspc.Q_IDX]
                for j in range(16, 2352):
                    if j in rspc.FIXED_POS:
                        continue
                    pok = j in rspc.BYTE_TO_P and p_ok[rspc.BYTE_TO_P[j]]
                    qok = j in rspc.BYTE_TO_Q and q_ok[rspc.BYTE_TO_Q[j]]
                    w = 1 + (3 if j not in c2 else 0) + (8 if pok else 0) + (10 if qok else 0) + (25 if pok and qok else 0)
                    votes[j][sec[j]] += w
            voted = bytearray(2352)
            rspc.init_sector_header(lba, voted)
            unc = set()
            for j in range(16, 2352):
                if j in rspc.FIXED_POS:
                    continue
                mc = votes[j].most_common(2)
                voted[j] = mc[0][0]
                if len(mc) > 1 and mc[1][1] * 4 >= mc[0][1]:
                    unc.add(j)

            for _ in range(25):
                if rspc.check_and_fix_q_parity(voted):
                    return True, 86, 52, voted, f"CLEANED_CONSENSUS_K{k}"
                ch = 0
                for cw_list in (rspc.P_IDX, rspc.Q_IDX):
                    for cw in cw_list:
                        if rspc.syndromes(voted, cw) == (0, 0):
                            for pos in cw:
                                unc.discard(pos)
                            continue
                        st1, c1 = rspc.solve_1err(voted, cw)
                        if st1 == 1 and c1[0][0] in unc:
                            voted[c1[0][0]] = c1[0][1]
                            unc.discard(c1[0][0])
                            ch += 1
                            continue
                        eras = [i for i, pos in enumerate(cw) if pos in unc]
                        if len(eras) == 1 or len(eras) == 2:
                            ok, corrs = rspc.solve_2eras(voted, cw, eras)
                            if ok and (len(eras) == 1 or all(v in cand_vals[p] for p, v in corrs)):
                                for p, v in corrs:
                                    voted[p] = v
                                for i in eras:
                                    unc.discard(cw[i])
                                ch += 1
                if ch == 0:
                    break
            if rspc.check_and_fix_q_parity(voted):
                return True, 86, 52, voted, f"CLEANED_CONSENSUS_K{k}"
            pz, qz = rspc.count_pq(voted)
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, voted, f"PARTIAL_C_K{k}")
        return best


@SolverRegistry.register
class AnchorC2GraftSolver(SectorSolverPlugin):
    name = "anchor_c2_graft"
    tier = 2
    description = "Anchor on cleanest C2 pass, graft C2-clean bytes, un-miscorrect 1-byte errors."

    def solve(self, lba, raw_passes, context=None):
        if len(raw_passes) < 2:
            return False, -1, -1, bytearray(2352), "PARTIAL_D"
        passes = []
        c2_sets = []
        cand_vals = [set() for _ in range(2352)]
        for raw_sec, c2_b in raw_passes:
            sec = bytearray(raw_sec)
            rspc.init_sector_header(lba, sec)
            passes.append(sec)
            c2_sets.append(rspc.c2_set(c2_b))
            for j in range(2352):
                cand_vals[j].add(sec[j])

        order = sorted(
            range(len(passes)),
            key=lambda i: (len(c2_sets[i]), -sum(1 for cw in rspc.P_IDX if rspc.syndromes_p(passes[i], cw) == (0, 0))),
        )
        best = (False, -1, -1, passes[order[0]], "PARTIAL_D")
        for a_idx in order[: min(3, len(order))]:
            sec = bytearray(passes[a_idx])
            a_c2 = c2_sets[a_idx]
            unc = set()
            for pos in a_c2:
                c = Counter()
                for idx, osec in enumerate(passes):
                    if idx != a_idx and pos not in c2_sets[idx]:
                        w = 10
                        if pos in rspc.BYTE_TO_P and rspc.syndromes_p(osec, rspc.P_IDX[rspc.BYTE_TO_P[pos]]) == (0, 0):
                            w += 25
                        if pos in rspc.BYTE_TO_Q and rspc.syndromes_q(osec, rspc.Q_IDX[rspc.BYTE_TO_Q[pos]]) == (0, 0):
                            w += 30
                        c[osec[pos]] += w
                if c:
                    mc = c.most_common(2)
                    sec[pos] = mc[0][0]
                    if len(mc) > 1:
                        unc.add(pos)
                else:
                    unc.add(pos)

            for _ in range(25):
                if rspc.check_and_fix_q_parity(sec):
                    return True, 86, 52, sec, "D_ANCHOR_GRAFT"
                ch = 0
                for cw_list in (rspc.P_IDX, rspc.Q_IDX):
                    for cw in cw_list:
                        if rspc.syndromes(sec, cw) == (0, 0):
                            continue
                        eras = [i for i, pos in enumerate(cw) if pos in unc]
                        if 1 <= len(eras) <= 2:
                            ok, corrs = rspc.solve_2eras(sec, cw, eras)
                            if ok:
                                for pos, val in corrs:
                                    sec[pos] = val
                                for i in eras:
                                    unc.discard(cw[i])
                                ch += 1
                if ch:
                    continue
                for is_p, cw_list in ((True, rspc.P_IDX), (False, rspc.Q_IDX)):
                    for cw in cw_list:
                        st1, c1 = rspc.solve_1err(sec, cw)
                        if st1 == 1 and c1[0][0] not in rspc.FIXED_POS:
                            pos, val = c1[0]
                            old = sec[pos]
                            sec[pos] = val
                            orth = _orth_cw(is_p, pos)
                            if pos in a_c2 or val in cand_vals[pos] or orth is None or rspc.solve_1err(sec, orth)[0] in (0, 1):
                                unc.discard(pos)
                                ch += 1
                            else:
                                sec[pos] = old
                if ch == 0:
                    break
            if rspc.check_and_fix_q_parity(sec):
                return True, 86, 52, sec, "D_ANCHOR_GRAFT"
            pz, qz = rspc.count_pq(sec)
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, sec, "PARTIAL_D")
        return best
