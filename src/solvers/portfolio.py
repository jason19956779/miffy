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
    """Single-pass C2-erasure RSPC decoding + orthogonal 1-error polish (FIXED_POS locked)."""
    sec = bytearray(raw_sec)
    rspc.init_sector_header(lba, sec)
    c2_eras = rspc.c2_set(c2_bytes)
    eras = set(c2_eras)

    for _ in range(15):
        if rspc.check_and_fix_q_parity(sec, expected_lba=lba):
            return True, sec, c2_eras
        changed = 0
        for cw_list in (rspc.P_IDX, rspc.Q_IDX):
            for cw in cw_list:
                if rspc.syndromes(sec, cw) == (0, 0):
                    for pos in cw:
                        eras.discard(pos)
                    continue
                cw_eras = [i for i, pos in enumerate(cw) if pos in eras and pos not in rspc.FIXED_POS]
                if 1 <= len(cw_eras) <= 2:
                    ok, corrs = rspc.solve_2eras(sec, cw, cw_eras)
                    if ok and all(pos not in rspc.FIXED_POS for pos, _ in corrs):
                        for pos, val in corrs:
                            sec[pos] = val
                        for i in cw_eras:
                            eras.discard(cw[i])
                        changed += 1
                        continue
                st1, corr1 = rspc.solve_1err(sec, cw)
                if st1 == 1 and corr1[0][0] in eras and corr1[0][0] not in rspc.FIXED_POS:
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
    return rspc.check_and_fix_q_parity(sec, expected_lba=lba), sec, c2_eras


def _try_plane_sweep(
    comp: bytearray, candidate_vals: list[set[int]], expected_lba: int | None = None
) -> bytearray | None:
    test = bytearray(comp)
    for plane in (0, 1):
        p_bad = [i for i in range(plane, 86, 2) if rspc.syndromes_p(test, rspc.P_IDX[i]) != (0, 0)]
        q_bad = [i for i in range(plane, 52, 2) if rspc.syndromes_q(test, rspc.Q_IDX[i]) != (0, 0)]
        if 1 <= len(p_bad) <= 2 and q_bad:
            p_set = set(p_bad)
            mod_q = set()
            for q_idx in q_bad:
                cw = rspc.Q_IDX[q_idx]
                eras = [
                    i
                    for i, pos in enumerate(cw[:43])
                    if pos not in rspc.FIXED_POS and rspc.BYTE_TO_P[pos] in p_set
                ]
                ok, corrs = rspc.solve_2eras(test, cw, eras)
                if ok and corrs and all(pos not in rspc.FIXED_POS for pos, _ in corrs):
                    for pos, val in corrs:
                        test[pos] = val
                    mod_q.add(q_idx)
            for p_idx in p_bad:
                cw = rspc.P_IDX[p_idx]
                if rspc.syndromes_p(test, cw) == (0, 0):
                    continue
                st1, c1 = rspc.solve_1err(test, cw)
                if st1 == 1 and c1[0][0] not in rspc.FIXED_POS:
                    test[c1[0][0]] = c1[0][1]
                else:
                    cand_i = [
                        i
                        for i, pos in enumerate(cw)
                        if pos not in rspc.FIXED_POS and rspc.BYTE_TO_Q.get(pos) in mod_q
                    ]
                    if 2 <= len(cand_i) <= 12:
                        for i0, i1 in combinations(cand_i, 2):
                            ok, corrs = rspc.solve_2eras(test, cw, [i0, i1])
                            if (
                                ok
                                and len(corrs) == 2
                                and all(p not in rspc.FIXED_POS and v in candidate_vals[p] for p, v in corrs)
                            ):
                                for p, v in corrs:
                                    test[p] = v
                                break
        elif 1 <= len(q_bad) <= 2 and p_bad:
            q_set = set(q_bad)
            for p_idx in p_bad:
                cw = rspc.P_IDX[p_idx]
                eras = [
                    i
                    for i, pos in enumerate(cw)
                    if pos not in rspc.FIXED_POS and pos in rspc.BYTE_TO_Q and rspc.BYTE_TO_Q[pos] in q_set
                ]
                ok, corrs = rspc.solve_2eras(test, cw, eras)
                if ok and corrs and all(pos not in rspc.FIXED_POS for pos, _ in corrs):
                    for pos, val in corrs:
                        test[pos] = val
    return test if rspc.check_and_fix_q_parity(test, expected_lba=expected_lba) else None


def subgrid_peel_decode(
    lba: int,
    raw_passes: list[tuple[bytes, bytes]],
    include_q_parity_in_uncov: bool = True,
    c2_strict: bool = False,
    pass_cache: list[dict] | None = None,
    indices: list[int] | None = None,
) -> tuple[bool, int, int, bytearray]:
    """Codeword-weighted vote + K43,26 subgrid peeling + weight-3 unmasking (FIXED_POS locked)."""
    if pass_cache is None:
        pass_cache = rspc.build_pass_cache(lba, raw_passes)
    sel = [pass_cache[i] for i in indices] if indices is not None else pass_cache
    if not sel:
        comp = bytearray(2352)
        rspc.init_sector_header(lba, comp)
        return False, 0, 0, comp

    c2_union: set[int] = set()
    candidate_vals: list[set[int]] = [set() for _ in range(2352)]
    votes = [Counter() for _ in range(2352)]

    for entry in sel:
        sec = entry["sec"]
        c2 = entry["c2"]
        c2_union.update(c2)
        for j in range(16, 2352):
            candidate_vals[j].add(sec[j])

        if c2_strict:
            p_ok = [entry["p_ok_raw"][i] and entry["p_c2_hits"][i] < 2 for i in range(86)]
            q_ok = [entry["q_ok_raw"][i] and entry["q_c2_hits"][i] < 2 for i in range(52)]
        else:
            p_ok = entry["p_ok_raw"]
            q_ok = entry["q_ok_raw"]

        for pos, val in entry["agreed_1err"].items():
            p_idx = rspc.BYTE_TO_P.get(pos)
            q_idx = rspc.BYTE_TO_Q.get(pos)
            if p_idx is not None and q_idx is not None and not p_ok[p_idx] and not q_ok[q_idx]:
                votes[pos][val] += 80
                candidate_vals[pos].add(val)

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
        if rspc.check_and_fix_q_parity(comp, expected_lba=lba):
            return True, 86, 52, comp
        swept = _try_plane_sweep(comp, candidate_vals, expected_lba=lba)
        if swept is not None:
            return True, 86, 52, swept

        p_bad = {i for i, cw in enumerate(rspc.P_IDX) if rspc.syndromes_p(comp, cw) != (0, 0)}
        q_bad = {i for i, cw in enumerate(rspc.Q_IDX) if rspc.syndromes_q(comp, cw) != (0, 0)}

        q_par_fixed = 0
        for idx in list(q_bad):
            st1, c1 = rspc.solve_1err(comp, rspc.Q_IDX[idx])
            if st1 == 1 and c1[0][0] >= 2248 and c1[0][0] not in rspc.FIXED_POS:
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
                eras = [i for i, pos in enumerate(cw) if pos in uncov and pos not in rspc.FIXED_POS]
                if len(eras) == 1 or (include_q_parity_in_uncov and len(eras) == 2):
                    ok, corrs = rspc.solve_2eras(comp, cw, eras)
                    if ok and corrs and all(pos not in rspc.FIXED_POS for pos, _ in corrs):
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
                if st1 == 1 and c1[0][0] in uncov and c1[0][0] not in rspc.FIXED_POS:
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
                ec = [i for i, pos in enumerate(cw) if pos in uncov and pos not in rspc.FIXED_POS]
                if 2 <= len(ec) <= 10:
                    for i0, i1 in combinations(ec, 2):
                        ok, corrs = rspc.solve_2eras(comp, cw, [i0, i1])
                        if not ok or len(corrs) != 2 or any(p in rspc.FIXED_POS for p, _ in corrs):
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

    if rspc.check_and_fix_q_parity(comp, expected_lba=lba):
        return True, 86, 52, comp
    pz, qz = rspc.count_pq(comp)
    return False, pz, qz, comp


def _get_sorted_pass_cache(
    lba: int, raw_passes: list[tuple[bytes, bytes]], context: dict | None
) -> tuple[list[dict], list[int]]:
    if context is not None and "pass_cache" in context:
        pass_cache = context["pass_cache"]
    else:
        pass_cache = rspc.build_pass_cache(lba, raw_passes)
        if context is not None:
            context["pass_cache"] = pass_cache
    order = sorted(range(len(pass_cache)), key=lambda i: pass_cache[i]["p_ok_count"], reverse=True)
    return pass_cache, order


@SolverRegistry.register
class SinglePassC2Solver(SectorSolverPlugin):
    name = "single_c2"
    tier = 1
    description = "Single-pass C2 erasure decoding + orthogonal 1-error polish (latest passes first)."

    def solve(self, lba, raw_passes, context=None):
        best = (False, -1, -1, bytearray(2352), "PARTIAL_A")
        for raw_sec, c2_b in reversed(raw_passes):
            if rspc.is_frame_slip(lba, raw_sec):
                continue
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
    description = "Fast top-K pass codeword-weighted voting + K43,26 subgrid peeling using shared pass cache."

    def solve(self, lba, raw_passes, context=None):
        filtered = [(r, c) for r, c in raw_passes if not rspc.is_frame_slip(lba, r)]
        if not filtered:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, 0, 0, sec, "PARTIAL_B_SLIP"
        pass_cache, order = _get_sorted_pass_cache(lba, filtered, context)
        n = len(order)
        fast_ks = sorted(
            {k for k in (n, n - 1, max(2, n // 2), 6, 4, 2) if 1 <= k <= n},
            reverse=True,
        )

        best = (False, -1, -1, bytearray(2352), "PARTIAL_B")
        for k in fast_ks:
            ok, pz, qz, v = subgrid_peel_decode(
                lba, filtered, include_q_parity_in_uncov=False, c2_strict=False, pass_cache=pass_cache, indices=order[:k]
            )
            if ok:
                return True, 86, 52, v, f"B_SUBGRID_K{k}_Q0_S0"
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, v, f"PARTIAL_B_K{k}_Q0_S0")

        for k in fast_ks[:3]:
            ok, pz, qz, v = subgrid_peel_decode(
                lba, filtered, include_q_parity_in_uncov=True, c2_strict=False, pass_cache=pass_cache, indices=order[:k]
            )
            if ok:
                return True, 86, 52, v, f"B_SUBGRID_K{k}_Q1_S0"
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, v, f"PARTIAL_B_K{k}_Q1_S0")
        return best


@SolverRegistry.register
class SubgridPeelExhaustiveSolver(SectorSolverPlugin):
    name = "subgrid_peel_exhaustive"
    tier = 3
    description = "Exhaustive k-subset and strict-C2 K43,26 subgrid peel sweep over stubborn sectors."

    def solve(self, lba, raw_passes, context=None):
        filtered = [(r, c) for r, c in raw_passes if not rspc.is_frame_slip(lba, r)]
        if len(filtered) < 2:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, 0, 0, sec, "PARTIAL_B_EX"
        pass_cache, order = _get_sorted_pass_cache(lba, filtered, context)
        n = len(order)
        all_ks = sorted(
            {k for k in (n, n - 1, round(0.8 * n), round(0.6 * n), n // 2, 6, 4, 3, 2) if 1 <= k <= n},
            reverse=True,
        )
        if n >= 8:
            all_ks.extend(k for k in range(n - 2, 1, -1) if k not in all_ks)
        fast_ks_s0 = sorted({k for k in (n, n - 1, max(2, n // 2), 6, 4, 2) if 1 <= k <= n}, reverse=True)
        already_q0_s0 = set(fast_ks_s0)
        already_q1_s0 = set(fast_ks_s0[:3])

        best = (False, -1, -1, bytearray(2352), "PARTIAL_B_EX")
        for c2_strict in (False, True):
            for k in all_ks:
                for inc_q in (False, True):
                    if not c2_strict and ((not inc_q and k in already_q0_s0) or (inc_q and k in already_q1_s0)):
                        continue
                    ok, pz, qz, v = subgrid_peel_decode(
                        lba,
                        filtered,
                        include_q_parity_in_uncov=inc_q,
                        c2_strict=c2_strict,
                        pass_cache=pass_cache,
                        indices=order[:k],
                    )
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
                if rspc.check_and_fix_q_parity(voted, expected_lba=lba):
                    return True, 86, 52, voted, f"CLEANED_CONSENSUS_K{k}"
                ch = 0
                for cw_list in (rspc.P_IDX, rspc.Q_IDX):
                    for cw in cw_list:
                        if rspc.syndromes(voted, cw) == (0, 0):
                            for pos in cw:
                                unc.discard(pos)
                            continue
                        st1, c1 = rspc.solve_1err(voted, cw)
                        if st1 == 1 and c1[0][0] in unc and c1[0][0] not in rspc.FIXED_POS:
                            voted[c1[0][0]] = c1[0][1]
                            unc.discard(c1[0][0])
                            ch += 1
                            continue
                        eras = [i for i, pos in enumerate(cw) if pos in unc and pos not in rspc.FIXED_POS]
                        if len(eras) == 1 or len(eras) == 2:
                            ok, corrs = rspc.solve_2eras(voted, cw, eras)
                            if ok and all(p not in rspc.FIXED_POS for p, _ in corrs) and (
                                len(eras) == 1 or all(v in cand_vals[p] for p, v in corrs)
                            ):
                                for p, v in corrs:
                                    voted[p] = v
                                for i in eras:
                                    unc.discard(cw[i])
                                ch += 1
                if ch == 0:
                    break
            if rspc.check_and_fix_q_parity(voted, expected_lba=lba):
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
                if rspc.check_and_fix_q_parity(sec, expected_lba=lba):
                    return True, 86, 52, sec, "D_ANCHOR_GRAFT"
                ch = 0
                for cw_list in (rspc.P_IDX, rspc.Q_IDX):
                    for cw in cw_list:
                        if rspc.syndromes(sec, cw) == (0, 0):
                            continue
                        eras = [i for i, pos in enumerate(cw) if pos in unc and pos not in rspc.FIXED_POS]
                        if 1 <= len(eras) <= 2:
                            ok, corrs = rspc.solve_2eras(sec, cw, eras)
                            if ok and all(pos not in rspc.FIXED_POS for pos, _ in corrs):
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
            if rspc.check_and_fix_q_parity(sec, expected_lba=lba):
                return True, 86, 52, sec, "D_ANCHOR_GRAFT"
            pz, qz = rspc.count_pq(sec)
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, sec, "PARTIAL_D")
        return best
