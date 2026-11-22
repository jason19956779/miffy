#!/usr/bin/env python3
"""Raw SCSI MMC READ CD (0xBE, Flags=0xFA) + C2 + RSPC P/Q/EDC CD recovery engine."""

import argparse
import csv
import ctypes
import fcntl
import json
import os
import re
import shutil
import sqlite3
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
MIFFY_DIR = REPO_DIR / "recovered" / "MIFFY"
SRC_ISO = MIFFY_DIR / "miffy1.iso"
REC_ISO = MIFFY_DIR / "miffy_recovered.iso"
BAD_LOG_PATH = MIFFY_DIR / "bad sector.txt"
LOGS_DIR = MIFFY_DIR / "logs"
DB_PATH = LOGS_DIR / "raw_passes.sqlite3"
STATE_PATH = LOGS_DIR / "recovery_state.json"
AFFECTED_CSV = LOGS_DIR / "affected_files.csv"
SG_DEV = "/dev/sg0"

TOTAL_VOLUME_SECTORS = 124079
TOTAL_VOLUME_BYTES = TOTAL_VOLUME_SECTORS * 2048
MAX_BATCH_SECTORS = 44  # 44 * 2646 = 116,424 B < 120 KB sysfs max_sectors_kb
SG_IO = 0x2285

RANGE_RE = re.compile(
    r"0x([0-9A-Fa-f]+)\s*-\s*0x([0-9A-Fa-f]+)\s*\((\d+)\s*sectors\)"
)

# ---------------------------------------------------------------------------
# ECMA-130 GF(2^8), 32-bit EDC, and RSPC P/Q Tables
# ---------------------------------------------------------------------------

GF_EXP = [0] * 512
GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    GF_EXP[_i] = _x
    GF_EXP[_i + 255] = _x
    GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D

GF_MUL = tuple(
    tuple(
        0 if (a == 0 or b == 0) else GF_EXP[GF_LOG[a] + GF_LOG[b]]
        for b in range(256)
    )
    for a in range(256)
)


def gf_div(a: int, b: int) -> int:
    if a == 0:
        return 0
    return GF_EXP[GF_LOG[a] - GF_LOG[b] + 255]


EDC_TABLE = [0] * 256
for _i in range(256):
    _edc = _i
    for _ in range(8):
        _edc = ((_edc >> 1) ^ 0xD8018001) if (_edc & 1) else (_edc >> 1)
    EDC_TABLE[_i] = _edc
EDC_TABLE = tuple(EDC_TABLE)


def compute_edc(data_2064: bytes | bytearray | memoryview) -> int:
    edc = 0
    tbl = EDC_TABLE
    for b in data_2064:
        edc = (edc >> 8) ^ tbl[(edc ^ b) & 0xFF]
    return edc


P_IDX: list[tuple[int, ...]] = []
BYTE_TO_P: dict[int, int] = {}
for _col in range(43):
    for _plane in range(2):
        _cw = [12 + (2 * (43 * _row + _col) + _plane) for _row in range(24)]
        _cw.append(2076 + 2 * _col + _plane)
        _cw.append(2076 + 86 + 2 * _col + _plane)
        _idx = len(P_IDX)
        P_IDX.append(tuple(_cw))
        for _pos in _cw:
            BYTE_TO_P[_pos] = _idx

Q_IDX: list[tuple[int, ...]] = []
BYTE_TO_Q: dict[int, int] = {}
for _row in range(26):
    for _plane in range(2):
        _cw = [
            12 + (2 * ((43 * ((_row + _col) % 26) + _col) % 1118) + _plane)
            for _col in range(43)
        ]
        _cw.append(2248 + 2 * _row + _plane)
        _cw.append(2248 + 52 + 2 * _row + _plane)
        _idx = len(Q_IDX)
        Q_IDX.append(tuple(_cw))
        for _pos in _cw:
            BYTE_TO_Q[_pos] = _idx

P_WEIGHTS = tuple(GF_EXP[25 - i] for i in range(26))
Q_WEIGHTS = tuple(GF_EXP[44 - i] for i in range(45))
FIXED_POS = frozenset(range(0, 16)) | frozenset(range(2068, 2076))
SYNC_12 = b"\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00"


def syndromes_p(sec: bytes | bytearray, cw: tuple[int, ...]) -> tuple[int, int]:
    s0 = s1 = 0
    mul = GF_MUL
    w = P_WEIGHTS
    for i in range(26):
        v = sec[cw[i]]
        if v:
            s0 ^= v
            s1 ^= mul[v][w[i]]
    return s0, s1


def syndromes_q(sec: bytes | bytearray, cw: tuple[int, ...]) -> tuple[int, int]:
    s0 = s1 = 0
    mul = GF_MUL
    w = Q_WEIGHTS
    for i in range(45):
        v = sec[cw[i]]
        if v:
            s0 ^= v
            s1 ^= mul[v][w[i]]
    return s0, s1


def syndromes(sec: bytes | bytearray, cw: tuple[int, ...]) -> tuple[int, int]:
    return syndromes_p(sec, cw) if len(cw) == 26 else syndromes_q(sec, cw)


def solve_1err(sec: bytes | bytearray, cw: tuple[int, ...]) -> tuple[int, list[tuple[int, int]]]:
    s0, s1 = syndromes(sec, cw)
    if s0 == 0 and s1 == 0:
        return 0, []
    if s0 == 0 or s1 == 0:
        return -1, []
    log_pos = (GF_LOG[s1] - GF_LOG[s0]) % 255
    n = len(cw)
    if log_pos >= n:
        return -1, []
    i = n - 1 - log_pos
    pos = cw[i]
    return 1, [(pos, sec[pos] ^ s0)]


def solve_2eras(
    sec: bytes | bytearray, cw: tuple[int, ...], eras_i: list[int]
) -> tuple[bool, list[tuple[int, int]]]:
    s0, s1 = syndromes(sec, cw)
    if s0 == 0 and s1 == 0:
        return True, []
    n = len(cw)
    if len(eras_i) == 1:
        i0 = eras_i[0]
        if GF_MUL[s0][GF_EXP[n - 1 - i0]] == s1:
            return True, [(cw[i0], sec[cw[i0]] ^ s0)]
        return False, []
    if len(eras_i) == 2:
        i0, i1 = eras_i
        a0 = GF_EXP[n - 1 - i0]
        a1 = GF_EXP[n - 1 - i1]
        e1 = gf_div(s1 ^ GF_MUL[a0][s0], a0 ^ a1)
        e0 = s0 ^ e1
        out = []
        if e0 != 0:
            out.append((cw[i0], sec[cw[i0]] ^ e0))
        if e1 != 0:
            out.append((cw[i1], sec[cw[i1]] ^ e1))
        return True, out
    return False, []


def to_bcd(x: int) -> int:
    return ((x // 10) << 4) | (x % 10)


def lba_header(lba: int) -> bytes:
    frames = lba + 150
    return bytes(
        [
            to_bcd(frames // 4500),
            to_bcd((frames % 4500) // 75),
            to_bcd(frames % 75),
            0x01,
        ]
    )


def count_pq(sec: bytes | bytearray) -> tuple[int, int]:
    pz = sum(1 for cw in P_IDX if syndromes_p(sec, cw) == (0, 0))
    qz = sum(1 for cw in Q_IDX if syndromes_q(sec, cw) == (0, 0))
    return pz, qz


def check_and_fix_q_parity(sec: bytearray) -> bool:
    """Exact verification oracle: 86/86 P-codewords == 0 AND 32-bit EDC match."""
    if not all(syndromes_p(sec, cw) == (0, 0) for cw in P_IDX):
        return False
    if compute_edc(sec[:2064]) != int.from_bytes(sec[2064:2068], "little"):
        return False
    for cw in Q_IDX:
        if syndromes_q(sec, cw) != (0, 0):
            ok, corrs = solve_2eras(sec, cw, [43, 44])
            if ok:
                for pos, val in corrs:
                    sec[pos] = val
    return all(syndromes_q(sec, cw) == (0, 0) for cw in Q_IDX)


# ---------------------------------------------------------------------------
# Three-Strategy Portfolio Decoder
# ---------------------------------------------------------------------------


def clean_single_pass(
    lba: int, raw_sec: bytes, c2_bytes: bytes
) -> tuple[bool, bytearray, set[int]]:
    """Strategy A: Single-pass C2-erasure + orthogonal-confirmed 1-error polish."""
    sec = bytearray(raw_sec)
    sec[0:12] = SYNC_12
    sec[12:16] = lba_header(lba)
    sec[2068:2076] = b"\x00" * 8
    c2_eras = {
        j
        for j in range(16, 2352)
        if ((c2_bytes[j // 8] >> (7 - (j % 8))) & 1) and j not in FIXED_POS
    }
    eras = set(c2_eras)

    for _ in range(15):
        if check_and_fix_q_parity(sec):
            return True, sec, c2_eras
        changed = 0
        for cw_list in (P_IDX, Q_IDX):
            for cw in cw_list:
                s0, s1 = syndromes(sec, cw)
                if s0 == 0 and s1 == 0:
                    for pos in cw:
                        eras.discard(pos)
                    continue
                cw_eras = [i for i, pos in enumerate(cw) if pos in eras]
                if 1 <= len(cw_eras) <= 2:
                    ok, corrs = solve_2eras(sec, cw, cw_eras)
                    if ok:
                        for pos, val in corrs:
                            sec[pos] = val
                        for i in cw_eras:
                            eras.discard(cw[i])
                        changed += 1
                        continue
                st1, corr1 = solve_1err(sec, cw)
                if st1 == 1:
                    pos, val = corr1[0]
                    if pos in eras:
                        sec[pos] = val
                        eras.discard(pos)
                        changed += 1
        if changed == 0:
            for is_p, cw_list in ((True, P_IDX), (False, Q_IDX)):
                for cw in cw_list:
                    st1, corr1 = solve_1err(sec, cw)
                    if st1 == 1:
                        pos, val = corr1[0]
                        if pos not in FIXED_POS:
                            old = sec[pos]
                            sec[pos] = val
                            orth = (
                                Q_IDX[BYTE_TO_Q[pos]]
                                if is_p and pos in BYTE_TO_Q
                                else (P_IDX[BYTE_TO_P[pos]] if not is_p and pos in BYTE_TO_P else None)
                            )
                            if orth is not None and syndromes(sec, orth) == (0, 0):
                                eras.discard(pos)
                                changed += 1
                            else:
                                sec[pos] = old
        if changed == 0:
            break
    return check_and_fix_q_parity(sec), sec, c2_eras


def _try_plane_sweep(comp: bytearray, candidate_vals: list[set[int]]) -> bytearray | None:
    """Solve remaining 1-2 failing P or Q columns/rows per plane; commit only if verified."""
    test = bytearray(comp)
    for plane in (0, 1):
        p_bad_pl = [idx for idx in range(plane, 86, 2) if syndromes_p(test, P_IDX[idx]) != (0, 0)]
        q_bad_pl = [idx for idx in range(plane, 52, 2) if syndromes_q(test, Q_IDX[idx]) != (0, 0)]
        if 1 <= len(p_bad_pl) <= 2 and q_bad_pl:
            p_set = set(p_bad_pl)
            modified_q = set()
            for q_idx in q_bad_pl:
                cw = Q_IDX[q_idx]
                eras = [i for i, pos in enumerate(cw[:43]) if BYTE_TO_P[pos] in p_set]
                ok, corrs = solve_2eras(test, cw, eras)
                if ok and corrs:
                    for pos, val in corrs:
                        test[pos] = val
                    modified_q.add(q_idx)
            for p_idx in p_bad_pl:
                cw = P_IDX[p_idx]
                if syndromes_p(test, cw) == (0, 0):
                    continue
                st1, c1 = solve_1err(test, cw)
                if st1 == 1:
                    test[c1[0][0]] = c1[0][1]
                else:
                    cand_i = [i for i, pos in enumerate(cw) if BYTE_TO_Q.get(pos) in modified_q]
                    if 2 <= len(cand_i) <= 12:
                        for i0, i1 in combinations(cand_i, 2):
                            ok, corrs = solve_2eras(test, cw, [i0, i1])
                            if ok and len(corrs) == 2 and all(v in candidate_vals[p] for p, v in corrs):
                                for p, v in corrs:
                                    test[p] = v
                                break
        elif 1 <= len(q_bad_pl) <= 2 and p_bad_pl:
            q_set = set(q_bad_pl)
            for p_idx in p_bad_pl:
                cw = P_IDX[p_idx]
                eras = [i for i, pos in enumerate(cw) if pos in BYTE_TO_Q and BYTE_TO_Q[pos] in q_set]
                ok, corrs = solve_2eras(test, cw, eras)
                if ok and corrs:
                    for pos, val in corrs:
                        test[pos] = val
    if check_and_fix_q_parity(test):
        return test
    return None


def _subgrid_peel_decode(
    lba: int,
    raw_passes: list[tuple[bytes, bytes]],
    include_q_parity_in_uncov: bool = True,
    c2_strict: bool = False,
) -> tuple[bool, int, int, bytearray]:
    """Strategy B: Unforged codeword-weighted vote + K43,26 subgrid peeling + weight-3 unmasking."""
    hdr4 = lba_header(lba)
    passes: list[bytearray] = []
    c2_sets: list[set[int]] = []
    c2_union: set[int] = set()
    candidate_vals: list[set[int]] = [set() for _ in range(2352)]

    for raw_sec, c2_bytes in raw_passes:
        sec = bytearray(raw_sec)
        sec[0:12] = SYNC_12
        sec[12:16] = hdr4
        sec[2068:2076] = b"\x00" * 8
        c2_eras = {
            j
            for j in range(16, 2352)
            if ((c2_bytes[j // 8] >> (7 - (j % 8))) & 1) and j not in FIXED_POS
        }
        passes.append(sec)
        c2_sets.append(c2_eras)
        c2_union.update(c2_eras)
        for j in range(2352):
            candidate_vals[j].add(sec[j])

    votes = [Counter() for _ in range(2352)]
    for p_idx, sec in enumerate(passes):
        c2_eras = c2_sets[p_idx]
        p_ok = []
        for cw in P_IDX:
            ok = syndromes_p(sec, cw) == (0, 0)
            if ok and c2_strict and sum(1 for pos in cw if pos in c2_eras) >= 2:
                ok = False
            p_ok.append(ok)
        q_ok = []
        for cw in Q_IDX:
            ok = syndromes_q(sec, cw) == (0, 0)
            if ok and c2_strict and sum(1 for pos in cw if pos in c2_eras) >= 2:
                ok = False
            q_ok.append(ok)

        p_1err = {}
        for idx, cw in enumerate(P_IDX):
            if not p_ok[idx]:
                st1, c1 = solve_1err(sec, cw)
                if st1 == 1:
                    p_1err[c1[0][0]] = c1[0][1]
        q_1err = {}
        for idx, cw in enumerate(Q_IDX):
            if not q_ok[idx]:
                st1, c1 = solve_1err(sec, cw)
                if st1 == 1:
                    q_1err[c1[0][0]] = c1[0][1]
        for pos, val in p_1err.items():
            if q_1err.get(pos) == val:
                votes[pos][val] += 80
                candidate_vals[pos].add(val)

        for j in range(16, 2352):
            if j in FIXED_POS:
                continue
            pok = j in BYTE_TO_P and p_ok[BYTE_TO_P[j]]
            qok = j in BYTE_TO_Q and q_ok[BYTE_TO_Q[j]]
            c2ok = j not in c2_eras
            if c2_strict:
                if pok and qok:
                    w = 120 if c2ok else 60
                elif qok:
                    w = 35 if c2ok else 12
                elif pok:
                    w = 30 if c2ok else 10
                else:
                    w = 5 if c2ok else 1
            else:
                if pok and qok:
                    w = 120
                elif qok:
                    w = 35 if c2ok else 25
                elif pok:
                    w = 30 if c2ok else 20
                else:
                    w = 4 if c2ok else 1
            votes[j][sec[j]] += w

    comp = bytearray(2352)
    comp[0:12] = SYNC_12
    comp[12:16] = hdr4
    comp[2068:2076] = b"\x00" * 8
    for j in range(16, 2352):
        if j not in FIXED_POS:
            comp[j] = votes[j].most_common(1)[0][0]

    used_unmask: set[tuple[int, int]] = set()
    for _ in range(45):
        if check_and_fix_q_parity(comp):
            return True, 86, 52, comp
        swept = _try_plane_sweep(comp, candidate_vals)
        if swept is not None:
            return True, 86, 52, swept

        p_bad = {idx for idx, cw in enumerate(P_IDX) if syndromes_p(comp, cw) != (0, 0)}
        q_bad = {idx for idx, cw in enumerate(Q_IDX) if syndromes_q(comp, cw) != (0, 0)}

        # Pre-clean 1-byte errors in Q-parity bytes (pos >= 2248) so Q can peel P
        q_par_fixed = 0
        for idx in list(q_bad):
            st1, c1 = solve_1err(comp, Q_IDX[idx])
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
                if j not in FIXED_POS
                and (
                    (j in BYTE_TO_P and BYTE_TO_P[j] in p_bad and j in BYTE_TO_Q and BYTE_TO_Q[j] in q_bad)
                    or (j >= 2248 and BYTE_TO_Q[j] in q_bad)
                )
            }
        else:
            uncov = {
                j
                for j in range(16, 2248)
                if j not in FIXED_POS and BYTE_TO_P[j] in p_bad and BYTE_TO_Q[j] in q_bad
            }
        changed = 0

        for is_p, bad_set, cw_list in ((True, p_bad, P_IDX), (False, q_bad, Q_IDX)):
            for idx in list(bad_set):
                cw = cw_list[idx]
                if syndromes(comp, cw) == (0, 0):
                    continue
                eras = [i for i, pos in enumerate(cw) if pos in uncov]
                if len(eras) == 1 or (include_q_parity_in_uncov and len(eras) == 2):
                    ok, corrs = solve_2eras(comp, cw, eras)
                    if ok and corrs:
                        for pos, val in corrs:
                            comp[pos] = val
                            uncov.discard(pos)
                        changed += 1
        if changed:
            continue

        for is_p, bad_set, cw_list in ((True, p_bad, P_IDX), (False, q_bad, Q_IDX)):
            for idx in list(bad_set):
                cw = cw_list[idx]
                st1, c1 = solve_1err(comp, cw)
                if st1 == 1 and c1[0][0] in uncov:
                    pos, val = c1[0]
                    old = comp[pos]
                    comp[pos] = val
                    orth = (
                        Q_IDX[BYTE_TO_Q[pos]]
                        if is_p and pos in BYTE_TO_Q
                        else (P_IDX[BYTE_TO_P[pos]] if not is_p and pos in BYTE_TO_P else None)
                    )
                    orth_ok = (
                        (orth is None)
                        or (syndromes(comp, orth) == (0, 0))
                        or (solve_1err(comp, orth)[0] == 1 and solve_1err(comp, orth)[1][0][0] in uncov)
                    )
                    if (val in candidate_vals[pos]) or orth_ok:
                        uncov.discard(pos)
                        changed += 1
                    else:
                        comp[pos] = old
        if changed:
            continue

        for is_p, bad_set, cw_list in ((True, p_bad, P_IDX), (False, q_bad, Q_IDX)):
            for idx in list(bad_set):
                cw = cw_list[idx]
                if syndromes(comp, cw) == (0, 0):
                    continue
                eras_candidates = [i for i, pos in enumerate(cw) if pos in uncov]
                if 2 <= len(eras_candidates) <= 10:
                    for i0, i1 in combinations(eras_candidates, 2):
                        ok, corrs = solve_2eras(comp, cw, [i0, i1])
                        if not ok or len(corrs) != 2:
                            continue
                        (pos0, val0), (pos1, val1) = corrs
                        in_cand0 = val0 in candidate_vals[pos0]
                        in_cand1 = val1 in candidate_vals[pos1]
                        old0, old1 = comp[pos0], comp[pos1]
                        comp[pos0], comp[pos1] = val0, val1
                        orth0 = (
                            Q_IDX[BYTE_TO_Q[pos0]]
                            if is_p and pos0 in BYTE_TO_Q
                            else (P_IDX[BYTE_TO_P[pos0]] if not is_p and pos0 in BYTE_TO_P else None)
                        )
                        orth1 = (
                            Q_IDX[BYTE_TO_Q[pos1]]
                            if is_p and pos1 in BYTE_TO_Q
                            else (P_IDX[BYTE_TO_P[pos1]] if not is_p and pos1 in BYTE_TO_P else None)
                        )
                        orth0_zero = orth0 is not None and syndromes(comp, orth0) == (0, 0)
                        orth1_zero = orth1 is not None and syndromes(comp, orth1) == (0, 0)
                        orth0_1err = (
                            orth0 is not None
                            and solve_1err(comp, orth0)[0] == 1
                            and solve_1err(comp, orth0)[1][0][0] in uncov
                        )
                        orth1_1err = (
                            orth1 is not None
                            and solve_1err(comp, orth1)[0] == 1
                            and solve_1err(comp, orth1)[1][0][0] in uncov
                        )
                        if (
                            (orth0_zero and orth1_zero)
                            or (orth0_zero and (in_cand1 or orth1_1err))
                            or (orth1_zero and (in_cand0 or orth0_1err))
                            or (in_cand0 and in_cand1 and (len(eras_candidates) == 2 or orth0_1err or orth1_1err))
                        ):
                            changed += 1
                            break
                        comp[pos0], comp[pos1] = old0, old1
                if changed:
                    break
            if changed:
                break
        if changed:
            continue

        # Weight-3 mis-correction unmasking across orthogonal zero-syndrome codewords
        small_residual = (len(p_bad) + len(q_bad)) <= 8
        q_to_p: dict[int, list[tuple[int, int]]] = {}
        for q_i in q_bad:
            st1, c1 = solve_1err(comp, Q_IDX[q_i])
            if st1 == 1 and c1[0][0] not in FIXED_POS and c1[0][0] in BYTE_TO_P:
                pos, val = c1[0]
                if (pos, val) not in used_unmask:
                    q_to_p.setdefault(BYTE_TO_P[pos], []).append((pos, val))
        for _, corrs in sorted(q_to_p.items(), key=lambda x: len(x[1]), reverse=True):
            if len(corrs) >= 2 or (
                len(corrs) == 1
                and (
                    corrs[0][1] in candidate_vals[corrs[0][0]]
                    or (small_residual and corrs[0][0] in c2_union)
                )
            ):
                for pos, val in corrs:
                    comp[pos] = val
                    used_unmask.add((pos, val))
                changed += 1
        if changed:
            continue

        p_to_q: dict[int, list[tuple[int, int]]] = {}
        for p_i in p_bad:
            st1, c1 = solve_1err(comp, P_IDX[p_i])
            if st1 == 1 and c1[0][0] not in FIXED_POS and c1[0][0] in BYTE_TO_Q:
                pos, val = c1[0]
                if (pos, val) not in used_unmask:
                    p_to_q.setdefault(BYTE_TO_Q[pos], []).append((pos, val))
        for _, corrs in sorted(p_to_q.items(), key=lambda x: len(x[1]), reverse=True):
            if len(corrs) >= 2 or (
                len(corrs) == 1
                and (
                    corrs[0][1] in candidate_vals[corrs[0][0]]
                    or (small_residual and corrs[0][0] in c2_union)
                )
            ):
                for pos, val in corrs:
                    comp[pos] = val
                    used_unmask.add((pos, val))
                changed += 1
        if changed == 0:
            break

    if check_and_fix_q_parity(comp):
        return True, 86, 52, comp
    pz, qz = count_pq(comp)
    return False, pz, qz, comp


def _polish_consensus(
    lba: int,
    voted: bytearray,
    uncertain: set[int],
    candidate_vals: list[set[int]],
) -> tuple[bool, int, int, bytearray]:
    """Strategy C helper: Polish a weighted consensus sector over cleaned passes."""
    voted = bytearray(voted)
    voted[0:12] = SYNC_12
    voted[12:16] = lba_header(lba)
    voted[2068:2076] = b"\x00" * 8
    uncertain = set(uncertain) - FIXED_POS

    for _ in range(25):
        if check_and_fix_q_parity(voted):
            return True, 86, 52, voted
        changed = 0
        for cw_list in (P_IDX, Q_IDX):
            for cw in cw_list:
                s0, s1 = syndromes(voted, cw)
                if s0 == 0 and s1 == 0:
                    for pos in cw:
                        uncertain.discard(pos)
                    continue
                st1, corr1 = solve_1err(voted, cw)
                if st1 == 1 and corr1[0][0] in uncertain:
                    pos, val = corr1[0]
                    voted[pos] = val
                    uncertain.discard(pos)
                    changed += 1
                    continue
                eras = [i for i, pos in enumerate(cw) if pos in uncertain]
                if len(eras) == 1:
                    ok, corrs = solve_2eras(voted, cw, eras)
                    if ok:
                        for pos, val in corrs:
                            voted[pos] = val
                        uncertain.discard(cw[eras[0]])
                        changed += 1
                        continue
                elif len(eras) == 2:
                    ok, corrs = solve_2eras(voted, cw, eras)
                    if ok and all(val in candidate_vals[pos] for pos, val in corrs):
                        for pos, val in corrs:
                            voted[pos] = val
                        for i in eras:
                            uncertain.discard(cw[i])
                        changed += 1
                        continue
        if changed == 0:
            for is_p, cw_list in ((True, P_IDX), (False, Q_IDX)):
                for cw in cw_list:
                    st1, corr1 = solve_1err(voted, cw)
                    if st1 == 1:
                        pos, val = corr1[0]
                        if pos not in FIXED_POS:
                            old = voted[pos]
                            voted[pos] = val
                            orth = (
                                Q_IDX[BYTE_TO_Q[pos]]
                                if is_p and pos in BYTE_TO_Q
                                else (P_IDX[BYTE_TO_P[pos]] if not is_p and pos in BYTE_TO_P else None)
                            )
                            if orth is None or solve_1err(voted, orth)[0] in (0, 1):
                                uncertain.discard(pos)
                                changed += 1
                            else:
                                voted[pos] = old
        if changed == 0:
            for cw_list in (P_IDX, Q_IDX):
                for cw in cw_list:
                    if syndromes(voted, cw) == (0, 0):
                        continue
                    eras = [i for i, pos in enumerate(cw) if pos in uncertain]
                    if len(eras) == 2:
                        ok, corrs = solve_2eras(voted, cw, eras)
                        if ok:
                            for pos, val in corrs:
                                voted[pos] = val
                            for i in eras:
                                uncertain.discard(cw[i])
                            changed += 1
        if changed == 0:
            break
    if check_and_fix_q_parity(voted):
        return True, 86, 52, voted
    pz, qz = count_pq(voted)
    return False, pz, qz, voted


def _cleaned_consensus_decode(
    lba: int, raw_passes: list[tuple[bytes, bytes]]
) -> tuple[bool, int, int, bytearray, str]:
    """Strategy C: Combine C2-cleaned passes across top-K subsets."""
    cleaned: list[tuple[int, bytearray, set[int]]] = []
    candidate_vals: list[set[int]] = [set() for _ in range(2352)]
    for raw_sec, c2_bytes in raw_passes:
        ok, c_sec, c2_eras = clean_single_pass(lba, raw_sec, c2_bytes)
        if ok:
            return True, 86, 52, c_sec, "C2_SINGLE_PASS"
        pz, qz = count_pq(c_sec)
        cleaned.append((pz + qz, c_sec, c2_eras))
        for j in range(2352):
            candidate_vals[j].add(c_sec[j])

    cleaned.sort(key=lambda x: x[0], reverse=True)
    best = (False, 0, 0, cleaned[0][1], "NONE")
    n = len(cleaned)
    ks = sorted({n, max(2, n - 1), min(6, n), min(4, n), min(3, n), min(2, n)}, reverse=True)
    for k in ks:
        if k < 2:
            continue
        sub = cleaned[:k]
        byte_votes = [Counter() for _ in range(2352)]
        for _, sec, c2_eras in sub:
            p_valid = [syndromes_p(sec, cw) == (0, 0) for cw in P_IDX]
            q_valid = [syndromes_q(sec, cw) == (0, 0) for cw in Q_IDX]
            for j in range(16, 2352):
                if j in FIXED_POS:
                    continue
                p_ok = j in BYTE_TO_P and p_valid[BYTE_TO_P[j]]
                q_ok = j in BYTE_TO_Q and q_valid[BYTE_TO_Q[j]]
                c2_ok = j not in c2_eras
                w = 1 + (3 if c2_ok else 0) + (8 if p_ok else 0) + (10 if q_ok else 0) + (25 if (p_ok and q_ok) else 0)
                byte_votes[j][sec[j]] += w
        voted = bytearray(2352)
        uncertain = set()
        for j in range(16, 2352):
            if j in FIXED_POS:
                continue
            mc = byte_votes[j].most_common(2)
            voted[j] = mc[0][0]
            if len(mc) > 1 and mc[1][1] * 4 >= mc[0][1]:
                uncertain.add(j)
        ok, pz, qz, v = _polish_consensus(lba, voted, uncertain, candidate_vals)
        if ok:
            return True, 86, 52, v, f"CLEANED_CONSENSUS_K{k}"
        if (pz, qz) > (best[1], best[2]):
            best = (False, pz, qz, v, f"PARTIAL_C_K{k}")
    return best


def _anchor_c2_graft_decode(
    lba: int, raw_passes: list[tuple[bytes, bytes]]
) -> tuple[bool, int, int, bytearray]:
    """Strategy D: Anchor on cleanest pass, graft C2-clean bytes, un-miscorrect 1-byte errors."""
    hdr4 = lba_header(lba)
    passes: list[bytearray] = []
    c2_sets: list[set[int]] = []
    candidate_vals: list[set[int]] = [set() for _ in range(2352)]
    for raw_sec, c2_bytes in raw_passes:
        sec = bytearray(raw_sec)
        sec[0:12] = SYNC_12
        sec[12:16] = hdr4
        sec[2068:2076] = b"\x00" * 8
        c2_eras = {
            j
            for j in range(16, 2352)
            if ((c2_bytes[j // 8] >> (7 - (j % 8))) & 1) and j not in FIXED_POS
        }
        passes.append(sec)
        c2_sets.append(c2_eras)
        for j in range(2352):
            candidate_vals[j].add(sec[j])

    order = sorted(
        range(len(passes)),
        key=lambda i: (len(c2_sets[i]), -sum(1 for cw in P_IDX if syndromes_p(passes[i], cw) == (0, 0))),
    )
    best = (False, 0, 0, passes[order[0]])
    for anchor_idx in order[: min(3, len(order))]:
        sec = bytearray(passes[anchor_idx])
        anchor_c2 = set(c2_sets[anchor_idx])
        still_uncertain: set[int] = set()
        for pos in anchor_c2:
            c = Counter()
            for idx, other_sec in enumerate(passes):
                if idx == anchor_idx:
                    continue
                if pos not in c2_sets[idx]:
                    w = 10
                    if pos in BYTE_TO_P and syndromes_p(other_sec, P_IDX[BYTE_TO_P[pos]]) == (0, 0):
                        w += 25
                    if pos in BYTE_TO_Q and syndromes_q(other_sec, Q_IDX[BYTE_TO_Q[pos]]) == (0, 0):
                        w += 30
                    c[other_sec[pos]] += w
            if c:
                mc = c.most_common(2)
                sec[pos] = mc[0][0]
                if len(mc) > 1:
                    still_uncertain.add(pos)
            else:
                still_uncertain.add(pos)

        for _ in range(25):
            if check_and_fix_q_parity(sec):
                return True, 86, 52, sec
            changed = 0
            for cw_list in (P_IDX, Q_IDX):
                for cw in cw_list:
                    if syndromes(sec, cw) == (0, 0):
                        continue
                    eras = [i for i, pos in enumerate(cw) if pos in still_uncertain]
                    if 1 <= len(eras) <= 2:
                        ok, corrs = solve_2eras(sec, cw, eras)
                        if ok:
                            for pos, val in corrs:
                                sec[pos] = val
                            for i in eras:
                                still_uncertain.discard(cw[i])
                            changed += 1
            if changed:
                continue
            for is_p, cw_list in ((True, P_IDX), (False, Q_IDX)):
                for cw in cw_list:
                    st1, c1 = solve_1err(sec, cw)
                    if st1 == 1:
                        pos, val = c1[0]
                        if pos in FIXED_POS:
                            continue
                        old = sec[pos]
                        sec[pos] = val
                        orth = (
                            Q_IDX[BYTE_TO_Q[pos]]
                            if is_p and pos in BYTE_TO_Q
                            else (P_IDX[BYTE_TO_P[pos]] if not is_p and pos in BYTE_TO_P else None)
                        )
                        if (
                            (pos in anchor_c2)
                            or (val in candidate_vals[pos])
                            or (orth is None)
                            or (solve_1err(sec, orth)[0] in (0, 1))
                        ):
                            still_uncertain.discard(pos)
                            changed += 1
                        else:
                            sec[pos] = old
            if changed == 0:
                break
        if check_and_fix_q_parity(sec):
            return True, 86, 52, sec
        pz, qz = count_pq(sec)
        if (pz, qz) > (best[1], best[2]):
            best = (False, pz, qz, sec)
    return best


def _partial_score(pz: int, qz: int) -> tuple[int, int]:
    """Rank unverified partial candidates, avoiding forged 86/86 loops that failed EDC."""
    return (0 if pz == 86 else pz, qz)


def portfolio_decode(
    lba: int, raw_passes: list[tuple[bytes, bytes]]
) -> tuple[bool, int, int, bytearray, str]:
    """Run Strategies A, B, C, and D to recover and mathematically verify a sector."""
    ok, sec_a, _ = clean_single_pass(lba, raw_passes[-1][0], raw_passes[-1][1])
    if ok:
        return True, 86, 52, sec_a, "A_SINGLE_C2"
    pz_a, qz_a = count_pq(sec_a)
    best = (False, pz_a, qz_a, sec_a, "PARTIAL_A")

    scored = sorted(
        raw_passes,
        key=lambda x: sum(1 for cw in P_IDX if syndromes_p(x[0], cw) == (0, 0)),
        reverse=True,
    )
    n = len(scored)
    ks = sorted(
        {
            k
            for k in (
                n,
                n - 1,
                round(0.8 * n),
                round(0.6 * n),
                n // 2,
                6,
                4,
                3,
                2,
            )
            if 1 <= k <= n
        },
        reverse=True,
    )
    for c2_strict in (False, True):
        for k in ks:
            sub = scored[:k]
            for inc_q in (False, True):
                ok, pz, qz, v = _subgrid_peel_decode(
                    lba, sub, include_q_parity_in_uncov=inc_q, c2_strict=c2_strict
                )
                if ok:
                    return True, 86, 52, v, f"B_SUBGRID_K{k}_Q{int(inc_q)}_S{int(c2_strict)}"
                if _partial_score(pz, qz) > _partial_score(best[1], best[2]):
                    best = (False, pz, qz, v, f"PARTIAL_B_K{k}_Q{int(inc_q)}_S{int(c2_strict)}")

    if n >= 2:
        ok, pz, qz, v, strat = _cleaned_consensus_decode(lba, raw_passes)
        if ok:
            return True, 86, 52, v, strat
        if _partial_score(pz, qz) > _partial_score(best[1], best[2]):
            best = (False, pz, qz, v, strat)

        ok, pz, qz, v = _anchor_c2_graft_decode(lba, raw_passes)
        if ok:
            return True, 86, 52, v, "D_ANCHOR_GRAFT"
        if _partial_score(pz, qz) > _partial_score(best[1], best[2]):
            best = (False, pz, qz, v, "PARTIAL_D")

    if n >= 8:
        tested = set(ks)
        for k in range(n - 2, 1, -1):
            if k in tested:
                continue
            sub = scored[:k]
            for c2_strict in (True, False):
                for inc_q in (False, True):
                    ok, pz, qz, v = _subgrid_peel_decode(
                        lba, sub, include_q_parity_in_uncov=inc_q, c2_strict=c2_strict
                    )
                    if ok:
                        return True, 86, 52, v, f"B_SUBGRID_K{k}_Q{int(inc_q)}_S{int(c2_strict)}"
                    if _partial_score(pz, qz) > _partial_score(best[1], best[2]):
                        best = (False, pz, qz, v, f"PARTIAL_B_K{k}_Q{int(inc_q)}_S{int(c2_strict)}")

    return best


# ---------------------------------------------------------------------------
# SCSI SG_IO Interface
# ---------------------------------------------------------------------------


class SgIoHdr(ctypes.Structure):
    _fields_ = [
        ("interface_id", ctypes.c_int),
        ("dxfer_direction", ctypes.c_int),
        ("cmd_len", ctypes.c_ubyte),
        ("mx_sb_len", ctypes.c_ubyte),
        ("iovec_count", ctypes.c_ushort),
        ("dxfer_len", ctypes.c_uint),
        ("dxferp", ctypes.c_void_p),
        ("cmdp", ctypes.c_void_p),
        ("sbp", ctypes.c_void_p),
        ("timeout", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("pack_id", ctypes.c_int),
        ("usr_ptr", ctypes.c_void_p),
        ("status", ctypes.c_ubyte),
        ("masked_status", ctypes.c_ubyte),
        ("msg_status", ctypes.c_ubyte),
        ("sb_len_wr", ctypes.c_ubyte),
        ("host_status", ctypes.c_ushort),
        ("driver_status", ctypes.c_ushort),
        ("resid", ctypes.c_int),
        ("duration", ctypes.c_uint),
        ("info", ctypes.c_uint),
    ]


def scsi_cmd(
    fd: int,
    cdb_bytes: list[int],
    buf_bytes: list[int] | None = None,
    buf_len: int = 256,
    direction: int = -3,
    timeout: int = 20000,
) -> tuple[int, tuple[int, int, int], int, bytes]:
    cdb = (ctypes.c_ubyte * len(cdb_bytes))(*cdb_bytes)
    if buf_bytes is not None:
        buf_len = len(buf_bytes)
        buf = (ctypes.c_ubyte * buf_len)(*buf_bytes)
    else:
        buf = (ctypes.c_ubyte * buf_len)()
    sense = (ctypes.c_ubyte * 64)()
    hdr = SgIoHdr(
        interface_id=ord("S"),
        dxfer_direction=direction,
        cmd_len=len(cdb_bytes),
        mx_sb_len=64,
        dxfer_len=buf_len,
        dxferp=ctypes.addressof(buf),
        cmdp=ctypes.addressof(cdb),
        sbp=ctypes.addressof(sense),
        timeout=timeout,
    )
    fcntl.ioctl(fd, SG_IO, hdr)
    sb = bytes(sense[: hdr.sb_len_wr])
    sk = sb[2] & 0x0F if len(sb) > 2 else 0
    asc = sb[12] if len(sb) > 12 else 0
    ascq = sb[13] if len(sb) > 13 else 0
    return hdr.status, (sk, asc, ascq), hdr.resid, bytes(buf)


def set_mode_01(fd: int, err_bits: int, retries: int):
    payload = [0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x0A, err_bits, retries, 0, 0, 0, 0, 0, 0, 0, 0]
    st, sense, _, _ = scsi_cmd(
        fd, [0x55, 0x10, 0, 0, 0, 0, 0, 0, len(payload), 0], buf_bytes=payload, direction=-2
    )
    if st != 0:
        raise RuntimeError(f"MODE SELECT Page 0x01 failed: status={st} sense={sense}")


def set_cd_speed(fd: int, kbps: int) -> int:
    """Issue SCSI SET CD SPEED (0xBB) for read speed diversity (0xFFFF = max)."""
    st, _, _, _ = scsi_cmd(
        fd,
        [0xBB, 0, (kbps >> 8) & 0xFF, kbps & 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0],
        buf_len=0,
        direction=0,
    )
    return st


def read_cd_raw(fd: int, start_lba: int, count: int, with_c2: bool = True) -> tuple[int, tuple[int, int, int], int, bytes]:
    flags = 0xFA if with_c2 else 0xF8
    sec_size = 2646 if with_c2 else 2352
    cdb = [
        0xBE,
        0,
        (start_lba >> 24) & 0xFF,
        (start_lba >> 16) & 0xFF,
        (start_lba >> 8) & 0xFF,
        start_lba & 0xFF,
        (count >> 16) & 0xFF,
        (count >> 8) & 0xFF,
        count & 0xFF,
        flags,
        0,
        0,
    ]
    return scsi_cmd(fd, cdb, buf_len=sec_size * count, timeout=25000)


# ---------------------------------------------------------------------------
# Data Loading, SQLite Persistence, ISO Maintenance
# ---------------------------------------------------------------------------


def load_bad_lbas() -> list[int]:
    lbas = []
    with BAD_LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RANGE_RE.search(line.strip())
            if m:
                lbas.extend(range(int(m.group(1), 16), int(m.group(2), 16)))
    return sorted(set(lbas))


def load_category_lbas(category: str) -> set[int]:
    if category == "all":
        return set()
    if not AFFECTED_CSV.exists():
        raise FileNotFoundError(
            f"{AFFECTED_CSV} not found. Run analyze_disc.py first."
        )
    cat_map = {
        "photo_s": lambda d: d == "/PHOTO/S",
        "photo_l": lambda d: d == "/PHOTO/L_PHOTO",
        "photo": lambda d: d.startswith("/PHOTO"),
        "pic": lambda d: d == "/PIC",
        "xtras": lambda d: d == "/XTRAS",
        "root": lambda d: d == "/",
    }
    pred = cat_map[category]
    target = set()
    with AFFECTED_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if pred(row["directory"]):
                for hx in row["bad_lbas_hex"].split():
                    target.add(int(hx, 16))
    return target


def init_db() -> sqlite3.Connection:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_passes (
            lba INTEGER NOT NULL,
            pass_id INTEGER NOT NULL,
            raw_2352 BLOB NOT NULL,
            c2_294 BLOB NOT NULL,
            p_ok INTEGER NOT NULL,
            q_ok INTEGER NOT NULL,
            PRIMARY KEY (lba, pass_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS solved_sectors (
            lba INTEGER PRIMARY KEY,
            user_2048 BLOB NOT NULL,
            verified INTEGER NOT NULL,
            best_p_ok INTEGER NOT NULL,
            best_q_ok INTEGER NOT NULL,
            edc_ok INTEGER NOT NULL,
            solved_pass INTEGER NOT NULL,
            strategy TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def ensure_recovered_iso():
    """Create miffy_recovered.iso from miffy1.iso and append trailing 150 volume sectors."""
    if not REC_ISO.exists():
        print(f"Copying {SRC_ISO.name} -> {REC_ISO.name} ...")
        shutil.copy2(SRC_ISO, REC_ISO)

    cur_size = REC_ISO.stat().st_size
    if cur_size == TOTAL_VOLUME_BYTES:
        return
    if cur_size > TOTAL_VOLUME_BYTES:
        raise RuntimeError(
            f"{REC_ISO} size {cur_size} exceeds expected {TOTAL_VOLUME_BYTES}"
        )

    start_tail = cur_size // 2048
    remaining = TOTAL_VOLUME_SECTORS - start_tail
    print(f"Appending trailing {remaining} sectors (LBA {start_tail}..{TOTAL_VOLUME_SECTORS - 1}) from {SG_DEV}...")
    fd = os.open(SG_DEV, os.O_RDWR | os.O_NONBLOCK)
    try:
        set_mode_01(fd, 0x00, 0x80)
        chunks = []
        pos = start_tail
        while pos < TOTAL_VOLUME_SECTORS:
            cnt = min(40, TOTAL_VOLUME_SECTORS - pos)
            st, sense, resid, raw = read_cd_raw(fd, pos, cnt, with_c2=False)
            if st != 0 or resid != 0:
                raise RuntimeError(f"Failed reading tail LBA {pos}..{pos+cnt}: st={st} sense={sense} resid={resid}")
            for i in range(cnt):
                chunks.append(raw[i * 2352 + 16 : i * 2352 + 2064])
            pos += cnt
    finally:
        os.close(fd)

    with REC_ISO.open("r+b") as f:
        f.seek(cur_size)
        f.write(b"".join(chunks))
    assert REC_ISO.stat().st_size == TOTAL_VOLUME_BYTES


def write_iso_sector(iso_fh, lba: int, user_2048: bytes | bytearray):
    iso_fh.seek(lba * 2048)
    iso_fh.write(user_2048)


def save_recovery_state(conn: sqlite3.Connection, all_bad_lbas: list[int]):
    rows = conn.execute(
        "SELECT lba, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy FROM solved_sectors"
    ).fetchall()
    by_lba = {r[0]: r for r in rows}
    verified_cnt = sum(1 for r in rows if r[1] == 1)
    partial_cnt = sum(1 for r in rows if r[1] == 0)
    untouched_cnt = len(all_bad_lbas) - len(rows)

    pass_rows = conn.execute(
        "SELECT lba, COUNT(*) FROM raw_passes GROUP BY lba"
    ).fetchall()
    max_pass_row = conn.execute("SELECT MAX(pass_id) FROM raw_passes").fetchone()

    strat_counts = Counter(r[6] for r in rows if r[1] == 1)
    state = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_bad_sectors": len(all_bad_lbas),
        "verified_sectors": verified_cnt,
        "partial_sectors": partial_cnt,
        "untouched_sectors": untouched_cnt,
        "verified_pct": round(100.0 * verified_cnt / max(1, len(all_bad_lbas)), 2),
        "total_raw_pass_reads": sum(r[1] for r in pass_rows),
        "max_pass_id": max_pass_row[0] if max_pass_row and max_pass_row[0] is not None else -1,
        "strategy_breakdown": dict(strat_counts),
        "unverified_sample": [
            {
                "lba": lba,
                "hex": f"0x{lba:05X}",
                "p_ok": by_lba[lba][2] if lba in by_lba else 0,
                "q_ok": by_lba[lba][3] if lba in by_lba else 0,
                "passes": dict(pass_rows).get(lba, 0),
            }
            for lba in all_bad_lbas
            if lba not in by_lba or by_lba[lba][1] == 0
        ][:30],
    }
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state


def print_status(conn: sqlite3.Connection, all_bad_lbas: list[int]):
    state = save_recovery_state(conn, all_bad_lbas)
    print(
        f"Recovery Status: {state['verified_sectors']}/{state['total_bad_sectors']} verified "
        f"({state['verified_pct']}%), {state['partial_sectors']} partial, "
        f"{state['untouched_sectors']} untouched | raw reads={state['total_raw_pass_reads']}"
    )
    if state["strategy_breakdown"]:
        print("  Strategies:", state["strategy_breakdown"])

    if AFFECTED_CSV.exists():
        rows = conn.execute(
            "SELECT lba, verified, best_p_ok, best_q_ok FROM solved_sectors"
        ).fetchall()
        status_map = {r[0]: (r[1], r[2], r[3]) for r in rows}
        by_dir = {}
        file_rows = []
        with AFFECTED_CSV.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbas = [int(hx, 16) for hx in row["bad_lbas_hex"].split()]
                d = row["directory"]
                st = by_dir.setdefault(d, {"total": 0, "verified": 0, "partial": 0, "untouched": 0, "clean_files": 0, "total_files": 0})
                st["total_files"] += 1
                v_cnt = sum(1 for x in lbas if status_map.get(x, (0,))[0] == 1)
                p_cnt = sum(1 for x in lbas if x in status_map and status_map[x][0] == 0)
                u_cnt = len(lbas) - v_cnt - p_cnt
                st["total"] += len(lbas)
                st["verified"] += v_cnt
                st["partial"] += p_cnt
                st["untouched"] += u_cnt
                if v_cnt == len(lbas):
                    st["clean_files"] += 1
                file_rows.append((row["path"], len(lbas), v_cnt, p_cnt, u_cnt))

        print("\nBy Directory:")
        for d, st in sorted(by_dir.items()):
            print(
                f"  {d:16s}: {st['verified']:4d}/{st['total']:4d} verified "
                f"({100*st['verified']/max(1,st['total']):5.1f}%), "
                f"files fully clean: {st['clean_files']}/{st['total_files']}"
            )
        incomplete = [x for x in file_rows if x[2] < x[1] and (x[2] > 0 or x[3] > 0)]
        if incomplete:
            print(f"\nPartially touched files with remaining bad sectors ({len(incomplete)}):")
            for path, tot, vc, pc, uc in incomplete[:15]:
                print(f"  {path:35s} {vc}/{tot} verified ({pc} partial, {uc} untouched)")


# ---------------------------------------------------------------------------
# Batched Multi-Pass Recovery Runner
# ---------------------------------------------------------------------------


def build_batches(target_lbas: list[int], lead: int) -> list[tuple[int, int, list[int]]]:
    """Group target LBAs into multi-sector READ CD batches (<= MAX_BATCH_SECTORS)."""
    batches = []
    i = 0
    n = len(target_lbas)
    while i < n:
        s_lba = max(0, target_lbas[i] - lead)
        if s_lba + MAX_BATCH_SECTORS > TOTAL_VOLUME_SECTORS:
            s_lba = max(0, TOTAL_VOLUME_SECTORS - MAX_BATCH_SECTORS)
        cnt = min(MAX_BATCH_SECTORS, TOTAL_VOLUME_SECTORS - s_lba)
        end_limit = s_lba + cnt - 2  # leave >= 1 trailing sector after last bad LBA
        covered = []
        while i < n and target_lbas[i] <= end_limit:
            if target_lbas[i] < s_lba:
                i += 1
                continue
            covered.append(target_lbas[i])
            i += 1
        if not covered and i < n:
            # Handle bad LBA right at the end of volume if needed
            covered.append(target_lbas[i])
            i += 1
        batches.append((s_lba, cnt, covered))
    return batches


def process_lba_decode(
    conn: sqlite3.Connection,
    iso_fh,
    lba: int,
    raw_passes: list[tuple[bytes, bytes]],
    cur_pass_id: int,
) -> bool:
    """Decode an LBA from its cached raw passes and update SQLite + ISO."""
    existing = conn.execute(
        "SELECT verified, best_p_ok, best_q_ok FROM solved_sectors WHERE lba = ?",
        (lba,),
    ).fetchone()
    if existing and existing[0] == 1:
        return True

    ok, pz, qz, sec, strat = portfolio_decode(lba, raw_passes)
    user_2048 = bytes(sec[16:2064])
    edc_ok = int(compute_edc(sec[:2064]) == int.from_bytes(sec[2064:2068], "little"))

    if ok:
        conn.execute(
            """
            INSERT OR REPLACE INTO solved_sectors
            (lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy)
            VALUES (?, ?, 1, 86, 52, 1, ?, ?)
            """,
            (lba, user_2048, cur_pass_id, strat),
        )
        write_iso_sector(iso_fh, lba, user_2048)
        return True

    prev_score = (
        (-1, -1)
        if (not existing or existing[1] == 86)
        else _partial_score(existing[1], existing[2])
    )
    if _partial_score(pz, qz) > prev_score:
        conn.execute(
            """
            INSERT OR REPLACE INTO solved_sectors
            (lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy)
            VALUES (?, ?, 0, ?, ?, ?, ?, ?)
            """,
            (lba, user_2048, pz, qz, edc_ok, cur_pass_id, strat),
        )
        write_iso_sector(iso_fh, lba, user_2048)
    return False


def run_redecode(conn: sqlite3.Connection, all_bad_lbas: list[int], target_lbas: list[int]):
    ensure_recovered_iso()
    unverified = [
        lba
        for lba in target_lbas
        if not conn.execute(
            "SELECT 1 FROM solved_sectors WHERE lba = ? AND verified = 1", (lba,)
        ).fetchone()
    ]
    print(f"Redecoding {len(unverified)} unverified target sectors from cached raw passes...", flush=True)
    solved_now = 0
    with REC_ISO.open("r+b") as iso_fh:
        for idx, lba in enumerate(unverified, 1):
            rows = conn.execute(
                "SELECT pass_id, raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
                (lba,),
            ).fetchall()
            if not rows:
                continue
            raw_passes = [(bytes(r[1]), bytes(r[2])) for r in rows]
            if process_lba_decode(conn, iso_fh, lba, raw_passes, rows[-1][0]):
                solved_now += 1
            if idx % 100 == 0 or idx == len(unverified):
                conn.commit()
                iso_fh.flush()
                print(f"  Redecoded {idx}/{len(unverified)} | newly_verified={solved_now}", flush=True)
        conn.commit()
    print(f"Redecode complete: newly verified {solved_now}/{len(unverified)} sectors.", flush=True)
    save_recovery_state(conn, all_bad_lbas)


SPEED_ROTATION = (0xFFFF, 1411, 706)


def run_recovery(
    conn: sqlite3.Connection,
    all_bad_lbas: list[int],
    target_lbas: list[int],
    max_passes: int,
    speed_mode: str = "auto",
):
    ensure_recovered_iso()

    for pass_round in range(max_passes):
        verified_set = {
            r[0]
            for r in conn.execute(
                "SELECT lba FROM solved_sectors WHERE verified = 1"
            ).fetchall()
        }
        remaining = [lba for lba in target_lbas if lba not in verified_set]
        if not remaining:
            print("All target bad sectors are verified!")
            break

        pass_counts = dict(
            conn.execute(
                "SELECT lba, COALESCE(MAX(pass_id), -1) FROM raw_passes GROUP BY lba"
            ).fetchall()
        )
        next_pass_base = min((pass_counts.get(lba, -1) + 1 for lba in remaining), default=pass_round)
        lead = 2 + ((next_pass_base * 3) % 15)
        if speed_mode == "auto":
            kbps = 0xFFFF if next_pass_base < 6 else SPEED_ROTATION[(next_pass_base - 6) % len(SPEED_ROTATION)]
        else:
            kbps = {"max": 0xFFFF, "8x": 1411, "4x": 706}[speed_mode]
        speed_label = "Max" if kbps == 0xFFFF else ("8x" if kbps == 1411 else "4x")

        batches = build_batches(remaining, lead)
        total_sectors_to_read = sum(b[1] for b in batches)
        print(
            f"\n--- Recovery Round {pass_round + 1}/{max_passes} (lead={lead}, speed={speed_label}): "
            f"{len(remaining)} unverified LBAs in {len(batches)} batches ({total_sectors_to_read} sectors) ---",
            flush=True,
        )

        fd = os.open(SG_DEV, os.O_RDWR | os.O_NONBLOCK)
        t_round = time.time()
        newly_verified = 0
        read_errors = 0
        try:
            set_mode_01(fd, 0x01, 0)
            st_spd = set_cd_speed(fd, kbps)
            if st_spd != 0:
                print(f"  WARN: SET CD SPEED ({speed_label}) returned status={st_spd}", flush=True)
            # Evict drive's 512 KB cache if batch count or span is short
            if len(batches) < 20 or (remaining[-1] - remaining[0] < 600):
                dummy_lba = max(0, remaining[0] - 1000)
                read_cd_raw(fd, dummy_lba, 1, with_c2=False)

            with REC_ISO.open("r+b") as iso_fh:
                for b_idx, (s_lba, cnt, covered) in enumerate(batches, 1):
                    st, sense, resid, raw = read_cd_raw(fd, s_lba, cnt, with_c2=True)
                    if st != 0 or resid != 0:
                        read_errors += 1
                        for lba in covered:
                            sub_s = max(0, lba - 2)
                            sub_cnt = min(4, TOTAL_VOLUME_SECTORS - sub_s)
                            st2, sense2, resid2, raw2 = read_cd_raw(fd, sub_s, sub_cnt, with_c2=True)
                            if st2 != 0 or resid2 != 0:
                                print(f"  WARN: SCSI read failed at LBA {lba}: st={st2} sense={sense2} resid={resid2}", flush=True)
                                continue
                            off = (lba - sub_s) * 2646
                            raw_2352 = raw2[off : off + 2352]
                            c2_294 = raw2[off + 2352 : off + 2646]
                            pid = pass_counts.get(lba, -1) + 1
                            pass_counts[lba] = pid
                            pz_raw, qz_raw = count_pq(raw_2352)
                            conn.execute(
                                "INSERT OR REPLACE INTO raw_passes (lba, pass_id, raw_2352, c2_294, p_ok, q_ok) VALUES (?, ?, ?, ?, ?, ?)",
                                (lba, pid, raw_2352, c2_294, pz_raw, qz_raw),
                            )
                            rows = conn.execute(
                                "SELECT raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
                                (lba,),
                            ).fetchall()
                            if process_lba_decode(conn, iso_fh, lba, [(bytes(r[0]), bytes(r[1])) for r in rows], pid):
                                newly_verified += 1
                        continue

                    for lba in covered:
                        off = (lba - s_lba) * 2646
                        raw_2352 = raw[off : off + 2352]
                        c2_294 = raw[off + 2352 : off + 2646]
                        pid = pass_counts.get(lba, -1) + 1
                        pass_counts[lba] = pid
                        pz_raw, qz_raw = count_pq(raw_2352)
                        conn.execute(
                            "INSERT OR REPLACE INTO raw_passes (lba, pass_id, raw_2352, c2_294, p_ok, q_ok) VALUES (?, ?, ?, ?, ?, ?)",
                            (lba, pid, raw_2352, c2_294, pz_raw, qz_raw),
                        )
                        rows = conn.execute(
                            "SELECT raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
                            (lba,),
                        ).fetchall()
                        if process_lba_decode(conn, iso_fh, lba, [(bytes(r[0]), bytes(r[1])) for r in rows], pid):
                            newly_verified += 1

                    if b_idx % 25 == 0 or b_idx == len(batches):
                        conn.commit()
                        iso_fh.flush()
                        elapsed = time.time() - t_round
                        print(
                            f"  Batch {b_idx}/{len(batches)} | elapsed={elapsed:.1f}s | "
                            f"newly_verified={newly_verified}/{len(remaining)} | errs={read_errors}",
                            flush=True,
                        )
        finally:
            try:
                set_cd_speed(fd, 0xFFFF)
            finally:
                try:
                    set_mode_01(fd, 0x00, 0x80)
                finally:
                    os.close(fd)

        conn.commit()
        state = save_recovery_state(conn, all_bad_lbas)
        dt_round = time.time() - t_round
        print(
            f"Round {pass_round + 1} finished in {dt_round:.1f}s: +{newly_verified} verified | "
            f"Overall: {state['verified_sectors']}/{state['total_bad_sectors']} ({state['verified_pct']}%)",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Recover bad CD sectors via raw SCSI C2 pointers + RSPC P/Q decoding."
    )
    parser.add_argument("--lba-start", type=int, default=0, help="Start LBA filter (inclusive)")
    parser.add_argument("--lba-end", type=int, default=999999, help="End LBA filter (inclusive)")
    parser.add_argument("--max-passes", type=int, default=6, help="Max read/decode passes")
    parser.add_argument(
        "--category",
        choices=["all", "photo_s", "photo_l", "photo", "pic", "xtras", "root"],
        default="all",
        help="Filter bad sectors by ISO directory category",
    )
    parser.add_argument(
        "--speed",
        choices=["auto", "max", "8x", "4x"],
        default="auto",
        help="CD read speed policy (auto rotates Max/8x/4x on passes >= 6)",
    )
    parser.add_argument("--status", action="store_true", help="Print status summary and exit")
    parser.add_argument("--redecode", action="store_true", help="Redecode cached raw passes without touching /dev/sg0")
    args = parser.parse_args()

    all_bad_lbas = load_bad_lbas()
    cat_set = load_category_lbas(args.category)
    target_lbas = [
        lba
        for lba in all_bad_lbas
        if args.lba_start <= lba <= args.lba_end and (not cat_set or lba in cat_set)
    ]

    conn = init_db()
    try:
        if args.status:
            print_status(conn, all_bad_lbas)
            return
        if args.redecode:
            run_redecode(conn, all_bad_lbas, target_lbas)
            print_status(conn, all_bad_lbas)
            return
        run_recovery(conn, all_bad_lbas, target_lbas, args.max_passes, speed_mode=args.speed)
        print_status(conn, all_bad_lbas)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
