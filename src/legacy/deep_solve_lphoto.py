#!/usr/bin/env python3
"""Deep algebraic RSPC+EDC solver for the 6 remaining /PHOTO/L_PHOTO bad sectors."""

import csv
import io
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFile

import recover_cd as r

TARGETS = [
    ("MYCD0619.JPG", 64208),
    ("MYCD0626.JPG", 64921),
    ("MYCD0642.JPG", 66870),
    ("MYCD0646.JPG", 67243),
    ("MYCD0664.JPG", 68685),
    ("MYCD0687.JPG", 70646),
]

S_DIR = r.MIFFY_DIR / "extracted" / "S"


def load_file_extents() -> dict[int, tuple[str, int, int]]:
    """Map target LBA -> (filename, start_lba, size_bytes)."""
    out = {}
    target_by_lba = {lba: fname for fname, lba in TARGETS}
    with r.AFFECTED_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for hx in row["bad_lbas_hex"].split():
                lba = int(hx, 16)
                if lba in target_by_lba:
                    out[lba] = (target_by_lba[lba], int(row["start_lba"]), int(row["size_bytes"]))
    return out


def c2_set(c2_bytes: bytes) -> set[int]:
    return {
        j
        for j in range(16, 2352)
        if ((c2_bytes[j // 8] >> (7 - (j % 8))) & 1) and j not in r.FIXED_POS
    }


def collect_plane_candidates(
    lba: int, raw_passes: list[tuple[bytes, bytes]], plane: int
) -> tuple[dict[int, Counter], dict[int, Counter]]:
    hdr4 = r.lba_header(lba)
    p_cands: dict[int, Counter] = defaultdict(Counter)
    q_cands: dict[int, Counter] = defaultdict(Counter)

    for raw_sec, c2_bytes in raw_passes:
        c2 = c2_set(c2_bytes)
        _, s_clean, _ = r.clean_single_pass(lba, raw_sec, c2_bytes)
        for w_mode, s_base in ((4, bytearray(raw_sec)), (1, s_clean)):
            s = bytearray(s_base)
            s[0:12] = r.SYNC_12
            s[12:16] = hdr4
            s[2068:2076] = b"\x00" * 8
            for c in range(43):
                cw = r.P_IDX[2 * c + plane]
                if r.syndromes_p(s, cw) == (0, 0):
                    b = bytes(s[pos] for pos in cw)
                    c2_hits = sum(1 for pos in cw if pos in c2)
                    p_cands[c][b] += (30 if c2_hits == 0 else (6 if c2_hits == 1 else 2)) * w_mode
                else:
                    eras = [i for i, pos in enumerate(cw) if pos in c2]
                    if 1 <= len(eras) <= 2:
                        ok, corrs = r.solve_2eras(s, cw, eras)
                        if ok:
                            st = bytearray(s)
                            for pos, val in corrs:
                                st[pos] = val
                            if r.syndromes_p(st, cw) == (0, 0):
                                p_cands[c][bytes(st[pos] for pos in cw)] += 2
            for q in range(26):
                cw = r.Q_IDX[2 * q + plane]
                if r.syndromes_q(s, cw) == (0, 0):
                    b = bytes(s[pos] for pos in cw)
                    c2_hits = sum(1 for pos in cw if pos in c2)
                    q_cands[q][b] += (30 if c2_hits == 0 else (6 if c2_hits == 1 else 2)) * w_mode
                else:
                    eras = [i for i, pos in enumerate(cw) if pos in c2]
                    if 1 <= len(eras) <= 2:
                        ok, corrs = r.solve_2eras(s, cw, eras)
                        if ok:
                            st = bytearray(s)
                            for pos, val in corrs:
                                st[pos] = val
                            if r.syndromes_q(st, cw) == (0, 0):
                                q_cands[q][bytes(st[pos] for pos in cw)] += 2
    return p_cands, q_cands


def solve_plane_bipartite_core(
    lba: int,
    raw_passes: list[tuple[bytes, bytes]],
    plane: int,
    prefer_p: bool = True,
) -> tuple[dict[int, bytes], dict[int, bytes], bytearray, set[int]]:
    hdr4 = r.lba_header(lba)
    p_cands, q_cands = collect_plane_candidates(lba, raw_passes, plane)
    passes = [bytearray(rp[0]) for rp in raw_passes]
    c2_sets = [c2_set(rp[1]) for rp in raw_passes]

    best_p = {c: p_cands[c].most_common(1)[0][0] for c in range(43) if p_cands[c]}
    best_q = {q: q_cands[q].most_common(1)[0][0] for q in range(26) if q_cands[q]}

    def prune_and_refill():
        for _ in range(60):
            for c in list(best_p.keys()):
                if sum(1 for q, b_q in best_q.items() if b_q[c] != best_p[c][(q + c) % 26]) > 0:
                    for b_p, _ in p_cands[c].most_common():
                        if sum(1 for q, b_q in best_q.items() if b_q[c] != b_p[(q + c) % 26]) == 0:
                            best_p[c] = b_p
                            break
            for q in list(best_q.keys()):
                if sum(1 for c, b_p in best_p.items() if best_q[q][c] != b_p[(q + c) % 26]) > 0:
                    for b_q, _ in q_cands[q].most_common():
                        if sum(1 for c, b_p in best_p.items() if b_q[c] != b_p[(q + c) % 26]) == 0:
                            best_q[q] = b_q
                            break
            p_conf = {c: sum(1 for q, b_q in best_q.items() if b_q[c] != b_p[(q + c) % 26]) for c, b_p in best_p.items()}
            q_conf = {q: sum(1 for c, b_p in best_p.items() if b_q[c] != b_p[(q + c) % 26]) for q, b_q in best_q.items()}
            max_pc = max(p_conf.values(), default=0)
            max_qc = max(q_conf.values(), default=0)
            if max(max_pc, max_qc) == 0:
                break
            worst_p = max(p_conf, key=lambda c: (p_conf[c], -p_cands[c][best_p[c]])) if p_conf else None
            worst_q = max(q_conf, key=lambda q: (q_conf[q], -q_cands[q][best_q[q]])) if q_conf else None
            if max_pc > max_qc or (max_pc == max_qc and not prefer_p):
                if worst_p is not None and (max_pc > max_qc or p_cands[worst_p][best_p[worst_p]] <= q_cands[worst_q][best_q[worst_q]] * (2 if not prefer_p else 1)):
                    del best_p[worst_p]
                elif worst_q is not None:
                    del best_q[worst_q]
            else:
                if worst_q is not None and (max_qc > max_pc or q_cands[worst_q][best_q[worst_q]] <= p_cands[worst_p][best_p[worst_p]]):
                    del best_q[worst_q]
                elif worst_p is not None:
                    del best_p[worst_p]

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

    for _ in range(8):
        prune_and_refill()
        known_pos = set(r.FIXED_POS)
        core_sec = bytearray(2352)
        core_sec[0:12] = r.SYNC_12
        core_sec[12:16] = hdr4
        core_sec[2068:2076] = b"\x00" * 8
        for c, b_p in best_p.items():
            for i, pos in enumerate(r.P_IDX[2 * c + plane]):
                core_sec[pos] = b_p[i]
                known_pos.add(pos)
        for q, b_q in best_q.items():
            for i, pos in enumerate(r.Q_IDX[2 * q + plane]):
                core_sec[pos] = b_q[i]
                known_pos.add(pos)

        added_any = False
        for s_raw, c2 in zip(passes, c2_sets):
            sh = bytearray(s_raw)
            for pos in known_pos:
                sh[pos] = core_sec[pos]
            for c in range(43):
                if c in best_p:
                    continue
                cw = r.P_IDX[2 * c + plane]
                un = [i for i, pos in enumerate(cw) if pos not in known_pos]
                un_c2 = [i for i in un if cw[i] in c2]
                trials = []
                if r.syndromes_p(sh, cw) == (0, 0):
                    trials.append((bytes(sh[pos] for pos in cw), 25))
                if 1 <= len(un) <= 2:
                    ok, corrs = r.solve_2eras(sh, cw, un)
                    if ok:
                        st = bytearray(sh)
                        for pos, val in corrs:
                            st[pos] = val
                        if r.syndromes_p(st, cw) == (0, 0):
                            trials.append((bytes(st[pos] for pos in cw), 60))
                if 1 <= len(un_c2) <= 2:
                    ok, corrs = r.solve_2eras(sh, cw, un_c2)
                    if ok:
                        st = bytearray(sh)
                        for pos, val in corrs:
                            st[pos] = val
                        if r.syndromes_p(st, cw) == (0, 0):
                            trials.append((bytes(st[pos] for pos in cw), 15))
                if len(un) == 3:
                    for keep_i in un:
                        if cw[keep_i] not in c2:
                            ok, corrs = r.solve_2eras(sh, cw, [i for i in un if i != keep_i])
                            if ok:
                                st = bytearray(sh)
                                for pos, val in corrs:
                                    st[pos] = val
                                if r.syndromes_p(st, cw) == (0, 0):
                                    trials.append((bytes(st[pos] for pos in cw), 5))
                for b_p, w in trials:
                    if all(b_q[c] == b_p[(q + c) % 26] for q, b_q in best_q.items()):
                        p_cands[c][b_p] += w
                        added_any = True

            for q in range(26):
                if q in best_q:
                    continue
                cw = r.Q_IDX[2 * q + plane]
                un = [i for i, pos in enumerate(cw) if pos not in known_pos]
                un_c2 = [i for i in un if cw[i] in c2]
                trials = []
                if r.syndromes_q(sh, cw) == (0, 0):
                    trials.append((bytes(sh[pos] for pos in cw), 25))
                if 1 <= len(un) <= 2:
                    ok, corrs = r.solve_2eras(sh, cw, un)
                    if ok:
                        st = bytearray(sh)
                        for pos, val in corrs:
                            st[pos] = val
                        if r.syndromes_q(st, cw) == (0, 0):
                            trials.append((bytes(st[pos] for pos in cw), 60))
                if 1 <= len(un_c2) <= 2:
                    ok, corrs = r.solve_2eras(sh, cw, un_c2)
                    if ok:
                        st = bytearray(sh)
                        for pos, val in corrs:
                            st[pos] = val
                        if r.syndromes_q(st, cw) == (0, 0):
                            trials.append((bytes(st[pos] for pos in cw), 15))
                if 3 <= len(un) <= 5:
                    for idx1 in range(len(un)):
                        for idx2 in range(idx1 + 1, len(un)):
                            eras2 = [un[idx1], un[idx2]]
                            others = [i for i in un if i not in eras2]
                            if all(cw[i] not in c2 for i in others):
                                ok, corrs = r.solve_2eras(sh, cw, eras2)
                                if ok:
                                    st = bytearray(sh)
                                    for pos, val in corrs:
                                        st[pos] = val
                                    if r.syndromes_q(st, cw) == (0, 0):
                                        trials.append((bytes(st[pos] for pos in cw), 4))
                for b_q, w in trials:
                    if all(b_q[c] == b_p[(q + c) % 26] for c, b_p in best_p.items()):
                        q_cands[q][b_q] += w
                        added_any = True

        # Algebraic peeling on this plane
        for _ in range(25):
            peeled = False
            for c in range(43):
                if c in best_p:
                    continue
                cw = r.P_IDX[2 * c + plane]
                un = [i for i, pos in enumerate(cw) if pos not in known_pos]
                if 1 <= len(un) <= 2:
                    ok, corrs = r.solve_2eras(core_sec, cw, un)
                    if ok:
                        for pos, val in corrs:
                            core_sec[pos] = val
                        if r.syndromes_p(core_sec, cw) == (0, 0):
                            for pos in cw:
                                known_pos.add(pos)
                            best_p[c] = bytes(core_sec[pos] for pos in cw)
                            peeled = True
            for q in range(26):
                if q in best_q:
                    continue
                cw = r.Q_IDX[2 * q + plane]
                un = [i for i, pos in enumerate(cw) if pos not in known_pos]
                if 1 <= len(un) <= 2:
                    ok, corrs = r.solve_2eras(core_sec, cw, un)
                    if ok:
                        for pos, val in corrs:
                            core_sec[pos] = val
                        if r.syndromes_q(core_sec, cw) == (0, 0):
                            for pos in cw:
                                known_pos.add(pos)
                            best_q[q] = bytes(core_sec[pos] for pos in cw)
                            peeled = True
            if not peeled:
                break

        if len(best_p) == 43 and len(best_q) == 26:
            break
        if not added_any:
            break

    # Final assembly of plane core
    known_pos = set(r.FIXED_POS)
    core_sec = bytearray(2352)
    core_sec[0:12] = r.SYNC_12
    core_sec[12:16] = hdr4
    core_sec[2068:2076] = b"\x00" * 8
    for c, b_p in best_p.items():
        for i, pos in enumerate(r.P_IDX[2 * c + plane]):
            core_sec[pos] = b_p[i]
            known_pos.add(pos)
    for q, b_q in best_q.items():
        for i, pos in enumerate(r.Q_IDX[2 * q + plane]):
            core_sec[pos] = b_q[i]
            known_pos.add(pos)
    return best_p, best_q, core_sec, known_pos


def assemble_core_sector(
    lba: int, raw_passes: list[tuple[bytes, bytes]], prefer_p: bool = True
) -> tuple[bytearray, set[int]]:
    s0_known: set[int]
    bp0, bq0, s0, k0 = solve_plane_bipartite_core(lba, raw_passes, 0, prefer_p=prefer_p)
    bp1, bq1, s1, k1 = solve_plane_bipartite_core(lba, raw_passes, 1, prefer_p=prefer_p)
    sec = bytearray(2352)
    sec[0:12] = r.SYNC_12
    sec[12:16] = r.lba_header(lba)
    sec[2068:2076] = b"\x00" * 8
    for pos in range(16, 2352):
        sec[pos] = s0[pos] if (pos % 2 == 0) else s1[pos]
    known = set(r.FIXED_POS) | {p for p in k0 if p % 2 == 0} | {p for p in k1 if p % 2 == 1}
    # Fill unknown bytes with best C2-weighted modal vote for partial fallback
    c2_sets = [c2_set(rp[1]) for rp in raw_passes]
    for pos in range(16, 2352):
        if pos not in known:
            cnt = Counter()
            for (raw_sec, _), c2 in zip(raw_passes, c2_sets):
                cnt[raw_sec[pos]] += 10 if pos not in c2 else 1
            sec[pos] = cnt.most_common(1)[0][0]
    return sec, known


def gf2_solve(equations: list[tuple[int, int]], n_vars: int) -> tuple[int, int, list[int]]:
    basis: dict[int, tuple[int, int]] = {}
    for m, b in equations:
        while m:
            lsb = m & -m
            bit = lsb.bit_length() - 1
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
    pivot_set = set(pivots)
    free_vars = [v for v in range(n_vars) if v not in pivot_set]
    for bit in pivots:
        m, b = basis[bit]
        if b ^ ((bin((m ^ (1 << bit)) & sol).count("1")) & 1):
            sol |= 1 << bit
    null_basis = []
    for fv in free_vars:
        vec = 1 << fv
        for bit in pivots:
            m, _ = basis[bit]
            if (bin((m ^ (1 << bit)) & vec).count("1")) & 1:
                vec |= 1 << bit
        null_basis.append(vec)
    return len(pivots), sol, null_basis


def edc_residual(sec_2068: bytes | bytearray) -> int:
    return r.compute_edc(sec_2068[:2064]) ^ int.from_bytes(sec_2068[2064:2068], "little")


def jpeg_post_sector_mae(
    fname: str, start_lba: int, size_bytes: int, bad_lba: int, sec_user_2048: bytes
) -> float:
    """Measure post-bad-sector MAE against the intact thumbnail S/MYCD06xx.JPG."""
    iso_bytes = bytearray()
    with r.REC_ISO.open("rb") as f:
        f.seek(start_lba * 2048)
        iso_bytes.extend(f.read(size_bytes))
    off = (bad_lba - start_lba) * 2048
    end_off = min(size_bytes, off + 2048)
    iso_bytes[off:end_off] = sec_user_2048[: end_off - off]

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        im_l = Image.open(io.BytesIO(bytes(iso_bytes)))
        im_l.load()
    except Exception:
        return 999.0

    im_s = Image.open(S_DIR / fname)
    im_s.load()
    im_s_up = im_s.resize(im_l.size, Image.Resampling.BILINEAR)
    bl = im_l.tobytes()
    bs = im_s_up.tobytes()
    start_row = max(0, (im_l.size[1] * off) // max(1, size_bytes) - 32)
    s0 = start_row * im_l.size[0] * 3
    if s0 >= len(bl):
        s0 = 0
    return sum(abs(a - b) for a, b in zip(bl[s0:], bs[s0:])) / max(1, len(bl) - s0)


def solve_with_gf2_edc(
    lba: int,
    raw_passes: list[tuple[bytes, bytes]],
    sec_core: bytearray,
    known: set[int],
    extent_info: tuple[str, int, int],
) -> tuple[bool, bytearray | None, str]:
    """Use GF(2) P-syndromes + 32-bit EDC (+ unanimous C2-clean pins if needed) to solve remaining unknowns."""
    un_all = sorted(pos for pos in range(16, 2248) if pos not in known)
    if not un_all or len(un_all) > 85:
        return False, None, ""

    c2_sets = [c2_set(rp[1]) for rp in raw_passes]
    # Rank unknown positions by C2-clean consensus so we can pin highest-confidence bytes if null_dim > 13
    pin_candidates = []
    for pos in un_all:
        clean_vals = [rp[0][pos] for rp, c2 in zip(raw_passes, c2_sets) if pos not in c2]
        if clean_vals:
            val, cnt = Counter(clean_vals).most_common(1)[0]
            ratio = cnt / len(clean_vals)
            pin_candidates.append((ratio, cnt, pos, val))
        else:
            raw_vals = [rp[0][pos] for rp in raw_passes]
            val, cnt = Counter(raw_vals).most_common(1)[0]
            pin_candidates.append((0.0, cnt, pos, val))
    pin_candidates.sort(reverse=True)

    fname, start_lba, size_bytes = extent_info

    # Try progressive pinning: 0 pins first, then pin top unanimous C2-clean bytes in un_all
    max_pins = max(0, len(un_all) - 2)
    tried_pin_counts = [0]
    for p_cnt in range(1, max_pins + 1):
        if p_cnt <= 12 or p_cnt % 2 == 0 or len(un_all) - p_cnt <= 24:
            tried_pin_counts.append(p_cnt)

    for n_pins in tried_pin_counts:
        sec_base = bytearray(sec_core)
        pinned_pos = set()
        for ratio, cnt, pos, val in pin_candidates[:n_pins]:
            if n_pins > 0 and ratio < 0.75 and len(un_all) - n_pins > 18:
                break
            sec_base[pos] = val
            pinned_pos.add(pos)

        un_pos = [p for p in un_all if p not in pinned_pos]
        if not un_pos or len(un_pos) > 45:
            continue
        p_touch = sorted({r.BYTE_TO_P[p] for p in un_pos})
        # Check non-touching P codewords
        sec0 = bytearray(sec_base)
        for p in un_pos:
            sec0[p] = 0
        if any(r.syndromes_p(sec0, r.P_IDX[idx]) != (0, 0) for idx in range(86) if idx not in p_touch):
            continue

        rhs_bits = []
        for idx in p_touch:
            s0, s1 = r.syndromes_p(sec0, r.P_IDX[idx])
            for b in range(8):
                rhs_bits.append((s0 >> b) & 1)
            for b in range(8):
                rhs_bits.append((s1 >> b) & 1)
        edc0 = edc_residual(sec0[:2068])
        for b in range(32):
            rhs_bits.append((edc0 >> b) & 1)

        n_eq = len(rhs_bits)
        n_vars = 8 * len(un_pos)
        col_vecs = [0] * n_eq
        zero_2352 = bytearray(2352)
        for k, pos in enumerate(un_pos):
            for bit_idx in range(8):
                var_id = 8 * k + bit_idx
                zero_2352[pos] = 1 << bit_idx
                eq_off = 0
                for idx in p_touch:
                    s0, s1 = r.syndromes_p(zero_2352, r.P_IDX[idx])
                    for b in range(8):
                        if (s0 >> b) & 1:
                            col_vecs[eq_off + b] |= 1 << var_id
                        if (s1 >> b) & 1:
                            col_vecs[eq_off + 8 + b] |= 1 << var_id
                    eq_off += 16
                if pos < 2068:
                    dedc = edc_residual(zero_2352[:2068])
                    for b in range(32):
                        if (dedc >> b) & 1:
                            col_vecs[eq_off + b] |= 1 << var_id
                zero_2352[pos] = 0

        rank, sol0, null_basis = gf2_solve(
            [(col_vecs[i], rhs_bits[i]) for i in range(n_eq)], n_vars
        )
        if rank == -1 or len(null_basis) > 13:
            continue

        print(
            f"    [GF2] LBA {lba}: n_pins={n_pins}, free_bytes={len(un_pos)}, null_dim={len(null_basis)} ({1<<len(null_basis)} cands)",
            flush=True,
        )
        # Precompute modal C2-clean Q-parity bytes (2248..2351) to cross-check solved Q-parity!
        qpar_clean = {}
        for pos in range(2248, 2352):
            vals = [rp[0][pos] for rp, c2 in zip(raw_passes, c2_sets) if pos not in c2]
            if not vals:
                vals = [rp[0][pos] for rp in raw_passes]
            qpar_clean[pos] = {v for v, c in Counter(vals).items() if c >= 2}

        best_cands = []
        for mask in range(1 << len(null_basis)):
            sol = sol0
            for idx, nb in enumerate(null_basis):
                if (mask >> idx) & 1:
                    sol ^= nb
            cand = bytearray(sec0)
            for k, pos in enumerate(un_pos):
                cand[pos] = (sol >> (8 * k)) & 0xFF
            if not r.check_and_fix_q_parity(cand):
                continue
            qpar_matches = sum(1 for pos in range(2248, 2352) if cand[pos] in qpar_clean[pos])
            raw_hits = sum(
                1 for pos in un_pos for rp in raw_passes if rp[0][pos] == cand[pos]
            )
            mae = jpeg_post_sector_mae(fname, start_lba, size_bytes, lba, bytes(cand[16:2064]))
            best_cands.append((qpar_matches, -mae, raw_hits, cand))

        if best_cands:
            best_cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            q_match, neg_mae, raw_hits, winner = best_cands[0]
            mae = -neg_mae
            print(
                f"    [GF2 HIT] LBA {lba}: 86/86 P, 52/52 Q, EDC=1 | qpar_match={q_match}/104 | JPEG post-MAE={mae:.2f}",
                flush=True,
            )
            if mae < 25.0 or q_match >= 98:
                return True, winner, f"GF2_P_EDC_PIN{n_pins}_ND{len(null_basis)}"
    return False, None, ""


def deep_solve_one(
    conn: sqlite3.Connection,
    iso_fh,
    lba: int,
    extent_info: tuple[str, int, int],
) -> bool:
    existing = conn.execute(
        "SELECT verified, best_p_ok, best_q_ok FROM solved_sectors WHERE lba = ?",
        (lba,),
    ).fetchone()
    if existing and existing[0] == 1:
        return True

    rows = conn.execute(
        "SELECT pass_id, raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
        (lba,),
    ).fetchall()
    raw_passes = [(bytes(row[1]), bytes(row[2])) for row in rows]
    max_pid = rows[-1][0]

    # 1. Try speed/direction subset splits + random subgrid splits before full GF(2)
    subsets = [
        ("ALL", raw_passes),
        ("LAST32", raw_passes[16:]),
        ("FWD_NEW", raw_passes[16::2]),
        ("REV_NEW", raw_passes[17::2]),
        ("OLD16", raw_passes[:16]),
    ]
    rng = random.Random(lba)
    for r_i in range(24):
        k = rng.choice([8, 12, 16, 20, 24, 32])
        if len(raw_passes) > k:
            subsets.append((f"RAND{r_i}_K{k}", rng.sample(raw_passes, k)))

    for label, sub in subsets:
        if len(sub) < 2:
            continue
        ok, pz, qz, sec, strat = r.portfolio_decode(lba, sub)
        if ok:
            print(f"  LBA {lba} VERIFIED by subset {label} ({strat})!", flush=True)
            user_2048 = bytes(sec[16:2064])
            conn.execute(
                "INSERT OR REPLACE INTO solved_sectors (lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy) VALUES (?, ?, 1, 86, 52, 1, ?, ?)",
                (lba, user_2048, max_pid, f"DEEP_{label}_{strat}"),
            )
            r.write_iso_sector(iso_fh, lba, user_2048)
            return True

    # 2. Bipartite Core + Core-Injected Hybrid Pass Harvesting + GF(2) P+EDC Solver
    best_partial_sec = None
    best_partial_score = (-1, -1)
    best_partial_strat = ""

    for prefer_p in (True, False):
        sec_core, known = assemble_core_sector(lba, raw_passes, prefer_p=prefer_p)
        sec_chk = bytearray(sec_core)
        if r.check_and_fix_q_parity(sec_chk):
            fname, start_lba, size_bytes = extent_info
            mae = jpeg_post_sector_mae(fname, start_lba, size_bytes, lba, bytes(sec_chk[16:2064]))
            print(f"  LBA {lba} VERIFIED by Bipartite Core (pref={prefer_p})! JPEG post-MAE={mae:.2f}", flush=True)
            user_2048 = bytes(sec_chk[16:2064])
            conn.execute(
                "INSERT OR REPLACE INTO solved_sectors (lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy) VALUES (?, ?, 1, 86, 52, 1, ?, ?)",
                (lba, user_2048, max_pid, f"BIPARTITE_CORE_P{int(prefer_p)}"),
            )
            r.write_iso_sector(iso_fh, lba, user_2048)
            return True

        ok_gf2, sec_gf2, strat_gf2 = solve_with_gf2_edc(lba, raw_passes, sec_core, known, extent_info)
        if ok_gf2 and sec_gf2 is not None:
            user_2048 = bytes(sec_gf2[16:2064])
            conn.execute(
                "INSERT OR REPLACE INTO solved_sectors (lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy) VALUES (?, ?, 1, 86, 52, 1, ?, ?)",
                (lba, user_2048, max_pid, strat_gf2),
            )
            r.write_iso_sector(iso_fh, lba, user_2048)
            return True

        pz, qz = r.count_pq(sec_core)
        if r._partial_score(pz, qz) > best_partial_score:
            best_partial_score = r._partial_score(pz, qz)
            best_partial_sec = sec_core
            best_partial_strat = f"PARTIAL_CORE_P{int(prefer_p)}"

    if best_partial_sec is not None:
        prev_score = (
            (-1, -1)
            if (not existing or existing[1] == 86)
            else r._partial_score(existing[1], existing[2])
        )
        if best_partial_score > prev_score:
            pz, qz = r.count_pq(best_partial_sec)
            user_2048 = bytes(best_partial_sec[16:2064])
            edc_ok = int(r.compute_edc(best_partial_sec[:2064]) == int.from_bytes(best_partial_sec[2064:2068], "little"))
            conn.execute(
                "INSERT OR REPLACE INTO solved_sectors (lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy) VALUES (?, ?, 0, ?, ?, ?, ?, ?)",
                (lba, user_2048, pz, qz, edc_ok, max_pid, best_partial_strat),
            )
            r.write_iso_sector(iso_fh, lba, user_2048)
    return False


def main() -> None:
    extents = load_file_extents()
    conn = r.init_db()
    all_bad_lbas = r.load_bad_lbas()
    try:
        verified_cnt = 0
        with r.REC_ISO.open("r+b") as iso_fh:
            for fname, lba in TARGETS:
                n_p = conn.execute("SELECT COUNT(*) FROM raw_passes WHERE lba = ?", (lba,)).fetchone()[0]
                print(f"\n=== Deep Solving {fname} LBA {lba} ({n_p} passes) ===", flush=True)
                ok = deep_solve_one(conn, iso_fh, lba, extents[lba])
                st = conn.execute(
                    "SELECT verified, best_p_ok, best_q_ok, edc_ok, strategy FROM solved_sectors WHERE lba = ?",
                    (lba,),
                ).fetchone()
                print(
                    f"  Result LBA {lba}: verified={st[0]}, P={st[1]}/86, Q={st[2]}/52, edc={st[3]}, strat={st[4]}",
                    flush=True,
                )
                if ok:
                    verified_cnt += 1
                conn.commit()
                iso_fh.flush()
        r.save_recovery_state(conn, all_bad_lbas)
        print(f"\nDeep solve summary: {verified_cnt}/{len(TARGETS)} target LBAs mathematically verified.", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
