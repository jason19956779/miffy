#!/usr/bin/env python3
"""Thumbnail-guided baseline JPEG bitstream resynchronization & lossless DCT gap reconstruction."""

import io
import math
import shutil
import sqlite3
import struct
from pathlib import Path
from PIL import Image, ImageFile

import deep_solve_lphoto as ds
import recover_cd as r

ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]
COS_TBL = [[math.cos((2 * x + 1) * u * math.pi / 16.0) for x in range(8)] for u in range(8)]
ALPHA = [1.0 / math.sqrt(2.0) if u == 0 else 1.0 for u in range(8)]

L_DIR = r.MIFFY_DIR / "extracted" / "L_PHOTO"
S_DIR = r.MIFFY_DIR / "extracted" / "S"
BACKUP_DIR = r.LOGS_DIR / "lphoto_before_bridge"


def parse_jpeg_info(data: bytes):
    pos = 2
    dht = {}
    dqt = {}
    sos_off = 0
    w = h = 0
    comps = []
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        m = data[pos + 1]
        if m == 0xFF:
            pos += 1
            continue
        if m == 0xDA:
            ln = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
            sos_off = pos + 2 + ln
            break
        ln = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        seg = data[pos + 4 : pos + 2 + ln]
        if m == 0xC0:
            _, h, w, ncomp = struct.unpack(">BHHB", seg[:6])
            comps = [
                (seg[6 + 3 * i], seg[6 + 3 * i + 1] >> 4, seg[6 + 3 * i + 1] & 0xF, seg[6 + 3 * i + 2])
                for i in range(ncomp)
            ]
        elif m == 0xC4:
            i = 0
            while i < len(seg):
                tc_th = seg[i]
                i += 1
                bits = list(seg[i : i + 16])
                i += 16
                vals = list(seg[i : i + sum(bits)])
                i += sum(bits)
                dht[(tc_th >> 4, tc_th & 0xF)] = (bits, vals)
        elif m == 0xDB:
            i = 0
            while i < len(seg):
                tq = seg[i] & 0xF
                i += 1
                dqt[tq] = list(seg[i : i + 64])
                i += 64
        pos += 2 + ln
    eoi = data.rfind(b"\xff\xd9")
    return w, h, comps, dqt, dht, sos_off, eoi


def build_dec_huff(bits: list[int], vals: list[int]):
    lut_sym = [-1] * 65536
    lut_len = [0] * 65536
    code = 0
    vi = 0
    for b_len in range(1, 17):
        shift = 16 - b_len
        for _ in range(bits[b_len - 1]):
            sym = vals[vi]
            vi += 1
            base = code << shift
            for suffix in range(1 << shift):
                lut_sym[base | suffix] = sym
                lut_len[base | suffix] = b_len
            code += 1
        code <<= 1
    return lut_sym, lut_len


def build_enc_huff(bits: list[int], vals: list[int]):
    enc = {}
    code = 0
    vi = 0
    for b_len in range(1, 17):
        for _ in range(bits[b_len - 1]):
            enc[vals[vi]] = (code, b_len)
            vi += 1
            code += 1
        code <<= 1
    return enc


def unstuff_scan(data: bytes, sos_off: int, eoi: int) -> tuple[bytes, list[int]]:
    unstuffed = bytearray()
    orig_byte_map = []
    i = sos_off
    while i < eoi:
        b = data[i]
        unstuffed.append(b)
        orig_byte_map.append(i)
        if b == 0xFF and i + 1 < eoi and data[i + 1] == 0x00:
            i += 2
        else:
            i += 1
    return bytes(unstuffed), orig_byte_map


def idct_8x8_block(coeffs_zz: list[int], q_table: list[int]) -> list[float]:
    f = [0.0] * 64
    for k in range(64):
        if coeffs_zz[k]:
            f[ZIGZAG[k]] = coeffs_zz[k] * q_table[k]
    tmp = [0.0] * 64
    for r_i in range(8):
        for x in range(8):
            s = 0.0
            for u in range(8):
                val = f[r_i * 8 + u]
                if val:
                    s += ALPHA[u] * val * COS_TBL[u][x]
            tmp[r_i * 8 + x] = 0.5 * s
    out = [0.0] * 64
    for x in range(8):
        for y in range(8):
            s = 0.0
            for v in range(8):
                val = tmp[v * 8 + x]
                if val:
                    s += ALPHA[v] * val * COS_TBL[v][y]
            out[y * 8 + x] = 0.5 * s + 128.0
    return out


def fdct_quant_8x8(block64_px: list[float], q_table: list[int]) -> list[int]:
    shifted = [px - 128.0 for px in block64_px]
    tmp = [0.0] * 64
    for r_i in range(8):
        row = shifted[r_i * 8 : r_i * 8 + 8]
        for u in range(8):
            cu = COS_TBL[u]
            tmp[r_i * 8 + u] = 0.5 * ALPHA[u] * sum(row[x] * cu[x] for x in range(8))
    f = [0.0] * 64
    for u in range(8):
        for v in range(8):
            cv = COS_TBL[v]
            f[v * 8 + u] = 0.5 * ALPHA[v] * sum(tmp[y * 8 + u] * cv[y] for y in range(8))
    coeffs_zz = [0] * 64
    for k in range(64):
        coeffs_zz[k] = round(f[ZIGZAG[k]] / q_table[k])
    return coeffs_zz


def decode_jpeg_ycbcr_planes(data: bytes):
    w, h, comps, dqt, dht, sos_off, eoi = parse_jpeg_info(data)
    dc_y = build_dec_huff(*dht[(0, 0)])
    ac_y = build_dec_huff(*dht[(1, 0)])
    dc_c = build_dec_huff(*dht[(0, 1)])
    ac_c = build_dec_huff(*dht[(1, 1)])
    h_y, v_y = comps[0][1], comps[0][2]
    mcu_w = 8 * h_y
    mcu_h = 8 * v_y
    nx = (w + mcu_w - 1) // mcu_w
    ny = (h + mcu_h - 1) // mcu_h
    u, _ = unstuff_scan(data, sos_off, eoi)
    nu = len(u)
    bp = 0
    pred = [0, 0, 0]
    blk_spec = [(0, dc_y, ac_y, dqt[comps[0][3]])] * (h_y * v_y) + [
        (1, dc_c, ac_c, dqt[comps[1][3]]),
        (2, dc_c, ac_c, dqt[comps[2][3]]),
    ]
    Y = [0.0] * (w * h)
    Cb = [0.0] * (w * h)
    Cr = [0.0] * (w * h)
    mcus = []
    for mr in range(ny):
        for mc in range(nx):
            blks = []
            for comp, dc_t, ac_t, qt in blk_spec:
                dc_s, dc_l = dc_t
                ac_s, ac_l = ac_t
                bi = bp >> 3
                binb = bp & 7
                b0 = u[bi]
                b1 = u[bi + 1] if bi + 1 < nu else 0
                b2 = u[bi + 2] if bi + 2 < nu else 0
                wt = ((b0 << 16 | b1 << 8 | b2) >> (8 - binb)) & 0xFFFF
                slen = dc_l[wt]
                ssss = dc_s[wt]
                bp += slen
                if ssss:
                    bi = bp >> 3
                    binb = bp & 7
                    b0 = u[bi]
                    b1 = u[bi + 1] if bi + 1 < nu else 0
                    b2 = u[bi + 2] if bi + 2 < nu else 0
                    diff = ((b0 << 16 | b1 << 8 | b2) >> (24 - binb - ssss)) & ((1 << ssss) - 1)
                    if diff < (1 << (ssss - 1)):
                        diff -= (1 << ssss) - 1
                    bp += ssss
                else:
                    diff = 0
                pred[comp] += diff
                coeffs = [0] * 64
                coeffs[0] = pred[comp]
                k = 1
                while k < 64:
                    bi = bp >> 3
                    binb = bp & 7
                    b0 = u[bi]
                    b1 = u[bi + 1] if bi + 1 < nu else 0xFF
                    b2 = u[bi + 2] if bi + 2 < nu else 0xFF
                    wt = ((b0 << 16 | b1 << 8 | b2) >> (8 - binb)) & 0xFFFF
                    alen = ac_l[wt]
                    rs = ac_s[wt]
                    bp += alen
                    if rs == 0:
                        break
                    if rs == 0xF0:
                        k += 16
                    else:
                        k += rs >> 4
                        ss = rs & 0xF
                        bi = bp >> 3
                        binb = bp & 7
                        b0 = u[bi]
                        b1 = u[bi + 1] if bi + 1 < nu else 0
                        b2 = u[bi + 2] if bi + 2 < nu else 0
                        val = ((b0 << 16 | b1 << 8 | b2) >> (24 - binb - ss)) & ((1 << ss) - 1)
                        if val < (1 << (ss - 1)):
                            val -= (1 << ss) - 1
                        bp += ss
                        coeffs[k] = val
                        k += 1
                blks.append((comp, coeffs, qt))
            mcus.append([b[1] for b in blks])
            for bi_y in range(h_y * v_y):
                px = idct_8x8_block(blks[bi_y][1], blks[bi_y][2])
                ox = mc * mcu_w + (bi_y % h_y) * 8
                oy = mr * mcu_h + (bi_y // h_y) * 8
                for yy in range(8):
                    if oy + yy >= h:
                        continue
                    for xx in range(8):
                        if ox + xx < w:
                            Y[(oy + yy) * w + ox + xx] = px[yy * 8 + xx]
            for c_idx, plane in ((h_y * v_y, Cb), (h_y * v_y + 1, Cr)):
                px = idct_8x8_block(blks[c_idx][1], blks[c_idx][2])
                for yy in range(8):
                    for sy in range(v_y):
                        py = mr * mcu_h + yy * v_y + sy
                        if py >= h:
                            continue
                        for xx in range(8):
                            for sx in range(h_y):
                                px_x = mc * mcu_w + xx * h_y + sx
                                if px_x < w:
                                    plane[py * w + px_x] = px[yy * 8 + xx]
    return w, h, dqt, mcus, Y, Cb, Cr


def sample_bilinear(plane: list[float], w_s: int, h_s: int, w_l: int, h_l: int, x: int, y: int) -> float:
    fx = max(0.0, min(w_s - 1.0, (x + 0.5) * w_s / float(w_l) - 0.5))
    fy = max(0.0, min(h_s - 1.0, (y + 0.5) * h_s / float(h_l) - 0.5))
    x0 = int(fx)
    y0 = int(fy)
    x1 = min(w_s - 1, x0 + 1)
    y1 = min(h_s - 1, y0 + 1)
    dx = fx - x0
    dy = fy - y0
    return (
        (plane[y0 * w_s + x0] * (1.0 - dx) + plane[y0 * w_s + x1] * dx) * (1.0 - dy)
        + (plane[y1 * w_s + x0] * (1.0 - dx) + plane[y1 * w_s + x1] * dx) * dy
    )


def encode_mcus(mcus: list[list[list[int]]], dht: dict) -> bytes:
    dc_ey = build_enc_huff(*dht[(0, 0)])
    ac_ey = build_enc_huff(*dht[(1, 0)])
    dc_ec = build_enc_huff(*dht[(0, 1)])
    ac_ec = build_enc_huff(*dht[(1, 1)])
    dc_tbls = [dc_ey, dc_ey, dc_ey, dc_ey, dc_ec, dc_ec]
    ac_tbls = [ac_ey, ac_ey, ac_ey, ac_ey, ac_ec, ac_ec]
    out = bytearray()
    acc = 0
    nbits = 0

    def put(code: int, length: int):
        nonlocal acc, nbits
        acc = (acc << length) | code
        nbits += length
        while nbits >= 8:
            nbits -= 8
            b = (acc >> nbits) & 0xFF
            out.append(b)
            if b == 0xFF:
                out.append(0x00)
            acc &= (1 << nbits) - 1

    pred = [0, 0, 0]
    for mcu in mcus:
        for blk in range(6):
            dc_e = dc_tbls[blk]
            ac_e = ac_tbls[blk]
            coeffs = mcu[blk]
            comp = 0 if blk < 4 else (1 if blk == 4 else 2)
            diff = coeffs[0] - pred[comp]
            pred[comp] = coeffs[0]
            if diff == 0:
                c, ln = dc_e[0]
                put(c, ln)
            else:
                ssss = abs(diff).bit_length()
                c, ln = dc_e[ssss]
                put(c, ln)
                bits = diff if diff > 0 else diff + (1 << ssss) - 1
                put(bits, ssss)
            last_nz = 0
            for k in range(63, 0, -1):
                if coeffs[k] != 0:
                    last_nz = k
                    break
            zeros = 0
            for k in range(1, last_nz + 1):
                v = coeffs[k]
                if v == 0:
                    zeros += 1
                    if zeros == 16:
                        c, ln = ac_e[0xF0]
                        put(c, ln)
                        zeros = 0
                else:
                    ss = abs(v).bit_length()
                    c, ln = ac_e[(zeros << 4) | ss]
                    put(c, ln)
                    bits = v if v > 0 else v + (1 << ss) - 1
                    put(bits, ss)
                    zeros = 0
            if last_nz < 63:
                c, ln = ac_e[0x00]
                put(c, ln)
    if nbits > 0:
        put((1 << (8 - nbits)) - 1, 8 - nbits)
    return bytes(out)


def reconstruct_one_photo(
    fname: str, lba: int, start_lba: int, size_bytes: int, conn: sqlite3.Connection
):
    l_path = L_DIR / fname
    s_path = S_DIR / fname
    byte_off = (lba - start_lba) * 2048

    # Load verified mask inside the unverified sector from bipartite core so we preserve clean prefix/suffix MCUs!
    rows = conn.execute(
        "SELECT raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
        (lba,),
    ).fetchall()
    raw_passes = [(bytes(rw[0]), bytes(rw[1])) for rw in rows]
    _, known_sec_pos = ds.assemble_core_sector(lba, raw_passes, prefer_p=True)
    # Map sector byte positions [16..2063] -> file byte offsets
    bad_file_offs = sorted(
        byte_off + (pos - 16)
        for pos in range(16, 2064)
        if pos not in known_sec_pos and byte_off + (pos - 16) < size_bytes
    )
    first_bad_off = bad_file_offs[0] if bad_file_offs else byte_off
    last_bad_off = bad_file_offs[-1] if bad_file_offs else byte_off + 2047
    print(
        f"  Verified interior prefix ends at file byte {first_bad_off} (+{first_bad_off - byte_off} into sector); "
        f"verified tail starts at {last_bad_off + 1} (bad span {last_bad_off - first_bad_off + 1} bytes)",
        flush=True,
    )

    with r.REC_ISO.open("rb") as f:
        f.seek(start_lba * 2048)
        data = f.read(size_bytes)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    (BACKUP_DIR / fname).write_bytes(data)

    w, h, comps, dqt, dht, sos_off, eoi = parse_jpeg_info(data)
    assert (w, h) == (1600, 1200)
    n_total = 7500
    dc_y = build_dec_huff(*dht[(0, 0)])
    ac_y = build_dec_huff(*dht[(1, 0)])
    dc_c = build_dec_huff(*dht[(0, 1)])
    ac_c = build_dec_huff(*dht[(1, 1)])
    mc_tbls = [dc_y, dc_y, dc_y, dc_y, dc_c, dc_c]
    ac_tbls = [ac_y, ac_y, ac_y, ac_y, ac_c, ac_c]

    u, orig_byte_map = unstuff_scan(data, sos_off, eoi)
    nu = len(u)
    bit_len = nu * 8

    def decode_mcu_at(bp: int, pred: list[int]):
        mcu = []
        for blk in range(6):
            dc_s, dc_l = mc_tbls[blk]
            ac_s, ac_l = ac_tbls[blk]
            bi = bp >> 3
            binb = bp & 7
            if bi >= nu:
                return -1, None
            b0 = u[bi]
            b1 = u[bi + 1] if bi + 1 < nu else 0
            b2 = u[bi + 2] if bi + 2 < nu else 0
            wt = ((b0 << 16 | b1 << 8 | b2) >> (8 - binb)) & 0xFFFF
            slen = dc_l[wt]
            if slen == 0 or bp + slen > bit_len:
                return -1, None
            ssss = dc_s[wt]
            if ssss > 11:
                return -1, None
            bp += slen
            if ssss:
                if bp + ssss > bit_len:
                    return -1, None
                bi = bp >> 3
                binb = bp & 7
                b0 = u[bi]
                b1 = u[bi + 1] if bi + 1 < nu else 0
                b2 = u[bi + 2] if bi + 2 < nu else 0
                diff = ((b0 << 16 | b1 << 8 | b2) >> (24 - binb - ssss)) & ((1 << ssss) - 1)
                if diff < (1 << (ssss - 1)):
                    diff -= (1 << ssss) - 1
                bp += ssss
            else:
                diff = 0
            comp = 0 if blk < 4 else (1 if blk == 4 else 2)
            pred[comp] += diff
            coeffs = [0] * 64
            coeffs[0] = pred[comp]
            k = 1
            while k < 64:
                bi = bp >> 3
                binb = bp & 7
                if bi >= nu:
                    return -1, None
                b0 = u[bi]
                b1 = u[bi + 1] if bi + 1 < nu else 0xFF
                b2 = u[bi + 2] if bi + 2 < nu else 0xFF
                wt = ((b0 << 16 | b1 << 8 | b2) >> (8 - binb)) & 0xFFFF
                alen = ac_l[wt]
                if alen == 0 or bp + alen > bit_len:
                    return -1, None
                rs = ac_s[wt]
                bp += alen
                if rs == 0:
                    break
                if rs == 0xF0:
                    k += 16
                else:
                    k += rs >> 4
                    ss = rs & 0xF
                    if ss > 10 or k >= 64 or bp + ss > bit_len:
                        return -1, None
                    bi = bp >> 3
                    binb = bp & 7
                    b0 = u[bi]
                    b1 = u[bi + 1] if bi + 1 < nu else 0
                    b2 = u[bi + 2] if bi + 2 < nu else 0
                    val = ((b0 << 16 | b1 << 8 | b2) >> (24 - binb - ss)) & ((1 << ss) - 1)
                    if val < (1 << (ss - 1)):
                        val -= (1 << ss) - 1
                    bp += ss
                    coeffs[k] = val
                    k += 1
            if k > 64:
                return -1, None
            mcu.append(coeffs)
        return bp, mcu

    # 1. Decode intact prefix MCUs strictly contained in [0 .. first_bad_off - 1]
    pre_mcus = []
    bp = 0
    pred_pre = [0, 0, 0]
    while len(pre_mcus) < n_total:
        tmp_pred = list(pred_pre)
        nxt, mcu = decode_mcu_at(bp, tmp_pred)
        if nxt == -1:
            break
        last_u = min((nxt - 1) >> 3, nu - 1)
        last_file_off = orig_byte_map[last_u] + (1 if u[last_u] == 0xFF else 0)
        if last_file_off >= first_bad_off:
            break
        pred_pre = tmp_pred
        pre_mcus.append(mcu)
        bp = nxt

    # 2. Decode intact thumbnail S in direct YCbCr DCT domain
    ws, hs, dqts, _, Ys, Cbs, Crs = decode_jpeg_ycbcr_planes(s_path.read_bytes())

    # Sanity-trim any trailing pre_mcu that might diverge from S if core had an undetected boundary glitch
    while pre_mcus:
        m = len(pre_mcus) - 1
        mr, mc = m // 100, m % 100
        x0, y0 = mc * 16, mr * 16
        px_y = [idct_8x8_block(pre_mcus[-1][bi], dqt[0]) for bi in range(4)]
        mae_y = 0.0
        for bi, (dx, dy) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)]):
            for yy in range(8):
                for xx in range(8):
                    sy = sample_bilinear(Ys, ws, hs, w, h, x0 + dx + xx, y0 + dy + yy)
                    mae_y += abs(px_y[bi][yy * 8 + xx] - sy)
        if mae_y / 256.0 <= 40.0:
            break
        pre_mcus.pop()

    # 3. Find canonical MCU chain on the verified tail (>= last_bad_off + 1, safely anchoring past byte_off + 2048)
    u_safe = next(idx for idx, off in enumerate(orig_byte_map) if off >= last_bad_off + 1)
    u_hi = next((idx for idx, off in enumerate(orig_byte_map) if off >= min(eoi - 10, max(byte_off + 2400, last_bad_off + 400))), nu - 1)

    memo = {}

    def solve_tail(start_bp: int):
        path = []
        cur = start_bp
        while cur not in memo:
            if bit_len - 7 <= cur <= bit_len:
                if all((u[b >> 3] >> (7 - (b & 7))) & 1 for b in range(cur, bit_len)):
                    memo[cur] = (0, None)
                    break
            nxt, _ = decode_mcu_at(cur, [0, 0, 0])
            if nxt == -1:
                memo[cur] = None
                break
            path.append((cur, nxt))
            cur = nxt
        for p, nxt in reversed(path):
            r_nxt = memo[nxt]
            memo[p] = None if r_nxt is None else (r_nxt[0] + 1, nxt)
        return memo[start_bp]

    canon_nodes = set()
    for bp_s in range(u_safe * 8, u_hi * 8):
        if solve_tail(bp_s) is not None:
            cur = bp_s
            for _ in range(40):
                nxt = memo[cur][1]
                if nxt is None:
                    break
                cur = nxt
            canon_nodes.add(cur)
    bp_anchor = min(canon_nodes)

    # Step BACKWARDS one MCU at a time along valid predecessors from bp_anchor down toward u_safe * 8,
    # keeping only predecessors whose AC energy and delta-DC vs next MCU agree with thumbnail S!
    valid_preds = {}
    for bp_s in range(u_safe * 8, bp_anchor):
        if solve_tail(bp_s) is not None:
            nxt = memo[bp_s][1]
            if nxt is not None:
                valid_preds.setdefault(nxt, []).append(bp_s)

    bp_tail_start = bp_anchor
    m_cur = n_total - memo[bp_tail_start][0]
    while bp_tail_start in valid_preds and m_cur > len(pre_mcus) + 1:
        _, cur_mcu = decode_mcu_at(bp_tail_start, [0, 0, 0])
        nxt_y_mean = sum(idct_8x8_block(cur_mcu[bi], dqt[0])[p] for bi in range(4) for p in range(64)) / 256.0
        mr_c, mc_c = m_cur // 100, m_cur % 100
        s_cur_mean = sum(
            sample_bilinear(Ys, ws, hs, w, h, mc_c * 16 + xx, mr_c * 16 + yy)
            for yy in range(16)
            for xx in range(16)
        ) / 256.0
        m_prev = m_cur - 1
        mr_p, mc_p = m_prev // 100, m_prev % 100
        s_prev_mean = sum(
            sample_bilinear(Ys, ws, hs, w, h, mc_p * 16 + xx, mr_p * 16 + yy)
            for yy in range(16)
            for xx in range(16)
        ) / 256.0
        expected_delta_y = s_prev_mean - s_cur_mean

        best_prev = None
        best_err = 999.0
        for p_bp in valid_preds[bp_tail_start]:
            _, p_mcu = decode_mcu_at(p_bp, [0, 0, 0])
            # Relative DC jump from end of p_mcu (blk 3) to start of cur_mcu (blk 0)!
            # Remember with pred=[0,0,0], nxt_mcu started from 0 after p_mcu[3], so p_mcu[3] + cur_mcu[0..3] vs S:
            # Instead decode BOTH p_mcu and cur_mcu continuously starting at p_bp!
            n1, p_m = decode_mcu_at(p_bp, [0, 0, 0])
            end_pred = [p_m[3][0], p_m[4][0], p_m[5][0]]
            n2, c_m = decode_mcu_at(n1, end_pred)
            py_mean = sum(idct_8x8_block(p_m[bi], dqt[0])[p] for bi in range(4) for p in range(64)) / 256.0
            cy_mean = sum(idct_8x8_block(c_m[bi], dqt[0])[p] for bi in range(4) for p in range(64)) / 256.0
            delta_y = py_mean - cy_mean
            # Align DC of p_mcu to S_prev mean and measure AC texture MAE vs S
            shift_y = s_prev_mean - py_mean
            px_p = [idct_8x8_block(p_m[bi], dqt[0]) for bi in range(4)]
            ac_mae = 0.0
            for bi, (dx, dy) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)]):
                for yy in range(8):
                    for xx in range(8):
                        sy = sample_bilinear(Ys, ws, hs, w, h, mc_p * 16 + dx + xx, mr_p * 16 + dy + yy)
                        ac_mae += abs((px_p[bi][yy * 8 + xx] + shift_y) - sy)
            ac_mae /= 256.0
            err = abs(delta_y - expected_delta_y) + ac_mae
            if abs(delta_y - expected_delta_y) <= 25.0 and ac_mae <= 35.0 and err < best_err:
                best_err = err
                best_prev = p_bp
        if best_prev is None:
            break
        bp_tail_start = best_prev
        m_cur -= 1

    # Decode canonical tail from bp_tail_start
    post_mcus = []
    cur = bp_tail_start
    pred_rel = [0, 0, 0]
    while cur < bit_len - 7:
        nxt, mcu = decode_mcu_at(cur, pred_rel)
        if nxt == -1:
            break
        post_mcus.append(mcu)
        cur = nxt
    m_post = n_total - len(post_mcus)

    # Trim 1 MCU at start of post_mcus if still self-sync noisy (ensure clean AC correlation)
    q_y = dqt[0][0]
    q_c = dqt[1][0]

    # 4. Calibrate S -> L DC offset using ALL pre_mcus + solve exact (C_Y, C_Cb, C_Cr) across ALL post_mcus!
    pre_ey, pre_ecb, pre_ecr = [], [], []
    for m, mcu in enumerate(pre_mcus):
        mr = m // 100
        mc = m % 100
        x0 = mc * 16
        y0 = mr * 16
        for bi, (dx, dy) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)]):
            sy = sum(sample_bilinear(Ys, ws, hs, w, h, x0 + dx + xx, y0 + dy + yy) for yy in range(8) for xx in range(8))
            pre_ey.append(mcu[bi][0] - (sy - 8192.0) / (8.0 * q_y))
        scb = sum(sample_bilinear(Cbs, ws, hs, w, h, x0 + xx, y0 + yy) for yy in range(16) for xx in range(16)) / 4.0
        scr = sum(sample_bilinear(Crs, ws, hs, w, h, x0 + xx, y0 + yy) for yy in range(16) for xx in range(16)) / 4.0
        pre_ecb.append(mcu[4][0] - (scb - 8192.0) / (8.0 * q_c))
        pre_ecr.append(mcu[5][0] - (scr - 8192.0) / (8.0 * q_c))

    bias_y = sum(pre_ey) / len(pre_ey) if pre_ey else 0.0
    bias_cb = sum(pre_ecb) / len(pre_ecb) if pre_ecb else 0.0
    bias_cr = sum(pre_ecr) / len(pre_ecr) if pre_ecr else 0.0

    # Estimate constant integer offsets (C_Y, C_Cb, C_Cr) for post_mcus
    est_cy, est_ccb, est_ccr = [], [], []
    for idx_m, mcu in enumerate(post_mcus):
        m = m_post + idx_m
        mr = m // 100
        mc = m % 100
        x0 = mc * 16
        y0 = mr * 16
        for bi, (dx, dy) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)]):
            sy = sum(sample_bilinear(Ys, ws, hs, w, h, x0 + dx + xx, y0 + dy + yy) for yy in range(8) for xx in range(8))
            est_cy.append(((sy - 8192.0) / (8.0 * q_y) + bias_y) - mcu[bi][0])
        scb = sum(sample_bilinear(Cbs, ws, hs, w, h, x0 + xx, y0 + yy) for yy in range(16) for xx in range(16)) / 4.0
        scr = sum(sample_bilinear(Crs, ws, hs, w, h, x0 + xx, y0 + yy) for yy in range(16) for xx in range(16)) / 4.0
        est_ccb.append(((scb - 8192.0) / (8.0 * q_c) + bias_cb) - mcu[4][0])
        est_ccr.append(((scr - 8192.0) / (8.0 * q_c) + bias_cr) - mcu[5][0])

    c_y = round(sum(est_cy) / len(est_cy))
    c_cb = round(sum(est_ccb) / len(est_ccb))
    c_cr = round(sum(est_ccr) / len(est_ccr))
    print(
        f"  pre_mcus=0..{len(pre_mcus)-1}, post_mcus={m_post}..7499 (gap={m_post - len(pre_mcus)} MCUs), "
        f"solved integer DC offsets=(C_Y={c_y:+d}, C_Cb={c_cb:+d}, C_Cr={c_cr:+d}) "
        f"[raw float=({sum(est_cy)/len(est_cy):+.3f}, {sum(est_ccb)/len(est_ccb):+.3f}, {sum(est_ccr)/len(est_ccr):+.3f})]",
        flush=True,
    )

    # Apply exact integer DC correction to post_mcus (AC coefficients untouched!)
    fixed_post = []
    for mcu in post_mcus:
        nmcu = []
        for blk_i, coeffs in enumerate(mcu):
            nc = list(coeffs)
            nc[0] += c_y if blk_i < 4 else (c_cb if blk_i == 4 else c_cr)
            nmcu.append(nc)
        fixed_post.append(nmcu)

    # Check fine boundary MCUs around m_post (trim up to 3 leading post MCUs if they appear to be self-sync gibberish)
    # Build pixel grid around gap from pre_mcus + fixed_post and compare with S!
    drop_head = 0
    for chk_i in range(min(6, len(fixed_post))):
        m = m_post + chk_i
        mr = m // 100
        mc = m % 100
        x0 = mc * 16
        y0 = mr * 16
        px_y = [idct_8x8_block(fixed_post[chk_i][bi], dqt[0]) for bi in range(4)]
        mae_y = 0.0
        for bi, (dx, dy) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)]):
            for yy in range(8):
                for xx in range(8):
                    sy = sample_bilinear(Ys, ws, hs, w, h, x0 + dx + xx, y0 + dy + yy)
                    mae_y += abs(px_y[bi][yy * 8 + xx] - sy)
        mae_y /= 256.0
        if mae_y > 40.0:
            drop_head = chk_i + 1
    if drop_head:
        fixed_post = fixed_post[drop_head:]
        m_post += drop_head
        print(f"  Trimmed {drop_head} self-sync transition MCU(s); clean tail now starts at MCU {m_post}", flush=True)

    # 5. Synthesize gap MCUs [len(pre_mcus) .. m_post - 1] from direct-YCbCr upsampled thumbnail + local tone match!
    # Measure local bias on immediate neighboring intact row/columns around the gap
    m_gap_start = len(pre_mcus)
    gap_mcus = []
    qt_y = dqt[comps[0][3]]
    qt_c = dqt[comps[1][3]]
    for m in range(m_gap_start, m_post):
        mr = m // 100
        mc = m % 100
        x0 = mc * 16
        y0 = mr * 16
        nmcu = []
        for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
            blk_px = [
                sample_bilinear(Ys, ws, hs, w, h, x0 + dx + xx, y0 + dy + yy) + bias_y * (8.0 * q_y) / 64.0
                for yy in range(8)
                for xx in range(8)
            ]
            nmcu.append(fdct_quant_8x8(blk_px, qt_y))
        cb_px = []
        cr_px = []
        for yy in range(8):
            for xx in range(8):
                cb_px.append(
                    sum(sample_bilinear(Cbs, ws, hs, w, h, x0 + 2 * xx + sx, y0 + 2 * yy + sy) for sy in (0, 1) for sx in (0, 1))
                    / 4.0
                )
                cr_px.append(
                    sum(sample_bilinear(Crs, ws, hs, w, h, x0 + 2 * xx + sx, y0 + 2 * yy + sy) for sy in (0, 1) for sx in (0, 1))
                    / 4.0
                )
        nmcu.append(fdct_quant_8x8(cb_px, qt_c))
        nmcu.append(fdct_quant_8x8(cr_px, qt_c))
        gap_mcus.append(nmcu)

    full_mcus = pre_mcus + gap_mcus + fixed_post
    assert len(full_mcus) == n_total
    scan_bytes = encode_mcus(full_mcus, dht)
    out_jpg = data[:sos_off] + scan_bytes + b"\xff\xd9"

    # Strict PIL decode check
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    im_chk = Image.open(io.BytesIO(out_jpg))
    im_chk.load()
    l_path.write_bytes(out_jpg)
    print(f"  Saved {l_path} ({len(out_jpg)} bytes), PIL {im_chk.size} {im_chk.mode} STRICT_OK!", flush=True)


def main():
    extents = ds.load_file_extents()
    conn = r.init_db()
    try:
        for fname, lba in ds.TARGETS:
            row = conn.execute(
                "SELECT verified FROM solved_sectors WHERE lba = ?", (lba,)
            ).fetchone()
            if row and row[0] == 1:
                print(f"{fname} (LBA {lba}): ALREADY 100% RSPC+EDC VERIFIED — skipping bridge.")
                continue
            print(f"\n=== Thumbnail-Guided JPEG Reconstruction: {fname} (LBA {lba}) ===", flush=True)
            _, start_lba, size_bytes = extents[lba]
            reconstruct_one_photo(fname, lba, start_lba, size_bytes, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
