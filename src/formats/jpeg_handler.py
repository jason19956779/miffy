"""Lossless quantized-DCT baseline JPEG resynchronizer and gap reconstructor."""

import io
import math
import struct
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFile

from src.formats import FormatRegistry
from src.formats.base import FileFormatHandler
from src.core.models import FileExtent

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


def parse_jpeg_info(data: bytes):
    """Extract SOF0 dimensions, component sampling factors, DQT, DHT, SOS, and EOI offsets."""
    pos = 2
    dht = {}
    dqt = {}
    sos_comps = []
    comps = []
    sos_off = 0
    w = h = 0
    has_dri = False
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        m = data[pos + 1]
        if m == 0xFF:
            pos += 1
            continue
        if m == 0x00 or (0xD0 <= m <= 0xD7):
            pos += 2
            continue
        if m == 0xDD:
            has_dri = True
        if m == 0xDA:
            if pos + 4 > len(data):
                break
            ln = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
            if pos + 2 + ln > len(data) or ln < 6:
                break
            nc = data[pos + 4]
            if 5 + 2 * nc > 2 + ln:
                break
            sos_comps = [
                (data[pos + 5 + 2 * i], data[pos + 5 + 2 * i + 1] >> 4, data[pos + 5 + 2 * i + 1] & 0xF)
                for i in range(nc)
            ]
            sos_off = pos + 2 + ln
            break
        if pos + 4 > len(data):
            break
        ln = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        if ln < 2 or pos + 2 + ln > len(data):
            break
        seg = data[pos + 4 : pos + 2 + ln]
        if m == 0xC0 and len(seg) >= 6:
            _, h, w, ncomp = struct.unpack(">BHHB", seg[:6])
            if len(seg) < 6 + 3 * ncomp:
                break
            comps = [
                (seg[6 + 3 * i], seg[6 + 3 * i + 1] >> 4, seg[6 + 3 * i + 1] & 0xF, seg[6 + 3 * i + 2])
                for i in range(ncomp)
            ]
        elif m == 0xC4:
            i = 0
            while i + 17 <= len(seg):
                tc_th = seg[i]
                i += 1
                bits = list(seg[i : i + 16])
                i += 16
                cnt = sum(bits)
                if i + cnt > len(seg):
                    break
                vals = list(seg[i : i + cnt])
                i += cnt
                dht[(tc_th >> 4, tc_th & 0xF)] = (bits, vals)
        elif m == 0xDB:
            i = 0
            while i < len(seg):
                tq = seg[i] & 0xF
                prec = seg[i] >> 4
                i += 1
                need = 64 if prec == 0 else 128
                if i + need > len(seg):
                    break
                if prec == 0:
                    dqt[tq] = list(seg[i : i + 64])
                    i += 64
                else:
                    dqt[tq] = list(struct.unpack(">64H", seg[i : i + 128]))
                    i += 128
        pos += 2 + ln
    eoi = data.rfind(b"\xff\xd9")
    if eoi <= sos_off:
        eoi = len(data)
    return w, h, comps, sos_comps, dqt, dht, sos_off, eoi, has_dri


def build_dec_huff(bits: list[int], vals: list[int]):
    if len(bits) != 16 or sum(bits) != len(vals):
        raise ValueError("Corrupted DHT bit counts")
    lut_sym = [-1] * 65536
    lut_len = [0] * 65536
    code = 0
    vi = 0
    for b_len in range(1, 17):
        shift = 16 - b_len
        for _ in range(bits[b_len - 1]):
            if code >= (1 << b_len):
                raise ValueError("Corrupted Huffman code table overflow")
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
    u = bytearray()
    orig_map = []
    i = sos_off
    while i < eoi:
        b = data[i]
        u.append(b)
        orig_map.append(i)
        if b == 0xFF and i + 1 < eoi and data[i + 1] == 0x00:
            i += 2
        else:
            i += 1
    return bytes(u), orig_map


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
    return [round(f[ZIGZAG[k]] / q_table[k]) for k in range(64)]


def build_mcu_layout(comps, sos_comps, dqt, dht):
    """Build generic per-block descriptor list matching SOF0 sampling & SOS tables."""
    h_max = max(c[1] for c in comps)
    v_max = max(c[2] for c in comps)
    sos_map = {cid: (td, ta) for cid, td, ta in sos_comps}
    dec_cache = {k: build_dec_huff(*v) for k, v in dht.items()}
    enc_cache = {k: build_enc_huff(*v) for k, v in dht.items()}
    blocks = []
    for comp_idx, (cid, ch, cv, tq) in enumerate(comps):
        td, ta = sos_map[cid]
        dc_dec = dec_cache[(0, td)]
        ac_dec = dec_cache[(1, ta)]
        dc_enc = enc_cache[(0, td)]
        ac_enc = enc_cache[(1, ta)]
        qt = dqt[tq]
        for by in range(cv):
            for bx in range(ch):
                blocks.append(
                    {
                        "comp_idx": comp_idx,
                        "h": ch,
                        "v": cv,
                        "bx": bx,
                        "by": by,
                        "h_max": h_max,
                        "v_max": v_max,
                        "dc_dec": dc_dec,
                        "ac_dec": ac_dec,
                        "dc_enc": dc_enc,
                        "ac_enc": ac_enc,
                        "qt": qt,
                    }
                )
    return h_max * 8, v_max * 8, blocks


def decode_jpeg_ycbcr_planes(data: bytes):
    w, h, comps, sos_comps, dqt, dht, sos_off, eoi, _ = parse_jpeg_info(data)
    mcu_w, mcu_h, blk_spec = build_mcu_layout(comps, sos_comps, dqt, dht)
    nx = (w + mcu_w - 1) // mcu_w
    ny = (h + mcu_h - 1) // mcu_h
    u, _ = unstuff_scan(data, sos_off, eoi)
    nu = len(u)
    bit_len = nu * 8
    bp = 0
    pred = [0] * len(comps)
    planes = [[128.0] * (w * h) for _ in range(len(comps))]
    mcus = []

    for mr in range(ny):
        for mc in range(nx):
            mcu_coeffs = []
            for spec in blk_spec:
                dc_s, dc_l = spec["dc_dec"]
                ac_s, ac_l = spec["ac_dec"]
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
                c_i = spec["comp_idx"]
                pred[c_i] += diff
                coeffs = [0] * 64
                coeffs[0] = pred[c_i]
                k = 1
                while k < 64 and bp < bit_len:
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
                        if k < 64:
                            coeffs[k] = val
                        k += 1
                mcu_coeffs.append(coeffs)
                px = idct_8x8_block(coeffs, spec["qt"])
                sx = spec["h_max"] // spec["h"]
                sy = spec["v_max"] // spec["v"]
                plane = planes[c_i]
                ox0 = mc * mcu_w + spec["bx"] * 8 * sx
                oy0 = mr * mcu_h + spec["by"] * 8 * sy
                for yy in range(8):
                    for dy in range(sy):
                        py = oy0 + yy * sy + dy
                        if py >= h:
                            continue
                        for xx in range(8):
                            v = px[yy * 8 + xx]
                            for dx in range(sx):
                                px_x = ox0 + xx * sx + dx
                                if px_x < w:
                                    plane[py * w + px_x] = v
            mcus.append(mcu_coeffs)
    return w, h, dqt, mcus, planes


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


def encode_mcus(mcus: list[list[list[int]]], blk_spec: list[dict], n_comps: int) -> bytes:
    out = bytearray()
    acc = 0
    nbits = 0

    def put(code: int, length: int) -> None:
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

    pred = [0] * n_comps
    for mcu in mcus:
        for b_i, spec in enumerate(blk_spec):
            coeffs = mcu[b_i]
            c_i = spec["comp_idx"]
            dc_e = spec["dc_enc"]
            ac_e = spec["ac_enc"]
            prev_dc = pred[c_i]
            diff = coeffs[0] - prev_dc
            ssss = abs(diff).bit_length() if diff != 0 else 0
            if ssss not in dc_e:
                avail_dc = [s for s in dc_e if s <= ssss]
                s_use = max(avail_dc) if avail_dc else min(dc_e)
                if s_use == 0:
                    diff = 0
                    ssss = 0
                else:
                    lim = (1 << s_use) - 1
                    diff = max(-lim, min(lim, diff))
                    ssss = s_use
            pred[c_i] = prev_dc + diff
            if diff == 0:
                c, ln = dc_e[0]
                put(c, ln)
            else:
                c, ln = dc_e[ssss]
                put(c, ln)
                bits = diff if diff > 0 else diff + (1 << ssss) - 1
                put(bits, ssss)

            # Deferred pending_zrl keeps intact MCUs 100% bit-exact while guarding sparse custom DHTs
            zeros = 0
            pending_zrl = 0
            encoded_through = 0
            for k in range(1, 64):
                v = coeffs[k]
                if v != 0:
                    ss = abs(v).bit_length()
                    sym = (zeros << 4) | ss
                    if sym not in ac_e:
                        found = False
                        for s2 in range(ss - 1, 0, -1):
                            cand_sym = (zeros << 4) | s2
                            if cand_sym in ac_e:
                                lim = (1 << s2) - 1
                                v = lim if v > 0 else -lim
                                ss = s2
                                sym = cand_sym
                                found = True
                                break
                        if not found:
                            v = 0
                if v == 0:
                    zeros += 1
                    if zeros == 16:
                        pending_zrl += 1
                        zeros = 0
                else:
                    if pending_zrl > 0 and 0xF0 not in ac_e:
                        break
                    while pending_zrl > 0:
                        c, ln = ac_e[0xF0]
                        put(c, ln)
                        pending_zrl -= 1
                    c, ln = ac_e[(zeros << 4) | ss]
                    put(c, ln)
                    bits = v if v > 0 else v + (1 << ss) - 1
                    put(bits, ss)
                    zeros = 0
                    encoded_through = k
            if encoded_through < 63 and 0x00 in ac_e:
                c, ln = ac_e[0x00]
                put(c, ln)
    if nbits > 0:
        put((1 << (8 - nbits)) - 1, 8 - nbits)
    return bytes(out)


def extract_exif_thumbnail(data: bytes) -> bytes | None:
    """Locate embedded JPEG SOI..EOI inside APP1 EXIF segment if intact."""
    idx = 2
    limit = min(len(data) - 4, 65536)
    while idx < limit:
        if data[idx] != 0xFF:
            idx += 1
            continue
        m = data[idx + 1]
        if m == 0xDA:
            break
        if m == 0xFF:
            idx += 1
            continue
        if idx + 4 > len(data):
            break
        ln = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
        if m == 0xE1:
            seg = data[idx + 4 : idx + 2 + ln]
            s_i = seg.find(b"\xff\xd8\xff")
            if s_i != -1:
                e_i = seg.find(b"\xff\xd9", s_i + 2)
                if e_i != -1:
                    return seg[s_i : e_i + 2]
        idx += 2 + ln
    return None


def validate_jpeg_t81_structure(data: bytes) -> tuple[bool, list[str], dict]:
    """Strictly verify ISO/IEC 10918-1 (ITU-T T.81) marker framing, Kraft DHTs, ECS stuffing, and EOI."""
    issues: list[str] = []
    n = len(data)
    if n < 4 or data[0:2] != b"\xff\xd8":
        return False, ["missing_SOI"], {"t81_valid": False}

    pos = 2
    dqt_ids: set[int] = set()
    dht_dc: set[int] = set()
    dht_ac: set[int] = set()
    sof_info: tuple[int, int, int, int, list[tuple[int, int, int, int]]] | None = None
    dri = 0
    scans = 0
    eoi_pos: int | None = None
    trailing_zero_bytes = 0

    while pos < n:
        if data[pos] != 0xFF:
            issues.append(f"non_marker_byte_at_{pos}")
            break
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            issues.append("truncated_marker")
            break
        m = data[pos]
        pos += 1
        if m == 0xD9:
            eoi_pos = pos
            break
        if m == 0x01 or (0xD0 <= m <= 0xD7):
            continue
        if pos + 2 > n:
            issues.append(f"truncated_seg_len_{m:02X}")
            break
        seg_len = int.from_bytes(data[pos : pos + 2], "big")
        if seg_len < 2 or pos + seg_len > n:
            issues.append(f"bad_seg_len_{m:02X}:{seg_len}")
            break
        payload = data[pos + 2 : pos + seg_len]
        pos += seg_len

        if m == 0xDB:
            p = 0
            while p < len(payload):
                pq_tq = payload[p]
                p += 1
                pq, tq = pq_tq >> 4, pq_tq & 0x0F
                if pq not in (0, 1) or tq > 3:
                    issues.append(f"bad_dqt_hdr:{pq},{tq}")
                    break
                step = 64 * (pq + 1)
                if p + step > len(payload):
                    issues.append("truncated_dqt")
                    break
                dqt_ids.add(tq)
                p += step
            if p != len(payload):
                issues.append("dqt_trailing_bytes")
        elif m == 0xC4:
            p = 0
            while p < len(payload):
                tc_th = payload[p]
                p += 1
                tc, th = tc_th >> 4, tc_th & 0x0F
                if tc not in (0, 1) or th > 3:
                    issues.append(f"bad_dht_hdr:{tc},{th}")
                    break
                if p + 16 > len(payload):
                    issues.append("truncated_dht_counts")
                    break
                counts = list(payload[p : p + 16])
                p += 16
                total = sum(counts)
                if p + total > len(payload):
                    issues.append("truncated_dht_syms")
                    break
                syms = list(payload[p : p + total])
                p += total
                kraft_num = sum(c * (1 << (16 - L)) for L, c in enumerate(counts, 1))
                if kraft_num >= (1 << 16):
                    issues.append(f"dht_kraft_violation_tc{tc}_th{th}")
                if len(set(syms)) != len(syms):
                    issues.append(f"dht_dup_sym_tc{tc}_th{th}")
                if tc == 0:
                    dht_dc.add(th)
                else:
                    dht_ac.add(th)
            if p != len(payload):
                issues.append("dht_trailing_bytes")
        elif m == 0xDD:
            if len(payload) != 2:
                issues.append("bad_dri_len")
            else:
                dri = int.from_bytes(payload, "big")
        elif m in (0xC0, 0xC1, 0xC2):
            if len(payload) < 6:
                issues.append("truncated_sof")
                break
            prec, H, W, ncomp = struct.unpack(">BHHB", payload[:6])
            if prec not in (8, 12) or W <= 0 or H <= 0 or ncomp not in (1, 3, 4):
                issues.append(f"bad_sof_params:{prec},{W}x{H},{ncomp}")
            if len(payload) != 6 + 3 * ncomp:
                issues.append("bad_sof_len")
                break
            comps = [
                (payload[6 + 3 * i], payload[7 + 3 * i] >> 4, payload[7 + 3 * i] & 0x0F, payload[8 + 3 * i])
                for i in range(ncomp)
            ]
            for cid, ch, cv, qt in comps:
                if not (1 <= ch <= 4 and 1 <= cv <= 4 and qt <= 3):
                    issues.append(f"bad_comp_spec:{cid},{ch},{cv},{qt}")
            sof_info = (m, prec, W, H, comps)
        elif m == 0xDA:
            if len(payload) < 1:
                issues.append("truncated_sos")
                break
            ns = payload[0]
            if ns < 1 or len(payload) != 1 + 2 * ns + 3:
                issues.append("bad_sos_len")
                break
            scomps = [
                (payload[1 + 2 * i], payload[2 + 2 * i] >> 4, payload[2 + 2 * i] & 0x0F)
                for i in range(ns)
            ]
            ss, se, ahal = payload[-3], payload[-2], payload[-1]
            if sof_info is not None:
                comp_ids = {c[0] for c in sof_info[4]}
                for cs, td, ta in scomps:
                    if cs not in comp_ids:
                        issues.append(f"undefined_sos_comp:{cs}")
                    if sof_info[0] in (0xC0, 0xC1):
                        if td not in dht_dc or ta not in dht_ac:
                            issues.append(f"undefined_sos_dht:{td},{ta}")
                if sof_info[0] in (0xC0, 0xC1) and (ss, se, ahal) != (0, 63, 0):
                    issues.append(f"non_baseline_spectral:{ss},{se},{ahal}")
            scans += 1
            cur = pos
            is_prog = sof_info is not None and sof_info[0] == 0xC2
            while cur < n:
                idx_ff = data.find(b"\xff", cur)
                if idx_ff == -1 or idx_ff + 1 >= n:
                    issues.append("missing_EOI_in_scan")
                    pos = n
                    break
                nxt = data[idx_ff + 1]
                if nxt == 0x00:
                    cur = idx_ff + 2
                elif nxt == 0xFF:
                    cur = idx_ff + 1
                elif 0xD0 <= nxt <= 0xD7:
                    if dri == 0:
                        issues.append(f"unexpected_RST_{nxt:02X}_with_dri0")
                    cur = idx_ff + 2
                elif nxt == 0xD9 or (is_prog and nxt in (0xDA, 0xC4, 0xDB, 0xDD, 0xFE)):
                    pos = idx_ff
                    break
                else:
                    issues.append(f"illegal_marker_in_scan:0xFF{nxt:02X}")
                    pos = idx_ff
                    break

    if eoi_pos is None:
        issues.append("missing_EOI")
    elif eoi_pos < n:
        trailing = data[eoi_pos:]
        if all(b == 0 for b in trailing):
            trailing_zero_bytes = len(trailing)
        else:
            issues.append(f"trailing_nonzero_after_EOI:{len(trailing)}")

    if sof_info is None or scans == 0:
        issues.append("missing_SOF_or_SOS")
    else:
        for _, _, _, qt in sof_info[4]:
            if qt not in dqt_ids:
                issues.append(f"undefined_DQT:{qt}")

    meta = {
        "t81_valid": len(issues) == 0,
        "sof_marker": f"0x{sof_info[0]:02X}" if sof_info else None,
        "width": sof_info[2] if sof_info else 0,
        "height": sof_info[3] if sof_info else 0,
        "dri": dri,
        "scans": scans,
        "trailing_zero_bytes": trailing_zero_bytes,
    }
    return len(issues) == 0, issues, meta


@FormatRegistry.register
class JpegFormatHandler(FileFormatHandler):
    name = "jpeg"
    extensions = (".JPG", ".JPEG")
    priority = 10
    description = "Baseline JPEG thumbnail pairing + DP suffix MCU resync + lossless DCT gap repair."

    def can_handle(self, ext: str, data: bytes | None = None) -> bool:
        if ext.upper() in self.extensions:
            return True
        return bool(data and data[:2] == b"\xff\xd8")

    def find_reference(
        self,
        extent: FileExtent,
        all_extents: dict[str, FileExtent],
        read_file_fn: Callable[[FileExtent], bytes],
        file_bytes: bytes | None = None,
    ) -> tuple[bytes | None, str | None]:
        parent = str(Path(extent.directory).parent).rstrip("/")
        stem = Path(extent.name).stem
        all_upper = {k.upper(): v for k, v in all_extents.items()}
        candidates = [
            f"/PHOTO/S/{extent.name}",
            f"/PHOTO/S/{stem}_T{extent.ext}",
            f"{parent}/S/{extent.name}",
            f"{parent}/S/{stem}_T{extent.ext}",
            f"{extent.directory.rstrip('/')}/S/{extent.name}",
            f"{extent.directory.rstrip('/')}/{stem}_T{extent.ext}",
        ]
        if stem.upper().endswith("_T"):
            orig_stem = stem[:-2]
            candidates.extend([
                f"/PHOTO/L_PHOTO/{orig_stem}{extent.ext}",
                f"{parent}/L_PHOTO/{orig_stem}{extent.ext}",
            ])
        for cand_path in candidates:
            ref_ext = all_upper.get(cand_path.upper())
            if ref_ext and ref_ext.path != extent.path and not ref_ext.bad_lbas:
                return read_file_fn(ref_ext), ref_ext.path
        for cand_path in candidates:
            ref_ext = all_upper.get(cand_path.upper())
            if ref_ext and ref_ext.path != extent.path:
                data = read_file_fn(ref_ext)
                ok, _, _ = self.validate(data)
                if ok:
                    return data, ref_ext.path

        if file_bytes:
            exif_thumb = extract_exif_thumbnail(file_bytes)
            if exif_thumb:
                ok, _, _ = self.validate(exif_thumb)
                if ok:
                    return exif_thumb, "EXIF_APP1_THUMBNAIL"
        return None, None

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        t81_ok, t81_issues, t81_meta = validate_jpeg_t81_structure(data)
        if not t81_ok:
            return False, f"T81_STRUCT_ERR ({'; '.join(t81_issues[:3])})", t81_meta
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im2:
                im2.load()
                meta = {"dims": f"{im2.size[0]}x{im2.size[1]} {im2.mode}", **t81_meta}
            return True, "STRICT_OK", meta
        except Exception as exc_strict:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                with Image.open(io.BytesIO(data)) as im3:
                    im3.load()
                    meta = {"dims": f"{im3.size[0]}x{im3.size[1]} {im3.mode}", **t81_meta}
                return False, f"TRUNCATED_OK ({type(exc_strict).__name__})", meta
            except Exception as exc_trunc:
                return False, f"DECODE_ERR ({type(exc_trunc).__name__}: {exc_trunc})", t81_meta
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = False

    def reconstruct(
        self,
        data: bytes,
        bad_intervals: list[tuple[int, int]],
        reference_data: bytes | None = None,
        context: dict | None = None,
    ) -> tuple[bytes, str, dict]:
        if not bad_intervals:
            return data, "NO_BAD_INTERVALS", {}
        first_bad_off = min(s for s, _ in bad_intervals)
        last_bad_off = max(e for _, e in bad_intervals) - 1

        try:
            w, h, comps, sos_comps, dqt, dht, sos_off, eoi, has_dri = parse_jpeg_info(data)
            if w == 0 or h == 0 or len(comps) != 3 or has_dri or not dht or sos_off <= 0 or sos_off >= len(data):
                return data, "UNSUPPORTED_JPEG_LAYOUT", {}
            if first_bad_off < sos_off:
                return data, "HEADER_CORRUPTED_UNSUPPORTED", {}
            mcu_w, mcu_h, blk_spec = build_mcu_layout(comps, sos_comps, dqt, dht)
        except Exception as exc:
            return data, f"HEADER_PARSE_ERROR ({type(exc).__name__})", {}
        nx = (w + mcu_w - 1) // mcu_w
        ny = (h + mcu_h - 1) // mcu_h
        n_total = nx * ny
        n_comps = len(comps)

        u, orig_map = unstuff_scan(data, sos_off, eoi)
        nu = len(u)
        bit_len = nu * 8

        def decode_mcu_at(bp: int, pred: list[int]):
            mcu = []
            for spec in blk_spec:
                dc_s, dc_l = spec["dc_dec"]
                ac_s, ac_l = spec["ac_dec"]
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
                c_i = spec["comp_idx"]
                pred[c_i] += diff
                coeffs = [0] * 64
                coeffs[0] = pred[c_i]
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

        # 1. Decode intact prefix MCUs strictly before first_bad_off
        pre_mcus = []
        bp = 0
        pred_pre = [0] * n_comps
        while len(pre_mcus) < n_total:
            tmp_pred = list(pred_pre)
            nxt, mcu = decode_mcu_at(bp, tmp_pred)
            if nxt == -1:
                break
            last_u = min((nxt - 1) >> 3, nu - 1)
            last_file_off = orig_map[last_u] + (1 if u[last_u] == 0xFF else 0)
            if last_file_off >= first_bad_off:
                break
            pred_pre = tmp_pred
            pre_mcus.append(mcu)
            bp = nxt

        # Optional reference thumbnail direct-YCbCr decoding
        ref_planes = None
        ref_w = ref_h = 0
        if reference_data:
            try:
                ref_w, ref_h, _, _, ref_planes = decode_jpeg_ycbcr_planes(reference_data)
            except Exception:
                ref_planes = None

        # Trim trailing prefix MCU if thumbnail detects a boundary corruption
        y_blocks = [i for i, sp in enumerate(blk_spec) if sp["comp_idx"] == 0]
        if ref_planes is not None:
            Ys = ref_planes[0]
            while pre_mcus:
                m = len(pre_mcus) - 1
                mr, mc = m // nx, m % nx
                x0, y0 = mc * mcu_w, mr * mcu_h
                mae_y = 0.0
                for bi in y_blocks:
                    sp = blk_spec[bi]
                    px = idct_8x8_block(pre_mcus[-1][bi], sp["qt"])
                    ox, oy = x0 + sp["bx"] * 8, y0 + sp["by"] * 8
                    for yy in range(8):
                        for xx in range(8):
                            mae_y += abs(px[yy * 8 + xx] - sample_bilinear(Ys, ref_w, ref_h, w, h, ox + xx, oy + yy))
                if mae_y / max(1, 64 * len(y_blocks)) <= 40.0:
                    break
                pre_mcus.pop()

        # 2. DP-memoized suffix scan starting safely past last_bad_off
        if last_bad_off + 1 >= eoi - 8:
            # Corruption spans the very end of the file; fill trailing MCUs from reference/edge
            m_post = n_total
            post_mcus = []
        else:
            u_safe = next((idx for idx, off in enumerate(orig_map) if off >= last_bad_off + 1), nu - 1)
            u_hi = next((idx for idx, off in enumerate(orig_map) if off >= min(eoi - 4, last_bad_off + 512)), nu - 1)
            memo: dict[int, tuple[int, int | None] | None] = {}

            def solve_tail(start_bp: int):
                path = []
                cur = start_bp
                while cur not in memo:
                    if bit_len - 7 <= cur <= bit_len:
                        if all((u[b >> 3] >> (7 - (b & 7))) & 1 for b in range(cur, bit_len)):
                            memo[cur] = (0, None)
                            break
                    nxt, _ = decode_mcu_at(cur, [0] * n_comps)
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
            for bp_s in range(u_safe * 8, max(u_safe * 8 + 1, u_hi * 8)):
                if solve_tail(bp_s) is not None:
                    cur = bp_s
                    for _ in range(40):
                        nxt = memo[cur][1]
                        if nxt is None:
                            break
                        cur = nxt
                    canon_nodes.add(cur)
            if not canon_nodes:
                return data, "TAIL_SYNC_NOT_FOUND", {}
            bp_anchor = min(canon_nodes)
            if bp_anchor < u_safe * 8 or memo[bp_anchor] is None:
                return data, "TAIL_SYNC_INVALID", {}

            # Backward step toward u_safe * 8 along clean canonical predecessors
            valid_preds: dict[int, list[int]] = {}
            for bp_s in range(u_safe * 8, bp_anchor):
                if solve_tail(bp_s) is not None:
                    nxt = memo[bp_s][1]
                    if nxt is not None:
                        valid_preds.setdefault(nxt, []).append(bp_s)

            bp_tail_start = bp_anchor
            m_cur = n_total - memo[bp_tail_start][0]
            if m_cur < 0 or m_cur > n_total:
                return data, "TAIL_SYNC_MCU_RANGE_ERR", {}
            while bp_tail_start in valid_preds and m_cur > len(pre_mcus) + 1:
                cands = valid_preds[bp_tail_start]
                chosen = None
                if ref_planes is not None:
                    Ys = ref_planes[0]
                    m_prev = m_cur - 1
                    mr_p, mc_p = m_prev // nx, m_prev % nx
                    s_prev = sum(
                        sample_bilinear(Ys, ref_w, ref_h, w, h, mc_p * mcu_w + xx, mr_p * mcu_h + yy)
                        for yy in range(mcu_h)
                        for xx in range(mcu_w)
                    ) / (mcu_w * mcu_h)
                    has_next = m_cur < n_total
                    if has_next:
                        mr_c, mc_c = m_cur // nx, m_cur % nx
                        s_cur = sum(
                            sample_bilinear(Ys, ref_w, ref_h, w, h, mc_c * mcu_w + xx, mr_c * mcu_h + yy)
                            for yy in range(mcu_h)
                            for xx in range(mcu_w)
                        ) / (mcu_w * mcu_h)
                        exp_delta = s_prev - s_cur
                    else:
                        exp_delta = 0.0

                    best_err = 999.0
                    for p_bp in cands:
                        n1, p_m = decode_mcu_at(p_bp, [0] * n_comps)
                        if p_m is None or n1 == -1:
                            continue
                        px_p = [idct_8x8_block(p_m[bi], blk_spec[bi]["qt"]) for bi in y_blocks]
                        py_m = sum(sum(p) for p in px_p) / (64 * len(y_blocks))
                        if has_next:
                            end_pred = [0] * n_comps
                            for bi, sp in enumerate(blk_spec):
                                end_pred[sp["comp_idx"]] = p_m[bi][0]
                            _, c_m = decode_mcu_at(n1, end_pred)
                            if c_m is None:
                                continue
                            px_c = [idct_8x8_block(c_m[bi], blk_spec[bi]["qt"]) for bi in y_blocks]
                            cy_m = sum(sum(p) for p in px_c) / (64 * len(y_blocks))
                            delta_err = abs((py_m - cy_m) - exp_delta)
                            if delta_err > 25.0:
                                continue
                        else:
                            delta_err = 0.0
                        shift = s_prev - py_m
                        ac_mae = 0.0
                        for idx_b, bi in enumerate(y_blocks):
                            sp = blk_spec[bi]
                            ox, oy = mc_p * mcu_w + sp["bx"] * 8, mr_p * mcu_h + sp["by"] * 8
                            for yy in range(8):
                                for xx in range(8):
                                    ac_mae += abs((px_p[idx_b][yy * 8 + xx] + shift) - sample_bilinear(Ys, ref_w, ref_h, w, h, ox + xx, oy + yy))
                        ac_mae /= 64 * len(y_blocks)
                        err = delta_err + ac_mae
                        if ac_mae <= 35.0 and err < best_err:
                            best_err = err
                            chosen = p_bp
                elif len(cands) == 1:
                    chosen = cands[0]
                if chosen is None:
                    break
                bp_tail_start = chosen
                m_cur -= 1

            post_mcus = []
            cur = bp_tail_start
            pred_rel = [0] * n_comps
            while cur < bit_len - 7 and len(post_mcus) < n_total:
                nxt, mcu = decode_mcu_at(cur, pred_rel)
                if nxt == -1 or mcu is None:
                    break
                post_mcus.append(mcu)
                cur = nxt
            m_post = n_total - len(post_mcus)
            if m_post < len(pre_mcus):
                pre_mcus = pre_mcus[: max(0, m_post)]

        # 3. Solve exact integer differential DC offsets [C_0 .. C_{n_comps-1}]
        c_offsets = [0] * n_comps
        mode_label = "DCT_THUMB_RESYNC"
        if post_mcus:
            if ref_planes is not None:
                # Calibrate direct-YCbCr residual on pre_mcus + solve integer C_i over post_mcus
                bias = [0.0] * n_comps
                for c_i in range(n_comps):
                    q_dc = dqt[comps[c_i][3]][0]
                    blk_indices = [i for i, sp in enumerate(blk_spec) if sp["comp_idx"] == c_i]
                    pre_diffs = []
                    for m, mcu in enumerate(pre_mcus):
                        mr, mc = m // nx, m % nx
                        for bi in blk_indices:
                            sp = blk_spec[bi]
                            sx, sy = sp["h_max"] // sp["h"], sp["v_max"] // sp["v"]
                            ox0 = mc * mcu_w + sp["bx"] * 8 * sx
                            oy0 = mr * mcu_h + sp["by"] * 8 * sy
                            s_sum = 0.0
                            for yy in range(8):
                                for xx in range(8):
                                    s_sum += sample_bilinear(
                                        ref_planes[c_i],
                                        ref_w,
                                        ref_h,
                                        w,
                                        h,
                                        min(w - 1, ox0 + xx * sx),
                                        min(h - 1, oy0 + yy * sy),
                                    )
                            pre_diffs.append(mcu[bi][0] - (s_sum - 8192.0) / (8.0 * q_dc))
                    bias[c_i] = sum(pre_diffs) / len(pre_diffs) if pre_diffs else 0.0

                    ests = []
                    for idx_m, mcu in enumerate(post_mcus):
                        m = m_post + idx_m
                        mr, mc = m // nx, m % nx
                        for bi in blk_indices:
                            sp = blk_spec[bi]
                            sx, sy = sp["h_max"] // sp["h"], sp["v_max"] // sp["v"]
                            ox0 = mc * mcu_w + sp["bx"] * 8 * sx
                            oy0 = mr * mcu_h + sp["by"] * 8 * sy
                            s_sum = 0.0
                            for yy in range(8):
                                for xx in range(8):
                                    s_sum += sample_bilinear(
                                        ref_planes[c_i],
                                        ref_w,
                                        ref_h,
                                        w,
                                        h,
                                        min(w - 1, ox0 + xx * sx),
                                        min(h - 1, oy0 + yy * sy),
                                    )
                            ests.append(((s_sum - 8192.0) / (8.0 * q_dc) + bias[c_i]) - mcu[bi][0])
                    c_offsets[c_i] = round(sum(ests) / len(ests))
            else:
                # Blind Self-Calibrated DC solver: match row mean across intact rows + [0,255] clipping bounds
                mode_label = "DCT_BLIND_RESYNC"
                for c_i in range(n_comps):
                    q_dc = dqt[comps[c_i][3]][0]
                    blk_indices = [i for i, sp in enumerate(blk_spec) if sp["comp_idx"] == c_i]
                    step_px = q_dc / 8.0
                    # Clipping bounds from IDCT block DC extrema
                    dc_vals = [mcu[bi][0] for mcu in post_mcus for bi in blk_indices]
                    c_lo = math.ceil((-128.0 / step_px) - min(dc_vals))
                    c_hi = math.floor((127.0 / step_px) - max(dc_vals))
                    if len(pre_mcus) >= nx and len(post_mcus) >= nx:
                        row_pre = [
                            mcu[bi][0]
                            for mcu in pre_mcus[max(0, len(pre_mcus) - nx) :]
                            for bi in blk_indices
                        ]
                        row_post = [
                            mcu[bi][0] for mcu in post_mcus[:nx] for bi in blk_indices
                        ]
                        est = round((sum(row_pre) / len(row_pre)) - (sum(row_post) / len(row_post)))
                    else:
                        est = round(-sum(dc_vals) / len(dc_vals))
                    if c_lo <= c_hi:
                        est = max(c_lo, min(c_hi, est))
                    c_offsets[c_i] = est

            fixed_post = []
            for mcu in post_mcus:
                nmcu = []
                for bi, coeffs in enumerate(mcu):
                    nc = list(coeffs)
                    nc[0] += c_offsets[blk_spec[bi]["comp_idx"]]
                    nmcu.append(nc)
                fixed_post.append(nmcu)

            if ref_planes is not None:
                Ys = ref_planes[0]
                drop_head = 0
                for chk_i in range(min(6, len(fixed_post))):
                    m = m_post + chk_i
                    mr, mc = m // nx, m % nx
                    x0, y0 = mc * mcu_w, mr * mcu_h
                    mae_y = 0.0
                    for bi in y_blocks:
                        sp = blk_spec[bi]
                        px = idct_8x8_block(fixed_post[chk_i][bi], sp["qt"])
                        ox, oy = x0 + sp["bx"] * 8, y0 + sp["by"] * 8
                        for yy in range(8):
                            for xx in range(8):
                                mae_y += abs(px[yy * 8 + xx] - sample_bilinear(Ys, ref_w, ref_h, w, h, ox + xx, oy + yy))
                    if mae_y / max(1, 64 * len(y_blocks)) > 40.0:
                        drop_head = chk_i + 1
                if drop_head:
                    fixed_post = fixed_post[drop_head:]
                    m_post += drop_head
        else:
            fixed_post = []

        # 4. Synthesize missing MCUs [m_gap_start .. m_post - 1]
        m_gap_start = len(pre_mcus)
        gap_mcus = []
        for m in range(m_gap_start, m_post):
            mr, mc = m // nx, m % nx
            nmcu = []
            for blk_idx, sp in enumerate(blk_spec):
                c_i = sp["comp_idx"]
                sx, sy = sp["h_max"] // sp["h"], sp["v_max"] // sp["v"]
                ox0 = mc * mcu_w + sp["bx"] * 8 * sx
                oy0 = mr * mcu_h + sp["by"] * 8 * sy
                if ref_planes is not None:
                    plane = ref_planes[c_i]
                    blk_px = []
                    for yy in range(8):
                        for xx in range(8):
                            vals = [
                                sample_bilinear(
                                    plane,
                                    ref_w,
                                    ref_h,
                                    w,
                                    h,
                                    min(w - 1, ox0 + xx * sx + dx),
                                    min(h - 1, oy0 + yy * sy + dy),
                                )
                                for dy in range(sy)
                                for dx in range(sx)
                            ]
                            blk_px.append(sum(vals) / len(vals))
                else:
                    # Smooth vertical blend between nearest above/below intact MCU DC planes
                    top_m = m - nx
                    bot_m = m + nx
                    top_val = 128.0
                    bot_val = 128.0
                    q_dc = sp["qt"][0]
                    if 0 <= top_m < len(pre_mcus):
                        top_val = 128.0 + pre_mcus[top_m][blk_idx][0] * q_dc / 8.0
                    elif len(pre_mcus) > 0:
                        top_val = 128.0 + pre_mcus[-1][blk_idx][0] * q_dc / 8.0
                    if m_post <= bot_m < n_total:
                        bot_val = 128.0 + fixed_post[bot_m - m_post][blk_idx][0] * q_dc / 8.0
                    elif fixed_post:
                        bot_val = 128.0 + fixed_post[0][blk_idx][0] * q_dc / 8.0
                    if len(pre_mcus) == 0 and fixed_post:
                        top_val = bot_val
                    elif not fixed_post and len(pre_mcus) > 0:
                        bot_val = top_val
                    blk_px = [
                        max(0.0, min(255.0, top_val * (1.0 - yy / 7.0) + bot_val * (yy / 7.0)))
                        for yy in range(8)
                        for _ in range(8)
                    ]
                nmcu.append(fdct_quant_8x8(blk_px, sp["qt"]))
            gap_mcus.append(nmcu)

        full_mcus = pre_mcus + gap_mcus + fixed_post
        if len(full_mcus) != n_total:
            return data, f"MCU_COUNT_MISMATCH ({len(full_mcus)}!={n_total})", {}

        scan_bytes = encode_mcus(full_mcus, blk_spec, n_comps)
        rebuilt = data[:sos_off] + scan_bytes + b"\xff\xd9"
        ok, status, _ = self.validate(rebuilt)
        if not ok:
            return data, f"RECONSTRUCT_INVALID ({status})", {}
        return rebuilt, mode_label, {
            "pre_mcus": len(pre_mcus),
            "post_mcus_start": m_post,
            "gap_mcus": m_post - len(pre_mcus),
            "dc_offsets": c_offsets,
        }
