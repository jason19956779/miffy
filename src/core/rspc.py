"""ECMA-130 Yellow Book Mode 1 GF(2^8), 32-bit EDC, and RSPC P/Q math primitives."""

from collections import Counter

SYNC_12 = b"\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00"
FIXED_POS = frozenset(range(0, 16)) | frozenset(range(2068, 2076))

# GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
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
    tuple(0 if (a == 0 or b == 0) else GF_EXP[GF_LOG[a] + GF_LOG[b]] for b in range(256))
    for a in range(256)
)


def gf_div(a: int, b: int) -> int:
    """Divide a by non-zero b in GF(2^8)."""
    if a == 0:
        return 0
    return GF_EXP[GF_LOG[a] - GF_LOG[b] + 255]


# ECMA-130 32-bit Error Detection Code table
_edc_tbl = [0] * 256
for _i in range(256):
    _edc = _i
    for _ in range(8):
        _edc = ((_edc >> 1) ^ 0xD8018001) if (_edc & 1) else (_edc >> 1)
    _edc_tbl[_i] = _edc
EDC_TABLE = tuple(_edc_tbl)


def compute_edc(data_2064: bytes | bytearray | memoryview) -> int:
    """Compute the 32-bit ECMA-130 EDC over sector bytes 0..2063."""
    edc = 0
    tbl = EDC_TABLE
    for b in data_2064:
        edc = (edc >> 8) ^ tbl[(edc ^ b) & 0xFF]
    return edc


def edc_residual(sec_2068: bytes | bytearray) -> int:
    """Return XOR between computed EDC(0..2063) and stored EDC(2064..2067)."""
    return compute_edc(sec_2068[:2064]) ^ int.from_bytes(sec_2068[2064:2068], "little")


# P (86 codewords of length 26) and Q (52 codewords of length 45) index tables
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


def to_bcd(val: int) -> int:
    return ((val // 10) << 4) | (val % 10)


def lba_header(lba: int) -> bytes:
    """Build canonical 4-byte Mode 1 MSF+Mode header for an LBA."""
    frames = lba + 150
    return bytes(
        [
            to_bcd(frames // 4500),
            to_bcd((frames % 4500) // 75),
            to_bcd(frames % 75),
            0x01,
        ]
    )


def msf_bytes_to_lba(hdr4: bytes | bytearray) -> int | None:
    """Decode a 4-byte Mode 1 MSF+Mode header back into an LBA, or None if invalid BCD/Mode."""
    if len(hdr4) < 4 or hdr4[3] != 0x01:
        return None
    vals = []
    for b in hdr4[:3]:
        hi, lo = (b >> 4) & 0x0F, b & 0x0F
        if hi > 9 or lo > 9:
            return None
        vals.append(hi * 10 + lo)
    mm, ss, ff = vals
    if ss >= 60 or ff >= 75:
        return None
    frames = (mm * 60 + ss) * 75 + ff
    return frames - 150 if frames >= 150 else None


def init_sector_header(lba: int, sec: bytearray) -> None:
    """Overwrite known constant Mode 1 Sync, MSF header, and 8 intermediate zero bytes."""
    sec[0:12] = SYNC_12
    sec[12:16] = lba_header(lba)
    sec[2068:2076] = b"\x00" * 8


def c2_set(c2_bytes: bytes) -> set[int]:
    """Decode 294-byte MSB-first C2 error pointer bitmap into byte positions in 16..2351."""
    return {
        j
        for j in range(16, 2352)
        if ((c2_bytes[j >> 3] >> (7 - (j & 7))) & 1) and j not in FIXED_POS
    }


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
    """Locate and correct a single-byte error in a P or Q codeword."""
    s0, s1 = syndromes(sec, cw)
    if s0 == 0 and s1 == 0:
        return 0, []
    if s0 == 0 or s1 == 0:
        return -1, []
    log_pos = (GF_LOG[s1] - GF_LOG[s0]) % 255
    n = len(cw)
    if log_pos >= n:
        return -1, []
    pos = cw[n - 1 - log_pos]
    return 1, [(pos, sec[pos] ^ s0)]


def solve_2eras(
    sec: bytes | bytearray, cw: tuple[int, ...], eras_i: list[int]
) -> tuple[bool, list[tuple[int, int]]]:
    """Solve 1 or 2 known erasure positions in a P or Q codeword."""
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


def count_pq(sec: bytes | bytearray) -> tuple[int, int]:
    """Count zero-syndrome P (out of 86) and Q (out of 52) codewords."""
    pz = sum(1 for cw in P_IDX if syndromes_p(sec, cw) == (0, 0))
    qz = sum(1 for cw in Q_IDX if syndromes_q(sec, cw) == (0, 0))
    return pz, qz


def partial_score(pz: int, qz: int) -> tuple[int, int]:
    """Rank unverified sectors while preventing forged 86/86 loops that fail EDC."""
    return (0 if pz == 86 else pz, qz)


def check_and_fix_q_parity(sec: bytearray, expected_lba: int | None = None) -> bool:
    """Verification oracle: Mode 1 header (if expected_lba), 86/86 P == 0, AND 32-bit EDC match (+ Q-parity fix)."""
    if expected_lba is not None:
        if (
            sec[0:12] != SYNC_12
            or sec[12:16] != lba_header(expected_lba)
            or sec[2068:2076] != b"\x00" * 8
        ):
            return False
    if not all(syndromes_p(sec, cw) == (0, 0) for cw in P_IDX):
        return False
    if edc_residual(sec[:2068]) != 0:
        return False
    for cw in Q_IDX:
        if syndromes_q(sec, cw) != (0, 0):
            ok, corrs = solve_2eras(sec, cw, [43, 44])
            if ok:
                for pos, val in corrs:
                    sec[pos] = val
    return all(syndromes_q(sec, cw) == (0, 0) for cw in Q_IDX)


def verify_physical_q_parity(
    sec: bytes | bytearray,
    raw_passes: list[tuple[bytes, bytes]],
    c2_sets: list[set[int]] | None = None,
) -> tuple[bool, int, int]:
    """Cross-check recomputed 104-byte Q-parity against unflagged C2-clean physical disc reads."""
    if not raw_passes or len(sec) < 2352:
        return True, 0, 0
    c2_list = c2_sets if c2_sets is not None else [c2_set(c2_b) for _, c2_b in raw_passes]
    q_clean_checked = 0
    q_clean_match = 0
    for pos in range(2248, 2352):
        if any(pos not in c2 for c2 in c2_list):
            q_clean_checked += 1
            if any(rp[0][pos] == sec[pos] for rp in raw_passes):
                q_clean_match += 1
    if q_clean_checked < 8:
        return True, q_clean_match, q_clean_checked
    max_miss = max(6, int(0.10 * q_clean_checked))
    return (q_clean_match >= q_clean_checked - max_miss), q_clean_match, q_clean_checked


def is_frame_slip(
    expected_lba: int, raw_2352: bytes | bytearray, max_delta: int = 16
) -> bool:
    """Detect physical drive frame slips where raw_2352 carries a valid neighboring LBA sector."""
    if len(raw_2352) < 2352:
        return False
    msf_lba = msf_bytes_to_lba(raw_2352[12:16])
    if msf_lba is None or msf_lba == expected_lba or abs(msf_lba - expected_lba) > max_delta:
        return False
    if edc_residual(raw_2352[:2068]) == 0:
        return True
    p_valid = sum(1 for cw in P_IDX if syndromes_p(raw_2352, cw) == (0, 0))
    return p_valid >= 40


def build_pass_cache(lba: int, raw_passes: list[tuple[bytes, bytes]]) -> list[dict]:
    """Precompute cleaned headers, C2 sets, P/Q validity, and agreed 1-errors once per LBA."""
    cache: list[dict] = []
    for raw_sec, c2_bytes in raw_passes:
        sec = bytearray(raw_sec)
        init_sector_header(lba, sec)
        c2 = c2_set(c2_bytes)
        p_ok_raw = [syndromes_p(sec, cw) == (0, 0) for cw in P_IDX]
        q_ok_raw = [syndromes_q(sec, cw) == (0, 0) for cw in Q_IDX]
        p_c2_hits = [sum(1 for pos in cw if pos in c2) for cw in P_IDX]
        q_c2_hits = [sum(1 for pos in cw if pos in c2) for cw in Q_IDX]

        p_1err: dict[int, int] = {}
        for idx, cw in enumerate(P_IDX):
            if not p_ok_raw[idx]:
                st1, c1 = solve_1err(sec, cw)
                if st1 == 1 and c1[0][0] not in FIXED_POS:
                    p_1err[c1[0][0]] = c1[0][1]

        agreed_1err: dict[int, int] = {}
        for idx, cw in enumerate(Q_IDX):
            if not q_ok_raw[idx]:
                st1, c1 = solve_1err(sec, cw)
                if st1 == 1 and p_1err.get(c1[0][0]) == c1[0][1]:
                    agreed_1err[c1[0][0]] = c1[0][1]

        cache.append(
            {
                "sec": sec,
                "c2": c2,
                "p_ok_raw": p_ok_raw,
                "q_ok_raw": q_ok_raw,
                "p_c2_hits": p_c2_hits,
                "q_c2_hits": q_c2_hits,
                "p_ok_count": sum(1 for x in p_ok_raw if x),
                "agreed_1err": agreed_1err,
            }
        )
    return cache


def compute_verified_byte_mask_2048(sec: bytes | bytearray) -> set[int]:
    """Return user byte offsets (0..2047) covered by valid P or valid Q codewords."""
    p_ok = {i for i, cw in enumerate(P_IDX) if syndromes_p(sec, cw) == (0, 0)}
    q_ok = {i for i, cw in enumerate(Q_IDX) if syndromes_q(sec, cw) == (0, 0)}
    good = set()
    for pos in range(16, 2064):
        if (pos in BYTE_TO_P and BYTE_TO_P[pos] in p_ok) and (
            pos in BYTE_TO_Q and BYTE_TO_Q[pos] in q_ok
        ):
            good.add(pos - 16)
    return good


def weighted_modal_vote(lba: int, raw_passes: list[tuple[bytes, bytes]]) -> bytearray:
    """Fast C2/P/Q-weighted byte vote fallback for unverified partial sectors."""
    sec = bytearray(2352)
    init_sector_header(lba, sec)
    votes = [Counter() for _ in range(2352)]
    for raw_sec, c2_b in raw_passes:
        c2 = c2_set(c2_b)
        p_ok = [syndromes_p(raw_sec, cw) == (0, 0) for cw in P_IDX]
        q_ok = [syndromes_q(raw_sec, cw) == (0, 0) for cw in Q_IDX]
        for j in range(16, 2352):
            if j in FIXED_POS:
                continue
            pok = j in BYTE_TO_P and p_ok[BYTE_TO_P[j]]
            qok = j in BYTE_TO_Q and q_ok[BYTE_TO_Q[j]]
            c2ok = j not in c2
            w = 1 + (4 if c2ok else 0) + (20 if pok else 0) + (25 if qok else 0) + (60 if pok and qok else 0)
            votes[j][raw_sec[j]] += w
    for j in range(16, 2352):
        if j not in FIXED_POS:
            sec[j] = votes[j].most_common(1)[0][0]
    return sec


ZERO_294 = b"\x00" * 294


def lba_header_mode(lba: int, mode: int = 0x01) -> bytes:
    """Build canonical 4-byte MSF+Mode header for an LBA and CD sector mode (1 or 2)."""
    frames = lba + 150
    return bytes(
        [
            to_bcd(frames // 4500),
            to_bcd((frames % 4500) // 75),
            to_bcd(frames % 75),
            mode & 0xFF,
        ]
    )


def msf_bytes_to_lba_any(
    hdr4: bytes | bytearray, expected_mode: int | None = None
) -> tuple[int, int] | None:
    """Decode a 4-byte MSF+Mode header into `(lba, mode)`, or None if invalid BCD/mode."""
    if len(hdr4) < 4:
        return None
    mode = hdr4[3]
    if expected_mode is not None:
        if mode != expected_mode:
            return None
    elif mode not in (1, 2):
        return None
    vals = []
    for b in hdr4[:3]:
        hi, lo = (b >> 4) & 0x0F, b & 0x0F
        if hi > 9 or lo > 9:
            return None
        vals.append(hi * 10 + lo)
    mm, ss, ff = vals
    if ss >= 60 or ff >= 75:
        return None
    frames = (mm * 60 + ss) * 75 + ff
    return (frames - 150, mode) if frames >= 150 else None


def is_mode2_form2(raw_2352: bytes | bytearray) -> bool:
    """Return True if a raw 2352-byte sector is CD-ROM XA Mode 2 Form 2."""
    return len(raw_2352) >= 24 and raw_2352[15] == 0x02 and bool(raw_2352[18] & 0x20)


def count_pq_mode2_f1(raw_2352: bytes | bytearray) -> tuple[int, int]:
    """Count zero-syndrome P/Q codewords for Mode 2 Form 1 (bytes 12..15 zeroed)."""
    sec = bytearray(raw_2352[:2352])
    sec[12:16] = b"\x00\x00\x00\x00"
    return count_pq(sec)


def verify_mode2_sector(
    expected_lba: int,
    raw_2352: bytes | bytearray,
    c2_294: bytes | None = None,
) -> tuple[bool, str, int, int]:
    """Verify Sync, Mode-2 MSF header, XA subheader mirror, EDC, and Form-1 P/Q or Form-2 C2."""
    if (
        len(raw_2352) < 2352
        or raw_2352[0:12] != SYNC_12
        or raw_2352[12:16] != lba_header_mode(expected_lba, 0x02)
    ):
        return False, "M2_HDR_ERR", 0, 0

    sub_ok = raw_2352[16:20] == raw_2352[20:24]
    if raw_2352[18] & 0x20:
        c2_clean = (c2_294 is None) or (c2_294 == ZERO_294)
        stored_edc = int.from_bytes(raw_2352[2348:2352], "little")
        calc_edc = compute_edc(raw_2352[16:2348])
        if stored_edc != 0:
            if sub_ok and calc_edc == stored_edc and c2_clean:
                return True, "M2F2_EDC_OK", 0, 0
            return False, "M2F2_ERR", 0, 0
        if sub_ok and c2_clean:
            return True, "M2F2_ZERO_EDC_C2_OK", 0, 0
        return False, "M2F2_ERR", 0, 0

    pz, qz = count_pq_mode2_f1(raw_2352)
    calc_edc = compute_edc(raw_2352[16:2072])
    stored_edc = int.from_bytes(raw_2352[2072:2076], "little")
    if sub_ok and calc_edc == stored_edc and pz == 86 and qz == 52:
        return True, "M2F1_OK", 86, 52
    return False, "M2F1_ERR", pz, qz


def is_frame_slip_mode2(
    expected_lba: int, raw_2352: bytes | bytearray, max_delta: int = 16
) -> bool:
    """Detect physical drive frame slips where raw_2352 carries a valid neighboring Mode 2 LBA."""
    if len(raw_2352) < 2352:
        return False
    decoded = msf_bytes_to_lba_any(raw_2352[12:16], expected_mode=2)
    if decoded is None:
        return False
    msf_lba, _ = decoded
    if msf_lba == expected_lba or abs(msf_lba - expected_lba) > max_delta:
        return False
    sub_ok = raw_2352[16:20] == raw_2352[20:24]
    if raw_2352[18] & 0x20:
        stored_edc = int.from_bytes(raw_2352[2348:2352], "little")
        return sub_ok and (stored_edc == 0 or compute_edc(raw_2352[16:2348]) == stored_edc)
    if sub_ok and compute_edc(raw_2352[16:2072]) == int.from_bytes(raw_2352[2072:2076], "little"):
        return True
    pz, _ = count_pq_mode2_f1(raw_2352)
    return pz >= 40


def clean_single_pass_mode2_f1(
    lba: int, raw_2352: bytes, c2_294: bytes
) -> tuple[bool, bytearray]:
    """Single-pass P/Q + C2 erasure solver for Mode 2 Form 1 (with bytes 12..15 zeroed)."""
    if len(raw_2352) < 2352 or len(c2_294) < 294:
        return False, bytearray(raw_2352)
    sec = bytearray(raw_2352[:2352])
    sec[0:12] = SYNC_12
    sec[12:16] = b"\x00\x00\x00\x00"
    c2 = {j for j in range(16, 2352) if (c2_294[j >> 3] >> (7 - (j & 7))) & 1}

    for _ in range(8):
        changed = False
        for cw in P_IDX:
            if syndromes_p(sec, cw) != (0, 0):
                eras = [idx for idx, pos in enumerate(cw) if pos in c2]
                if 1 <= len(eras) <= 2:
                    ok_e, corrs = solve_2eras(sec, cw, eras)
                    if ok_e:
                        for pos, val in corrs:
                            sec[pos] = val
                        for idx in eras:
                            c2.discard(cw[idx])
                        changed = True
                elif not eras:
                    st1, c1 = solve_1err(sec, cw)
                    if st1 == 1 and c1[0][0] >= 16:
                        sec[c1[0][0]] = c1[0][1]
                        c2.discard(c1[0][0])
                        changed = True
        for cw in Q_IDX:
            if syndromes_q(sec, cw) != (0, 0):
                eras = [idx for idx, pos in enumerate(cw) if pos in c2]
                if 1 <= len(eras) <= 2:
                    ok_e, corrs = solve_2eras(sec, cw, eras)
                    if ok_e:
                        for pos, val in corrs:
                            sec[pos] = val
                        for idx in eras:
                            c2.discard(cw[idx])
                        changed = True
                elif not eras:
                    st1, c1 = solve_1err(sec, cw)
                    if st1 == 1 and c1[0][0] >= 16:
                        sec[c1[0][0]] = c1[0][1]
                        c2.discard(c1[0][0])
                        changed = True
        if not changed:
            break

    sec[12:16] = lba_header_mode(lba, mode=0x02)
    ok, _, _, _ = verify_mode2_sector(lba, sec)
    return ok, sec
