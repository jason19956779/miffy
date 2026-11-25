"""Structural and mathematical correctness tests for JPEG & BMP photo recovery."""

import hashlib
import io
import struct
import unittest

from PIL import Image, ImageFile

from src.core import rspc
from src.formats import FormatRegistry
from src.formats.jpeg_handler import (
    build_mcu_layout,
    decode_jpeg_ycbcr_planes,
    encode_mcus,
    parse_jpeg_info,
)
from src.solvers import SolverPipeline
from test.core.test_rspc_and_state import encode_mode1_sector


def _make_synthetic_photo_suite() -> tuple[bytes, bytes, bytes]:
    """Create deterministic 640x480 RGB photo JPEG, 160x120 thumbnail, and 400x300 24-bit BMP."""
    w, h = 640, 480
    px = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            edge = 40 if ((x // 32 + y // 32) & 1) == 0 else 0
            px[i] = min(255, (x * 220 // w) + (y % 29) + edge)
            px[i + 1] = min(255, (y * 210 // h) + ((x * 3) % 23) + (edge // 2))
            px[i + 2] = min(255, ((x + y) * 160 // (w + h)) + 30)
    im = Image.frombytes("RGB", (w, h), bytes(px))
    buf_j = io.BytesIO()
    im.save(buf_j, format="JPEG", quality=92, subsampling=2)
    buf_t = io.BytesIO()
    im.resize((160, 120), Image.Resampling.BILINEAR).save(buf_t, format="JPEG", quality=90)
    buf_b = io.BytesIO()
    im.resize((400, 300), Image.Resampling.BILINEAR).save(buf_b, format="BMP")
    return buf_j.getvalue(), buf_t.getvalue(), buf_b.getvalue()


def _strict_pil_open(data: bytes) -> Image.Image:
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        im_v = Image.open(io.BytesIO(data))
        im_v.verify()
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev


def _assert_valid_jpeg_grammar(tc: unittest.TestCase, jpg_bytes: bytes) -> tuple[int, int]:
    tc.assertEqual(jpg_bytes[:2], b"\xff\xd8")
    tc.assertEqual(jpg_bytes[-2:], b"\xff\xd9")
    _, _, _, _, _, _, sos_off, eoi_off, _ = parse_jpeg_info(jpg_bytes)
    tc.assertLess(sos_off, eoi_off)
    scan = jpg_bytes[sos_off:eoi_off]
    i = 0
    while i < len(scan) - 1:
        if scan[i] == 0xFF:
            nxt = scan[i + 1]
            # Inside entropy scan, only stuffed 0x00 or RST0..RST7 (0xD0..0xD7) are legal
            tc.assertTrue(nxt == 0x00 or 0xD0 <= nxt <= 0xD7, f"Illegal marker 0xFF{nxt:02X} in scan")
            i += 2
        else:
            i += 1
    im = _strict_pil_open(jpg_bytes)
    return im.size


def _pixel_mae(jpg_a: bytes, jpg_b: bytes) -> float:
    ba = _strict_pil_open(jpg_a).tobytes()
    bb = _strict_pil_open(jpg_b).tobytes()
    return sum(abs(x - y) for x, y in zip(ba, bb)) / len(ba)


def _c2_from_positions(positions: list[int]) -> bytes:
    c2 = bytearray(294)
    for pos in positions:
        c2[pos >> 3] |= 1 << (7 - (pos & 7))
    return bytes(c2)


class TestPhotoIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_jpg, cls.thumb_jpg, cls.orig_bmp = _make_synthetic_photo_suite()

    def test_jpeg_structural_grammar_and_lossless_dct_roundtrip(self):
        size = _assert_valid_jpeg_grammar(self, self.orig_jpg)
        self.assertEqual(size, (640, 480))
        _, _, comps, sos_comps, dqt, dht, sos_off, _, _ = parse_jpeg_info(self.orig_jpg)
        _, _, _, mcus, _ = decode_jpeg_ycbcr_planes(self.orig_jpg)
        _, _, blk_spec = build_mcu_layout(comps, sos_comps, dqt, dht)
        re_scan = encode_mcus(mcus, blk_spec, len(comps))
        self.assertEqual(self.orig_jpg[:sos_off] + re_scan + b"\xff\xd9", self.orig_jpg)

    def test_jpeg_repair_preserves_intact_mcus_losslessly_and_pixel_accuracy(self):
        jh = FormatRegistry.get_handler(".JPG")
        _, _, comps, sos_comps, dqt, dht, sos_off, eoi_off, _ = parse_jpeg_info(self.orig_jpg)
        _, _, blk_spec = build_mcu_layout(comps, sos_comps, dqt, dht)
        _, _, _, orig_mcus, _ = decode_jpeg_ycbcr_planes(self.orig_jpg)
        mid = sos_off + (eoi_off - sos_off) // 2

        def _mcu_count_before(byte_limit: int) -> int:
            lo, hi = 0, len(orig_mcus)
            while lo < hi:
                m = (lo + hi + 1) // 2
                if sos_off + len(encode_mcus(orig_mcus[:m], blk_spec, len(comps))) <= byte_limit:
                    lo = m
                else:
                    hi = m - 1
            return lo

        for span_len in (200, 2048):
            c_start = mid
            c_end = min(eoi_off - 64, mid + span_len)
            corrupt = bytearray(self.orig_jpg)
            for pos in range(c_start, c_end):
                corrupt[pos] ^= 0x5A
                if corrupt[pos - 1] == 0xFF and corrupt[pos] != 0x00:
                    corrupt[pos] = 0x00
                if corrupt[pos] == 0xFF and corrupt[pos + 1] != 0x00:
                    corrupt[pos] = 0xAA

            rebuilt_t, st_t, meta_t = jh.reconstruct(
                bytes(corrupt), [(c_start, c_end)], reference_data=self.thumb_jpg
            )
            self.assertEqual(st_t, "DCT_THUMB_RESYNC")
            self.assertEqual(_assert_valid_jpeg_grammar(self, rebuilt_t), (640, 480))
            self.assertLess(_pixel_mae(rebuilt_t, self.orig_jpg), 6.0)

            # Mathematical DCT losslessness: 100% intact before gap, >=95% exact AC/DC match after resync
            _, _, _, reb_mcus, _ = decode_jpeg_ycbcr_planes(rebuilt_t)
            self.assertEqual(len(reb_mcus), len(orig_mcus))
            pre_count = max(1, _mcu_count_before(c_start) - 1)
            gap_mcus = int(meta_t["gap_mcus"])
            after_start = pre_count + gap_mcus + 8
            before_ids = list(range(pre_count))
            after_ids = list(range(after_start, len(orig_mcus)))
            self.assertGreater(len(before_ids), 20)
            self.assertGreater(len(after_ids), 20)
            for idx in before_ids:
                self.assertEqual(reb_mcus[idx], orig_mcus[idx], f"Pre-gap MCU {idx} altered!")
            after_exact = sum(1 for idx in after_ids if reb_mcus[idx] == orig_mcus[idx])
            self.assertGreaterEqual(after_exact / len(after_ids), 0.95)
            self.assertGreater(gap_mcus, 0)

            rebuilt_b, st_b, _ = jh.reconstruct(bytes(corrupt), [(c_start, c_end)], reference_data=None)
            self.assertEqual(st_b, "DCT_BLIND_RESYNC")
            self.assertEqual(_assert_valid_jpeg_grammar(self, rebuilt_b), (640, 480))
            self.assertLess(_pixel_mae(rebuilt_b, self.orig_jpg), 20.0)

    def test_jpeg_corrupt_header_fails_safely(self):
        jh = FormatRegistry.get_handler(".JPG")
        bad_hdr = bytearray(self.orig_jpg)
        bad_hdr[0:2] = b"\x00\x00"
        self.assertFalse(jh.validate(bytes(bad_hdr))[0])
        _, st, _ = jh.reconstruct(bytes(bad_hdr), [(0, 128)], reference_data=self.thumb_jpg)
        self.assertNotIn("RESYNC", st)

    def test_bmp_structural_and_mathematical_scanline_correctness(self):
        bh = FormatRegistry.get_handler(".BMP")
        px_off = struct.unpack_from("<I", self.orig_bmp, 10)[0]
        w, h, _, bpp = struct.unpack_from("<IiHH", self.orig_bmp, 18)
        self.assertEqual((w, h, bpp), (400, 300, 24))
        row_stride = ((w * (bpp // 8) + 3) // 4) * 4

        bad_start = px_off + 80 * row_stride
        bad_end = bad_start + 2048
        corrupt = bytearray(self.orig_bmp)
        corrupt[bad_start:bad_end] = b"\x00" * 2048

        repaired, st, meta = bh.reconstruct(bytes(corrupt), [(bad_start, bad_end)])
        self.assertEqual(st, "BMP_SCANLINE_REPAIRED")
        self.assertTrue(bh.validate(repaired)[0])
        self.assertGreater(meta["repaired_rows"], 0)
        self.assertEqual(len(repaired), len(self.orig_bmp))
        self.assertEqual(repaired[:px_off], self.orig_bmp[:px_off])
        im = _strict_pil_open(repaired)
        self.assertEqual(im.size, (400, 300))

        # Every untouched scanline must remain 100% bit-exact
        first_bad_row = (bad_start - px_off) // row_stride
        last_bad_row = (bad_end - 1 - px_off) // row_stride
        for r in range(h):
            if r < first_bad_row or r > last_bad_row:
                rs = px_off + r * row_stride
                self.assertEqual(repaired[rs : rs + row_stride], self.orig_bmp[rs : rs + row_stride])

        orig_rgb = _strict_pil_open(self.orig_bmp).tobytes()
        rep_rgb = im.tobytes()
        mae = sum(abs(a - b) for a, b in zip(orig_rgb, rep_rgb)) / len(orig_rgb)
        self.assertLess(mae, 2.0)

    def test_end_to_end_sector_to_jpeg_bit_exact_recovery(self):
        start_lba = 58340
        n_secs = (len(self.orig_jpg) + 2047) // 2048
        padded = self.orig_jpg + b"\x00" * (n_secs * 2048 - len(self.orig_jpg))
        clean_raws = [
            bytes(encode_mode1_sector(start_lba + i, padded[i * 2048 : (i + 1) * 2048]))
            for i in range(n_secs)
        ]

        damaged_indices = [1, 3]
        pipeline = SolverPipeline(max_tier=3)
        recovered_chunks = [bytes(padded[i * 2048 : (i + 1) * 2048]) for i in range(n_secs)]

        for idx in damaged_indices:
            lba = start_lba + idx
            good = clean_raws[idx]
            passes = []
            for p_i in range(3):
                cor = bytearray(good)
                bad_pos = list(range(120 + p_i * 550, 120 + p_i * 550 + 280))
                for pos in bad_pos:
                    cor[pos] ^= 0x7E
                passes.append((bytes(cor), _c2_from_positions(bad_pos)))
            # Inject clean -1 LBA frame slip from preceding sector of the same JPEG!
            passes.append((clean_raws[idx - 1], b"\x00" * 294))

            ctx: dict = {}
            ok, pz, qz, sec, _ = pipeline.solve_sector(lba, passes, context=ctx)
            self.assertTrue(ok)
            self.assertEqual((pz, qz), (86, 52))
            self.assertTrue(rspc.check_and_fix_q_parity(sec, expected_lba=lba))
            self.assertEqual(ctx.get("frame_slips_filtered", 0), 1)
            recovered_chunks[idx] = bytes(sec[16:2064])

        reassembled = b"".join(recovered_chunks)[: len(self.orig_jpg)]
        self.assertEqual(hashlib.sha256(reassembled).hexdigest(), hashlib.sha256(self.orig_jpg).hexdigest())
        self.assertEqual(_assert_valid_jpeg_grammar(self, reassembled), (640, 480))

    def test_encode_mcus_handles_sparse_custom_dht_without_keyerror(self):
        _, _, comps, sos_comps, dqt, dht, sos_off, _, _ = parse_jpeg_info(self.orig_jpg)
        _, _, blk_spec = build_mcu_layout(comps, sos_comps, dqt, dht)
        _, _, _, mcus, _ = decode_jpeg_ycbcr_planes(self.orig_jpg)

        # Remove AC symbol 0x14 (20: run=1, size=4) and DC category 11 from block 0's encoder tables
        blk_spec[0]["ac_enc"] = {k: v for k, v in blk_spec[0]["ac_enc"].items() if k != 0x14}
        blk_spec[0]["dc_enc"] = {k: v for k, v in blk_spec[0]["dc_enc"].items() if k != 11}

        mod_mcus = [list(list(c) for c in m) for m in mcus]
        mod_mcus[0][0][0] = 1500  # triggers DC ssss=11
        mod_mcus[0][0][1] = 0
        mod_mcus[0][0][2] = 12    # triggers AC (1 << 4) | 4 = 20

        scan_bytes = encode_mcus(mod_mcus, blk_spec, len(comps))
        rebuilt = self.orig_jpg[:sos_off] + scan_bytes + b"\xff\xd9"
        im = _strict_pil_open(rebuilt)
        self.assertEqual(im.size, (640, 480))

    def test_jpeg_validate_enforces_strict_t81_structure(self):
        from src.formats.jpeg_handler import validate_jpeg_t81_structure

        jh = FormatRegistry.get_handler(".JPG")
        t81_ok, issues, meta = validate_jpeg_t81_structure(self.orig_jpg)
        self.assertTrue(t81_ok, issues)
        self.assertEqual(issues, [])
        self.assertTrue(meta.get("t81_valid"))

        ok_val, status_val, meta_val = jh.validate(self.orig_jpg)
        self.assertTrue(ok_val)
        self.assertEqual(status_val, "STRICT_OK")
        self.assertTrue(meta_val.get("t81_valid"))

        # 1. Trailing non-zero garbage after EOI (Pillow alone ignores this)
        bad_tail = self.orig_jpg + b"\xde\xad\xbe\xef"
        ok_tail, status_tail, _ = jh.validate(bad_tail)
        self.assertFalse(ok_tail)
        self.assertIn("T81_STRUCT_ERR", status_tail)

        # 2. Illegal unstuffed marker inside entropy-coded scan
        _, _, _, _, _, _, sos_off, eoi_off, _ = parse_jpeg_info(self.orig_jpg)
        bad_scan = bytearray(self.orig_jpg)
        mid = (sos_off + eoi_off) // 2
        bad_scan[mid : mid + 2] = b"\xff\xaa"
        self.assertFalse(validate_jpeg_t81_structure(bytes(bad_scan))[0])
        self.assertFalse(jh.validate(bytes(bad_scan))[0])

        # 3. Corrupted DHT violating Kraft-McMillan inequality (>= 1.0)
        dht_pos = self.orig_jpg.find(b"\xff\xc4")
        self.assertGreater(dht_pos, 0)
        bad_dht = bytearray(self.orig_jpg)
        cnt_off = dht_pos + 5  # marker(2) + len(2) + tc_th(1)
        orig_sum = sum(bad_dht[cnt_off : cnt_off + 16])
        self.assertEqual(bad_dht[cnt_off], 0)
        bad_dht[cnt_off] = 2
        dec_left = 2
        for i in range(1, 16):
            while bad_dht[cnt_off + i] > 0 and dec_left > 0:
                bad_dht[cnt_off + i] -= 1
                dec_left -= 1
        self.assertEqual(sum(bad_dht[cnt_off : cnt_off + 16]), orig_sum)
        ok_k, issues_k, _ = validate_jpeg_t81_structure(bytes(bad_dht))
        self.assertFalse(ok_k)
        self.assertTrue(any("kraft" in it.lower() for it in issues_k), issues_k)
        self.assertFalse(jh.validate(bytes(bad_dht))[0])

    def test_bmp_validate_enforces_header_and_bounds(self):
        from src.formats.bmp_handler import validate_bmp_structure

        bh = FormatRegistry.get_handler(".BMP")
        ok_b, issues_b, meta_b = validate_bmp_structure(self.orig_bmp)
        self.assertTrue(ok_b, issues_b)
        self.assertTrue(meta_b.get("bmp_struct_valid"))

        ok_val, status_val, meta_val = bh.validate(self.orig_bmp)
        self.assertTrue(ok_val)
        self.assertEqual(status_val, "STRICT_OK")
        self.assertTrue(meta_val.get("bmp_struct_valid"))

        # 1. Non-zero bfReserved1
        bad_res = bytearray(self.orig_bmp)
        bad_res[6:8] = b"\x01\x00"
        ok_r, st_r, _ = bh.validate(bytes(bad_res))
        self.assertFalse(ok_r)
        self.assertIn("BMP_STRUCT_ERR", st_r)

        # 2. Mismatched bfSize / truncated pixel array
        bad_sz = bytearray(self.orig_bmp)
        bad_sz[2:6] = (12345).to_bytes(4, "little")
        self.assertFalse(bh.validate(bytes(bad_sz))[0])
        self.assertFalse(bh.validate(self.orig_bmp[:-100])[0])


if __name__ == "__main__":
    unittest.main()