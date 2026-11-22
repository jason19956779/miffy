"""Functional tests for JPEG DCT resync, BMP scanlines, PNG CRC, text, RIFX, and PE handlers."""

import io
import struct
import unittest
from pathlib import Path
from PIL import Image

from src.formats import FormatRegistry
from src.formats.jpeg_handler import (
    build_mcu_layout,
    decode_jpeg_ycbcr_planes,
    encode_mcus,
    parse_jpeg_info,
)

REPO_DIR = Path(__file__).resolve().parents[2]
EXTRACTED_MIFFY = REPO_DIR / "recovered" / "MIFFY" / "extracted"


def _row_mae(img_a: bytes, img_b: bytes) -> float:
    im_a = Image.open(io.BytesIO(img_a))
    im_a.load()
    im_b = Image.open(io.BytesIO(img_b))
    im_b.load()
    im_b_up = im_b.resize(im_a.size, Image.Resampling.BILINEAR)
    ba, bb = im_a.tobytes(), im_b_up.tobytes()
    return sum(abs(x - y) for x, y in zip(ba, bb)) / len(ba)


class TestFormatHandlers(unittest.TestCase):
    def test_format_registry_and_non_image_validators(self):
        handlers = {h.name for h in FormatRegistry.list_handlers()}
        self.assertTrue({"jpeg", "bmp", "image_generic", "text_ini", "director_rifx", "pe_binary", "generic_binary"}.issubset(handlers))

        txt_h = FormatRegistry.get_handler(".INI")
        self.assertTrue(txt_h.validate(b"[Photo]\r\nCount=688\r\n")[0])
        self.assertFalse(txt_h.validate(b"[Photo]\x00\x00")[0])

        rifx_h = FormatRegistry.get_handler(".DXR")
        self.assertTrue(rifx_h.validate(b"XFIR\x10\x00\x00\x00MV93" + b"\x00" * 12)[0])
        self.assertFalse(rifx_h.validate(b"BAD!" + b"\x00" * 12)[0])

        pe_h = FormatRegistry.get_handler(".X32")
        pe_blob = bytearray(128)
        pe_blob[0:2] = b"MZ"
        pe_blob[60:64] = struct.pack("<I", 64)
        pe_blob[64:68] = b"PE\x00\x00"
        self.assertTrue(pe_h.validate(bytes(pe_blob))[0])

        # PNG CRC check
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 200, 30)).save(buf, format="PNG")
        png_ok = bytearray(buf.getvalue())
        png_h = FormatRegistry.get_handler(".PNG")
        self.assertTrue(png_h.validate(bytes(png_ok))[0])
        png_ok[30] ^= 0xFF  # Corrupt IHDR CRC byte cleanly while keeping header type valid
        ok_bad_crc, st_bad_crc, _ = png_h.validate(bytes(png_ok))
        self.assertFalse(ok_bad_crc)
        self.assertEqual(st_bad_crc, "PNG_CRC_ERROR")

    @unittest.skipUnless((EXTRACTED_MIFFY / "L_PHOTO" / "MYCD0001.JPG").exists(), "Sample JPEG not present")
    def test_jpeg_lossless_dct_roundtrip(self):
        orig = (EXTRACTED_MIFFY / "L_PHOTO" / "MYCD0001.JPG").read_bytes()
        w, h, comps, sos_comps, dqt, dht, sos_off, eoi, _ = parse_jpeg_info(orig)
        self.assertEqual((w, h), (1600, 1200))
        _, _, blk_spec = build_mcu_layout(comps, sos_comps, dqt, dht)
        _, _, _, mcus, _ = decode_jpeg_ycbcr_planes(orig)
        re_scan = encode_mcus(mcus, blk_spec, len(comps))
        self.assertEqual(orig[:sos_off] + re_scan + b"\xff\xd9", orig)

    @unittest.skipUnless((EXTRACTED_MIFFY / "L_PHOTO" / "MYCD0646.JPG").exists(), "Sample JPEG not present")
    def test_jpeg_synthetic_thumb_and_blind_resync(self):
        orig = (EXTRACTED_MIFFY / "L_PHOTO" / "MYCD0646.JPG").read_bytes()
        thumb = (EXTRACTED_MIFFY / "S" / "MYCD0646.JPG").read_bytes()
        jh = FormatRegistry.get_handler(".JPG")

        corrupt = bytearray(orig)
        for pos in range(31420, 31620):
            corrupt[pos] ^= 0x55
            if corrupt[pos - 1] == 0xFF and corrupt[pos] != 0x00:
                corrupt[pos] = 0x00
            if corrupt[pos] == 0xFF and corrupt[pos + 1] != 0x00:
                corrupt[pos] = 0xAA

        rebuilt_thumb, st_t, meta_t = jh.reconstruct(bytes(corrupt), [(31420, 31622)], reference_data=thumb)
        self.assertEqual(st_t, "DCT_THUMB_RESYNC")
        self.assertTrue(jh.validate(rebuilt_thumb)[0])
        self.assertLess(_row_mae(rebuilt_thumb, thumb), 10.0)
        self.assertGreater(meta_t["gap_mcus"], 0)

        rebuilt_blind, st_b, _ = jh.reconstruct(bytes(corrupt), [(31420, 31622)], reference_data=None)
        self.assertEqual(st_b, "DCT_BLIND_RESYNC")
        self.assertTrue(jh.validate(rebuilt_blind)[0])
        self.assertLess(_row_mae(rebuilt_blind, thumb), 15.0)

    @unittest.skipUnless((EXTRACTED_MIFFY / "PIC" / "GL_12.BMP").exists(), "Sample BMP not present")
    def test_bmp_scanline_repair(self):
        bmp_orig = (EXTRACTED_MIFFY / "PIC" / "GL_12.BMP").read_bytes()
        bmp_corrupt = bytearray(bmp_orig)
        bmp_corrupt[20000:22048] = b"\x00" * 2048
        bh = FormatRegistry.get_handler(".BMP")
        repaired, st, meta = bh.reconstruct(bytes(bmp_corrupt), [(20000, 22048)])
        self.assertEqual(st, "BMP_SCANLINE_REPAIRED")
        self.assertTrue(bh.validate(repaired)[0])
        self.assertGreater(meta["repaired_rows"], 0)


if __name__ == "__main__":
    unittest.main()
