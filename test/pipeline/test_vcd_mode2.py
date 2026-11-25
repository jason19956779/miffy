"""TDD unit and synthetic integration tests for CD-ROM XA Mode 2 / VCD 2.0 ripping."""

import contextlib
import io
import struct
import tempfile
import unittest
from pathlib import Path

from src.core import rspc
from src.core.models import DeviceCapabilities
from src.core.state import RecoveryStateDB
from src.pipeline.vcd_mode2 import run_mode2_pipeline, write_cue_sheet


def encode_mode2_form1_sector(lba: int, sub4: bytes, user_2048: bytes) -> bytes:
    """Build a physically valid Mode 2 Form 1 2352-byte sector with EDC and RSPC P/Q."""
    assert len(sub4) == 4
    assert len(user_2048) == 2048
    sec = bytearray(2352)
    sec[0:12] = rspc.SYNC_12
    sec[12:16] = b"\x00\x00\x00\x00"
    sec[16:20] = sub4
    sec[20:24] = sub4
    sec[24:2072] = user_2048
    sec[2072:2076] = rspc.compute_edc(sec[16:2072]).to_bytes(4, "little")
    for cw in rspc.P_IDX:
        ok, corrs = rspc.solve_2eras(sec, cw, [24, 25])
        assert ok
        for pos, val in corrs:
            sec[pos] = val
    for cw in rspc.Q_IDX:
        ok, corrs = rspc.solve_2eras(sec, cw, [43, 44])
        assert ok
        for pos, val in corrs:
            sec[pos] = val
    sec[12:16] = rspc.lba_header_mode(lba, mode=0x02)
    return bytes(sec)


def encode_mode2_form2_sector(
    lba: int, sub4: bytes, payload_2324: bytes, include_edc: bool = True
) -> bytes:
    """Build a Mode 2 Form 2 2352-byte sector with optional 32-bit EDC over bytes 16..2347."""
    assert len(sub4) == 4
    assert len(payload_2324) == 2324
    sec = bytearray(2352)
    sec[0:12] = rspc.SYNC_12
    sec[12:16] = rspc.lba_header_mode(lba, mode=0x02)
    sec[16:20] = sub4
    sec[20:24] = sub4
    sec[24:2348] = payload_2324
    if include_edc:
        sec[2348:2352] = rspc.compute_edc(sec[16:2348]).to_bytes(4, "little")
    return bytes(sec)


class _FakeMode2Drive:
    """In-memory CD-ROM XA Mode 2 optical drive mock for synthetic pipeline tests."""

    def __init__(self, sectors: list[tuple[bytes, bytes]], tracks: list[dict]):
        self._sectors = sectors
        self._tracks = tracks
        self.max_batch_sectors = 8
        self.caps = DeviceCapabilities(
            sg_device="/dev/sg_fake",
            sr_device="/dev/sr_fake",
            vendor_model="FAKE VCD DRIVE",
            supports_c2=True,
            max_speed_kbps=2823,
            cur_speed_kbps=2823,
            max_sectors_kb=120,
            max_batch_sectors=8,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def set_speed(self, kbps: int) -> int:
        return 0

    def read_toc(self) -> list[dict]:
        return list(self._tracks)

    def detect_disc_mode(self) -> tuple[str, list[dict], int]:
        leadout = next(t["start_lba"] for t in self._tracks if t["track"] == 170)
        return "mode2", list(self._tracks), leadout

    def directional_seek_flush(
        self, target_lba: int, direction: str, total_sectors: int, offset_seed: int = 0
    ) -> None:
        return None

    def read_cd_raw(
        self, start_lba: int, count: int, with_c2: bool = True
    ) -> tuple[int, tuple[int, int, int], int, bytes]:
        out = bytearray()
        for off in range(count):
            idx = start_lba + off
            if idx < 0 or idx >= len(self._sectors):
                return 2, (5, 0x21, 0), count * (2646 if with_c2 else 2352), b""
            raw, c2 = self._sectors[idx]
            out.extend(raw)
            if with_c2:
                out.extend(c2)
        return 0, (0, 0, 0), 0, bytes(out)


def _dir_rec(ident: bytes, ext_lba: int, size: int, is_dir: bool) -> bytes:
    pad = 1 if (len(ident) % 2 == 0) else 0
    rec_len = 33 + len(ident) + pad
    r = bytearray(rec_len)
    r[0] = rec_len
    struct.pack_into("<II", r, 2, ext_lba, ext_lba)
    struct.pack_into("<II", r, 10, size, size)
    r[18:25] = bytes([126, 11, 25, 14, 0, 0, 0])
    r[25] = 0x02 if is_dir else 0x00
    struct.pack_into("<HH", r, 28, 1, 1)
    r[32] = len(ident)
    r[33 : 33 + len(ident)] = ident
    return bytes(r)


class TestVcdMode2(unittest.TestCase):
    def test_mode2_header_and_frame_slip_detection(self):
        self.assertEqual(rspc.lba_header_mode(16, mode=0x02), bytes.fromhex("00021602"))
        self.assertEqual(rspc.lba_header_mode(3150, mode=0x02), bytes.fromhex("00440002"))
        self.assertEqual(rspc.msf_bytes_to_lba_any(bytes.fromhex("00021602")), (16, 2))
        self.assertEqual(rspc.msf_bytes_to_lba_any(bytes.fromhex("00440002"), expected_mode=2), (3150, 2))
        self.assertIsNone(rspc.msf_bytes_to_lba_any(bytes.fromhex("00021602"), expected_mode=1))
        self.assertIsNone(rspc.msf_bytes_to_lba_any(bytes.fromhex("00990002")))

        f1_prev = encode_mode2_form1_sector(199, b"\x00\x00\x09\x00", bytes(range(256)) * 8)
        f1_cur = encode_mode2_form1_sector(200, b"\x00\x00\x09\x00", bytes(range(256)) * 8)
        self.assertTrue(rspc.is_frame_slip_mode2(200, f1_prev))
        self.assertFalse(rspc.is_frame_slip_mode2(200, f1_cur))

        f2_prev = encode_mode2_form2_sector(3149, b"\x01\x01\x62\x0f", b"\xab" * 2324)
        self.assertTrue(rspc.is_frame_slip_mode2(3150, f2_prev))
        self.assertFalse(rspc.is_frame_slip_mode2(3149, f2_prev))

    def test_mode2_form1_and_form2_verification_and_c2_clean(self):
        user_2048 = bytes(((i * 37 + 11) & 0xFF) for i in range(2048))
        raw_f1 = encode_mode2_form1_sector(150, b"\x00\x00\x09\x00", user_2048)

        ok, tag, pz, qz = rspc.verify_mode2_sector(150, raw_f1, rspc.ZERO_294)
        self.assertTrue(ok)
        self.assertEqual(tag, "M2F1_OK")
        self.assertEqual((pz, qz), (86, 52))

        # Corrupt 1 byte in user payload and flag it in C2; clean_single_pass_mode2_f1 must restore exact bytes
        bad_f1 = bytearray(raw_f1)
        corrupt_pos = 1200
        bad_f1[corrupt_pos] ^= 0x5A
        c2 = bytearray(294)
        c2[corrupt_pos >> 3] |= 1 << (7 - (corrupt_pos & 7))
        ok_bad, tag_bad, _, _ = rspc.verify_mode2_sector(150, bad_f1, bytes(c2))
        self.assertFalse(ok_bad)
        self.assertEqual(tag_bad, "M2F1_ERR")

        ok_clean, cleaned = rspc.clean_single_pass_mode2_f1(150, bytes(bad_f1), bytes(c2))
        self.assertTrue(ok_clean)
        self.assertEqual(bytes(cleaned), raw_f1)
        self.assertTrue(rspc.verify_mode2_sector(150, cleaned)[0])

        # Mode 2 Form 2 verification (with EDC and zero-EDC C2 fallback)
        mpeg_2324 = b"\x00\x00\x01\xba" + bytes(((i * 19 + 5) & 0xFF) for i in range(2320))
        raw_f2 = encode_mode2_form2_sector(3300, b"\x01\x01\x62\x0f", mpeg_2324, include_edc=True)
        self.assertEqual(rspc.verify_mode2_sector(3300, raw_f2, rspc.ZERO_294), (True, "M2F2_EDC_OK", 0, 0))

        # Bit corruption or non-zero C2 fails Form 2
        bad_f2 = bytearray(raw_f2)
        bad_f2[500] ^= 0x01
        self.assertFalse(rspc.verify_mode2_sector(3300, bad_f2, rspc.ZERO_294)[0])
        c2_nz = bytearray(294)
        c2_nz[10] = 0x80
        self.assertFalse(rspc.verify_mode2_sector(3300, raw_f2, bytes(c2_nz))[0])

        # Zero stored EDC + zero C2 -> M2F2_ZERO_EDC_C2_OK; zero stored EDC + non-zero C2 -> False
        raw_f2_zero_edc = encode_mode2_form2_sector(3301, b"\x01\x00\x60\x00", b"\x00" * 2324, include_edc=False)
        self.assertEqual(
            rspc.verify_mode2_sector(3301, raw_f2_zero_edc, rspc.ZERO_294),
            (True, "M2F2_ZERO_EDC_C2_OK", 0, 0),
        )
        self.assertFalse(rspc.verify_mode2_sector(3301, raw_f2_zero_edc, bytes(c2_nz))[0])

    def test_write_cue_sheet_formatting(self):
        with tempfile.TemporaryDirectory() as tmp:
            cue = Path(tmp) / "vcd_test.cue"
            tracks = [
                {"track": 1, "control": 0x04, "is_data": True, "start_lba": 0},
                {"track": 2, "control": 0x04, "is_data": True, "start_lba": 3150},
                {"track": 170, "control": 0x04, "is_data": True, "start_lba": 230237},
            ]
            write_cue_sheet(cue, "vcd_test.bin", tracks)
            text = cue.read_text(encoding="utf-8")
            self.assertIn('FILE "vcd_test.bin" BINARY', text)
            self.assertIn("TRACK 01 MODE2/2352", text)
            self.assertIn("INDEX 01 00:00:00", text)
            self.assertIn("TRACK 02 MODE2/2352", text)
            self.assertIn("INDEX 00 00:40:00", text)
            self.assertIn("INDEX 01 00:42:00", text)

    def test_synthetic_mode2_vcd_pipeline_end_to_end(self):
        total_sectors = 40
        tracks = [
            {"track": 1, "control": 0x04, "is_data": True, "start_lba": 0},
            {"track": 2, "control": 0x04, "is_data": True, "start_lba": 30},
            {"track": 170, "control": 0x04, "is_data": True, "start_lba": total_sectors},
        ]

        info_vcd = (b"VIDEO_CD\x02\x00" + b"TEST_VCD".ljust(20, b" ")).ljust(2048, b"\x00")
        mpeg_payloads = [
            b"\x00\x00\x01\xba" + bytes([0x21 + idx]) * 2320
            for idx in range(3)
        ]

        root_dir = (
            _dir_rec(b"\x00", 18, 2048, True)
            + _dir_rec(b"\x01", 18, 2048, True)
            + _dir_rec(b"VCD", 19, 2048, True)
            + _dir_rec(b"MPEGAV", 20, 2048, True)
            + _dir_rec(b"CDI", 22, 2048, True)
        )
        vcd_dir = (
            _dir_rec(b"\x00", 19, 2048, True)
            + _dir_rec(b"\x01", 18, 2048, True)
            + _dir_rec(b"INFO.VCD", 21, 2048, False)
        )
        # Non-MPEG Form 2 file under /CDI must retain its .RTF extension
        cdi_dir = (
            _dir_rec(b"\x00", 22, 2048, True)
            + _dir_rec(b"\x01", 18, 2048, True)
            + _dir_rec(b"CDI_ALL.RTF", 23, 2048, False)
        )
        # Track 2 spans LBAs 30..35 (6 sectors): 2 zero-pad 0x60 + 3 MPEG 0x62 + 1 EOF pad 0xE0
        mpegav_dir = (
            _dir_rec(b"\x00", 20, 2048, True)
            + _dir_rec(b"\x01", 18, 2048, True)
            + _dir_rec(b"AVSEQ01.DAT", 30, 6 * 2048, False)
        )

        pvd = bytearray(2048)
        pvd[0] = 1
        pvd[1:6] = b"CD001"
        pvd[6] = 1
        pvd[40:72] = b"VCD_SYNTH".ljust(32, b" ")
        struct.pack_into("<II", pvd, 80, total_sectors, total_sectors)
        struct.pack_into("<HH", pvd, 128, 2048, 2048)
        pvd[156 : 156 + 34] = _dir_rec(b"\x00", 18, 2048, True)

        term = bytearray(2048)
        term[0] = 255
        term[1:6] = b"CD001"
        term[6] = 1

        sectors: list[tuple[bytes, bytes]] = []
        for lba in range(total_sectors):
            if lba == 16:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x09\x00", bytes(pvd))
            elif lba == 17:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", bytes(term))
            elif lba == 18:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", root_dir.ljust(2048, b"\x00"))
            elif lba == 19:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", vcd_dir.ljust(2048, b"\x00"))
            elif lba == 20:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", mpegav_dir.ljust(2048, b"\x00"))
            elif lba == 21:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", info_vcd)
            elif lba == 22:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", cdi_dir.ljust(2048, b"\x00"))
            elif lba == 23:
                raw = encode_mode2_form2_sector(lba, b"\x01\x00\xa9\x00", b"{\\rtf1}" + b"\x00" * 2317)
            elif lba in (30, 31):
                raw = encode_mode2_form2_sector(lba, b"\x01\x00\x60\x00", b"\x00" * 2324)
            elif lba in (32, 33, 34):
                raw = encode_mode2_form2_sector(lba, b"\x01\x01\x62\x0f", mpeg_payloads[lba - 32])
            elif lba == 35:
                raw = encode_mode2_form2_sector(lba, b"\x01\x00\xe0\x00", b"\x00" * 2324)
            else:
                raw = encode_mode2_form1_sector(lba, b"\x00\x00\x00\x00", b"\x00" * 2048)
            sectors.append((raw, rspc.ZERO_294))

        # Intentionally corrupt LBA 33 (one active MPEG Form 2 sector)
        corrupt_raw = bytearray(sectors[33][0])
        corrupt_raw[100] ^= 0xFF
        c2_bad = bytearray(294)
        c2_bad[100 >> 3] |= 1 << (7 - (100 & 7))
        sectors[33] = (bytes(corrupt_raw), bytes(c2_bad))

        fake_drive = _FakeMode2Drive(sectors, tracks)
        with tempfile.TemporaryDirectory() as tmp:
            rec_root = Path(tmp) / "recovered"
            buf_stdout = io.StringIO()

            with contextlib.redirect_stdout(buf_stdout):
                report = run_mode2_pipeline(
                    drive=fake_drive,
                    tracks=tracks,
                    total_sectors=total_sectors,
                    recovered_root=rec_root,
                    session_name="VCD_SYNTH",
                    categories=["all"],
                    max_passes=0,
                    debug=True,
                )

            out_text = buf_stdout.getvalue()
            self.assertNotIn("{", out_text)
            self.assertNotIn("}", out_text)
            self.assertEqual(report["disc_mode"], "mode2")

            bin_path = Path(report["bin_out"])
            cue_path = Path(report["cue_out"])
            self.assertTrue(bin_path.exists())
            self.assertTrue(cue_path.exists())
            self.assertEqual(bin_path.stat().st_size, total_sectors * 2352)

            # LBA 33 must stay zero-filled in .bin and logged in SQLite raw_passes
            bin_data = bin_path.read_bytes()
            self.assertEqual(bin_data[33 * 2352 : 34 * 2352], b"\x00" * 2352)
            db = RecoveryStateDB(Path(report["session_dir"]) / "logs" / "raw_passes.sqlite3")
            try:
                self.assertEqual(db.all_tracked_bad_lbas(), [33])
            finally:
                db.close()

            # Extracted INFO.VCD matches byte-for-byte; CDI_ALL.RTF keeps .RTF; AVSEQ01.mpg skips 0x60/0xE0 padding
            ext_info = Path(report["session_dir"]) / "extracted" / "VCD" / "INFO.VCD"
            ext_rtf = Path(report["session_dir"]) / "extracted" / "CDI" / "CDI_ALL.RTF"
            ext_mpg = Path(report["session_dir"]) / "extracted" / "MPEGAV" / "AVSEQ01.mpg"
            self.assertTrue(ext_info.exists())
            self.assertTrue(ext_rtf.exists())
            self.assertEqual(ext_rtf.read_bytes()[:7], b"{\\rtf1}")
            self.assertTrue(ext_mpg.exists())
            self.assertEqual(ext_info.read_bytes(), info_vcd)
            mpg_bytes = ext_mpg.read_bytes()
            self.assertEqual(len(mpg_bytes), 3 * 2324)
            self.assertEqual(mpg_bytes[:2324], mpeg_payloads[0])
            self.assertEqual(mpg_bytes[2324 : 2 * 2324], b"\x00" * 2324)
            self.assertEqual(mpg_bytes[2 * 2324 : 3 * 2324], mpeg_payloads[2])

            self.assertTrue(report["manifest_csv"].endswith("manifest.csv"))
            self.assertTrue((Path(report["session_dir"]) / "extracted" / "manifest.csv").exists())
            self.assertFalse((Path(report["session_dir"]) / "extracted" / "photo_manifest.csv").exists())
            self.assertTrue((Path(report["session_dir"]) / "logs" / "debug.log").exists())
            self.assertTrue((Path(report["session_dir"]) / "logs" / "auto_recovery_report.json").exists())

            # Offline re-extraction via run_automated_pipeline(no_hardware=True) must handle Mode 2 BIN and preserve .cue
            cue_before = cue_path.read_text(encoding="utf-8")
            from src.pipeline.orchestrator import run_automated_pipeline

            with contextlib.redirect_stdout(io.StringIO()):
                offline_rep = run_automated_pipeline(
                    no_hardware=True,
                    recovered_root=rec_root,
                    session_name="VCD_SYNTH",
                    categories=["all"],
                )
            self.assertEqual(offline_rep["disc_mode"], "mode2")
            self.assertEqual(cue_path.read_text(encoding="utf-8"), cue_before)
            self.assertTrue((Path(report["session_dir"]) / "extracted" / "manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
