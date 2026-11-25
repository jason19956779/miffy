"""TDD tests for built-in image/disc inspection, session audit, baseline preserve, and drive status."""

import csv
import json
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.core import rspc
from src.core.state import RecoveryStateDB, preserve_baseline_image
from src.hardware.device import read_drive_status
from src.pipeline.diagnostics import audit_session, inspect_image
from test.core.test_rspc_and_state import encode_mode1_sector
from test.formats.test_media_handlers import _make_valid_mp3
from test.pipeline.test_vcd_mode2 import _dir_rec, encode_mode2_form1_sector, encode_mode2_form2_sector

REPO_DIR = Path(__file__).resolve().parents[2]


def _build_synthetic_mp3_iso(iso_path: Path, vol_id: str = "AUDIO_CD") -> bytes:
    mp3 = _make_valid_mp3(num_frames=6)
    mp3_len = len(mp3)
    n_mp3_sec = (mp3_len + 2047) // 2048

    root_dir = (
        _dir_rec(b"\x00", 18, 2048, True)
        + _dir_rec(b"\x01", 18, 2048, True)
        + _dir_rec(b"AUDIO", 19, 2048, True)
    )
    audio_dir = (
        _dir_rec(b"\x00", 19, 2048, True)
        + _dir_rec(b"\x01", 18, 2048, True)
        + _dir_rec(b"01.MP3", 20, mp3_len, False)
    )

    total_sectors = 20 + n_mp3_sec + 2
    img = bytearray(total_sectors * 2048)
    pvd = bytearray(2048)
    pvd[0] = 1
    pvd[1:6] = b"CD001"
    pvd[6] = 1
    pvd[40:72] = vol_id.encode("ascii").ljust(32, b" ")[:32]
    struct.pack_into("<II", pvd, 80, total_sectors, total_sectors)
    struct.pack_into("<HH", pvd, 128, 2048, 2048)
    pvd[156 : 156 + 34] = _dir_rec(b"\x00", 18, 2048, True)
    img[16 * 2048 : 17 * 2048] = pvd

    term = bytearray(2048)
    term[0] = 255
    term[1:6] = b"CD001"
    term[6] = 1
    img[17 * 2048 : 18 * 2048] = term

    img[18 * 2048 : 18 * 2048 + len(root_dir)] = root_dir
    img[19 * 2048 : 19 * 2048 + len(audio_dir)] = audio_dir
    img[20 * 2048 : 20 * 2048 + mp3_len] = mp3
    iso_path.write_bytes(bytes(img))
    return mp3


def _build_synthetic_mode2_bin(bin_path: Path) -> None:
    total_sectors = 25
    root_dir = _dir_rec(b"\x00", 18, 2048, True) + _dir_rec(b"\x01", 18, 2048, True) + _dir_rec(b"INFO.DAT", 19, 64, False)
    pvd = bytearray(2048)
    pvd[0] = 1
    pvd[1:6] = b"CD001"
    pvd[6] = 1
    pvd[40:72] = b"VCD_IMG".ljust(32, b" ")
    struct.pack_into("<II", pvd, 80, total_sectors, total_sectors)
    struct.pack_into("<HH", pvd, 128, 2048, 2048)
    pvd[156 : 156 + 34] = _dir_rec(b"\x00", 18, 2048, True)
    term = bytearray(2048)
    term[0] = 255
    term[1:6] = b"CD001"
    term[6] = 1

    out = bytearray()
    for lba in range(total_sectors):
        if lba == 16:
            out.extend(encode_mode2_form1_sector(lba, b"\x00\x00\x09\x00", bytes(pvd)))
        elif lba == 17:
            out.extend(encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", bytes(term)))
        elif lba == 18:
            out.extend(encode_mode2_form1_sector(lba, b"\x00\x00\x89\x00", root_dir.ljust(2048, b"\x00")))
        elif lba == 19:
            out.extend(encode_mode2_form2_sector(lba, b"\x01\x01\x62\x0f", b"\x00\x00\x01\xba" + b"\x11" * 2320))
        else:
            out.extend(encode_mode2_form1_sector(lba, b"\x00\x00\x00\x00", b"\x00" * 2048))
    bin_path.write_bytes(bytes(out))


class TestDiagnosticsAndBaselines(unittest.TestCase):
    def test_inspect_image_mode1_iso_and_mode2_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            iso_p = t / "audio.iso"
            _build_synthetic_mp3_iso(iso_p)
            info_iso = inspect_image(iso_p)
            self.assertEqual(info_iso["disc_mode"], "mode1")
            self.assertEqual(info_iso["sector_size"], 2048)
            self.assertEqual(info_iso["pvd"]["volume_id"], "AUDIO_CD")
            self.assertEqual(info_iso["file_count"], 1)
            self.assertEqual(info_iso["extensions"][".mp3"], 1)

            bin_p = t / "vcd.bin"
            _build_synthetic_mode2_bin(bin_p)
            info_bin = inspect_image(bin_p)
            self.assertEqual(info_bin["disc_mode"], "mode2")
            self.assertEqual(info_bin["sector_size"], 2352)
            self.assertEqual(info_bin["pvd"]["volume_id"], "VCD_IMG")
            self.assertEqual(info_bin["file_count"], 1)

    def test_preserve_baseline_image_iso_and_bin_cue_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            src_bin = t / "my_vcd_recovered.bin"
            src_cue = t / "my_vcd_recovered.cue"
            base_bin = t / "baseline.bin"
            base_cue = t / "baseline.cue"
            src_bin.write_bytes(b"RAWBINBYTES")
            src_cue.write_text('FILE "my_vcd_recovered.bin" BINARY\n  TRACK 01 MODE2/2352\n', encoding="utf-8")

            created = preserve_baseline_image(src_bin, base_bin, cue_src=src_cue, cue_baseline=base_cue)
            self.assertTrue(created)
            self.assertEqual(base_bin.read_bytes(), b"RAWBINBYTES")
            self.assertIn('FILE "baseline.bin" BINARY', base_cue.read_text(encoding="utf-8"))
            self.assertEqual(stat.S_IMODE(base_bin.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(base_cue.stat().st_mode), 0o444)

            # Second call refuses to overwrite existing baseline
            src_bin.write_bytes(b"MODIFIED")
            self.assertFalse(preserve_baseline_image(src_bin, base_bin, cue_src=src_cue, cue_baseline=base_cue))
            self.assertEqual(base_bin.read_bytes(), b"RAWBINBYTES")

    def test_read_drive_status_uses_only_readonly_inquiry_and_mode_sense(self):
        seen_opcodes: list[tuple[int, int | None]] = []

        def fake_scsi(self_drive, cdb_bytes, buf_bytes=None, buf_len=256, direction=-3, timeout=25000):
            opcode = cdb_bytes[0]
            page = cdb_bytes[2] if opcode == 0x5A else None
            seen_opcodes.append((opcode, page))
            if opcode == 0x12:
                data = bytearray(36)
                data[8:32] = b"FAKE OPTICAL DRIVE      "
                return 0, (0, 0, 0), 0, bytes(data)
            if opcode == 0x5A and page == 0x01:
                p01 = bytearray(64)
                p01[10] = 0x00
                p01[11] = 0x80
                return 0, (0, 0, 0), 0, bytes(p01)
            if opcode == 0x5A and page == 0x2A:
                p2a = bytearray(64)
                p2a[13] = 0x77
                p2a[16:18] = (4234).to_bytes(2, "big")
                p2a[22:24] = (4234).to_bytes(2, "big")
                return 0, (0, 0, 0), 0, bytes(p2a)
            raise AssertionError(f"Unexpected mutating SCSI opcode: 0x{opcode:02X}")

        with mock.patch("src.hardware.device.auto_detect_sg_device", return_value=("/dev/sg9", "/dev/sr9", 120)):
            with mock.patch("src.hardware.device.os.open", return_value=999):
                with mock.patch("src.hardware.device.os.close") as mock_close:
                    with mock.patch("src.hardware.device.OpticalDrive.scsi_cmd", new=fake_scsi):
                        status = read_drive_status("/dev/sg9")
        mock_close.assert_called_once_with(999)
        self.assertEqual(seen_opcodes, [(0x12, None), (0x5A, 0x01), (0x5A, 0x2A)])
        self.assertTrue(status["mode_page_01"]["restored_ok"])
        self.assertEqual(status["mode_page_01"]["err_bits"], 0)
        self.assertEqual(status["mode_page_01"]["retries"], 0x80)
        self.assertEqual(status["mode_page_2a"]["max_speed_kbps"], 4234)

    def test_audit_session_with_manifest_fallback_and_parity_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            s1 = t / "SESSION_A"
            s2 = t / "SESSION_B"
            for idx, sdir in enumerate((s1, s2)):
                (sdir / "extracted" / "AUDIO").mkdir(parents=True)
                (sdir / "logs").mkdir(parents=True)
                iso_p = sdir / f"{sdir.name.lower()}_recovered.iso"
                mp3 = _build_synthetic_mp3_iso(iso_p)
                mp3_file = sdir / "extracted" / "AUDIO" / "01.MP3"
                mp3_file.write_bytes(mp3)

                # Exercise SQLite solved sector + physical Q-parity check on SESSION_A
                db = RecoveryStateDB(sdir / "logs" / "raw_passes.sqlite3")
                raw20 = encode_mode1_sector(20, mp3[:2048].ljust(2048, b"\x00"))
                db.insert_raw_pass(20, 0, raw20, rspc.ZERO_294)
                db.upsert_solution(20, raw20, True, 86, 52, 0, "TEST_QPAR")
                db.conn.commit()
                db.close()

                # Write new manifest.csv in SESSION_A, legacy photo_manifest.csv in SESSION_B
                mname = "manifest.csv" if idx == 0 else "photo_manifest.csv"
                with (sdir / "extracted" / mname).open("w", newline="", encoding="utf-8") as cf:
                    w = csv.DictWriter(
                        cf,
                        fieldnames=[
                            "iso_path",
                            "saved_path",
                            "directory",
                            "size_bytes",
                            "bad_sectors_total",
                            "bad_sectors_verified",
                            "min_p_ok",
                            "zero_holes_2k",
                            "recovery_class",
                            "format_status",
                            "handler",
                            "details",
                        ],
                    )
                    w.writeheader()
                    w.writerow(
                        {
                            "iso_path": "/AUDIO/01.MP3",
                            "saved_path": str(mp3_file),
                            "directory": "/AUDIO",
                            "size_bytes": len(mp3),
                            "bad_sectors_total": 0,
                            "bad_sectors_verified": 0,
                            "min_p_ok": 86,
                            "zero_holes_2k": 0,
                            "recovery_class": "INTACT_ORIGINAL",
                            "format_status": "AUDIO_MPEG_OK",
                            "handler": "audio",
                            "details": "{}",
                        }
                    )

            rep = audit_session(s1, run_ffmpeg=False, compare_session=s2)
            self.assertEqual(rep["manifest_name"], "manifest.csv")
            self.assertEqual(rep["validation"]["passed"], 1)
            self.assertEqual(rep["validation"]["failed"], 0)
            self.assertEqual(rep["q_parity_gate"]["checked"], 1)
            self.assertEqual(rep["q_parity_gate"]["passed"], 1)
            self.assertEqual(rep["compare"]["exact_match"], 1)
            self.assertEqual(rep["compare"]["mismatches"], [])

            # CLI --inspect-image --json and --audit-session --json
            out_ins = subprocess.check_output(
                [sys.executable, str(REPO_DIR / "miffy.py"), "--inspect-image", str(s1 / "session_a_recovered.iso"), "--json"],
                cwd=str(REPO_DIR),
                text=True,
            )
            ins_json = json.loads(out_ins)
            self.assertEqual(ins_json["pvd"]["volume_id"], "AUDIO_CD")

            out_aud = subprocess.check_output(
                [sys.executable, str(REPO_DIR / "miffy.py"), "--audit-session", str(s1), "--compare-session", str(s2), "--json"],
                cwd=str(REPO_DIR),
                text=True,
            )
            aud_json = json.loads(out_aud)
            self.assertEqual(aud_json["validation"]["passed"], 1)
            self.assertEqual(aud_json["compare"]["exact_match"], 1)

    def test_audit_session_allows_verified_zeros_and_flags_partial_zero_holes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdir = Path(tmp) / "ZERO_AUDIT"
            (sdir / "extracted" / "CDI").mkdir(parents=True)
            (sdir / "logs").mkdir(parents=True)
            rtf = sdir / "extracted" / "CDI" / "CDI_ALL.RTF"
            bad_dat = sdir / "extracted" / "CDI" / "BAD_HOLE.DAT"
            rtf.write_bytes(b"\x00" * 2324)
            bad_dat.write_bytes(b"\x00" * 2324)

            with (sdir / "extracted" / "manifest.csv").open("w", newline="", encoding="utf-8") as cf:
                w = csv.DictWriter(
                    cf,
                    fieldnames=[
                        "iso_path",
                        "saved_path",
                        "directory",
                        "size_bytes",
                        "bad_sectors_total",
                        "bad_sectors_verified",
                        "min_p_ok",
                        "zero_holes_2k",
                        "recovery_class",
                        "format_status",
                        "handler",
                        "details",
                    ],
                )
                w.writeheader()
                w.writerow(
                    {
                        "iso_path": "/CDI/CDI_ALL.RTF",
                        "saved_path": str(rtf),
                        "directory": "/CDI",
                        "size_bytes": 2324,
                        "bad_sectors_total": 0,
                        "bad_sectors_verified": 0,
                        "min_p_ok": 0,
                        "zero_holes_2k": 0,
                        "recovery_class": "INTACT_ORIGINAL",
                        "format_status": "M2F2_RAW_OK",
                        "handler": "mode2_form2_raw",
                        "details": "{}",
                    }
                )
                w.writerow(
                    {
                        "iso_path": "/CDI/BAD_HOLE.DAT",
                        "saved_path": str(bad_dat),
                        "directory": "/CDI",
                        "size_bytes": 2324,
                        "bad_sectors_total": 1,
                        "bad_sectors_verified": 0,
                        "min_p_ok": 0,
                        "zero_holes_2k": 1,
                        "recovery_class": "PARTIAL_RSPC_RECONSTRUCTED",
                        "format_status": "ZERO_HOLES (1)",
                        "handler": "generic_binary",
                        "details": "{}",
                    }
                )

            rep = audit_session(sdir, run_ffmpeg=False)
            self.assertEqual(rep["validation"]["passed"], 1)
            self.assertEqual(rep["validation"]["failed"], 1)
            by_p = {f["path"]: f for f in rep["validation"]["files"]}
            self.assertTrue(by_p["CDI/CDI_ALL.RTF"]["valid"])
            self.assertEqual(by_p["CDI/CDI_ALL.RTF"]["status"], "BINARY_OK (verified_zeros)")
            self.assertFalse(by_p["CDI/BAD_HOLE.DAT"]["valid"])


if __name__ == "__main__":
    unittest.main()
