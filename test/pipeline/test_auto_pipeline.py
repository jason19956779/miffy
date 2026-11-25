"""Integration tests for session directory auto-naming, priority scheduling, --debug logs, and CLI."""

import io
import json
import re
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.core import rspc
from src.core.logging_utils import RecoveryLogger
from src.core.models import FileExtent
from src.core.state import RecoveryStateDB
from src.filesystem import iso9660
from src.pipeline.orchestrator import (
    SPEEDS_MULTI,
    _match_category,
    _run_tier3_phase,
    partition_targets_by_priority,
    resolve_session_dir,
    sanitize_volume_id,
    select_adaptive_focus_speeds,
)
from test.core.test_rspc_and_state import encode_mode1_sector

REPO_DIR = Path(__file__).resolve().parents[2]
MIFFY_ISO = REPO_DIR / "recovered" / "MIFFY" / "miffy_recovered.iso"


def _make_minimal_jpeg_iso(iso_path: Path, vol_id: str = "DBG_TEST") -> None:
    """Build a minimal 24-sector ISO-9660 image with /PHOTO/IMG01.JPG at LBA 20."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 180, 60)).save(buf, format="JPEG", quality=90)
    jpg = buf.getvalue()
    assert len(jpg) <= 4096

    def dir_rec(ident: bytes, ext_lba: int, size: int, is_dir: bool) -> bytes:
        pad = 1 if (len(ident) % 2 == 0) else 0
        rec_len = 33 + len(ident) + pad
        r = bytearray(rec_len)
        r[0] = rec_len
        struct.pack_into("<II", r, 2, ext_lba, ext_lba)
        struct.pack_into("<II", r, 10, size, size)
        r[18:25] = bytes([126, 11, 24, 12, 0, 0, 0])
        r[25] = 0x02 if is_dir else 0x00
        struct.pack_into("<HH", r, 28, 1, 1)
        r[32] = len(ident)
        r[33 : 33 + len(ident)] = ident
        return bytes(r)

    root_dir = dir_rec(b"\x00", 18, 2048, True) + dir_rec(b"\x01", 18, 2048, True) + dir_rec(b"PHOTO", 19, 2048, True)
    photo_dir = dir_rec(b"\x00", 19, 2048, True) + dir_rec(b"\x01", 18, 2048, True) + dir_rec(b"IMG01.JPG", 20, len(jpg), False)

    img = bytearray(24 * 2048)
    # PVD at LBA 16
    pvd = bytearray(2048)
    pvd[0] = 1
    pvd[1:6] = b"CD001"
    pvd[6] = 1
    pvd[40:72] = vol_id.encode("ascii").ljust(32, b" ")[:32]
    struct.pack_into("<II", pvd, 80, 24, 24)
    struct.pack_into("<HH", pvd, 128, 2048, 2048)
    pvd[156 : 156 + 34] = dir_rec(b"\x00", 18, 2048, True)
    img[16 * 2048 : 17 * 2048] = pvd

    # Terminator SVD at LBA 17
    term = bytearray(2048)
    term[0] = 255
    term[1:6] = b"CD001"
    term[6] = 1
    img[17 * 2048 : 18 * 2048] = term

    img[18 * 2048 : 18 * 2048 + len(root_dir)] = root_dir
    img[19 * 2048 : 19 * 2048 + len(photo_dir)] = photo_dir
    img[20 * 2048 : 20 * 2048 + len(jpg)] = jpg
    iso_path.write_bytes(bytes(img))


class TestAutoPipeline(unittest.TestCase):
    def test_sanitize_volume_id(self):
        self.assertEqual(sanitize_volume_id("_________MIFFY   "), "MIFFY")
        self.assertEqual(sanitize_volume_id("Vacation 2004/CD#1"), "Vacation_2004_CD_1")
        self.assertEqual(sanitize_volume_id("___   "), "")

    @unittest.skipUnless(MIFFY_ISO.exists(), "Recovered MIFFY ISO not present")
    def test_resolve_session_dir_from_iso_and_uuid_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec_root = Path(tmp) / "recovered"
            name_pvd, sdir_pvd = resolve_session_dir(
                recovered_root=rec_root,
                no_hardware=True,
                iso_out=MIFFY_ISO,
            )
            self.assertEqual(name_pvd, "MIFFY")
            self.assertEqual(sdir_pvd, rec_root / "MIFFY")

            name_custom, sdir_custom = resolve_session_dir(
                recovered_root=rec_root,
                session_name="My Custom Disc",
                no_hardware=True,
            )
            self.assertEqual(name_custom, "My_Custom_Disc")
            self.assertEqual(sdir_custom, rec_root / "My_Custom_Disc")

            name_uuid, sdir_uuid = resolve_session_dir(
                recovered_root=rec_root,
                no_hardware=True,
            )
            self.assertRegex(name_uuid, r"^cd_[0-9a-f]{8}$")
            self.assertEqual(sdir_uuid, rec_root / name_uuid)

    def test_cli_list_plugins(self):
        out = subprocess.check_output(
            [sys.executable, str(REPO_DIR / "miffy.py"), "--list-plugins"],
            cwd=str(REPO_DIR),
            text=True,
        )
        self.assertIn("joint_gf2_edc", out)
        self.assertIn("jpeg", out)

    def test_partition_targets_by_priority(self):
        extents = [
            FileExtent(path="/MAIN.DXR", directory="/", name="MAIN.DXR", ext=".DXR", start_lba=900, end_lba=902, size_bytes=4096, sectors=2, bad_lbas=[900, 901], priority=40),
            FileExtent(path="/PHOTO/L_PHOTO/A.JPG", directory="/PHOTO/L_PHOTO", name="A.JPG", ext=".JPG", start_lba=200, end_lba=202, size_bytes=4096, sectors=2, bad_lbas=[200, 201], priority=10),
            FileExtent(path="/XTRAS/X.X32", directory="/XTRAS", name="X.X32", ext=".X32", start_lba=700, end_lba=701, size_bytes=2048, sectors=1, bad_lbas=[700], priority=40),
            FileExtent(path="/PHOTO/S/A_T.JPG", directory="/PHOTO/S", name="A_T.JPG", ext=".JPG", start_lba=300, end_lba=301, size_bytes=2048, sectors=1, bad_lbas=[300], priority=20),
            FileExtent(path="/PIC/B.BMP", directory="/PIC", name="B.BMP", ext=".BMP", start_lba=400, end_lba=401, size_bytes=2048, sectors=1, bad_lbas=[400, 201], priority=10),
            FileExtent(path="/PHOTO/PHOTO.INI", directory="/PHOTO", name="PHOTO.INI", ext=".INI", start_lba=500, end_lba=501, size_bytes=2048, sectors=1, bad_lbas=[500], priority=30),
        ]
        media_lbas, bulk_lbas = partition_targets_by_priority(extents, ["all"])
        self.assertEqual(media_lbas, [200, 201, 300, 400])
        self.assertEqual(bulk_lbas, [500, 700, 900, 901])
        self.assertTrue(set(media_lbas).isdisjoint(set(bulk_lbas)))

        aud_ext = FileExtent(path="/AUDIO/01.MP3", directory="/AUDIO", name="01.MP3", ext=".MP3", start_lba=10, end_lba=12, size_bytes=3000, sectors=2, bad_lbas=[10], priority=10)
        vid_ext = FileExtent(path="/MPEGAV/AVSEQ01.DAT", directory="/MPEGAV", name="AVSEQ01.DAT", ext=".DAT", start_lba=20, end_lba=22, size_bytes=4096, sectors=2, bad_lbas=[20], priority=10)
        self.assertTrue(_match_category(aud_ext, ["audio"]))
        self.assertTrue(_match_category(aud_ext, ["media"]))
        self.assertTrue(_match_category(vid_ext, ["video"]))
        self.assertTrue(_match_category(vid_ext, ["media"]))
        self.assertIn(".MP3", iso9660.AUDIO_EXTS)
        self.assertIn(".MPG", iso9660.VIDEO_EXTS)

    def test_cli_debug_flag_and_timestamps(self):
        help_out = subprocess.check_output(
            [sys.executable, str(REPO_DIR / "miffy.py"), "--help"],
            cwd=str(REPO_DIR),
            text=True,
        )
        for flag in ("--debug", "--focus-passes", "--workers"):
            self.assertIn(flag, help_out)

        line_re = re.compile(r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\] ")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rec_root = tmp_dir / "recovered"
            iso_in = tmp_dir / "dbg_test.iso"
            _make_minimal_jpeg_iso(iso_in, vol_id="DBG_TEST")
            out = subprocess.check_output(
                [
                    sys.executable,
                    str(REPO_DIR / "miffy.py"),
                    "--no-hardware",
                    "--debug",
                    "--recovered-root",
                    str(rec_root),
                    "--name",
                    "DBG_TEST",
                    "--iso-out",
                    str(iso_in),
                    "--categories",
                    "all",
                ],
                cwd=str(REPO_DIR),
                text=True,
            )
            # Strip prefixed log lines and parse trailing JSON report
            json_start = out.index("{")
            report = json.loads(out[json_start:])
            self.assertTrue(report["debug_enabled"])
            self.assertIn("timing", report)
            for k in ("started_at", "finished_at", "total_elapsed_s"):
                self.assertIn(k, report["timing"])

            logs_dir = rec_root / "DBG_TEST" / "logs"
            for fname in ("recovery.log", "debug.log", "debug_events.jsonl"):
                p = logs_dir / fname
                self.assertTrue(p.exists(), f"Missing {fname}")
                self.assertGreater(p.stat().st_size, 0)
            for line in (logs_dir / "recovery.log").read_text(encoding="utf-8").splitlines():
                self.assertRegex(line, line_re)
            for line in (logs_dir / "debug.log").read_text(encoding="utf-8").splitlines():
                self.assertRegex(line, line_re)

            conn = sqlite3.connect(logs_dir / "raw_passes.sqlite3")
            evt_cnt = conn.execute("SELECT COUNT(*) FROM debug_events").fetchone()[0]
            conn.close()
            self.assertGreaterEqual(evt_cnt, 1)

    def test_select_adaptive_focus_speeds_prioritizes_best_speeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = RecoveryStateDB(Path(tmp) / "state.sqlite3")
            try:
                # Empty pass_metrics fallback must return valid cycle of length focus_passes
                empty_sched = select_adaptive_focus_speeds(db, [64921], focus_passes=24)
                self.assertEqual(len(empty_sched), 24)
                self.assertTrue(all(s in SPEEDS_MULTI for s in empty_sched))

                # Populate pass_metrics favoring 4234, 1411, 2822 KB/s
                samples = [
                    (4234, 67, 33),
                    (1411, 56, 21),
                    (2822, 51, 23),
                    (176, 32, 10),
                    (353, 30, 9),
                    (706, 28, 8),
                    (1058, 25, 7),
                    (2117, 35, 11),
                ]
                for idx, (spd, pok, qok) in enumerate(samples):
                    db.record_pass_metric(
                        64921,
                        idx,
                        "2026-11-25T10:00:00.000",
                        "media",
                        idx + 1,
                        "FWD" if idx % 2 == 0 else "REV",
                        spd,
                        pok,
                        qok,
                        "0e263301",
                        1,
                        0,
                        1.0,
                    )
                db.conn.commit()

                sched = select_adaptive_focus_speeds(db, [64921], focus_passes=24)
                self.assertEqual(len(sched), 24)
                # Initial exploration slice covers multiple distinct speeds
                self.assertGreaterEqual(len(set(sched[:8])), 4)
                # Dominant share uses the top-3 speeds
                top3_hits = sum(1 for s in sched if s in {4234, 1411, 2822})
                self.assertGreaterEqual(top3_hits, 12)
            finally:
                db.close()

    def test_debug_mode_emits_stage3_to_6_live_progress_to_console(self):
        def make_textured_jpeg(seed: int, size: tuple[int, int] = (64, 64), quality: int = 88) -> bytes:
            w, h = size
            px = bytearray(w * h * 3)
            for y in range(h):
                for x in range(w):
                    i = (y * w + x) * 3
                    px[i] = ((x * 5 + y * 9 + seed) + ((x ^ y) & 15) * 4) & 0xFF
                    px[i + 1] = ((x * 7 + y * 3 + seed * 3) + ((x * 3 ^ y) & 15) * 4) & 0xFF
                    px[i + 2] = ((x * 3 + y * 11 + seed * 5) + (((x + y) >> 1) & 15) * 4) & 0xFF
            buf = io.BytesIO()
            Image.frombytes("RGB", (w, h), bytes(px)).save(buf, format="JPEG", quality=quality)
            return buf.getvalue()

        def dir_rec(ident: bytes, ext_lba: int, size: int, is_dir: bool) -> bytes:
            pad = 1 if (len(ident) % 2 == 0) else 0
            rec_len = 33 + len(ident) + pad
            r = bytearray(rec_len)
            r[0] = rec_len
            struct.pack_into("<II", r, 2, ext_lba, ext_lba)
            struct.pack_into("<II", r, 10, size, size)
            r[18:25] = bytes([126, 11, 24, 12, 0, 0, 0])
            r[25] = 0x02 if is_dir else 0x00
            struct.pack_into("<HH", r, 28, 1, 1)
            r[32] = len(ident)
            r[33 : 33 + len(ident)] = ident
            return bytes(r)

        buf1 = io.BytesIO()
        Image.new("RGB", (32, 32), (120, 180, 60)).save(buf1, format="JPEG", quality=90)
        jpg1 = buf1.getvalue()
        jpg2 = make_textured_jpeg(17, size=(64, 64), quality=88)
        jpg3 = make_textured_jpeg(73, size=(80, 80), quality=88)
        self.assertLessEqual(len(jpg1), 2048)
        self.assertGreater(len(jpg2), 2048)
        self.assertLessEqual(len(jpg2), 4096)
        self.assertGreater(len(jpg3), 4096)
        self.assertLessEqual(len(jpg3), 6144)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img = bytearray(32 * 2048)
            pvd = bytearray(2048)
            pvd[0] = 1
            pvd[1:6] = b"CD001"
            pvd[6] = 1
            pvd[40:72] = b"DBG_LIVE".ljust(32, b" ")
            struct.pack_into("<II", pvd, 80, 32, 32)
            struct.pack_into("<HH", pvd, 128, 2048, 2048)
            pvd[156 : 156 + 34] = dir_rec(b"\x00", 18, 2048, True)
            img[16 * 2048 : 17 * 2048] = pvd

            term = bytearray(2048)
            term[0] = 255
            term[1:6] = b"CD001"
            term[6] = 1
            img[17 * 2048 : 18 * 2048] = term

            root_dir = (
                dir_rec(b"\x00", 18, 2048, True)
                + dir_rec(b"\x01", 18, 2048, True)
                + dir_rec(b"PHOTO", 19, 2048, True)
            )
            photo_dir = (
                dir_rec(b"\x00", 19, 2048, True)
                + dir_rec(b"\x01", 18, 2048, True)
                + dir_rec(b"IMG01.JPG", 20, len(jpg1), False)
                + dir_rec(b"IMG02.JPG", 21, len(jpg2), False)
                + dir_rec(b"IMG03.JPG", 23, len(jpg3), False)
            )
            img[18 * 2048 : 18 * 2048 + len(root_dir)] = root_dir
            img[19 * 2048 : 19 * 2048 + len(photo_dir)] = photo_dir
            img[20 * 2048 : 20 * 2048 + len(jpg1)] = jpg1
            img[21 * 2048 : 21 * 2048 + 2048] = jpg2[:2048]
            # LBA 22 left zero-filled on initial ISO (solvable from SQLite raw_passes)
            img[23 * 2048 : 23 * 2048 + 2048] = jpg3[:2048]
            # LBA 24 (middle sector of IMG03.JPG) left zero-filled on initial ISO -> repaired by JpegFormatHandler
            img[25 * 2048 : 25 * 2048 + (len(jpg3) - 4096)] = jpg3[4096:]

            iso_dbg = tmp_dir / "dbg_live.iso"
            iso_dbg.write_bytes(bytes(img))
            db_dbg = tmp_dir / "dbg_live.sqlite3"
            db = RecoveryStateDB(db_dbg)
            try:
                sec22 = encode_mode1_sector(22, jpg2[2048:].ljust(2048, b"\x00"))
                sec22_err = bytearray(sec22)
                sec22_err[16 + 50] ^= 0x5A
                c2_22 = bytearray(294)
                pos = 16 + 50
                c2_22[pos >> 3] |= 1 << (7 - (pos & 7))
                db.insert_raw_pass(22, 0, bytes(sec22_err), bytes(c2_22))

                # Unsolvable corrupted sector 24
                sec24_bad = bytearray(2352)
                rspc.init_sector_header(24, sec24_bad)
                for p in range(16, 2064, 17):
                    sec24_bad[p] = (p * 37) & 0xFF
                db.insert_raw_pass(24, 0, bytes(sec24_bad), b"\xff" * 294)
                db.conn.commit()
            finally:
                db.close()

            rec_root = tmp_dir / "recovered"
            out_dbg = subprocess.check_output(
                [
                    sys.executable,
                    str(REPO_DIR / "miffy.py"),
                    "--no-hardware",
                    "--debug",
                    "--recovered-root",
                    str(rec_root),
                    "--name",
                    "DBG_LIVE",
                    "--iso-out",
                    str(iso_dbg),
                    "--db",
                    str(db_dbg),
                    "--categories",
                    "all",
                ],
                cwd=str(REPO_DIR),
                text=True,
            )
            self.assertIn("[Stage 3: Triage]", out_dbg)
            self.assertIn("[Stage 3: Target Plan]", out_dbg)
            self.assertIn("[DEBUG]", out_dbg)
            self.assertIn("[Stage 3 dir]", out_dbg)
            self.assertIn("[Offline Solver", out_dbg)
            self.assertIn("VERIFIED", out_dbg)
            self.assertIn("/PHOTO/IMG02.JPG", out_dbg)
            self.assertIn("[Extract", out_dbg)
            self.assertIn("[Format Repair", out_dbg)
            self.assertIn("/PHOTO/IMG03.JPG", out_dbg)
            self.assertIn("[Stages 5-6 Complete]", out_dbg)

            json_start = out_dbg.index("{")
            report_dbg = json.loads(out_dbg[json_start:])
            self.assertTrue(report_dbg["debug_enabled"])
            self.assertTrue(report_dbg["manifest_csv"].endswith("manifest.csv"))
            self.assertFalse((Path(report_dbg["session_dir"]) / "extracted" / "photo_manifest.csv").exists())
            self.assertTrue((Path(report_dbg["session_dir"]) / "extracted" / "manifest.csv").exists())

            # Normal mode (--debug omitted) must keep concise milestones without [DEBUG] lines
            iso_q = tmp_dir / "quiet_live.iso"
            iso_q.write_bytes(bytes(img))
            db_q = tmp_dir / "quiet_live.sqlite3"
            import shutil

            shutil.copy2(db_dbg, db_q)
            out_q = subprocess.check_output(
                [
                    sys.executable,
                    str(REPO_DIR / "miffy.py"),
                    "--no-hardware",
                    "--recovered-root",
                    str(rec_root),
                    "--name",
                    "QUIET_LIVE",
                    "--iso-out",
                    str(iso_q),
                    "--db",
                    str(db_q),
                    "--categories",
                    "all",
                ],
                cwd=str(REPO_DIR),
                text=True,
            )
            self.assertIn("[Stage 3: Target Plan]", out_q)
            self.assertIn("[Stages 5-6 Complete]", out_q)
            self.assertNotIn("[DEBUG]", out_q)

            # Direct Tier-3 phase check with 2 workers & extents mapping
            t3_iso = tmp_dir / "tier3.iso"
            t3_iso.write_bytes(bytes(img))
            t3_db = RecoveryStateDB(tmp_dir / "tier3.sqlite3")
            t3_stream = io.StringIO()
            t3_logger = RecoveryLogger(tmp_dir / "t3_logs", db=t3_db, debug=True, stream=t3_stream)
            try:
                sec22 = encode_mode1_sector(22, jpg2[2048:].ljust(2048, b"\x00"))
                sec22_e = bytearray(sec22)
                sec22_e[16 + 40] ^= 0x3C
                c2_22 = bytearray(294)
                pos = 16 + 40
                c2_22[pos >> 3] |= 1 << (7 - (pos & 7))
                t3_db.insert_raw_pass(22, 0, bytes(sec22_e), bytes(c2_22))

                sec24_b = bytearray(2352)
                rspc.init_sector_header(24, sec24_b)
                for p in range(16, 2064, 13):
                    sec24_b[p] = (p * 19) & 0xFF
                t3_db.insert_raw_pass(24, 0, bytes(sec24_b), b"\xff" * 294)
                t3_db.conn.commit()

                extents = [
                    FileExtent(
                        path="/PHOTO/T1.JPG",
                        directory="/PHOTO",
                        name="T1.JPG",
                        ext=".JPG",
                        start_lba=21,
                        end_lba=23,
                        size_bytes=len(jpg2),
                        sectors=2,
                        bad_lbas=[22],
                        priority=10,
                    ),
                    FileExtent(
                        path="/PHOTO/T2.JPG",
                        directory="/PHOTO",
                        name="T2.JPG",
                        ext=".JPG",
                        start_lba=23,
                        end_lba=25,
                        size_bytes=len(jpg3),
                        sectors=2,
                        bad_lbas=[24],
                        priority=10,
                    ),
                ]
                _run_tier3_phase(
                    t3_db,
                    t3_iso,
                    [22, 24],
                    max_tier=3,
                    workers=2,
                    extents=extents,
                    phase="media",
                    logger=t3_logger,
                )
            finally:
                t3_logger.close()
                t3_db.close()

            t3_out = t3_stream.getvalue()
            self.assertIn("1/2", t3_out)
            self.assertIn("2/2", t3_out)
            self.assertIn("/PHOTO/T1.JPG", t3_out)
            self.assertIn("/PHOTO/T2.JPG", t3_out)
            self.assertIn("VERIFIED", t3_out)
            self.assertIn("PARTIAL", t3_out)
            self.assertIn("solvers=", t3_out)


if __name__ == "__main__":
    unittest.main()
