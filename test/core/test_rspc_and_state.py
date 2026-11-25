"""Unit tests for ECMA-130 RSPC P/Q/EDC primitives and SQLite RecoveryStateDB."""

import io
import re
import tempfile
import unittest
from pathlib import Path

from src.core import rspc
from src.core.logging_utils import RecoveryLogger, format_timestamp
from src.core.state import RecoveryStateDB, load_wincdemu_bad_lbas


def encode_mode1_sector(lba: int, user_2048: bytes) -> bytearray:
    """Construct a mathematically valid 2,352-byte ECMA-130 Mode 1 raw sector."""
    assert len(user_2048) == 2048
    sec = bytearray(2352)
    rspc.init_sector_header(lba, sec)
    sec[16:2064] = user_2048
    edc = rspc.compute_edc(sec[:2064])
    sec[2064:2068] = edc.to_bytes(4, "little")
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
    assert rspc.check_and_fix_q_parity(sec)
    return sec


class TestRspcAndState(unittest.TestCase):
    def test_gf_arithmetic_identities(self):
        for a in (1, 2, 7, 42, 137, 255):
            self.assertEqual(rspc.gf_div(rspc.GF_MUL[a][99], 99), a)
            self.assertEqual(rspc.GF_MUL[a][0], 0)
            self.assertEqual(rspc.GF_MUL[0][a], 0)

    def test_p_and_q_codeword_coverage(self):
        self.assertEqual(len(rspc.P_IDX), 86)
        self.assertEqual(len(rspc.Q_IDX), 52)
        p_data_bytes = {pos for cw in rspc.P_IDX for pos in cw[:24]}
        q_data_bytes = {pos for cw in rspc.Q_IDX for pos in cw[:43]}
        self.assertEqual(p_data_bytes, set(range(12, 2076)))
        self.assertEqual(q_data_bytes, set(range(12, 2248)))

    def test_edc_gf2_linearity_and_rspc_roundtrip(self):
        payload = bytes((i * 37 + 13) & 0xFF for i in range(2048))
        sec = encode_mode1_sector(64208, payload)
        self.assertEqual(rspc.count_pq(sec), (86, 52))
        self.assertEqual(rspc.edc_residual(sec[:2068]), 0)

        # Linearity check: EDC(A ^ B) == EDC(A) ^ EDC(B) when initial zero
        a = bytearray(2064)
        b = bytearray(2064)
        a[100] = 0x5A
        b[900] = 0xC3
        ab = bytes(x ^ y for x, y in zip(a, b))
        self.assertEqual(rspc.compute_edc(ab), rspc.compute_edc(a) ^ rspc.compute_edc(b))

    def test_solve_1err_and_2eras(self):
        payload = bytes((i * 19 + 5) & 0xFF for i in range(2048))
        sec = encode_mode1_sector(12345, payload)
        orig_200 = sec[200]
        sec[200] ^= 0xA5
        cw = rspc.P_IDX[rspc.BYTE_TO_P[200]]
        st, corrs = rspc.solve_1err(sec, cw)
        self.assertEqual(st, 1)
        self.assertEqual(corrs, [(200, orig_200)])

        # 2-erasure recovery on same codeword
        i0 = cw.index(200)
        p1 = cw[3]
        orig_p1 = sec[p1]
        sec[p1] ^= 0x3C
        ok, corrs2 = rspc.solve_2eras(sec, cw, [3, i0])
        self.assertTrue(ok)
        self.assertEqual(dict(corrs2), {p1: orig_p1, 200: orig_200})

    def test_sqlite_state_db_and_bad_log_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            db = RecoveryStateDB(tmp_dir / "state.sqlite3")
            payload = b"\xAB" * 2048
            sec = encode_mode1_sector(500, payload)
            c2 = b"\x00" * 294
            pz, qz = db.insert_raw_pass(500, 0, bytes(sec), c2)
            self.assertEqual((pz, qz), (86, 52))
            self.assertTrue(db.upsert_solution(500, sec, True, 86, 52, 0, "TEST_OK"))
            self.assertIn(500, db.get_verified_set())
            # Partial candidate never overwrites verified sector
            self.assertTrue(db.upsert_solution(500, bytearray(2352), False, 10, 5, 1, "PARTIAL"))
            self.assertEqual(db.get_solved_map()[500][0], 1)
            st_json = db.export_state_json(tmp_dir / "state.json", [500])
            self.assertEqual(st_json["verified_sectors"], 1)
            db.close()

            log_f = tmp_dir / "bad.txt"
            log_f.write_text("0x0010 - 0x0013 (3 sectors)\n", encoding="utf-8")
            self.assertEqual(load_wincdemu_bad_lbas(log_f), [16, 17, 18])

    def test_msf_decoding_and_expected_lba_verification(self):
        for lba in (0, 16, 51838, 58348, 80434, 124078):
            self.assertEqual(rspc.msf_bytes_to_lba(rspc.lba_header(lba)), lba)

        for bad_hdr in (
            b"\x00\x02\x00\x02",  # invalid mode 0x02
            b"\x9e\x07\x09\x01",  # non-BCD nibble
            b"\x01\x60\x00\x01",  # SS >= 60
            b"\x01\x00\x75\x01",  # FF >= 75
            b"\x00\x02\x00",      # too short
        ):
            self.assertIsNone(rspc.msf_bytes_to_lba(bad_hdr))

        payload = bytes((i * 29 + 7) & 0xFF for i in range(2048))
        sec = encode_mode1_sector(58347, payload)
        self.assertTrue(rspc.check_and_fix_q_parity(bytearray(sec), expected_lba=58347))

        sec_copy = bytearray(sec)
        self.assertFalse(rspc.check_and_fix_q_parity(sec_copy, expected_lba=58348))
        self.assertEqual(bytes(sec_copy), bytes(sec))

        sec_bad_sync = bytearray(sec)
        sec_bad_sync[0] ^= 0xFF
        self.assertFalse(rspc.check_and_fix_q_parity(sec_bad_sync, expected_lba=58347))

        sec_bad_zero8 = bytearray(sec)
        sec_bad_zero8[2068] = 0x55
        self.assertFalse(rspc.check_and_fix_q_parity(sec_bad_zero8, expected_lba=58347))

    def test_is_frame_slip_distinguishes_slip_from_scratched_header(self):
        payload_prev = bytes((i * 17 + 1) & 0xFF for i in range(2048))
        payload_curr = bytes((i * 31 + 9) & 0xFF for i in range(2048))
        sec_prev = encode_mode1_sector(58347, payload_prev)
        sec_curr = encode_mode1_sector(58348, payload_curr)

        self.assertTrue(rspc.is_frame_slip(58348, bytes(sec_prev)))
        self.assertFalse(rspc.is_frame_slip(58348, bytes(sec_curr)))

        # Mildly noisy slipped sector (5 P-columns damaged, >=40 P-codewords still valid under 58347)
        noisy_prev = bytearray(sec_prev)
        for c in range(5):
            noisy_prev[rspc.P_IDX[c][2]] ^= 0xA5
        self.assertTrue(rspc.is_frame_slip(58348, bytes(noisy_prev)))

        # Scratched header byte on true 58348 sector (non-BCD nibble) -> NOT a frame slip
        scratched_hdr = bytearray(sec_curr)
        scratched_hdr[12] = 0x9E
        self.assertFalse(rspc.is_frame_slip(58348, bytes(scratched_hdr)))

        # Heavily scratched 58348 sector whose frame byte coincidentally equals 58347's BCD byte
        heavy_scratch = bytearray(sec_curr)
        heavy_scratch[12:16] = sec_prev[12:16]
        for c in range(0, 60, 1):
            heavy_scratch[rspc.P_IDX[c][3]] ^= 0xFF
        self.assertFalse(rspc.is_frame_slip(58348, bytes(heavy_scratch)))

    def test_sqlite_debug_tables_and_timestamped_logger(self):
        ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}$")
        line_re = re.compile(r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\] ")
        ts = format_timestamp()
        self.assertRegex(ts, ts_re)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            db = RecoveryStateDB(tmp_dir / "state.sqlite3")
            logger = RecoveryLogger(tmp_dir / "logs", db=db, debug=True)
            logger.info("stage start")
            logger.debug("detail msg", lba=123)
            logger.log_event("optical", "slip_detected", lba=58348)
            db.record_pass_metric(58348, 0, ts, "media", 0, "FWD", 4234, 5, 12, "12597301", 1, 0, 4.2)
            db.record_round_log(ts, "media", 1, "FWD", 4234, 5, 10, 4, 1, 0, 1.2, 0.4, 1.6)
            db.record_solver_attempt(ts, 58348, 3, "single_c2", 1, 1, 86, 52, "A_SINGLE_C2", 12.5)
            db.conn.commit()
            logger.close()

            rec_lines = (tmp_dir / "logs" / "recovery.log").read_text(encoding="utf-8").splitlines()
            dbg_lines = (tmp_dir / "logs" / "debug.log").read_text(encoding="utf-8").splitlines()
            evt_lines = (tmp_dir / "logs" / "debug_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(rec_lines), 1)
            self.assertGreaterEqual(len(dbg_lines), 2)
            self.assertGreaterEqual(len(evt_lines), 1)
            for line in rec_lines + dbg_lines:
                self.assertRegex(line, line_re)

            for tbl in ("pass_metrics", "round_logs", "solver_attempts", "debug_events"):
                cnt = db.conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                self.assertGreaterEqual(cnt, 1, f"Expected rows in {tbl}")
            db.close()

            # Non-debug mode must still write timestamped recovery.log without debug.log
            quiet_logs = tmp_dir / "logs_quiet"
            logger2 = RecoveryLogger(quiet_logs, db=None, debug=False)
            logger2.info("quiet info line")
            logger2.debug("should not create debug.log")
            logger2.close()
            self.assertTrue((quiet_logs / "recovery.log").exists())
            self.assertFalse((quiet_logs / "debug.log").exists())
            for line in (quiet_logs / "recovery.log").read_text(encoding="utf-8").splitlines():
                self.assertRegex(line, line_re)

    def test_verify_physical_q_parity_accepts_valid_and_rejects_mismatched(self):
        from src.core.rspc import verify_physical_q_parity

        payload = bytes((i * 29 + 7) & 0xFF for i in range(2048))
        good = encode_mode1_sector(68310, payload)

        # Case A: 25 Q-parity bytes scratched & C2-flagged across passes, unflagged Q-par intact
        scratched_q = list(range(2248, 2273))
        c2_b = bytearray(294)
        for pos in scratched_q:
            c2_b[pos >> 3] |= 1 << (7 - (pos & 7))
        passes_a = []
        for idx in range(4):
            s = bytearray(good)
            for pos in scratched_q:
                s[pos] ^= (idx + 0x41) & 0xFF
            passes_a.append((bytes(s), bytes(c2_b)))
        ok_a, match_a, checked_a = verify_physical_q_parity(good, passes_a)
        self.assertTrue(ok_a)
        self.assertEqual(checked_a, 104 - len(scratched_q))
        self.assertEqual(match_a, checked_a)

        # Case B: 20 unflagged Q-parity bytes disagree with computed Q-parity
        passes_b = []
        bad_q = list(range(2248, 2268))
        for idx in range(4):
            s = bytearray(good)
            for pos in bad_q:
                s[pos] ^= 0x5A
            passes_b.append((bytes(s), b"\x00" * 294))
        ok_b, match_b, checked_b = verify_physical_q_parity(good, passes_b)
        self.assertFalse(ok_b)
        self.assertEqual(checked_b, 104)
        self.assertEqual(match_b, 84)

        # Case C: LBA 48194 pattern - Pass 0 has 90 unflagged errors while Pass 1 is exact with 24 conservative C2 flags
        noisy_p0 = bytearray(good)
        for pos in range(2248, 2338):
            noisy_p0[pos] ^= 0x5A
        c2_p1 = bytearray(294)
        for pos in range(2248, 2272):
            c2_p1[pos >> 3] |= 1 << (7 - (pos & 7))
        ok_c, match_c, checked_c = verify_physical_q_parity(
            good, [(bytes(noisy_p0), b"\x00" * 294), (bytes(good), bytes(c2_p1))]
        )
        self.assertTrue(ok_c)
        self.assertEqual(checked_c, 104)
        self.assertEqual(match_c, 104)

    def test_debug_logger_streams_to_console_only_when_debug_enabled(self):
        line_re = re.compile(r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\] ")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            buf_dbg = io.StringIO()
            logger_dbg = RecoveryLogger(tmp_dir / "logs_dbg", db=None, debug=True, stream=buf_dbg)
            logger_dbg.debug("solver step", lba=56007, strategy="PER_PLANE")
            logger_dbg.debug("file only", console=False, lba=56007)
            logger_dbg.close()

            console_lines = buf_dbg.getvalue().splitlines()
            self.assertEqual(len(console_lines), 1)
            self.assertRegex(console_lines[0], line_re)
            self.assertIn("[DEBUG] solver step | lba=56007 strategy=PER_PLANE", console_lines[0])
            self.assertNotIn("file only", buf_dbg.getvalue())

            dbg_file_text = (tmp_dir / "logs_dbg" / "debug.log").read_text(encoding="utf-8")
            self.assertIn("solver step", dbg_file_text)
            self.assertIn("file only", dbg_file_text)

            buf_quiet = io.StringIO()
            logger_quiet = RecoveryLogger(tmp_dir / "logs_quiet", db=None, debug=False, stream=buf_quiet)
            logger_quiet.info("visible info")
            logger_quiet.debug("hidden debug", lba=1)
            logger_quiet.close()

            quiet_out = buf_quiet.getvalue()
            self.assertIn("visible info", quiet_out)
            self.assertNotIn("hidden debug", quiet_out)
            self.assertNotIn("[DEBUG]", quiet_out)
            self.assertFalse((tmp_dir / "logs_quiet" / "debug.log").exists())


if __name__ == "__main__":
    unittest.main()
