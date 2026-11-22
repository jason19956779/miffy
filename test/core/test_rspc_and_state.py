"""Unit tests for ECMA-130 RSPC P/Q/EDC primitives and SQLite RecoveryStateDB."""

import tempfile
import unittest
from pathlib import Path

from src.core import rspc
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


if __name__ == "__main__":
    unittest.main()
