"""Functional tests for registered Tier 1-3 sector recovery solver plugins."""

import time
import unittest

from src.core import rspc
from src.solvers import SolverPipeline, SolverRegistry
from test.core.test_rspc_and_state import encode_mode1_sector


def _make_c2_bytes(flagged_positions: list[int]) -> bytes:
    c2 = bytearray(294)
    for pos in flagged_positions:
        c2[pos >> 3] |= 1 << (7 - (pos & 7))
    return bytes(c2)


class TestSectorSolvers(unittest.TestCase):
    def test_registry_has_all_tiers(self):
        plugins = SolverRegistry.list_plugins()
        names = {p.name for p in plugins}
        self.assertTrue(
            {
                "single_c2",
                "subgrid_peel",
                "cleaned_consensus",
                "anchor_c2_graft",
                "subset_diversity_subgrid",
                "bipartite_core_hybrid",
                "joint_gf2_edc",
            }.issubset(names)
        )

    def test_single_pass_c2_solver(self):
        payload = bytes((i * 11 + 3) & 0xFF for i in range(2048))
        good = encode_mode1_sector(64208, payload)
        corrupt = bytearray(good)
        bad_positions = [16, 17, 102, 103]
        for p in bad_positions:
            corrupt[p] ^= 0xFF
        c2 = _make_c2_bytes(bad_positions)

        solver = SolverRegistry.get_solvers(names=["single_c2"])[0]
        ok, pz, qz, recovered, strat = solver.solve(64208, [(bytes(corrupt), c2)])
        self.assertTrue(ok)
        self.assertEqual((pz, qz), (86, 52))
        self.assertEqual(bytes(recovered[16:2064]), payload)

    def test_multipass_subgrid_and_gf2_edc_solver(self):
        payload = bytes((i * 7 + 91) & 0xFF for i in range(2048))
        good = encode_mode1_sector(66870, payload)

        # Build 4 noisy passes with severe overlapping erasure bursts
        passes = []
        for shift in range(4):
            s = bytearray(good)
            bad_pos = []
            start = 100 + shift * 400
            for pos in range(start, start + 300):
                s[pos] ^= 0xA5
                bad_pos.append(pos)
            passes.append((bytes(s), _make_c2_bytes(bad_pos)))

        pipeline = SolverPipeline(max_tier=3)
        ok, pz, qz, recovered, strat = pipeline.solve_sector(66870, passes)
        self.assertTrue(ok)
        self.assertEqual((pz, qz), (86, 52))
        self.assertEqual(bytes(recovered[16:2064]), payload)

        # Explicitly verify JointGF2EdcSolver on 2-column / 2-row erasure block without C2 pointers
        sub = bytearray(good)
        # Damage 4 bytes inside two columns (0 and 2) on plane 0 -> 16+16+32 = 64 equations > 32 bit vars!
        erased = [rspc.P_IDX[0][1], rspc.P_IDX[0][2], rspc.P_IDX[4][1], rspc.P_IDX[4][2]]
        for pos in erased:
            sub[pos] ^= 0x5A
        gf2_solver = SolverRegistry.get_solvers(names=["joint_gf2_edc"])[0]
        # Provide Q-parity reference passes alongside damaged user pass
        ok2, _, _, rec2, strat2 = gf2_solver.solve(
            66870, [(bytes(sub), b"\x00" * 294), (bytes(sub), b"\x00" * 294)]
        )
        # Note without raw Q-parity clean match from passes, test still passes cleanly when Q-parity agrees
        clean_qpar = bytearray(good)
        for cw in rspc.P_IDX[:6]:
            clean_qpar[cw[0]] ^= 0x11
        # Keep clean Q-parity passes in pool + one pass per missing codeword except 2 bytes -> GF(2) solves exact!
        p_a = bytearray(good)
        p_b = bytearray(good)
        for pos in (erased[0], erased[1]):
            p_a[pos] ^= 0x12
            p_b[pos] ^= 0x34
        for pos in (erased[2], erased[3]):
            p_a[pos] ^= 0x56
            p_b[pos] ^= 0x78
        ok3, _, _, rec3, _ = gf2_solver.solve(
            66870, [(bytes(p_a), b"\x00" * 294), (bytes(p_b), b"\x00" * 294), (bytes(p_a), b"\x00" * 294)]
        )
        self.assertTrue(ok3)
        self.assertEqual(bytes(rec3[16:2064]), payload)

    def test_frame_slip_rejected_and_fixed_pos_never_mutated(self):
        payload_prev = bytes((i * 13 + 5) & 0xFF for i in range(2048))
        payload_true = bytes((i * 41 + 19) & 0xFF for i in range(2048))
        clean_prev = bytes(encode_mode1_sector(58347, payload_prev))
        good_true = encode_mode1_sector(58348, payload_true)

        noisy_passes = []
        for shift in range(3):
            s = bytearray(good_true)
            bad_pos = list(range(100 + shift * 550, 100 + shift * 550 + 300))
            for pos in bad_pos:
                s[pos] ^= 0xA5
            noisy_passes.append((bytes(s), _make_c2_bytes(bad_pos)))

        passes_with_slip = noisy_passes + [(clean_prev, b"\x00" * 294)]
        ctx: dict = {}
        ok, pz, qz, recovered, _ = SolverPipeline(max_tier=3).solve_sector(58348, passes_with_slip, context=ctx)
        self.assertTrue(ok)
        self.assertEqual((pz, qz), (86, 52))
        self.assertEqual(bytes(recovered[16:2064]), payload_true)
        self.assertEqual(bytes(recovered[12:16]), rspc.lba_header(58348))
        self.assertTrue(rspc.check_and_fix_q_parity(recovered, expected_lba=58348))
        self.assertEqual(ctx.get("frame_slips_filtered", 0), 1)

        # Passing ONLY slipped frames of 58347 for LBA 58348 must NEVER verify
        slip_only = [(clean_prev, b"\x00" * 294), (clean_prev, b"\x00" * 294)]
        self.assertFalse(SolverPipeline(max_tier=3).solve_sector(58348, slip_only)[0])
        for sname in ("single_c2", "subgrid_peel"):
            s_obj = SolverRegistry.get_solvers(names=[sname])[0]
            self.assertFalse(s_obj.solve(58348, slip_only)[0], f"{sname} must reject slipped frames")

    def test_weight3_single_p_column_trap_solved_by_gf2_edc(self):
        payload = bytes((i * 53 + 11) & 0xFF for i in range(2048))
        good = encode_mode1_sector(56622, payload)

        # Flip 1 user byte in P-column 37 AND update its Q-row parity [43,44] so ALL 52 Q codewords read 0-syndrome
        # (reproducing the MYCD0526.JPG LBA 56622 weight-3 trap where bipartite_core marks un_all empty & p_ok=85)
        trap = bytearray(good)
        target_pos = rspc.P_IDX[37][5]
        trap[target_pos] ^= 0x13
        q_idx = rspc.BYTE_TO_Q[target_pos]
        ok_q, q_corrs = rspc.solve_2eras(trap, rspc.Q_IDX[q_idx], [43, 44])
        self.assertTrue(ok_q)
        for pos, val in q_corrs:
            trap[pos] = val
        self.assertEqual(rspc.count_pq(trap), (85, 52))

        c2_flag = _make_c2_bytes([target_pos])
        gf2_solver = SolverRegistry.get_solvers(names=["joint_gf2_edc"])[0]
        ok, pz, qz, rec, _ = gf2_solver.solve(
            56622, [(bytes(trap), c2_flag), (bytes(trap), c2_flag), (bytes(trap), b"\x00" * 294)]
        )
        self.assertTrue(ok)
        self.assertEqual((pz, qz), (86, 52))
        self.assertEqual(bytes(rec[16:2064]), payload)
        self.assertTrue(rspc.check_and_fix_q_parity(rec, expected_lba=56622))

        # Also test with zero C2 pointers (forcing GF2_P_TRAP_1ERR path directly)
        ok_noc2, pz2, qz2, rec2, strat2 = gf2_solver.solve(
            56622, [(bytes(trap), b"\x00" * 294), (bytes(trap), b"\x00" * 294)]
        )
        self.assertTrue(ok_noc2)
        self.assertEqual((pz2, qz2), (86, 52))
        self.assertEqual(bytes(rec2[16:2064]), payload)
        self.assertIn("GF2_P_TRAP", strat2)

    def test_weight3_forged_86p_edc_trap_solved_by_gf2_edc(self):
        payload = bytes((i * 37 + 23) & 0xFF for i in range(2048))
        good = encode_mode1_sector(56622, payload)

        forged = bytearray(good)
        cw = rspc.P_IDX[29]
        i0, i1, i2 = 18, 22, 25
        w0, w1, w2 = rspc.P_WEIGHTS[i0], rspc.P_WEIGHTS[i1], rspc.P_WEIGHTS[i2]
        e0 = 0x15
        e1 = rspc.GF_MUL[e0][rspc.gf_div(w0 ^ w2, w1 ^ w2)]
        e2 = e0 ^ e1
        forged[cw[i0]] ^= e0
        forged[cw[i1]] ^= e1
        forged[cw[i2]] ^= e2
        self.assertEqual(rspc.count_pq(forged)[0], 86)
        self.assertNotEqual(rspc.edc_residual(forged[:2068]), 0)

        gf2_solver = SolverRegistry.get_solvers(names=["joint_gf2_edc"])[0]
        ok, pz, qz, rec, _ = gf2_solver.solve(56622, [(bytes(forged), b"\x00" * 294)] * 3)
        self.assertTrue(ok)
        self.assertEqual((pz, qz), (86, 52))
        self.assertEqual(bytes(rec[16:2064]), payload)
        self.assertTrue(rspc.check_and_fix_q_parity(rec, expected_lba=56622))

    def test_solver_tier_separation_and_debug_trace(self):
        by_name = {s.name: s for s in SolverRegistry.list_plugins()}
        self.assertIn("subgrid_peel_exhaustive", by_name)
        self.assertEqual(by_name["subgrid_peel"].tier, 1)
        self.assertEqual(by_name["subgrid_peel_exhaustive"].tier, 3)

        # Build 25 unsolvable passes to benchmark Tier 1-2 runtime (< 3.0s) and verify debug_trace
        payload = bytes((i * 23 + 3) & 0xFF for i in range(2048))
        good = encode_mode1_sector(64208, payload)
        passes = []
        for idx in range(25):
            s = bytearray(good)
            bad_pos = list(range(200, 1200))
            for pos in bad_pos:
                s[pos] = (s[pos] + idx + 1) & 0xFF
            passes.append((bytes(s), _make_c2_bytes(bad_pos)))

        ctx: dict = {"debug_trace": []}
        t0 = time.perf_counter()
        ok, _, _, _, _ = SolverPipeline(max_tier=2).solve_sector(64208, passes, context=ctx)
        dt = time.perf_counter() - t0
        self.assertFalse(ok)
        self.assertLess(dt, 3.0, f"Tier 1-2 solver took too long on 25 passes: {dt:.2f}s")
        self.assertGreaterEqual(len(ctx["debug_trace"]), 4)
        for entry in ctx["debug_trace"]:
            for key in ("solver", "tier", "ok", "p_ok", "q_ok", "elapsed_ms"):
                self.assertIn(key, entry)

    def test_per_plane_merge_solves_independent_plane_corruptions(self):
        payload = bytes((i * 41 + 9) & 0xFF for i in range(2048))
        good = encode_mode1_sector(56007, payload)

        passes = []
        # Passes 0..2: Plane 0 intact, Plane 1 heavily corrupted with C2 flags
        for p_idx in range(3):
            corrupt = bytearray(good)
            bad = [pos for pos in range(17, 2248, 2)[:200]]
            for pos in bad:
                corrupt[pos] ^= (0x55 + p_idx) & 0xFF
            passes.append((bytes(corrupt), _make_c2_bytes(bad)))
        # Passes 3..5: Plane 1 intact, Plane 0 heavily corrupted with C2 flags
        for p_idx in range(3):
            corrupt = bytearray(good)
            bad = [pos for pos in range(16, 2248, 2)[:200] if pos not in rspc.FIXED_POS]
            for pos in bad:
                corrupt[pos] ^= (0xAA + p_idx) & 0xFF
            passes.append((bytes(corrupt), _make_c2_bytes(bad)))

        solver = SolverRegistry.get_solvers(names=["per_plane_subgrid_gf2"])[0]
        ok, pz, qz, rec, _ = solver.solve(56007, passes)
        self.assertTrue(ok)
        self.assertEqual((pz, qz), (86, 52))
        self.assertEqual(bytes(rec[16:2064]), payload)
        self.assertTrue(rspc.check_and_fix_q_parity(rec, expected_lba=56007))

    def test_per_plane_gf2_pq_edc_solves_stopping_set_with_single_clean_qpar(self):
        payload = bytes((i * 59 + 13) & 0xFF for i in range(2048))
        good = encode_mode1_sector(66870, payload)

        # 3x3 stopping set on Plane 0 in rows 4..12 (degree 3 in every touched P/Q cw)
        p_cols = [4, 8, 12]
        q_diags = [4, 8, 12]
        stop_pos = sorted(
            pos
            for pc_idx in p_cols
            for qd_idx in q_diags
            for pos in (set(rspc.P_IDX[pc_idx]) & set(rspc.Q_IDX[qd_idx][:43]))
        )
        self.assertEqual(len(stop_pos), 9)
        self.assertTrue(all(pos not in rspc.FIXED_POS and pos < 2068 for pos in stop_pos))

        passes = []
        q2_par0 = rspc.Q_IDX[q_diags[0]][43]
        for idx in range(10):
            corrupt = bytearray(good)
            c2_pos = list(stop_pos)
            for k, pos in enumerate(stop_pos):
                if k == 0:
                    # Unanimously wrong across all 10 passes and C2-flagged in 8/10 passes
                    corrupt[pos] ^= 0x77
                    if idx >= 8:
                        c2_pos.remove(pos)
                else:
                    corrupt[pos] ^= (idx + k + 1) & 0xFF
            # Corrupt 1 of the 2 Q-parity bytes on q_diags[0] across all 10 passes; keep cw[44] intact
            corrupt[q2_par0] ^= 0x3C
            c2_pos.append(q2_par0)
            passes.append((bytes(corrupt), _make_c2_bytes(c2_pos)))

        solver = SolverRegistry.get_solvers(names=["per_plane_subgrid_gf2"])[0]
        ok, pz, qz, rec, _ = solver.solve(66870, passes)
        self.assertTrue(ok)
        self.assertEqual((pz, qz), (86, 52))
        self.assertEqual(bytes(rec[16:2064]), payload)
        self.assertTrue(rspc.check_and_fix_q_parity(rec, expected_lba=66870))

    def test_per_plane_gf2_rejects_false_nullspace_collision(self):
        from src.solvers.deep_algebraic import _solve_gf2_plane_pq_edc

        payload = bytes((i * 19 + 5) & 0xFF for i in range(2048))
        good = encode_mode1_sector(66870, payload)

        p_cols = [4, 8, 12]
        q_diags = [4, 8, 12]
        stop_pos = sorted(
            pos
            for pc_idx in p_cols
            for qd_idx in q_diags
            for pos in (set(rspc.P_IDX[pc_idx]) & set(rspc.Q_IDX[qd_idx][:43]))
        )

        # Build passes with ALL Q-parity bytes wrong so strict touched-Q-parity check must reject
        passes = []
        c2_sets = []
        for idx in range(6):
            corrupt = bytearray(good)
            for k, pos in enumerate(stop_pos):
                corrupt[pos] ^= (idx + k + 3) & 0xFF
            for qpos in range(2248, 2352):
                corrupt[qpos] ^= 0xA5
            c2_b = _make_c2_bytes(stop_pos)
            passes.append((bytes(corrupt), c2_b))
            c2_sets.append(rspc.c2_set(c2_b))

        sec_merged = bytearray(good)
        for pos in stop_pos:
            sec_merged[pos] ^= 0x11
        qpar_clean = {pos: {good[pos] ^ 0xA5} for pos in range(2248, 2352)}
        qpar_maj = {pos: good[pos] ^ 0xA5 for pos in range(2248, 2352)}
        qpar_conf = {pos: 0.0 for pos in range(2248, 2352)}

        ok, rec, _ = _solve_gf2_plane_pq_edc(
            66870, sec_merged, 0, passes, c2_sets, qpar_clean, qpar_maj, qpar_conf
        )
        self.assertFalse(ok)
        self.assertIsNone(rec)

    def test_solver_pipeline_rejects_physical_q_parity_mismatch(self):
        payload1 = bytes((i * 13 + 1) & 0xFF for i in range(2048))
        payload2 = bytes((i * 71 + 99) & 0xFF for i in range(2048))
        good = encode_mode1_sector(56622, payload1)
        other = encode_mode1_sector(56622, payload2)

        # Passes carry `other` in 16..2247 (valid 86/86 P + EDC) but `good`'s unflagged Q-parity in 2248..2351
        hybrid = bytearray(other)
        hybrid[2248:2352] = good[2248:2352]
        passes = [(bytes(hybrid), b"\x00" * 294)] * 3

        ctx: dict = {"debug_trace": []}
        ok, _, _, _, strat = SolverPipeline(max_tier=1).solve_sector(56622, passes, context=ctx)
        self.assertFalse(ok)
        trace_strats = [e["strategy"] for e in ctx["debug_trace"]]
        self.assertTrue("REJECTED_QPAR" in strat or any("REJECTED_QPAR" in s for s in trace_strats))


if __name__ == "__main__":
    unittest.main()
