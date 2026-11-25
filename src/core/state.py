"""SQLite pass-cache, ISO image creation/patching, and state JSON exporters."""

import json
import re
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

from src.core import rspc

RANGE_RE = re.compile(r"0x([0-9A-Fa-f]+)\s*-\s*0x([0-9A-Fa-f]+)\s*\((\d+)\s*sectors\)")


def load_wincdemu_bad_lbas(bad_log_path: Path) -> list[int]:
    """Parse optional WinCDEmu 0xSTART - 0xEND (N sectors) report into sorted LBAs."""
    lbas: list[int] = []
    with bad_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RANGE_RE.search(line.strip())
            if m:
                lbas.extend(range(int(m.group(1), 16), int(m.group(2), 16)))
    return sorted(set(lbas))


class RecoveryStateDB:
    """Thread/resume-safe SQLite store for raw passes and solved ISO sectors."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_passes (
                lba INTEGER NOT NULL,
                pass_id INTEGER NOT NULL,
                raw_2352 BLOB NOT NULL,
                c2_294 BLOB NOT NULL,
                p_ok INTEGER NOT NULL,
                q_ok INTEGER NOT NULL,
                PRIMARY KEY (lba, pass_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS solved_sectors (
                lba INTEGER PRIMARY KEY,
                user_2048 BLOB NOT NULL,
                verified INTEGER NOT NULL,
                best_p_ok INTEGER NOT NULL,
                best_q_ok INTEGER NOT NULL,
                edc_ok INTEGER NOT NULL,
                solved_pass INTEGER NOT NULL,
                strategy TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pass_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lba INTEGER NOT NULL,
                pass_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                phase TEXT NOT NULL,
                round_idx INTEGER NOT NULL,
                direction TEXT NOT NULL,
                speed_kbps INTEGER NOT NULL,
                p_ok INTEGER NOT NULL,
                q_ok INTEGER NOT NULL,
                msf_hex TEXT NOT NULL,
                msf_valid INTEGER NOT NULL,
                frame_slip INTEGER NOT NULL,
                solve_ms REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS round_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                phase TEXT NOT NULL,
                round_idx INTEGER NOT NULL,
                direction TEXT NOT NULL,
                speed_kbps INTEGER NOT NULL,
                lead INTEGER NOT NULL,
                remaining_before INTEGER NOT NULL,
                batch_count INTEGER NOT NULL,
                newly_verified INTEGER NOT NULL,
                frame_slips INTEGER NOT NULL,
                read_elapsed_s REAL NOT NULL,
                solve_elapsed_s REAL NOT NULL,
                total_elapsed_s REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS solver_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                lba INTEGER NOT NULL,
                max_tier INTEGER NOT NULL,
                solver_name TEXT NOT NULL,
                pass_count INTEGER NOT NULL,
                ok INTEGER NOT NULL,
                p_ok INTEGER NOT NULL,
                q_ok INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                elapsed_ms REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS debug_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                stage TEXT NOT NULL,
                event TEXT NOT NULL,
                lba INTEGER,
                payload_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_max_pass_ids(self) -> dict[int, int]:
        rows = self.conn.execute(
            "SELECT lba, COALESCE(MAX(pass_id), -1) FROM raw_passes GROUP BY lba"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_verified_set(self) -> set[int]:
        return {
            r[0]
            for r in self.conn.execute(
                "SELECT lba FROM solved_sectors WHERE verified = 1"
            ).fetchall()
        }

    def get_solved_map(self) -> dict[int, tuple[int, int, int, str]]:
        return {
            r[0]: (r[1], r[2], r[3], r[4])
            for r in self.conn.execute(
                "SELECT lba, verified, best_p_ok, best_q_ok, strategy FROM solved_sectors"
            ).fetchall()
        }

    def get_raw_passes(self, lba: int) -> list[tuple[bytes, bytes]]:
        rows = self.conn.execute(
            "SELECT raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
            (lba,),
        ).fetchall()
        return [(bytes(r[0]), bytes(r[1])) for r in rows]

    def get_pass_count(self, lba: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM raw_passes WHERE lba = ?", (lba,)
        ).fetchone()
        return row[0] if row else 0

    def all_tracked_bad_lbas(self) -> list[int]:
        lbas = {
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT lba FROM raw_passes UNION SELECT lba FROM solved_sectors"
            ).fetchall()
        }
        return sorted(lbas)

    def insert_raw_pass(
        self, lba: int, pass_id: int, raw_2352: bytes, c2_294: bytes
    ) -> tuple[int, int]:
        pz, qz = rspc.count_pq(raw_2352)
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_passes "
            "(lba, pass_id, raw_2352, c2_294, p_ok, q_ok) VALUES (?, ?, ?, ?, ?, ?)",
            (lba, pass_id, raw_2352, c2_294, pz, qz),
        )
        return pz, qz

    def record_pass_metric(
        self,
        lba: int,
        pass_id: int,
        ts: str,
        phase: str,
        round_idx: int,
        direction: str,
        speed_kbps: int,
        p_ok: int,
        q_ok: int,
        msf_hex: str,
        msf_valid: int,
        frame_slip: int,
        solve_ms: float,
    ) -> None:
        self.conn.execute(
            "INSERT INTO pass_metrics "
            "(lba, pass_id, ts, phase, round_idx, direction, speed_kbps, p_ok, q_ok, "
            "msf_hex, msf_valid, frame_slip, solve_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (lba, pass_id, ts, phase, round_idx, direction, speed_kbps, p_ok, q_ok, msf_hex, msf_valid, frame_slip, solve_ms),
        )

    def record_round_log(
        self,
        ts: str,
        phase: str,
        round_idx: int,
        direction: str,
        speed_kbps: int,
        lead: int,
        remaining_before: int,
        batch_count: int,
        newly_verified: int,
        frame_slips: int,
        read_elapsed_s: float,
        solve_elapsed_s: float,
        total_elapsed_s: float,
    ) -> None:
        self.conn.execute(
            "INSERT INTO round_logs "
            "(ts, phase, round_idx, direction, speed_kbps, lead, remaining_before, batch_count, "
            "newly_verified, frame_slips, read_elapsed_s, solve_elapsed_s, total_elapsed_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                phase,
                round_idx,
                direction,
                speed_kbps,
                lead,
                remaining_before,
                batch_count,
                newly_verified,
                frame_slips,
                read_elapsed_s,
                solve_elapsed_s,
                total_elapsed_s,
            ),
        )

    def record_solver_attempt(
        self,
        ts: str,
        lba: int,
        max_tier: int,
        solver_name: str,
        pass_count: int,
        ok: int,
        p_ok: int,
        q_ok: int,
        strategy: str,
        elapsed_ms: float,
    ) -> None:
        self.conn.execute(
            "INSERT INTO solver_attempts "
            "(ts, lba, max_tier, solver_name, pass_count, ok, p_ok, q_ok, strategy, elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, lba, max_tier, solver_name, pass_count, ok, p_ok, q_ok, strategy, elapsed_ms),
        )

    def record_debug_event(
        self, ts: str, stage: str, event: str, lba: int | None, payload: dict
    ) -> None:
        self.conn.execute(
            "INSERT INTO debug_events (ts, stage, event, lba, payload_json) VALUES (?, ?, ?, ?, ?)",
            (ts, stage, event, lba, json.dumps(payload, default=str)),
        )

    def upsert_solution(
        self,
        lba: int,
        sec: bytearray | bytes,
        verified: bool,
        pz: int,
        qz: int,
        pass_id: int,
        strategy: str,
        iso_fh=None,
    ) -> bool:
        existing = self.conn.execute(
            "SELECT verified, best_p_ok, best_q_ok FROM solved_sectors WHERE lba = ?",
            (lba,),
        ).fetchone()
        if existing and existing[0] == 1 and not verified:
            return True

        user_2048 = bytes(sec[16:2064]) if len(sec) >= 2064 else bytes(sec[:2048])
        edc_ok = (
            1
            if len(sec) >= 2068 and rspc.edc_residual(sec[:2068]) == 0
            else (1 if verified else 0)
        )

        if verified:
            self.conn.execute(
                "INSERT OR REPLACE INTO solved_sectors "
                "(lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy) "
                "VALUES (?, ?, 1, 86, 52, 1, ?, ?)",
                (lba, user_2048, pass_id, strategy),
            )
            if iso_fh is not None:
                iso_fh.seek(lba * 2048)
                iso_fh.write(user_2048)
            return True

        prev_score = (
            (-1, -1)
            if (not existing or existing[1] == 86)
            else rspc.partial_score(existing[1], existing[2])
        )
        if rspc.partial_score(pz, qz) > prev_score:
            self.conn.execute(
                "INSERT OR REPLACE INTO solved_sectors "
                "(lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy) "
                "VALUES (?, ?, 0, ?, ?, ?, ?, ?)",
                (lba, user_2048, pz, qz, edc_ok, pass_id, strategy),
            )
            if iso_fh is not None:
                iso_fh.seek(lba * 2048)
                iso_fh.write(user_2048)
        return False

    def export_state_json(self, state_path: Path, all_bad_lbas: list[int]) -> dict:
        rows = self.conn.execute(
            "SELECT lba, verified, best_p_ok, best_q_ok, strategy FROM solved_sectors"
        ).fetchall()
        by_lba = {r[0]: r for r in rows}
        verified_cnt = sum(1 for r in rows if r[1] == 1)
        partial_cnt = sum(1 for r in rows if r[1] == 0)
        untouched_cnt = max(0, len(all_bad_lbas) - len(rows))
        pass_rows = self.conn.execute(
            "SELECT lba, COUNT(*) FROM raw_passes GROUP BY lba"
        ).fetchall()
        max_pass_row = self.conn.execute("SELECT MAX(pass_id) FROM raw_passes").fetchone()
        strat_counts = Counter(r[4] for r in rows if r[1] == 1)
        state = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_bad_sectors": len(all_bad_lbas),
            "verified_sectors": verified_cnt,
            "partial_sectors": partial_cnt,
            "untouched_sectors": untouched_cnt,
            "verified_pct": round(100.0 * verified_cnt / max(1, len(all_bad_lbas)), 2),
            "total_raw_pass_reads": sum(r[1] for r in pass_rows),
            "max_pass_id": max_pass_row[0] if max_pass_row and max_pass_row[0] is not None else -1,
            "strategy_breakdown": dict(strat_counts),
            "unverified_sample": [
                {
                    "lba": lba,
                    "hex": f"0x{lba:05X}",
                    "p_ok": by_lba[lba][2] if lba in by_lba else 0,
                    "q_ok": by_lba[lba][3] if lba in by_lba else 0,
                    "passes": dict(pass_rows).get(lba, 0),
                }
                for lba in all_bad_lbas
                if lba not in by_lba or by_lba[lba][1] == 0
            ][:30],
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state


def prepare_output_iso(
    iso_out: Path,
    seed_iso: Path | None,
    total_volume_sectors: int | None = None,
) -> None:
    """Create or verify the writable output ISO without ever modifying seed_iso."""
    iso_out.parent.mkdir(parents=True, exist_ok=True)
    if not iso_out.exists():
        if seed_iso is not None and seed_iso.exists():
            if seed_iso.resolve() == iso_out.resolve():
                raise ValueError("seed_iso and iso_out must be distinct files to preserve baseline")
            shutil.copy2(seed_iso, iso_out)
        elif total_volume_sectors is not None:
            with iso_out.open("wb") as f:
                f.truncate(total_volume_sectors * 2048)
        else:
            raise FileNotFoundError(f"Neither {iso_out} nor seed_iso exists")

    if total_volume_sectors is not None:
        target_bytes = total_volume_sectors * 2048
        cur_bytes = iso_out.stat().st_size
        if cur_bytes < target_bytes:
            with iso_out.open("r+b") as f:
                f.truncate(target_bytes)
