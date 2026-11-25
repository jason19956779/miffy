"""Timestamped console, file, JSONL, and SQLite structured logging for CD recovery."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def format_timestamp(dt: datetime | None = None) -> str:
    """Format local wall-clock time as ISO-8601 `YYYY-MM-DDTHH:MM:SS.mmm`."""
    cur = dt if dt is not None else datetime.now()
    return cur.strftime("%Y-%m-%dT%H:%M:%S.") + f"{cur.microsecond // 1000:03d}"


class RecoveryLogger:
    """Writes ISO-8601 timestamped logs to stdout, `recovery.log`, and optional `--debug` sinks."""

    def __init__(
        self,
        logs_dir: Path,
        db: Any = None,
        debug: bool = False,
        stream=None,
    ):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.debug_enabled = bool(debug)
        self.stream = sys.stdout if stream is None else stream

        self.recovery_log_path = self.logs_dir / "recovery.log"
        self._rec_fh = self.recovery_log_path.open("a", encoding="utf-8")

        self.debug_log_path = self.logs_dir / "debug.log" if self.debug_enabled else None
        self.events_jsonl_path = self.logs_dir / "debug_events.jsonl" if self.debug_enabled else None
        self._dbg_fh = self.debug_log_path.open("a", encoding="utf-8") if self.debug_log_path else None
        self._evt_fh = self.events_jsonl_path.open("a", encoding="utf-8") if self.events_jsonl_path else None

    def close(self) -> None:
        for fh in (self._rec_fh, self._dbg_fh, self._evt_fh):
            if fh is not None:
                fh.flush()
                fh.close()
        self._rec_fh = None
        self._dbg_fh = None
        self._evt_fh = None

    def _fmt_kwargs(self, kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return ""
        parts = [f"{k}={kwargs[k]}" for k in sorted(kwargs)]
        return " | " + " ".join(parts)

    def info(self, msg: str, console: bool = True, **kwargs: Any) -> str:
        ts = format_timestamp()
        line = f"[{ts}] {msg}{self._fmt_kwargs(kwargs)}"
        if console and self.stream is not None:
            print(line, file=self.stream, flush=True)
        if self._rec_fh is not None:
            self._rec_fh.write(line + "\n")
            self._rec_fh.flush()
        if self._dbg_fh is not None:
            self._dbg_fh.write(f"[{ts}] [INFO] {msg}{self._fmt_kwargs(kwargs)}\n")
            self._dbg_fh.flush()
        return ts

    def debug(self, msg: str, console: bool | None = None, **kwargs: Any) -> str:
        ts = format_timestamp()
        line = f"[{ts}] [DEBUG] {msg}{self._fmt_kwargs(kwargs)}"
        if self._dbg_fh is not None:
            self._dbg_fh.write(line + "\n")
            self._dbg_fh.flush()
        to_console = self.debug_enabled if console is None else bool(console)
        if to_console and self.stream is not None:
            print(line, file=self.stream, flush=True)
        return ts

    def log_event(
        self,
        stage: str,
        event: str,
        lba: int | None = None,
        persist_db: bool = True,
        **payload: Any,
    ) -> str:
        ts = format_timestamp()
        rec = {"ts": ts, "stage": stage, "event": event, "lba": lba, **payload}
        if self._dbg_fh is not None:
            self._dbg_fh.write(f"[{ts}] [EVENT] {stage}.{event} lba={lba}{self._fmt_kwargs(payload)}\n")
            self._dbg_fh.flush()
        if self._evt_fh is not None:
            self._evt_fh.write(json.dumps(rec, default=str) + "\n")
            self._evt_fh.flush()
        if persist_db and self.db is not None and (self.debug_enabled or event in ("frame_slip", "stage_complete")):
            self.db.record_debug_event(ts, stage, event, lba, rec)
        return ts
