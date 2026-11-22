"""Core dataclasses, ECMA-130 RSPC/EDC primitives, and SQLite/ISO persistence."""

from src.core.models import (
    DeviceCapabilities,
    EtiologyReport,
    FileExtent,
    FileRecoveryResult,
    SolvedSector,
)
from src.core.state import RecoveryStateDB, load_wincdemu_bad_lbas, prepare_output_iso

__all__ = [
    "DeviceCapabilities",
    "EtiologyReport",
    "FileExtent",
    "FileRecoveryResult",
    "SolvedSector",
    "RecoveryStateDB",
    "load_wincdemu_bad_lbas",
    "prepare_output_iso",
]
