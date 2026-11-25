"""End-to-end 6-stage optical recovery orchestrator and session workspace resolver."""

from src.pipeline.diagnostics import audit_session, inspect_image, inspect_optical_disc
from src.pipeline.orchestrator import (
    extract_and_repair_files,
    resolve_session_dir,
    run_automated_pipeline,
    run_synthetic_repair_self_test,
    sanitize_volume_id,
)
from src.pipeline.vcd_mode2 import run_mode2_pipeline

__all__ = [
    "audit_session",
    "extract_and_repair_files",
    "inspect_image",
    "inspect_optical_disc",
    "resolve_session_dir",
    "run_automated_pipeline",
    "run_mode2_pipeline",
    "run_synthetic_repair_self_test",
    "sanitize_volume_id",
]
