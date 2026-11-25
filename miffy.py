#!/usr/bin/env python3
"""Unified CLI entry point for automated optical CD ripping, RSPC+EDC solving, and format repair."""

import argparse
import json
import sys
from pathlib import Path

from src.formats import FormatRegistry
from src.hardware.device import OpticalDrive
from src.pipeline import (
    audit_session,
    inspect_image,
    inspect_optical_disc,
    run_automated_pipeline,
    run_synthetic_repair_self_test,
)
from src.pipeline.diagnostics import (
    format_drive_status,
    format_image_or_disc_info,
    format_session_audit,
)
from src.solvers import SolverRegistry

REPO_DIR = Path(__file__).resolve().parent


def print_plugins() -> None:
    print("Registered Sector Recovery Solvers (src.solvers):")
    for s in SolverRegistry.list_plugins():
        print(f"  [tier {s.tier}] {s.name:28s} - {s.description}")

    print("\nRegistered File Format Handlers (src.formats):")
    for h in FormatRegistry.list_handlers():
        exts = ", ".join(h.extensions)
        print(f"  [prio {h.priority:2d}] {h.name:18s} ({exts}) - {h.description}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated modular CD optical ripper, RSPC+EDC solver, and format repair framework."
    )
    parser.add_argument("--device", default=None, help="SCSI generic device (auto-detects /dev/sgN if omitted)")
    parser.add_argument("--no-hardware", action="store_true", help="Run offline using cached SQLite passes and ISO")
    parser.add_argument(
        "--recovered-root",
        type=Path,
        default=REPO_DIR / "recovered",
        help="Root directory holding auto-named per-CD recovery folders (default: ./recovered)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional session folder name under --recovered-root (auto-uses CD Volume ID or UUID if omitted)",
    )
    parser.add_argument("--iso-out", type=Path, default=None, help="Override output repaired ISO path")
    parser.add_argument("--seed-iso", type=Path, default=None, help="Optional baseline read-only ISO to seed from")
    parser.add_argument("--bad-log", type=Path, default=None, help="Optional WinCDEmu bad-sector report")
    parser.add_argument("--full-rip", action="store_true", help="Force full pass-0 optical rip from disc")
    parser.add_argument("--db", type=Path, default=None, help="Override SQLite pass cache path")
    parser.add_argument("--extract-dir", type=Path, default=None, help="Override extracted files directory")
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated directory/extension/alias filter (audio,video,media,photo_l,photo_s,pic,photo) or 'all'",
    )
    parser.add_argument("--max-passes", type=int, default=48, help="Maximum multi-tier optical pass rounds")
    parser.add_argument("--focus-passes", type=int, default=24, help="Focused directional burst rounds when <=40 media LBAs remain")
    parser.add_argument("--max-tier", type=int, default=3, help="Maximum algebraic solver tier (1..3)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker processes for Tier 3 solving (0=auto, 1=serial)")
    parser.add_argument("--save-baseline", action="store_true", help="Preserve read-only (0444) Pass-0 baseline image (+ .cue)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose timestamped debug logs, JSONL events, and SQLite telemetry")
    parser.add_argument("--report-out", type=Path, default=None, help="Override summary JSON report path")
    parser.add_argument("--list-plugins", action="store_true", help="List all registered solver and format plugins and exit")
    parser.add_argument("--test-synthetic-repair", action="store_true", help="Run in-memory synthetic JPEG & BMP repair test and exit")
    parser.add_argument("--drive-status", action="store_true", help="Read-only SCSI Mode Page 0x01/0x2A drive status check and exit")
    parser.add_argument("--inspect-disc", action="store_true", help="Probe inserted optical disc TOC, mode, sample sectors, and PVD/files and exit")
    parser.add_argument("--inspect-image", type=Path, default=None, help="Inspect an offline .iso or Mode-2 .bin image and exit")
    parser.add_argument("--audit-session", type=Path, default=None, help="Audit a recovered session directory (image, SQLite, Q-parity, file formats) and exit")
    parser.add_argument("--compare-session", type=Path, default=None, help="Optional second session directory to compare SHA-256 parity during --audit-session")
    parser.add_argument("--ffmpeg", action="store_true", help="Run ffmpeg -v error -xerror + ffprobe on audio/video files during --audit-session")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON for diagnostic commands")
    args = parser.parse_args()

    if args.list_plugins:
        print_plugins()
        return 0

    if args.test_synthetic_repair:
        ok = run_synthetic_repair_self_test(REPO_DIR)
        print("Synthetic repair self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.drive_status:
        status = OpticalDrive.read_drive_status(args.device)
        print(json.dumps(status, indent=2) if args.json else format_drive_status(status))
        return 0 if status["mode_page_01"]["restored_ok"] else 1

    if args.inspect_disc:
        info = inspect_optical_disc(args.device)
        print(json.dumps(info, indent=2) if args.json else format_image_or_disc_info(info, is_disc=True))
        return 0

    if args.inspect_image is not None:
        info = inspect_image(args.inspect_image)
        print(json.dumps(info, indent=2) if args.json else format_image_or_disc_info(info, is_disc=False))
        return 0

    if args.audit_session is not None:
        audit = audit_session(
            args.audit_session,
            run_ffmpeg=args.ffmpeg,
            compare_session=args.compare_session,
        )
        print(json.dumps(audit, indent=2) if args.json else format_session_audit(audit))
        val = audit["validation"]
        qp = audit["q_parity_gate"]
        cmp_d = audit.get("compare", {})
        ok_all = val["failed"] == 0 and qp["failed"] == 0 and not cmp_d.get("mismatches")
        return 0 if ok_all else 1

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    report = run_automated_pipeline(
        device=args.device,
        no_hardware=args.no_hardware,
        recovered_root=args.recovered_root,
        session_name=args.name,
        iso_out=args.iso_out,
        seed_iso=args.seed_iso,
        bad_log=args.bad_log,
        full_rip=args.full_rip,
        db_path=args.db,
        extract_dir=args.extract_dir,
        categories=categories,
        max_passes=args.max_passes,
        focus_passes=args.focus_passes,
        max_tier=args.max_tier,
        workers=args.workers,
        debug=args.debug,
        report_out=args.report_out,
        save_baseline=args.save_baseline,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
