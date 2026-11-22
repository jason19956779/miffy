#!/usr/bin/env python3
"""Unified CLI entry point for automated optical CD ripping, RSPC+EDC solving, and format repair."""

import argparse
import json
import sys
from pathlib import Path

from src.formats import FormatRegistry
from src.pipeline import run_automated_pipeline, run_synthetic_repair_self_test
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
        default="photo_l,photo_s,pic,photo",
        help="Comma-separated directory/extension/alias filter or 'all'",
    )
    parser.add_argument("--max-passes", type=int, default=48, help="Maximum multi-tier optical pass rounds")
    parser.add_argument("--max-tier", type=int, default=3, help="Maximum algebraic solver tier (1..3)")
    parser.add_argument("--report-out", type=Path, default=None, help="Override summary JSON report path")
    parser.add_argument("--list-plugins", action="store_true", help="List all registered solver and format plugins and exit")
    parser.add_argument("--test-synthetic-repair", action="store_true", help="Run in-memory synthetic JPEG & BMP repair test and exit")
    args = parser.parse_args()

    if args.list_plugins:
        print_plugins()
        return 0

    if args.test_synthetic_repair:
        ok = run_synthetic_repair_self_test(REPO_DIR)
        print("Synthetic repair self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

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
        max_tier=args.max_tier,
        report_out=args.report_out,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
