"""Six-stage modular CD optical ripping, triage, RSPC solving, and format recovery pipeline."""

import csv
import io
import json
import re
import time
import uuid
from pathlib import Path

from PIL import Image

from src.core import rspc
from src.core.models import FileExtent, FileRecoveryResult
from src.core.state import RecoveryStateDB, load_wincdemu_bad_lbas, prepare_output_iso
from src.filesystem import iso9660
from src.formats import FormatRegistry
from src.hardware.device import OpticalDrive
from src.solvers import SolverPipeline, SolverRegistry

SPEEDS_MULTI = (4234, 2822, 2117, 1411, 1058, 706, 353, 176)
LEADS_WIDE = (2, 5, 8, 11, 14, 18, 22, 26, 30, 34, 38)


def sanitize_volume_id(vol_id: str) -> str:
    """Convert an ISO-9660 volume identifier (e.g. '_________MIFFY') to a clean folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", vol_id.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned


def resolve_session_dir(
    recovered_root: Path,
    session_name: str | None = None,
    device: str | None = None,
    no_hardware: bool = False,
    seed_iso: Path | None = None,
    iso_out: Path | None = None,
) -> tuple[str, Path]:
    """Determine per-CD session directory under `recovered/` from --name, PVD volume ID, or UUID."""
    resolved = sanitize_volume_id(session_name) if session_name else ""

    if not resolved:
        for candidate_iso in (iso_out, seed_iso):
            if candidate_iso is not None and candidate_iso.exists():
                reader = iso9660.make_iso_sector_reader(candidate_iso)
                try:
                    pvd = iso9660.parse_pvd(reader)
                    resolved = sanitize_volume_id(pvd.get("volume_id", ""))
                    if resolved:
                        break
                except Exception:
                    pass
                finally:
                    reader.fh.close()  # type: ignore[attr-defined]

    if not resolved and not no_hardware:
        try:
            with OpticalDrive(device) as drive:
                pvd_sec = drive.read_cooked_sector(16)
                if pvd_sec and len(pvd_sec) >= 72 and pvd_sec[1:6] == b"CD001":
                    resolved = sanitize_volume_id(pvd_sec[40:72].decode("ascii", errors="ignore"))
        except Exception:
            pass

    if not resolved:
        resolved = f"cd_{uuid.uuid4().hex[:8]}"

    session_dir = recovered_root / resolved
    return resolved, session_dir


def _match_category(ext_obj: FileExtent, categories: list[str]) -> bool:
    if not categories or "all" in [c.lower() for c in categories]:
        return True
    d_up = ext_obj.directory.upper()
    alias_map = {
        "photo_l": lambda e: d_up == "/PHOTO/L_PHOTO",
        "photo_s": lambda e: d_up == "/PHOTO/S",
        "photo": lambda e: d_up.startswith("/PHOTO"),
        "pic": lambda e: d_up == "/PIC",
        "xtras": lambda e: d_up == "/XTRAS",
        "root": lambda e: d_up == "/",
        "images": lambda e: e.ext in iso9660.PHOTO_MAIN_EXTS,
    }
    for cat in categories:
        fn = alias_map.get(cat.lower())
        if fn and fn(ext_obj):
            return True
        if d_up.startswith(cat.upper()) or ext_obj.ext.lower().lstrip(".") == cat.lower().lstrip("."):
            return True
    return False


def _build_batches(target_lbas: list[int], lead: int, max_batch: int, total_sectors: int) -> list[tuple[int, int, list[int]]]:
    batches = []
    i = 0
    n = len(target_lbas)
    while i < n:
        s_lba = max(0, target_lbas[i] - lead)
        if s_lba + max_batch > total_sectors:
            s_lba = max(0, total_sectors - max_batch)
        cnt = min(max_batch, total_sectors - s_lba)
        end_lim = s_lba + cnt - 2
        covered = []
        while i < n and target_lbas[i] <= end_lim:
            if target_lbas[i] >= s_lba:
                covered.append(target_lbas[i])
            i += 1
        if not covered and i < n:
            covered.append(target_lbas[i])
            i += 1
        batches.append((s_lba, cnt, covered))
    return batches


def pass0_full_rip(
    drive: OpticalDrive,
    db: RecoveryStateDB,
    iso_out: Path,
    total_sectors: int,
) -> list[int]:
    """Stream the entire CD once with DCR=1 + C2 pointers to build ISO and discover bad LBAs."""
    bad_lbas: list[int] = []
    pipeline = SolverPipeline(max_tier=1)
    max_batch = drive.max_batch_sectors
    t0 = time.time()

    with iso_out.open("r+b") as iso_fh:
        lba = 0
        while lba < total_sectors:
            cnt = min(max_batch, total_sectors - lba)
            st, _, resid, raw = drive.read_cd_raw(lba, cnt, with_c2=True)
            if st != 0 or resid != 0:
                # Bisect to single-sector reads on SCSI error
                for sub_lba in range(lba, lba + cnt):
                    st2, _, res2, raw2 = drive.read_cd_raw(sub_lba, 1, with_c2=True)
                    if st2 != 0 or res2 != 0:
                        bad_lbas.append(sub_lba)
                        continue
                    _handle_pass0_sector(sub_lba, raw2[:2352], raw2[2352:2646], db, pipeline, iso_fh, bad_lbas)
            else:
                for i in range(cnt):
                    off = i * 2646
                    _handle_pass0_sector(
                        lba + i,
                        raw[off : off + 2352],
                        raw[off + 2352 : off + 2646],
                        db,
                        pipeline,
                        iso_fh,
                        bad_lbas,
                    )
            lba += cnt
            if lba % (max_batch * 80) == 0 or lba >= total_sectors:
                db.conn.commit()
                iso_fh.flush()
                print(
                    f"  [Pass-0 Rip] {lba}/{total_sectors} sectors "
                    f"({100.0 * lba / total_sectors:.1f}%) | bad_discovered={len(bad_lbas)} | "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
        db.conn.commit()
        iso_fh.flush()
    return sorted(set(bad_lbas))


def _handle_pass0_sector(
    lba: int,
    raw_2352: bytes,
    c2_294: bytes,
    db: RecoveryStateDB,
    pipeline: SolverPipeline,
    iso_fh,
    bad_lbas: list[int],
) -> None:
    sec = bytearray(raw_2352)
    if rspc.check_and_fix_q_parity(sec):
        iso_fh.seek(lba * 2048)
        iso_fh.write(sec[16:2064])
        return
    # Fast single-pass C2 clean before flagging sector as bad
    ok, pz, qz, decoded, strat = pipeline.solve_sector(lba, [(raw_2352, c2_294)], max_tier=1)
    if ok:
        iso_fh.seek(lba * 2048)
        iso_fh.write(decoded[16:2064])
        return
    bad_lbas.append(lba)
    db.insert_raw_pass(lba, 0, raw_2352, c2_294)
    db.upsert_solution(lba, decoded, False, pz, qz, 0, strat, iso_fh=iso_fh)


def run_tiered_optical_recovery(
    drive: OpticalDrive,
    db: RecoveryStateDB,
    iso_out: Path,
    target_lbas: list[int],
    total_sectors: int,
    max_passes: int,
    max_tier: int,
) -> None:
    """Escalate across Tier 1 (FWD streaming), Tier 2 (speed+FWD/REV seeks), and Tier 3 (deep GF2/subset)."""
    if not target_lbas:
        return
    pipeline = SolverPipeline(max_tier=max_tier)
    pass_counts = db.get_max_pass_ids()

    with iso_out.open("r+b") as iso_fh:
        for round_idx in range(max_passes):
            verified = db.get_verified_set()
            remaining = [lba for lba in target_lbas if lba not in verified]
            if not remaining:
                print("All target sectors mathematically verified.", flush=True)
                break

            next_pid = min((pass_counts.get(lba, -1) + 1 for lba in remaining), default=round_idx)
            # Determine optical tier by pass count
            if next_pid < 6:
                direction = "FWD"
                kbps = 0xFFFF
                lead = 2 + ((next_pid * 3) % 15)
                tier_now = min(2, max_tier)
            else:
                direction = "FWD" if (next_pid % 2 == 0) else "REV"
                kbps = SPEEDS_MULTI[(next_pid - 6) % len(SPEEDS_MULTI)]
                lead = LEADS_WIDE[(next_pid * 5) % len(LEADS_WIDE)]
                tier_now = min(2, max_tier)

            drive.set_speed(kbps)
            t_r = time.time()
            newly_ok = 0

            # Use focused per-LBA directional seeks when <= 16 targets remain, else batch streaming
            if len(remaining) <= 16:
                for lba in remaining:
                    drive.directional_seek_flush(lba, direction, total_sectors, offset_seed=next_pid)
                    s_lba = max(0, lba - lead)
                    cnt = min(drive.max_batch_sectors, max(lead + 4, 4), total_sectors - s_lba)
                    st, _, res, raw = drive.read_cd_raw(s_lba, cnt, with_c2=True)
                    if st != 0 or res != 0:
                        continue
                    off = (lba - s_lba) * 2646
                    pid = pass_counts.get(lba, -1) + 1
                    pass_counts[lba] = pid
                    db.insert_raw_pass(lba, pid, raw[off : off + 2352], raw[off + 2352 : off + 2646])
                    passes = db.get_raw_passes(lba)
                    ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=tier_now)
                    if db.upsert_solution(lba, sec, ok, pz, qz, pid, strat, iso_fh=iso_fh):
                        newly_ok += 1
            else:
                batches = _build_batches(remaining, lead, drive.max_batch_sectors, total_sectors)
                if len(batches) < 20:
                    drive.directional_seek_flush(remaining[0], direction, total_sectors, offset_seed=next_pid)
                for s_lba, cnt, covered in batches:
                    st, _, res, raw = drive.read_cd_raw(s_lba, cnt, with_c2=True)
                    if st != 0 or res != 0:
                        continue
                    for lba in covered:
                        off = (lba - s_lba) * 2646
                        pid = pass_counts.get(lba, -1) + 1
                        pass_counts[lba] = pid
                        db.insert_raw_pass(lba, pid, raw[off : off + 2352], raw[off + 2352 : off + 2646])
                        passes = db.get_raw_passes(lba)
                        ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=tier_now)
                        if db.upsert_solution(lba, sec, ok, pz, qz, pid, strat, iso_fh=iso_fh):
                            newly_ok += 1
            db.conn.commit()
            iso_fh.flush()
            print(
                f"  [Optical Round {round_idx + 1}/{max_passes}] {direction} {kbps}KB/s lead={lead} "
                f"remaining={len(remaining)} -> +{newly_ok} verified ({time.time() - t_r:.1f}s)",
                flush=True,
            )

        # Tier 3 escalation over any still-unverified target LBAs
        if max_tier >= 3:
            still_bad = [lba for lba in target_lbas if lba not in db.get_verified_set()]
            if still_bad:
                print(f"\n[Tier 3 Deep Algebraic Solver] Solving {len(still_bad)} stubborn LBAs...", flush=True)
                for lba in still_bad:
                    passes = db.get_raw_passes(lba)
                    if not passes:
                        continue
                    pid = pass_counts.get(lba, 0)
                    ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=3)
                    if db.upsert_solution(lba, sec, ok, pz, qz, pid, strat, iso_fh=iso_fh):
                        print(f"  LBA {lba} VERIFIED by {strat}!", flush=True)
                db.conn.commit()
                iso_fh.flush()


def _contiguous_intervals(byte_offsets: list[int], total_size: int) -> list[tuple[int, int]]:
    valid = sorted(off for off in byte_offsets if 0 <= off < total_size)
    if not valid:
        return []
    out = []
    s = valid[0]
    e = valid[0] + 1
    for off in valid[1:]:
        if off == e:
            e += 1
        else:
            out.append((s, e))
            s, e = off, off + 1
    out.append((s, e))
    return out


def extract_and_repair_files(
    iso_out: Path,
    extract_dir: Path,
    db: RecoveryStateDB,
    extents: list[FileExtent],
    categories: list[str],
    run_offline_solve: bool = False,
    max_tier: int = 3,
) -> list[FileRecoveryResult]:
    """Extract target files, run offline solver if requested, apply format-level repair, and validate."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    files = [e for e in extents if not e.is_dir and _match_category(e, categories)]
    ext_by_path = {e.path: e for e in extents if not e.is_dir}
    solved_map = db.get_solved_map()
    pipeline = SolverPipeline(max_tier=max_tier)

    known_masks: dict[int, set[int]] = {}
    if run_offline_solve:
        bad_targets = sorted({lba for f in files for lba in f.bad_lbas if solved_map.get(lba, (0,))[0] != 1})
        if bad_targets:
            print(f"[Offline Solver] Checking {len(bad_targets)} unverified target LBAs in SQLite...", flush=True)
            with iso_out.open("r+b") as iso_fh:
                for lba in bad_targets:
                    passes = db.get_raw_passes(lba)
                    if not passes:
                        continue
                    ctx: dict = {}
                    ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=max_tier, context=ctx)
                    if "known_mask_2048" in ctx:
                        known_masks[lba] = ctx["known_mask_2048"]
                    db.upsert_solution(lba, sec, ok, pz, qz, len(passes) - 1, strat, iso_fh=iso_fh)
                db.conn.commit()
                iso_fh.flush()
            solved_map = db.get_solved_map()

    results: list[FileRecoveryResult] = []
    with iso_out.open("rb") as iso_f:
        def read_file_bytes(ext_obj: FileExtent) -> bytes:
            iso_f.seek(ext_obj.start_lba * 2048)
            return iso_f.read(ext_obj.size_bytes)

        for ext_obj in files:
            data = read_file_bytes(ext_obj)
            handler = FormatRegistry.get_handler(ext_obj.ext, data)
            verified_bad = [lba for lba in ext_obj.bad_lbas if solved_map.get(lba, (0,))[0] == 1]
            unverified_bad = [lba for lba in ext_obj.bad_lbas if solved_map.get(lba, (0,))[0] != 1]
            min_p_ok = min((solved_map.get(lba, (0, 0, 0, ""))[1] for lba in ext_obj.bad_lbas), default=86)
            zero_holes = sum(
                1 for off in range(0, max(0, len(data) - 2047), 2048) if data[off : off + 2048] == b"\x00" * 2048
            )

            is_valid, fmt_status, meta = handler.validate(data)
            ref_used = None
            if not ext_obj.bad_lbas:
                rec_class = "INTACT_ORIGINAL"
            elif not unverified_bad:
                rec_class = "RECOVERED_100PCT_VERIFIED"
            else:
                # Partial sectors remain: extract byte-level verified mask inside unverified sectors & run format handler
                bad_byte_offs: list[int] = []
                for lba in unverified_bad:
                    sec_base = (lba - ext_obj.start_lba) * 2048
                    if lba in known_masks:
                        known_mask = known_masks[lba]
                    else:
                        passes = db.get_raw_passes(lba)
                        ctx = {}
                        if passes and max_tier >= 3:
                            core_solvers = SolverRegistry.get_solvers(names=["bipartite_core_hybrid"])
                            if core_solvers:
                                core_solvers[0].solve(lba, passes, context=ctx)
                        known_mask = ctx.get("known_mask_2048", set())
                        known_masks[lba] = known_mask
                    for rel_b in range(2048):
                        if rel_b not in known_mask and sec_base + rel_b < ext_obj.size_bytes:
                            bad_byte_offs.append(sec_base + rel_b)
                bad_intervals = _contiguous_intervals(bad_byte_offs, ext_obj.size_bytes)
                ref_data, ref_used = handler.find_reference(ext_obj, ext_by_path, read_file_bytes, data)
                repaired, rep_status, rep_meta = handler.reconstruct(
                    data, bad_intervals, reference_data=ref_data, context={"known_mask": True}
                )
                if rep_status.endswith("RESYNC") or rep_status.endswith("REPAIRED"):
                    data = repaired
                    is_valid, fmt_status, meta = handler.validate(data)
                    rec_class = rep_status
                    meta.update(rep_meta)
                else:
                    rec_class = "PARTIAL_RSPC_RECONSTRUCTED"

            rel = ext_obj.path.lstrip("/")
            dest = extract_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

            results.append(
                FileRecoveryResult(
                    iso_path=ext_obj.path,
                    saved_path=str(dest),
                    directory=ext_obj.directory,
                    size_bytes=ext_obj.size_bytes,
                    bad_sectors_total=len(ext_obj.bad_lbas),
                    bad_sectors_verified=len(verified_bad),
                    min_p_ok=min_p_ok,
                    zero_holes_2k=zero_holes,
                    recovery_class=rec_class,
                    format_status=fmt_status,
                    details={"handler": handler.name, "reference": ref_used, **meta},
                )
            )
    return results


def run_automated_pipeline(
    device: str | None = None,
    no_hardware: bool = False,
    recovered_root: Path = Path("recovered"),
    session_name: str | None = None,
    iso_out: Path | None = None,
    seed_iso: Path | None = None,
    bad_log: Path | None = None,
    full_rip: bool = False,
    db_path: Path | None = None,
    extract_dir: Path | None = None,
    categories: list[str] | None = None,
    max_passes: int = 48,
    max_tier: int = 3,
    report_out: Path | None = None,
) -> dict:
    """Execute the full 6-stage automated CD ripping, solving, repair, and validation pipeline."""
    cats = categories if categories is not None else ["photo_l", "photo_s", "pic", "photo"]
    resolved_name, session_dir = resolve_session_dir(
        recovered_root=recovered_root,
        session_name=session_name,
        device=device,
        no_hardware=no_hardware,
        seed_iso=seed_iso,
        iso_out=iso_out,
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    iso_out = iso_out or (session_dir / f"{resolved_name.lower()}_recovered.iso")
    db_path = db_path or (logs_dir / "raw_passes.sqlite3")
    extract_dir = extract_dir or (session_dir / "extracted")
    report_out = report_out or (logs_dir / "auto_recovery_report.json")
    print(f"[Session] name={resolved_name} -> {session_dir}", flush=True)

    db = RecoveryStateDB(db_path)
    try:
        bad_lbas: list[int] = []
        if not no_hardware:
            with OpticalDrive(device) as drive:
                caps = drive.caps
                print(
                    f"[Stage 1: Hardware Probe] {caps.sg_device} ({caps.sr_device}) | "
                    f"{caps.vendor_model} | C2={caps.supports_c2} | max_batch={caps.max_batch_sectors}",
                    flush=True,
                )
                total_sectors = drive.read_total_volume_sectors()
                prepare_output_iso(iso_out, seed_iso, total_sectors)
                if full_rip or seed_iso is None:
                    print(f"[Stage 2: Pass-0 Raw+C2 Rip] Ripping {total_sectors} sectors...", flush=True)
                    bad_lbas = pass0_full_rip(drive, db, iso_out, total_sectors)
        else:
            print("[Stage 1: Offline Mode] Skipping optical device probe.", flush=True)
            prepare_output_iso(iso_out, seed_iso, None)
            total_sectors = iso_out.stat().st_size // 2048

        if bad_log and bad_log.exists():
            bad_lbas = sorted(set(bad_lbas) | set(load_wincdemu_bad_lbas(bad_log)))
        bad_lbas = sorted(set(bad_lbas) | set(db.all_tracked_bad_lbas()))

        reader = iso9660.make_iso_sector_reader(iso_out)
        try:
            pvd, extents = iso9660.parse_iso9660_tree(reader, bad_lbas=set(bad_lbas))
        finally:
            reader.fh.close()  # type: ignore[attr-defined]

        etiology = iso9660.classify_damage_etiology(bad_lbas)
        print(
            f"[Stage 3: Triage] Vol={pvd['volume_id']!r} ({pvd['volume_sectors']} sectors), "
            f"files={sum(1 for e in extents if not e.is_dir)}, "
            f"bad_sectors={etiology.total_bad_sectors}, etiology={etiology.classification}",
            flush=True,
        )

        # Prioritize bad LBAs from high-priority target categories
        prio_files = [e for e in extents if not e.is_dir and e.bad_lbas and _match_category(e, cats)]
        prio_files.sort(key=lambda e: (e.priority, e.start_lba))
        target_lbas: list[int] = []
        seen = set()
        for f in prio_files:
            for lba in f.bad_lbas:
                if lba not in seen:
                    seen.add(lba)
                    target_lbas.append(lba)
        target_lbas.sort()

        if not no_hardware and target_lbas:
            with OpticalDrive(device) as drive:
                run_tiered_optical_recovery(
                    drive, db, iso_out, target_lbas, total_sectors, max_passes, max_tier
                )

        print(f"[Stages 5-6: Format Repair & Extraction -> {extract_dir}]", flush=True)
        results = extract_and_repair_files(
            iso_out,
            extract_dir,
            db,
            extents,
            cats,
            run_offline_solve=no_hardware,
            max_tier=max_tier,
        )

        # Write state JSON, file manifest CSV, and summary report JSON
        state_json = db.export_state_json(report_out.parent / "recovery_state.json", bad_lbas)
        manifest_csv = extract_dir / "photo_manifest.csv"
        if results:
            rows_dict = [
                {
                    "iso_path": r.iso_path,
                    "saved_path": r.saved_path,
                    "directory": r.directory,
                    "size_bytes": r.size_bytes,
                    "bad_sectors_total": r.bad_sectors_total,
                    "bad_sectors_verified": r.bad_sectors_verified,
                    "min_p_ok": r.min_p_ok,
                    "zero_holes_2k": r.zero_holes_2k,
                    "recovery_class": r.recovery_class,
                    "format_status": r.format_status,
                    "handler": r.details.get("handler", ""),
                    "details": json.dumps(r.details),
                }
                for r in results
            ]
            with manifest_csv.open("w", newline="", encoding="utf-8") as cf:
                w = csv.DictWriter(cf, fieldnames=list(rows_dict[0].keys()))
                w.writeheader()
                w.writerows(rows_dict)

        summary_by_dir: dict[str, dict[str, int]] = {}
        for r in results:
            d = summary_by_dir.setdefault(
                r.directory,
                {
                    "files": 0,
                    "intact_original": 0,
                    "verified_100pct": 0,
                    "format_repaired": 0,
                    "partial_other": 0,
                },
            )
            d["files"] += 1
            if r.recovery_class == "INTACT_ORIGINAL":
                d["intact_original"] += 1
            elif r.recovery_class == "RECOVERED_100PCT_VERIFIED":
                d["verified_100pct"] += 1
            elif "RESYNC" in r.recovery_class or "REPAIRED" in r.recovery_class:
                d["format_repaired"] += 1
            else:
                d["partial_other"] += 1

        report = {
            "session_dir": str(session_dir),
            "iso_out": str(iso_out),
            "pvd": pvd,
            "etiology": etiology.__dict__,
            "sector_state": state_json,
            "extracted_files": len(results),
            "by_directory": summary_by_dir,
            "manifest_csv": str(manifest_csv),
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        db.close()


def _mae_vs_thumb(jpg_bytes: bytes, thumb_bytes: bytes) -> tuple[float, float, int]:
    im_l = Image.open(io.BytesIO(jpg_bytes))
    im_l.load()
    im_s = Image.open(io.BytesIO(thumb_bytes))
    im_s.load()
    im_s_up = im_s.resize(im_l.size, Image.Resampling.BILINEAR)
    bl = im_l.tobytes()
    bs = im_s_up.tobytes()
    stride = im_l.size[0] * 3 * 16
    row_maes = [
        sum(abs(a - b) for a, b in zip(bl[r * stride : (r + 1) * stride], bs[r * stride : (r + 1) * stride]))
        / stride
        for r in range(im_l.size[1] // 16)
    ]
    return sum(abs(a - b) for a, b in zip(bl, bs)) / len(bl), max(row_maes), sum(1 for m in row_maes if m > 18.0)


def _build_synthetic_test_images() -> tuple[bytes, bytes, bytes]:
    """Generate deterministic in-memory JPEG, thumbnail, and BMP fixtures for clean-clone tests."""
    w, h = 640, 480
    px = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            px[i] = (x * 255 // w + (y // 16) * 7) & 0xFF
            px[i + 1] = (y * 255 // h + (x // 16) * 5) & 0xFF
            px[i + 2] = ((x + y) * 127 // (w + h) + 40) & 0xFF
    im = Image.frombytes("RGB", (w, h), bytes(px))
    buf_j = io.BytesIO()
    im.save(buf_j, format="JPEG", quality=92, subsampling=2)
    buf_t = io.BytesIO()
    im.resize((160, 120), Image.Resampling.BILINEAR).save(buf_t, format="JPEG", quality=90)
    buf_b = io.BytesIO()
    im.resize((400, 300), Image.Resampling.BILINEAR).save(buf_b, format="BMP")
    return buf_j.getvalue(), buf_t.getvalue(), buf_b.getvalue()


def run_synthetic_repair_self_test(repo_dir: Path) -> bool:
    """In-memory verification of JpegFormatHandler (thumbnail + blind) and BmpFormatHandler."""
    all_ok = True
    jpeg_h = FormatRegistry.get_handler(".JPG")
    bmp_h = FormatRegistry.get_handler(".BMP")

    print("=== Synthetic Fallback Self-Test (Strictly In-Memory; Zero Disk Mutation) ===", flush=True)
    sample_base = repo_dir / "recovered" / "MIFFY" / "extracted"
    if (sample_base / "L_PHOTO" / "MYCD0646.JPG").exists() and (sample_base / "PIC" / "GL_12.BMP").exists():
        jpeg_cases = [
            (
                "MYCD0646.JPG",
                (sample_base / "L_PHOTO" / "MYCD0646.JPG").read_bytes(),
                (sample_base / "S" / "MYCD0646.JPG").read_bytes(),
                31420,
                31620,
            ),
            (
                "MYCD0687.JPG",
                (sample_base / "L_PHOTO" / "MYCD0687.JPG").read_bytes(),
                (sample_base / "S" / "MYCD0687.JPG").read_bytes(),
                127676,
                127876,
            ),
        ]
        bmp_name = "GL_12.BMP"
        bmp_orig = (sample_base / "PIC" / "GL_12.BMP").read_bytes()
    else:
        syn_j, syn_t, syn_b = _build_synthetic_test_images()
        mid = len(syn_j) // 2
        jpeg_cases = [("SYNTHETIC_640x480.JPG", syn_j, syn_t, mid, min(len(syn_j) - 64, mid + 160))]
        bmp_name = "SYNTHETIC_400x300.BMP"
        bmp_orig = syn_b

    for fname, orig, thumb, c_start, c_end in jpeg_cases:
        corrupt = bytearray(orig)
        for pos in range(c_start, c_end):
            corrupt[pos] ^= 0x55
            if pos > 0 and corrupt[pos - 1] == 0xFF and corrupt[pos] != 0x00:
                corrupt[pos] = 0x00
            if corrupt[pos] == 0xFF and pos + 1 < len(corrupt) and corrupt[pos + 1] != 0x00:
                corrupt[pos] = 0xAA

        b_mae, b_rmax, b_bad = _mae_vs_thumb(bytes(corrupt), thumb)

        rebuilt_thumb, st_t, meta_t = jpeg_h.reconstruct(
            bytes(corrupt), [(c_start, c_end + 2)], reference_data=thumb
        )
        t_mae, t_rmax, t_bad = _mae_vs_thumb(rebuilt_thumb, thumb)
        ok_t, val_t, _ = jpeg_h.validate(rebuilt_thumb)

        rebuilt_blind, st_b, meta_b = jpeg_h.reconstruct(
            bytes(corrupt), [(c_start, c_end + 2)], reference_data=None
        )
        bl_mae, bl_rmax, bl_bad = _mae_vs_thumb(rebuilt_blind, thumb)
        ok_b, val_b, _ = jpeg_h.validate(rebuilt_blind)

        total_rows = Image.open(io.BytesIO(orig)).size[1] // 16
        print(
            f"[JPEG {fname}] corrupt(MAE={b_mae:.2f}, maxRow={b_rmax:.1f}, badRows={b_bad}/{total_rows})\n"
            f"  -> {st_t} ({val_t}, gap={meta_t.get('gap_mcus')}): MAE={t_mae:.2f}, maxRow={t_rmax:.1f}, badRows={t_bad}/{total_rows}\n"
            f"  -> {st_b} ({val_b}, gap={meta_b.get('gap_mcus')}): MAE={bl_mae:.2f}, maxRow={bl_rmax:.1f}, badRows={bl_bad}/{total_rows}",
            flush=True,
        )
        if not (ok_t and t_mae < 10.0 and t_bad == 0 and ok_b and bl_mae < 25.0):
            all_ok = False

    bmp_corrupt = bytearray(bmp_orig)
    bmp_corrupt[20000:22048] = b"\x00" * 2048
    repaired_bmp, bmp_st, bmp_meta = bmp_h.reconstruct(bytes(bmp_corrupt), [(20000, 22048)])
    ok_bmp, val_bmp, dims_bmp = bmp_h.validate(repaired_bmp)
    mae_bmp = sum(abs(a - b) for a, b in zip(bmp_orig, repaired_bmp)) / len(bmp_orig)
    print(
        f"[BMP {bmp_name}] {bmp_st} ({val_bmp}, {dims_bmp}, rows={bmp_meta.get('repaired_rows')}, fullMAE={mae_bmp:.3f})",
        flush=True,
    )
    if not (ok_bmp and mae_bmp < 2.0):
        all_ok = False
    return all_ok
