"""CD-ROM XA Mode 2 / VCD 2.0 (.bin + .cue) optical ripper, verifier, and MPEG-1 extractor."""

import csv
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from src.core import rspc
from src.core.logging_utils import RecoveryLogger, format_timestamp
from src.core.models import FileExtent, FileRecoveryResult
from src.core.state import RecoveryStateDB, load_wincdemu_bad_lbas, preserve_baseline_image
from src.filesystem import iso9660
from src.formats import FormatRegistry
from src.pipeline.orchestrator import (
    LEADS_WIDE,
    SPEEDS_MULTI,
    _build_batches,
    _match_category,
    partition_targets_by_priority,
    sanitize_volume_id,
)

ZERO_2352 = b"\x00" * 2352


def lba_to_cue_msf(lba: int) -> str:
    """Format a 0-based BIN sector LBA as CUE MM:SS:FF timestamp (75 frames/sec)."""
    if lba < 0:
        lba = 0
    mm = lba // 4500
    ss = (lba % 4500) // 75
    ff = lba % 75
    return f"{mm:02d}:{ss:02d}:{ff:02d}"


def write_cue_sheet(cue_path: Path, bin_filename: str, tracks: list[dict]) -> None:
    """Write a CDRWIN CUE sheet for a raw 2352-byte MODE2/2352 BIN image."""
    lines = [f'FILE "{bin_filename}" BINARY']
    data_tracks = sorted(
        (t for t in tracks if t.get("track") != 170),
        key=lambda t: int(t["track"]),
    )
    if not data_tracks:
        data_tracks = [{"track": 1, "is_data": True, "start_lba": 0}]

    for trk in data_tracks:
        trk_no = int(trk["track"])
        trk_type = "MODE2/2352" if trk.get("is_data", True) else "AUDIO"
        start_lba = int(trk["start_lba"])
        lines.append(f"  TRACK {trk_no:02d} {trk_type}")
        if trk_no == 1:
            lines.append(f"    INDEX 01 {lba_to_cue_msf(start_lba)}")
        else:
            if start_lba >= 150:
                lines.append(f"    INDEX 00 {lba_to_cue_msf(start_lba - 150)}")
            lines.append(f"    INDEX 01 {lba_to_cue_msf(start_lba)}")

    cue_path.parent.mkdir(parents=True, exist_ok=True)
    cue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_bin(
    bin_out: Path,
    seed_bin: Path | None,
    total_sectors: int | None = None,
) -> None:
    """Allocate or verify writable 2352-byte/sector BIN image without modifying seed_bin."""
    bin_out.parent.mkdir(parents=True, exist_ok=True)
    if not bin_out.exists():
        if seed_bin is not None and seed_bin.exists():
            if seed_bin.resolve() == bin_out.resolve():
                raise ValueError("seed_bin and bin_out must be distinct files to preserve baseline")
            shutil.copy2(seed_bin, bin_out)
        elif total_sectors is not None:
            with bin_out.open("wb") as f:
                f.truncate(total_sectors * 2352)
        else:
            raise FileNotFoundError(f"Neither {bin_out} nor seed_bin exists")

    if total_sectors is not None:
        target_bytes = total_sectors * 2352
        if bin_out.stat().st_size < target_bytes:
            with bin_out.open("r+b") as f:
                f.truncate(target_bytes)


def make_mode2_bin_sector_reader(bin_path: Path) -> Callable[[int], bytes]:
    """Return a 2048-byte Mode 2 Form 1 user-data reader (offset +24 inside 2352-byte frames)."""
    fh = bin_path.open("rb")

    def _read(lba: int) -> bytes:
        fh.seek(lba * 2352 + 24)
        data = fh.read(2048)
        if len(data) < 2048:
            data = data + b"\x00" * (2048 - len(data))
        return data

    _read.fh = fh  # type: ignore[attr-defined]
    return _read


def _verify_or_clean_mode2_sector(
    lba: int, raw_2352: bytes, c2_294: bytes
) -> tuple[bool, str, int, int, bytes]:
    """Verify Mode 2 Form 1/2 sector and run Form 1 single-pass C2 cleanup if needed."""
    ok, tag, pz, qz = rspc.verify_mode2_sector(lba, raw_2352, c2_294)
    if ok:
        return True, tag, pz, qz, raw_2352
    if not rspc.is_mode2_form2(raw_2352):
        ok_c, cleaned = rspc.clean_single_pass_mode2_f1(lba, raw_2352, c2_294)
        if ok_c:
            return True, "M2F1_C2_CLEANED", 86, 52, bytes(cleaned)
    return False, tag, pz, qz, raw_2352


def pass0_mode2_bin_rip(
    drive,
    db: RecoveryStateDB,
    bin_out: Path,
    total_sectors: int,
    logger: RecoveryLogger | None = None,
) -> tuple[list[int], dict]:
    """Stream all 2352-byte Mode 2 sectors once with DCR=1 + C2, logging verified stats."""
    bad_lbas: list[int] = []
    max_batch = drive.max_batch_sectors
    t0 = time.time()
    slips_retried = 0
    stats = {
        "m2f1_ok": 0,
        "m2f1_c2_cleaned": 0,
        "m2f2_edc_ok": 0,
        "m2f2_zero_edc_ok": 0,
        "bad_form1": 0,
        "bad_form2": 0,
        "read_errors": 0,
    }

    def _retry_slip(sub_lba: int) -> tuple[bool, bytes, bytes]:
        drive.directional_seek_flush(sub_lba, "FWD", total_sectors, offset_seed=97)
        st2, _, res2, raw2 = drive.read_cd_raw(sub_lba, 1, with_c2=True)
        if st2 != 0 or res2 != 0 or len(raw2) < 2646:
            return False, b"", b""
        return True, raw2[:2352], raw2[2352:2646]

    def _process_sector(sub_lba: int, r2352: bytes, rc2: bytes, bin_fh) -> None:
        nonlocal slips_retried
        if rspc.is_frame_slip_mode2(sub_lba, r2352):
            slips_retried += 1
            if logger is not None:
                logger.log_event(
                    "pass0_mode2",
                    "frame_slip",
                    lba=sub_lba,
                    got_msf=r2352[12:16].hex(),
                    expected_msf=rspc.lba_header_mode(sub_lba, 0x02).hex(),
                )
            ok_r, rr, rc = _retry_slip(sub_lba)
            if ok_r and not rspc.is_frame_slip_mode2(sub_lba, rr):
                r2352, rc2 = rr, rc
            else:
                bad_lbas.append(sub_lba)
                if rspc.is_mode2_form2(r2352):
                    stats["bad_form2"] += 1
                else:
                    stats["bad_form1"] += 1
                return

        ok, tag, pz, qz, final_2352 = _verify_or_clean_mode2_sector(sub_lba, r2352, rc2)
        if ok:
            if tag == "M2F1_OK":
                stats["m2f1_ok"] += 1
            elif tag == "M2F1_C2_CLEANED":
                stats["m2f1_c2_cleaned"] += 1
            elif tag == "M2F2_EDC_OK":
                stats["m2f2_edc_ok"] += 1
            else:
                stats["m2f2_zero_edc_ok"] += 1
            bin_fh.seek(sub_lba * 2352)
            bin_fh.write(final_2352)
            return

        bad_lbas.append(sub_lba)
        if rspc.is_mode2_form2(r2352):
            stats["bad_form2"] += 1
        else:
            stats["bad_form1"] += 1
        db.insert_raw_pass(sub_lba, 0, r2352, rc2, p_ok=pz, q_ok=qz)
        if logger is not None and logger.debug_enabled:
            db.record_pass_metric(
                sub_lba,
                0,
                format_timestamp(),
                "pass0_mode2",
                0,
                "FWD",
                0xFFFF,
                pz,
                qz,
                r2352[12:16].hex(),
                int(r2352[12:16] == rspc.lba_header_mode(sub_lba, 0x02)),
                0,
                0.0,
            )

    with bin_out.open("r+b") as bin_fh:
        lba = 0
        next_report = max_batch * 80
        while lba < total_sectors:
            cnt = min(max_batch, total_sectors - lba)
            st, _, resid, raw = drive.read_cd_raw(lba, cnt, with_c2=True)
            if st != 0 or resid != 0 or len(raw) < cnt * 2646:
                for sub_lba in range(lba, lba + cnt):
                    st2, _, res2, raw2 = drive.read_cd_raw(sub_lba, 1, with_c2=True)
                    if st2 != 0 or res2 != 0 or len(raw2) < 2646:
                        bad_lbas.append(sub_lba)
                        stats["read_errors"] += 1
                        continue
                    _process_sector(sub_lba, raw2[:2352], raw2[2352:2646], bin_fh)
            else:
                for i in range(cnt):
                    off = i * 2646
                    _process_sector(lba + i, raw[off : off + 2352], raw[off + 2352 : off + 2646], bin_fh)
            lba += cnt
            if lba >= next_report or lba >= total_sectors:
                db.conn.commit()
                bin_fh.flush()
                msg = (
                    f"  [Pass-0 BIN Rip] {lba}/{total_sectors} sectors "
                    f"({100.0 * lba / total_sectors:.1f}%) | bad_discovered={len(bad_lbas)} | "
                    f"slips_retried={slips_retried} | elapsed={time.time() - t0:.1f}s"
                )
                logger.info(msg) if logger is not None else print(msg, flush=True)
                next_report += max_batch * 80
        db.conn.commit()
        bin_fh.flush()
    stats["slips_retried"] = slips_retried
    return sorted(set(bad_lbas)), stats


def _recover_mode2_bad_lbas(
    drive,
    db: RecoveryStateDB,
    bin_out: Path,
    target_lbas: list[int],
    total_sectors: int,
    max_passes: int,
    logger: RecoveryLogger | None = None,
) -> int:
    """Run directional multi-speed Mode 2 re-reads for any broken Pass-0 sectors."""
    if not target_lbas or max_passes <= 0:
        return 0
    pass_counts = db.get_max_pass_ids()
    verified_total = 0
    verified_set: set[int] = set()

    with bin_out.open("r+b") as bin_fh:
        for round_idx in range(max_passes):
            remaining = [lba for lba in target_lbas if lba not in verified_set]
            if not remaining:
                break
            direction = "FWD" if (round_idx % 2 == 0) else "REV"
            kbps = SPEEDS_MULTI[round_idx % len(SPEEDS_MULTI)]
            lead = LEADS_WIDE[(round_idx * 5) % len(LEADS_WIDE)]
            drive.set_speed(kbps)
            batches = _build_batches(remaining, lead, drive.max_batch_sectors, total_sectors)
            if direction == "REV":
                batches = list(reversed(batches))
            newly_ok = 0
            for b_idx, (s_lba, cnt, covered) in enumerate(batches):
                if b_idx == 0 or (direction == "REV" and b_idx % 8 == 0):
                    drive.directional_seek_flush(covered[0], direction, total_sectors, offset_seed=round_idx + b_idx)
                st, _, res, raw = drive.read_cd_raw(s_lba, cnt, with_c2=True)
                if st != 0 or res != 0 or len(raw) < cnt * 2646:
                    continue
                for lba in covered:
                    off = (lba - s_lba) * 2646
                    r2352 = raw[off : off + 2352]
                    rc2 = raw[off + 2352 : off + 2646]
                    if rspc.is_frame_slip_mode2(lba, r2352):
                        continue
                    ok, tag, pz, qz, final_2352 = _verify_or_clean_mode2_sector(lba, r2352, rc2)
                    pid = pass_counts.get(lba, -1) + 1
                    pass_counts[lba] = pid
                    db.insert_raw_pass(lba, pid, r2352, rc2, p_ok=pz, q_ok=qz)
                    if ok:
                        bin_fh.seek(lba * 2352)
                        bin_fh.write(final_2352)
                        user_2048 = final_2352[24:2072]
                        db.conn.execute(
                            "INSERT OR REPLACE INTO solved_sectors "
                            "(lba, user_2048, verified, best_p_ok, best_q_ok, edc_ok, solved_pass, strategy) "
                            "VALUES (?, ?, 1, ?, ?, 1, ?, ?)",
                            (lba, user_2048, pz, qz, pid, tag),
                        )
                        verified_set.add(lba)
                        newly_ok += 1
                        verified_total += 1
            db.conn.commit()
            bin_fh.flush()
            msg = (
                f"  [Optical Mode2 {round_idx + 1}/{max_passes}] {direction} {kbps}KB/s "
                f"rem={len(remaining)} -> +{newly_ok} verified"
            )
            logger.info(msg) if logger else print(msg, flush=True)
    return verified_total


def _is_extent_mode2_form2(bin_f, ext_obj: FileExtent) -> bool:
    for lba in range(ext_obj.start_lba, ext_obj.end_lba):
        bin_f.seek(lba * 2352)
        sec = bin_f.read(2352)
        if len(sec) < 2352 or sec == ZERO_2352:
            continue
        if sec[15] == 0x02:
            return bool(sec[18] & 0x20)
        return False
    return ext_obj.directory.upper() in ("/MPEGAV", "/SEGMENT")


def _peek_first_form2_active_payload(bin_f, ext_obj: FileExtent) -> bytes:
    """Return the first active Mode 2 Form 2 user payload bytes for MPEG-PS sniffing."""
    for lba in range(ext_obj.start_lba, ext_obj.end_lba):
        bin_f.seek(lba * 2352)
        sec = bin_f.read(2352)
        if len(sec) < 2352 or sec == ZERO_2352:
            continue
        submode = sec[18]
        if (submode & 0x0E) != 0 or sec[24:28] == b"\x00\x00\x01\xba":
            return sec[24:40]
    return b""


def extract_mode2_vcd_files(
    bin_path: Path,
    extract_dir: Path,
    db: RecoveryStateDB,
    extents: list[FileExtent],
    categories: list[str],
    logger: RecoveryLogger | None = None,
) -> list[FileRecoveryResult]:
    """Extract Mode 2 Form 1 files (2048 B/sec) and Form 2 streams/MPEG (2324 B/sec)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    files = [e for e in extents if not e.is_dir and _match_category(e, categories)]
    verified_set = db.get_verified_set()
    results: list[FileRecoveryResult] = []
    t0 = time.time()

    with bin_path.open("rb") as bin_f:
        for f_idx, ext_obj in enumerate(files, 1):
            rel = ext_obj.path.lstrip("/")
            verified_bad = [lba for lba in ext_obj.bad_lbas if lba in verified_set]
            unverified_bad = [lba for lba in ext_obj.bad_lbas if lba not in verified_set]
            is_m2f2 = _is_extent_mode2_form2(bin_f, ext_obj)

            if is_m2f2:
                first_active = _peek_first_form2_active_payload(bin_f, ext_obj)
                is_mpeg_stream = (
                    ext_obj.directory.upper() in ("/MPEGAV", "/SEGMENT")
                    or first_active[:4] == b"\x00\x00\x01\xba"
                )
                dest = (extract_dir / rel).with_suffix(".mpg") if is_mpeg_stream else (extract_dir / rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                started_mpeg = False
                ended_mpeg = False
                written_bytes = 0
                zero_holes = 0
                active_sectors = 0

                with dest.open("wb") as out_f:
                    for lba in range(ext_obj.start_lba, ext_obj.end_lba):
                        bin_f.seek(lba * 2352)
                        sec = bin_f.read(2352)
                        if len(sec) < 2352:
                            break
                        if sec == ZERO_2352:
                            if not is_mpeg_stream or (started_mpeg and not ended_mpeg):
                                out_f.write(b"\x00" * 2324)
                                written_bytes += 2324
                                zero_holes += 1
                            continue
                        submode = sec[18]
                        if (submode & 0x0E) != 0 or sec[24:28] == b"\x00\x00\x01\xba" or not is_mpeg_stream:
                            started_mpeg = True
                            ended_mpeg = False
                            out_f.write(sec[24:2348])
                            written_bytes += 2324
                            active_sectors += 1
                        elif (submode & 0x80) and started_mpeg:
                            ended_mpeg = True

                rec_class = (
                    "INTACT_ORIGINAL"
                    if not ext_obj.bad_lbas
                    else ("RECOVERED_100PCT_VERIFIED" if not unverified_bad else "PARTIAL_RSPC_RECONSTRUCTED")
                )
                min_p_ok = 0
                if is_mpeg_stream:
                    vid_h = FormatRegistry.get_handler(".MPG")
                    v_ok, v_st, v_meta = vid_h.validate_file(dest)
                    handler_name = vid_h.name
                    fmt_status = (
                        f"M2F2_MPEG_PARTIAL (holes={zero_holes})"
                        if unverified_bad
                        else (v_st if v_ok else v_st)
                    )
                    details = {
                        "handler": handler_name,
                        "mode2_form": 2,
                        "active_mpeg_sectors": active_sectors,
                        **v_meta,
                    }
                else:
                    handler_name = "mode2_form2_raw"
                    fmt_status = (
                        "M2F2_RAW_OK"
                        if not unverified_bad
                        else f"M2F2_RAW_PARTIAL (holes={zero_holes})"
                    )
                    details = {
                        "handler": handler_name,
                        "mode2_form": 2,
                        "active_sectors": active_sectors,
                        "size": written_bytes,
                    }
                final_size = written_bytes
            else:
                dest = extract_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                chunks = []
                zero_holes = 0
                for lba in range(ext_obj.start_lba, ext_obj.end_lba):
                    bin_f.seek(lba * 2352)
                    sec = bin_f.read(2352)
                    user = sec[24:2072] if len(sec) >= 2072 else b"\x00" * 2048
                    if user == b"\x00" * 2048 and lba in ext_obj.bad_lbas and lba not in verified_set:
                        zero_holes += 1
                    chunks.append(user)
                data = b"".join(chunks)[: ext_obj.size_bytes]
                dest.write_bytes(data)

                handler = FormatRegistry.get_handler(ext_obj.ext, data)
                _, fmt_status, meta = handler.validate(data)
                rec_class = (
                    "INTACT_ORIGINAL"
                    if not ext_obj.bad_lbas
                    else ("RECOVERED_100PCT_VERIFIED" if not unverified_bad else "PARTIAL_RSPC_RECONSTRUCTED")
                )
                min_p_ok = 86 if not unverified_bad else 0
                handler_name = handler.name
                details = {"handler": handler_name, "mode2_form": 1, **meta}
                final_size = ext_obj.size_bytes

            if logger and logger.debug_enabled:
                logger.debug(
                    f"  [Extract {f_idx}/{len(files)}] {ext_obj.path} -> {rec_class}",
                    handler=handler_name,
                    status=fmt_status,
                    bad_lbas=f"{len(verified_bad)}/{len(ext_obj.bad_lbas)}",
                    size=final_size,
                )

            results.append(
                FileRecoveryResult(
                    iso_path=ext_obj.path,
                    saved_path=str(dest),
                    directory=ext_obj.directory,
                    size_bytes=final_size,
                    bad_sectors_total=len(ext_obj.bad_lbas),
                    bad_sectors_verified=len(verified_bad),
                    min_p_ok=min_p_ok,
                    zero_holes_2k=zero_holes,
                    recovery_class=rec_class,
                    format_status=fmt_status,
                    details=details,
                )
            )

    n_intact = sum(1 for r in results if r.recovery_class == "INTACT_ORIGINAL")
    n_ver = sum(1 for r in results if r.recovery_class == "RECOVERED_100PCT_VERIFIED")
    n_part = len(results) - n_intact - n_ver
    msg_done = (
        f"[Stages 5-6 Complete] extracted={len(results)} | intact={n_intact} | "
        f"verified_100pct={n_ver} | repaired=0 | partial={n_part} | elapsed={time.time() - t0:.1f}s"
    )
    logger.info(msg_done) if logger else print(msg_done, flush=True)
    return results


def _resolve_mode2_session_dir(
    drive, recovered_root: Path, session_name: str | None
) -> tuple[str, Path]:
    resolved = sanitize_volume_id(session_name) if session_name else ""
    if not resolved and drive is not None:
        st, _, res, raw16 = drive.read_cd_raw(16, 1, with_c2=False)
        if st == 0 and res == 0 and len(raw16) >= 96 and raw16[25:30] == b"CD001":
            resolved = sanitize_volume_id(raw16[64:96].decode("ascii", errors="ignore"))
    if not resolved:
        resolved = f"vcd_{uuid.uuid4().hex[:8]}"
    return resolved, recovered_root / resolved


def run_mode2_pipeline(
    drive,
    tracks: list[dict],
    total_sectors: int,
    recovered_root: Path = Path("recovered"),
    session_name: str | None = None,
    bad_log: Path | None = None,
    full_rip: bool = False,
    categories: list[str] | None = None,
    max_passes: int = 0,
    focus_passes: int = 0,
    max_tier: int = 1,
    workers: int = 0,
    debug: bool = False,
    report_out: Path | None = None,
    existing_db: RecoveryStateDB | None = None,
    existing_logger: RecoveryLogger | None = None,
    t_stage1_start: float | None = None,
    t_pipeline_start: float | None = None,
    save_baseline: bool = False,
) -> dict:
    """Execute the 6-stage CD-ROM XA Mode 2 / VCD 2.0 BIN+CUE rip, triage, and extraction pipeline."""
    t_pipeline0 = t_pipeline_start if t_pipeline_start is not None else time.time()
    t_s1 = t_stage1_start if t_stage1_start is not None else t_pipeline0
    cats = categories if categories is not None else ["all"]

    resolved_name, session_dir = _resolve_mode2_session_dir(drive, recovered_root, session_name)
    session_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    bin_out = session_dir / f"{resolved_name.lower()}_recovered.bin"
    cue_out = session_dir / f"{resolved_name.lower()}_recovered.cue"
    db_path = logs_dir / "raw_passes.sqlite3"
    extract_dir = session_dir / "extracted"
    report_out = report_out or (logs_dir / "auto_recovery_report.json")

    owns_db = existing_db is None
    owns_logger = existing_logger is None
    db = existing_db or RecoveryStateDB(db_path)
    logger = existing_logger or RecoveryLogger(logs_dir, db=db, debug=debug)

    try:
        if owns_logger:
            logger.info(f"[Session] name={resolved_name} -> {session_dir}", debug=debug)
            caps = getattr(drive, "caps", None)
            if caps is not None:
                logger.info(
                    f"[Stage 1: Hardware Probe] {caps.sg_device} ({caps.sr_device}) | "
                    f"{caps.vendor_model} | C2={caps.supports_c2} | max_batch={caps.max_batch_sectors}"
                )
        logger.info(
            f"[Stage 1: Mode 2 / VCD Detected] tracks={sum(1 for t in tracks if t.get('track') != 170)} | "
            f"total_sectors={total_sectors} | raw_size={total_sectors * 2352}"
        )
        bin_existed = bin_out.exists() and bin_out.stat().st_size > 0
        prepare_output_bin(bin_out, None, total_sectors)
        if tracks or not cue_out.exists():
            write_cue_sheet(cue_out, bin_out.name, tracks)
        timing: dict[str, float | str] = {
            "started_at": format_timestamp(),
            "stage1_probe_s": round(time.time() - t_s1, 3),
        }

        bad_lbas: list[int] = []
        pass0_stats: dict = {}
        if drive is not None and (full_rip or not bin_existed):
            t_s2 = time.time()
            logger.info(f"[Stage 2: Pass-0 Mode-2 Raw+C2 BIN Rip] Ripping {total_sectors} sectors...")
            bad_lbas, pass0_stats = pass0_mode2_bin_rip(drive, db, bin_out, total_sectors, logger=logger)
            timing["stage2_pass0_s"] = round(time.time() - t_s2, 3)
            if save_baseline:
                if preserve_baseline_image(
                    bin_out,
                    session_dir / "baseline.bin",
                    cue_src=cue_out,
                    cue_baseline=session_dir / "baseline.cue",
                ):
                    logger.info(f"[Baseline Preserved] {session_dir / 'baseline.bin'} (0444)")

        t_s3 = time.time()
        if bad_log and bad_log.exists():
            bad_lbas = sorted(set(bad_lbas) | set(load_wincdemu_bad_lbas(bad_log)))
        bad_lbas = sorted(set(bad_lbas) | set(db.all_tracked_bad_lbas()))

        reader = make_mode2_bin_sector_reader(bin_out)
        try:
            pvd, extents = iso9660.parse_iso9660_tree(reader, bad_lbas=set(bad_lbas))
        finally:
            reader.fh.close()  # type: ignore[attr-defined]

        etiology = iso9660.classify_damage_etiology(bad_lbas)
        timing["stage3_triage_s"] = round(time.time() - t_s3, 3)
        logger.info(
            f"[Stage 3: Triage] Vol={pvd['volume_id']!r} (Mode2 BIN={total_sectors} sectors), "
            f"files={sum(1 for e in extents if not e.is_dir)}, "
            f"bad_sectors={etiology.total_bad_sectors}, etiology={etiology.classification}"
        )

        media_lbas, bulk_lbas = partition_targets_by_priority(extents, cats)
        target_files = [e for e in extents if not e.is_dir and _match_category(e, cats)]
        logger.info(
            f"[Stage 3: Target Plan] categories={','.join(cats)} | "
            f"media_lbas={len(media_lbas)} | bulk_lbas={len(bulk_lbas)} | target_files={len(target_files)}"
        )
        if logger.debug_enabled:
            for ext_obj in target_files:
                logger.debug(
                    f"  [Stage 3 file] {ext_obj.path}",
                    lba_range=f"{ext_obj.start_lba}..{ext_obj.end_lba}",
                    sectors=ext_obj.sectors,
                    bad_lbas=len(ext_obj.bad_lbas),
                )
        logger.log_event("triage", "stage_complete", media_lbas=len(media_lbas), bulk_lbas=len(bulk_lbas))

        all_target_lbas = sorted(set(media_lbas) | set(bulk_lbas) | set(bad_lbas))
        if drive is not None and max_passes > 0 and all_target_lbas:
            t_s4 = time.time()
            logger.info(f"[Stage 4: Mode 2 Multi-Pass Recovery] {len(all_target_lbas)} LBAs")
            _recover_mode2_bad_lbas(drive, db, bin_out, all_target_lbas, total_sectors, max_passes, logger=logger)
            timing["stage4_mode2_s"] = round(time.time() - t_s4, 3)

        t_s5 = time.time()
        logger.info(f"[Stages 5-6: Mode 2 Extraction -> {extract_dir}]")
        results = extract_mode2_vcd_files(bin_out, extract_dir, db, extents, cats, logger=logger)
        timing["stage5_6_repair_extract_s"] = round(time.time() - t_s5, 3)

        state_json = db.export_state_json(report_out.parent / "recovery_state.json", bad_lbas)
        manifest_csv = extract_dir / "manifest.csv"
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
            else:
                d["partial_other"] += 1

        timing["finished_at"] = format_timestamp()
        timing["total_elapsed_s"] = round(time.time() - t_pipeline0, 3)
        log_files = {"recovery_log": str(logger.recovery_log_path)}
        if logger.debug_log_path is not None:
            log_files["debug_log"] = str(logger.debug_log_path)
        if logger.events_jsonl_path is not None:
            log_files["debug_events_jsonl"] = str(logger.events_jsonl_path)

        report = {
            "disc_mode": "mode2",
            "session_dir": str(session_dir),
            "bin_out": str(bin_out),
            "cue_out": str(cue_out),
            "iso_out": str(bin_out),
            "toc": tracks,
            "pass0_stats": pass0_stats,
            "debug_enabled": bool(debug),
            "timing": timing,
            "log_files": log_files,
            "pvd": pvd,
            "etiology": etiology.__dict__,
            "sector_state": state_json,
            "extracted_files": len(results),
            "by_directory": summary_by_dir,
            "manifest_csv": str(manifest_csv),
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.log_event("pipeline", "stage_complete", total_elapsed_s=timing["total_elapsed_s"])
        db.conn.commit()
        return report
    finally:
        if owns_logger:
            logger.close()
        if owns_db:
            db.close()
