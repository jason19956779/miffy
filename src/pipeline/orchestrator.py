"""Six-stage modular CD optical ripping, triage, RSPC solving, and format recovery pipeline."""

import csv
import io
import json
import os
import re
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageFile

from src.core import rspc
from src.core.logging_utils import RecoveryLogger, format_timestamp
from src.core.models import FileExtent, FileRecoveryResult
from src.core.state import (
    RecoveryStateDB,
    load_wincdemu_bad_lbas,
    prepare_output_iso,
    preserve_baseline_image,
)
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
                if not resolved:
                    st_m2, _, res_m2, raw_m2 = drive.read_cd_raw(16, 1, with_c2=False)
                    if st_m2 == 0 and res_m2 == 0 and len(raw_m2) >= 96 and raw_m2[25:30] == b"CD001":
                        resolved = sanitize_volume_id(raw_m2[64:96].decode("ascii", errors="ignore"))
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
        "audio": lambda e: e.ext in iso9660.AUDIO_EXTS,
        "video": lambda e: e.ext in iso9660.VIDEO_EXTS or (e.ext == ".DAT" and d_up in ("/MPEGAV", "/SEGMENT")),
        "media": lambda e: (
            e.ext in iso9660.PHOTO_MAIN_EXTS
            or e.ext in iso9660.AUDIO_EXTS
            or e.ext in iso9660.VIDEO_EXTS
            or (e.ext == ".DAT" and d_up in ("/MPEGAV", "/SEGMENT"))
        ),
    }
    for cat in categories:
        fn = alias_map.get(cat.lower())
        if fn and fn(ext_obj):
            return True
        if d_up.startswith(cat.upper()) or ext_obj.ext.lower().lstrip(".") == cat.lower().lstrip("."):
            return True
    return False


def partition_targets_by_priority(
    extents: list[FileExtent],
    categories: list[str],
    high_priority_max: int = 20,
) -> tuple[list[int], list[int]]:
    """Split target bad LBAs into high-priority media (`priority <= 20`) and bulk file LBAs."""
    matching = [e for e in extents if not e.is_dir and e.bad_lbas and _match_category(e, categories)]
    matching.sort(key=lambda e: (e.priority, e.start_lba))

    media_set: set[int] = set()
    bulk_set: set[int] = set()
    for ext_obj in matching:
        if ext_obj.priority <= high_priority_max:
            media_set.update(ext_obj.bad_lbas)
        else:
            bulk_set.update(ext_obj.bad_lbas)
    bulk_set -= media_set
    return sorted(media_set), sorted(bulk_set)


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


class _JpegThumbOracle:
    """Picklable post-bad-sector JPEG MAE scorer against companion thumbnail for Tier 3 ties."""

    def __init__(self, iso_extent_ref: tuple[str, int, int], bad_lba: int, thumb_bytes: bytes):
        self.iso_path, self.start_lba, self.size_bytes, self.bad_lba = (
            Path(iso_extent_ref[0]),
            iso_extent_ref[1],
            iso_extent_ref[2],
            bad_lba,
        )
        self.thumb_bytes = thumb_bytes
        self._base_bytes: bytes | None = None

    def _get_base(self) -> bytearray:
        if self._base_bytes is None:
            with self.iso_path.open("rb") as f:
                f.seek(self.start_lba * 2048)
                self._base_bytes = f.read(self.size_bytes)
        return bytearray(self._base_bytes)

    def __call__(self, user_2048: bytes) -> float:
        buf = self._get_base()
        off = (self.bad_lba - self.start_lba) * 2048
        end_off = min(self.size_bytes, off + 2048)
        buf[off:end_off] = user_2048[: max(0, end_off - off)]
        prev_trunc = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            im_l = Image.open(io.BytesIO(bytes(buf)))
            im_l.load()
        except Exception:
            return -999.0
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_trunc
        try:
            im_s = Image.open(io.BytesIO(self.thumb_bytes))
            im_s.load()
            im_s_up = im_s.resize(im_l.size, Image.Resampling.BILINEAR)
            bl = im_l.tobytes()
            bs = im_s_up.tobytes()
            start_row = max(0, (im_l.size[1] * off) // max(1, self.size_bytes) - 32)
            s0 = start_row * im_l.size[0] * 3
            if s0 >= len(bl):
                s0 = 0
            mae = sum(abs(a - b) for a, b in zip(bl[s0:], bs[s0:])) / max(1, len(bl) - s0)
            return -mae
        except Exception:
            return -999.0


def _tier3_worker(
    args: tuple[int, list[tuple[bytes, bytes]], int, _JpegThumbOracle | None, bool],
) -> tuple[int, bool, int, int, bytes, str, list[dict], int, float]:
    lba, passes, max_tier, oracle, collect_trace = args
    pipeline = SolverPipeline(max_tier=max_tier)
    ctx: dict = {}
    if oracle is not None:
        ctx["oracle_fn"] = oracle
    if collect_trace:
        ctx["debug_trace"] = []
    t0 = time.perf_counter()
    ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=max_tier, context=ctx)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (
        lba,
        bool(ok),
        pz,
        qz,
        bytes(sec),
        strat,
        ctx.get("debug_trace", []),
        int(ctx.get("frame_slips_filtered", 0)),
        elapsed_ms,
    )


def _build_lba_to_file_map(extents: list[FileExtent] | None) -> dict[int, str]:
    """Map each bad LBA to its first owning ISO file path for live recovery logs."""
    mapping: dict[int, str] = {}
    if not extents:
        return mapping
    for ext_obj in extents:
        if ext_obj.is_dir:
            continue
        for lba in ext_obj.bad_lbas:
            mapping.setdefault(lba, ext_obj.path)
    return mapping


def _format_trace_summary(trace: list[dict]) -> str:
    """Format per-solver trace metrics on one brace-free line."""
    if not trace:
        return "none"
    parts = []
    for tr in trace:
        ms = int(round(float(tr.get("elapsed_ms", 0.0))))
        if tr.get("ok"):
            parts.append(f"{tr['solver']}:OK:{tr['strategy']}({ms}ms)")
        else:
            parts.append(f"{tr['solver']}:p{tr['p_ok']}/q{tr['q_ok']}({ms}ms)")
    return ",".join(parts)


def _format_safe_meta(meta: dict) -> str:
    """Format scalar metadata entries cleanly without JSON braces."""
    parts = []
    for k, v in sorted(meta.items()):
        if isinstance(v, (int, float, str, bool)):
            sv = str(v).replace("{", "(").replace("}", ")")
            parts.append(f"{k}={sv}")
    return " ".join(parts)


def _record_solve_telemetry(
    db: RecoveryStateDB,
    logger: RecoveryLogger | None,
    lba: int,
    max_tier: int,
    pass_count: int,
    ctx: dict,
) -> None:
    if logger is None or not logger.debug_enabled:
        return
    ts = format_timestamp()
    for tr in ctx.get("debug_trace", []):
        db.record_solver_attempt(
            ts,
            lba,
            max_tier,
            tr["solver"],
            pass_count,
            int(tr["ok"]),
            tr["p_ok"],
            tr["q_ok"],
            tr["strategy"],
            tr["elapsed_ms"],
        )


def _solve_lba_with_telemetry(
    pipeline: SolverPipeline,
    db: RecoveryStateDB,
    iso_fh,
    logger: RecoveryLogger | None,
    lba: int,
    passes: list[tuple[bytes, bytes]],
    pass_id: int,
    max_tier: int,
    oracle_fn=None,
) -> tuple[bool, float]:
    ctx: dict = {}
    if logger is not None and logger.debug_enabled:
        ctx["debug_trace"] = []
    if oracle_fn is not None:
        ctx["oracle_fn"] = oracle_fn
    t0 = time.perf_counter()
    ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=max_tier, context=ctx)
    solve_ms = (time.perf_counter() - t0) * 1000.0
    _record_solve_telemetry(db, logger, lba, max_tier, len(passes), ctx)
    verified = db.upsert_solution(lba, sec, ok, pz, qz, pass_id, strat, iso_fh=iso_fh)
    return verified, solve_ms


def pass0_full_rip(
    drive: OpticalDrive,
    db: RecoveryStateDB,
    iso_out: Path,
    total_sectors: int,
    logger: RecoveryLogger | None = None,
) -> list[int]:
    """Stream the entire CD once with DCR=1 + C2 pointers, enforcing exact MSF+RSPC+EDC integrity."""
    bad_lbas: list[int] = []
    pipeline = SolverPipeline(solver_names=["single_c2"])
    max_batch = drive.max_batch_sectors
    t0 = time.time()
    slips_retried = 0

    def _read_single_retry(sub_lba: int) -> tuple[bool, bytes, bytes]:
        drive.directional_seek_flush(sub_lba, "FWD", total_sectors, offset_seed=97)
        st2, _, res2, raw2 = drive.read_cd_raw(sub_lba, 1, with_c2=True)
        if st2 != 0 or res2 != 0:
            return False, b"", b""
        return True, raw2[:2352], raw2[2352:2646]

    def _log_bad(sub_lba: int, raw_2352: bytes, c2_294: bytes, is_slip: bool) -> None:
        nonlocal slips_retried
        if is_slip:
            slips_retried += 1
            if logger is not None:
                logger.log_event(
                    "pass0",
                    "frame_slip",
                    lba=sub_lba,
                    got_msf=raw_2352[12:16].hex(),
                    expected_msf=rspc.lba_header(sub_lba).hex(),
                )
            ok_r, r_raw, r_c2 = _read_single_retry(sub_lba)
            if ok_r and not rspc.is_frame_slip(sub_lba, r_raw):
                raw_2352, c2_294 = r_raw, r_c2
                is_slip = False
            elif ok_r:
                if logger is not None:
                    logger.debug("Pass-0 slip persisted after retry", lba=sub_lba, got_msf=r_raw[12:16].hex())
                bad_lbas.append(sub_lba)
                return
            else:
                bad_lbas.append(sub_lba)
                return

        if is_slip:
            bad_lbas.append(sub_lba)
            return

        bad_lbas.append(sub_lba)
        pz_r, qz_r = db.insert_raw_pass(sub_lba, 0, raw_2352, c2_294)
        ok_s, solved = _solve_lba_and_upsert(sub_lba, raw_2352, c2_294)
        if logger is not None and logger.debug_enabled:
            db.record_pass_metric(
                sub_lba,
                0,
                format_timestamp(),
                "pass0",
                0,
                "FWD",
                0xFFFF,
                pz_r,
                qz_r,
                raw_2352[12:16].hex(),
                int(raw_2352[12:16] == rspc.lba_header(sub_lba)),
                0,
                solved,
            )

    with iso_out.open("r+b") as iso_fh:
        def _solve_lba_and_upsert(sub_lba: int, raw_2352: bytes, c2_294: bytes) -> tuple[bool, float]:
            t_s = time.perf_counter()
            ok, pz, qz, decoded, strat = pipeline.solve_sector(sub_lba, [(raw_2352, c2_294)], max_tier=1)
            ms = (time.perf_counter() - t_s) * 1000.0
            ver = db.upsert_solution(sub_lba, decoded, ok, pz, qz, 0, strat, iso_fh=iso_fh)
            return ver, ms

        lba = 0
        while lba < total_sectors:
            cnt = min(max_batch, total_sectors - lba)
            st, _, resid, raw = drive.read_cd_raw(lba, cnt, with_c2=True)
            if st != 0 or resid != 0:
                for sub_lba in range(lba, lba + cnt):
                    st2, _, res2, raw2 = drive.read_cd_raw(sub_lba, 1, with_c2=True)
                    if st2 != 0 or res2 != 0:
                        bad_lbas.append(sub_lba)
                        continue
                    r2352, rc2 = raw2[:2352], raw2[2352:2646]
                    if not _try_pass0_fast(sub_lba, r2352, rc2, iso_fh, bad_lbas):
                        _log_bad(sub_lba, r2352, rc2, rspc.is_frame_slip(sub_lba, r2352))
            else:
                for i in range(cnt):
                    off = i * 2646
                    sub_lba = lba + i
                    r2352 = raw[off : off + 2352]
                    rc2 = raw[off + 2352 : off + 2646]
                    if not _try_pass0_fast(sub_lba, r2352, rc2, iso_fh, bad_lbas):
                        _log_bad(sub_lba, r2352, rc2, rspc.is_frame_slip(sub_lba, r2352))
            lba += cnt
            if lba % (max_batch * 80) == 0 or lba >= total_sectors:
                db.conn.commit()
                iso_fh.flush()
                msg = (
                    f"  [Pass-0 Rip] {lba}/{total_sectors} sectors "
                    f"({100.0 * lba / total_sectors:.1f}%) | bad_discovered={len(bad_lbas)} | "
                    f"slips_retried={slips_retried} | elapsed={time.time() - t0:.1f}s"
                )
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg, flush=True)
        db.conn.commit()
        iso_fh.flush()
    return sorted(set(bad_lbas))


def _try_pass0_fast(
    lba: int,
    raw_2352: bytes,
    c2_294: bytes,
    iso_fh,
    bad_lbas: list[int],
) -> bool:
    """Strictly verify expected MSF header + 86/86 P + 52/52 Q + 32-bit EDC on Pass-0."""
    if rspc.is_frame_slip(lba, raw_2352):
        return False
    sec = bytearray(raw_2352)
    if rspc.check_and_fix_q_parity(sec, expected_lba=lba):
        iso_fh.seek(lba * 2048)
        iso_fh.write(sec[16:2064])
        return True
    # Fast single-pass C2 clean using Tier 1 SinglePassC2Solver only (exact header + EDC enforced)
    from src.solvers.portfolio import clean_single_pass

    ok, decoded, _ = clean_single_pass(lba, raw_2352, c2_294)
    if ok and rspc.check_and_fix_q_parity(decoded, expected_lba=lba):
        iso_fh.seek(lba * 2048)
        iso_fh.write(decoded[16:2064])
        return True
    return False


def _handle_pass0_sector(
    lba: int,
    raw_2352: bytes,
    c2_294: bytes,
    db: RecoveryStateDB,
    pipeline: SolverPipeline,
    iso_fh,
    bad_lbas: list[int],
) -> None:
    """Backward-compatible single-sector Pass-0 handler with strict expected_lba guard."""
    if rspc.is_frame_slip(lba, raw_2352):
        bad_lbas.append(lba)
        return
    sec = bytearray(raw_2352)
    if rspc.check_and_fix_q_parity(sec, expected_lba=lba):
        iso_fh.seek(lba * 2048)
        iso_fh.write(sec[16:2064])
        return
    ok, pz, qz, decoded, strat = pipeline.solve_sector(lba, [(raw_2352, c2_294)], max_tier=1)
    if ok and rspc.check_and_fix_q_parity(decoded, expected_lba=lba):
        iso_fh.seek(lba * 2048)
        iso_fh.write(decoded[16:2064])
        return
    bad_lbas.append(lba)
    db.insert_raw_pass(lba, 0, raw_2352, c2_294)
    db.upsert_solution(lba, decoded, False, pz, qz, 0, strat, iso_fh=iso_fh)


def _build_lba_oracles(
    iso_out: Path,
    extents: list[FileExtent],
    still_bad: list[int],
) -> dict[int, _JpegThumbOracle]:
    oracles: dict[int, _JpegThumbOracle] = {}
    bad_set = set(still_bad)
    ext_by_path = {e.path.upper(): e for e in extents if not e.is_dir}
    reader = iso9660.make_iso_sector_reader(iso_out)
    try:
        for ext_obj in extents:
            if ext_obj.is_dir or ext_obj.ext.upper() not in (".JPG", ".JPEG"):
                continue
            hit_lbas = [lba for lba in ext_obj.bad_lbas if lba in bad_set]
            if not hit_lbas:
                continue
            stem = Path(ext_obj.name).stem
            thumb_ext = None
            for cand in (f"/PHOTO/S/{ext_obj.name}".upper(), f"/PHOTO/S/{stem}_T{ext_obj.ext}".upper()):
                if cand in ext_by_path:
                    thumb_ext = ext_by_path[cand]
                    break
            if thumb_ext is None or thumb_ext.bad_lbas:
                continue
            # Read thumbnail bytes from iso_out
            n_t = (thumb_ext.size_bytes + 2047) // 2048
            t_buf = b"".join(reader(thumb_ext.start_lba + i) for i in range(n_t))[: thumb_ext.size_bytes]
            for lba in hit_lbas:
                oracles[lba] = _JpegThumbOracle((str(iso_out), ext_obj.start_lba, ext_obj.size_bytes), lba, t_buf)
    finally:
        reader.fh.close()  # type: ignore[attr-defined]
    return oracles


def _run_tier3_phase(
    db: RecoveryStateDB,
    iso_out: Path,
    target_lbas: list[int],
    max_tier: int,
    workers: int,
    extents: list[FileExtent],
    phase: str,
    logger: RecoveryLogger | None,
) -> int:
    if max_tier < 3 or not target_lbas:
        return 0
    verified_now = db.get_verified_set()
    still_bad = [lba for lba in target_lbas if lba not in verified_now]
    if not still_bad:
        return 0

    msg = f"[Tier 3 Deep Algebraic Solver ({phase})] Solving {len(still_bad)} stubborn LBAs..."
    if logger:
        logger.info(msg)
    else:
        print(msg, flush=True)

    lba_to_file = _build_lba_to_file_map(extents)
    pass_counts = db.get_max_pass_ids()
    pass_len_by_lba: dict[int, int] = {}
    oracles = _build_lba_oracles(iso_out, extents, still_bad)
    tasks = []
    for lba in still_bad:
        passes = db.get_raw_passes(lba)
        if passes:
            pass_len_by_lba[lba] = len(passes)
            tasks.append((lba, passes, max_tier, oracles.get(lba), bool(logger and logger.debug_enabled)))
    if not tasks:
        return 0

    n_workers = workers if workers > 0 else min(8, os.cpu_count() or 1)
    solved_t3 = 0
    done_idx = 0
    last_hb = time.monotonic()
    handled_lbas: set[int] = set()
    t_t3 = time.time()

    with iso_out.open("r+b") as iso_fh:
        def _handle_t3_result(res_item: tuple[int, bool, int, int, bytes, str, list[dict], int, float]) -> None:
            nonlocal solved_t3, done_idx, last_hb
            lba, ok, pz, qz, sec_bytes, strat, trace, slips_f, elapsed_ms = res_item
            handled_lbas.add(lba)
            done_idx += 1
            pid = pass_counts.get(lba, 0)
            lba_file = lba_to_file.get(lba, "-")
            if trace and logger and logger.debug_enabled:
                _record_solve_telemetry(
                    db, logger, lba, max_tier, pass_len_by_lba.get(lba, 1), {"debug_trace": trace}
                )
            if slips_f and logger:
                logger.debug(
                    "Tier 3 filtered stored frame slips",
                    phase=phase,
                    lba=lba,
                    file=lba_file,
                    slips=slips_f,
                )
            if db.upsert_solution(lba, sec_bytes, ok, pz, qz, pid, strat, iso_fh=iso_fh):
                solved_t3 += 1
                db.conn.commit()
                iso_fh.flush()
                m2 = f"  [Tier 3 {phase} {done_idx}/{len(tasks)}] LBA {lba} VERIFIED by {strat} ({elapsed_ms:.0f}ms)"
                if logger:
                    logger.info(m2, file=lba_file)
                    if trace and logger.debug_enabled:
                        logger.debug(
                            f"  [Tier 3 {phase} trace] LBA {lba}",
                            file=lba_file,
                            solvers=_format_trace_summary(trace),
                        )
                else:
                    print(f"{m2} | file={lba_file}", flush=True)
            else:
                now_m = time.monotonic()
                if logger and logger.debug_enabled:
                    logger.debug(
                        f"  [Tier 3 {phase} {done_idx}/{len(tasks)}] LBA {lba} PARTIAL",
                        file=lba_file,
                        p_ok=pz,
                        q_ok=qz,
                        strategy=strat,
                        ms=round(elapsed_ms, 1),
                        solvers=_format_trace_summary(trace),
                    )
                    last_hb = now_m
                elif len(tasks) <= 20 or (now_m - last_hb >= 10.0):
                    m_p = (
                        f"  [Tier 3 {phase} {done_idx}/{len(tasks)}] LBA {lba} partial "
                        f"(p={pz}/86 q={qz}/52, {strat}, {elapsed_ms:.0f}ms)"
                    )
                    if logger:
                        logger.info(m_p, file=lba_file)
                    else:
                        print(f"{m_p} | file={lba_file}", flush=True)
                    last_hb = now_m

        if n_workers > 1 and len(tasks) > 1:
            try:
                with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
                    futs = [pool.submit(_tier3_worker, t) for t in tasks]
                    for fut in as_completed(futs):
                        _handle_t3_result(fut.result())
            except Exception as exc:
                if logger:
                    logger.debug("ProcessPoolExecutor fallback to serial", error=str(exc))
                for t in tasks:
                    if t[0] not in handled_lbas:
                        _handle_t3_result(_tier3_worker(t))
        else:
            for t in tasks:
                _handle_t3_result(_tier3_worker(t))
        db.conn.commit()
        iso_fh.flush()
    msg_done = f"[Tier 3 ({phase}) Complete] +{solved_t3}/{len(tasks)} verified in {time.time() - t_t3:.1f}s"
    if logger:
        logger.info(msg_done)
    else:
        print(msg_done, flush=True)
    return solved_t3


def _round_solve_worker(
    args: tuple[int, list[tuple[bytes, bytes]], int, int, int, int, str, int, int, bool],
) -> tuple[int, bool, int, int, bytes, str, int, int, int, str, int, int, list[dict], float]:
    lba, passes, tier_now, pid, pz_r, qz_r, msf_hex, msf_valid, was_slip, collect_trace = args
    pipeline = SolverPipeline(max_tier=tier_now)
    ctx: dict = {"debug_trace": []} if collect_trace else {}
    t0 = time.perf_counter()
    ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=tier_now, context=ctx)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (
        lba,
        bool(ok),
        pz,
        qz,
        bytes(sec),
        strat,
        pid,
        pz_r,
        qz_r,
        msf_hex,
        msf_valid,
        was_slip,
        ctx.get("debug_trace", []),
        elapsed_ms,
    )


def _solve_updated_lbas_parallel(
    db: RecoveryStateDB,
    iso_fh,
    logger: RecoveryLogger | None,
    updated_items: list[tuple[int, int, int, int, str, int, int]],
    tier_now: int,
    workers: int,
    phase: str,
    round_num: int,
    direction: str,
    kbps: int,
    lba_to_file: dict[int, str] | None = None,
) -> tuple[int, float]:
    """Solve updated LBAs in parallel across CPU cores while writing verified sectors to ISO."""
    if not updated_items:
        return 0, 0.0
    t_s0 = time.perf_counter()
    collect_trace = bool(logger and logger.debug_enabled)
    tasks = []
    pass_len_by_lba = {}
    for lba, pid, pz_r, qz_r, msf_hex, msf_valid, was_slip in updated_items:
        passes = db.get_raw_passes(lba)
        if not passes:
            continue
        pass_len_by_lba[lba] = len(passes)
        tasks.append((lba, passes, tier_now, pid, pz_r, qz_r, msf_hex, msf_valid, was_slip, collect_trace))
    if not tasks:
        return 0, 0.0

    n_workers = workers if workers > 0 else min(8, os.cpu_count() or 1)
    newly_ok = 0
    completed = 0
    last_hb = time.monotonic()
    handled_lbas: set[int] = set()
    prog_step = max(1, len(tasks) // 10)
    file_map = lba_to_file or {}

    def _handle_round_result(
        item: tuple[int, bool, int, int, bytes, str, int, int, int, str, int, int, list[dict], float],
    ) -> None:
        nonlocal newly_ok, completed, last_hb
        (
            lba,
            ok,
            pz,
            qz,
            sec_b,
            strat,
            pid,
            pz_r,
            qz_r,
            msf_hex,
            msf_valid,
            was_slip,
            trace,
            solve_ms,
        ) = item
        handled_lbas.add(lba)
        completed += 1
        pass_cnt = pass_len_by_lba.get(lba, 1)
        _record_solve_telemetry(db, logger, lba, tier_now, pass_cnt, {"debug_trace": trace})
        verified_ok = db.upsert_solution(lba, sec_b, ok, pz, qz, pid, strat, iso_fh=iso_fh)
        if verified_ok:
            newly_ok += 1
        if logger and logger.debug_enabled:
            db.record_pass_metric(
                lba,
                pid,
                format_timestamp(),
                phase,
                round_num,
                direction,
                kbps,
                pz_r,
                qz_r,
                msf_hex,
                msf_valid,
                was_slip,
                solve_ms,
            )
            lba_file = file_map.get(lba, "-")
            now_m = time.monotonic()
            if verified_ok:
                logger.debug(
                    f"  [Solve {phase} r{round_num}] LBA {lba} VERIFIED",
                    file=lba_file,
                    strategy=strat,
                    p_ok=pz,
                    q_ok=qz,
                    passes=pass_cnt,
                    ms=round(solve_ms, 1),
                )
                last_hb = now_m
            elif len(tasks) <= 24:
                logger.debug(
                    f"  [Solve {phase} r{round_num}] LBA {lba} PARTIAL",
                    file=lba_file,
                    strategy=strat,
                    p_ok=pz,
                    q_ok=qz,
                    passes=pass_cnt,
                    ms=round(solve_ms, 1),
                )
                last_hb = now_m
            elif completed % prog_step == 0 or (now_m - last_hb >= 3.0) or completed == len(tasks):
                logger.debug(
                    f"  [Solve {phase} r{round_num}] progress {completed}/{len(tasks)}",
                    verified=newly_ok,
                    last_lba=lba,
                    file=lba_file,
                    strategy=strat,
                )
                last_hb = now_m
        elif logger and (time.monotonic() - last_hb >= 15.0):
            last_hb = time.monotonic()
            logger.info(
                f"  [Solve {phase} r{round_num}] {completed}/{len(tasks)} solved (+{newly_ok} verified)"
            )

    if n_workers > 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
                futs = [pool.submit(_round_solve_worker, t) for t in tasks]
                for fut in as_completed(futs):
                    _handle_round_result(fut.result())
        except Exception as exc:
            if logger:
                logger.debug("Per-round ProcessPoolExecutor fallback to serial", error=str(exc))
            for t in tasks:
                if t[0] not in handled_lbas:
                    _handle_round_result(_round_solve_worker(t))
    else:
        for t in tasks:
            _handle_round_result(_round_solve_worker(t))

    return newly_ok, time.perf_counter() - t_s0


def select_adaptive_focus_speeds(
    db: RecoveryStateDB, lbas: list[int], focus_passes: int
) -> list[int]:
    """Select an adaptive per-speed read schedule favoring top historical speeds for `lbas`."""
    if focus_passes <= 0:
        return []
    fallback = [SPEEDS_MULTI[i % len(SPEEDS_MULTI)] for i in range(focus_passes)]
    if not lbas:
        return fallback

    placeholders = ",".join("?" for _ in lbas)
    rows = db.conn.execute(
        f"SELECT lba, speed_kbps, MAX(p_ok), MAX(q_ok), AVG(p_ok), AVG(q_ok) "
        f"FROM pass_metrics WHERE lba IN ({placeholders}) GROUP BY lba, speed_kbps",
        tuple(lbas),
    ).fetchall()
    if not rows:
        return fallback

    allowed = set(SPEEDS_MULTI)
    per_speed: dict[int, list[float]] = {s: [] for s in SPEEDS_MULTI}
    for _, spd, max_p, max_q, avg_p, avg_q in rows:
        if spd in allowed:
            sc = float(max_p) + 2.0 * float(max_q) + 0.5 * (float(avg_p) + 2.0 * float(avg_q))
            per_speed[int(spd)].append(sc)

    avg_scores = {
        s: (sum(vals) / len(vals)) if vals else -1.0 for s, vals in per_speed.items()
    }
    if all(v < 0.0 for v in avg_scores.values()):
        return fallback

    ranked_speeds = sorted(
        SPEEDS_MULTI,
        key=lambda s: (avg_scores[s], -SPEEDS_MULTI.index(s)),
        reverse=True,
    )
    top3 = ranked_speeds[:3]
    rest = ranked_speeds[3:] or top3

    explore_n = min(len(SPEEDS_MULTI), focus_passes // 3)
    schedule = [ranked_speeds[i % len(ranked_speeds)] for i in range(explore_n)]
    top_i = 0
    rest_i = 0
    for step in range(focus_passes - explore_n):
        if step % 4 == 3:
            schedule.append(rest[rest_i % len(rest)])
            rest_i += 1
        else:
            schedule.append(top3[top_i % len(top3)])
            top_i += 1
    return schedule


def run_tiered_optical_recovery(
    drive: OpticalDrive,
    db: RecoveryStateDB,
    iso_out: Path,
    target_lbas: list[int],
    total_sectors: int,
    max_passes: int,
    max_tier: int,
    phase: str = "all",
    focus_passes: int = 0,
    workers: int = 0,
    extents: list[FileExtent] | None = None,
    logger: RecoveryLogger | None = None,
) -> None:
    """Escalate across decoupled optical reads + parallel Tier 1/2 solving, focused bursts, and Tier 3."""
    if not target_lbas:
        return
    pass_counts = db.get_max_pass_ids()
    lba_to_file = _build_lba_to_file_map(extents)

    def _read_with_slip_guard(
        lba: int, raw_2352: bytes, c2_294: bytes, direction: str, seed: int
    ) -> tuple[bool, bytes, bytes, bool]:
        """Return `(accepted, raw_2352, c2_294, was_slip)` rejecting drive -1 LBA frame slips."""
        if not rspc.is_frame_slip(lba, raw_2352):
            return True, raw_2352, c2_294, False
        drive.directional_seek_flush(lba, direction, total_sectors, offset_seed=seed + 13)
        st_r, _, res_r, raw_r = drive.read_cd_raw(lba, 1, with_c2=True)
        if st_r == 0 and res_r == 0 and not rspc.is_frame_slip(lba, raw_r[:2352]):
            return True, raw_r[:2352], raw_r[2352:2646], True
        return False, raw_2352, c2_294, True

    def _store_checked_pass(
        lba: int,
        r2352: bytes,
        rc2: bytes,
        direction: str,
        seed: int,
        round_num: int,
        slips_counter: list[int],
        updated_items: list[tuple[int, int, int, int, str, int, int]],
    ) -> tuple[int, int, int, bool, bool] | None:
        accepted, r2352, rc2, was_slip = _read_with_slip_guard(lba, r2352, rc2, direction, seed)
        if was_slip:
            slips_counter[0] += 1
        if not accepted:
            if logger is not None:
                logger.log_event("optical", "frame_slip", lba=lba, phase=phase, round_idx=round_num)
            return None
        pid = pass_counts.get(lba, -1) + 1
        pass_counts[lba] = pid
        pz_r, qz_r = db.insert_raw_pass(lba, pid, r2352, rc2)
        msf_ok = r2352[12:16] == rspc.lba_header(lba)
        updated_items.append(
            (lba, pid, pz_r, qz_r, r2352[12:16].hex(), int(msf_ok), int(was_slip))
        )
        return pid, pz_r, qz_r, msf_ok, was_slip

    with iso_out.open("r+b") as iso_fh:
        for round_idx in range(max_passes):
            verified = db.get_verified_set()
            remaining = [lba for lba in target_lbas if lba not in verified]
            if not remaining:
                msg = f"[{phase}] All target sectors mathematically verified."
                logger.info(msg) if logger else print(msg, flush=True)
                break

            next_pid = min((pass_counts.get(lba, -1) + 1 for lba in remaining), default=round_idx)
            if next_pid < 6:
                direction = "FWD"
                kbps = 0xFFFF
                lead = 2 + ((next_pid * 3) % 15)
            else:
                direction = "FWD" if (next_pid % 2 == 0) else "REV"
                kbps = SPEEDS_MULTI[(next_pid - 6) % len(SPEEDS_MULTI)]
                lead = LEADS_WIDE[(next_pid * 5) % len(LEADS_WIDE)]

            # Decouple heavy Tier 2: run Tier 1 every round, Tier 2 every 3rd round & final round
            tier_now = min(2, max_tier) if ((round_idx + 1) % 3 == 0 or round_idx == max_passes - 1) else 1

            if logger and logger.debug_enabled:
                logger.debug(
                    f"  [Optical {phase} {round_idx + 1}/{max_passes} start]",
                    direction=direction,
                    kbps=kbps,
                    lead=lead,
                    tier=f"T{tier_now}",
                    remaining=len(remaining),
                )

            drive.set_speed(kbps)
            t_r = time.time()
            slips_counter = [0]
            updated_items: list[tuple[int, int, int, int, str, int, int]] = []
            batch_count = 0

            # 1. Stream optical reads continuously without blocking on CPU decoding
            t_rd_start = time.perf_counter()
            last_hb = time.monotonic()
            if len(remaining) <= 24:
                batch_count = len(remaining)
                for idx_l, lba in enumerate(remaining, 1):
                    drive.directional_seek_flush(lba, direction, total_sectors, offset_seed=next_pid)
                    s_lba = max(0, lba - lead)
                    cnt = min(drive.max_batch_sectors, max(lead + 4, 4), total_sectors - s_lba)
                    st, _, res, raw = drive.read_cd_raw(s_lba, cnt, with_c2=True)
                    if st != 0 or res != 0:
                        if logger and logger.debug_enabled:
                            logger.debug(
                                f"  [Optical {phase} r{round_idx + 1}] read {idx_l}/{len(remaining)} LBA {lba} ERR",
                                file=lba_to_file.get(lba, "-"),
                                st=st,
                                resid=res,
                            )
                        continue
                    off = (lba - s_lba) * 2646
                    stored = _store_checked_pass(
                        lba,
                        raw[off : off + 2352],
                        raw[off + 2352 : off + 2646],
                        direction,
                        next_pid,
                        round_idx + 1,
                        slips_counter,
                        updated_items,
                    )
                    if logger and logger.debug_enabled:
                        lba_file = lba_to_file.get(lba, "-")
                        if stored is not None:
                            _, pz_r, qz_r, _, was_slip = stored
                            logger.debug(
                                f"  [Optical {phase} r{round_idx + 1}] read {idx_l}/{len(remaining)} LBA {lba}",
                                file=lba_file,
                                p_ok=pz_r,
                                q_ok=qz_r,
                                slip=int(was_slip),
                            )
                        else:
                            logger.debug(
                                f"  [Optical {phase} r{round_idx + 1}] read {idx_l}/{len(remaining)} LBA {lba} SLIP_REJECTED",
                                file=lba_file,
                            )
            else:
                batches = _build_batches(remaining, lead, drive.max_batch_sectors, total_sectors)
                if direction == "REV":
                    batches = list(reversed(batches))
                batch_count = len(batches)
                batch_step = max(1, batch_count // 10)
                for b_idx, (s_lba, cnt, covered) in enumerate(batches):
                    if b_idx == 0 or (direction == "REV" and len(batches) <= 60 and b_idx % 8 == 0):
                        drive.directional_seek_flush(covered[0], direction, total_sectors, offset_seed=next_pid + b_idx)
                    st, _, res, raw = drive.read_cd_raw(s_lba, cnt, with_c2=True)
                    if st != 0 or res != 0:
                        continue
                    for lba in covered:
                        off = (lba - s_lba) * 2646
                        _store_checked_pass(
                            lba,
                            raw[off : off + 2352],
                            raw[off + 2352 : off + 2646],
                            direction,
                            next_pid,
                            round_idx + 1,
                            slips_counter,
                            updated_items,
                        )
                    now_m = time.monotonic()
                    if logger and logger.debug_enabled:
                        if (b_idx + 1) % batch_step == 0 or (now_m - last_hb >= 3.0) or (b_idx + 1 == batch_count):
                            logger.debug(
                                f"  [Optical {phase} r{round_idx + 1}] batch {b_idx + 1}/{batch_count}",
                                lba_range=f"{covered[0]}..{covered[-1]}",
                                captured=len(updated_items),
                                slips=slips_counter[0],
                                elapsed_s=round(time.perf_counter() - t_rd_start, 1),
                            )
                            last_hb = now_m
                    elif logger and (now_m - last_hb >= 15.0):
                        last_hb = now_m
                        logger.info(
                            f"  [Optical {phase} r{round_idx + 1}] reading batch {b_idx + 1}/{batch_count} "
                            f"captured={len(updated_items)}/{len(remaining)} ({time.perf_counter() - t_rd_start:.1f}s)"
                        )
            db.conn.commit()
            read_s = time.perf_counter() - t_rd_start

            # 2. Solve all updated LBAs in parallel across CPU cores
            newly_ok, solve_s = _solve_updated_lbas_parallel(
                db,
                iso_fh,
                logger,
                updated_items,
                tier_now,
                workers,
                phase,
                round_idx + 1,
                direction,
                kbps,
                lba_to_file=lba_to_file,
            )

            tot_s = time.time() - t_r
            round_slips = slips_counter[0]
            db.record_round_log(
                format_timestamp(),
                phase,
                round_idx + 1,
                direction,
                kbps,
                lead,
                len(remaining),
                batch_count,
                newly_ok,
                round_slips,
                round(read_s, 3),
                round(solve_s, 3),
                round(tot_s, 3),
            )
            db.conn.commit()
            iso_fh.flush()
            msg = (
                f"  [Optical {phase} {round_idx + 1}/{max_passes}] {direction} {kbps}KB/s lead={lead} T{tier_now} "
                f"rem={len(remaining)} -> +{newly_ok} verified | slips={round_slips} | "
                f"read={read_s:.1f}s solve={solve_s:.1f}s ({tot_s:.1f}s)"
            )
            logger.info(msg) if logger else print(msg, flush=True)

        # Adaptive Focused Directional Burst for remaining <=40 high-priority media LBAs (like scan_lphoto_6.py)
        still_rem = [lba for lba in target_lbas if lba not in db.get_verified_set()]
        if focus_passes > 0 and 1 <= len(still_rem) <= 40:
            focus_speed_schedule = select_adaptive_focus_speeds(db, still_rem, focus_passes)
            msg_b = (
                f"[{phase} Focused Directional Burst] {focus_passes} FWD/REV reads on {len(still_rem)} stubborn LBAs "
                f"(speeds={focus_speed_schedule[:6]}...)..."
            )
            logger.info(msg_b) if logger else print(msg_b, flush=True)
            t_b0 = time.time()
            last_hb_f = time.monotonic()
            last_burst_items: dict[int, tuple[int, int, int, int, str, int, int]] = {}
            for f_idx in range(focus_passes):
                direction = "FWD" if (f_idx % 2 == 0) else "REV"
                kbps = focus_speed_schedule[f_idx]
                lead = LEADS_WIDE[(f_idx * 5) % len(LEADS_WIDE)]
                drive.set_speed(kbps)
                pass_cap = 0
                best_p = 0
                best_q = 0
                for lba in still_rem:
                    drive.directional_seek_flush(lba, direction, total_sectors, offset_seed=f_idx)
                    s_lba = max(0, lba - lead)
                    cnt = min(drive.max_batch_sectors, max(lead + 4, 4), total_sectors - s_lba)
                    st, _, res, raw = drive.read_cd_raw(s_lba, cnt, with_c2=True)
                    if st != 0 or res != 0:
                        continue
                    off = (lba - s_lba) * 2646
                    r2352, rc2 = raw[off : off + 2352], raw[off + 2352 : off + 2646]
                    accepted, r2352, rc2, was_slip = _read_with_slip_guard(lba, r2352, rc2, direction, f_idx + 100)
                    if not accepted:
                        continue
                    pid = pass_counts.get(lba, -1) + 1
                    pass_counts[lba] = pid
                    pz_r, qz_r = db.insert_raw_pass(lba, pid, r2352, rc2)
                    pass_cap += 1
                    best_p = max(best_p, pz_r)
                    best_q = max(best_q, qz_r)
                    msf_hex = r2352[12:16].hex()
                    msf_valid = int(r2352[12:16] == rspc.lba_header(lba))
                    last_burst_items[lba] = (lba, pid, pz_r, qz_r, msf_hex, msf_valid, int(was_slip))
                    if logger and logger.debug_enabled:
                        db.record_pass_metric(
                            lba,
                            pid,
                            format_timestamp(),
                            f"{phase}_focus",
                            f_idx + 1,
                            direction,
                            kbps,
                            pz_r,
                            qz_r,
                            msf_hex,
                            msf_valid,
                            int(was_slip),
                            0.0,
                        )
                db.conn.commit()
                now_f = time.monotonic()
                if logger and logger.debug_enabled:
                    logger.debug(
                        f"  [{phase} Focus {f_idx + 1}/{focus_passes}] {direction} {kbps}KB/s lead={lead}",
                        lbas=len(still_rem),
                        captured=pass_cap,
                        best_p=best_p,
                        best_q=best_q,
                        elapsed_s=round(time.time() - t_b0, 1),
                    )
                    last_hb_f = now_f
                elif logger and ((f_idx + 1) % 4 == 0 or (now_f - last_hb_f >= 10.0) or f_idx + 1 == focus_passes):
                    last_hb_f = now_f
                    logger.info(
                        f"  [{phase} Focus {f_idx + 1}/{focus_passes}] {direction} {kbps}KB/s "
                        f"captured={pass_cap}/{len(still_rem)} best=p{best_p}/q{best_q} ({time.time() - t_b0:.1f}s)"
                    )
            burst_solved, _ = _solve_updated_lbas_parallel(
                db,
                iso_fh,
                logger,
                list(last_burst_items.values()),
                min(2, max_tier),
                workers,
                f"{phase}_focus",
                focus_passes,
                "MIX",
                0,
                lba_to_file=lba_to_file,
            )
            db.conn.commit()
            iso_fh.flush()
            msg_bd = (
                f"[{phase} Focused Burst Done] +{burst_solved}/{len(still_rem)} verified after {focus_passes} passes "
                f"in {time.time() - t_b0:.1f}s"
            )
            logger.info(msg_bd) if logger else print(msg_bd, flush=True)

    _run_tier3_phase(db, iso_out, target_lbas, max_tier, workers, extents or [], phase, logger)


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
    logger: RecoveryLogger | None = None,
) -> list[FileRecoveryResult]:
    """Extract target files, run offline solver if requested, apply format-level repair, and validate."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    files = [e for e in extents if not e.is_dir and _match_category(e, categories)]
    ext_by_path = {e.path: e for e in extents if not e.is_dir}
    lba_to_file = _build_lba_to_file_map(extents)
    solved_map = db.get_solved_map()
    pipeline = SolverPipeline(max_tier=max_tier)

    known_masks: dict[int, set[int]] = {}
    if run_offline_solve:
        bad_targets = sorted({lba for f in files for lba in f.bad_lbas if solved_map.get(lba, (0,))[0] != 1})
        if bad_targets:
            msg_os = f"[Offline Solver] Checking {len(bad_targets)} unverified target LBAs in SQLite..."
            logger.info(msg_os) if logger else print(msg_os, flush=True)
            off_step = max(1, len(bad_targets) // 10)
            last_hb_os = time.monotonic()
            with iso_out.open("r+b") as iso_fh:
                for idx, lba in enumerate(bad_targets, 1):
                    passes = db.get_raw_passes(lba)
                    lba_file = lba_to_file.get(lba, "-")
                    if not passes:
                        if logger and logger.debug_enabled:
                            logger.debug(
                                f"  [Offline Solver {idx}/{len(bad_targets)}] LBA {lba} SKIPPED (no raw passes)",
                                file=lba_file,
                            )
                        continue
                    ctx: dict = {"debug_trace": []} if (logger and logger.debug_enabled) else {}
                    t_os0 = time.perf_counter()
                    ok, pz, qz, sec, strat = pipeline.solve_sector(lba, passes, max_tier=max_tier, context=ctx)
                    elapsed_ms = (time.perf_counter() - t_os0) * 1000.0
                    _record_solve_telemetry(db, logger, lba, max_tier, len(passes), ctx)
                    if "known_mask_2048" in ctx:
                        known_masks[lba] = ctx["known_mask_2048"]
                    verified_ok = db.upsert_solution(
                        lba, sec, ok, pz, qz, len(passes) - 1, strat, iso_fh=iso_fh
                    )
                    trace = ctx.get("debug_trace", [])
                    now_m = time.monotonic()
                    if verified_ok:
                        m_ok = (
                            f"  [Offline Solver {idx}/{len(bad_targets)}] LBA {lba} VERIFIED by {strat} "
                            f"({elapsed_ms:.0f}ms)"
                        )
                        if logger:
                            logger.info(m_ok, file=lba_file)
                            if trace and logger.debug_enabled:
                                logger.debug(
                                    f"  [Offline Solver trace] LBA {lba}",
                                    file=lba_file,
                                    solvers=_format_trace_summary(trace),
                                )
                        else:
                            print(f"{m_ok} | file={lba_file}", flush=True)
                        last_hb_os = now_m
                    elif logger and logger.debug_enabled:
                        logger.debug(
                            f"  [Offline Solver {idx}/{len(bad_targets)}] LBA {lba} PARTIAL",
                            file=lba_file,
                            p_ok=pz,
                            q_ok=qz,
                            strategy=strat,
                            passes=len(passes),
                            ms=round(elapsed_ms, 1),
                            solvers=_format_trace_summary(trace),
                        )
                        last_hb_os = now_m
                    elif logger and (idx % off_step == 0 or (now_m - last_hb_os >= 10.0)):
                        last_hb_os = now_m
                        logger.info(
                            f"  [Offline Solver {idx}/{len(bad_targets)}] LBA {lba} partial "
                            f"(p={pz}/86 q={qz}/52, {strat})",
                            file=lba_file,
                        )
                db.conn.commit()
                iso_fh.flush()
            solved_map = db.get_solved_map()

    results: list[FileRecoveryResult] = []
    n_intact = 0
    n_ver = 0
    n_rep = 0
    n_part = 0
    t_ext0 = time.time()
    last_hb_ext = time.monotonic()

    with iso_out.open("rb") as iso_f:
        def read_file_bytes(ext_obj: FileExtent) -> bytes:
            iso_f.seek(ext_obj.start_lba * 2048)
            return iso_f.read(ext_obj.size_bytes)

        for f_idx, ext_obj in enumerate(files, 1):
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
                n_intact += 1
            elif not unverified_bad:
                rec_class = "RECOVERED_100PCT_VERIFIED"
                n_ver += 1
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
                if logger and logger.debug_enabled:
                    iv_str = ",".join(f"{s}..{e}" for s, e in bad_intervals[:6])
                    if len(bad_intervals) > 6:
                        iv_str += f"...({len(bad_intervals)})"
                    logger.debug(
                        f"  [Format Repair start] {ext_obj.path}",
                        handler=handler.name,
                        unverified_lbas=",".join(str(x) for x in unverified_bad),
                        intervals=iv_str or "none",
                        ref=ref_used or "none",
                    )
                repaired, rep_status, rep_meta = handler.reconstruct(
                    data, bad_intervals, reference_data=ref_data, context={"known_mask": True}
                )
                if rep_status.endswith("RESYNC") or rep_status.endswith("REPAIRED"):
                    data = repaired
                    is_valid, fmt_status, meta = handler.validate(data)
                    rec_class = rep_status
                    meta.update(rep_meta)
                    n_rep += 1
                else:
                    rec_class = "PARTIAL_RSPC_RECONSTRUCTED"
                    n_part += 1

                msg_rep = (
                    f"  [Format Repair {f_idx}/{len(files)}] {ext_obj.path} -> {rec_class} ({fmt_status})"
                )
                if logger:
                    logger.info(
                        msg_rep,
                        handler=handler.name,
                        bad_lbas=f"{len(verified_bad)}/{len(ext_obj.bad_lbas)}",
                        ref=ref_used or "none",
                    )
                    if logger.debug_enabled:
                        safe_meta = _format_safe_meta(rep_meta)
                        logger.debug(
                            f"  [Format Repair meta] {ext_obj.path}",
                            rep_status=rep_status,
                            details=safe_meta or "none",
                        )
                else:
                    print(f"{msg_rep} | handler={handler.name}", flush=True)

            rel = ext_obj.path.lstrip("/")
            dest = extract_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

            if logger and logger.debug_enabled:
                logger.debug(
                    f"  [Extract {f_idx}/{len(files)}] {ext_obj.path} -> {rec_class}",
                    handler=handler.name,
                    status=fmt_status,
                    bad_lbas=f"{len(verified_bad)}/{len(ext_obj.bad_lbas)}",
                    size=ext_obj.size_bytes,
                )
            elif logger:
                now_ext = time.monotonic()
                if f_idx % 100 == 0 or (now_ext - last_hb_ext >= 5.0):
                    last_hb_ext = now_ext
                    logger.info(
                        f"  [Extract {f_idx}/{len(files)}] intact={n_intact} verified={n_ver} "
                        f"repaired={n_rep} partial={n_part} ({time.time() - t_ext0:.1f}s)"
                    )

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
    msg_done = (
        f"[Stages 5-6 Complete] extracted={len(results)} | intact={n_intact} | "
        f"verified_100pct={n_ver} | repaired={n_rep} | partial={n_part} | elapsed={time.time() - t_ext0:.1f}s"
    )
    logger.info(msg_done) if logger else print(msg_done, flush=True)
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
    focus_passes: int = 24,
    max_tier: int = 3,
    workers: int = 0,
    debug: bool = False,
    report_out: Path | None = None,
    save_baseline: bool = False,
) -> dict:
    """Execute the full 6-stage automated CD ripping, solving, repair, and validation pipeline."""
    t_pipeline0 = time.time()
    started_at = format_timestamp()
    timing: dict[str, float | str] = {"started_at": started_at}

    cats = categories if categories is not None else ["all"]
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

    user_iso_out = iso_out
    iso_out = iso_out or (session_dir / f"{resolved_name.lower()}_recovered.iso")
    db_path = db_path or (logs_dir / "raw_passes.sqlite3")
    extract_dir = extract_dir or (session_dir / "extracted")
    report_out = report_out or (logs_dir / "auto_recovery_report.json")

    db = RecoveryStateDB(db_path)
    logger = RecoveryLogger(logs_dir, db=db, debug=debug)
    try:
        logger.info(f"[Session] name={resolved_name} -> {session_dir}", debug=debug)
        bad_lbas: list[int] = []
        if not no_hardware:
            t_s1 = time.time()
            with OpticalDrive(device) as drive:
                drive.set_speed(0xFFFF)
                caps = drive.caps
                logger.info(
                    f"[Stage 1: Hardware Probe] {caps.sg_device} ({caps.sr_device}) | "
                    f"{caps.vendor_model} | C2={caps.supports_c2} | max_batch={caps.max_batch_sectors}"
                )
                disc_mode = "mode1"
                tracks: list[dict] = []
                if user_iso_out is None and seed_iso is None:
                    disc_mode, tracks, total_sectors = drive.detect_disc_mode()
                else:
                    total_sectors = drive.read_total_volume_sectors()
                if disc_mode == "mode2":
                    from src.pipeline.vcd_mode2 import run_mode2_pipeline

                    return run_mode2_pipeline(
                        drive=drive,
                        tracks=tracks,
                        total_sectors=total_sectors,
                        recovered_root=recovered_root,
                        session_name=resolved_name,
                        bad_log=bad_log,
                        full_rip=full_rip,
                        categories=cats,
                        max_passes=max_passes,
                        focus_passes=focus_passes,
                        max_tier=max_tier,
                        workers=workers,
                        debug=debug,
                        report_out=report_out,
                        existing_db=db,
                        existing_logger=logger,
                        t_stage1_start=t_s1,
                        t_pipeline_start=t_pipeline0,
                        save_baseline=save_baseline,
                    )
                iso_existed = iso_out.exists() and iso_out.stat().st_size > 0
                prepare_output_iso(iso_out, seed_iso, total_sectors)
                timing["stage1_probe_s"] = round(time.time() - t_s1, 3)
                if full_rip or (seed_iso is None and not iso_existed):
                    t_s2 = time.time()
                    logger.info(f"[Stage 2: Pass-0 Raw+C2 Rip] Ripping {total_sectors} sectors...")
                    bad_lbas = pass0_full_rip(drive, db, iso_out, total_sectors, logger=logger)
                    timing["stage2_pass0_s"] = round(time.time() - t_s2, 3)
                    if save_baseline and seed_iso is None:
                        if preserve_baseline_image(iso_out, session_dir / "baseline.iso"):
                            logger.info(f"[Baseline Preserved] {session_dir / 'baseline.iso'} (0444)")
        else:
            logger.info("[Stage 1: Offline Mode] Skipping optical device probe.")
            bin_fallback = session_dir / f"{resolved_name.lower()}_recovered.bin"
            if user_iso_out is None and seed_iso is None and bin_fallback.exists() and not iso_out.exists():
                from src.pipeline.vcd_mode2 import run_mode2_pipeline

                total_bin_sectors = bin_fallback.stat().st_size // 2352
                return run_mode2_pipeline(
                    drive=None,
                    tracks=[],
                    total_sectors=total_bin_sectors,
                    recovered_root=recovered_root,
                    session_name=resolved_name,
                    bad_log=bad_log,
                    full_rip=False,
                    categories=cats,
                    max_passes=0,
                    focus_passes=0,
                    max_tier=max_tier,
                    workers=workers,
                    debug=debug,
                    report_out=report_out,
                    existing_db=db,
                    existing_logger=logger,
                    t_stage1_start=t_pipeline0,
                    t_pipeline_start=t_pipeline0,
                    save_baseline=save_baseline,
                )
            prepare_output_iso(iso_out, seed_iso, None)
            total_sectors = iso_out.stat().st_size // 2048

        t_s3 = time.time()
        if bad_log and bad_log.exists():
            bad_lbas = sorted(set(bad_lbas) | set(load_wincdemu_bad_lbas(bad_log)))
        bad_lbas = sorted(set(bad_lbas) | set(db.all_tracked_bad_lbas()))

        reader = iso9660.make_iso_sector_reader(iso_out)
        try:
            pvd, extents = iso9660.parse_iso9660_tree(reader, bad_lbas=set(bad_lbas))
        finally:
            reader.fh.close()  # type: ignore[attr-defined]

        etiology = iso9660.classify_damage_etiology(bad_lbas)
        timing["stage3_triage_s"] = round(time.time() - t_s3, 3)
        logger.info(
            f"[Stage 3: Triage] Vol={pvd['volume_id']!r} ({pvd['volume_sectors']} sectors), "
            f"files={sum(1 for e in extents if not e.is_dir)}, "
            f"bad_sectors={etiology.total_bad_sectors}, etiology={etiology.classification}"
        )

        media_lbas, bulk_lbas = partition_targets_by_priority(extents, cats)
        verified_now = db.get_verified_set()
        media_rem = sum(1 for lba in media_lbas if lba not in verified_now)
        bulk_rem = sum(1 for lba in bulk_lbas if lba not in verified_now)
        target_files = [e for e in extents if not e.is_dir and _match_category(e, cats)]
        logger.info(
            f"[Stage 3: Target Plan] categories={','.join(cats)} | "
            f"media_lbas={len(media_lbas)} (unverified={media_rem}) | "
            f"bulk_lbas={len(bulk_lbas)} (unverified={bulk_rem}) | "
            f"target_files={len(target_files)}"
        )
        if logger.debug_enabled:
            dir_stats: dict[str, dict[str, int | set[int]]] = {}
            for ext_obj in target_files:
                st_d = dir_stats.setdefault(
                    ext_obj.directory,
                    {"files": 0, "bad_files": 0, "bad_lbas": set(), "unverified_lbas": set()},
                )
                st_d["files"] = int(st_d["files"]) + 1
                if ext_obj.bad_lbas:
                    st_d["bad_files"] = int(st_d["bad_files"]) + 1
                    st_d["bad_lbas"].update(ext_obj.bad_lbas)  # type: ignore[union-attr]
                    st_d["unverified_lbas"].update(  # type: ignore[union-attr]
                        lba for lba in ext_obj.bad_lbas if lba not in verified_now
                    )
            for d_name in sorted(dir_stats):
                st_d = dir_stats[d_name]
                logger.debug(
                    f"  [Stage 3 dir] {d_name}",
                    files=st_d["files"],
                    bad_files=st_d["bad_files"],
                    bad_lbas=len(st_d["bad_lbas"]),  # type: ignore[arg-type]
                    unverified_lbas=len(st_d["unverified_lbas"]),  # type: ignore[arg-type]
                )
        logger.log_event(
            "triage",
            "stage_complete",
            media_lbas=len(media_lbas),
            bulk_lbas=len(bulk_lbas),
        )

        if not no_hardware and (media_lbas or bulk_lbas):
            with OpticalDrive(device) as drive:
                if media_lbas:
                    t_m = time.time()
                    logger.info(f"[Stage 4A: High-Priority Media Recovery] {len(media_lbas)} LBAs")
                    run_tiered_optical_recovery(
                        drive,
                        db,
                        iso_out,
                        media_lbas,
                        total_sectors,
                        max_passes,
                        max_tier,
                        phase="media",
                        focus_passes=focus_passes,
                        workers=workers,
                        extents=extents,
                        logger=logger,
                    )
                    timing["stage4_media_s"] = round(time.time() - t_m, 3)
                if bulk_lbas:
                    t_b = time.time()
                    logger.info(f"[Stage 4B: Bulk File Recovery] {len(bulk_lbas)} LBAs")
                    run_tiered_optical_recovery(
                        drive,
                        db,
                        iso_out,
                        bulk_lbas,
                        total_sectors,
                        max_passes,
                        max_tier,
                        phase="bulk",
                        focus_passes=0,
                        workers=workers,
                        extents=extents,
                        logger=logger,
                    )
                    timing["stage4_bulk_s"] = round(time.time() - t_b, 3)

        t_s5 = time.time()
        logger.info(f"[Stages 5-6: Format Repair & Extraction -> {extract_dir}]")
        results = extract_and_repair_files(
            iso_out,
            extract_dir,
            db,
            extents,
            cats,
            run_offline_solve=no_hardware,
            max_tier=max_tier,
            logger=logger,
        )
        timing["stage5_6_repair_extract_s"] = round(time.time() - t_s5, 3)

        # Write state JSON, file manifest CSV, and summary report JSON
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
            elif "RESYNC" in r.recovery_class or "REPAIRED" in r.recovery_class:
                d["format_repaired"] += 1
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
            "session_dir": str(session_dir),
            "iso_out": str(iso_out),
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
        logger.close()
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
