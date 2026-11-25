"""Built-in optical disc inspection, ISO/BIN image inspection, and session integrity auditing."""

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from src.core import rspc
from src.filesystem import iso9660
from src.formats import FormatRegistry
from src.formats.media_handlers import verify_media_with_ffmpeg
from src.hardware.device import OpticalDrive


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _summarize_extents(pvd: dict, extents: list, max_files: int) -> dict:
    files = [e for e in extents if not e.is_dir]
    dirs = Counter(e.directory for e in files)
    exts = Counter((e.ext.lower() or "<none>") for e in files)
    sample = [
        {
            "path": e.path,
            "directory": e.directory,
            "start_lba": e.start_lba,
            "end_lba": e.end_lba,
            "sectors": e.sectors,
            "size_bytes": e.size_bytes,
            "priority": e.priority,
        }
        for e in sorted(files, key=lambda x: x.start_lba)[:max_files]
    ]
    return {
        "pvd": pvd,
        "file_count": len(files),
        "dir_count": sum(1 for e in extents if e.is_dir),
        "directories": dict(sorted(dirs.items())),
        "extensions": dict(sorted(exts.items())),
        "files": sample,
        "truncated_files": max(0, len(files) - len(sample)),
    }


def _probe_raw_sample(drive: OpticalDrive, lba: int) -> dict:
    st, sense, resid, raw = drive.read_cd_raw(lba, 1, with_c2=True)
    if st != 0 or resid != 0 or len(raw) < 2646:
        return {"lba": lba, "ok": False, "status": st, "sense": list(sense), "resid": resid}
    r2352 = raw[:2352]
    c2 = raw[2352:2646]
    c2_sum = sum(c2)
    mode = r2352[15]
    info: dict = {
        "lba": lba,
        "ok": True,
        "mode": mode,
        "hdr_hex": r2352[12:16].hex(),
        "c2_nonzero_bytes": sum(1 for b in c2 if b),
        "c2_sum": c2_sum,
    }
    if mode == 2:
        info["subhdr_hex"] = r2352[16:24].hex()
        info["form"] = 2 if (r2352[18] & 0x20) else 1
        v_ok, v_tag, pz, qz = rspc.verify_mode2_sector(lba, r2352, c2)
        info["verify_ok"] = v_ok
        info["verify_tag"] = v_tag
        info["p_ok"] = pz
        info["q_ok"] = qz
    elif mode == 1 and r2352[:12] == rspc.SYNC_12:
        sec_m1 = bytearray(r2352)
        pz, qz = rspc.count_pq(sec_m1)
        edc_ok = rspc.edc_residual(sec_m1[:2068]) == 0
        info["cd001"] = r2352[17:22] == b"CD001"
        info["edc_ok"] = edc_ok
        info["p_ok"] = pz
        info["q_ok"] = qz
    else:
        info["pcm_nonzero_bytes"] = sum(1 for b in r2352 if b)
    return info


def inspect_optical_disc(device: str | None = None, max_files: int = 50) -> dict:
    """Probe the inserted optical disc (TOC, CD-TEXT, MCN, Mode 1/2/CD-DA, PVD, file tree)."""
    drive_status = OpticalDrive.read_drive_status(device)
    with OpticalDrive(device) as drive:
        caps = drive.caps
        tracks = drive.read_toc()
        sessions = drive.read_toc_sessions()
        cd_text = drive.read_cd_text()
        mcn = drive.read_mcn()

        has_data_track = any(t.get("is_data") and t.get("track") != 170 for t in tracks)
        probe_lbas = [0, 16, 17, 18]
        for t in tracks:
            if t.get("track") != 170 and t["start_lba"] not in probe_lbas:
                probe_lbas.append(int(t["start_lba"]))

        samples = [_probe_raw_sample(drive, lba) for lba in probe_lbas[:8]]
        pvd_tree: dict = {}

        st16, _, res16, raw16 = drive.read_cd_raw(16, 1, with_c2=False)
        if not has_data_track and tracks:
            disc_mode = "cdda"
            leadout = next((int(t["start_lba"]) for t in tracks if t["track"] == 170), 0)
            total_sectors = leadout
        elif st16 == 0 and res16 == 0 and len(raw16) >= 30 and raw16[25:30] == b"CD001":
            disc_mode = "mode2"
            leadout = next((int(t["start_lba"]) for t in tracks if t["track"] == 170), 0)
            total_sectors = leadout or drive.read_total_volume_sectors()

            def _read_m2(lba: int) -> bytes:
                stx, _, resx, rx = drive.read_cd_raw(lba, 1, with_c2=False)
                return rx[24:2072] if stx == 0 and resx == 0 and len(rx) >= 2072 else b"\x00" * 2048

            pvd, extents = iso9660.parse_iso9660_tree(_read_m2)
            pvd_tree = _summarize_extents(pvd, extents, max_files)
        elif st16 == 0 and res16 == 0 and len(raw16) >= 22 and raw16[17:22] == b"CD001":
            disc_mode = "mode1"
            total_sectors = drive.read_total_volume_sectors()

            def _read_m1(lba: int) -> bytes:
                return drive.read_cooked_sector(lba) or (b"\x00" * 2048)

            pvd, extents = iso9660.parse_iso9660_tree(_read_m1)
            pvd_tree = _summarize_extents(pvd, extents, max_files)
        else:
            disc_mode = "unknown"
            total_sectors = next((int(t["start_lba"]) for t in tracks if t["track"] == 170), 0)

        return {
            "device": caps.sg_device if caps else "",
            "sr_device": caps.sr_device if caps else None,
            "vendor_model": caps.vendor_model if caps else "",
            "supports_c2": caps.supports_c2 if caps else False,
            "disc_mode": disc_mode,
            "total_sectors": total_sectors,
            "tracks": tracks,
            "sessions": sessions,
            "cd_text": cd_text,
            "mcn": mcn,
            "sector_samples": samples,
            "drive_status_before": drive_status,
            **pvd_tree,
        }


def inspect_image(image_path: Path, max_files: int = 50) -> dict:
    """Inspect a local .iso (2048 B/sec) or raw .bin (2352 B/sec Mode 1/Mode 2) image."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    size_bytes = image_path.stat().st_size

    with image_path.open("rb") as fh:
        fh.seek(16 * 2352)
        raw16 = fh.read(2352)
        fh.seek(16 * 2048)
        cooked16 = fh.read(2048)

    is_raw_2352 = (
        len(raw16) >= 30
        and raw16[0:12] == rspc.SYNC_12
        and (size_bytes % 2352 == 0 or size_bytes % 2048 != 0 or raw16[15] in (1, 2))
        and (raw16[25:30] == b"CD001" or raw16[17:22] == b"CD001")
    )

    if is_raw_2352:
        sector_size = 2352
        total_sectors = size_bytes // 2352
        if raw16[15] == 2 or raw16[25:30] == b"CD001":
            disc_mode = "mode2"
            user_off = 24
        else:
            disc_mode = "mode1_raw"
            user_off = 16
        fh_img = image_path.open("rb")

        def _read_raw_user(lba: int) -> bytes:
            fh_img.seek(lba * 2352 + user_off)
            d = fh_img.read(2048)
            return d if len(d) == 2048 else d + b"\x00" * (2048 - len(d))

        try:
            pvd, extents = iso9660.parse_iso9660_tree(_read_raw_user)
        finally:
            fh_img.close()
    elif len(cooked16) >= 6 and cooked16[1:6] == b"CD001":
        sector_size = 2048
        total_sectors = size_bytes // 2048
        disc_mode = "mode1"
        reader = iso9660.make_iso_sector_reader(image_path)
        try:
            pvd, extents = iso9660.parse_iso9660_tree(reader)
        finally:
            reader.fh.close()  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unable to locate ISO-9660 PVD (CD001) in {image_path}")

    return {
        "image_path": str(image_path),
        "size_bytes": size_bytes,
        "sector_size": sector_size,
        "total_sectors": total_sectors,
        "disc_mode": disc_mode,
        **_summarize_extents(pvd, extents, max_files),
    }


def _resolve_extracted_path(extract_dir: Path, row: dict) -> Path:
    saved = row.get("saved_path")
    if saved:
        sp = Path(saved)
        if sp.exists():
            return sp
        parts = sp.parts
        if "extracted" in parts:
            idx = len(parts) - 1 - parts[::-1].index("extracted")
            cand = extract_dir.joinpath(*parts[idx + 1 :])
            if cand.exists():
                return cand
    iso_rel = str(row.get("iso_path", "")).lstrip("/")
    return extract_dir / iso_rel


def _audit_sqlite(db_path: Path, report_out: dict) -> dict:
    if not db_path.exists():
        return {"present": False, "tables": {}, "q_parity_gate": {"checked": 0, "passed": 0, "failed": 0}}
    con = sqlite3.connect(db_path)
    try:
        tbls = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        counts = {t: int(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tbls}

        q_checked = 0
        q_passed = 0
        q_failed = 0
        q_fails: list[int] = []
        min_clean_ratio = 1.0

        solved_rows = con.execute(
            "SELECT lba, user_2048, verified FROM solved_sectors WHERE verified = 1"
        ).fetchall()
        for lba, user_2048, _ in solved_rows:
            rp_rows = con.execute(
                "SELECT raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
                (lba,),
            ).fetchall()
            if not rp_rows:
                continue
            raw0 = bytes(rp_rows[0][0])
            if len(raw0) < 2352 or raw0[15] != 1:
                continue
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            sec[16:2064] = bytes(user_2048)
            sec[2064:2068] = rspc.compute_edc(sec[:2064]).to_bytes(4, "little")
            for cw in rspc.P_IDX:
                ok_p, corrs_p = rspc.solve_2eras(sec, cw, [24, 25])
                if ok_p:
                    for pos, val in corrs_p:
                        sec[pos] = val
            for cw in rspc.Q_IDX:
                ok_q, corrs_q = rspc.solve_2eras(sec, cw, [43, 44])
                if ok_q:
                    for pos, val in corrs_q:
                        sec[pos] = val
            raw_passes = [(bytes(r[0]), bytes(r[1])) for r in rp_rows]
            ok_qp, q_match, q_chk = rspc.verify_physical_q_parity(sec, raw_passes)
            q_checked += 1
            if q_chk > 0:
                min_clean_ratio = min(min_clean_ratio, q_match / q_chk)
            if ok_qp:
                q_passed += 1
            else:
                q_failed += 1
                q_fails.append(lba)

        report_out["q_parity_gate"] = {
            "checked": q_checked,
            "passed": q_passed,
            "failed": q_failed,
            "min_clean_ratio": round(min_clean_ratio, 4) if q_checked else None,
            "failed_lbas": q_fails[:20],
        }
        return {"present": True, "tables": counts}
    finally:
        con.close()


def audit_session(
    session_dir: Path,
    run_ffmpeg: bool = False,
    compare_session: Path | None = None,
) -> dict:
    """Audit a recovered session directory (image, SQLite, Q-parity, file formats, optional FFmpeg/diff)."""
    session_dir = session_dir.resolve()
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    logs_dir = session_dir / "logs"
    extract_dir = (session_dir / "extracted").resolve()
    report_path = logs_dir / "auto_recovery_report.json"
    saved_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    primary_images = [
        p
        for p in sorted(session_dir.iterdir())
        if p.is_file() and p.suffix.lower() in (".iso", ".bin") and not p.name.startswith("baseline.")
    ]
    baselines = [
        p for p in sorted(session_dir.iterdir()) if p.is_file() and p.name.startswith("baseline.")
    ]

    result: dict = {
        "session_dir": str(session_dir),
        "disc_mode": saved_report.get("disc_mode", "mode2" if any(p.suffix == ".bin" for p in primary_images) else "mode1"),
        "images": [{"name": p.name, "size_bytes": p.stat().st_size} for p in primary_images],
        "baselines": [{"name": p.name, "size_bytes": p.stat().st_size} for p in baselines],
        "q_parity_gate": {"checked": 0, "passed": 0, "failed": 0},
    }
    result["sqlite"] = _audit_sqlite(logs_dir / "raw_passes.sqlite3", result)

    manifest_path = None
    for cand in (extract_dir / "manifest.csv", extract_dir / "photo_manifest.csv"):
        if cand.exists():
            manifest_path = cand
            break
    result["manifest_name"] = manifest_path.name if manifest_path else None

    file_entries: list[dict] = []
    if manifest_path is not None:
        with manifest_path.open("r", newline="", encoding="utf-8") as cf:
            for row in csv.DictReader(cf):
                fp = _resolve_extracted_path(extract_dir, row)
                file_entries.append({"row": row, "path": fp})
    elif extract_dir.exists():
        for fp in sorted(extract_dir.rglob("*")):
            if fp.is_file() and fp.name not in ("manifest.csv", "photo_manifest.csv"):
                file_entries.append({"row": {"iso_path": "/" + fp.relative_to(extract_dir).as_posix()}, "path": fp})

    passed = 0
    failed = 0
    missing = 0
    ffmpeg_checked = 0
    ffmpeg_failed = 0
    file_hashes: dict[str, str] = {}
    file_details: list[dict] = []

    media_exts = iso9660.AUDIO_EXTS | iso9660.VIDEO_EXTS
    for entry in file_entries:
        fp: Path = entry["path"].resolve()
        rel_key = (
            fp.relative_to(extract_dir).as_posix()
            if fp.exists() and extract_dir in fp.parents
            else str(entry["row"].get("iso_path", fp.name)).lstrip("/")
        )
        if not fp.exists():
            missing += 1
            failed += 1
            file_details.append({"path": rel_key, "exists": False, "valid": False, "status": "MISSING"})
            continue

        sha = _sha256_file(fp)
        file_hashes[rel_key] = sha
        with fp.open("rb") as fh:
            head = fh.read(4096)
        handler = FormatRegistry.get_handler(fp.suffix, head)
        valid, fmt_status, meta = handler.validate_file(fp)
        row = entry["row"]
        if (
            not valid
            and handler.name == "generic_binary"
            and fmt_status.startswith("ZERO_HOLES")
            and row.get("recovery_class") in ("INTACT_ORIGINAL", "RECOVERED_100PCT_VERIFIED")
            and int(row.get("bad_sectors_total", 0) or 0) == int(row.get("bad_sectors_verified", 0) or 0)
            and int(row.get("zero_holes_2k", 0) or 0) == 0
        ):
            valid = True
            fmt_status = "BINARY_OK (verified_zeros)"
        ff_ok = True
        ff_status = None
        if run_ffmpeg and (handler.name in ("audio", "video") or fp.suffix.upper() in media_exts):
            ff_ok, ff_status, ff_meta = verify_media_with_ffmpeg(fp)
            ffmpeg_checked += 1
            if not ff_ok:
                ffmpeg_failed += 1
            meta["ffmpeg_status"] = ff_status
            if "duration_s" in ff_meta:
                meta["duration_s"] = ff_meta["duration_s"]
        overall_ok = bool(valid and ff_ok)
        if overall_ok:
            passed += 1
        else:
            failed += 1
        file_details.append(
            {
                "path": rel_key,
                "size_bytes": fp.stat().st_size,
                "sha256": sha,
                "handler": handler.name,
                "valid": overall_ok,
                "status": fmt_status,
                "ffmpeg_status": ff_status,
            }
        )

    result["validation"] = {
        "total_files": len(file_entries),
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "ffmpeg_checked": ffmpeg_checked,
        "ffmpeg_failed": ffmpeg_failed,
        "files": file_details,
    }

    if compare_session is not None:
        cmp_ext = (compare_session.resolve() / "extracted").resolve()
        if not cmp_ext.exists():
            raise FileNotFoundError(f"Comparison extracted dir not found: {cmp_ext}")
        cmp_hashes: dict[str, str] = {}
        for fp in sorted(cmp_ext.rglob("*")):
            if fp.is_file() and fp.name not in ("manifest.csv", "photo_manifest.csv"):
                cmp_hashes[fp.relative_to(cmp_ext).as_posix()] = _sha256_file(fp)
        common = sorted(set(file_hashes) & set(cmp_hashes))
        mismatches = [k for k in common if file_hashes[k] != cmp_hashes[k]]
        result["compare"] = {
            "compare_session": str(compare_session),
            "common_files": len(common),
            "exact_match": len(common) - len(mismatches),
            "mismatches": mismatches,
            "only_in_primary": sorted(set(file_hashes) - set(cmp_hashes)),
            "only_in_compare": sorted(set(cmp_hashes) - set(file_hashes)),
        }

    return result


def format_drive_status(status: dict) -> str:
    p01 = status["mode_page_01"]
    p2a = status["mode_page_2a"]
    lines = [
        f"Drive: {status['sg_device']} ({status['sr_device']}) | {status['vendor_model']}",
        f"Mode Page 0x01: st={p01['status']} err_bits={p01['err_bits_hex']} retries={p01['retries_hex']} restored={p01['restored_ok']}",
        f"Mode Page 0x2A: st={p2a['status']} max_speed={p2a['max_speed_kbps']}KB/s cur_speed={p2a['cur_speed_kbps']}KB/s c2={p2a['c2']} cdda={p2a['cdda']}",
    ]
    return "\n".join(lines)


def format_image_or_disc_info(info: dict, is_disc: bool = False) -> str:
    lines = []
    if is_disc:
        lines.append(
            f"Disc: {info['device']} ({info['sr_device']}) | {info['vendor_model']} | mode={info['disc_mode']} | sectors={info['total_sectors']}"
        )
        for t in info.get("tracks", []):
            kind = "LEADOUT" if t["track"] == 170 else ("DATA" if t["is_data"] else "AUDIO")
            lines.append(f"  Track {t['track']:3d} ({kind:7s}): ctrl=0x{t['control']:x} start_lba={t['start_lba']}")
        for s in info.get("sector_samples", []):
            lines.append(
                f"  Sample LBA {s['lba']:6d}: ok={s['ok']} mode={s.get('mode')} hdr={s.get('hdr_hex')} c2={s.get('c2_sum')} tag={s.get('verify_tag', '-')}"
            )
    else:
        lines.append(
            f"Image: {info['image_path']} | mode={info['disc_mode']} | sector_size={info['sector_size']} | sectors={info['total_sectors']} | bytes={info['size_bytes']}"
        )

    pvd = info.get("pvd")
    if pvd:
        lines.append(
            f"PVD: volume_id={pvd['volume_id']!r} app={pvd['application_id']!r} sectors={pvd['volume_sectors']} enc={pvd['encoding']}"
        )
        lines.append(
            f"Tree: files={info['file_count']} dirs={info['dir_count']} extensions={info['extensions']}"
        )
        for f in info.get("files", []):
            lines.append(
                f"  {f['path']} | LBA {f['start_lba']}..{f['end_lba']} ({f['sectors']} sec, {f['size_bytes']} B, prio={f['priority']})"
            )
    return "\n".join(lines)


def format_session_audit(report: dict) -> str:
    val = report["validation"]
    qp = report["q_parity_gate"]
    lines = [
        f"Session: {report['session_dir']} | mode={report['disc_mode']} | manifest={report.get('manifest_name')}",
        f"Images: {', '.join(i['name'] for i in report['images']) or 'none'} | Baselines: {', '.join(b['name'] for b in report['baselines']) or 'none'}",
        f"SQLite tables: {report['sqlite'].get('tables', {})}",
        f"Physical Q-parity gate: {qp['passed']}/{qp['checked']} passed (failed={qp['failed']}, min_clean_ratio={qp.get('min_clean_ratio')})",
        f"Validation: {val['passed']}/{val['total_files']} passed | failed={val['failed']} | missing={val['missing']} | ffmpeg_checked={val['ffmpeg_checked']} (failed={val['ffmpeg_failed']})",
    ]
    for fd in val["files"]:
        if not fd["valid"] or len(val["files"]) <= 20:
            lines.append(
                f"  {fd['path']} | handler={fd.get('handler')} | valid={fd['valid']} | status={fd['status']} | ffmpeg={fd.get('ffmpeg_status') or '-'}"
            )
    if "compare" in report:
        cmp_d = report["compare"]
        lines.append(
            f"Compare vs {cmp_d['compare_session']}: exact_match={cmp_d['exact_match']}/{cmp_d['common_files']} mismatches={len(cmp_d['mismatches'])}"
        )
    return "\n".join(lines)
