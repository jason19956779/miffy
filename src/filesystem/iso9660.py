"""Pure-Python ISO-9660 filesystem parser and bad-sector geometry etiology analyzer."""

from collections import Counter
from pathlib import Path
from typing import Callable

from src.core.models import EtiologyReport, FileExtent

PHOTO_MAIN_EXTS = {".JPG", ".JPEG", ".BMP", ".PNG", ".TIF", ".TIFF", ".GIF", ".WEBP"}
AUDIO_EXTS = {".MP3", ".MP2", ".WAV", ".FLAC", ".OGG", ".AAC", ".M4A", ".WMA", ".OPUS", ".AIFF"}
VIDEO_EXTS = {".MPG", ".MPEG", ".MP4", ".AVI", ".MOV", ".MKV", ".WMV", ".VOB", ".M2V", ".M1V", ".ASF"}


def parse_pvd(read_sector: Callable[[int], bytes]) -> dict:
    """Read LBA 16 Primary Volume Descriptor (and optional Joliet SVD)."""
    pvd = read_sector(16)
    if pvd[0] != 1 or pvd[1:6] != b"CD001":
        raise ValueError("Invalid ISO-9660 Primary Volume Descriptor at LBA 16")
    volume_size = int.from_bytes(pvd[80:84], "little")
    block_size = int.from_bytes(pvd[128:130], "little") or 2048
    vol_id = pvd[40:72].decode("ascii", errors="replace").strip()
    app_id = pvd[574:702].decode("ascii", errors="replace").strip()
    root_rec = pvd[156:190]
    root_lba = int.from_bytes(root_rec[2:6], "little")
    root_size = int.from_bytes(root_rec[10:14], "little")

    # Check if Joliet Supplementary Volume Descriptor (type 2, UCS-2 escape) exists
    encoding = "latin-1"
    tree_root_lba = root_lba
    tree_root_size = root_size
    for lba in range(17, 32):
        vd = read_sector(lba)
        if vd[1:6] != b"CD001" or vd[0] == 255:
            break
        if vd[0] == 2 and vd[88:91] in (b"%/@", b"%/C", b"%/E"):
            j_root = vd[156:190]
            j_lba = int.from_bytes(j_root[2:6], "little")
            j_sz = int.from_bytes(j_root[10:14], "little")
            if j_lba > 0 and j_sz > 0:
                encoding = "utf-16-be"
                tree_root_lba = j_lba
                tree_root_size = j_sz
                break

    return {
        "volume_id": vol_id,
        "application_id": app_id,
        "block_size": block_size,
        "volume_sectors": volume_size,
        "root_lba": tree_root_lba,
        "root_size": tree_root_size,
        "encoding": encoding,
    }


def parse_iso9660_tree(
    read_sector: Callable[[int], bytes],
    bad_lbas: set[int] | None = None,
) -> tuple[dict, list[FileExtent]]:
    """Traverse ISO-9660 PVD directory tree without external binaries."""
    pvd = parse_pvd(read_sector)
    bad_set = bad_lbas or set()
    extents: list[FileExtent] = []
    visited_dirs: set[tuple[int, str]] = set()

    def read_range(start_lba: int, size_bytes: int) -> bytes:
        n_sec = (size_bytes + 2047) // 2048
        chunks = [read_sector(start_lba + i) for i in range(n_sec)]
        return b"".join(chunks)[:size_bytes]

    def walk_dir(dir_lba: int, dir_size: int, cur_dir: str) -> None:
        key = (dir_lba, cur_dir)
        if key in visited_dirs:
            return
        visited_dirs.add(key)
        raw = read_range(dir_lba, dir_size)
        pos = 0
        total = len(raw)
        subdirs: list[tuple[int, int, str]] = []

        while pos < total:
            rec_len = raw[pos]
            if rec_len == 0:
                # ECMA-119: records do not cross 2048-byte sector boundaries
                next_sec = ((pos // 2048) + 1) * 2048
                if next_sec >= total:
                    break
                pos = next_sec
                continue
            if pos + rec_len > total or rec_len < 34:
                break

            ext_lba = int.from_bytes(raw[pos + 2 : pos + 6], "little")
            data_len = int.from_bytes(raw[pos + 10 : pos + 14], "little")
            flags = raw[pos + 25]
            name_len = raw[pos + 32]
            raw_name = raw[pos + 33 : pos + 33 + name_len]
            pos += rec_len

            if raw_name in (b"\x00", b"\x01"):
                continue
            enc = pvd.get("encoding", "latin-1")
            name = raw_name.decode(enc, errors="replace").rstrip("\x00").split(";")[0]
            if not name:
                continue

            is_dir = bool(flags & 0x02)
            full_path = ("/" if cur_dir == "/" else cur_dir + "/") + name
            if is_dir:
                extents.append(
                    FileExtent(
                        path=full_path,
                        directory=cur_dir,
                        name=name,
                        ext="",
                        start_lba=ext_lba,
                        end_lba=ext_lba + ((data_len + 2047) // 2048),
                        size_bytes=data_len,
                        sectors=(data_len + 2047) // 2048,
                        is_dir=True,
                        priority=5,
                    )
                )
                subdirs.append((ext_lba, data_len, full_path))
            else:
                n_sec = (data_len + 2047) // 2048 if data_len > 0 else 0
                ext = Path(name).suffix.upper()
                overlap = [lba for lba in range(ext_lba, ext_lba + n_sec) if lba in bad_set]
                d_up = cur_dir.upper()
                if ext in PHOTO_MAIN_EXTS and not d_up.endswith("/S") and "/S/" not in d_up:
                    prio = 10
                elif ext in AUDIO_EXTS or ext in VIDEO_EXTS or (ext == ".DAT" and d_up in ("/MPEGAV", "/SEGMENT")):
                    prio = 10
                elif ext in PHOTO_MAIN_EXTS:
                    prio = 20
                elif ext in {".INI", ".TXT", ".CFG", ".XML", ".JSON"}:
                    prio = 30
                else:
                    prio = 40
                extents.append(
                    FileExtent(
                        path=full_path,
                        directory=cur_dir,
                        name=name,
                        ext=ext,
                        start_lba=ext_lba,
                        end_lba=ext_lba + n_sec,
                        size_bytes=data_len,
                        sectors=n_sec,
                        is_dir=False,
                        bad_lbas=overlap,
                        priority=prio,
                    )
                )

        for sub_lba, sub_sz, sub_path in subdirs:
            walk_dir(sub_lba, sub_sz, sub_path)

    walk_dir(pvd["root_lba"], pvd["root_size"], "/")
    extents.sort(key=lambda e: (e.priority, e.start_lba, e.path))
    return pvd, extents


def make_iso_sector_reader(iso_path: Path) -> Callable[[int], bytes]:
    """Return a cached sector reader over a file on disk."""
    fh = iso_path.open("rb")

    def _read(lba: int) -> bytes:
        fh.seek(lba * 2048)
        data = fh.read(2048)
        if len(data) < 2048:
            data = data + b"\x00" * (2048 - len(data))
        return data

    _read.fh = fh  # type: ignore[attr-defined]
    return _read


def classify_damage_etiology(bad_lbas: list[int]) -> EtiologyReport:
    """Analyze spatial distribution & single-turn LBA pitch to classify CD defect cause."""
    if not bad_lbas:
        return EtiologyReport(
            total_bad_sectors=0,
            total_ranges=0,
            run_length_distribution={},
            single_turn_pitch_samples=0,
            avg_single_turn_pitch=None,
            early_track_ratio=0.0,
            classification="CLEAN_DISC",
            notes="Zero bad sectors detected.",
        )

    s_lbas = sorted(set(bad_lbas))
    ranges: list[tuple[int, int]] = []
    cur_s = s_lbas[0]
    cur_e = s_lbas[0] + 1
    for lba in s_lbas[1:]:
        if lba == cur_e:
            cur_e += 1
        else:
            ranges.append((cur_s, cur_e))
            cur_s, cur_e = lba, lba + 1
    ranges.append((cur_s, cur_e))

    run_lengths = Counter(e - s for s, e in ranges)
    pitches = [
        ranges[i + 1][0] - ranges[i][0]
        for i in range(len(ranges) - 1)
        if 9 <= (ranges[i + 1][0] - ranges[i][0]) <= 24
    ]
    avg_pitch = round(sum(pitches) / len(pitches), 3) if pitches else None
    early_cnt = sum(1 for x in s_lbas if 500 <= x <= 12000)
    early_ratio = round(early_cnt / len(s_lbas), 3)
    max_run = max(run_lengths.keys())

    if len(s_lbas) > 300 and early_ratio > 0.7 and len(pitches) < 10:
        cls = "SUSPECTED_DRM"
        notes = "Dense bad-sector ring in early track LBAs (500..12000); check for SafeDisc/SecuROM."
    elif len(pitches) >= max(5, len(ranges) // 8) and max_run <= 8:
        cls = "RADIAL_SCRATCH"
        notes = (
            f"Single-revolution pitch {avg_pitch} sectors/rev across {len(pitches)} crossings; "
            "strongly recoverable via multi-pass optical Diversity + RSPC."
        )
    elif max_run >= 20:
        cls = "CIRCUMFERENTIAL_OR_BLOCK_DAMAGE"
        notes = f"Long contiguous bad runs (max={max_run} sectors); likely arc scratch or delamination."
    else:
        cls = "LOCALIZED_SURFACE_DEFECTS"
        notes = "Isolated bad sectors across disc surface."

    return EtiologyReport(
        total_bad_sectors=len(s_lbas),
        total_ranges=len(ranges),
        run_length_distribution=dict(sorted(run_lengths.items())),
        single_turn_pitch_samples=len(pitches),
        avg_single_turn_pitch=avg_pitch,
        early_track_ratio=early_ratio,
        classification=cls,
        notes=notes,
    )
