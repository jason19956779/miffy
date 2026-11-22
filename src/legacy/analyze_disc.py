#!/usr/bin/env python3
"""Triage bad sectors in bad sector.txt and map them to ISO-9660 file extents."""

import csv
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
MIFFY_DIR = REPO_DIR / "recovered" / "MIFFY"
ISO_PATH = MIFFY_DIR / "miffy1.iso"
BAD_LOG_PATH = MIFFY_DIR / "bad sector.txt"
LOGS_DIR = MIFFY_DIR / "logs"

RANGE_RE = re.compile(
    r"0x([0-9A-Fa-f]+)\s*-\s*0x([0-9A-Fa-f]+)\s*\((\d+)\s*sectors\)"
)
ISOINFO_ENTRY_RE = re.compile(
    r"^([d\-][\-rwx]{9})\s+\d+\s+\d+\s+\d+\s+(\d+)\s+"
    r"[A-Za-z]+\s+\d+\s+\d+\s+\[\s*(\d+)\s+[0-9A-Fa-f]+\]\s+(.+)$"
)


def parse_bad_sectors(bad_log_path: Path):
    """Parse WinCDEmu bad sector ranges and verify start/end/count consistency."""
    ranges = []
    lbas = []
    with bad_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RANGE_RE.search(line.strip())
            if not m:
                continue
            start = int(m.group(1), 16)
            end = int(m.group(2), 16)
            count = int(m.group(3))
            if end - start != count:
                raise ValueError(
                    f"Unit mismatch in bad sector.txt: {start:#x} - {end:#x} != {count}"
                )
            ranges.append({"start": start, "end": end, "count": count})
            lbas.extend(range(start, end))
    lbas.sort()
    return ranges, lbas


def verify_iso_zero_fill(iso_path: Path, bad_lbas: list[int]) -> dict:
    """Confirm all bad sectors in miffy1.iso are zero-filled and check ISO size."""
    iso_size = iso_path.stat().st_size
    zero_count = 0
    nonzero_lbas = []
    out_of_range = []
    zeros = b"\x00" * 2048
    with iso_path.open("rb") as f:
        for lba in bad_lbas:
            off = lba * 2048
            if off + 2048 > iso_size:
                out_of_range.append(lba)
                continue
            f.seek(off)
            if f.read(2048) == zeros:
                zero_count += 1
            else:
                nonzero_lbas.append(lba)
    return {
        "iso_size_bytes": iso_size,
        "iso_sectors": iso_size // 2048,
        "iso_remainder_bytes": iso_size % 2048,
        "bad_lbas_zero_filled": zero_count,
        "bad_lbas_nonzero_count": len(nonzero_lbas),
        "bad_lbas_out_of_range": len(out_of_range),
    }


def analyze_geometry(ranges: list[dict], bad_lbas: list[int]) -> dict:
    """Analyze run lengths, gap sizes, LBA distribution, and revolution pitch."""
    run_lengths = Counter(r["count"] for r in ranges)
    gaps = [
        ranges[i + 1]["start"] - ranges[i]["end"] for i in range(len(ranges) - 1)
    ]
    gap_dist = Counter(gaps)

    buckets = {}
    for b in range(0, 130000, 10000):
        cnt = sum(1 for x in bad_lbas if b <= x < b + 10000)
        pitches = [
            ranges[i + 1]["start"] - ranges[i]["start"]
            for i in range(len(ranges) - 1)
            if b <= ranges[i]["start"] < b + 10000
            and 10 <= (ranges[i + 1]["start"] - ranges[i]["start"]) <= 20
        ]
        avg_pitch = round(sum(pitches) / len(pitches), 3) if pitches else None
        buckets[f"{b}..{b + 9999}"] = {
            "bad_sectors": cnt,
            "single_turn_pitch_samples": len(pitches),
            "avg_single_turn_pitch_sectors": avg_pitch,
        }

    return {
        "total_ranges": len(ranges),
        "total_bad_sectors": len(bad_lbas),
        "min_bad_lba": bad_lbas[0] if bad_lbas else None,
        "max_bad_lba": bad_lbas[-1] if bad_lbas else None,
        "run_length_distribution": {str(k): v for k, v in sorted(run_lengths.items())},
        "top_15_gap_sizes": [
            {"gap_sectors": k, "count": v} for k, v in gap_dist.most_common(15)
        ],
        "lba_buckets_10k": buckets,
    }


def parse_isoinfo(iso_path: Path) -> tuple[dict, list[dict]]:
    """Parse PVD metadata and all file/directory extents via isoinfo."""
    pvd_out = subprocess.check_output(
        ["isoinfo", "-d", "-i", str(iso_path)], text=True, errors="replace"
    )
    pvd_info = {}
    for line in pvd_out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            pvd_info[k.strip()] = v.strip()

    listing = subprocess.check_output(
        ["isoinfo", "-l", "-i", str(iso_path)], text=True, errors="replace"
    )
    cur_dir = "/"
    entries = []
    for line in listing.splitlines():
        if line.startswith("Directory listing of "):
            cur_dir = line[len("Directory listing of ") :].strip()
            continue
        m = ISOINFO_ENTRY_RE.match(line)
        if not m:
            continue
        ftype, size_s, lba_s, name = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        if name in (".", ".."):
            continue
        if name.endswith(";1"):
            name = name[:-2]
        size = int(size_s)
        start_lba = int(lba_s)
        n_sectors = (size + 2047) // 2048 if size > 0 else 0
        entries.append(
            {
                "path": cur_dir + name,
                "directory": cur_dir.rstrip("/") or "/",
                "type": "dir" if ftype.startswith("d") else "file",
                "ext": Path(name).suffix.upper() if not ftype.startswith("d") else "",
                "start_lba": start_lba,
                "end_lba": start_lba + n_sectors,
                "size_bytes": size,
                "sectors": n_sectors,
            }
        )
    entries.sort(key=lambda x: (x["start_lba"], x["path"]))
    return pvd_info, entries


def map_bad_sectors_to_files(entries: list[dict], bad_lbas: list[int]) -> dict:
    """Identify all files/dirs overlapping bad LBAs and summarize by dir/ext."""
    bad_set = set(bad_lbas)
    covered_bad = set()
    affected = []
    by_dir = {}
    by_ext = {}

    for entry in entries:
        overlap = sorted(
            lba for lba in range(entry["start_lba"], entry["end_lba"]) if lba in bad_set
        )
        if not overlap:
            continue
        covered_bad.update(overlap)
        d = entry["directory"]
        ext = entry["ext"] or "(NOEXT)"
        by_dir.setdefault(d, {"affected_files": 0, "bad_sectors": 0, "total_bytes": 0})
        by_dir[d]["affected_files"] += 1
        by_dir[d]["bad_sectors"] += len(overlap)
        by_dir[d]["total_bytes"] += entry["size_bytes"]

        by_ext.setdefault(ext, {"affected_files": 0, "bad_sectors": 0})
        by_ext[ext]["affected_files"] += 1
        by_ext[ext]["bad_sectors"] += len(overlap)

        affected.append(
            {
                **entry,
                "bad_sector_count": len(overlap),
                "bad_pct": round(100.0 * len(overlap) / max(1, entry["sectors"]), 2),
                "bad_lbas": overlap,
            }
        )

    unallocated = sorted(bad_set - covered_bad)
    return {
        "total_entries": len(entries),
        "max_extent_end_lba": max((e["end_lba"] for e in entries), default=0),
        "affected_file_count": len(affected),
        "allocated_bad_sectors": len(covered_bad),
        "unallocated_bad_sectors": len(unallocated),
        "by_directory": by_dir,
        "by_extension": by_ext,
        "affected_files": affected,
    }


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ranges, bad_lbas = parse_bad_sectors(BAD_LOG_PATH)
    iso_check = verify_iso_zero_fill(ISO_PATH, bad_lbas)
    geometry = analyze_geometry(ranges, bad_lbas)
    pvd_info, entries = parse_isoinfo(ISO_PATH)
    fs_map = map_bad_sectors_to_files(entries, bad_lbas)

    report = {
        "pvd": pvd_info,
        "iso_check": iso_check,
        "geometry": geometry,
        "filesystem_summary": {
            k: v for k, v in fs_map.items() if k != "affected_files"
        },
        "affected_files_count": fs_map["affected_file_count"],
    }

    report_path = LOGS_DIR / "triage_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = LOGS_DIR / "affected_files.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "directory",
                "ext",
                "start_lba",
                "end_lba",
                "sectors",
                "size_bytes",
                "bad_sector_count",
                "bad_pct",
                "bad_lbas_hex",
            ]
        )
        for item in fs_map["affected_files"]:
            writer.writerow(
                [
                    item["path"],
                    item["directory"],
                    item["ext"],
                    item["start_lba"],
                    item["end_lba"],
                    item["sectors"],
                    item["size_bytes"],
                    item["bad_sector_count"],
                    item["bad_pct"],
                    " ".join(f"0x{x:05X}" for x in item["bad_lbas"]),
                ]
            )

    print(f"Wrote {report_path} and {csv_path}")
    print(f"Bad sectors: {geometry['total_bad_sectors']} across {geometry['total_ranges']} ranges")
    print(
        f"ISO check: {iso_check['iso_sectors']} sectors ({iso_check['iso_size_bytes']} B), "
        f"PVD Volume size={pvd_info.get('Volume size is')}, zero-filled bad={iso_check['bad_lbas_zero_filled']}"
    )
    print(
        f"Filesystem: {fs_map['total_entries']} entries, {fs_map['affected_file_count']} affected files, "
        f"unallocated bad={fs_map['unallocated_bad_sectors']}"
    )
    print("By directory:", json.dumps(fs_map["by_directory"], indent=2))
    print("By extension:", json.dumps(fs_map["by_extension"], indent=2))


if __name__ == "__main__":
    main()
