#!/usr/bin/env python3
"""Extract and validate recovered /PHOTO/L_PHOTO, /PHOTO/S, PHOTO.INI, and /PIC files."""

import csv
import io
import json
import re
import sqlite3
import subprocess
from pathlib import Path

from PIL import Image, ImageFile

REPO_DIR = Path(__file__).resolve().parents[2]
MIFFY_DIR = REPO_DIR / "recovered" / "MIFFY"
REC_ISO = MIFFY_DIR / "miffy_recovered.iso"
OUT_DIR = MIFFY_DIR / "extracted"
DB_PATH = MIFFY_DIR / "logs" / "raw_passes.sqlite3"
LOGS_DIR = MIFFY_DIR / "logs"
BAD_LOG_PATH = MIFFY_DIR / "bad sector.txt"

RANGE_RE = re.compile(
    r"0x([0-9A-Fa-f]+)\s*-\s*0x([0-9A-Fa-f]+)\s*\((\d+)\s*sectors\)"
)
ISOINFO_ENTRY_RE = re.compile(
    r"^([d\-][\-rwx]{9})\s+\d+\s+\d+\s+\d+\s+(\d+)\s+"
    r"[A-Za-z]+\s+\d+\s+\d+\s+\[\s*(\d+)\s+[0-9A-Fa-f]+\]\s+(.+)$"
)


def load_bad_lbas() -> set[int]:
    bad = set()
    with BAD_LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RANGE_RE.search(line.strip())
            if m:
                bad.update(range(int(m.group(1), 16), int(m.group(2), 16)))
    return bad


def list_iso_targets(iso_path: Path) -> list[dict]:
    listing = subprocess.check_output(
        ["isoinfo", "-l", "-i", str(iso_path)], text=True, errors="replace"
    )
    cur_dir = "/"
    targets = []
    for line in listing.splitlines():
        if line.startswith("Directory listing of "):
            cur_dir = line[len("Directory listing of ") :].strip()
            continue
        m = ISOINFO_ENTRY_RE.match(line)
        if not m:
            continue
        ftype, size_s, lba_s, name = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        if name in (".", "..") or ftype.startswith("d"):
            continue
        if name.endswith(";1"):
            name = name[:-2]
        if cur_dir in ("/PHOTO/L_PHOTO/", "/PHOTO/S/", "/PIC/") or (
            cur_dir == "/PHOTO/" and name == "PHOTO.INI"
        ):
            size = int(size_s)
            start_lba = int(lba_s)
            n_sec = (size + 2047) // 2048 if size > 0 else 0
            targets.append(
                {
                    "iso_path": cur_dir + name,
                    "directory": cur_dir.rstrip("/"),
                    "name": name,
                    "start_lba": start_lba,
                    "end_lba": start_lba + n_sec,
                    "size_bytes": size,
                    "sectors": n_sec,
                }
            )
    targets.sort(key=lambda x: (x["directory"], x["name"]))
    return targets


def check_image_decode(data: bytes) -> tuple[str, str]:
    """Return (decode_status, dimensions_mode)."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return "STRICT_OK", f"{im.size[0]}x{im.size[1]} {im.mode}"
    except Exception as exc_strict:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            return f"TRUNCATED_OK ({type(exc_strict).__name__})", f"{im.size[0]}x{im.size[1]} {im.mode}"
        except Exception as exc_trunc:
            return f"DECODE_ERR ({type(exc_trunc).__name__}: {exc_trunc})", ""
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "L_PHOTO").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "S").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "PIC").mkdir(parents=True, exist_ok=True)

    bad_set = load_bad_lbas()
    conn = sqlite3.connect(DB_PATH)
    solved_map = {
        r[0]: (r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT lba, verified, best_p_ok, best_q_ok FROM solved_sectors"
        ).fetchall()
    }
    conn.close()

    targets = list_iso_targets(REC_ISO)
    manifest_rows = []
    summary = {}

    with REC_ISO.open("rb") as iso_f:
        for item in targets:
            iso_f.seek(item["start_lba"] * 2048)
            data = iso_f.read(item["size_bytes"])

            if item["directory"] == "/PHOTO/L_PHOTO":
                dest = OUT_DIR / "L_PHOTO" / item["name"]
            elif item["directory"] == "/PHOTO/S":
                dest = OUT_DIR / "S" / item["name"]
            elif item["directory"] == "/PIC":
                dest = OUT_DIR / "PIC" / item["name"]
            else:
                dest = OUT_DIR / item["name"]

            dest.write_bytes(data)

            overlap_bad = [
                lba for lba in range(item["start_lba"], item["end_lba"]) if lba in bad_set
            ]
            verified_bad = [lba for lba in overlap_bad if solved_map.get(lba, (0,))[0] == 1]
            min_p_ok = (
                min((solved_map.get(lba, (0, 0, 0))[1] for lba in overlap_bad), default=86)
            )
            zero_holes = sum(
                1
                for off in range(0, max(0, len(data) - 2047), 2048)
                if data[off : off + 2048] == b"\x00" * 2048
            )

            if not overlap_bad:
                rec_class = "INTACT_ORIGINAL"
            elif len(verified_bad) == len(overlap_bad):
                rec_class = "RECOVERED_100PCT_VERIFIED"
            else:
                rec_class = "PARTIAL_RSPC_RECONSTRUCTED"

            ext = dest.suffix.upper()
            if ext in (".JPG", ".BMP"):
                pil_status, img_dims = check_image_decode(data)
            else:
                pil_status, img_dims = "TEXT_INI_OK", ""

            d_key = item["directory"]
            st = summary.setdefault(
                d_key,
                {
                    "total_files": 0,
                    "intact_original": 0,
                    "recovered_100pct_verified": 0,
                    "partial_rspc_reconstructed": 0,
                    "strict_pil_ok": 0,
                    "truncated_pil_ok": 0,
                    "pil_error": 0,
                    "bad_sectors_total": 0,
                    "bad_sectors_verified": 0,
                },
            )
            st["total_files"] += 1
            st["bad_sectors_total"] += len(overlap_bad)
            st["bad_sectors_verified"] += len(verified_bad)
            if rec_class == "INTACT_ORIGINAL":
                st["intact_original"] += 1
            elif rec_class == "RECOVERED_100PCT_VERIFIED":
                st["recovered_100pct_verified"] += 1
            else:
                st["partial_rspc_reconstructed"] += 1

            if pil_status == "STRICT_OK":
                st["strict_pil_ok"] += 1
            elif pil_status.startswith("TRUNCATED_OK"):
                st["truncated_pil_ok"] += 1
            elif pil_status.startswith("DECODE_ERR"):
                st["pil_error"] += 1

            manifest_rows.append(
                {
                    "iso_path": item["iso_path"],
                    "saved_path": str(dest),
                    "directory": item["directory"],
                    "size_bytes": item["size_bytes"],
                    "bad_sectors_total": len(overlap_bad),
                    "bad_sectors_verified": len(verified_bad),
                    "min_p_ok": min_p_ok,
                    "zero_holes_2k": zero_holes,
                    "recovery_class": rec_class,
                    "pil_status": pil_status,
                    "image_dims": img_dims,
                }
            )

    manifest_path = OUT_DIR / "photo_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    report_path = LOGS_DIR / "photo_recovery_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Extracted {len(manifest_rows)} files to {OUT_DIR}")
    print("Manifest:", manifest_path)
    print("Summary by directory:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
