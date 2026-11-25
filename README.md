# Automated Optical CD Ripper & Format-Aware Recovery (`miffy`)

**`miffy`** helps you back up and rescue files from aging or scratched CDs—including photo CDs, MP3/audio data CDs, Video CDs (VCD 2.0), and general data discs. Where ordinary file managers freeze or abort on read errors, `miffy` automatically identifies the disc format, retries damaged areas from multiple angles and speeds, checks every recovered file for structural corruption, and repairs what cannot be read cleanly.

> [!WARNING]
> - **Linux Only**: This tool currently runs only on Linux (optical drive hardware access relies on Linux `SG_IO` ioctls and `/dev/sgN` devices).
> - **Under Active Development**: This project is still under active development and is **not production-ready**. CLI flags, internal APIs, and output schemas may change without notice—always preserve read-only baselines (`--save-baseline`) and verify critical recovered files independently.

## What It Can Do

- **Rescue Scratched CDs Automatically**: Detects your optical drive, scans the disc, retries unreadable areas across multiple read speeds and seek directions, and reassembles damaged sectors into a clean disc image.
- **Handle Photo, Audio & Video CDs**: Supports standard Mode 1 data/photo/MP3 discs (saved as `.iso`) as well as White Book Video CDs (saved as raw `.bin` + `.cue` with extracted playable `.mpg` video streams).
- **Verify Every Recovered File**: Validates extracted JPEGs, BMPs, PNGs, MP3/audio tracks, MPEG videos, and text/binary assets so silent corruption is never mistaken for a clean copy, and supports cross-checking two recovery runs byte-for-byte.
- **Inspect Discs & Audit Backups Without Extra Scripts**: Built-in CLI commands let you inspect an inserted disc, check drive health/status, inspect offline `.iso`/`.bin` images, and audit finished recovery sessions.
- **Protect Baselines & Hardware State**: Preserves optional read-only (`0444`) Pass-0 baseline snapshots (`--save-baseline`) and always restores your optical drive's default error-correction and speed settings before exiting.

---

## Quickstart & Common Workflows

**Requirements**: Python 3.10+ and `Pillow` (`PIL`). Optional `ffmpeg` and `ffprobe` enable deep audio/video decoding checks during `--audit-session --ffmpeg`. Hardware commands require read/write access to `/dev/sgN` (run outside restrictive sandboxes).

### 1. Inspect an Inserted Disc & Check Drive State

```bash
# Inspect inserted CD tracks, Mode 1 / Mode 2 / CD-DA type, sample sectors, and file tree
python3 miffy.py --inspect-disc

# Read-only check confirming drive Mode Page 0x01 (0x00/0x80) and speed settings are at defaults
python3 miffy.py --drive-status
```

### 2. Rip & Recover a CD

```bash
# Full automated recovery with live --debug telemetry and a read-only Pass-0 baseline backup
python3 -u miffy.py --full-rip --save-baseline --debug

# Pass-0 rip first (ideal for clean discs, audio/video CDs, or capturing a baseline before repair passes)
python3 -u miffy.py --name MY_DISC --max-passes 0 --focus-passes 0 --max-tier 1 --save-baseline --debug

# Target specific categories or folders (default: all; aliases include audio,video,media,photo_l,photo_s,pic,photo)
python3 -u miffy.py --name MY_DISC --categories photo_l --max-passes 48 --focus-passes 24 --max-tier 3 --save-baseline --debug
```

Sessions are saved automatically under `recovered/<VOLUME_ID>/` (or `--name`, or `cd_<uuid>`/`vcd_<uuid>` if unnamed). Mode 2 / VCD 2.0 discs automatically output `<volume>_recovered.bin` + `<volume>_recovered.cue` with `.mpg` streams extracted into `extracted/`.

### 3. Inspect Images, Audit Sessions & Run Offline Recovery

```bash
# Inspect an offline .iso (2048 B/sec) or Mode 2 .bin (2352 B/sec) image
python3 miffy.py --inspect-image recovered/MY_DISC/my_disc_recovered.iso

# Audit a recovered session (SQLite telemetry, Q-parity gate, format checks, optional ffmpeg & parity diff)
python3 miffy.py --audit-session recovered/MY_DISC --ffmpeg --compare-session recovered/MY_DISC_PREV

# Re-extract or run offline Tier 1-3 solving on an existing .iso or Mode 2 .bin cache
python3 miffy.py --no-hardware --name MY_DISC --categories all

# List registered solver/format plugins or run strictly in-memory synthetic JPEG & BMP repair tests
python3 miffy.py --list-plugins
python3 miffy.py --test-synthetic-repair
```

Add `--json` to `--drive-status`, `--inspect-disc`, `--inspect-image`, or `--audit-session` for machine-readable JSON output.

---

## Design Philosophy & Technical Architecture

### Design Philosophy
- **Data Integrity First**: Never treat a plausible-looking read as verified without exact mathematical proof (full sector header checks, ECMA-130 RSPC P/Q + 32-bit EDC, physical Q-parity cross-checking, and strict file-format parsing).
- **Physical & Algebraic Recovery Before Reconstruction**: Exhaust multi-speed directional hardware diversity and exact linear error-correction solving before applying lossless format-level gap repair.
- **Non-Destructive Baselines & Clean Hardware Teardown**: Never mutate baseline images (`baseline.iso` / `baseline.bin` + `baseline.cue`); always restore drive `Mode Page 0x01` and read speed in `finally:` blocks.
- **Transparent Telemetry**: Every `--debug` run streams live Stage 1–6 progress and records timestamped logs (`debug.log`, `debug_events.jsonl`) plus per-pass/per-solver SQLite metrics (`raw_passes.sqlite3`).

### Under the Hood

1. **Low-Level Raw+C2 Acquisition (`src/hardware/device.py`)**:
   - Opens `/dev/sgN` with `O_RDWR | O_NONBLOCK`, probes drive capabilities (`INQUIRY`, `MODE SENSE 0x2A`), configures `Mode Page 0x01` (`DCR=1`, `retries=0`), and issues `READ CD (0xBE, Flags=0xFA)` for `2,352`-byte raw frames + `294`-byte MSB-first C2 pointers.
   - Detects `mode1` vs CD-ROM XA `mode2` discs automatically via `READ TOC (0x43)` and LBA 16 probing (`detect_disc_mode`).
   - Filters and retries `-1 LBA` physical drive frame slips (`is_frame_slip`, `is_frame_slip_mode2`), rotates speeds via `SET CD SPEED (0xBB)`, performs `FWD`/`REV` sled cache-flush sweeps, and runs adaptive `--focus-passes` directional bursts when `<= 40` media LBAs remain.
2. **Tiered Algebraic & $GF(2)$ Linear Solver (`src/solvers/`, `src/core/rspc.py`)**:
   - **Tier 1–2**: Single-pass C2 erasure cleaning (`single_c2`), fast $K_{43,26}$ subgrid peeling (`subgrid_peel`), C2-anchor grafting (`anchor_c2_graft`), and C2-cleaned consensus voting (`cleaned_consensus`).
   - **Tier 3 (parallel via `--workers`)**: Priority-ordered per-plane RSPC+$GF(2)$+EDC solving (`per_plane_subgrid_gf2`), $K_{43,26}$ zero-conflict bipartite core + hybrid harvesting (`bipartite_core_hybrid`), Joint $GF(2)$ Gaussian elimination over missing P-column syndromes + 32-bit EDC with weight-3 forged-P trap solving (`joint_gf2_edc`), exhaustive subgrid peeling (`subgrid_peel_exhaustive`), and speed/direction diversity (`subset_diversity_subgrid`).
   - Enforces expected Mode 1 headers (`SYNC_12`, MSF LBA, `ZERO8`), `86/86 P`, `52/52 Q`, 32-bit EDC, and physical Q-parity verification (`verify_physical_q_parity`) against unflagged disc reads.
3. **Mode 2 / VCD 2.0 & ISO-9660/Joliet Triage (`src/pipeline/vcd_mode2.py`, `src/filesystem/iso9660.py`)**:
   - Parses ISO-9660 and Joliet (`utf-16-be`) directory trees in pure Python, assigning `priority=10` to full-size photos, `AUDIO_EXTS`, `VIDEO_EXTS`, and VCD `/MPEGAV`|`/SEGMENT` `.DAT` streams, and classifying physical damage geometry (`RADIAL_SCRATCH`, `SUSPECTED_DRM`, `CLEAN_DISC`, etc.).
   - Verifies Mode 2 Form 1 (`86/86 P`, `52/52 Q`, 32-bit EDC with zeroed MSF header + single-pass C2 cleanup) and Form 2 sectors (32-bit EDC over bytes `16..2347` + zero C2 + mirrored subheaders), writing `.mpg` streams from 2,324-byte Form 2 payloads while preserving non-MPEG Form 2 files in their original names.
4. **Format-Aware Validators & Reconstructors (`src/formats/`)**:
   - **JPEG (`jpeg_handler.py`)**: Strict ITU-T T.81 structural validation (`validate_jpeg_t81_structure`) plus thumbnail-paired and blind DP MCU suffix resynchronization in quantized DCT space (`0%` generation loss on intact MCUs).
   - **BMP & Images (`bmp_handler.py`, `image_generic.py`)**: Strict DIB header/bounds checks (`validate_bmp_structure`), 24-bit scanline repair, and PNG CRC / Pillow image checks.
   - **Audio & Video (`media_handlers.py`)**: MPEG-1/2/2.5 audio frame walker + WAV/FLAC/OGG validation (`AudioFormatHandler`), streaming C-accelerated MPEG-1/2 Program Stream (with PES-aware zero-stuffing detection) + AVI/MP4/MKV validation (`VideoFormatHandler`), and optional `ffmpeg`/`ffprobe` verification (`verify_media_with_ffmpeg`).
   - **Other Formats (`binary_handlers.py`)**: Text/INI, Macromedia Director RIFX/XFIR, PE executable/Xtra, and zero-hole binary validators.
5. **Built-In Diagnostics (`src/pipeline/diagnostics.py`)**:
   - Implements `inspect_optical_disc`, `inspect_image`, `audit_session`, and non-mutating `OpticalDrive.read_drive_status`.

---

## Repository Layout

```
.
├── miffy.py                       # Top-level CLI entry point
├── README.md                      # Project documentation & usage guide
├── AGENTS.md                      # Operational & hardware safety rules
├── LICENSE                        # GNU General Public License v3.0
├── src/
│   ├── core/                      # Dataclasses (models.py), ECMA-130 + Mode 2 math (rspc.py), SQLite/ISO/baseline persistence (state.py), telemetry logger (logging_utils.py)
│   ├── hardware/                  # Linux SG_IO /dev/sgN auto-detection, TOC/mode probing, Mode Page 0x01/0x2A status & restore, raw+C2 I/O (device.py)
│   ├── filesystem/                # Pure-Python ISO-9660/Joliet parser & etiology classifier (iso9660.py)
│   ├── solvers/                   # Pluggable Tier 1–3 RSPC/EDC/GF(2) solvers (base.py, portfolio.py, deep_algebraic.py)
│   ├── formats/                   # File format plugins (jpeg_handler.py, bmp_handler.py, image_generic.py, media_handlers.py, binary_handlers.py)
│   ├── pipeline/                  # Mode 1 orchestrator (orchestrator.py), Mode 2/VCD BIN+CUE ripper (vcd_mode2.py), disc/image/session diagnostics (diagnostics.py)
│   └── legacy/                    # Original standalone scripts archived from initial recovery work
├── test/
│   ├── core/                      # ECMA-130, Q-parity gate, SQLite & debug logger unit tests
│   ├── filesystem/                # ISO-9660/Joliet & etiology tests
│   ├── solvers/                   # Multi-tier solver & GF(2) trap tests
│   ├── formats/                   # JPEG T.81/DCT, BMP, PNG CRC, audio/video & binary validator tests
│   └── pipeline/                  # Session resolution, Stage 3–6 telemetry, Mode 2 VCD & diagnostics CLI tests
└── recovered/                     # Git-ignored per-CD recovery outputs (auto-named by Volume ID, --name, or UUID)
    └── <VOLUME_ID>/               # Session folder
        ├── <volume>_recovered.iso # (or <volume>_recovered.bin + .cue for Mode 2 / VCD)
        ├── baseline.iso           # Optional read-only (0444) Pass-0 snapshot via --save-baseline (or baseline.bin + .cue)
        ├── logs/                  # raw_passes.sqlite3, recovery.log, debug.log, debug_events.jsonl, reports
        └── extracted/             # Extracted/repaired files + manifest.csv
```

---

## Running the Test Suite

```bash
python3 -m unittest discover -s test -t . -v
```

---

## Extending the Framework

### Adding a New Sector Recovery Strategy
Create a module inside `src/solvers/` implementing `SectorSolverPlugin`:

```python
from src.solvers import SolverRegistry
from src.solvers.base import SectorSolverPlugin

@SolverRegistry.register
class MyCustomSolver(SectorSolverPlugin):
    name = "my_custom_solver"
    tier = 2  # Runs after Tier 1 fast solvers, before Tier 3 deep GF(2)
    priority = 50  # Lower value runs earlier within the same tier
    description = "Custom iterative C2/RSPC heuristic."

    def solve(self, lba, raw_passes, context=None):
        ...
```
Import it in `src/solvers/__init__.py` to enable auto-registration.

### Adding a New File Format Handler
Create a module inside `src/formats/` implementing `FileFormatHandler`:

```python
from pathlib import Path
from src.formats import FormatRegistry
from src.formats.base import FileFormatHandler

@FormatRegistry.register
class TiffRepairHandler(FileFormatHandler):
    name = "tiff_repair"
    extensions = (".TIF", ".TIFF")
    priority = 16
    description = "TIFF IFD validator and strip reconstructor."

    def validate(self, data: bytes):
        ...

    def validate_file(self, path: Path):
        # Optional streaming override for large media files
        return self.validate(path.read_bytes())

    def reconstruct(self, data: bytes, bad_intervals, reference_data=None, context=None):
        ...
```
Import it in `src/formats/__init__.py` to enable auto-registration.

---

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
