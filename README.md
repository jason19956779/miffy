# Automated Optical CD Ripper & Format-Aware Recovery (`miffy`)

A modular Python recovery framework for extracting bit-exact ISO images and rescuing damaged photos and files from physically scratched CD-ROMs. Built around low-level Linux SCSI `SG_IO` raw+C2 acquisition, ECMA-130 Reed-Solomon Product Code (RSPC) + 32-bit EDC algebraic solving, and lossless quantized-DCT bitstream repair.

## Key Capabilities

1. **Raw+C2 Hardware Acquisition (`Mode Page 0x01`, `DCR=1`)**:
   - Reads full `2,352`-byte raw sectors plus `294`-byte MSB-first C2 error pointers via SCSI `READ CD (0xBE, Flags=0xFA)`, bypassing firmware read aborts while always restoring drive defaults on exit.
   - Uses rotating read speeds (`1x`–`24x` via `SET CD SPEED 0xBB`) and bidirectional inner/outer sled cache-flush seeks (`FWD`/`REV`) to capture stochastic read diversity across radial scratches.
2. **Tiered Algebraic & $GF(2)$ Linear Solver (`src/solvers/`)**:
   - **Tier 1–2**: Single-pass C2 erasure cleaning, $K_{43,26}$ subgrid peeling with DSP weight-3 mis-correction unmasking, C2-cleaned consensus voting, and C2-anchor grafting.
   - **Tier 3**: Speed/direction random-subset sampling, $K_{43,26}$ zero-conflict bipartite core + core-injected hybrid harvesting, and Joint $GF(2)$ Gaussian elimination over missing P-column syndromes + linear 32-bit ECMA-130 EDC.
3. **Zero-Config Session Management & ISO-9660/Joliet Triage (`src/filesystem/`, `src/pipeline/`)**:
   - Pure-Python ISO-9660 & Joliet (`utf-16-be`) parser maps bad LBAs directly to file extents and classifies defect etiology (`RADIAL_SCRATCH`, `SUSPECTED_DRM`, etc.).
   - Automatically stores each disc's ISO, `logs/` SQLite pass cache, and `extracted/` files under `recovered/<VOLUME_ID>/` (falling back to `recovered/cd_<uuid>/` if unnamed).
4. **Lossless Format-Aware Reconstruction (`src/formats/`)**:
   - **Baseline JPEGs**: Pairs photos with companion thumbnails (`/S/`, `_T` $\leftrightarrow$ full-res, or EXIF `APP1`), aligns shifted post-sector MCU streams via DP-memoized suffix search, solves exact integer differential DC offsets `(C_Y, C_Cb, C_Cr)`, and bridges only missing gap MCUs in quantized DCT space (`0%` generation loss on all intact MCUs; also supports blind self-calibrated fallback without a thumbnail).
   - **BMP & Other Formats**: Scanline repair for uncompressed BMPs, PNG chunk CRC validation, plus validators for `.INI`/`.TXT`, Macromedia Director `.DXR`/`.CXT`, and PE `.DLL`/`.X32`/`.EXE` files.

---

## Repository Layout

```
.
├── auto_cd_recover.py             # Top-level CLI entry point
├── README.md                      # Project documentation & usage guide
├── AGENTS.md                      # Operational & hardware safety rules
├── src/
│   ├── core/                      # Dataclasses (models.py), ECMA-130 math (rspc.py), SQLite/ISO persistence (state.py)
│   ├── hardware/                  # Linux SG_IO /dev/sgN auto-detection, Mode Page 0x01/0x2A, raw+C2 I/O (device.py)
│   ├── filesystem/                # Pure-Python ISO-9660/Joliet parser & etiology classifier (iso9660.py)
│   ├── solvers/                   # Pluggable Tier 1–3 RSPC/EDC solvers (base.py, portfolio.py, deep_algebraic.py)
│   ├── formats/                   # Pluggable file format handlers (jpeg_handler.py, bmp_handler.py, image_generic.py, binary_handlers.py)
│   ├── pipeline/                  # 6-stage orchestrator & per-CD session manager (orchestrator.py)
│   └── legacy/                    # Original standalone scripts from the miffy CD-ROM recovery investigation
├── test/
│   ├── core/                      # ECMA-130 & SQLite unit tests
│   ├── filesystem/                # ISO-9660/Joliet & etiology tests
│   ├── solvers/                   # Multi-tier solver functional tests
│   ├── formats/                   # JPEG DCT, BMP, PNG CRC & binary validator tests
│   └── pipeline/                  # Session folder resolution & CLI integration tests
└── recovered/
    └── MIFFY/                     # Recovered MIFFY CD-ROM artifacts (100% of 688 /Photo/L_Photo photos verified)
        ├── miffy1.iso             # Untouched baseline WinCDEmu image (253,806,592 B)
        ├── miffy_recovered.iso    # Patched full-volume ISO image (254,113,792 B)
        ├── bad sector.txt         # WinCDEmu bad-sector report
        ├── logs/                  # SQLite pass cache (raw_passes.sqlite3), recovery_state.json, run logs
        └── extracted/             # Extracted media, photo_manifest.csv, L_PHOTO/, S/, PIC/, PHOTO.INI
```

---

## Quickstart & Usage

**Requirements**: Python 3.10+ and `Pillow` (`PIL`). Hardware mode requires read/write access to `/dev/sgN` (standard permissions or outside restrictive sandboxes).

### 1. Rip & Automatically Recover a Newly Inserted CD
Insert a disc into your optical drive and run:

```bash
python3 auto_cd_recover.py --full-rip --categories all --max-passes 24
```

This automatically:
1. Detects `/dev/sgN` and reads the disc's ISO-9660 Volume ID (creating e.g. `recovered/<VOLUME_ID>/`).
2. Streams Pass-0 raw+C2 sectors to build `<volume>_recovered.iso` while logging bad sectors into `logs/raw_passes.sqlite3`.
3. Runs Tier 1–3 optical + algebraic solving (prioritizing photos/images).
4. Applies format-aware reconstruction (JPEG DCT resync, BMP scanline repair) on any still-partial files and writes verified outputs to `recovered/<VOLUME_ID>/extracted/`.

### 2. Useful CLI Options

```bash
# Inspect registered solver and format handler plugins
python3 auto_cd_recover.py --list-plugins

# Run strictly in-memory synthetic JPEG & BMP repair verification
python3 auto_cd_recover.py --test-synthetic-repair

# Run offline against the cached MIFFY session (explicit session name or ISO path)
python3 auto_cd_recover.py \
  --no-hardware \
  --name MIFFY \
  --iso-out recovered/MIFFY/miffy_recovered.iso \
  --categories photo_l,photo_s,pic,photo
```

### 3. Run the Test Suite

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
    description = "Custom iterative C2/RSPC heuristic."

    def solve(self, lba, raw_passes, context=None):
        ...
```
Import it in `src/solvers/__init__.py` to enable auto-registration.

### Adding a New File Format Handler
Create a module inside `src/formats/` implementing `FileFormatHandler`:

```python
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

    def reconstruct(self, data: bytes, bad_intervals, reference_data=None, context=None):
        ...
```
Import it in `src/formats/__init__.py` to enable auto-registration.
