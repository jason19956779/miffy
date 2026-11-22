# AGENTS.md

General repository rules and operational guidelines for inspecting and recovering data from the `miffy` CD-ROM.

## Repository Rules

1. **Direct Optical Device Access (`BypassSandbox` & `O_RDWR`)**:
   - When running any command that reads data from the CD (`/dev/sr0` or `/dev/sg0`), launch it from Antigravity with the sandbox disabled (`BypassSandbox: true`).
   - Do not block read-write access (`O_RDWR`): open `/dev/sg0` with `os.O_RDWR | os.O_NONBLOCK` so SCSI `MODE SELECT (10)` and `READ CD` `SG_IO` ioctls succeed without permission errors.
2. **Baseline Image Preservation (`miffy1.iso` vs. `miffy_recovered.iso`)**:
   - `recovered/MIFFY/miffy1.iso` (also referred to as `miffy.iso`) was extracted from the original CD using WinCDEmu on Windows 11 (`253,806,592` bytes = `123,929` sectors).
   - Never overwrite or modify `miffy1.iso` in place. Write all recovered sectors to `recovered/MIFFY/miffy_recovered.iso` (expanded to the full ISO-9660 volume size of `124,079` sectors = `254,113,792` bytes), or to a per-disc folder under `recovered/<VOLUME_ID>/`.
3. **WinCDEmu Reference Log (`bad sector.txt`)**:
   - `recovered/MIFFY/bad sector.txt` is the bad-sector log recorded by WinCDEmu (`2,519` ranges totaling `2,612` bad sectors, formatted as `0xSTART - 0xEND (N sectors)` in LBA units).
   - Every bad sector in `miffy1.iso` is zero-filled (`0x00 * 2048`) at its exact LBA offset.
4. **Rule Documentation**:
   - All general rules for this repository must be documented in `AGENTS.md`.
5. **Inspection Notes & Debugging (`NOTE.md`)**:
   - Record all inspection logs, diagnostic findings, hardware observations, recovery progress, and debugging notes in `NOTE.md` (local-only, git-ignored; documented exclusively in `AGENTS.md`).
6. **Modular Repository Layout (`auto_cd_recover.py`, `src/`, `test/`, `recovered/`)**:
   - Keep `auto_cd_recover.py` at top level as the main CLI entry point, source modules under `src/` grouped by functionality, tests under `test/` grouped by functionality, and per-CD recovery outputs (ISO, `logs/`, `extracted/`) under `recovered/<NAME>/`.
   - Historical standalone scripts are archived under `src/legacy/`.
7. **Target Optical Device (`/dev/sr0` / `/dev/sg0`)**:
   - The original CD is in `/dev/sr0` (paired SCSI generic device `/dev/sg0`).
   - Whenever configuring Mode Page `0x01` (`DCR=1`, `retries=0`) via `MODE SELECT (10)` for raw sector + C2 reading, always restore the drive's default Mode Page `0x01` (`err_bits=0x00, retries=0x80`) in a `finally:` block before closing the file descriptor.
8. **Elevated Privileges (`sudo`)**:
   - If any tool or operation requires `sudo`, provide the complete command to the user to execute instead of failing or attempting workarounds.
9. **Commit Message Style**:
   - **Subject line**: Write a single imperative sentence starting with a capitalized verb, with no Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) and no trailing period. Summarize the core outcomes concisely (joining 2–3 cohesive themes with commas and `and` when appropriate).
   - **Body**: Separate from the subject with one blank line and use a dash-prefixed (`- `) bullet list. Each bullet must start with a capitalized imperative verb (`Add`, `Build`, `Configure`, `Convert`, `Enforce`, `Harden`, `Implement`, `Remove`, `Set`, `Split`, `Update`, etc.), cite concrete modules, classes, functions, file paths, commands, and invariants (explaining why via `so ...` where helpful), and conclude with a bullet summarizing added or updated tests and verification coverage.
   - Never mention local-only `NOTE.md` or untracked recovery dumps in commit messages.
