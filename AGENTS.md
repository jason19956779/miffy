# AGENTS.md

General repository rules and operational guidelines for inspecting and recovering data from scratched optical CD-ROMs.

## Repository Rules

1. **Direct Optical Device Access (`BypassSandbox` & `O_RDWR`)**:
   - When running any command that reads data from an optical disc (`/dev/sr0` or `/dev/sg0`), launch it from Antigravity with the sandbox disabled (`BypassSandbox: true`).
   - Do not block read-write access (`O_RDWR`): open `/dev/sgN` (e.g., `/dev/sg0`) with `os.O_RDWR | os.O_NONBLOCK` so SCSI `MODE SELECT (10)` and `READ CD` `SG_IO` ioctls succeed without permission errors.
2. **Baseline Image Preservation (`baseline.iso` vs. `<volume>_recovered.iso`)**:
   - Never overwrite or modify an existing read-only baseline ISO image (e.g., `recovered/<VOLUME_ID>/baseline.iso`) in place.
   - Write all recovered sectors to a separate output image (e.g., `recovered/<VOLUME_ID>/<volume>_recovered.iso`), expanded to the full ISO-9660 volume size.
   - Keep unrecovered bad sectors zero-filled (`0x00 * 2048`) at their exact LBA offsets.
3. **Rule Documentation**:
   - All general rules for this repository must be documented in `AGENTS.md`.
4. **Inspection Notes & Debugging (`NOTE.md`)**:
   - Record all inspection logs, diagnostic findings, hardware observations, recovery progress, and debugging notes in `NOTE.md` (local-only, git-ignored; documented exclusively in `AGENTS.md`).
5. **Modular Repository Layout (`miffy.py`, `src/`, `test/`, `recovered/`)**:
   - Keep `miffy.py` at top level as the main CLI entry point, source modules under `src/` grouped by functionality, tests under `test/` grouped by functionality, and per-CD recovery outputs (ISO, `logs/`, `extracted/`) under `recovered/<NAME>/`.
   - Historical standalone scripts are archived under `src/legacy/`.
6. **Target Optical Device (`/dev/srN` / `/dev/sgN`)**:
   - The default target optical drive is `/dev/sr0` (paired SCSI generic device `/dev/sg0`), or the auto-detected `/dev/sgN` device.
   - Whenever configuring Mode Page `0x01` (`DCR=1`, `retries=0`) via `MODE SELECT (10)` for raw sector + C2 reading, always restore the drive's default Mode Page `0x01` (`err_bits=0x00, retries=0x80`) in a `finally:` block before closing the file descriptor.
7. **Elevated Privileges (`sudo`)**:
   - If any tool or operation requires `sudo`, provide the complete command to the user to execute instead of failing or attempting workarounds.
8. **Commit Message Style**:
   - **Subject line**: Write a single imperative sentence starting with a capitalized verb, with no Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) and no trailing period. Summarize the core outcomes concisely (joining 2–3 cohesive themes with commas and `and` when appropriate).
   - **Body**: Separate from the subject with one blank line and use a dash-prefixed (`- `) bullet list. Each bullet must start with a capitalized imperative verb (`Add`, `Build`, `Configure`, `Convert`, `Enforce`, `Harden`, `Implement`, `Remove`, `Set`, `Split`, `Update`, etc.), cite concrete modules, classes, functions, file paths, commands, and invariants (explaining why via `so ...` where helpful), and conclude with a bullet summarizing added or updated tests and verification coverage.
   - Never mention local-only `NOTE.md` or untracked recovery dumps in commit messages.
