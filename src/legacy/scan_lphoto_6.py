#!/usr/bin/env python3
"""Intensive multi-speed, bidirectional-seek hardware re-scan for the 6 remaining L_PHOTO bad LBAs."""

import os
import time
import recover_cd as r

TARGET_LBAS = [64208, 64921, 66870, 67243, 68685, 70646]
SPEEDS_KBPS = [176, 353, 706, 1058, 1411, 2117, 2822, 4234]
LEADS = [1, 3, 5, 8, 11, 14, 18, 22, 26, 30, 34, 38]
PASSES_PER_DIRECTION = 16


def configure_mode_01(fd: int) -> int:
    """Enable DCR (try 0x21 TB+DCR first, fallback to 0x01 DCR only) with retries=0."""
    for bits in (0x21, 0x01):
        try:
            r.set_mode_01(fd, bits, 0)
            print(f"Configured Mode Page 0x01 with err_bits=0x{bits:02X}, retries=0x00", flush=True)
            return bits
        except RuntimeError as exc:
            print(f"Mode Page 0x01 err_bits=0x{bits:02X} not accepted ({exc}); trying fallback.", flush=True)
    raise RuntimeError("Failed to configure Mode Page 0x01")


def main() -> None:
    r.ensure_recovered_iso()
    conn = r.init_db()
    all_bad_lbas = r.load_bad_lbas()
    fd = os.open(r.SG_DEV, os.O_RDWR | os.O_NONBLOCK)
    t_start = time.time()
    new_reads = 0
    newly_verified = []

    try:
        configure_mode_01(fd)
        pass_counts = {
            lba: (
                conn.execute(
                    "SELECT COALESCE(MAX(pass_id), -1) FROM raw_passes WHERE lba = ?",
                    (lba,),
                ).fetchone()[0]
            )
            for lba in TARGET_LBAS
        }
        print("Initial max pass_id per target LBA:", pass_counts, flush=True)

        # Interleave forward and reverse passes across the 8 speeds and 12 lead-ins
        schedule = []
        for idx in range(PASSES_PER_DIRECTION):
            kbps = SPEEDS_KBPS[idx % len(SPEEDS_KBPS)]
            lead_f = LEADS[(idx * 5) % len(LEADS)]
            schedule.append(("FWD", idx, kbps, lead_f))
            kbps_r = SPEEDS_KBPS[(idx + 3) % len(SPEEDS_KBPS)]
            lead_r = LEADS[(idx * 7 + 3) % len(LEADS)]
            schedule.append(("REV", idx, kbps_r, lead_r))

        with r.REC_ISO.open("r+b") as iso_fh:
            for round_num, (direction, dir_idx, kbps, lead) in enumerate(schedule, 1):
                st_spd = r.set_cd_speed(fd, kbps)
                t_r = time.time()
                round_p_scores = []
                for lba in TARGET_LBAS:
                    # Cache eviction + directional sled approach before every LBA read
                    if direction == "FWD":
                        seek_lba = max(0, lba - 800 - ((dir_idx * 17) % 100))
                    else:
                        seek_lba = min(r.TOTAL_VOLUME_SECTORS - 1, lba + 800 + ((dir_idx * 17) % 100))
                    r.read_cd_raw(fd, seek_lba, 1, with_c2=False)

                    s_lba = max(0, lba - lead)
                    cnt = min(r.MAX_BATCH_SECTORS, max(lead + 4, lead + 2), r.TOTAL_VOLUME_SECTORS - s_lba)
                    if lba - s_lba >= cnt:
                        cnt = min(r.MAX_BATCH_SECTORS, r.TOTAL_VOLUME_SECTORS - s_lba)
                    st, sense, resid, raw = r.read_cd_raw(fd, s_lba, cnt, with_c2=True)
                    if st != 0 or resid != 0:
                        print(
                            f"  WARN round {round_num} {direction} LBA {lba}: st={st} sense={sense} resid={resid}",
                            flush=True,
                        )
                        continue
                    off = (lba - s_lba) * 2646
                    raw_2352 = raw[off : off + 2352]
                    c2_294 = raw[off + 2352 : off + 2646]
                    pid = pass_counts[lba] + 1
                    pass_counts[lba] = pid
                    pz_raw, qz_raw = r.count_pq(raw_2352)
                    conn.execute(
                        "INSERT OR REPLACE INTO raw_passes (lba, pass_id, raw_2352, c2_294, p_ok, q_ok) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (lba, pid, raw_2352, c2_294, pz_raw, qz_raw),
                    )
                    new_reads += 1
                    round_p_scores.append(f"{lba}:{pz_raw}/{qz_raw}")

                conn.commit()
                dt_r = time.time() - t_r
                print(
                    f"Pass {round_num:2d}/32 [{direction} {kbps:4d}KB/s lead={lead:2d} spd_st={st_spd}] "
                    f"({dt_r:.1f}s): " + " ".join(round_p_scores),
                    flush=True,
                )

            # Run portfolio decoder on all accumulated passes for each of the 6 LBAs
            print("\nRunning portfolio_decode across all accumulated passes...", flush=True)
            for lba in TARGET_LBAS:
                rows = conn.execute(
                    "SELECT pass_id, raw_2352, c2_294 FROM raw_passes WHERE lba = ? ORDER BY pass_id",
                    (lba,),
                ).fetchall()
                raw_passes = [(bytes(row[1]), bytes(row[2])) for row in rows]
                ok = r.process_lba_decode(conn, iso_fh, lba, raw_passes, rows[-1][0])
                st_row = conn.execute(
                    "SELECT verified, best_p_ok, best_q_ok, edc_ok, strategy FROM solved_sectors WHERE lba = ?",
                    (lba,),
                ).fetchone()
                print(
                    f"  LBA {lba}: total_passes={len(rows)}, verified={st_row[0]}, "
                    f"best_P={st_row[1]}/86, best_Q={st_row[2]}/52, edc={st_row[3]}, strat={st_row[4]}",
                    flush=True,
                )
                if ok:
                    newly_verified.append(lba)
            conn.commit()
            iso_fh.flush()
    finally:
        try:
            r.set_cd_speed(fd, 0xFFFF)
        finally:
            try:
                r.set_mode_01(fd, 0x00, 0x80)
            finally:
                os.close(fd)
        r.save_recovery_state(conn, all_bad_lbas)
        conn.close()

    print(
        f"\nHardware scan finished in {time.time() - t_start:.1f}s: "
        f"{new_reads} new raw passes stored, newly_verified={newly_verified}",
        flush=True,
    )


if __name__ == "__main__":
    main()
