"""Low-level Linux SG_IO optical drive control, auto-detection, and safe restore."""

import ctypes
import fcntl
import os
from pathlib import Path

from src.core.models import DeviceCapabilities

SG_IO = 0x2285


class SgIoHdr(ctypes.Structure):
    _fields_ = [
        ("interface_id", ctypes.c_int),
        ("dxfer_direction", ctypes.c_int),
        ("cmd_len", ctypes.c_ubyte),
        ("mx_sb_len", ctypes.c_ubyte),
        ("iovec_count", ctypes.c_ushort),
        ("dxfer_len", ctypes.c_uint),
        ("dxferp", ctypes.c_void_p),
        ("cmdp", ctypes.c_void_p),
        ("sbp", ctypes.c_void_p),
        ("timeout", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("pack_id", ctypes.c_int),
        ("usr_ptr", ctypes.c_void_p),
        ("status", ctypes.c_ubyte),
        ("masked_status", ctypes.c_ubyte),
        ("msg_status", ctypes.c_ubyte),
        ("sb_len_wr", ctypes.c_ubyte),
        ("host_status", ctypes.c_ushort),
        ("driver_status", ctypes.c_ushort),
        ("resid", ctypes.c_int),
        ("duration", ctypes.c_uint),
        ("info", ctypes.c_uint),
    ]


def auto_detect_sg_device(preferred: str | None = None) -> tuple[str, str | None, int]:
    """Locate optical SCSI generic (/dev/sgN), paired /dev/srM, and max_sectors_kb."""
    if preferred:
        sg_name = Path(preferred).name
        sys_sg = Path("/sys/class/scsi_generic") / sg_name
    else:
        sys_sg = None
        for cand in sorted(Path("/sys/class/scsi_generic").glob("sg*")):
            t_file = cand / "device" / "type"
            if t_file.exists() and t_file.read_text(errors="ignore").strip() == "5":
                sys_sg = cand
                break
        if sys_sg is None:
            sys_sg = Path("/sys/class/scsi_generic/sg0")
        sg_name = sys_sg.name

    sr_name = None
    max_kb = 120
    block_dir = sys_sg / "device" / "block"
    if block_dir.exists():
        bdevs = sorted(p.name for p in block_dir.iterdir())
        if bdevs:
            sr_name = f"/dev/{bdevs[0]}"
            kb_path = Path(f"/sys/block/{bdevs[0]}/queue/max_sectors_kb")
            if kb_path.exists():
                try:
                    max_kb = int(kb_path.read_text().strip())
                except ValueError:
                    max_kb = 120
    return f"/dev/{sg_name}", sr_name, max_kb


class OpticalDrive:
    """Context-managed SG_IO interface that always restores Mode Page 0x01 and speed."""

    def __init__(self, device_path: str | None = None):
        sg_dev, sr_dev, max_kb = auto_detect_sg_device(device_path)
        self.sg_device = sg_dev
        self.sr_device = sr_dev
        self.max_sectors_kb = max_kb
        self.max_batch_sectors = min(44, max(1, (max_kb * 1024 - 4096) // 2646))
        self.fd: int | None = None
        self.caps: DeviceCapabilities | None = None

    def __enter__(self) -> "OpticalDrive":
        self.fd = os.open(self.sg_device, os.O_RDWR | os.O_NONBLOCK)
        try:
            self.caps = self.probe_capabilities()
            self.enable_dcr_raw_c2()
            return self
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.fd is None:
            return
        try:
            self.set_speed(0xFFFF)
        finally:
            try:
                self.set_mode_01(0x00, 0x80)
            finally:
                os.close(self.fd)
                self.fd = None

    def scsi_cmd(
        self,
        cdb_bytes: list[int],
        buf_bytes: list[int] | None = None,
        buf_len: int = 256,
        direction: int = -3,
        timeout: int = 25000,
    ) -> tuple[int, tuple[int, int, int], int, bytes]:
        if self.fd is None:
            raise RuntimeError("OpticalDrive not opened")
        cdb = (ctypes.c_ubyte * len(cdb_bytes))(*cdb_bytes)
        if buf_bytes is not None:
            buf_len = len(buf_bytes)
            buf = (ctypes.c_ubyte * buf_len)(*buf_bytes)
        else:
            buf = (ctypes.c_ubyte * buf_len)()
        sense = (ctypes.c_ubyte * 64)()
        hdr = SgIoHdr(
            interface_id=ord("S"),
            dxfer_direction=direction,
            cmd_len=len(cdb_bytes),
            mx_sb_len=64,
            dxfer_len=buf_len,
            dxferp=ctypes.addressof(buf),
            cmdp=ctypes.addressof(cdb),
            sbp=ctypes.addressof(sense),
            timeout=timeout,
        )
        fcntl.ioctl(self.fd, SG_IO, hdr)
        sb = bytes(sense[: hdr.sb_len_wr])
        sk = sb[2] & 0x0F if len(sb) > 2 else 0
        asc = sb[12] if len(sb) > 12 else 0
        ascq = sb[13] if len(sb) > 13 else 0
        return hdr.status, (sk, asc, ascq), hdr.resid, bytes(buf)

    def probe_capabilities(self) -> DeviceCapabilities:
        st_inq, _, _, inq = self.scsi_cmd([0x12, 0, 0, 0, 36, 0], buf_len=36, direction=-3)
        vendor = (
            inq[8:32].decode("ascii", errors="replace").strip()
            if st_inq == 0 and len(inq) >= 32
            else "Unknown Optical Drive"
        )
        st_2a, _, _, p2a = self.scsi_cmd(
            [0x5A, 0, 0x2A, 0, 0, 0, 0, 0, 64, 0], buf_len=64, direction=-3
        )
        supports_c2 = bool(st_2a == 0 and len(p2a) >= 24 and (p2a[13] & 0x10))
        max_spd = int.from_bytes(p2a[16:18], "big") if st_2a == 0 and len(p2a) >= 18 else 0
        cur_spd = int.from_bytes(p2a[22:24], "big") if st_2a == 0 and len(p2a) >= 24 else 0
        return DeviceCapabilities(
            sg_device=self.sg_device,
            sr_device=self.sr_device,
            vendor_model=vendor,
            supports_c2=supports_c2,
            max_speed_kbps=max_spd,
            cur_speed_kbps=cur_spd,
            max_sectors_kb=self.max_sectors_kb,
            max_batch_sectors=self.max_batch_sectors,
        )

    def set_mode_01(self, err_bits: int, retries: int) -> None:
        payload = [0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x0A, err_bits, retries, 0, 0, 0, 0, 0, 0, 0, 0]
        st, sense, _, _ = self.scsi_cmd(
            [0x55, 0x10, 0, 0, 0, 0, 0, 0, len(payload), 0],
            buf_bytes=payload,
            direction=-2,
        )
        if st != 0:
            raise RuntimeError(f"MODE SELECT Page 0x01 (0x{err_bits:02X}) failed: st={st} sense={sense}")

    def enable_dcr_raw_c2(self) -> int:
        for bits in (0x01, 0x21):
            try:
                self.set_mode_01(bits, 0x00)
                return bits
            except RuntimeError:
                continue
        raise RuntimeError("Drive rejected DCR Mode Page 0x01 configuration")

    def set_speed(self, kbps: int) -> int:
        st, _, _, _ = self.scsi_cmd(
            [0xBB, 0, (kbps >> 8) & 0xFF, kbps & 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0],
            buf_len=0,
            direction=0,
        )
        return st

    def read_cd_raw(
        self, start_lba: int, count: int, with_c2: bool = True
    ) -> tuple[int, tuple[int, int, int], int, bytes]:
        flags = 0xFA if with_c2 else 0xF8
        sec_size = 2646 if with_c2 else 2352
        cdb = [
            0xBE,
            0,
            (start_lba >> 24) & 0xFF,
            (start_lba >> 16) & 0xFF,
            (start_lba >> 8) & 0xFF,
            start_lba & 0xFF,
            (count >> 16) & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
            flags,
            0,
            0,
        ]
        return self.scsi_cmd(cdb, buf_len=sec_size * count, timeout=25000)

    def read_cooked_sector(self, lba: int) -> bytes | None:
        st, _, resid, raw = self.read_cd_raw(lba, 1, with_c2=False)
        if st == 0 and resid == 0 and len(raw) >= 2064:
            return raw[16:2064]
        return None

    def read_total_volume_sectors(self) -> int:
        """Read ISO-9660 PVD (LBA 16) or READ CAPACITY (0x25) to find volume sector count."""
        pvd = self.read_cooked_sector(16)
        if pvd and pvd[1:6] == b"CD001":
            return int.from_bytes(pvd[80:84], "little")
        st, _, _, cap = self.scsi_cmd([0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0], buf_len=8, direction=-3)
        if st == 0 and len(cap) >= 8:
            return int.from_bytes(cap[0:4], "big") + 1
        raise RuntimeError("Unable to determine CD volume sector count")

    def directional_seek_flush(
        self, target_lba: int, direction: str, total_sectors: int, offset_seed: int = 0
    ) -> None:
        """Evict drive cache and force inner-outward (FWD) or outer-inward (REV) sled seek."""
        jitter = (offset_seed * 17) % 100
        if direction == "REV":
            seek_lba = min(total_sectors - 1, target_lba + 800 + jitter)
        else:
            seek_lba = max(0, target_lba - 800 - jitter)
        self.read_cd_raw(seek_lba, 1, with_c2=False)
