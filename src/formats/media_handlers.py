"""Audio and video container/bitstream validation handlers plus optional FFmpeg audit."""

import io
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import BinaryIO

from src.formats import FormatRegistry
from src.formats.base import FileFormatHandler

_MPEG_SR_TABLES = {
    3: (44100, 48000, 32000),  # MPEG-1
    2: (22050, 24000, 16000),  # MPEG-2
    0: (11025, 12000, 8000),   # MPEG-2.5
}

_MPEG_BR_V1 = {
    3: (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0),  # Layer I
    2: (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0),     # Layer II
    1: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0),      # Layer III
}

_MPEG_BR_V2 = {
    3: (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0),     # Layer I
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),          # Layer II
    1: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),          # Layer III
}


def _parse_mpeg_audio_frame_header(b0: int, b1: int, b2: int) -> tuple[int, int, int, int, int] | None:
    """Return `(frame_len, sample_rate, bitrate_kbps, version_id, layer_bits)` or None."""
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    ver_id = (b1 >> 3) & 0x03
    layer_bits = (b1 >> 1) & 0x03
    br_idx = (b2 >> 4) & 0x0F
    sr_idx = (b2 >> 2) & 0x03
    pad = (b2 >> 1) & 0x01
    if ver_id == 1 or layer_bits == 0 or br_idx in (0, 15) or sr_idx == 3:
        return None
    sr = _MPEG_SR_TABLES[ver_id][sr_idx]
    br_tbl = _MPEG_BR_V1 if ver_id == 3 else _MPEG_BR_V2
    br = br_tbl[layer_bits][br_idx]
    if br <= 0 or sr <= 0:
        return None
    if layer_bits == 3:
        flen = ((12 * br * 1000 // sr) + pad) * 4
    elif layer_bits == 2 or ver_id == 3:
        flen = (144 * br * 1000 // sr) + pad
    else:
        flen = (72 * br * 1000 // sr) + pad
    return flen, sr, br, ver_id, layer_bits


def _validate_mpeg_audio(data: bytes) -> tuple[bool, str, dict]:
    n = len(data)
    if n < 4:
        return False, "AUDIO_STRUCT_ERR (too_short)", {"bytes": n}

    pos = 0
    has_id3v2 = False
    if data[:3] == b"ID3" and n >= 10:
        has_id3v2 = True
        tag_sz = (
            ((data[6] & 0x7F) << 21)
            | ((data[7] & 0x7F) << 14)
            | ((data[8] & 0x7F) << 7)
            | (data[9] & 0x7F)
        )
        pos = min(n, 10 + tag_sz + (10 if (data[5] & 0x10) else 0))

    end = n
    has_id3v1 = False
    if end - pos >= 128 and data[end - 128 : end - 125] == b"TAG":
        has_id3v1 = True
        end -= 128
    # Strip trailing APEv2 tag if present before ID3v1/EOF
    if end - pos >= 32 and data[end - 32 : end - 24] == b"APETAGEX":
        ape_sz = struct.unpack_from("<I", data, end - 20)[0]
        if 32 <= ape_sz + 32 <= end - pos:
            end -= ape_sz + 32

    if data.find(b"\x00" * 2048, pos, end) != -1:
        return False, "AUDIO_STRUCT_ERR (zero_hole_2k)", {"bytes": n}

    # Allow small leading zero/tag padding before the first MPEG sync header
    scan_limit = min(end - 3, pos + 4096)
    first_pos = None
    while pos < scan_limit:
        if data[pos] == 0xFF and _parse_mpeg_audio_frame_header(data[pos], data[pos + 1], data[pos + 2]):
            first_pos = pos
            break
        pos += 1

    if first_pos is None:
        return False, "AUDIO_STRUCT_ERR (no_mpeg_audio_sync)", {"bytes": n}

    frames = 0
    sample_rate = 0
    bitrate_sum = 0
    layer_bits = 0
    pos = first_pos
    while pos + 4 <= end:
        hdr = _parse_mpeg_audio_frame_header(data[pos], data[pos + 1], data[pos + 2])
        if hdr is None:
            if data[pos : pos + 3] == b"TAG":
                break
            rem = data[pos:end]
            if len(rem) <= 32 and all(b == 0 for b in rem):
                break
            return False, f"AUDIO_STRUCT_ERR (frame_desync_at_{pos})", {
                "frames": frames,
                "offset": pos,
            }
        flen, sr, br, _, lyr = hdr
        if pos + flen > end:
            if end - pos < flen and frames >= 1 and pos + 4 == end:
                break
            return False, f"AUDIO_STRUCT_ERR (truncated_frame_at_{pos})", {
                "frames": frames,
                "offset": pos,
            }
        frames += 1
        sample_rate = sr
        bitrate_sum += br
        layer_bits = lyr
        pos += flen

    if frames == 0:
        return False, "AUDIO_STRUCT_ERR (zero_valid_frames)", {"bytes": n}

    return True, "AUDIO_MPEG_OK", {
        "bytes": n,
        "frames": frames,
        "sample_rate": sample_rate,
        "bitrate_kbps": round(bitrate_sum / frames, 1),
        "layer": {3: 1, 2: 2, 1: 3}.get(layer_bits, 0),
        "id3v2": has_id3v2,
        "id3v1": has_id3v1,
    }


def _validate_wav(data: bytes) -> tuple[bool, str, dict]:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False, "AUDIO_STRUCT_ERR (bad_riff_wave_header)", {"bytes": len(data)}
    riff_sz = struct.unpack_from("<I", data, 4)[0]
    if 8 + riff_sz > len(data) + 8:
        return False, f"AUDIO_STRUCT_ERR (truncated_riff:{len(data)}<{8 + riff_sz})", {"bytes": len(data)}
    pos = 12
    has_fmt = False
    has_data = False
    channels = 0
    sr = 0
    bps = 0
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csz = struct.unpack_from("<I", data, pos + 4)[0]
        body_s = pos + 8
        if body_s + csz > len(data):
            return False, f"AUDIO_STRUCT_ERR (truncated_chunk:{cid!r})", {"bytes": len(data)}
        if cid == b"fmt " and csz >= 16:
            has_fmt = True
            _, channels, sr, _, _, bps = struct.unpack_from("<HHIIHH", data, body_s)
        elif cid == b"data":
            has_data = True
        pos = body_s + csz + (csz & 1)
    if not (has_fmt and has_data):
        return False, "AUDIO_STRUCT_ERR (missing_fmt_or_data)", {"bytes": len(data)}
    return True, "AUDIO_WAV_OK", {"bytes": len(data), "channels": channels, "sample_rate": sr, "bits": bps}


def _validate_flac(data: bytes) -> tuple[bool, str, dict]:
    if len(data) < 42 or data[:4] != b"fLaC":
        return False, "AUDIO_STRUCT_ERR (bad_flac_magic)", {"bytes": len(data)}
    blk_hdr = int.from_bytes(data[4:8], "big")
    blk_type = (blk_hdr >> 24) & 0x7F
    blk_len = blk_hdr & 0xFFFFFF
    if blk_type != 0 or blk_len != 34 or len(data) < 8 + blk_len:
        return False, "AUDIO_STRUCT_ERR (invalid_flac_streaminfo)", {"bytes": len(data)}
    if data.find(b"\x00" * 2048, 42) != -1:
        return False, "AUDIO_STRUCT_ERR (zero_hole_2k)", {"bytes": len(data)}
    sr_ch_bps = int.from_bytes(data[18:22], "big")
    sr = (sr_ch_bps >> 12) & 0xFFFFF
    channels = ((sr_ch_bps >> 9) & 0x07) + 1
    return True, "AUDIO_FLAC_OK", {"bytes": len(data), "sample_rate": sr, "channels": channels}


def _validate_ogg(data: bytes) -> tuple[bool, str, dict]:
    if len(data) < 28 or data[:4] != b"OggS" or data[4] != 0:
        return False, "AUDIO_STRUCT_ERR (bad_ogg_header)", {"bytes": len(data)}
    n_segs = data[26]
    if len(data) < 27 + n_segs:
        return False, "AUDIO_STRUCT_ERR (truncated_ogg_page)", {"bytes": len(data)}
    if data.find(b"\x00" * 2048) != -1:
        return False, "AUDIO_STRUCT_ERR (zero_hole_2k)", {"bytes": len(data)}
    return True, "AUDIO_OGG_OK", {"bytes": len(data)}


def _has_corrupt_mpeg_zero_hole(data: bytes, is_last_chunk: bool) -> bool:
    """Distinguish real >=2048B zero holes from legitimate VCD PES zero stuffing (<2324B)."""
    if data.find(b"\x00" * 2324) != -1:
        return True
    z2k = b"\x00" * 2048
    search_pos = 0
    while True:
        pos = data.find(z2k, search_pos)
        if pos == -1:
            return False
        z_start = pos
        while z_start > 0 and data[z_start - 1] == 0:
            z_start -= 1
        z_end = pos + 2048
        while z_end < len(data) and data[z_end] == 0:
            z_end += 1
        if z_end - z_start >= 2324:
            return True
        if not is_last_chunk and z_start + 2350 >= len(data):
            return False

        # Search backward up to 2324B for enclosing PES header whose length cleanly frames the zeros
        lookback = max(0, z_start - 2324)
        valid_pes_stuffing = False
        sc_pos = lookback
        pes_candidates: list[int] = []
        while True:
            idx = data.find(b"\x00\x00\x01", sc_pos, z_start)
            if idx == -1:
                break
            if idx + 6 <= len(data) and 0xBB <= data[idx + 3] <= 0xEF:
                pes_candidates.append(idx)
            sc_pos = idx + 3

        for p_idx in reversed(pes_candidates):
            pes_len = struct.unpack_from(">H", data, p_idx + 4)[0]
            pes_end = p_idx + 6 + pes_len
            if pes_end >= z_end - 2 and pes_end <= len(data):
                if pes_end == len(data) or data[pes_end : pes_end + 3] == b"\x00\x00\x01":
                    valid_pes_stuffing = True
                    break
        if not valid_pes_stuffing:
            return True
        search_pos = z_end


def _scan_mpeg_ps_stream(fh: BinaryIO, total_bytes: int) -> tuple[bool, str, dict]:
    """Constant-memory C-accelerated MPEG-1/2 Program Stream scanner."""
    chunk_sz = 2 * 1024 * 1024
    pack_code = b"\x00\x00\x01\xba"
    pack_count = 0
    video_packets = 0
    audio_packets = 0
    zero_holes = 0
    mpeg_version = 0

    buf = b""
    offset = 0
    while True:
        chunk = fh.read(chunk_sz)
        if not chunk:
            break
        data = buf + chunk
        is_last = offset + len(chunk) >= total_bytes
        overlap = min(len(data), 4096) if not is_last else 0
        scan_end = len(data) - 4

        pos = 0
        while True:
            idx = data.find(pack_code, pos, scan_end + 1)
            if idx == -1:
                break
            if idx < len(buf) and offset > 0:
                pos = idx + 4
                continue
            pack_count += 1
            if mpeg_version == 0 and idx + 4 < len(data):
                b4 = data[idx + 4]
                if (b4 & 0xF0) == 0x20:
                    mpeg_version = 1
                elif (b4 & 0xC0) == 0x40:
                    mpeg_version = 2
            pos = idx + 4

        # Fast scan for 0x000001 start codes (video 0xE0..0xEF, audio 0xC0..0xDF)
        p3 = 0
        while True:
            idx3 = data.find(b"\x00\x00\x01", p3, len(data) - 1)
            if idx3 == -1:
                break
            if idx3 >= (len(buf) if offset > 0 else 0):
                sc = data[idx3 + 3]
                if 0xE0 <= sc <= 0xEF:
                    video_packets += 1
                elif 0xC0 <= sc <= 0xDF:
                    audio_packets += 1
            p3 = idx3 + 3

        if zero_holes == 0 and _has_corrupt_mpeg_zero_hole(data, is_last):
            zero_holes += 1

        offset += len(chunk)
        buf = data[-overlap:] if overlap else b""

    meta = {
        "bytes": total_bytes,
        "mpeg_version": mpeg_version or 1,
        "pack_count": pack_count,
        "video_packets": video_packets,
        "audio_packets": audio_packets,
        "zero_holes_2k": zero_holes,
    }
    if zero_holes > 0:
        return False, "VIDEO_STRUCT_ERR (zero_hole_2k)", meta
    if pack_count == 0 or (video_packets == 0 and audio_packets == 0):
        return False, "VIDEO_STRUCT_ERR (no_mpeg_ps_packs)", meta
    return True, "VIDEO_MPEG_PS_OK", meta


def _validate_avi(data: bytes) -> tuple[bool, str, dict]:
    if len(data) < 32 or data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        return False, "VIDEO_STRUCT_ERR (bad_avi_header)", {"bytes": len(data)}
    riff_sz = struct.unpack_from("<I", data, 4)[0]
    if 8 + riff_sz > len(data) + 8:
        return False, f"VIDEO_STRUCT_ERR (truncated_avi:{len(data)}<{8 + riff_sz})", {"bytes": len(data)}
    if data.find(b"\x00" * 2048) != -1:
        return False, "VIDEO_STRUCT_ERR (zero_hole_2k)", {"bytes": len(data)}
    return True, "VIDEO_AVI_OK", {"bytes": len(data)}


def _validate_mp4(data: bytes) -> tuple[bool, str, dict]:
    if len(data) < 24 or data[4:8] != b"ftyp":
        return False, "VIDEO_STRUCT_ERR (missing_ftyp_box)", {"bytes": len(data)}
    boxes: list[str] = []
    pos = 0
    while pos + 8 <= len(data):
        bsz = struct.unpack_from(">I", data, pos)[0]
        btype = data[pos + 4 : pos + 8].decode("latin-1", errors="replace")
        if bsz == 1 and pos + 16 <= len(data):
            bsz = struct.unpack_from(">Q", data, pos + 8)[0]
        elif bsz == 0:
            bsz = len(data) - pos
        if bsz < 8 or pos + bsz > len(data):
            return False, f"VIDEO_STRUCT_ERR (bad_box:{btype})", {"bytes": len(data)}
        boxes.append(btype)
        pos += bsz
    if "moov" not in boxes or "mdat" not in boxes:
        return False, "VIDEO_STRUCT_ERR (missing_moov_or_mdat)", {"bytes": len(data), "boxes": boxes}
    return True, "VIDEO_MP4_OK", {"bytes": len(data), "boxes": boxes}


def _validate_mkv(data: bytes) -> tuple[bool, str, dict]:
    if len(data) < 16 or data[:4] != b"\x1a\x45\xdf\xa3":
        return False, "VIDEO_STRUCT_ERR (bad_ebml_magic)", {"bytes": len(data)}
    if data.find(b"\x00" * 2048) != -1:
        return False, "VIDEO_STRUCT_ERR (zero_hole_2k)", {"bytes": len(data)}
    return True, "VIDEO_MKV_OK", {"bytes": len(data)}


def verify_media_with_ffmpeg(path: Path) -> tuple[bool, str, dict]:
    """Run `ffmpeg -v error -xerror` + `ffprobe` on an audio/video file if available."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return True, "FFMPEG_NOT_INSTALLED", {}
    res = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    meta: dict = {"ffmpeg_rc": res.returncode, "ffmpeg_stderr": res.stderr.strip()}
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        pr = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=format_name,duration,bit_rate:stream=codec_type,codec_name,width,height,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if pr.returncode == 0 and pr.stdout.strip():
            try:
                parsed = json.loads(pr.stdout)
                fmt = parsed.get("format", {})
                streams = parsed.get("streams", [])
                if fmt.get("duration"):
                    meta["duration_s"] = round(float(fmt["duration"]), 3)
                if fmt.get("bit_rate"):
                    meta["bit_rate"] = int(fmt["bit_rate"])
                meta["streams"] = [
                    {k: s[k] for k in ("codec_type", "codec_name", "width", "height", "sample_rate", "channels") if k in s}
                    for s in streams
                ]
            except Exception:
                pass
    if res.returncode == 0 and not res.stderr.strip():
        return True, "FFMPEG_OK", meta
    return False, f"FFMPEG_ERR (rc={res.returncode})", meta


@FormatRegistry.register
class AudioFormatHandler(FileFormatHandler):
    name = "audio"
    extensions = (".MP3", ".MP2", ".WAV", ".FLAC", ".OGG", ".M4A", ".AAC", ".WMA")
    priority = 18
    description = "MPEG-1/2 audio frame & RIFF/FLAC/OGG container structural validator."

    def can_handle(self, ext: str, data: bytes | None = None) -> bool:
        return ext.upper() in self.extensions

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return _validate_wav(data)
        if data[:4] == b"fLaC":
            return _validate_flac(data)
        if data[:4] == b"OggS":
            return _validate_ogg(data)
        if data[:3] == b"ID3" or (len(data) >= 3 and _parse_mpeg_audio_frame_header(data[0], data[1], data[2])):
            return _validate_mpeg_audio(data)
        if data.find(b"\x00" * 2048) != -1:
            return False, "AUDIO_STRUCT_ERR (zero_hole_2k)", {"bytes": len(data)}
        if len(data) >= 8 and b"ftyp" in data[4:12]:
            return True, "AUDIO_M4A_OK", {"bytes": len(data)}
        return _validate_mpeg_audio(data)

    def validate_file(self, path: Path) -> tuple[bool, str, dict]:
        return self.validate(path.read_bytes())


@FormatRegistry.register
class VideoFormatHandler(FileFormatHandler):
    name = "video"
    extensions = (".MPG", ".MPEG", ".MP4", ".AVI", ".MOV", ".MKV", ".WMV", ".VOB")
    priority = 18
    description = "Streaming MPEG-1/2 Program Stream & AVI/MP4/MKV container structural validator."

    def can_handle(self, ext: str, data: bytes | None = None) -> bool:
        return ext.upper() in self.extensions

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
            return _validate_avi(data)
        if len(data) >= 8 and data[4:8] == b"ftyp":
            return _validate_mp4(data)
        if data[:4] == b"\x1a\x45\xdf\xa3":
            return _validate_mkv(data)
        return _scan_mpeg_ps_stream(io.BytesIO(data), len(data))

    def validate_file(self, path: Path) -> tuple[bool, str, dict]:
        ext = path.suffix.upper()
        sz = path.stat().st_size
        if ext in (".MPG", ".MPEG", ".VOB"):
            with path.open("rb") as fh:
                return _scan_mpeg_ps_stream(fh, sz)
        return self.validate(path.read_bytes())
