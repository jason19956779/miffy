"""TDD tests for audio/video media format handlers and BMP sniffing bounds."""

import struct
import tempfile
import unittest
from pathlib import Path

from src.formats import FormatRegistry


def _make_valid_mp3(num_frames: int = 8, with_id3v2: bool = True, with_id3v1: bool = True) -> bytes:
    """Construct a valid MPEG-1 Layer III (128 kbps, 44.1 kHz, stereo) bitstream."""
    frame_hdr = bytes([0xFF, 0xFB, 0x90, 0x00])  # MPEG-1 Layer III 128kbps 44100Hz
    frame_len = 417
    frame = frame_hdr + bytes(((i * 13) & 0x7F) for i in range(frame_len - 4))
    out = bytearray()
    if with_id3v2:
        out.extend(b"ID3\x03\x00\x00\x00\x00\x00\x10")
        out.extend(b"\x00" * 16)
    for _ in range(num_frames):
        out.extend(frame)
    if with_id3v1:
        out.extend(b"TAG" + b"TestTitle".ljust(125, b"\x00"))
    return bytes(out)


def _make_valid_wav() -> bytes:
    pcm = b"\x01\x00\xff\x7f" * 16
    fmt_chunk = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 2, 44100, 176400, 4, 16)
    data_chunk = b"data" + struct.pack("<I", len(pcm)) + pcm
    riff_sz = 4 + len(fmt_chunk) + len(data_chunk)
    return b"RIFF" + struct.pack("<I", riff_sz) + b"WAVE" + fmt_chunk + data_chunk


def _make_valid_mpeg1_ps(packs: int = 4) -> bytes:
    pack_hdr = bytes.fromhex("000001ba2100010001801b91")
    vid_pes = b"\x00\x00\x01\xe0" + struct.pack(">H", 16) + (b"\x81" * 16)
    aud_pes = b"\x00\x00\x01\xc0" + struct.pack(">H", 16) + (b"\x91" * 16)
    return (pack_hdr + vid_pes + aud_pes) * packs + b"\x00\x00\x01\xb9"


class TestMediaHandlers(unittest.TestCase):
    def test_registry_registers_audio_and_video_handlers(self):
        names = {h.name for h in FormatRegistry.list_handlers()}
        self.assertIn("audio", names)
        self.assertIn("video", names)

    def test_audio_mp3_wav_flac_ogg_validation_and_corruption_detection(self):
        audio_h = FormatRegistry.get_handler(".MP3")
        self.assertEqual(audio_h.name, "audio")

        mp3 = _make_valid_mp3(num_frames=8, with_id3v2=True, with_id3v1=True)
        ok, status, meta = audio_h.validate(mp3)
        self.assertTrue(ok, f"Expected valid MP3, got {status}: {meta}")
        self.assertEqual(status, "AUDIO_MPEG_OK")
        self.assertEqual(meta["frames"], 8)
        self.assertEqual(meta["sample_rate"], 44100)
        self.assertEqual(meta["bitrate_kbps"], 128)

        # Streaming validate_file must match in-memory validate
        with tempfile.TemporaryDirectory() as tmp:
            p_mp3 = Path(tmp) / "track01.mp3"
            p_mp3.write_bytes(mp3)
            ok_f, st_f, meta_f = audio_h.validate_file(p_mp3)
            self.assertTrue(ok_f)
            self.assertEqual(st_f, "AUDIO_MPEG_OK")
            self.assertEqual(meta_f["frames"], 8)

            # Mid-stream 2048-byte zero hole must fail loudly
            p_bad = Path(tmp) / "bad.mp3"
            bad_mp3 = mp3[:500] + (b"\x00" * 2048) + mp3[500:]
            p_bad.write_bytes(bad_mp3)
            ok_b, st_b, _ = audio_h.validate_file(p_bad)
            self.assertFalse(ok_b)
            self.assertIn("AUDIO_STRUCT_ERR", st_b)

        # Truncated MP3 / garbage fails
        ok_garb, st_garb, _ = audio_h.validate(b"\x00" * 512)
        self.assertFalse(ok_garb)
        self.assertIn("AUDIO_STRUCT_ERR", st_garb)

        # WAV validation
        wav_h = FormatRegistry.get_handler(".WAV")
        wav_ok = _make_valid_wav()
        self.assertTrue(wav_h.validate(wav_ok)[0])
        self.assertFalse(wav_h.validate(wav_ok[:-10])[0])

        # FLAC validation (fLaC + STREAMINFO 34 bytes)
        flac_h = FormatRegistry.get_handler(".FLAC")
        si = bytearray(34)
        si[0:4] = struct.pack(">HH", 4096, 4096)
        si[10:14] = (44100 << 12 | 1 << 9 | 15 << 4).to_bytes(4, "big")
        flac_ok = b"fLaC" + bytes([0x80, 0x00, 0x00, 34]) + bytes(si)
        self.assertTrue(flac_h.validate(flac_ok)[0])
        self.assertFalse(flac_h.validate(b"fLaC" + b"\x00" * 8)[0])

        # OGG validation
        ogg_h = FormatRegistry.get_handler(".OGG")
        ogg_ok = b"OggS\x00\x02" + b"\x00" * 20 + b"\x01\x08\x01vorbis\x00"
        self.assertTrue(ogg_h.validate(ogg_ok)[0])
        self.assertFalse(ogg_h.validate(b"BAD!" + b"\x00" * 32)[0])

    def test_video_mpeg_ps_avi_mp4_mkv_validation_and_file_streaming(self):
        vid_h = FormatRegistry.get_handler(".MPG")
        self.assertEqual(vid_h.name, "video")

        mpg = _make_valid_mpeg1_ps(packs=5)
        ok, status, meta = vid_h.validate(mpg)
        self.assertTrue(ok, f"Expected valid MPEG PS, got {status}: {meta}")
        self.assertEqual(status, "VIDEO_MPEG_PS_OK")
        self.assertGreaterEqual(meta["pack_count"], 5)
        self.assertEqual(meta["mpeg_version"], 1)
        self.assertGreaterEqual(meta["video_packets"], 5)
        self.assertGreaterEqual(meta["audio_packets"], 5)

        with tempfile.TemporaryDirectory() as tmp:
            p_mpg = Path(tmp) / "AVSEQ01.mpg"
            p_mpg.write_bytes(mpg)
            ok_f, st_f, meta_f = vid_h.validate_file(p_mpg)
            self.assertTrue(ok_f)
            self.assertEqual(st_f, "VIDEO_MPEG_PS_OK")
            self.assertEqual(meta_f["pack_count"], meta["pack_count"])

            p_hole = Path(tmp) / "hole.mpg"
            p_hole.write_bytes(mpg[:30] + (b"\x00" * 2048) + mpg[30:])
            ok_h, st_h, meta_h = vid_h.validate_file(p_hole)
            self.assertFalse(ok_h)
            self.assertIn("VIDEO_STRUCT_ERR", st_h)
            self.assertGreaterEqual(meta_h["zero_holes_2k"], 1)

            # Unrecovered 2324-byte Mode 2 Form 2 zero hole must fail
            p_hole2324 = Path(tmp) / "hole2324.mpg"
            p_hole2324.write_bytes(mpg + (b"\x00" * 2324) + mpg)
            ok_h2, st_h2, _ = vid_h.validate_file(p_hole2324)
            self.assertFalse(ok_h2)
            self.assertIn("VIDEO_STRUCT_ERR", st_h2)

            # Valid VCD 2324-byte pack with PES_packet_length=0x0902 (2306) & 2292 zero stuffing bytes must PASS
            pack_hdr = bytes.fromhex("000001ba2100033ec1801b91")
            stuffed_pes = b"\x00\x00\x01\xe0" + struct.pack(">H", 2306) + (b"\xff" * 13) + b"\x0f" + (b"\x00" * 2292)
            vcd_stuffed_pack = pack_hdr + stuffed_pes
            self.assertEqual(len(vcd_stuffed_pack), 2324)
            p_vcd = Path(tmp) / "vcd_fade.mpg"
            p_vcd.write_bytes(vcd_stuffed_pack + mpg)
            ok_v, st_v, _ = vid_h.validate_file(p_vcd)
            self.assertTrue(ok_v, f"Legitimate VCD PES stuffing should pass, got {st_v}")
            self.assertEqual(st_v, "VIDEO_MPEG_PS_OK")

        self.assertFalse(vid_h.validate(b"\xff" * 256)[0])

        # AVI
        avi_h = FormatRegistry.get_handler(".AVI")
        avi_ok = b"RIFF" + struct.pack("<I", 20) + b"AVI " + b"LIST\x0c\x00\x00\x00hdrlavih\x00\x00\x00\x00"
        self.assertTrue(avi_h.validate(avi_ok)[0])
        self.assertFalse(avi_h.validate(avi_ok[:-6])[0])

        # MP4
        mp4_h = FormatRegistry.get_handler(".MP4")
        mp4_ok = (
            struct.pack(">I", 20) + b"ftypisom\x00\x00\x02\x00isom"
            + struct.pack(">I", 12) + b"moov\x00\x00\x00\x04"
            + struct.pack(">I", 12) + b"mdat\x00\x00\x00\x04"
        )
        self.assertTrue(mp4_h.validate(mp4_ok)[0])
        self.assertFalse(mp4_h.validate(b"\x00" * 32)[0])

        # MKV
        mkv_h = FormatRegistry.get_handler(".MKV")
        mkv_ok = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\x82\x88matroska" + b"\x00" * 16
        self.assertTrue(mkv_h.validate(mkv_ok)[0])
        self.assertFalse(mkv_h.validate(b"nope" * 8)[0])

    def test_bmp_handler_does_not_claim_fnt_files(self):
        bmp_h = FormatRegistry.get_handler(".BMP")
        fake_fnt = b"BM" + b"\x00" * 128
        self.assertFalse(bmp_h.can_handle(".FNT", fake_fnt))
        self.assertTrue(bmp_h.can_handle(".BMP", fake_fnt))
        self.assertTrue(bmp_h.can_handle("", fake_fnt))
        self.assertEqual(FormatRegistry.get_handler(".FNT", fake_fnt).name, "generic_binary")


if __name__ == "__main__":
    unittest.main()
