"""Generic Pillow image validation handler for PNG, GIF, TIFF, and WebP."""

import io
import zlib

from PIL import Image, ImageFile

from src.formats import FormatRegistry
from src.formats.base import FileFormatHandler


@FormatRegistry.register
class PillowGenericImageHandler(FileFormatHandler):
    name = "image_generic"
    extensions = (".PNG", ".GIF", ".TIF", ".TIFF", ".WEBP")
    priority = 18
    description = "Structure/CRC & strict PIL validation for PNG, GIF, TIFF, and WebP images."

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        meta: dict = {}
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            pos = 8
            bad_crc = 0
            while pos + 12 <= len(data):
                ln = int.from_bytes(data[pos : pos + 4], "big")
                ctype = data[pos + 4 : pos + 8]
                if pos + 12 + ln > len(data):
                    break
                crc_calc = zlib.crc32(data[pos + 4 : pos + 8 + ln]) & 0xFFFFFFFF
                crc_file = int.from_bytes(data[pos + 8 + ln : pos + 12 + ln], "big")
                if crc_calc != crc_file:
                    bad_crc += 1
                pos += 12 + ln
                if ctype == b"IEND":
                    break
            meta["png_bad_crc_chunks"] = bad_crc
            if bad_crc > 0:
                return False, "PNG_CRC_ERROR", meta

        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im2:
                im2.load()
                meta["dims"] = f"{im2.size[0]}x{im2.size[1]} {im2.mode}"
            if meta.get("png_bad_crc_chunks", 0) > 0:
                return False, "PNG_CRC_ERROR", meta
            return True, "STRICT_OK", meta
        except Exception as exc:
            return False, f"DECODE_ERR ({type(exc).__name__}: {exc})", meta
