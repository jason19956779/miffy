"""Non-image file format handlers: text/INI, Macromedia Director RIFX/CXT, PE binaries, generic."""

from src.formats import FormatRegistry
from src.formats.base import FileFormatHandler


@FormatRegistry.register
class TextIniHandler(FileFormatHandler):
    name = "text_ini"
    extensions = (".INI", ".TXT", ".CFG", ".CSV", ".LOG", ".HTM", ".HTML", ".INF")
    priority = 30
    description = "Plain-text / INI encoding and control-character integrity checker."

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        if not data:
            return True, "EMPTY_TEXT", {}
        nul_cnt = data.count(0)
        printable = sum(1 for b in data if b in (9, 10, 13) or 32 <= b <= 126 or b >= 128)
        ratio = printable / len(data)
        if nul_cnt == 0 and ratio >= 0.98:
            return True, "TEXT_OK", {"bytes": len(data)}
        return False, f"TEXT_CORRUPT (nuls={nul_cnt}, printable={ratio:.2f})", {"nuls": nul_cnt}


@FormatRegistry.register
class DirectorRifxHandler(FileFormatHandler):
    name = "director_rifx"
    extensions = (".DXR", ".CXT", ".DIR", ".CST")
    priority = 40
    description = "Macromedia Director RIFX/XFIR multimedia container validator."

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        if len(data) < 12:
            return False, "TRUNCATED_RIFX", {}
        sig = data[:4]
        if sig == b"RIFX":
            declared = int.from_bytes(data[4:8], "big") + 8
            codec = data[8:12].decode("ascii", errors="replace")
        elif sig == b"XFIR":
            declared = int.from_bytes(data[4:8], "little") + 8
            codec = data[8:12].decode("ascii", errors="replace")
        else:
            return False, f"BAD_RIFX_MAGIC ({sig!r})", {}
        return True, "RIFX_HEADER_OK", {"codec": codec, "declared_size": declared, "actual_size": len(data)}


@FormatRegistry.register
class PeBinaryHandler(FileFormatHandler):
    name = "pe_binary"
    extensions = (".DLL", ".EXE", ".X32")
    priority = 40
    description = "Windows PE/COFF MZ executable & Xtra plugin header validator."

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        if len(data) < 64 or data[:2] != b"MZ":
            return False, "BAD_MZ_HEADER", {}
        pe_off = int.from_bytes(data[60:64], "little")
        if pe_off + 24 <= len(data) and data[pe_off : pe_off + 4] == b"PE\x00\x00":
            n_sec = int.from_bytes(data[pe_off + 6 : pe_off + 8], "little")
            return True, "PE_HEADER_OK", {"sections": n_sec}
        return False, "BAD_PE_SIGNATURE", {}


@FormatRegistry.register
class GenericBinaryHandler(FileFormatHandler):
    name = "generic_binary"
    extensions = ("*",)
    priority = 99
    description = "Fallback binary file integrity handler for unrecognized extensions."

    def can_handle(self, ext: str, data: bytes | None = None) -> bool:
        return True

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        zero_2k = sum(
            1 for off in range(0, max(0, len(data) - 2047), 2048) if data[off : off + 2048] == b"\x00" * 2048
        )
        return (zero_2k == 0), ("BINARY_OK" if zero_2k == 0 else f"ZERO_HOLES ({zero_2k})"), {"size": len(data)}
