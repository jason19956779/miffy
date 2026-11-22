"""Windows BMP format validation and scanline-level repair handler."""

import io
import struct
from typing import Callable

from PIL import Image

from src.formats import FormatRegistry
from src.formats.base import FileFormatHandler
from src.core.models import FileExtent


@FormatRegistry.register
class BmpFormatHandler(FileFormatHandler):
    name = "bmp"
    extensions = (".BMP", ".DIB")
    priority = 15
    description = "Uncompressed BMP header validator and scanline inpainting handler."

    def can_handle(self, ext: str, data: bytes | None = None) -> bool:
        if ext.upper() in self.extensions:
            return True
        return bool(data and data[:2] == b"BM")

    def find_reference(
        self,
        extent: FileExtent,
        all_extents: dict[str, FileExtent],
        read_file_fn: Callable[[FileExtent], bytes],
        file_bytes: bytes | None = None,
    ) -> tuple[bytes | None, str | None]:
        # Pair companion small/large BMP variants if present (e.g., GL_12.BMP <-> GS_12.BMP)
        if extent.name.upper().startswith("GL_"):
            all_upper = {k.upper(): v for k, v in all_extents.items()}
            cand = (extent.directory.rstrip("/") + "/GS_" + extent.name[3:]).upper()
            if cand in all_upper and not all_upper[cand].bad_lbas:
                return read_file_fn(all_upper[cand]), all_upper[cand].path
        return None, None

    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im2:
                im2.load()
                return True, "STRICT_OK", {"dims": f"{im2.size[0]}x{im2.size[1]} {im2.mode}"}
        except Exception as exc:
            return False, f"DECODE_ERR ({type(exc).__name__}: {exc})", {}

    def reconstruct(
        self,
        data: bytes,
        bad_intervals: list[tuple[int, int]],
        reference_data: bytes | None = None,
        context: dict | None = None,
    ) -> tuple[bytes, str, dict]:
        if len(data) < 54 or data[:2] != b"BM":
            return data, "INVALID_BMP_HEADER", {}
        _, _, _, _, data_off = struct.unpack("<2sIHHI", data[:14])
        dib_sz, w, h, planes, bpp, comp, _ = struct.unpack("<IiiHHII", data[14:38])
        if comp != 0 or bpp != 24 or planes != 1 or w <= 0 or h == 0:
            return data, "UNSUPPORTED_BMP_ENCODING", {}

        abs_h = abs(h)
        stride = (w * 3 + 3) & ~3
        if data_off + stride * abs_h > len(data):
            return data, "TRUNCATED_BMP", {}

        out = bytearray(data)
        bad_rows = set()
        for s_off, e_off in bad_intervals:
            p0 = max(data_off, s_off)
            p1 = min(data_off + stride * abs_h, e_off)
            for p in range(p0, p1):
                bad_rows.add((p - data_off) // stride)

        if not bad_rows:
            return data, "NO_PIXEL_DAMAGE", {}

        # Optional reference image overlay
        ref_pixels = None
        if reference_data:
            try:
                with Image.open(io.BytesIO(reference_data)) as ref_im:
                    ref_rgb = ref_im.convert("RGB").resize((w, abs_h), Image.Resampling.BILINEAR)
                    ref_pixels = ref_rgb.tobytes()
            except Exception:
                ref_pixels = None

        good_rows = [r for r in range(abs_h) if r not in bad_rows]
        if not good_rows and ref_pixels is None:
            return data, "NO_INTACT_SCANLINES", {}

        for s_off, e_off in bad_intervals:
            p0 = max(data_off, s_off)
            p1 = min(data_off + stride * abs_h, e_off)
            for p in range(p0, p1):
                rel = p - data_off
                row = rel // stride
                col_b = rel % stride
                if col_b >= w * 3:
                    out[p] = 0
                    continue
                if ref_pixels is not None:
                    img_y = (abs_h - 1 - row) if h > 0 else row
                    x = col_b // 3
                    c_bgr = col_b % 3
                    c_rgb = 2 - c_bgr
                    out[p] = ref_pixels[(img_y * w + x) * 3 + c_rgb]
                else:
                    above = max((r for r in good_rows if r < row), default=None)
                    below = min((r for r in good_rows if r > row), default=None)
                    if above is not None and below is not None:
                        alpha = (row - above) / float(below - above)
                        va = out[data_off + above * stride + col_b]
                        vb = out[data_off + below * stride + col_b]
                        out[p] = max(0, min(255, round(va * (1.0 - alpha) + vb * alpha)))
                    elif above is not None:
                        out[p] = out[data_off + above * stride + col_b]
                    elif below is not None:
                        out[p] = out[data_off + below * stride + col_b]

        ok, st, _ = self.validate(bytes(out))
        return bytes(out), ("BMP_SCANLINE_REPAIRED" if ok else f"BMP_REPAIR_FAILED ({st})"), {
            "repaired_rows": len(bad_rows),
        }
