"""Windows BMP format validation and scanline-level repair handler."""

import io
import struct
from typing import Callable

from PIL import Image

from src.formats import FormatRegistry
from src.formats.base import FileFormatHandler
from src.core.models import FileExtent


def validate_bmp_structure(data: bytes) -> tuple[bool, list[str], dict]:
    """Strictly verify BITMAPFILEHEADER, BITMAPINFOHEADER, and pixel buffer bounds."""
    issues: list[str] = []
    if len(data) < 54 or data[0:2] != b"BM":
        return False, ["bad_magic_or_short_header"], {"bmp_struct_valid": False}

    bf_size, r1, r2, bf_off = struct.unpack_from("<IHHI", data, 2)
    bi_size, w, h, planes, bpp, comp, img_sz = struct.unpack_from("<IiiHHII", data, 14)

    if r1 != 0 or r2 != 0:
        issues.append(f"nonzero_reserved:{r1},{r2}")
    if bf_size != len(data):
        issues.append(f"bfSize_mismatch:{bf_size}!={len(data)}")
    if bi_size not in (40, 52, 56, 108, 124):
        issues.append(f"unsupported_biSize:{bi_size}")
    if bf_off < 14 + bi_size:
        issues.append(f"bad_bfOffBits:{bf_off}<{14 + bi_size}")
    if w <= 0 or h == 0:
        issues.append(f"bad_dims:{w}x{h}")
    if planes != 1:
        issues.append(f"bad_planes:{planes}")
    if bpp not in (1, 4, 8, 16, 24, 32):
        issues.append(f"bad_bpp:{bpp}")

    if comp == 0 and w > 0 and h != 0 and bpp in (1, 4, 8, 16, 24, 32):
        stride = ((w * bpp + 31) // 32) * 4
        expected_img = stride * abs(h)
        if len(data) < bf_off + expected_img:
            issues.append(f"truncated_pixels:{len(data)}<{bf_off + expected_img}")
        if img_sz not in (0, expected_img):
            issues.append(f"biSizeImage_mismatch:{img_sz}!={expected_img}")

    meta = {
        "bmp_struct_valid": len(issues) == 0,
        "width": w,
        "height": h,
        "bpp": bpp,
        "compression": comp,
    }
    return len(issues) == 0, issues, meta


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
        bmp_ok, bmp_issues, bmp_meta = validate_bmp_structure(data)
        if not bmp_ok:
            return False, f"BMP_STRUCT_ERR ({'; '.join(bmp_issues[:3])})", bmp_meta
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im2:
                im2.load()
                return True, "STRICT_OK", {"dims": f"{im2.size[0]}x{im2.size[1]} {im2.mode}", **bmp_meta}
        except Exception as exc:
            return False, f"DECODE_ERR ({type(exc).__name__}: {exc})", bmp_meta

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
