"""Format plugin registry mapping extensions and magic signatures to handlers."""

from src.formats.base import FileFormatHandler


class FormatRegistry:
    """Central registry for pluggable file-format validators and reconstructors."""

    _handlers: dict[str, type[FileFormatHandler]] = {}

    @classmethod
    def register(cls, handler_cls: type[FileFormatHandler]) -> type[FileFormatHandler]:
        cls._handlers[handler_cls.name] = handler_cls
        return handler_cls

    @classmethod
    def list_handlers(cls) -> list[FileFormatHandler]:
        return sorted((c() for c in cls._handlers.values()), key=lambda h: (h.priority, h.name))

    @classmethod
    def get_handler(cls, ext: str, data: bytes | None = None) -> FileFormatHandler:
        handlers = cls.list_handlers()
        for h in handlers:
            if "*" not in h.extensions and h.can_handle(ext, data):
                return h
        for h in handlers:
            if h.can_handle(ext, data):
                return h
        raise KeyError(f"No handler registered for extension {ext!r}")


# Auto-register all built-in handlers upon package import
from src.formats import jpeg_handler as _jpeg  # noqa: E402, F401
from src.formats import bmp_handler as _bmp  # noqa: E402, F401
from src.formats import image_generic as _img  # noqa: E402, F401
from src.formats import media_handlers as _media  # noqa: E402, F401
from src.formats import binary_handlers as _bin  # noqa: E402, F401

__all__ = ["FormatRegistry", "FileFormatHandler"]
