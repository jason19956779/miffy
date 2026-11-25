"""Abstract plugin interface for format-specific validation, reference matching, and repair."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from src.core.models import FileExtent


class FileFormatHandler(ABC):
    """Base plugin class for handling recovered files by extension or magic bytes."""

    name: str = "base"
    extensions: tuple[str, ...] = ()
    priority: int = 50
    description: str = ""

    def can_handle(self, ext: str, data: bytes | None = None) -> bool:
        return ext.upper() in self.extensions

    def find_reference(
        self,
        extent: FileExtent,
        all_extents: dict[str, FileExtent],
        read_file_fn: Callable[[FileExtent], bytes],
        file_bytes: bytes | None = None,
    ) -> tuple[bytes | None, str | None]:
        """Locate an intact companion reference asset (e.g., thumbnail) if applicable."""
        return None, None

    @abstractmethod
    def validate(self, data: bytes) -> tuple[bool, str, dict]:
        """Validate recovered file bytes and return `(is_valid, status_label, metadata)`."""

    def validate_file(self, path: Path) -> tuple[bool, str, dict]:
        """Validate recovered file on disk (streaming override optional for large files)."""
        return self.validate(path.read_bytes())

    def reconstruct(
        self,
        data: bytes,
        bad_intervals: list[tuple[int, int]],
        reference_data: bytes | None = None,
        context: dict | None = None,
    ) -> tuple[bytes, str, dict]:
        """Optional format-specific reconstruction for remaining unverified byte intervals.

        Default implementation leaves `data` unchanged.
        """
        return data, "NO_REPAIR_APPLIED", {}
