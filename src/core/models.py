"""Core dataclasses shared across optical ripping, sector solving, and format repair."""

from dataclasses import dataclass, field


@dataclass
class DeviceCapabilities:
    """Hardware identity and limits discovered from an optical SG/SR device pair."""

    sg_device: str
    sr_device: str | None
    vendor_model: str
    supports_c2: bool
    max_speed_kbps: int
    cur_speed_kbps: int
    max_sectors_kb: int
    max_batch_sectors: int


@dataclass
class FileExtent:
    """ISO-9660 file or directory extent mapped onto CD LBAs and recovery priority."""

    path: str
    directory: str
    name: str
    ext: str
    start_lba: int
    end_lba: int
    size_bytes: int
    sectors: int
    is_dir: bool = False
    bad_lbas: list[int] = field(default_factory=list)
    priority: int = 50
    reference_path: str | None = None


@dataclass
class EtiologyReport:
    """Diagnostic classification of bad-sector physical geometry."""

    total_bad_sectors: int
    total_ranges: int
    run_length_distribution: dict[int, int]
    single_turn_pitch_samples: int
    avg_single_turn_pitch: float | None
    early_track_ratio: float
    classification: str
    notes: str


@dataclass
class SolvedSector:
    """Decoded or best-effort sector payload and RSPC verification status."""

    lba: int
    user_2048: bytes
    verified: bool
    best_p_ok: int
    best_q_ok: int
    edc_ok: bool
    solved_pass: int
    strategy: str
    known_mask_2048: set[int] | None = None


@dataclass
class FileRecoveryResult:
    """Per-file outcome after sector recovery, format repair, and extraction."""

    iso_path: str
    saved_path: str
    directory: str
    size_bytes: int
    bad_sectors_total: int
    bad_sectors_verified: int
    min_p_ok: int
    zero_holes_2k: int
    recovery_class: str
    format_status: str
    details: dict = field(default_factory=dict)
