"""Abstract plugin interface for CD-ROM Mode 1 sector recovery algorithms."""

from abc import ABC, abstractmethod


class SectorSolverPlugin(ABC):
    """Base class for pluggable raw+C2 RSPC/EDC sector recovery strategies.

    Subclasses register with `@SolverRegistry.register` and implement `solve()`.
    Tiers control execution order:
      - tier 1: fast single/multi-pass C2 & subgrid peeling
      - tier 2: cleaned consensus & anchor-graft heuristics
      - tier 3: deep algebraic / subset sampling / GF(2) linear solvers
    """

    name: str = "base"
    tier: int = 1
    priority: int = 50
    description: str = ""

    @abstractmethod
    def solve(
        self,
        lba: int,
        raw_passes: list[tuple[bytes, bytes]],
        context: dict | None = None,
    ) -> tuple[bool, int, int, bytearray, str]:
        """Attempt to recover and mathematically verify a sector.

        Returns `(verified, p_ok_count, q_ok_count, raw_2352_or_2064, strategy_label)`.
        """
