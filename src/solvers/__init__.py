"""Registry and tiered pipeline orchestrator for CD sector solver plugins."""

from src.core import rspc
from src.solvers.base import SectorSolverPlugin


class SolverRegistry:
    """Central registry allowing zero-friction addition of new sector recovery solvers."""

    _plugins: dict[str, type[SectorSolverPlugin]] = {}

    @classmethod
    def register(cls, solver_cls: type[SectorSolverPlugin]) -> type[SectorSolverPlugin]:
        cls._plugins[solver_cls.name] = solver_cls
        return solver_cls

    @classmethod
    def list_plugins(cls) -> list[SectorSolverPlugin]:
        return sorted((c() for c in cls._plugins.values()), key=lambda s: (s.tier, s.name))

    @classmethod
    def get_solvers(
        cls, max_tier: int = 3, names: list[str] | None = None
    ) -> list[SectorSolverPlugin]:
        all_solvers = cls.list_plugins()
        if names:
            wanted = set(names)
            return [s for s in all_solvers if s.name in wanted]
        return [s for s in all_solvers if s.tier <= max_tier]


class SolverPipeline:
    """Runs registered solver plugins in ascending tier order until mathematical verification."""

    def __init__(self, max_tier: int = 3, solver_names: list[str] | None = None):
        self.solvers = SolverRegistry.get_solvers(max_tier=max_tier, names=solver_names)

    def solve_sector(
        self,
        lba: int,
        raw_passes: list[tuple[bytes, bytes]],
        max_tier: int | None = None,
        context: dict | None = None,
    ) -> tuple[bool, int, int, bytearray, str]:
        if not raw_passes:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, 0, 0, sec, "NO_PASSES"

        best = (False, -1, -1, bytearray(2352), "NONE")
        for solver in self.solvers:
            if max_tier is not None and solver.tier > max_tier:
                continue
            ok, pz, qz, sec, strat = solver.solve(lba, raw_passes, context=context)
            if ok:
                return True, 86, 52, sec, strat
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, sec, strat)
        return best


# Register built-in Tier 1–3 solvers upon package import
from src.solvers import portfolio as _portfolio  # noqa: E402, F401
from src.solvers import deep_algebraic as _deep  # noqa: E402, F401

__all__ = ["SolverRegistry", "SolverPipeline", "SectorSolverPlugin"]
