"""Registry and tiered pipeline orchestrator for CD sector solver plugins."""

import time

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
        return sorted(
            (c() for c in cls._plugins.values()),
            key=lambda s: (s.tier, getattr(s, "priority", 50), s.name),
        )

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

        filtered_passes = []
        slips_filtered = 0
        for raw_sec, c2_b in raw_passes:
            if rspc.is_frame_slip(lba, raw_sec):
                slips_filtered += 1
            else:
                filtered_passes.append((raw_sec, c2_b))

        ctx = context if context is not None else {}
        ctx["frame_slips_filtered"] = slips_filtered

        if not filtered_passes:
            sec = bytearray(2352)
            rspc.init_sector_header(lba, sec)
            return False, 0, 0, sec, "ALL_FRAME_SLIPS"

        if len(filtered_passes) >= 2 and "pass_cache" not in ctx:
            ctx["pass_cache"] = rspc.build_pass_cache(lba, filtered_passes)

        trace = ctx.get("debug_trace")
        best = (False, -1, -1, bytearray(2352), "NONE")
        for solver in self.solvers:
            if max_tier is not None and solver.tier > max_tier:
                continue
            t0 = time.perf_counter()
            ok, pz, qz, sec, strat = solver.solve(lba, filtered_passes, context=ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if ok and not rspc.check_and_fix_q_parity(sec, expected_lba=lba):
                ok = False
                pz, qz = rspc.count_pq(sec)
                strat = f"REJECTED_{strat}"
            if ok:
                c2_list = (
                    [e["c2"] for e in ctx["pass_cache"]]
                    if "pass_cache" in ctx and len(ctx["pass_cache"]) == len(filtered_passes)
                    else None
                )
                q_phys_ok, q_cm, q_cc = rspc.verify_physical_q_parity(sec, filtered_passes, c2_sets=c2_list)
                ctx["physical_q_match"] = (q_cm, q_cc)
                if not q_phys_ok:
                    ok = False
                    pz, qz = rspc.count_pq(sec)
                    strat = f"REJECTED_QPAR_{strat}"
            if isinstance(trace, list):
                trace.append(
                    {
                        "solver": solver.name,
                        "tier": solver.tier,
                        "ok": bool(ok),
                        "p_ok": pz,
                        "q_ok": qz,
                        "strategy": strat,
                        "elapsed_ms": round(elapsed_ms, 2),
                    }
                )
            if ok:
                return True, 86, 52, sec, strat
            if rspc.partial_score(pz, qz) > rspc.partial_score(best[1], best[2]):
                best = (False, pz, qz, sec, strat)
        return best


# Register built-in Tier 1–3 solvers upon package import
from src.solvers import portfolio as _portfolio  # noqa: E402, F401
from src.solvers import deep_algebraic as _deep  # noqa: E402, F401

__all__ = ["SolverRegistry", "SolverPipeline", "SectorSolverPlugin"]
