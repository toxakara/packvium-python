from __future__ import annotations

from ._compat import dataclass
from enum import Enum

from .effort import EffortBudget
from .units import Length


class SolverProfile(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    EXACT_SMALL = "exact_small"


@dataclass(frozen=True, slots=True)
class PackingConfig:
    profile: SolverProfile = SolverProfile.BALANCED
    time_limit_ms: int = 1000
    top_k: int = 3
    seed: int = 42
    max_containers: int | None = None
    clearance: Length = Length(0)
    minimum_support_ratio: float = 0.0
    exact_item_limit: int = 7
    multi_start_orders: int = 8
    validate_result: bool = True
    max_candidates_per_item: int = 1
    max_candidate_points: int = 4096
    solvers: tuple[str, ...] = ()
    objective: str = "default"
    effort_budget: EffortBudget | None = None
    #: A carrier's published dimensional-weight divisor. Required only when
    #: `objective` is `"shipping_cost"` -- see `extensions.ShippingCostSolutionScorer`.
    dimensional_weight_divisor: int | None = None
    dimensional_weight_length_unit: str = "in"
    dimensional_weight_weight_unit: str = "lb"
    # Default True reproduces every existing result byte-for-byte. False lets
    # GridSolver's regular-lattice fast path skip materializing one `Placement`
    # object per instance -- see `lattice_summary.py` -- when a caller
    # only needs to know the packing fits, not each item's own coordinates.
    require_placement_coordinates: bool = True
    #: Opt-in worker-process ceiling for the multi-start portfolio. `1`
    #: (the default) reproduces the original fully sequential orchestration
    #: exactly, with zero behavioural change. A value above `1` lets
    #: `SolverOrchestrator` run starts after the first concurrently in separate
    #: processes -- see docs/ALGORITHMS-AND-COMPLEXITY.md's "Multi-start
    #: orchestration" section for the eligibility rules and determinism argument.
    parallel_starts: int = 1
    #: Width of the deterministic beam over partial container plans.  The default
    #: keeps the historical greedy path; ``quality()`` opts into the bounded search.
    container_plan_beam_width: int = 1
    #: Hard counted-work ceiling for container-plan nodes, independent of wall time.
    container_plan_node_limit: int = 1
    #: The container walls an item may be unloaded through, for the
    #: stop-accessibility constraint. Empty (the default) disables the check entirely and
    #: reproduces every existing result byte-for-byte.
    #:
    #: Programmatic only, and deliberately so: the request schema has no access-directions
    #: field yet (docs/STOP-ACCESSIBILITY.md files it as deferred until a contract freeze),
    #: and defaulting to all six walls would enforce a rule true of no real vehicle. So a
    #: caller who wants the check states the doors in code, the same non-request path
    #: `safe_route_removal_order` is driven through today.
    access_directions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.time_limit_ms <= 0 or self.top_k <= 0 or self.exact_item_limit <= 0
                or self.multi_start_orders <= 0 or self.max_candidates_per_item <= 0
                or self.parallel_starts <= 0 or self.container_plan_beam_width <= 0
                or self.container_plan_node_limit <= 0):
            raise ValueError("positive configuration values required")
        if self.max_candidate_points < 16:
            raise ValueError("max_candidate_points must be at least 16")
        if not 0 <= self.minimum_support_ratio <= 1: raise ValueError("minimum_support_ratio must be between 0 and 1")
        if self.dimensional_weight_divisor is not None and self.dimensional_weight_divisor <= 0:
            raise ValueError("dimensional_weight_divisor must be positive")

    @classmethod
    def fast(cls, time_limit_ms: int = 200, **kwargs) -> "PackingConfig":
        return cls(profile=SolverProfile.FAST, time_limit_ms=time_limit_ms, top_k=1, multi_start_orders=1, **kwargs)

    @classmethod
    def balanced(cls, time_limit_ms: int = 1000, top_k: int = 3, **kwargs) -> "PackingConfig":
        return cls(profile=SolverProfile.BALANCED, time_limit_ms=time_limit_ms, top_k=top_k, **kwargs)

    @classmethod
    def quality(cls, time_limit_ms: int = 5000, top_k: int = 5, **kwargs) -> "PackingConfig":
        kwargs.setdefault("container_plan_beam_width", 16)
        kwargs.setdefault("container_plan_node_limit", 100_000)
        return cls(profile=SolverProfile.QUALITY, time_limit_ms=time_limit_ms, top_k=top_k, multi_start_orders=24, max_candidates_per_item=16, **kwargs)

    @classmethod
    def exact_small(cls, time_limit_ms: int = 10000, **kwargs) -> "PackingConfig":
        return cls(profile=SolverProfile.EXACT_SMALL, time_limit_ms=time_limit_ms, **kwargs)
