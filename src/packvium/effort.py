from __future__ import annotations

from ._compat import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solvers import SearchStats


@dataclass(frozen=True, slots=True)
class EffortBudget:
    """Counted-work limits, checked alongside (never instead of) the wall clock.

    A wall clock is not a reproducible measure of work: scheduler, CPU frequency,
    background load, cache state and GC all move where it trips. Counting instead
    lets a caller who needs bit-identical results across machines and loads buy
    that -- as long as the wall-clock cutoff stays generous enough not to trip
    first, since it remains a live safety bound even when an effort budget is set.
    `max_restarts` bounds the portfolio's multi-start loop directly; the other three
    mirror `SearchStats` fields and are checked against one solver call's own count.
    """

    max_candidates_evaluated: int | None = None
    max_placement_attempts: int | None = None
    max_search_nodes: int | None = None
    max_restarts: int | None = None

    def exceeded(self, stats: "SearchStats") -> bool:
        return ((self.max_candidates_evaluated is not None and stats.candidates_evaluated >= self.max_candidates_evaluated)
                or (self.max_placement_attempts is not None and stats.placements_attempted >= self.max_placement_attempts)
                or (self.max_search_nodes is not None and stats.search_nodes_expanded >= self.max_search_nodes))
