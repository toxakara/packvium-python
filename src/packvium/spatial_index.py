"""Uniform-grid broad-phase index for axis-aligned box collision queries.

`find_candidates`'s inner loop tests a candidate box against every already-placed box
in the container -- `docs/ALGORITHMS-AND-COMPLEXITY.md` already flagged this as
`O(n*p*r*m)` and named "replace pairwise checks with a spatial index" as the seam.
`SpatialIndex` buckets boxes into a uniform grid so a query only has to exact-check the
boxes sharing at least one cell with it, not every box ever placed.

Safety property the whole design leans on: a grid cell assignment only needs to be a
superset of "boxes this query could possibly intersect" -- `query()` returns bound
*indices*, and the caller (unchanged from before this module existed) still runs the
exact 6-comparison AABB test on each one before trusting it. Over-inclusive buckets
only make the index slower, but under-inclusive buckets can miss collisions: the
exact check cannot recover an omitted candidate. Differential property tests in
`test_spatial_index.py` compare queries against a naive O(n) scan, including cached
queries after insertions and forks, to guard that completeness invariant.
"""

from __future__ import annotations

from itertools import chain, product
from typing import Iterable

Bound = tuple[int, int, int, int, int, int]
_QUERY_CACHE_LIMIT = 1024


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


class SpatialIndex:
    """Buckets axis-aligned box extents into a uniform 3D grid.

    Cell size is derived once from the container's own inner dimensions, sized so a
    typical container has roughly `cells_per_axis` cells along each axis -- coarse
    enough that a handful of items don't each get their own cell (which would just move
    the O(n) cost into building the index), fine enough that a container with hundreds
    of placements spreads them across many buckets instead of one.
    """

    __slots__ = ("cell_x", "cell_y", "cell_z", "cells", "_answers", "_inherited", "_changed")

    def __init__(self, length_ticks: int, width_ticks: int, height_ticks: int, cells_per_axis: int = 8):
        self.cell_x = max(1, _ceil_div(max(1, length_ticks), cells_per_axis))
        self.cell_y = max(1, _ceil_div(max(1, width_ticks), cells_per_axis))
        self.cell_z = max(1, _ceil_div(max(1, height_ticks), cells_per_axis))
        # Cell contents are tuples, never lists: a bucket is replaced on insert rather than
        # appended to, so a fork only has to copy the dict, and `query` can hand a bucket
        # straight back without exposing anything a caller could mutate.
        self.cells: dict[tuple[int, int, int], tuple[int, ...]] = {}
        # Query answers by cell range, valid for the current contents only. A candidate
        # scan asks about far more boxes than there are distinct cell ranges -- the grid
        # is coarse by design -- and the answer for a range is a pure function of the
        # contents. Disjoint insertions preserve answers; overlapping insertions and
        # the retention ceiling cause lazy recomputation.
        self._answers: dict[tuple[int, int, int, int, int, int], tuple[int, ...]] = {}
        # The answers valid just before the latest insertion, and the cell range that
        # insertion changed. An answer for a range disjoint from it is still exact, and
        # is adopted into `_answers` the first time it is asked for. Inheriting by
        # reference costs an insertion nothing, so a search branch that is discarded
        # without being queried never pays for a copy (see `add`).
        self._inherited: dict[tuple[int, int, int, int, int, int], tuple[int, ...]] | None = None
        self._changed: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)

    def copy(self) -> "SpatialIndex":
        """A structural fork: independent of the original from this point on.

        Search branches copy a `ContainerState` and then diverge, each adding its own
        placements -- sharing the buckets would let one branch's insert corrupt
        another's index, the same reason `ContainerState.copy()` already copies its own
        `bounds` list rather than aliasing it. Buckets are immutable, so a shallow copy
        of the dict is a complete fork.
        """
        clone = SpatialIndex.__new__(SpatialIndex)
        clone.cell_x, clone.cell_y, clone.cell_z = self.cell_x, self.cell_y, self.cell_z
        clone.cells = dict(self.cells)
        # Same contents, same answers. Safe to share: an insert on either side replaces
        # the dict on that side rather than mutating it, so the other keeps a valid one.
        clone._answers = self._answers
        clone._inherited, clone._changed = self._inherited, self._changed
        return clone

    def _cell_range(self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int):
        ix1, ix2 = x1 // self.cell_x, _ceil_div(max(x2, x1 + 1), self.cell_x)
        iy1, iy2 = y1 // self.cell_y, _ceil_div(max(y2, y1 + 1), self.cell_y)
        iz1, iz2 = z1 // self.cell_z, _ceil_div(max(z2, z1 + 1), self.cell_z)
        return ix1, ix2, iy1, iy2, iz1, iz2

    def add(self, index: int, bound: Bound) -> None:
        x1, y1, z1, x2, y2, z2 = bound
        ix1, ix2, iy1, iy2, iz1, iz2 = self._cell_range(x1, y1, z1, x2, y2, z2)
        cells = self.cells
        for ix in range(ix1, ix2):
            for iy in range(iy1, iy2):
                for iz in range(iz1, iz2):
                    key = (ix, iy, iz)
                    cells[key] = cells.get(key, ()) + (index,)
        # Forks share immutable answers, so this branch starts a cache of its own and
        # keeps the old one, as it stood for the contents before this insertion, to
        # inherit from. Only one generation is kept: the old dictionary links to nothing,
        # so live states hold at most their own cache and their parent's, and siblings
        # share that parent's. Copying the disjoint answers here instead made every
        # child pay for answers most children never ask for, and every live state carry
        # its own copy -- which is why that copy had to stop at 128 answers.
        self._inherited = self._answers
        self._changed = (ix1, ix2, iy1, iy2, iz1, iz2)
        self._answers = {}

    def query(self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> tuple[int, ...]:
        """Bound indices sharing at least one cell with the given box, each at most once.

        Cells are visited in `(x, y, z)` order and an index keeps its first position, so
        the sequence is a deterministic function of the index contents. Returned as an
        immutable tuple rather than a generator: this is the innermost call of the
        candidate scan. Buckets and cached answers can be shared safely across forks.
        """
        cell_x, cell_y, cell_z = self.cell_x, self.cell_y, self.cell_z
        # Conditional expressions rather than `max()`: three builtin calls per query were
        # 4% of a profiled `quality` solve.
        ix1, ix2 = x1 // cell_x, -(-(x2 if x2 > x1 else x1 + 1) // cell_x)
        iy1, iy2 = y1 // cell_y, -(-(y2 if y2 > y1 else y1 + 1) // cell_y)
        iz1, iz2 = z1 // cell_z, -(-(z2 if z2 > z1 else z1 + 1) // cell_z)
        if ix1 + 1 == ix2 and iy1 + 1 == iy2 and iz1 + 1 == iz2:
            return self.cells.get((ix1, iy1, iz1), ())
        cell_range = (ix1, ix2, iy1, iy2, iz1, iz2)
        answers = self._answers
        answer = answers.get(cell_range)
        if answer is not None: return answer
        inherited = self._inherited
        if inherited is not None and (answer := inherited.get(cell_range)) is not None:
            cx1, cx2, cy1, cy2, cz1, cz2 = self._changed
            if not (ix2 <= cx1 or cx2 <= ix1 or iy2 <= cy1 or cy2 <= iy1 or iz2 <= cz1 or cz2 <= iz1):
                answer = None
        if answer is None:
            answer = self._collect(ix1, ix2, iy1, iy2, iz1, iz2)
        if len(answers) >= _QUERY_CACHE_LIMIT:
            # Detach rather than clear: another search branch may share this cache.
            self._answers = answers = {}
        answers[cell_range] = answer
        return answer

    def _collect(self, ix1: int, ix2: int, iy1: int, iy2: int, iz1: int, iz2: int) -> tuple[int, ...]:
        # `product` yields the cells in the same (x, y, z) order as three nested loops,
        # without an interpreted loop body per cell.
        hits = [bucket for bucket in map(self.cells.get, product(range(ix1, ix2), range(iy1, iy2), range(iz1, iz2)))
                if bucket]
        if not hits: return ()
        if len(hits) == 1: return hits[0]
        return tuple(dict.fromkeys(chain.from_iterable(hits)))


def build(bounds: Iterable[Bound], length_ticks: int, width_ticks: int, height_ticks: int) -> SpatialIndex:
    index = SpatialIndex(length_ticks, width_ticks, height_ticks)
    for position, bound in enumerate(bounds):
        index.add(position, bound)
    return index
