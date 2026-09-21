"""Uniform-grid broad-phase index for axis-aligned box collision queries.

`find_candidates`'s inner loop tests a candidate box against every already-placed box
in the container -- `docs/ALGORITHMS-AND-COMPLEXITY.md` already flagged this as
`O(n*p*r*m)` and named "replace pairwise checks with a spatial index" as the seam.
`SpatialIndex` buckets boxes into a uniform grid so a query only has to exact-check the
boxes sharing at least one cell with it, not every box ever placed.

Safety property the whole design leans on: a grid cell assignment only needs to be a
superset of "boxes this query could possibly intersect" -- `query()` returns bound
*indices*, and the caller (unchanged from before this module existed) still runs the
exact 6-comparison AABB test on each one before trusting it. A bug here can only make
the index slower (over-inclusive buckets) or, if under-inclusive, is caught immediately
by the differential property tests in `test_spatial_index.py`, which compare every
query against a naive O(n) scan across many random configurations. It can never by
itself cause a false "no collision" silently, because the exact check is still the
final word.
"""

from __future__ import annotations

from itertools import chain
from typing import Iterable, Sequence

Bound = tuple[int, int, int, int, int, int]


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

    __slots__ = ("cell_x", "cell_y", "cell_z", "cells", "_answers")

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
        # contents, so it is computed once per range per generation of the index.
        self._answers: dict[tuple[int, int, int, int, int, int], Sequence[int]] = {}

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
        self._answers = {}

    def query(self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> Sequence[int]:
        """Bound indices sharing at least one cell with the given box, each at most once.

        Cells are visited in `(x, y, z)` order and an index keeps its first position, so
        the sequence is a deterministic function of the index contents. Returned as a
        sequence rather than a generator: this is the innermost call of the candidate
        scan, and a bucket can be handed back as-is when the box touches only one.
        """
        cell_x, cell_y, cell_z = self.cell_x, self.cell_y, self.cell_z
        ix1, ix2 = x1 // cell_x, -(-max(x2, x1 + 1) // cell_x)
        iy1, iy2 = y1 // cell_y, -(-max(y2, y1 + 1) // cell_y)
        iz1, iz2 = z1 // cell_z, -(-max(z2, z1 + 1) // cell_z)
        cell_range = (ix1, ix2, iy1, iy2, iz1, iz2)
        answers = self._answers
        answer = answers.get(cell_range)
        if answer is not None: return answer
        answers[cell_range] = answer = self._collect(ix1, ix2, iy1, iy2, iz1, iz2)
        return answer

    def _collect(self, ix1: int, ix2: int, iy1: int, iy2: int, iz1: int, iz2: int) -> Sequence[int]:
        cells = self.cells
        hits: list[tuple[int, ...]] = []
        for ix in range(ix1, ix2):
            for iy in range(iy1, iy2):
                for iz in range(iz1, iz2):
                    bucket = cells.get((ix, iy, iz))
                    if bucket: hits.append(bucket)
        if not hits: return ()
        if len(hits) == 1: return hits[0]
        return list(dict.fromkeys(chain.from_iterable(hits)))


def build(bounds: Iterable[Bound], length_ticks: int, width_ticks: int, height_ticks: int) -> SpatialIndex:
    index = SpatialIndex(length_ticks, width_ticks, height_ticks)
    for position, bound in enumerate(bounds):
        index.add(position, bound)
    return index
