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

from typing import Iterable, Iterator

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

    __slots__ = ("cell_x", "cell_y", "cell_z", "cells")

    def __init__(self, length_ticks: int, width_ticks: int, height_ticks: int, cells_per_axis: int = 8):
        self.cell_x = max(1, _ceil_div(max(1, length_ticks), cells_per_axis))
        self.cell_y = max(1, _ceil_div(max(1, width_ticks), cells_per_axis))
        self.cell_z = max(1, _ceil_div(max(1, height_ticks), cells_per_axis))
        self.cells: dict[tuple[int, int, int], list[int]] = {}

    def copy(self) -> "SpatialIndex":
        """A structural fork: independent of the original from this point on.

        Search branches copy a `ContainerState` and then diverge, each adding its own
        placements -- sharing the cell lists would let one branch's insert corrupt
        another's index, the same reason `ContainerState.copy()` already copies its own
        `bounds` list rather than aliasing it.
        """
        clone = SpatialIndex.__new__(SpatialIndex)
        clone.cell_x, clone.cell_y, clone.cell_z = self.cell_x, self.cell_y, self.cell_z
        clone.cells = {key: list(indices) for key, indices in self.cells.items()}
        return clone

    def _cell_range(self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int):
        ix1, ix2 = x1 // self.cell_x, _ceil_div(max(x2, x1 + 1), self.cell_x)
        iy1, iy2 = y1 // self.cell_y, _ceil_div(max(y2, y1 + 1), self.cell_y)
        iz1, iz2 = z1 // self.cell_z, _ceil_div(max(z2, z1 + 1), self.cell_z)
        return ix1, ix2, iy1, iy2, iz1, iz2

    def add(self, index: int, bound: Bound) -> None:
        x1, y1, z1, x2, y2, z2 = bound
        ix1, ix2, iy1, iy2, iz1, iz2 = self._cell_range(x1, y1, z1, x2, y2, z2)
        for ix in range(ix1, ix2):
            for iy in range(iy1, iy2):
                for iz in range(iz1, iz2):
                    self.cells.setdefault((ix, iy, iz), []).append(index)

    def query(self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> Iterator[int]:
        """Bound indices sharing at least one cell with the given box, each at most once."""
        ix1, ix2, iy1, iy2, iz1, iz2 = self._cell_range(x1, y1, z1, x2, y2, z2)
        seen: set[int] = set()
        for ix in range(ix1, ix2):
            for iy in range(iy1, iy2):
                for iz in range(iz1, iz2):
                    for index in self.cells.get((ix, iy, iz), ()):
                        if index not in seen:
                            seen.add(index)
                            yield index


def build(bounds: Iterable[Bound], length_ticks: int, width_ticks: int, height_ticks: int) -> SpatialIndex:
    index = SpatialIndex(length_ticks, width_ticks, height_ticks)
    for position, bound in enumerate(bounds):
        index.add(position, bound)
    return index
