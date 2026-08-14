from __future__ import annotations

from ._compat import dataclass
from typing import Iterator, Sequence

from .geometry import AxisAlignedBox


@dataclass(frozen=True, slots=True)
class ContactEdge:
    index: int
    area: int


class _LevelIndex:
    """A uniform spatial hash over one z-level's boxes (those sharing one `by_top` or
    `origin.z` bucket), bucketed by XY cell so a query only scans the boxes actually
    near it instead of every other box at that level.

    A regular lattice (`GridSolver`) can put thousands of items on one shared level,
    and the plain "for each box, scan every other box at this level" this replaces was
    exactly quadratic in that case -- fine for the few-dozen-item scenes this module
    was written against, but O(level_size^2) is what actually made a 10,000-item
    request take longer than its own configured time limit.

    `cell` must be at least as large as the largest footprint dimension of *every* box
    that will ever be inserted into or queried against this index -- not just the
    candidates being indexed. Only then is a box guaranteed to span no more than a 2x2
    block of cells, which is what guarantees two overlapping boxes always share at
    least one cell. Sizing `cell` from the indexed candidates alone was exactly wrong:
    a querying box larger than that could skip cells in the middle of its own
    footprint and silently miss a real overlap.
    """

    __slots__ = ("_cell", "_buckets")

    def __init__(self, candidates: list[tuple[int, AxisAlignedBox]], cell: int):
        self._cell = cell
        self._buckets: dict[tuple[int, int], list[tuple[int, AxisAlignedBox]]] = {}
        for entry in candidates:
            for cell in self._cells(entry[1]):
                self._buckets.setdefault(cell, []).append(entry)

    def _cells(self, box: AxisAlignedBox) -> set[tuple[int, int]]:
        cell = self._cell
        return {
            (box.origin.x // cell, box.origin.y // cell),
            ((box.x2 - 1) // cell, box.origin.y // cell),
            (box.origin.x // cell, (box.y2 - 1) // cell),
            ((box.x2 - 1) // cell, (box.y2 - 1) // cell),
        }

    def near(self, box: AxisAlignedBox) -> Iterator[tuple[int, AxisAlignedBox]]:
        """Every candidate that could possibly overlap `box` in XY -- a superset of
        the true overlaps, not an exact answer; the caller still checks
        `overlap_area_xy` itself. May yield the same candidate more than once when its
        footprint spans more than one of `box`'s cells; the caller's `> 0` area check
        and edge-list append are idempotent-safe against that only if it also dedupes,
        so this dedupes here instead, once, rather than trusting every caller to."""
        seen: set[int] = set()
        for cell in self._cells(box):
            for other_index, other in self._buckets.get(cell, ()):
                if other_index not in seen:
                    seen.add(other_index)
                    yield other_index, other


class ContactGraph:
    """Direct top/bottom contact between boxes, derived from geometry alone.

    Built once per whole-container check and shared by every load-propagation
    computation that used to each re-derive their own by-plane index or nested
    pairwise scan (`top_loads`, `stacked_counts`) -- the two disagreeing on
    what "touches" means was a standing risk neither test caught, because nothing
    forced them to share the definition.
    """

    __slots__ = ("_supporters", "_children")

    def __init__(self, boxes: Sequence[AxisAlignedBox]):
        by_top: dict[int, list[tuple[int, AxisAlignedBox]]] = {}
        for index, box in enumerate(boxes):
            by_top.setdefault(box.z2, []).append((index, box))
        supporters: list[list[ContactEdge]] = [[] for _ in boxes]
        children: list[list[int]] = [[] for _ in boxes]
        # A single global cell size, not one derived per level from that level's own
        # candidates: a querying box can be any size in this scene, and `_LevelIndex`
        # is only correct when its cell is at least as large as every box it will ever
        # index or be queried with.
        cell = max((max(box.x2 - box.origin.x, box.y2 - box.origin.y) for box in boxes), default=1)
        indexes: dict[int, _LevelIndex] = {}
        for index, box in enumerate(boxes):
            candidates = by_top.get(box.origin.z)
            if not candidates:
                continue
            level = indexes.get(box.origin.z)
            if level is None:
                level = _LevelIndex(candidates, cell)
                indexes[box.origin.z] = level
            # `top_loads` (constraints.py) splits a conserved integer total across
            # `supporters(index)` and hands the rounding remainder to whichever edge
            # is *last* in that tuple -- so this must land in the same ascending
            # other_index order the original all-pairs scan produced (it visited
            # `by_top`'s bucket, itself built by a single increasing-index pass), not
            # whatever order the spatial hash's cells happen to iterate in.
            matches = sorted(
                (other_index, other.overlap_area_xy(box))
                for other_index, other in level.near(box)
                if other_index != index
            )
            for other_index, area in matches:
                if area > 0:
                    supporters[index].append(ContactEdge(other_index, area))
                    children[other_index].append(index)
        self._supporters = tuple(tuple(s) for s in supporters)
        self._children = tuple(tuple(c) for c in children)

    def supporters(self, index: int) -> tuple[ContactEdge, ...]:
        """What `index` directly rests on, each with the contact area."""
        return self._supporters[index]

    def children(self, index: int) -> tuple[int, ...]:
        """What rests directly on `index` (not transitively)."""
        return self._children[index]
