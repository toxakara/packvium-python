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

    __slots__ = ("_supporters", "_children", "_boxes", "_cell", "_by_top", "_by_bottom",
                 "_top_indexes", "_bottom_indexes")

    def __init__(self, boxes: Sequence[AxisAlignedBox], cell_hint: int = 1):
        """`cell_hint` is an upper bound on the footprint of any box that may later be
        appended with `with_box`.

        Without it the cell is sized from the boxes present now, and appending anything
        wider has to fall back to a full rebuild -- which is correct but defeats the
        point, because in a search the base is what is already placed and the candidate
        is a *new* item that may well be the widest thing in the request. A caller that
        knows the item set passes its widest footprint once and the delta path then
        always applies. Too large a hint only makes each bucket coarser; too small a one
        is impossible to get wrong, because the fallback covers it.
        """
        boxes = tuple(boxes)
        by_top: dict[int, list[tuple[int, AxisAlignedBox]]] = {}
        by_bottom: dict[int, list[tuple[int, AxisAlignedBox]]] = {}
        for index, box in enumerate(boxes):
            by_top.setdefault(box.z2, []).append((index, box))
            by_bottom.setdefault(box.origin.z, []).append((index, box))
        supporters: list[list[ContactEdge]] = [[] for _ in boxes]
        children: list[list[int]] = [[] for _ in boxes]
        # A single global cell size, not one derived per level from that level's own
        # candidates: a querying box can be any size in this scene, and `_LevelIndex`
        # is only correct when its cell is at least as large as every box it will ever
        # index or be queried with.
        cell = max(
            (max(box.x2 - box.origin.x, box.y2 - box.origin.y) for box in boxes),
            default=1,
        )
        cell = max(cell, cell_hint)
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
        self._boxes = boxes
        self._cell = cell
        self._by_top = by_top
        self._by_bottom = by_bottom
        # Only the top-plane indexes are populated by the build above; the bottom-plane
        # ones are built on demand, because a from-scratch build never needs them and
        # paying for them here would slow the common path to speed up the incremental one.
        self._top_indexes = indexes
        self._bottom_indexes: dict[int, _LevelIndex] = {}

    def with_box(self, box: AxisAlignedBox) -> "ContactGraph":
        """This graph plus one more box, appended at the next index.

        Adding a box cannot create or destroy contact between two boxes that were
        already here: contact is a pairwise geometric predicate over two boxes and
        nothing else. That is the whole reason a delta is sound, and it is why this
        returns a new graph that shares the base's edge tuples instead of recomputing
        them -- only the new box's own two planes are queried.

        The result is required to be identical to `ContactGraph(list(boxes) + [box])`,
        not merely equivalent: `top_loads` splits a conserved integer across the
        supporter tuple and hands the rounding remainder to its last edge, so edge
        *order* is contract, not presentation. Appending the new box's index keeps every
        existing tuple ascending because the new index is the largest one.
        """
        index = len(self._boxes)
        footprint = max(box.x2 - box.origin.x, box.y2 - box.origin.y)
        if footprint > self._cell:
            # `_LevelIndex` is only correct while its cell is at least as large as every
            # box indexed in or queried against it. A larger box could step over cells
            # in the middle of its own footprint and miss a real overlap, so this is a
            # correctness fallback, not an optimisation choice.
            return ContactGraph(self._boxes + (box,), cell_hint=footprint)

        supporters = list(self._supporters)
        children = list(self._children)

        # What the new box rests on: boxes whose top plane is its bottom plane.
        own_supporters: list[ContactEdge] = []
        for other_index, area in sorted(
            (other_index, other.overlap_area_xy(box))
            for other_index, other in self._near(self._by_top, self._top_indexes, box.origin.z, box)
        ):
            if area > 0:
                own_supporters.append(ContactEdge(other_index, area))
                children[other_index] = children[other_index] + (index,)

        # What now rests on it: boxes whose bottom plane is its top plane. Their
        # supporter tuples gain the new index, which is larger than every index already
        # in them, so ascending order is preserved by appending.
        own_children: list[int] = []
        for other_index, area in sorted(
            (other_index, other.overlap_area_xy(box))
            for other_index, other in self._near(self._by_bottom, self._bottom_indexes, box.z2, box)
        ):
            if area > 0:
                own_children.append(other_index)
                supporters[other_index] = supporters[other_index] + (ContactEdge(index, area),)

        supporters.append(tuple(own_supporters))
        children.append(tuple(own_children))

        # The by-plane buckets are carried forward rather than rederived: one box joins
        # exactly two planes, so copying the outer dict (one entry per distinct plane,
        # not per box) and rewriting those two buckets is all that changed. Rebuilding
        # both dicts from `boxes` would put an O(n) dict-insert pass on a path whose
        # whole purpose is to avoid touching the boxes that did not move.
        by_top = dict(self._by_top)
        by_top[box.z2] = by_top.get(box.z2, []) + [(index, box)]
        by_bottom = dict(self._by_bottom)
        by_bottom[box.origin.z] = by_bottom.get(box.origin.z, []) + [(index, box)]

        # `_LevelIndex` is immutable once built, so every cached one may be shared with
        # the base -- except on the two planes whose bucket just gained a member, where
        # the cached index no longer describes its bucket and must be rebuilt on demand.
        top_indexes = dict(self._top_indexes)
        top_indexes.pop(box.z2, None)
        bottom_indexes = dict(self._bottom_indexes)
        bottom_indexes.pop(box.origin.z, None)

        return ContactGraph._from_parts(
            self._boxes + (box,), self._cell, tuple(supporters), tuple(children),
            by_top, by_bottom, top_indexes, bottom_indexes)

    @classmethod
    def _from_parts(cls, boxes, cell, supporters, children,
                    by_top, by_bottom, top_indexes, bottom_indexes) -> "ContactGraph":
        graph = cls.__new__(cls)
        graph._boxes = boxes
        graph._cell = cell
        graph._supporters = supporters
        graph._children = children
        graph._by_top = by_top
        graph._by_bottom = by_bottom
        graph._top_indexes = top_indexes
        graph._bottom_indexes = bottom_indexes
        return graph

    def _near(self, buckets, cache, plane: int, box: AxisAlignedBox):
        """Every box on `plane` that could overlap `box` in XY, deduped, via the hash."""
        entries = buckets.get(plane)
        if not entries:
            return ()
        level = cache.get(plane)
        if level is None:
            level = _LevelIndex(entries, self._cell)
            cache[plane] = level
        return level.near(box)

    def supporters(self, index: int) -> tuple[ContactEdge, ...]:
        """What `index` directly rests on, each with the contact area."""
        return self._supporters[index]

    def children(self, index: int) -> tuple[int, ...]:
        """What rests directly on `index` (not transitively)."""
        return self._children[index]
