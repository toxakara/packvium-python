from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from . import hull
from .compression import applied_pressure, effective_height_ticks
from .geometry import ShapeType

if TYPE_CHECKING:
    from .models import Placement


def is_valid_nesting(a: "Placement", b: "Placement") -> bool:
    """Whether `a` and `b` are exactly the nested-column relationship
    `Item.nesting_height` describes: the same item type, the same footprint, one
    sunk into the other by precisely its declared nesting height --
    narrow and exact, not a blanket "same item type never collides" bypass.
    """
    if a.instance.item.id != b.instance.item.id:
        return False
    nesting = a.instance.item.nesting_height
    if nesting is None:
        return False
    box_a, box_b = a.envelope_box, b.envelope_box
    if (box_a.origin.x, box_a.origin.y, box_a.x2, box_a.y2) != (box_b.origin.x, box_b.origin.y, box_b.x2, box_b.y2):
        return False
    low, high = (box_a, box_b) if box_a.origin.z <= box_b.origin.z else (box_b, box_a)
    if low.origin.z == high.origin.z:
        return False
    return low.z2 - high.origin.z == nesting.ticks


def _nesting_overlap_volume(placements: Sequence["Placement"]) -> int:
    """Total double-counted volume between adjacent nested layers.

    A naive `sum(p.dimensions.volume for p in placements)` overstates how much
    space a nested column actually fills, since each pair of nested neighbours
    shares `nesting_height * base_area` of the same physical space. Grouped by
    (item, footprint) and checked only against the immediate z-neighbour -- O(n log
    n), not the O(n^2) an all-pairs scan would cost -- since a valid nest can only
    ever involve two immediately adjacent layers of an identical footprint.
    """
    groups: dict[tuple, list["Placement"]] = {}
    for p in placements:
        if p.instance.item.nesting_height is None:
            continue
        box = p.envelope_box
        key = (p.instance.item.id, box.origin.x, box.origin.y, box.x2, box.y2)
        groups.setdefault(key, []).append(p)
    overlap = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda p: p.envelope_box.origin.z)
        for lower, upper in zip(ordered, ordered[1:]):
            if is_valid_nesting(lower, upper):
                overlap += lower.instance.item.nesting_height.ticks * lower.envelope_dimensions.base_area
    return overlap


def occupied_volume(placement: "Placement") -> int:
    """How much space a placement actually takes, which is its box only if it is one.

    A `convex_hull` item occupies its hull. Counting its bounding box instead is not a
    conservative approximation of utilisation, it is a wrong number: two interlocking wedges
    filling one crate would report it 200% full, and the independent validator refuses that
    before any caller sees it.
    """
    item = placement.instance.item
    if item.shape_type is ShapeType.CONVEX_HULL:
        # Collision may conservatively fall back to the envelope for route-bound items or a
        # clearance margin. That does not turn the physical item into a box: utilisation and
        # reserve accounting always use the authored solid.
        assert item.hull_vertices is not None
        return hull.shape_for(item.hull_vertices, placement.rotation.value).volume
    if item.max_compression_pressure_kpa is None:
        return placement.dimensions.volume
    # Compression follows the item's rotated height axis, and `dimensions` is already rotated.
    # The load is the one the placement reports, so this needs no support graph and stays in
    # this layer; `top_load` is resolved before a result is ever built.
    dimensions = placement.dimensions
    footprint = dimensions.length.ticks * dimensions.width.ticks
    pressure = applied_pressure(placement.top_load, footprint)
    if pressure.exceeds_kpa(item.max_compression_pressure_kpa):
        # A crushed item has no meaningful occupied volume, and the arrangement is already
        # invalid -- `crushed` refuses it in the solver and the validator reports it. Reporting
        # the uncompressed figure here keeps that a reported issue rather than an exception
        # thrown out of a volume property.
        return dimensions.volume
    return footprint * effective_height_ticks(
        dimensions.height.ticks, item.compression_ratio_ppm,
        item.max_compression_pressure_kpa, pressure,
    )


def used_volume(placements: Sequence["Placement"]) -> int:
    """Physical volume actually occupied by `placements`, nesting overlap removed."""
    return sum(occupied_volume(p) for p in placements) - _nesting_overlap_volume(placements)


def used_volume_delta(placements: Sequence["Placement"], placement: "Placement") -> int:
    """Exact volume added by appending ``placement`` in O(n) time and O(1) space.

    Existing overlap pairs do not change when a placement is appended.  The delta is
    therefore its physical volume minus only the valid nesting pairs that contain the
    new placement.  The common, non-nesting case is O(1); the scan is paid only for an
    item that actually declares ``nesting_height``.
    """
    nesting = placement.instance.item.nesting_height
    if nesting is None:
        return occupied_volume(placement)
    overlap = 0
    for existing in placements:
        if is_valid_nesting(existing, placement):
            overlap += nesting.ticks * placement.envelope_dimensions.base_area
    return placement.dimensions.volume - overlap
