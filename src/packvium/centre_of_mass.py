from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .geometry import Dimensions

if TYPE_CHECKING:
    from .models import Placement


def centre_of_mass_offset_ppm(inner_dimensions: Dimensions, placements: Sequence[Placement]) -> int:
    """How far the weighted centre of mass sits from the container's own centre.

    Reported as parts per million of the half-extent along whichever of the two
    horizontal axes (length, width) is worse -- the Chebyshev, not Euclidean, offset,
    so the result stays exact: a Euclidean distance would need a square root, and
    this library's whole premise is exact fixed-point arithmetic. 0% means
    centred; 100% means the mass sits at the very edge along that axis. Needed for
    axle load and side-to-side balance, both of which care about the worst axis, not
    a single blended number that could hide either one.

    Every item contributes its own physical centre, weighted by its own mass -- an
    item's clearance envelope is packing buffer, not part of what it weighs, so
    `position`/`dimensions` are used rather than the (possibly larger) envelope.

    Kept exact throughout via one combined numerator/denominator per axis and a
    single final floor division, rather than rounding an intermediate centroid --
    the same discipline `required_area`/`unused_volume_ppm` already use elsewhere.
    """
    total_weight = sum(p.instance.weight.ticks for p in placements)
    if total_weight == 0:
        return 0
    length = inner_dimensions.length.ticks
    width = inner_dimensions.width.ticks
    doubled_weighted_x = sum(p.instance.weight.ticks * (2 * p.position.x + p.dimensions.length.ticks) for p in placements)
    doubled_weighted_y = sum(p.instance.weight.ticks * (2 * p.position.y + p.dimensions.width.ticks) for p in placements)
    numerator_x = doubled_weighted_x - total_weight * length
    numerator_y = doubled_weighted_y - total_weight * width
    offset_x_ppm = abs(numerator_x) * 1_000_000 // (total_weight * length)
    offset_y_ppm = abs(numerator_y) * 1_000_000 // (total_weight * width)
    return max(offset_x_ppm, offset_y_ppm)
