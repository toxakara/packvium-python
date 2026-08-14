from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .constraints import LoadUnit
    from .models import Axle


def axle_reactions(
    axles: tuple["Axle", "Axle"],
    units: Sequence["LoadUnit"],
    tare_weight_ticks: int = 0,
    tare_doubled_center_x: int = 0,
) -> tuple[int, int, int]:
    """Return ``(denominator, front_numerator, rear_numerator)`` exactly.

    Axle limits are gross-load limits. Payload contributes at each item's physical
    centre; tare contributes at the container's geometric longitudinal centre (the
    caller passes its doubled coordinate, preserving an odd half-tick).
    """
    front, rear = axles
    total_weight = tare_weight_ticks + sum(unit.weight_ticks for unit in units)
    doubled_weighted_x = tare_weight_ticks * tare_doubled_center_x + sum(
        unit.weight_ticks * (2 * unit.box.origin.x + unit.box.dimensions.length.ticks)
        for unit in units
    )
    denominator = 2 * (rear.position.ticks - front.position.ticks)
    return (
        denominator,
        2 * total_weight * rear.position.ticks - doubled_weighted_x,
        doubled_weighted_x - 2 * total_weight * front.position.ticks,
    )


def axle_load_exceeded(
    axles: tuple["Axle", "Axle"],
    units: Sequence["LoadUnit"],
    tare_weight_ticks: int = 0,
    tare_doubled_center_x: int = 0,
) -> tuple[str, str] | None:
    """Whether either axle bears more than its own limit, as (code, detail).

    Two-point beam statics: taking moments about each axle in turn gives an exact
    fraction for what the other axle carries, with no assumption weaker than "the
    container behaves like a rigid beam resting on exactly two supports" -- true by
    construction since `Container.axles` only ever has two entries.

    Takes `LoadUnit`s, the same "box + weight, real or hypothetical" abstraction
    `top_loads`/`stack_density_exceeded` already share, rather than requiring a real
    `Placement` -- a placement-time candidate does not have one yet. A unit's own box
    is already its envelope; the envelope's centre coincides exactly with the
    physical item's centre because clearance pads both sides of every axis equally,
    so no separate physical/envelope distinction is needed here.

    Never rounds an intermediate axle load. Both loads are compared to their limits
    by cross-multiplying the same exact fraction (numerator over `2 * (rear - front)`
    as the shared denominator), the same discipline `stack_density_exceeded` and
    `SupportConstraint` already use -- a rounded intermediate could hide a real
    overload or manufacture one that was not there.
    """
    front, rear = axles
    denominator, numerator_front, numerator_rear = axle_reactions(
        axles, units, tare_weight_ticks, tare_doubled_center_x
    )

    if front.max_load is not None and numerator_front > front.max_load.ticks * denominator:
        return ("axle_overloaded", "front")
    if rear.max_load is not None and numerator_rear > rear.max_load.ticks * denominator:
        return ("axle_overloaded", "rear")
    return None


def axle_balanced_origins(
    axles: tuple["Axle", "Axle"],
    other_units: Sequence["LoadUnit"],
    tare_weight_ticks: int,
    tare_doubled_center_x: int,
    item_weight_ticks: int,
    item_length_ticks: int,
) -> list[int]:
    """The tightest x-origins that seat this item exactly on either axle's limit.

    Candidate-point generation (`find_candidates`) only ever proposes positions
    flush against the container wall or another placed box's own corner -- correct
    for plain volume packing, where nothing is ever gained by leaving a gap, but
    incomplete once axle limits are in play: sometimes the only feasible spot for
    an item is floating away from every wall and every other box, specifically to
    keep this item's own moment from tipping one axle over its limit. A floor-level
    item needs no lateral contact for support (see `_support_ratio`), so that
    position is otherwise unreachable by any point this module already generates.

    Solving `axle_reactions`'s own boundary equation for this item's centre --
    "where would the front/rear reaction land exactly on its limit if this item's
    centre were here" -- turns the two axle limits into two extra x-origins worth
    trying, on top of the ordinary extreme points. Both are exact integer ticks,
    biased toward the safe side of their own limit (never past it) since a
    placement that lands exactly on the boundary is still allowed by
    `axle_load_exceeded`'s own strict `>` comparison; the caller runs every
    candidate through the same collision and constraint checks regardless, so a
    boundary that turns out unreachable (blocked, out of bounds, or infeasible for
    the other axle) is simply rejected same as any other candidate.

    Never assumes there is room: the caller clamps and discards out-of-range
    results.
    """
    if item_weight_ticks <= 0:
        return []
    front, rear = axles
    denominator = 2 * (rear.position.ticks - front.position.ticks)
    other_total = tare_weight_ticks + sum(unit.weight_ticks for unit in other_units)
    other_doubled_x = tare_weight_ticks * tare_doubled_center_x + sum(
        unit.weight_ticks * (2 * unit.box.origin.x + unit.box.dimensions.length.ticks)
        for unit in other_units
    )
    total = other_total + item_weight_ticks
    origins: list[int] = []
    if front.max_load is not None:
        # Smallest doubled centre for which numerator_front <= front limit * denominator.
        required = 2 * total * rear.position.ticks - front.max_load.ticks * denominator - other_doubled_x
        doubled_centre = -(-required // item_weight_ticks)  # ceil: never understate what front needs
        origins.append(-(-(doubled_centre - item_length_ticks) // 2))  # ceil: stay on the safe side
    if rear.max_load is not None:
        # Largest doubled centre for which numerator_rear <= rear limit * denominator.
        allowed = rear.max_load.ticks * denominator + 2 * total * front.position.ticks - other_doubled_x
        doubled_centre = allowed // item_weight_ticks  # floor: never overstate what rear allows
        origins.append((doubled_centre - item_length_ticks) // 2)  # floor: stay on the safe side
    return origins
