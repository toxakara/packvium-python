"""Made-to-measure containers (docs/MADE-TO-MEASURE-CONTAINERS.md "1.6 implementation
contract").

A made-to-measure container is cut by an on-demand box machine. Its declared
`inner_dimensions` is the largest box the machine can cut; `MadeToMeasure` adds the smallest
size, the cut step and the prices. A trial packs the round's items into up to four solving
envelopes ("rungs", `solving_envelopes`) and keeps the first that places every item; the
committed box is the achieved extent rounded up onto the machine grid (`fabricate`).

Everything here is exact integer arithmetic on ticks, square ticks, weight ticks and minor
units, so every engine derives the same rungs, the same box and the same price.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ._compat import dataclass
from ._request_error import refused
from .geometry import Dimensions
from .units import Length, Weight

if TYPE_CHECKING:
    from .models import Container

#: `docs/UNITS-AND-NUMERICS.md`: one length tick is 1/16 000 000 m.
LENGTH_TICKS_PER_METRE = 16_000_000
SQUARE_METRE_TICKS2 = LENGTH_TICKS_PER_METRE * LENGTH_TICKS_PER_METRE
#: The largest integer every binding returns unchanged; no fabricated price may pass it.
MAX_EXACT_INTEGER = 2**53 - 1
#: Each compact rung's side as a fraction of the integer cube root of the round's volume.
RUNG_FACTORS: tuple[tuple[int, int], ...] = ((5, 4), (3, 2), (2, 1))

Triple = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MadeToMeasure:
    """What the machine can cut beyond the declared maximum, per axis in ticks.

    `minimum` equal to the maximum pins an axis. `tare_per_square_metre` is weight ticks per
    square metre of inner surface; `outer_allowance` is the board added to each inner side.
    """

    minimum: Triple
    increment: Triple
    cost_per_square_metre_minor: int = 0
    tare_per_square_metre: int = 0
    outer_allowance: Triple = (0, 0, 0)

    def __post_init__(self) -> None:
        if min(self.minimum) <= 0:
            raise refused("invalid_value", "made_to_measure minimum must be positive")
        if min(self.increment) <= 0:
            raise refused("invalid_value", "made_to_measure increment must be positive")
        if self.cost_per_square_metre_minor < 0 or self.tare_per_square_metre < 0:
            raise refused("below_minimum", "made_to_measure prices cannot be negative")
        if min(self.outer_allowance) < 0:
            raise refused("negative_measure", "made_to_measure outer_allowance cannot be negative")

    def admits(self, maximum: Triple) -> None:
        """Refuse a machine whose grid does not land on its own maximum. O(1).

        Every reachable size is then unambiguous, and rounding an achieved extent that lies
        inside the maximum up onto the grid can never pass it (D3, D4)."""
        for axis, name in enumerate(("length", "width", "height")):
            if self.minimum[axis] > maximum[axis]:
                raise refused("invalid_value", f"made_to_measure minimum {name} exceeds inner_dimensions")
            if (maximum[axis] - self.minimum[axis]) % self.increment[axis]:
                raise refused(
                    "invalid_value", f"made_to_measure increment {name} does not divide inner_dimensions minus minimum"
                )

    def grid(self, axis: int, value: int) -> int:
        """The smallest cuttable size on `axis` that is at least `value`."""
        minimum, step = self.minimum[axis], self.increment[axis]
        return minimum + -(-(max(value, minimum) - minimum) // step) * step

    def clamp(self, axis: int, value: int, maximum: int) -> int:
        """`grid`, capped at the machine maximum: sizes a solving envelope, never a box."""
        return min(maximum, self.grid(axis, value))


def surface_area(sides: Triple) -> int:
    """The inner surface of a closed box, `2(LW + LH + WH)`, in square ticks."""
    length, width, height = sides
    return 2 * (length * width + length * height + width * height)


def per_square_metre(rate: int, area: int) -> int:
    """`ceil(rate * area / T^2)` without forming `rate * area`.

    `p*q + ceil(p*r / T^2)` with `q, r = divmod(area, T^2)`: every term stays below 2^128
    for any request admission lets through, which is what lets fixed-width engines agree."""
    whole, fraction = divmod(area, SQUARE_METRE_TICKS2)
    return rate * whole + -(-rate * fraction // SQUARE_METRE_TICKS2)


def box_cost_minor_for(price_minor: int, fee_minor: int, sides: Triple) -> int:
    """A box's price: the container's own fee plus its material at `price_minor` per square
    metre, rounded up. Cost grows with every side, so the machine maximum bounds them all."""
    return fee_minor + per_square_metre(price_minor, surface_area(sides))


def box_cost_minor(spec: MadeToMeasure, fee_minor: int, sides: Triple) -> int:
    return box_cost_minor_for(spec.cost_per_square_metre_minor, fee_minor, sides)


def box_tare_ticks(spec: MadeToMeasure, declared_tare: int, sides: Triple) -> int:
    """The fabricated box's weight: the declared tare plus its board, rounded up."""
    return declared_tare + per_square_metre(spec.tare_per_square_metre, surface_area(sides))


def icbrt_ceil(value: int) -> int:
    """The smallest non-negative `c` with `c**3 >= value`: doubling, then bisection.
    O(log value) exact steps, under 90 for any volume below 2^128."""
    high = 1
    while high**3 < value:
        high *= 2
    low = 0
    while low < high:
        middle = (low + high) // 2
        if middle**3 >= value:
            high = middle
        else:
            low = middle + 1
    return low


def solving_envelopes(spec: MadeToMeasure, maximum: Triple, edges: Iterable[Triple]) -> tuple[Triple, ...]:
    """The rungs one trial tries, ascending on every axis, the machine maximum last.

    `edges` are each remaining instance's envelope edges (nominal plus twice the clearance).
    O(r) over the `r` instances plus the cube root; the list depends only on their multiset.
    """
    volume = 0
    floors = [0, 0, 0]
    for instance in edges:
        volume += instance[0] * instance[1] * instance[2]
        ranked = sorted(instance, reverse=True)
        for rank in range(3):
            floors[rank] = max(floors[rank], ranked[rank])
    side = icbrt_ceil(volume)
    rungs: list[Triple] = []
    for numerator, denominator in RUNG_FACTORS:
        scaled = -(-side * numerator // denominator)
        rung = tuple(spec.clamp(axis, max(scaled, floors[axis]), maximum[axis]) for axis in range(3))
        if rung[0] * rung[1] * rung[2] >= volume and rung not in rungs:
            rungs.append(rung)
    if maximum not in rungs:
        rungs.append(maximum)
    return tuple(rungs)


def instance_edges(dimensions: Iterable[Dimensions], clearance_ticks: int) -> list[Triple]:
    """Each instance's envelope edges, the input `solving_envelopes` sizes rungs from."""
    grow = 2 * clearance_ticks
    return [
        (size.length.ticks + grow, size.width.ticks + grow, size.height.ticks + grow)
        for size in dimensions
    ]


def as_triple(dimensions: Dimensions) -> Triple:
    return (dimensions.length.ticks, dimensions.width.ticks, dimensions.height.ticks)


def as_dimensions(sides: Sequence[int]) -> Dimensions:
    return Dimensions(Length(sides[0]), Length(sides[1]), Length(sides[2]))


def with_envelope(container: "Container", rung: Triple) -> "Container":
    """The declared container solved inside `rung`: same id, everything else unchanged."""
    return replace(container, inner_dimensions=as_dimensions(rung))


def fabricate(declared: "Container", achieved: Triple) -> "Container":
    """The box cut for an achieved extent: equal to the declared container except its size,
    outer size, price and tare. O(1). Containment is a theorem, not a check: every placement
    lies inside `achieved`, and `grid` never rounds below it or past the maximum (D4)."""
    spec = declared.made_to_measure
    assert spec is not None
    maximum = as_triple(declared.inner_dimensions)
    sides = tuple(spec.grid(axis, achieved[axis]) for axis in range(3))
    assert all(sides[axis] <= maximum[axis] for axis in range(3)), "fabricated box exceeds the machine"
    outer = tuple(sides[axis] + spec.outer_allowance[axis] for axis in range(3))
    return replace(
        declared,
        inner_dimensions=as_dimensions(sides),
        outer_dimensions=as_dimensions(outer),
        cost_minor=box_cost_minor(spec, declared.cost_minor, sides),
        tare_weight=Weight(box_tare_ticks(spec, declared.tare_weight.ticks, sides)),
    )
