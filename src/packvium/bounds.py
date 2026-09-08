"""Lower bounds on the objective vector, and the gap against an incumbent.

The mathematics is fixed by [docs/OPTIMALITY-CERTIFICATES.md](../../../docs/OPTIMALITY-CERTIFICATES.md)
and `scripts/optimality_bounds.py` is the independent oracle. This module is written from
the document and never imports the oracle, so the property tests compare two
implementations rather than one implementation with itself -- the discipline set
for irregular items and restated here.

What a bound is for. Every solver in this project is a heuristic: it returns an
arrangement and has no notion of what it did not try. A bound is the other half of that
sentence -- the best score the request *could* admit, computed by relaxing the problem
until it becomes arithmetic. Where the achieved score meets the bound, the search is over
whether or not it explored anything, and the engine knows it.

What it is not. These bounds relax geometry away entirely. Attaining one certifies
optimality *of the relaxation*, never of the packing, and the difference between those two
claims is the difference between a true statement and a marketing one.

Instances that occupy less than their box. The document states this rule for
`nesting_height`: such an instance occupies less than its nominal volume, so nominal volumes
stop summing and every capacity argument built on them stops being a bound. The volume terms
are *dropped* rather than scaled -- a bound that is sometimes wrong is not a bound.

`convex_hull` and `compressible` have exactly the same property and the document does not
say so, because was written before the irregular shapes existed. A hull occupies
its hull rather than its bounding box, and a compressible item gives up height under load;
summing nominal box volumes over-states what a solution must carry, which over-states the
container count.
Measured on the golden corpus, that made the bound unsound on three fixtures --
`feature-convex-hull-wedges-share-one-crate`, `feature-route-hull-volume-reserve` and
`feature-compressible-volume-reserve-recomputes-loads` -- each claiming two containers were
necessary where one sufficed. They are treated here by the document's own rule.

That is the conservative repair, not the best one. A hull's exact volume is already computed
by `hull.shape_for`, and using it would keep the volume argument alive and tighter instead of
dropping it; a compressible item has a computable fully-compressed floor. Both are new
mathematics and belong to the design task, not to this implementation of it -- recorded in
`docs/OPTIMALITY-CERTIFICATES.md` rather than invented here.

Objective key order. The bounds are keyed to the *default* objective,
`(unpacked_count, container_count, total_cost_minor, unused_volume_ppm, stack_height_ppm)`.
`lowest_cost` and `maximum_value` reorder those keys, so a bound vector compared against
their score vectors compares different quantities and means nothing. Callers pass the
objective they scored with.

Complexity. `O(n log n + c log c)` for `n` instances and `c` container types -- one sort of
the instance volumes, one of the weights, one of the per-unit costs. No geometry is touched
and no candidate position is generated, which is why this can be computed at the root of a
search rather than inside it.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from ._compat import dataclass
from .constraints import usable_volume
from .geometry import ShapeType
from .models import Container, Item, ItemInstance

#: Parts per million, the fixed-point scale keys 3 and 4 of the objective are carried at.
PPM = 1_000_000

#: Every sum in the bound path must stay below this.
#:
#: Declared rather than inherited from the language. Python's integers are unbounded, PHP's
#: silently become doubles on overflow, JavaScript's `Number` stops being exact past 2^53 and
#: Rust's `i128` wraps -- so if each engine refused at its own limit the four would disagree
#: about which requests are answerable. The value comes from the widest intermediate the
#: formulas form: keys 3 and 4 multiply a summed volume by `PPM`, so `10^30 * 10^6 = 10^36`
#: sits about 170-fold inside `i128`. It never binds on a real request: `10^30` cubic ticks is
#: 244 million cubic metres, and a request that trips it has a data error in it.
MAX_BOUND_SUM = 10 ** 30

#: Largest integer every binding can return without changing its value.
#:
#: Intermediate arithmetic deliberately has the wider ``MAX_BOUND_SUM`` ceiling, but the
#: result crosses JSON and JavaScript's ``Number`` boundary.  Keeping the five public result
#: keys inside ``2**53 - 1`` makes "byte-identical" mean numerically identical as well as
#: textually similar; a larger cost must be refused before one engine rounds it.
MAX_BOUND_VALUE = 2 ** 53 - 1


class BoundOverflowError(ValueError):
    """A sum in the bound path exceeded the declared ceiling.

    A structured refusal rather than a number, because the alternative is the failure
    Baldacci et al. document for floating-point bin-packing solvers: a bound that is quietly
    wrong and carries no signal that it is. Every engine refuses at the same declared
    ceiling, so a request is either answerable everywhere or refused everywhere.
    """


class UnsoundBoundError(ValueError):
    """An achieved score fell below its own lower bound.

    One of the two is wrong, and saying so is more useful than reporting a negative gap.
    This is the assertion that fires first if a bound is ever made unsound, which is why it
    raises rather than clamping.
    """


@dataclass(frozen=True, slots=True)
class Bounds:
    """The five lower bounds, in the objective's own key order.

    Each is conditional on the keys before it: `container_count` is the least number of
    containers *given* that `unpacked_count` items were left behind, and so on down. That
    is what makes them comparable to a score vector key by key, and also why the gap stops
    at the first key that misses.
    """

    unpacked_count: int
    container_count: int
    total_cost_minor: int
    unused_volume_ppm: int
    stack_height_ppm: int

    def as_tuple(self) -> Tuple[int, int, int, int, int]:
        return (self.unpacked_count, self.container_count, self.total_cost_minor,
                self.unused_volume_ppm, self.stack_height_ppm)


@dataclass(frozen=True, slots=True)
class Gap:
    """How far an incumbent stands from the bound, on the first key that misses.

    `relative` is an exact rational pair `(numerator, denominator)` rather than a float:
    the reference arithmetic stays exact and each binding formats its own percentage
    without binary floating point becoming part of a certificate. It is `None` -- undefined,
    not substituted -- when the bound on that key is zero, which is the common case for
    `unpacked_count`. Dividing by a stand-in denominator would produce a number that looks
    like a percentage and is not one.
    """

    #: Index of the first key where the incumbent exceeds its bound, or `None` when every
    #: key is attained and the incumbent is optimal for this relaxation.
    key: "int | None"
    absolute: int
    relative: "Tuple[int, int] | None"

    @property
    def attained(self) -> bool:
        return self.key is None


def _capacity_total(values: Iterable["int | None"], quantities: Iterable["int | None"],
                    unbounded_when_value_infinite: bool) -> "int | None":
    """`Σc value · quantity`, or `None` for an unbounded total.

    `None` means infinity on the way in and on the way out. A limit nobody declared cannot
    be summed, and a single unlimited type makes the whole capacity unbounded -- except for
    volume, where a container with no usable volume adds nothing however many of it there
    are, which is why the caller says which rule applies.
    """
    total = 0
    for value, quantity in zip(values, quantities):
        if value is None:
            if unbounded_when_value_infinite:
                return None
            continue
        if quantity is None:
            if value > 0:
                return None
            continue
        total += value * quantity
    return _guard(total, "container capacity")


def _fit(ascending: Sequence[int], capacity: "int | None") -> int:
    """The largest `n` such that the `n` smallest values sum to at most `capacity`.

    Smallest first, and that is the whole soundness argument: taking the cheapest units
    maximises how many fit under one capacity, so this over-estimates what any real packing
    achieves. Geometry, support ratios, stacking rules, incompatible tags and route order
    can each make the real answer worse and none of them can make it better.

    The caller passes an already-ascending sequence. `compute` sorts the volumes and weights
    once for the later keys, and sorting them again here would have doubled the only
    superlinear work this module does.
    """
    if capacity is None:
        return len(ascending)
    used = 0
    for taken, cost in enumerate(ascending):
        used += cost
        if used > capacity:
            return taken
    return len(ascending)


def _guard(total: int, quantity: str) -> int:
    """Refuse a sum past the declared ceiling instead of carrying it further."""
    if total > MAX_BOUND_SUM:
        raise BoundOverflowError(
            f"{quantity} sums to {total}, above the {MAX_BOUND_SUM} ceiling the bound path "
            "declares; refusing rather than returning a number no engine can agree on"
        )
    return total


def _guard_output(value: int, quantity: str) -> int:
    """Refuse a result that cannot cross every binding exactly."""
    if value > MAX_BOUND_VALUE:
        raise BoundOverflowError(
            f"{quantity} bound is {value}, above the {MAX_BOUND_VALUE} exact portable "
            "result ceiling; refusing rather than rounding it in JavaScript"
        )
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _finite_max(values: Sequence["int | None"]) -> "int | None":
    """The largest declared limit, or `None` if any type declares none.

    One unlimited type makes the maximum unbounded and every term conditioned on it
    vacuous, which is why this collapses to `None` rather than ignoring the gap.
    """
    return None if any(value is None for value in values) else max(values)


def _occupies_less_than_its_box(item: Item) -> bool:
    """Can this item take up less room than its declared dimensions?

    Three ways, and every one of them breaks the same argument -- that nominal volumes sum
    to something a solution must carry. A nested item sinks into the one below it; a
    `convex_hull` occupies its hull and leaves the rest of its bounding box free, which is
    the entire reason that shape exists; a `compressible` item gives up height under load.

    The document names only the first because it predates the other two. Asking the question
    once, here, is what keeps a future fourth shape from reintroducing the same unsoundness
    silently -- a new `ShapeType` that is anything but a solid box has to pass this line.
    """
    if item.nesting_height is not None:
        return True
    return item.shape_type in (ShapeType.CONVEX_HULL, ShapeType.COMPRESSIBLE)


def compute(instances: Sequence[ItemInstance], containers: Sequence[Container]) -> Bounds:
    """Every bound for one request, at the root of a search.

    The instances are the quantity-expanded list the solver is about to place, and the
    containers are the types it may open -- both exactly as the request declares them,
    before any placement decision has been taken.
    """
    volumes = sorted(instance.item.dimensions.volume for instance in instances)
    weights = sorted(instance.weight.ticks for instance in instances)
    _guard(sum(volumes), "instance volume")
    _guard(sum(weights), "instance weight")
    nests = any(_occupies_less_than_its_box(instance.item) for instance in instances)
    count = len(instances)

    usable = [usable_volume(container) for container in containers]
    inner = [container.inner_dimensions.volume for container in containers]
    areas = [container.inner_dimensions.base_area for container in containers]
    heights = [container.inner_dimensions.height.ticks for container in containers]
    payloads = [None if c.max_payload is None else c.max_payload.ticks for c in containers]
    slots = [c.max_items for c in containers]
    quantities = [c.quantity for c in containers]
    costs = [c.cost_minor for c in containers]

    unpacked = _unpacked_bound(volumes, weights, nests, count, usable, payloads, slots,
                               quantities)
    placed = count - unpacked
    opened = _container_bound(volumes, weights, nests, placed, containers, usable, payloads,
                              slots)
    cost = _cost_bound(costs, quantities, opened)
    unused = _unused_volume_bound(volumes, nests, placed, inner, opened)
    height = _stack_height_bound(volumes, nests, placed, areas, heights, opened)
    return Bounds(
        _guard_output(unpacked, "unpacked count"),
        _guard_output(opened, "container count"),
        _guard_output(cost, "opening cost"),
        _guard_output(unused, "unused volume"),
        _guard_output(height, "stack height"),
    )


def _unpacked_bound(volumes, weights, nests, count, usable, payloads, slots,
                    quantities) -> int:
    """`L0`: remove the geometry entirely and ask what the declared resources alone forbid.

    Every instance becomes freely selectable, and only the three additive resources the
    request declares can stop one being placed. With two resources the minimum over each
    separately is a genuine relaxation and can fall below the true two-resource optimum; it
    remains a bound, which is all that is claimed.
    """
    volume_capacity = _capacity_total(usable, quantities, unbounded_when_value_infinite=False)
    payload_capacity = _capacity_total(payloads, quantities, unbounded_when_value_infinite=True)
    slot_capacity = _capacity_total(slots, quantities, unbounded_when_value_infinite=True)

    placeable = count
    if not nests:
        placeable = min(placeable, _fit(volumes, volume_capacity))
    placeable = min(placeable, _fit(weights, payload_capacity))
    if slot_capacity is not None:
        placeable = min(placeable, slot_capacity)
    return count - placeable


def _container_bound(volumes, weights, nests, placed, containers, usable, payloads,
                     slots) -> int:
    """`L1`: grant every container the largest capacity available, which can only understate
    how many are needed."""
    if placed <= 0 or not containers:
        return 0
    bound = 1
    if not nests:
        largest_usable = max(usable)
        if largest_usable > 0:
            bound = max(bound, _ceil_div(sum(volumes[:placed]), largest_usable))
    largest_payload = _finite_max(payloads)
    if largest_payload is not None and largest_payload > 0:
        bound = max(bound, _ceil_div(sum(weights[:placed]), largest_payload))
    largest_slots = _finite_max(slots)
    if largest_slots is not None and largest_slots > 0:
        bound = max(bound, _ceil_div(placed, largest_slots))
    return bound


def _cost_bound(costs, quantities, opened) -> int:
    """`L2`: at least `L1` containers open, each costing at least the cheapest the inventory
    still holds.

    Inventory is respected rather than assumed unlimited. Charging the cheapest type `L1`
    times would also be a bound, and a weaker one whenever that type is nearly exhausted;
    the difference costs one sort.
    """
    if opened <= 0:
        return 0
    available: list[int] = []
    for cost, quantity in zip(costs, quantities):
        available.extend([cost] * (opened if quantity is None else min(quantity, opened)))
    available.sort()
    return _guard(sum(available[:opened]), "opening cost")


def _unused_volume_bound(volumes, nests, placed, inner, opened) -> int:
    """`L3`: the fill is largest when every container is the smallest available and holds the
    greatest volume that could be placed at all.

    Key 3 sums a *per-container* ratio, so a tight bound would have to know which items went
    where -- that is the packing problem, not a relaxation of it. The `L1 - 1` term is the
    exact worst case by which summing `k` ceilings can exceed the ceiling of the sum.
    """
    if nests or opened <= 0 or not inner:
        return 0
    smallest_inner = min(inner)
    if smallest_inner <= 0:
        return 0
    largest_placed = sum(volumes[len(volumes) - placed:]) if placed > 0 else 0
    return max(0, opened * PPM - _ceil_div(largest_placed * PPM, smallest_inner) - (opened - 1))


def _stack_height_bound(volumes, nests, placed, areas, heights, opened) -> int:
    """`L4`: the volume that must be placed has to stand at least as tall as itself spread
    across the widest floor available, in the tallest container available.

    The `L1 - 1` correction is not a cautionary fudge. The objective floors each container's
    ratio *before* summing, and the sum of `k` floors can be one less than the floor of the
    sum at each of the first `k - 1` boundaries. Without it, two one-tick loads in height-6
    containers score `166666 + 166666 = 333332` while flooring their aggregate claims
    `333333` -- one ppm above a feasible solution, and therefore not a bound.
    """
    if nests or opened <= 0 or not areas:
        return 0
    widest = max(areas)
    tallest = max(heights)
    if widest <= 0 or tallest <= 0:
        return 0
    required = _ceil_div(sum(volumes[:placed]), widest) if placed > 0 else 0
    return max(0, required * PPM // tallest - (opened - 1))


def gap(score: Sequence[int], bound: Bounds) -> Gap:
    """The distance from an incumbent score to its bound, on the first key that misses.

    One key, not five: the keys after the first miss were computed on an assumption that has
    just been shown false -- that the earlier keys were attained -- so they say nothing and
    are not reported.
    """
    limits = bound.as_tuple()
    for index, (achieved, least) in enumerate(zip(score, limits)):
        if achieved < least:
            raise UnsoundBoundError(
                f"key {index} scored {achieved}, below its lower bound of {least}; "
                "one of the two is wrong"
            )
        if achieved > least:
            absolute = achieved - least
            return Gap(index, absolute, (absolute, least) if least > 0 else None)
    return Gap(None, 0, None)
