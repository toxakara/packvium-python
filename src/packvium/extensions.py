from __future__ import annotations

from ._compat import dataclass
from typing import Callable, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PackingConfig
    from .constraints import PlacementConstraint
    from .models import Container, ItemInstance, PackedContainer, UnpackedItem
    from .solvers import RawSolution, SingleContainerSolution, SingleContainerSolver


class ItemOrderStrategy(Protocol):
    name: str
    def order(self, items: Sequence["ItemInstance"], seed: int) -> Sequence["ItemInstance"]: ...


class ContainerSelector(Protocol):
    def score(self, container: "Container", solution: "SingleContainerSolution") -> tuple: ...


class SolutionScorer(Protocol):
    def score(self, solution: "RawSolution") -> tuple[int, ...]: ...


class DefaultSolutionScorer:
    """Canonical objective vector — see docs/OBJECTIVE.md.

    Five lexicographic keys, ascending, lower is better. Every key is an exact
    integer: the ratios are floored per container rather than computed in binary
    floating point, so Python, PHP, Rust and the JavaScript fallback all agree
    bit-for-bit.
    """

    #: Ratios are expressed as parts per million so the vector stays exactly comparable.
    SCALE = 1_000_000

    def score(self, solution: "RawSolution") -> tuple[int, ...]:
        return self.score_containers(solution.containers, solution.unpacked)

    @classmethod
    def score_containers(
        cls, containers: Sequence["PackedContainer"], unpacked: Sequence["UnpackedItem"]
    ) -> tuple[int, ...]:
        cost = sum(c.container.cost_minor for c in containers)
        unused = 0
        height = 0
        for c in containers:
            volume = c.container.inner_dimensions.volume
            unused += (volume - c.used_volume) * cls.SCALE // volume
            height += c.max_z_ticks * cls.SCALE // c.container.inner_dimensions.height.ticks
        return (len(unpacked), len(containers), cost, unused, height)


class LowestCostSolutionScorer:
    """Ranks by total container cost ahead of unused volume and stack height.

    Completeness still dominates every other key -- a solution that leaves items
    unpacked never wins for being cheaper, only among otherwise-complete solutions is
    cost the deciding factor. See docs/OBJECTIVE.md.
    """

    def score(self, solution: "RawSolution") -> tuple[int, ...]:
        return self.score_containers(solution.containers, solution.unpacked)

    @classmethod
    def score_containers(
        cls, containers: Sequence["PackedContainer"], unpacked: Sequence["UnpackedItem"]
    ) -> tuple[int, ...]:
        default = DefaultSolutionScorer.score_containers(containers, unpacked)
        unpacked_count, container_count, cost, unused, height = default
        return (unpacked_count, cost, container_count, unused, height)


class ShippingCostSolutionScorer:
    """Ranks by total carrier-billable weight ahead of container count, unused volume
    and stack height.

    Each container's billable weight is `max(actual gross weight, dimensional weight)`
    -- the standard carrier rule that a light-but-bulky shipment is billed by volume,
    not scale weight. Dimensional weight is computed from the container's *outer*
    (shipped-package) dimensions when declared, falling back to *inner* (usable)
    dimensions only when no outer size was given -- a carrier measures the box that
    moves, not its usable interior, and the two can differ once wall thickness or
    packaging is involved. Exact-integer throughout via `geometry.dimensional_weight`,
    using the caller-supplied divisor -- there is no library-chosen default, since a
    wrong guess would silently misprice every shipment. Every input (each container's
    reported `gross_weight`, its request-declared dimensions, and the caller's own
    divisor) is already public in the request/result pair, so this number is meant to
    be reproduced by hand against a carrier's own calculator, the same audit property
    `geometry.dimensional_weight` itself documents.
    """

    def __init__(self, divisor: int, length_unit: str = "in", weight_unit: str = "lb"):
        self.divisor = divisor
        self.length_unit = length_unit
        self.weight_unit = weight_unit

    @classmethod
    def from_config(cls, config: "PackingConfig | None") -> "ShippingCostSolutionScorer":
        if config is None or config.dimensional_weight_divisor is None:
            raise UnknownObjectiveError(
                "the shipping_cost objective requires configuration.dimensional_weight_divisor"
            )
        return cls(
            config.dimensional_weight_divisor,
            config.dimensional_weight_length_unit,
            config.dimensional_weight_weight_unit,
        )

    def score(self, solution: "RawSolution") -> tuple[int, ...]:
        return self.score_containers(solution.containers, solution.unpacked)

    def score_containers(
        self, containers: Sequence["PackedContainer"], unpacked: Sequence["UnpackedItem"]
    ) -> tuple[int, ...]:
        from .geometry import dimensional_weight

        default = DefaultSolutionScorer.score_containers(containers, unpacked)
        unpacked_count, container_count, _cost, unused, height = default
        billable = 0
        for c in containers:
            dimensions = c.container.outer_dimensions or c.container.inner_dimensions
            dim_weight = dimensional_weight(dimensions, self.divisor, self.length_unit, self.weight_unit)
            billable += max(c.gross_weight.ticks, dim_weight.ticks)
        return (unpacked_count, billable, container_count, unused, height)


class LandedCostSolutionScorer(ShippingCostSolutionScorer):
    """Ranks by the money a shipment actually costs, not by the grams it is billed at
.

    `shipping_cost` already ranks by billable weight, and for one tariff whose price rises
    with weight the two orderings agree -- which is why this exists only for the cases
    where they do not. A bracket step means two packings a gram apart can cost the same or
    differ by a whole band, and a minimum charge flattens every light shipment onto one
    price; in both, the cheaper answer is not the lighter one. Reusing
    `ShippingCostSolutionScorer`'s billed weight rather than recomputing it is deliberate:
    the two objectives must never disagree about *what* is being priced.

    Every container must carry a `rate_table`. Rating some containers and not others would
    silently rank a priced packing against an unpriced one as though the unpriced were
    free, so a missing table is a rejection.

    A billed weight past the last bracket is different: it is a property of how the search
    happened to fill the box, not of the request, so it loses a candidate rather than
    aborting a run that has a perfectly shippable alternative. Scoring it `UNPRICEABLE_MINOR`
    is what makes the priceable alternative win; `pack` refuses if that sentinel is still
    standing when an answer is about to be returned.
    """

    def score_containers(
        self, containers: Sequence["PackedContainer"], unpacked: Sequence["UnpackedItem"]
    ) -> tuple[int, ...]:
        from .geometry import dimensional_weight

        default = DefaultSolutionScorer.score_containers(containers, unpacked)
        unpacked_count, container_count, _cost, unused, height = default
        landed = 0
        for c in containers:
            table = c.container.rate_table
            if table is None:
                raise UnknownObjectiveError(
                    f"the lowest_landed_cost objective requires a rate_table on every "
                    f"container; {c.container.id!r} has none"
                )
            dimensions = c.container.outer_dimensions or c.container.inner_dimensions
            dim_weight = dimensional_weight(dimensions, self.divisor, self.length_unit, self.weight_unit)
            billed_ticks = max(c.gross_weight.ticks, dim_weight.ticks)
            charge = table.charge_minor_or_none(_grams(billed_ticks))
            if charge is None:
                landed = UNPRICEABLE_MINOR
                break
            landed += charge
        return (unpacked_count, landed, container_count, unused, height)


#: Ranks a packing the tariff cannot price behind every priceable one during search.
#: It is a search device, never an answer -- `pack` refuses before a solution carrying it
#: can be returned -- so its exact magnitude only has to dominate any real total. The value
#: is Rust's `i128::MAX as i64` so the three engines that need a sentinel share one.
UNPRICEABLE_MINOR = 2**63 - 1


def unpriceable_container(
    containers: Sequence["PackedContainer"], config: "PackingConfig"
) -> "tuple[str, int, int] | None":
    """The first container in a finished answer its own rate table cannot price, as
    `(container id, billed grams, last bracket)`.

    Ranking an unpriceable candidate worst is what lets a priceable alternative win the
    round. This is the guard that stops the sentinel from surfacing: returning a packing
    the tariff cannot price would quote a number the carrier never published.
    """
    from .geometry import dimensional_weight

    if config.objective != "lowest_landed_cost" or config.dimensional_weight_divisor is None:
        return None
    for c in containers:
        table = c.container.rate_table
        dimensions = c.container.outer_dimensions or c.container.inner_dimensions
        dim_weight = dimensional_weight(
            dimensions,
            config.dimensional_weight_divisor,
            config.dimensional_weight_length_unit,
            config.dimensional_weight_weight_unit,
        )
        grams = _grams(max(c.gross_weight.ticks, dim_weight.ticks))
        if table is None:
            return (c.container.id, grams, 0)
        if table.charge_minor_or_none(grams) is None:
            return (c.container.id, grams, table.weight_brackets_g[-1])
    return None


def _grams(weight_ticks: int) -> int:
    """Billed weight in whole grams, rounded up.

    The rate table is published in grams while the engine carries sub-gram ticks. Rounding
    up matches how a carrier reads a scale: a shipment fractionally over a bracket is in
    the next bracket, and rounding down would price it below what the carrier charges.
    """
    from .units import Weight

    return -(-weight_ticks // Weight.TICKS_PER_G)


class OpenDimensionSolutionScorer:
    """Ranks by the raw achieved stack height ahead of container count, cost and
    unused volume.

    `default`/`lowest_cost` express height as a *ratio* against the container's own
    declared inner height (`stack_height_ppm`) -- a sensible number when that height is
    a real, meaningful upper bound the caller cares about staying under. It stops being
    meaningful once a caller declares a container whose height is a generous, otherwise
    arbitrary ceiling rather than a real limit -- the "open dimension" case,
    where the real question is "how tall does this end up", not "what fraction of some
    height did it use". This objective answers that question directly: it ranks by the
    summed raw achieved height (`PackedContainer.max_z_ticks`, already computed by the
    solver -- no new geometry work) across every opened container, in ticks, ahead of
    `container_count`/`total_cost_minor`/`unused_volume_ppm`. Every solver already
    treats a container's declared inner height as a hard upper bound during placement
    (there is no actually-unbounded container in this release -- see the scope
    note in the project's issue tracker), so a caller wanting an open-dimension answer supplies a
    height generous enough not to bind, and lets this objective find the shortest
    arrangement that still fits everything else.

    O(c) additional work for `c` opened containers: one already-computed integer field
    read per container, no new placement or geometry computation -- the same bound as
    `DefaultSolutionScorer`/`LowestCostSolutionScorer` above.
    """

    def score(self, solution: "RawSolution") -> tuple[int, ...]:
        return self.score_containers(solution.containers, solution.unpacked)

    @classmethod
    def score_containers(
        cls, containers: Sequence["PackedContainer"], unpacked: Sequence["UnpackedItem"]
    ) -> tuple[int, ...]:
        default = DefaultSolutionScorer.score_containers(containers, unpacked)
        unpacked_count, container_count, cost, unused, _height_ppm = default
        achieved_height = sum(c.max_z_ticks for c in containers)
        return (unpacked_count, achieved_height, container_count, cost, unused)


class MaximumValueSolutionScorer:
    """Ranks by the total declared value of unpacked items ahead of container count,
    cost and unused volume.

    Every objective begins with `unpacked_count`: no selectable trade-off may prefer
    an incomplete answer over a complete one, this one included. Its purpose is the
    tie-break among solutions that leave the same *number* of items unpacked but not
    the same *value* -- the number alone is indifferent between leaving behind a
    pallet of high-value goods or a pallet of packing foam. `value_forgone` is the
    sum of `Item.value` (zero when unset) across every unpacked item.

    O(u) additional work for `u` unpacked items -- one already-known integer field
    read per unpacked item, the same bound class as every other named objective here.
    """

    def score(self, solution: "RawSolution") -> tuple[int, ...]:
        return self.score_containers(solution.containers, solution.unpacked)

    @classmethod
    def score_containers(
        cls, containers: Sequence["PackedContainer"], unpacked: Sequence["UnpackedItem"]
    ) -> tuple[int, ...]:
        default = DefaultSolutionScorer.score_containers(containers, unpacked)
        unpacked_count, container_count, cost, unused, _height = default
        value_forgone = sum(u.instance.item.value or 0 for u in unpacked)
        return (unpacked_count, value_forgone, container_count, cost, unused)


#: Named objectives a caller may select through `PackingConfig.objective`. Each factory
#: takes the active `PackingConfig` (possibly `None`, for callers that score without
#: one) so a parametrized objective like `shipping_cost` can pull its own inputs from
#: it without widening the `SolutionScorer.score(solution)` protocol every extension
#: author already implements against.
OBJECTIVE_SCORERS: dict[str, "Callable[[PackingConfig | None], SolutionScorer]"] = {
    "default": lambda config: DefaultSolutionScorer(),
    "lowest_cost": lambda config: LowestCostSolutionScorer(),
    "shipping_cost": ShippingCostSolutionScorer.from_config,
    # Inherits `from_config`, and therefore the same divisor requirement: landed cost is
    # priced off billed weight, so an objective that could not compute dimensional weight
    # would be pricing the wrong number.
    "lowest_landed_cost": LandedCostSolutionScorer.from_config,
    "open_dimension_height": lambda config: OpenDimensionSolutionScorer(),
    "maximum_value": lambda config: MaximumValueSolutionScorer(),
}


class UnknownObjectiveError(ValueError):
    """`PackingConfig.objective` named an objective this library does not implement,
    or named one whose required configuration (e.g. `shipping_cost`'s divisor) is
    missing."""


def resolve_objective_scorer(name: str, config: "PackingConfig | None" = None) -> SolutionScorer:
    try:
        factory = OBJECTIVE_SCORERS[name]
    except KeyError:
        raise UnknownObjectiveError(
            f"unknown objective {name!r}; expected one of {sorted(OBJECTIVE_SCORERS)}"
        ) from None
    return factory(config)


@dataclass(frozen=True, slots=True)
class ExtensionRegistry:
    placement_constraints: tuple["PlacementConstraint", ...] = ()
    item_order_strategies: tuple[ItemOrderStrategy, ...] = ()
    solvers: tuple["SingleContainerSolver", ...] = ()
    container_selector: ContainerSelector | None = None
