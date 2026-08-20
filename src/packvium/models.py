from __future__ import annotations

from dataclasses import field

from ._compat import dataclass
from typing import Any, Iterable, Mapping

from .centre_of_mass import centre_of_mass_offset_ppm
from .geometry import AxisAlignedBox, Dimensions, Point, Rotation
from .lattice_summary import LatticeSummary
from .nesting import used_volume as nesting_used_volume
from .units import Length, Weight


GROUND_CONTACT_RULES = frozenset({"free", "covered", "single", "multiple"})


@dataclass(frozen=True, slots=True)
class Obstacle:
    """An excluded zone. `additional_boxes` expresses a non-rectangular shape (a
    wheel arch, a tapered roof) as a union of exact boxes rather than modelling
    slants directly, so the geometry stays integral. Every existing
    single-box obstacle is exactly the one-box case of this union, so nothing about
    the default construction changes.
    """
    id: str
    box: AxisAlignedBox
    additional_boxes: tuple[AxisAlignedBox, ...] = ()

    @property
    def boxes(self) -> tuple[AxisAlignedBox, ...]:
        return (self.box, *self.additional_boxes)


@dataclass(frozen=True, slots=True)
class Axle:
    """One axle's position along the container's length axis and its own limit.

    Two-axle only (front, rear), matching the usual two-axle scope: the load on more
    than two supports is statically indeterminate without further assumptions this
    library has no basis for making, so it is not modelled.
    """
    position: Length
    max_load: Weight | None = None


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    dimensions: Dimensions
    weight: Weight = Weight(0)
    quantity: int = 1
    allowed_rotations: tuple[Rotation, ...] = field(default_factory=Rotation.all)
    keep_upright: bool = False
    stackable: bool = True
    must_be_on_floor: bool = False
    max_top_load: Weight | None = None
    max_stacked_items: int | None = None
    minimum_support_ratio: float = 0.0
    ground_contact_rule: str | None = None
    group: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    incompatible_tags: frozenset[str] = field(default_factory=frozenset)
    priority: int = 0
    eligible_container_tags: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # How much this item sinks into an identical item directly beneath it when
    # stacked, so a nested column needs less than the sum of every unit's height
    #. Only ``GridSolver``'s uniform lattice exploits this -- every other
    # solver places nestable items at their full height, which is always safe, just
    # not space-optimal, a tracked scope limit rather than a silent gap.
    nesting_height: Length | None = None
    # Which stop along a multi-stop route this item is unloaded at, lower leaving
    # first. `None` means the item is not on a route at all -- every
    # existing single-stop request leaves this unset, so `packing_sequence`'s route
    # check has nothing to enforce and behaves exactly as it did before this field
    # existed. Only order between items matters, not any absolute stop identity, so
    # this stays a bare non-negative integer rather than a richer stop object.
    stop_index: int | None = None
    # Exact, unit-less economic worth for the `maximum_value` objective,
    # which ranks by the total value of unpacked items rather than treating every
    # item as equally worth leaving behind. `None` (the default) never affects
    # placement or scoring under any objective, `maximum_value` included -- an
    # item with no declared value simply contributes zero to value forgone.
    value: int | None = None

    def __post_init__(self) -> None:
        if not self.id: raise ValueError("item id is required")
        if self.quantity <= 0: raise ValueError("item quantity must be positive")
        if not 0 <= self.minimum_support_ratio <= 1: raise ValueError("minimum_support_ratio must be between 0 and 1")
        if self.max_stacked_items is not None and self.max_stacked_items < 1:
            raise ValueError("max_stacked_items must be at least 1")
        if self.stop_index is not None and self.stop_index < 0:
            raise ValueError("stop_index must be non-negative")
        if self.value is not None and self.value < 0:
            raise ValueError("value must be non-negative")
        if self.nesting_height is not None and not 0 <= self.nesting_height.ticks < self.dimensions.height.ticks:
            raise ValueError("nesting_height must be at least zero and strictly less than the item's own height")
        if self.ground_contact_rule is not None and self.ground_contact_rule not in GROUND_CONTACT_RULES:
            raise ValueError(f"ground_contact_rule must be one of {sorted(GROUND_CONTACT_RULES)}")
        rotations = tuple(r for r in self.allowed_rotations if not self.keep_upright or r in Rotation.upright())
        if not rotations: raise ValueError("at least one rotation must be allowed")
        object.__setattr__(self, "allowed_rotations", rotations)
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "incompatible_tags", frozenset(self.incompatible_tags))
        object.__setattr__(self, "eligible_container_tags", frozenset(self.eligible_container_tags))
        # Coerce here rather than in `create`, so every construction path is covered. An
        # unparsed weight used to travel all the way into the solver before failing.
        object.__setattr__(self, "weight", Weight.parse(self.weight))
        if self.max_top_load is not None:
            object.__setattr__(self, "max_top_load", Weight.parse(self.max_top_load))

    @classmethod
    def create(cls, id: str, dimensions: Dimensions, weight=0, **kwargs) -> "Item":
        return cls(id=id, dimensions=dimensions, weight=Weight.parse(weight), **kwargs)

    def instances(self) -> tuple["ItemInstance", ...]:
        return tuple(ItemInstance(self, sequence) for sequence in range(1, self.quantity + 1))


@dataclass(frozen=True, slots=True)
class ItemInstance:
    item: Item
    sequence: int

    @property
    def id(self) -> str: return f"{self.item.id}#{self.sequence}"
    @property
    def dimensions(self) -> Dimensions: return self.item.dimensions
    @property
    def weight(self) -> Weight: return self.item.weight


@dataclass(frozen=True, slots=True)
class RateTable:
    """A carrier rate card carried by the request rather than by a code plugin.

    `weight_brackets_g` is strictly ascending and the same length as `prices_minor`; the
    price charged is the one at the first bracket at or above the billed weight, which is
    how a carrier's own published table reads. Everything is an exact integer -- grams,
    minor currency units, parts per thousand -- because a landed cost that participates in
    the objective must be reproducible bit for bit in four engines, and a float is not.

    A price is *not* required to be non-decreasing. A promotional band that dips is a real
    rate card, and the table is read by bracket rather than by comparing prices, so a dip
    prices correctly instead of being rejected as malformed.
    """

    weight_brackets_g: tuple[int, ...]
    prices_minor: tuple[int, ...]
    minimum_charge_minor: int = 0
    fuel_surcharge_permille: int = 0

    def __post_init__(self) -> None:
        if not self.weight_brackets_g:
            raise ValueError("rate_table requires at least one weight bracket")
        if len(self.weight_brackets_g) != len(self.prices_minor):
            raise ValueError("rate_table weight_brackets_g and prices_minor must be the same length")
        if any(bound <= 0 for bound in self.weight_brackets_g):
            raise ValueError("rate_table weight brackets must be positive")
        if any(
            later <= earlier
            for earlier, later in zip(self.weight_brackets_g, self.weight_brackets_g[1:])
        ):
            raise ValueError("rate_table weight brackets must be strictly ascending")
        if any(price < 0 for price in self.prices_minor):
            raise ValueError("rate_table prices cannot be negative")
        if self.minimum_charge_minor < 0:
            raise ValueError("rate_table minimum_charge_minor cannot be negative")
        if self.fuel_surcharge_permille < 0:
            raise ValueError("rate_table fuel_surcharge_permille cannot be negative")

    def charge_minor_or_none(self, billed_weight_g: int) -> int | None:
        """The exact landed cost of one shipment at this billed weight, or `None` when the
        tariff does not price it.

        The search needs to *compare* an unpriceable candidate rather than abort on one:
        a container whose tariff runs out at this weight must lose to one that can price
        the load, which it cannot do if asking the question raises. `charge_minor` is the
        same walk for callers who want the refusal.
        """
        for bound, price in zip(self.weight_brackets_g, self.prices_minor):
            if billed_weight_g <= bound:
                base = max(price, self.minimum_charge_minor)
                # Ceiling division, never a float: the surcharge is a share of the base,
                # and rounding down would let a fractional unit of revenue vanish.
                surcharge = -(-base * self.fuel_surcharge_permille // 1000)
                return base + surcharge
        return None

    def charge_minor(self, billed_weight_g: int) -> int:
        """The exact landed cost of one shipment at this billed weight.

        Raises `UnratedWeightError` above the last bracket rather than clamping to the top
        price. Clamping would quietly under-price every oversize shipment and, worse, make
        the objective prefer a packing the caller cannot actually ship at that price.
        """
        charge = self.charge_minor_or_none(billed_weight_g)
        if charge is None:
            raise UnratedWeightError(
                f"billed weight {billed_weight_g} g is above the rate table's last bracket "
                f"({self.weight_brackets_g[-1]} g); the shipment has no published price"
            )
        return charge


class UnratedWeightError(ValueError):
    """A billed weight the rate table does not price. Structured rather than a silent
    top-bracket clamp, so a caller learns the tariff is short rather than being quoted a
    number the carrier never published."""


@dataclass(frozen=True, slots=True)
class Container:
    id: str
    inner_dimensions: Dimensions
    outer_dimensions: Dimensions | None = None
    tare_weight: Weight = Weight(0)
    max_payload: Weight | None = None
    cost_minor: int = 0
    rate_table: RateTable | None = None
    quantity: int | None = None
    obstacles: tuple[Obstacle, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    max_items: int | None = None
    void_fill_reserve_ratio: float = 0.0
    tag_limits: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Weight per square metre of base area, not a total: floor loading.
    max_stack_density: Weight | None = None
    # (front, rear), front nearer the container's own x origin.
    axles: tuple[Axle, Axle] | None = None

    def __post_init__(self) -> None:
        if not self.id: raise ValueError("container id is required")
        if self.quantity is not None and self.quantity <= 0: raise ValueError("container quantity must be positive")
        if self.cost_minor < 0: raise ValueError("container cost cannot be negative")
        if not 0 <= self.void_fill_reserve_ratio <= 1: raise ValueError("void_fill_reserve_ratio must be between 0 and 1")
        if self.axles is not None:
            front, rear = self.axles
            if front.position.ticks >= rear.position.ticks:
                raise ValueError("the front axle must be strictly nearer the origin than the rear axle")
            if front.position.ticks < 0 or rear.position.ticks > self.inner_dimensions.length.ticks:
                raise ValueError("axle positions must lie within the container's length")
        if any(limit < 1 for limit in self.tag_limits.values()): raise ValueError("tag_limits must be at least 1")
        if self.outer_dimensions and not self.inner_dimensions.fits_inside(self.outer_dimensions): raise ValueError("outer dimensions cannot be smaller than inner dimensions")
        boundary = AxisAlignedBox(Point(0, 0, 0), self.inner_dimensions)
        for obstacle in self.obstacles:
            if not all(boundary.contains(box) for box in obstacle.boxes):
                raise ValueError(f"obstacle {obstacle.id} lies outside container")
        object.__setattr__(self, "tags", frozenset(self.tags))
        if self.max_stack_density is not None:
            object.__setattr__(self, "max_stack_density", Weight.parse(self.max_stack_density))

    @classmethod
    def create(cls, id: str, inner_dimensions: Dimensions, tare_weight=0, max_payload=None, **kwargs) -> "Container":
        return cls(id=id, inner_dimensions=inner_dimensions, tare_weight=Weight.parse(tare_weight), max_payload=None if max_payload is None else Weight.parse(max_payload), **kwargs)


@dataclass(frozen=True, slots=True)
class Placement:
    instance: ItemInstance
    position: Point
    rotation: Rotation
    dimensions: Dimensions
    envelope_origin: Point
    envelope_dimensions: Dimensions
    support_ratio: float = 1.0
    top_load: Weight = Weight(0)

    @property
    def box(self) -> AxisAlignedBox: return AxisAlignedBox(self.position, self.dimensions)
    @property
    def envelope_box(self) -> AxisAlignedBox: return AxisAlignedBox(self.envelope_origin, self.envelope_dimensions)


@dataclass(frozen=True, slots=True)
class PackedContainer:
    container: Container
    sequence: int
    placements: tuple[Placement, ...]
    # Populated instead of `placements` (which stays `()`) when `GridSolver` took the
    # quantity-compression fast path (`configuration.require_placement_coordinates
    # = False`): the lattice parameters plus how many instances were placed, in O(1)
    # additional space rather than one `Placement` object per instance.
    lattice_summary: LatticeSummary | None = None
    # The specific instances the lattice fast path consumed, in placement order --
    # needed to reconstruct identical `Placement` objects via `expand_placements()`.
    # A reference slice of already-existing `ItemInstance` objects, not new domain
    # objects, so retaining it is far cheaper than the `Placement` construction it
    # replaces, even though its length is still `len(lattice_summary)`.
    lattice_items: tuple["ItemInstance", ...] = ()

    @property
    def id(self) -> str: return f"{self.container.id}#{self.sequence}"
    @property
    def placement_count(self) -> int:
        return self.lattice_summary.count if self.lattice_summary is not None else len(self.placements)
    @property
    def payload_weight(self) -> Weight:
        if self.lattice_summary is not None: return Weight(self.lattice_summary.total_weight_ticks)
        return Weight(sum(p.instance.weight.ticks for p in self.placements))
    @property
    def gross_weight(self) -> Weight: return Weight(self.container.tare_weight.ticks + self.payload_weight.ticks)
    @property
    def used_volume(self) -> int:
        if self.lattice_summary is not None: return self.lattice_summary.used_volume_ticks
        return nesting_used_volume(self.placements)
    @property
    def utilization(self) -> float: return self.used_volume / self.container.inner_dimensions.volume
    @property
    def max_z_ticks(self) -> int:
        if self.lattice_summary is not None: return self.lattice_summary.max_z_ticks
        return max((p.envelope_box.z2 for p in self.placements), default=0)
    @property
    def centre_of_mass_offset_ppm(self) -> int:
        if self.lattice_summary is not None:
            inner = self.container.inner_dimensions
            return self.lattice_summary.centre_of_mass_offset_ppm(inner.length.ticks, inner.width.ticks)
        return centre_of_mass_offset_ppm(self.container.inner_dimensions, self.placements)

    def expand_placements(self) -> tuple[Placement, ...]:
        """The full per-item `Placement` tuple, reconstructed from `lattice_summary`
        if the compact fast path built this container, otherwise `placements`
        unchanged. Reconstruction is exact: same order, same coordinates, same
        rotation as the O(n) loop this fast path replaces."""
        if self.lattice_summary is not None:
            return self.lattice_summary.expand(self.lattice_items)
        return self.placements

    def as_item(self, id: str | None = None, keep_upright: bool = True, max_top_load: Weight | None = None) -> Item:
        return Item(id or self.id, self.container.outer_dimensions or self.container.inner_dimensions, self.gross_weight, keep_upright=keep_upright, max_top_load=max_top_load, metadata={"source_packed_container": self.id})


@dataclass(frozen=True, slots=True)
class RejectionObservation:
    code: str
    count: int = 1
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"code": self.code, "count": self.count, "details": list(self.details)}


@dataclass(frozen=True, slots=True)
class ReasonProof:
    level: str
    observations: tuple[RejectionObservation, ...]

    @classmethod
    def for_reason(cls, reason: str, details: tuple[str, ...] = ()) -> "ReasonProof":
        # `policy_rule` is proven, not inferred: it is only ever reported when a rule
        # rules the item out of *every* offered container, which is a statement about
        # the request that no search outcome can change.
        if reason in {"no_compatible_container_dimensions", "payload_exceeded", "no_eligible_container",
                      "rotation_restricted", "policy_rule"}:
            level = "proven"
        elif reason in {"time_limit", "effort_limit"}:
            level = "unknown_due_to_limit"
        elif reason in {"no_feasible_placement", "search_exhausted", "exact_search_incomplete",
                        "insufficient_support"}:
            level = "observed"
        else:
            level = "inferred"
        return cls(level, (RejectionObservation(reason, 1, details),))

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "observations": [observation.to_dict() for observation in self.observations],
        }


@dataclass(frozen=True, slots=True)
class UnpackedItem:
    instance: ItemInstance
    reason: str
    details: tuple[str, ...] = ()
    proof: ReasonProof | None = None

    def __post_init__(self) -> None:
        if self.proof is None:
            object.__setattr__(self, "proof", ReasonProof.for_reason(self.reason, self.details))


@dataclass(frozen=True, slots=True)
class PackingRequest:
    items: tuple[Item, ...]
    containers: tuple[Container, ...]

    def __post_init__(self) -> None:
        if not self.items: raise ValueError("at least one item is required")
        if not self.containers: raise ValueError("at least one container is required")
        if len({item.id for item in self.items}) != len(self.items): raise ValueError("item ids must be unique")
        if len({container.id for container in self.containers}) != len(self.containers): raise ValueError("container ids must be unique")

    @property
    def instances(self) -> tuple[ItemInstance, ...]:
        return tuple(instance for item in self.items for instance in item.instances())
