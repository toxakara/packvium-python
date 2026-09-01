from __future__ import annotations

from dataclasses import field

from ._compat import dataclass
from typing import Any, Iterable, Mapping

from .centre_of_mass import centre_of_mass_offset_ppm
from . import compression, hull
from .geometry import AxisAlignedBox, Dimensions, Point, Rotation, ShapeType
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


#: The largest `stop_index` every engine can carry identically (/KI defect found
#: under ). Route order is decided by comparing stop indices, and JavaScript holds
#: numbers as doubles: `JSON.parse` already collapses 2**53 + 1 to 2**53 before any
#: constraint sees it, so two consecutive stops above this bound become one number there
#: and one engine silently disagrees with the other three. Since the value cannot cross
#: the wire identically, it is refused rather than accepted and mis-ordered -- the same
#: choice `InvalidDirectionError` makes for a direction outside its six.
MAX_EXACT_STOP_INDEX = 2 ** 53 - 1


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
    # How much of `dimensions` this item actually occupies. The default is the
    # contract as it stood before this epic -- the item is its box -- and the three fields
    # below are the data the two narrower shapes need. Each belongs to exactly one shape;
    # setting one against the wrong shape is refused rather than ignored, because a
    # `compression_ratio` silently dropped on a `convex_hull` reads as an item that was
    # packed to its declared limits when it never was.
    shape_type: "ShapeType" = ShapeType.RIGID_CUBOID
    hull_vertices: tuple[tuple[int, int, int], ...] | None = None
    compression_ratio_ppm: int | None = None
    max_compression_pressure_kpa: int | None = None
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
        if self.stop_index is not None and not 0 <= self.stop_index <= MAX_EXACT_STOP_INDEX:
            raise ValueError("stop_index must be a non-negative safe integer")
        if self.value is not None and self.value < 0:
            raise ValueError("value must be non-negative")
        if self.nesting_height is not None and not 0 <= self.nesting_height.ticks < self.dimensions.height.ticks:
            raise ValueError("nesting_height must be at least zero and strictly less than the item's own height")
        if self.ground_contact_rule is not None and self.ground_contact_rule not in GROUND_CONTACT_RULES:
            raise ValueError(f"ground_contact_rule must be one of {sorted(GROUND_CONTACT_RULES)}")
        self._validate_shape()
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

    def _validate_shape(self) -> None:
        """Admit an item's shape, or refuse it with the reason.

        Kept out of `__post_init__` because it is the only rule here that spans four fields
        at once: which of them are required, which are forbidden, and what the survivors
        have to be consistent with.
        """
        object.__setattr__(self, "shape_type", ShapeType(self.shape_type))
        for name, value in self._fields_foreign_to_shape():
            if value is not None:
                raise ValueError(f"{name} is not part of a {self.shape_type.value} item")
        if self.nesting_height is not None and self.shape_type is not ShapeType.RIGID_CUBOID:
            # Both rewrite occupied height. Picking an order silently would give four
            # engines four contracts; the interaction gets its own task before it is allowed.
            raise ValueError(
                f"nesting_height with shape_type {self.shape_type.value} is not supported yet"
            )
        if self.shape_type is ShapeType.CONVEX_HULL:
            self._validate_hull()
        elif self.shape_type is ShapeType.COMPRESSIBLE:
            self._validate_compression()

    def _fields_foreign_to_shape(self) -> tuple[tuple[str, Any], ...]:
        compression = (
            ("compression_ratio_ppm", self.compression_ratio_ppm),
            ("max_compression_pressure_kpa", self.max_compression_pressure_kpa),
        )
        vertices = (("hull_vertices", self.hull_vertices),)
        if self.shape_type is ShapeType.CONVEX_HULL:
            return compression
        if self.shape_type is ShapeType.COMPRESSIBLE:
            return vertices
        return vertices + compression

    def _validate_hull(self) -> None:
        if self.hull_vertices is None:
            raise ValueError("a convex_hull item requires hull_vertices")
        object.__setattr__(self, "hull_vertices", hull.validate(self.hull_vertices))
        lower, upper = hull.bounding_extent(self.hull_vertices)
        extent = tuple(high - low for high, low in zip(upper, lower))
        declared = (self.dimensions.length.ticks, self.dimensions.width.ticks,
                    self.dimensions.height.ticks)
        if any(span > limit for span, limit in zip(extent, declared)):
            # `dimensions` stays the broad phase and the candidate-generation envelope, so a
            # hull poking out of it would be tested for collision against space the solver
            # never reserved.
            raise ValueError(
                f"hull_vertices span {extent} does not fit inside dimensions {declared}"
            )

    def _validate_compression(self) -> None:
        if self.compression_ratio_ppm is None or self.max_compression_pressure_kpa is None:
            raise ValueError(
                "a compressible item requires both compression_ratio and "
                "max_compression_pressure_kpa"
            )
        if not 0 <= self.compression_ratio_ppm <= compression.PPM:
            raise ValueError("compression_ratio must be between zero and one")
        if self.max_compression_pressure_kpa < 0:
            raise ValueError("max_compression_pressure_kpa cannot be negative")

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


def is_stack_sensitive(item: "Item") -> bool:
    """Whether what rests on this item can change a verdict.

    The three original reasons are about the item refusing load. The fourth is about the
    item *yielding* to it: a compressible item needs the cumulative mass above it computed
    before its occupied height -- or its crush limit -- means anything, and that mass only
    exists once the support graph is built.
    """
    return (not item.stackable
            or item.max_top_load is not None
            or item.max_stacked_items is not None
            or item.max_compression_pressure_kpa is not None)


def hull_collision_is_exact(item: "Item", envelope_matches_physical: bool) -> bool:
    """Whether this item's collisions may be decided by its hull rather than by its box.

    Three conditions, and each falls back to the box for its own reason:

    * not a `convex_hull` -- there is no hull to be exact about;
    * a clearance has inflated the envelope past the physical box -- a margin around a hull
      is not a hull, and refining here would hand back space the caller asked to keep empty;
    * the item is on a route -- `packing_sequence` reasons about reachability with box sweeps
      only, and a solver that packed hulls tighter than the sequence replay can verify would
      produce arrangements it then reported as unloadable. Better one conservative answer in
      both places than two that disagree. Lifting this needs a hull-aware sweep, which is a
      task of its own rather than a line here.

    Every fallback over-reserves space, which is the only safe direction.
    """
    return (item.shape_type is ShapeType.CONVEX_HULL
            and envelope_matches_physical
            and item.stop_index is None)


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

    @property
    def hull_shape(self) -> "hull.HullShape | None":
        """This placement's rotated hull, or `None` when its box is the honest answer.

        `None` for every `rigid_cuboid`, and also whenever a clearance has inflated the
        envelope past the physical box: a clearance is a margin around whatever the item is,
        and the margin around a hull is not a hull. Falling back to the envelope over-reserves
        space, which is the only safe direction to be wrong in.
        """
        item = self.instance.item
        if not hull_collision_is_exact(item, self.envelope_dimensions == self.dimensions):
            return None
        return hull.shape_for(item.hull_vertices, self.rotation.value)


def placements_collide(left: Placement, right: Placement) -> bool:
    """Do two placed items actually overlap?

    The axis-aligned envelope test is the broad phase and stays mandatory; this only refines
    its answer when a hull is one of the two solids, so a request of ordinary boxes reaches
    the same verdict by the same route it always did. One definition, so the solver, the
    sequence check and the final validation cannot disagree about what "collides" means.
    """
    left_box, right_box = left.envelope_box, right.envelope_box
    if not left_box.intersects(right_box):
        return False
    left_shape, right_shape = left.hull_shape, right.hull_shape
    if left_shape is None and right_shape is None:
        return True
    return hull.collide(
        left_shape if left_shape is not None else _box_shape(left_box),
        (left_box.origin.x, left_box.origin.y, left_box.origin.z),
        right_shape if right_shape is not None else _box_shape(right_box),
        (right_box.origin.x, right_box.origin.y, right_box.origin.z),
    )


def placement_hits_box(placement: Placement, box: AxisAlignedBox) -> bool:
    """Whether a placed item overlaps a plain box -- an obstacle, or any other fixed solid.

    Same two-phase rule as `placements_collide`, with the second solid known to be a cuboid.
    """
    envelope = placement.envelope_box
    if not envelope.intersects(box):
        return False
    shape = placement.hull_shape
    if shape is None:
        return True
    return hull.collide(
        shape, (envelope.origin.x, envelope.origin.y, envelope.origin.z),
        _box_shape(box), (box.origin.x, box.origin.y, box.origin.z),
    )


def _box_shape(box: AxisAlignedBox) -> "hull.HullShape":
    return hull.HullShape.box(box.dimensions.length.ticks, box.dimensions.width.ticks,
                              box.dimensions.height.ticks)


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
