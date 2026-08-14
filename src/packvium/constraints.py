from __future__ import annotations

from ._compat import dataclass
from typing import Protocol, Sequence

from .axle_load import axle_load_exceeded
from .contact import ContactEdge, ContactGraph
from .geometry import AxisAlignedBox, Dimensions, Point, Rotation
from .models import Container, ItemInstance, Placement
from .support_polygon import contact_hull_points, convex_hull, doubled_centroid, point_in_hull
from .units import Length

# Support ratios arrive as floats from the public API but must never decide feasibility
# in floating point. They are converted once to a scaled integer and every comparison
# below is exact. See docs/UNITS-AND-NUMERICS.md.
SUPPORT_SCALE = 1_000_000


def scaled_ratio(ratio: float) -> int:
    return int(ratio * SUPPORT_SCALE + 0.5)


def required_area(base_area: int, scaled: int) -> int:
    """Exact floor(base_area * scaled / SUPPORT_SCALE) without an overflowing product."""
    whole, remainder = divmod(base_area, SUPPORT_SCALE)
    return whole * scaled + (remainder * scaled) // SUPPORT_SCALE


def reserved_volume(container: Container) -> int:
    """Volume set aside for packing material, exact even for a metre-scale container.

    Not floating-point `ratio * volume`: that is exactly the double-precision loss
    this codebase's fixed-point design exists to avoid for a feasibility number.
    """
    return required_area(container.inner_dimensions.volume, scaled_ratio(container.void_fill_reserve_ratio))


def usable_volume(container: Container) -> int:
    return container.inner_dimensions.volume - reserved_volume(container)


@dataclass(frozen=True, slots=True)
class ConstraintContext:
    container: Container
    placements: tuple[Placement, ...]
    item: ItemInstance
    point: Point
    rotation: Rotation
    dimensions: Dimensions
    envelope_dimensions: Dimensions
    # False only when nothing in the container can refuse or be crushed by a load, which
    # lets the bearing check skip a per-candidate walk of the whole stack.
    stack_sensitive: bool = True
    # False when no item in play declares a `stop_index`, which is every request that is
    # not a multi-stop route -- the same opt-in shape as `stack_sensitive`.
    route_sensitive: bool = True

    @property
    def envelope_box(self) -> AxisAlignedBox:
        return AxisAlignedBox(self.point, self.envelope_dimensions)


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    allowed: bool
    code: str = "allowed"
    detail: str = ""

    @classmethod
    def allow(cls) -> "ConstraintResult": return _ALLOWED
    @classmethod
    def reject(cls, code: str, detail: str = "") -> "ConstraintResult": return cls(False, code, detail)


_ALLOWED = ConstraintResult(True)


class PlacementConstraint(Protocol):
    def evaluate(self, context: ConstraintContext) -> ConstraintResult: ...


class ProvenRejection(Protocol):
    """A constraint that can rule an item out of *every* offered container without
    searching for a placement first.

    `evaluate` answers about one candidate, which is never enough to say why an item is
    unpacked: a rejected candidate only means *this* position failed. A constraint that
    implements this can say more -- that no position in any offered container could have
    worked -- and that is what lets the unpacked reason be reported as `proven` and name
    the rule responsible, rather than falling through to the generic observed reason
.

    Returns a `(reason_code, detail)` pair, or `None` when the constraint cannot prove
    anything about this item. Proving nothing is the correct answer for most rules: a
    segregation or per-container cap depends on what else was packed, so whether it
    leaves an item behind is a property of the search, not of the request.
    """

    def proves_unplaceable(
        self, item: ItemInstance, containers: Sequence[Container]
    ) -> tuple[str, str] | None: ...


@dataclass(frozen=True, slots=True)
class LoadUnit:
    """The only facts load propagation needs about a box, real or hypothetical."""
    box: AxisAlignedBox
    weight_ticks: int
    max_top_load_ticks: int | None
    max_stacked_items: int | None
    label: str
    nesting_item_id: str | None = None
    nesting_height_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class DirectSupport:
    index: int
    box: AxisAlignedBox
    area: int


@dataclass(frozen=True, slots=True)
class DirectSupportView:
    """Candidate support surfaces and positive-area direct supporters."""
    surfaces: tuple[AxisAlignedBox, ...]
    supporters: tuple[DirectSupport, ...]

    @property
    def supporting_area(self) -> int:
        return sum(support.area for support in self.supporters)


def _same_nesting_column(placement: Placement, item: ItemInstance, box: AxisAlignedBox) -> bool:
    other = placement.envelope_box
    return (
        placement.instance.item.id == item.item.id
        and placement.instance.item.nesting_height is not None
        and item.item.nesting_height is not None
        and placement.instance.item.nesting_height.ticks == item.item.nesting_height.ticks
        and (other.origin.x, other.origin.y, other.x2, other.y2)
        == (box.origin.x, box.origin.y, box.x2, box.y2)
    )


def direct_support_view(
    placements: Sequence[Placement], item: ItemInstance, box: AxisAlignedBox
) -> DirectSupportView:
    """Direct supports for one active candidate in O(m) time and O(m) output space.

    Ordinary face semantics stay byte-for-byte equivalent: every box whose top is on
    the candidate's base plane remains a ``surface``, while only positive-area overlaps
    are supporters. If an exact same-type/same-footprint nesting predecessor exists,
    shadowed face surfaces from that nesting column are removed and the immediate
    predecessor becomes one full-area supporter. Lists remain in placement-index order.
    """
    face_surfaces: list[tuple[int, AxisAlignedBox]] = []
    face_supporters: list[DirectSupport] = []
    nearest_lower: tuple[int, AxisAlignedBox] | None = None
    nesting = item.item.nesting_height
    for index, placement in enumerate(placements):
        other = placement.envelope_box
        other_nesting = placement.instance.item.nesting_height
        if other.z2 == box.origin.z:
            face_surfaces.append((index, other))
            area = other.overlap_area_xy(box)
            if area > 0:
                face_supporters.append(DirectSupport(index, other, area))
        if (
            nesting is not None
            and other_nesting is not None
            and _same_nesting_column(placement, item, box)
            and other.origin.z < box.origin.z
            and (nearest_lower is None or other.origin.z > nearest_lower[1].origin.z)
        ):
            nearest_lower = (index, other)
    predecessor = (
        nearest_lower
        if nearest_lower is not None
        and nesting is not None
        and nearest_lower[1].z2 - box.origin.z == nesting.ticks
        else None
    )
    if predecessor is None:
        return DirectSupportView(
            tuple(surface for _, surface in face_surfaces), tuple(face_supporters)
        )

    surfaces = [
        entry for entry in face_surfaces
        if not _same_nesting_column(placements[entry[0]], item, box)
    ]
    supporters = [
        support for support in face_supporters
        if not _same_nesting_column(placements[support.index], item, box)
    ]
    predecessor_index, predecessor_box = predecessor
    surface_at = 0
    while surface_at < len(surfaces) and surfaces[surface_at][0] < predecessor_index:
        surface_at += 1
    surfaces.insert(surface_at, (predecessor_index, predecessor_box))
    supporter_at = 0
    while supporter_at < len(supporters) and supporters[supporter_at].index < predecessor_index:
        supporter_at += 1
    supporters.insert(
        supporter_at,
        DirectSupport(predecessor_index, predecessor_box, predecessor_box.overlap_area_xy(box)),
    )
    return DirectSupportView(
        tuple(surface for _, surface in surfaces), tuple(supporters)
    )


class LoadSupportGraph:
    """Face contact plus exact direct nesting support for load-bearing rules.

    Nesting candidates are sorted by declared item type, exact XY footprint and height,
    so only adjacent layers of one potential column are compared. Sorting is O(n log n)
    and scanning/rebuilding the existing contact edges is O(E), not an all-pairs scan.
    Once an adjacent nesting predecessor exists, all same-column face edges are replaced
    by that predecessor alone. Final edge and child lists retain ascending unit-index
    order, preserving ContactGraph's integer-remainder and traversal contract.
    """

    __slots__ = ("_supporters", "_children")

    def __init__(self, units: Sequence[LoadUnit]):
        face = ContactGraph([unit.box for unit in units])
        nesting = sorted(
            (
                unit.nesting_item_id,
                unit.nesting_height_ticks,
                unit.box.origin.x,
                unit.box.origin.y,
                unit.box.x2,
                unit.box.y2,
                unit.box.origin.z,
                unit.box.z2,
                index,
            )
            for index, unit in enumerate(units)
            if unit.nesting_item_id is not None and unit.nesting_height_ticks is not None
        )
        if not nesting:
            self._supporters = tuple(face.supporters(index) for index in range(len(units)))
            self._children = tuple(face.children(index) for index in range(len(units)))
            return
        supporters = [list(face.supporters(index)) for index in range(len(units))]
        for lower, upper in zip(nesting, nesting[1:]):
            if lower[:6] != upper[:6] or lower[6] >= upper[6]:
                continue
            lower_index, upper_index = lower[8], upper[8]
            nesting_depth = units[lower_index].nesting_height_ticks
            if units[lower_index].box.z2 - units[upper_index].box.origin.z != nesting_depth:
                continue
            retained: list[ContactEdge] = []
            for edge in supporters[upper_index]:
                supporter = units[edge.index]
                supporter_column = (
                    supporter.nesting_item_id,
                    supporter.nesting_height_ticks,
                    supporter.box.origin.x,
                    supporter.box.origin.y,
                    supporter.box.x2,
                    supporter.box.y2,
                )
                if supporter_column != upper[:6]:
                    retained.append(edge)
            area = units[lower_index].box.overlap_area_xy(units[upper_index].box)
            insert_at = 0
            while insert_at < len(retained) and retained[insert_at].index < lower_index:
                insert_at += 1
            retained.insert(insert_at, ContactEdge(lower_index, area))
            supporters[upper_index] = retained
        children: list[list[int]] = [[] for _ in units]
        for child_index, edges in enumerate(supporters):
            for edge in edges:
                children[edge.index].append(child_index)
        self._supporters = tuple(tuple(edges) for edges in supporters)
        self._children = tuple(tuple(indices) for indices in children)

    def supporters(self, index: int) -> tuple[ContactEdge, ...]:
        return self._supporters[index]

    def children(self, index: int) -> tuple[int, ...]:
        return self._children[index]


def non_stackable_failure(
    placements: Sequence[Placement], item: ItemInstance,
    graph: LoadSupportGraph, candidate_index: int,
) -> ConstraintResult | None:
    """A direct-support edge may never originate at a non-stackable item."""
    for edge in graph.supporters(candidate_index):
        supporter = placements[edge.index]
        if not supporter.instance.item.stackable:
            return ConstraintResult.reject("non_stackable", supporter.instance.item.id)
    if not item.item.stackable and graph.children(candidate_index):
        return ConstraintResult.reject("non_stackable", item.item.id)
    return None


def load_units(placements: Sequence[Placement], extra: LoadUnit | None = None) -> tuple[LoadUnit, ...]:
    units = [
        LoadUnit(p.envelope_box, p.instance.weight.ticks,
                 None if p.instance.item.max_top_load is None else p.instance.item.max_top_load.ticks,
                 p.instance.item.max_stacked_items,
                 p.instance.id,
                 None if p.instance.item.nesting_height is None else p.instance.item.id,
                 None if p.instance.item.nesting_height is None else p.instance.item.nesting_height.ticks)
        for p in placements
    ]
    if extra is not None: units.append(extra)
    return tuple(units)


def top_loads(units: Sequence[LoadUnit], graph: LoadSupportGraph | None = None) -> list[int]:
    """Weight borne by each unit, propagated down the whole stack.

    Every box pushes its own weight plus everything already resting on it onto its
    direct face or nesting supports, split by contact area. The integer remainder goes
    to the last supporter so the distributed total is conserved exactly.
    """
    graph = graph if graph is not None else LoadSupportGraph(units)
    loads = [0] * len(units)
    descending = sorted(range(len(units)), key=lambda i: (-units[i].box.z2, -units[i].box.origin.z, i))
    for upper_index in descending:
        supports = graph.supporters(upper_index)
        total_area = sum(edge.area for edge in supports)
        if total_area == 0: continue
        downward = units[upper_index].weight_ticks + loads[upper_index]
        assigned = 0
        for position, edge in enumerate(supports):
            share = downward - assigned if position == len(supports) - 1 else downward * edge.area // total_area
            assigned += share
            loads[edge.index] += share
    return loads


def overloaded(units: Sequence[LoadUnit], graph: LoadSupportGraph | None = None) -> tuple[str, str] | None:
    """First unit whose bearing limit is exceeded, as (code, detail)."""
    if all(unit.max_top_load_ticks is None for unit in units): return None
    for unit, load in zip(units, top_loads(units, graph)):
        if unit.max_top_load_ticks is not None and load > unit.max_top_load_ticks:
            return ("top_load_exceeded", unit.label)
    return None


def resting_above(units: Sequence[LoadUnit], graph: LoadSupportGraph | None = None) -> list[frozenset[int]]:
    """Everything resting anywhere above each unit, following the support graph upward.

    An item stacked through two levels of intermediaries appears in the set of every
    level beneath it, not merely the one it directly touches.
    """
    graph = graph if graph is not None else LoadSupportGraph(units)
    memo: dict[int, frozenset[int]] = {}

    def above_set(index: int) -> frozenset[int]:
        cached = memo.get(index)
        if cached is not None: return cached
        result: set[int] = set()
        for child in graph.children(index):
            result.add(child)
            result |= above_set(child)
        memo[index] = frozenset(result)
        return memo[index]

    return [above_set(index) for index in range(len(units))]


def stacked_counts(units: Sequence[LoadUnit], graph: LoadSupportGraph | None = None) -> list[int]:
    """Items resting anywhere above each unit. A count, not a weight."""
    return [len(above) for above in resting_above(units, graph)]


#: A placement with no `stop_index` is never scheduled for removal and rides the whole
#: route, so nothing can be blocking it -- but it does block everything beneath it.
#: Ordering it after every real stop expresses both halves at once.
RIDES_THE_WHOLE_ROUTE = float("inf")


def route_order_violated(units: Sequence[LoadUnit], stops: Sequence[float]) -> tuple[str, str] | None:
    """First item due at an earlier stop with a later-stop item resting above it.

    docs/VALIDATION-CONTRACT.md requires every item due at a stop to be removable
    before the route moves on to the next one. Vertical blocking is the half a
    placement decision can see by itself: an item bound for a later stop, resting
    anywhere above an earlier-stop one, has to come off first, and nothing happening at
    the earlier stop can move it. The other half -- whether a box that is unblocked can
    also slide out of some wall -- is a whole-container reachability question the
    post-validator answers; this check is the necessary condition, not the whole of it.
    """
    if all(stop == RIDES_THE_WHOLE_ROUTE for stop in stops): return None
    for index, above in enumerate(resting_above(units)):
        for upper in above:
            if stops[upper] > stops[index]:
                return ("unloading_order_violation", f"{units[index].label} blocked by {units[upper].label}")
    return None


def stack_limit_exceeded(units: Sequence[LoadUnit], graph: LoadSupportGraph | None = None) -> tuple[str, str] | None:
    """First unit whose stacked-item limit is exceeded, as (code, detail)."""
    if all(unit.max_stacked_items is None for unit in units): return None
    for unit, count in zip(units, stacked_counts(units, graph)):
        if unit.max_stacked_items is not None and count > unit.max_stacked_items:
            return ("stacked_item_limit_exceeded", unit.label)
    return None


# One square metre in length ticks squared, the reference area a stack density limit
# is expressed per (e.g. "1500 kg/m^2 floor loading"), derived from the single tick
# scale rather than hard-coded so it can never drift from it.
SQUARE_METRE_TICKS = Length.mm(1000).ticks ** 2


def stack_density_exceeded(
    units: Sequence[LoadUnit], max_density_ticks: int | None, graph: LoadSupportGraph | None = None
) -> tuple[str, str] | None:
    """First unit whose cumulative load crushes its own footprint, as (code, detail).

    `max_density_ticks` is weight ticks allowed per square metre of a unit's own base
    area (`Container.max_stack_density`), checked at every level of the stack, not
    only the floor: a wide item narrowing to a small waist part-way up is exactly the
    case a flat per-item `max_top_load` cannot express, since the same absolute load
    becomes crushing once concentrated onto less area. Cross-multiplied rather than
    divided, so no fractional precision is lost and nothing overflows in languages
    without arbitrary-precision integers.
    """
    if max_density_ticks is None: return None
    for unit, load_from_above in zip(units, top_loads(units, graph)):
        # `top_loads` gives only what rests *on* a unit, the right meaning for a flat
        # `max_top_load`. Floor loading is about everything bearing down *through* a
        # unit's own footprint, which includes the unit's own weight too.
        total = unit.weight_ticks + load_from_above
        if total * SQUARE_METRE_TICKS > max_density_ticks * unit.box.dimensions.base_area:
            return ("stack_density_exceeded", unit.label)
    return None


class FloorConstraint:
    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if context.item.item.must_be_on_floor and context.point.z != 0:
            return ConstraintResult.reject("must_be_on_floor")
        return ConstraintResult.allow()


class ContainerEligibilityConstraint:
    """Rejects a placement into a container the item did not opt into.

    An item with no `eligible_container_tags` may go anywhere, matching the same
    "empty means unconstrained" convention as `Item.tags`/`incompatible_tags`.
    """

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        item = context.item.item
        if not item.eligible_container_tags:
            return ConstraintResult.allow()
        if item.eligible_container_tags & context.container.tags:
            return ConstraintResult.allow()
        return ConstraintResult.reject("container_ineligible", context.container.id)


class CompatibilityConstraint:
    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        item = context.item.item
        if not item.tags and not item.incompatible_tags:
            return ConstraintResult.allow()
        for placement in context.placements:
            other = placement.instance.item
            if item.incompatible_tags & other.tags or other.incompatible_tags & item.tags:
                return ConstraintResult.reject("incompatible_items", f"{item.id} is incompatible with {other.id}")
        return ConstraintResult.allow()


class TagCountConstraint:
    """Caps how many items carrying a tag may share one container.

    A group is atomic but cannot express a count: "at most two of this class per
    container" is a policy about the container, not about any one item's neighbours,
    so it lives on `Container.tag_limits` rather than on the item.
    """

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        limits = context.container.tag_limits
        if not limits:
            return ConstraintResult.allow()
        relevant = context.item.item.tags & limits.keys()
        if not relevant:
            return ConstraintResult.allow()
        for tag in sorted(relevant):
            limit = limits[tag]
            count = sum(1 for p in context.placements if tag in p.instance.item.tags)
            if count + 1 > limit:
                return ConstraintResult.reject("tag_count_exceeded", f"{tag}: limit {limit}, would be {count + 1}")
        return ConstraintResult.allow()


def _touches_corners(candidate, surfaces) -> bool:
    """Whether every one of the candidate's four base corners rests on some surface.

    A ratio cannot express this: 90% of the footprint concentrated in the middle
    passes any ratio below that but leaves every corner hanging.
    """
    corners = ((candidate.origin.x, candidate.origin.y), (candidate.x2, candidate.origin.y),
              (candidate.origin.x, candidate.y2), (candidate.x2, candidate.y2))
    return all(
        any(surface.origin.x <= x <= surface.x2 and surface.origin.y <= y <= surface.y2 for surface in surfaces)
        for x, y in corners
    )


@dataclass(frozen=True, slots=True)
class SupportConstraint:
    global_minimum: float = 0.0

    def applies_to(self, item: ItemInstance) -> bool:
        """Whether this constraint can reject an above-floor placement of ``item``."""
        rule = item.item.ground_contact_rule
        required = max(self.global_minimum, item.item.minimum_support_ratio)
        return rule not in (None, "free") or required > 0

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if context.point.z == 0 or not self.applies_to(context.item):
            return ConstraintResult.allow()
        candidate = context.envelope_box
        support = direct_support_view(context.placements, context.item, candidate)
        surfaces = support.surfaces
        supporters = [entry.box for entry in support.supporters]
        rule = context.item.item.ground_contact_rule
        if rule is not None and rule != "free":
            if rule == "covered" and not _touches_corners(candidate, surfaces):
                return ConstraintResult.reject("ground_contact_violation", f"covered: {len(supporters)} supporter(s), not all four corners touched")
            if rule == "single" and len(supporters) != 1:
                return ConstraintResult.reject("ground_contact_violation", f"single: rests on {len(supporters)} item(s)")
            if rule == "multiple" and len(supporters) < 2:
                return ConstraintResult.reject("ground_contact_violation", f"multiple: rests on {len(supporters)} item(s)")
        required = max(self.global_minimum, context.item.item.minimum_support_ratio)
        if required <= 0:
            return ConstraintResult.allow()
        supporting_area = support.supporting_area
        base_area = context.envelope_dimensions.base_area
        if supporting_area < required_area(base_area, scaled_ratio(required)):
            return ConstraintResult.reject("insufficient_support", f"{supporting_area}/{base_area} < {required:.6f}")
        # Area alone is not stability: a candidate can clear the ratio while
        # overhanging its own centre of gravity, which is exactly what the ratio check
        # above cannot see. Checked only once a minimum ratio is already being
        # enforced, so a caller who never asked for support checking sees no new
        # rejection code and no behaviour change.
        hull = convex_hull(contact_hull_points(candidate, supporters))
        if not point_in_hull(doubled_centroid(candidate), hull):
            return ConstraintResult.reject("centre_of_gravity_unsupported", f"{supporting_area}/{base_area} met but centroid outside the {len(hull)}-point support hull")
        return ConstraintResult.allow()


class TopLoadConstraint:
    """Rejects a placement that would rest on something unable to carry it.

    Bearing limits are checked against the cumulative load of the whole stack, not
    only the box directly underneath, so a tower of light items cannot crush its base.
    """

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if not context.stack_sensitive: return ConstraintResult.allow()
        candidate = context.envelope_box
        item = context.item.item
        units = load_units(
            context.placements,
            LoadUnit(candidate, context.item.weight.ticks,
                     None if item.max_top_load is None else item.max_top_load.ticks,
                     item.max_stacked_items, context.item.id,
                     None if item.nesting_height is None else item.id,
                     None if item.nesting_height is None else item.nesting_height.ticks),
        )
        graph = LoadSupportGraph(units)
        failure = non_stackable_failure(
            context.placements, context.item, graph, len(units) - 1
        )
        if failure is not None:
            return failure
        density_limit = None if context.container.max_stack_density is None else context.container.max_stack_density.ticks
        failure = (
            overloaded(units, graph)
            or stack_limit_exceeded(units, graph)
            or stack_density_exceeded(units, density_limit, graph)
        )
        if failure is not None:
            return ConstraintResult.reject(*failure)
        return ConstraintResult.allow()


class RouteOrderConstraint:
    """Rejects a placement that would bury an item due off at an earlier stop.

    A no-op unless some item declares a `stop_index`: route enforcement is opt-in,
    matching every other optional rule in this module, so a request that never mentions
    a route sees no new rejection code and no behaviour change.
    """

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if not context.route_sensitive: return ConstraintResult.allow()
        units = load_units(
            context.placements,
            LoadUnit(
                context.envelope_box, context.item.weight.ticks, None, None, context.item.id,
                None if context.item.item.nesting_height is None else context.item.item.id,
                None if context.item.item.nesting_height is None else context.item.item.nesting_height.ticks,
            ),
        )
        stops = [
            RIDES_THE_WHOLE_ROUTE if p.instance.item.stop_index is None else float(p.instance.item.stop_index)
            for p in context.placements
        ]
        item_stop = context.item.item.stop_index
        stops.append(RIDES_THE_WHOLE_ROUTE if item_stop is None else float(item_stop))
        failure = route_order_violated(units, stops)
        if failure is not None:
            return ConstraintResult.reject(*failure)
        return ConstraintResult.allow()


class AxleLoadConstraint:
    """Rejects a placement that would push either of a two-axle container's axles
    over its own limit.

    A no-op unless `Container.axles` is set: axle enforcement is opt-in, matching
    every other whole-container weight rule in this module, so a caller who never
    configured axles sees no new rejection code and no behaviour change.
    """

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if context.container.axles is None:
            return ConstraintResult.allow()
        candidate = context.envelope_box
        nesting = context.item.item.nesting_height
        units = load_units(context.placements, LoadUnit(
            candidate, context.item.weight.ticks, None, None, context.item.id,
            None if nesting is None else context.item.item.id,
            None if nesting is None else nesting.ticks,
        ))
        failure = axle_load_exceeded(
            context.container.axles,
            units,
            context.container.tare_weight.ticks,
            context.container.inner_dimensions.length.ticks,
        )
        if failure is not None:
            return ConstraintResult.reject(*failure)
        return ConstraintResult.allow()
