from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import heapq

from ._compat import dataclass
from time import monotonic_ns
from typing import Callable, Iterable, Protocol, Sequence, cast

from .axle_load import axle_balanced_origins
from .config import PackingConfig
from .effort import EffortBudget
# `_grams` rather than a local ceil-div: the round-up from ticks to whole grams is the
# published billing rule, and a second copy of it could drift a bracket.
from .extensions import UNPRICEABLE_MINOR, _grams
from .constraints import (AxleLoadConstraint, CompatibilityConstraint, ConstraintContext,
                          ContainerEligibilityConstraint, FloorConstraint, LoadUnit, PlacementConstraint,
                          RIDES_THE_WHOLE_ROUTE, RouteOrderConstraint,
                          StopAccessibilityConstraint, SupportConstraint,
                          TagCountConstraint, TopLoadConstraint, active_constraints, direct_support_view,
                          load_units, top_loads, usable_volume)
from . import bounds, hull
from .geometry import (AxisAlignedBox, Dimensions, Point, Rotation, ShapeType,
                       dimensional_weight)
from .lattice_summary import LatticeSummary
from .models import (Container, ItemInstance, PackedContainer, Placement, UnpackedItem,
                     hull_collision_is_exact, is_stack_sensitive)
from .nesting import (is_valid_nesting, occupied_volume,
                      used_volume as nesting_used_volume, used_volume_delta)
from .result import SolverMetrics, StartRecord
from .spatial_index import SpatialIndex
from . import trace
from .units import Length, Weight

# A multi-start never gets less than this, otherwise a long start list would hand out
# slices too small for any solver to place a single item.
MINIMUM_SLICE_NS = 1_000_000


class TimeLimitReached(RuntimeError): pass


@dataclass(slots=True)
class SearchStats:
    candidates_evaluated: int = 0
    placements_attempted: int = 0
    candidate_points_considered: int = 0
    collision_checks: int = 0
    support_checks: int = 0
    space_partitions: int = 0
    search_nodes_expanded: int = 0
    # Times the exact hull test overruled an axis-aligned collision. Deliberately
    # absent from `to_metrics`: `algorithm.metrics` is serialised into every result, so a new
    # key there changes the bytes of every existing golden and has to land in all four engines
    # at once. This one stays internal, where tests can prove the refinement actually fired
    # rather than infer it from a placement that might have succeeded anyway.
    hull_refinements: int = 0
    # The request-level lower bound on the objective vector, computed once at the root of an
    # `exact_small` or global-beam solve. Internal for the same reason as
    # `hull_refinements` above, and for one more: reporting a gap to a caller is a new public
    # result field, and this project reserves and rejects such a field before a contract
    # freeze rather than adding it mid-line, following the precedent set for the reserved
    # container field and restated by the objective-bound wave.
    # `None` when no bound was computed -- a non-default objective keys its score vector
    # differently, so a bound compared against it would compare different quantities.
    objective_lower_bound: "tuple[int, ...] | None" = None

    def to_metrics(self) -> SolverMetrics:
        return SolverMetrics(
            candidate_points_considered=self.candidate_points_considered,
            orientations_considered=self.placements_attempted,
            feasible_candidates=self.candidates_evaluated,
            collision_checks=self.collision_checks,
            support_checks=self.support_checks,
            space_partitions=self.space_partitions,
            search_nodes_expanded=self.search_nodes_expanded,
        )


class Deadline:
    """Monotonic budget that can hand bounded sub-budgets to individual multi-starts.

    An attached `EffortBudget` is read against a live `SearchStats`, so `expired`/
    `check()` trip on whichever bound -- time or counted work -- is hit first. Every
    existing call site already polls `expired`/`check()`, so attaching an effort
    budget (via `with_effort`) needs no changes anywhere else; leaving it unattached
    (the default) reproduces the exact prior time-only behaviour.
    """

    __slots__ = ("started", "limit_ns", "_clock", "_effort_budget", "_stats")

    def __init__(
        self,
        limit_ms: int,
        limit_ns: int | None = None,
        *,
        clock: Callable[[], int] = monotonic_ns,
        effort_budget: EffortBudget | None = None,
        stats: "SearchStats | None" = None,
    ):
        self._clock = clock
        self.started = clock()
        self.limit_ns = limit_ms * 1_000_000 if limit_ns is None else limit_ns
        self._effort_budget = effort_budget
        self._stats = stats

    @property
    def elapsed_ms(self) -> int: return int((self._clock() - self.started) / 1_000_000)
    @property
    def remaining_ns(self) -> int: return self.limit_ns - (self._clock() - self.started)
    @property
    def effort_exceeded(self) -> bool:
        return self._effort_budget is not None and self._stats is not None and self._effort_budget.exceeded(self._stats)
    @property
    def expired(self) -> bool: return self.effort_exceeded or self.remaining_ns <= 0

    def slice(self, parts: int) -> "Deadline":
        remaining = self.remaining_ns
        if parts <= 1:
            return Deadline(0, limit_ns=remaining, clock=self._clock, effort_budget=self._effort_budget, stats=self._stats)
        return Deadline(
            0,
            limit_ns=max(remaining // parts, min(remaining, MINIMUM_SLICE_NS)),
            clock=self._clock,
            effort_budget=self._effort_budget,
            stats=self._stats,
        )

    def with_effort(self, effort_budget: "EffortBudget | None", stats: "SearchStats") -> "Deadline":
        """Bind an effort budget to this start's own stats, keeping the same time bound."""
        return Deadline(0, limit_ns=self.remaining_ns, clock=self._clock, effort_budget=effort_budget, stats=stats)

    def check(self) -> None:
        # Polled once per candidate point; the same test as `expired`, without the three
        # chained property calls that made it measurable there.
        budget = self._effort_budget
        if budget is not None and self._stats is not None and budget.exceeded(self._stats):
            raise TimeLimitReached("packing time limit reached")
        if self.limit_ns - (self._clock() - self.started) <= 0:
            raise TimeLimitReached("packing time limit reached")

    @property
    def uses_real_clock(self) -> bool:
        """True only for the unmodified system monotonic clock.

        An injected test/simulation clock (a closure or bound object) has no
        meaning outside the process that owns it, so the concurrent portfolio
        path (`SolverOrchestrator`) only ever hands work to a worker
        process when this is true -- otherwise it falls back to the fully
        sequential path, which is the only one an injected clock can drive.
        """
        return self._clock is monotonic_ns

    @classmethod
    def until(cls, absolute_ns: int) -> "Deadline":
        """Rebuild a deadline bound to a shared absolute monotonic instant.

        `CLOCK_MONOTONIC` (what `time.monotonic_ns` reads) is machine-wide, not
        per-process, so an absolute instant computed in the parent before
        spawning a worker is a valid, directly comparable deadline in the child
        using its own fresh `monotonic_ns()` call -- no clock object needs to
        cross the process boundary. Only ever called with `uses_real_clock`
        deadlines; see that property.
        """
        return cls(0, limit_ns=absolute_ns - monotonic_ns())


@dataclass(frozen=True, slots=True)
class Candidate:
    point: Point
    position: Point
    rotation: Rotation
    dimensions: Dimensions
    envelope_dimensions: Dimensions
    score: tuple[int, ...]


def _extent(box: AxisAlignedBox) -> tuple[int, int, int, int, int, int]:
    return (box.origin.x, box.origin.y, box.origin.z, box.x2, box.y2, box.z2)


def _solids_collide(left: "hull.HullShape | None", left_origin: tuple[int, int, int],
                    left_extent: tuple[int, int, int],
                    right: "hull.HullShape | None", right_origin: tuple[int, int, int],
                    right_extent: tuple[int, int, int]) -> bool:
    """Exact overlap between two solids of which at least one is a hull.

    Reached only after the axis-aligned test has already said their envelopes overlap, so the
    cost is paid on the small set of pairs where a box answer would have been wrong.
    """
    return hull.collide(
        left if left is not None else hull.HullShape.box(*left_extent), left_origin,
        right if right is not None else hull.HullShape.box(*right_extent), right_origin,
    )


class ContainerState:
    """Placed boxes plus the candidate points they expose.

    Points are corner points of every solid in the container together with their
    projections onto the surfaces below, behind and to the left of them. Points are
    only ever discarded when they fall outside the container or inside a solid --
    a free point is never pruned for being "dominated", because a lower-left point
    can be blocked while the point it would dominate is perfectly usable.
    """

    __slots__ = ("container", "sequence", "placements", "points", "ordered_points", "payload_ticks", "used_volume_ticks",
                 "stack_sensitive", "route_sensitive", "compression_sensitive", "max_z", "occupied", "bounds", "index", "hull_shapes",
                 "lattice_summary", "lattice_items")

    def __init__(self, container: Container, sequence: int):
        self.container = container
        self.sequence = sequence
        self.placements: list[Placement] = []
        self.payload_ticks = 0
        self.used_volume_ticks = 0
        self.stack_sensitive = False
        self.route_sensitive = False
        self.compression_sensitive = False
        self.max_z = 0
        # Set instead of appending to `placements` when GridSolver's quantity-
        # compression fast path applies -- see `lattice_summary.py`.
        self.lattice_summary: "LatticeSummary | None" = None
        self.lattice_items: tuple[ItemInstance, ...] = ()
        self.occupied: list[AxisAlignedBox] = [box for obstacle in container.obstacles for box in obstacle.boxes]
        # Plain integer extents of everything solid. The candidate scan tests these
        # millions of times, and recomputing box properties there dominated the search.
        self.bounds: list[tuple[int, int, int, int, int, int]] = [_extent(b) for b in self.occupied]
        # Parallel to `bounds`: the rotated hull of that solid, or `None` where the solid is
        # an ordinary box. Obstacles are always boxes, so every entry starts `None` and the
        # cuboid-only request never allocates anything beyond this list.
        self.hull_shapes: list["hull.HullShape | None"] = [None] * len(self.occupied)
        dims = container.inner_dimensions
        self.index = SpatialIndex(dims.length.ticks, dims.width.ticks, dims.height.ticks)
        for position, bound in enumerate(self.bounds):
            self.index.add(position, bound)
        self.points: dict[tuple[int, int, int], Point] = {}
        self.ordered_points: list[Point] = []
        self._absorb([Point(0, 0, 0)])
        for box in list(self.occupied):
            self._absorb(self._exposed_points(box))

    def copy(self) -> "ContainerState":
        other = ContainerState.__new__(ContainerState)
        other.container = self.container
        other.sequence = self.sequence
        other.placements = list(self.placements)
        other.points = dict(self.points)
        other.ordered_points = list(self.ordered_points)
        other.payload_ticks = self.payload_ticks
        other.used_volume_ticks = self.used_volume_ticks
        other.stack_sensitive = self.stack_sensitive
        other.route_sensitive = self.route_sensitive
        other.compression_sensitive = self.compression_sensitive
        other.max_z = self.max_z
        other.occupied = list(self.occupied)
        other.bounds = list(self.bounds)
        other.hull_shapes = list(self.hull_shapes)
        other.index = self.index.copy()
        other.lattice_summary = self.lattice_summary
        other.lattice_items = self.lattice_items
        return other

    @property
    def placement_count(self) -> int:
        return self.lattice_summary.count if self.lattice_summary is not None else len(self.placements)

    def add_lattice(self, summary: "LatticeSummary", items: tuple[ItemInstance, ...]) -> None:
        """Record an entire `GridSolver` regular-lattice run in O(1) beyond the
        rotation search already paid for by `summary` -- no per-item `Placement` is
        built. `items` is the placed-instance slice, kept only so this container's
        specific instances remain accounted for (`expand_placements`, unplaced-item
        bookkeeping); it is a reference slice of what the caller already holds, not a
        new allocation per item."""
        self.lattice_summary = summary
        self.lattice_items = items
        self.payload_ticks += summary.total_weight_ticks
        self.used_volume_ticks += summary.used_volume_ticks
        self.stack_sensitive = self.stack_sensitive or any(
            is_stack_sensitive(item.item) for item in items
        )
        self.route_sensitive = self.route_sensitive or any(item.item.stop_index is not None for item in items)
        if summary.max_z_ticks > self.max_z: self.max_z = summary.max_z_ticks

    def add(self, placement: Placement) -> None:
        box = placement.envelope_box
        compression_sensitive = (
            self.compression_sensitive
            or placement.instance.item.shape_type is ShapeType.COMPRESSIBLE
        )
        if compression_sensitive:
            self.used_volume_ticks = _used_volume_with_current_loads((*self.placements, placement))
        else:
            self.used_volume_ticks += used_volume_delta(self.placements, placement)
        self.placements.append(placement)
        self.payload_ticks += placement.instance.weight.ticks
        item = placement.instance.item
        self.stack_sensitive = self.stack_sensitive or is_stack_sensitive(item)
        self.route_sensitive = self.route_sensitive or item.stop_index is not None
        self.compression_sensitive = compression_sensitive
        if box.z2 > self.max_z: self.max_z = box.z2
        self.occupied.append(box)
        bound = _extent(box)
        self.index.add(len(self.bounds), bound)
        self.bounds.append(bound)
        shape = placement.hull_shape
        self.hull_shapes.append(shape)
        # Retiring a point because it falls inside a solid's box assumes the box *is* the
        # solid. For a hull it is not: a placement origin is a corner of a bounding box, and
        # a hull leaves most of that box -- including, for a wedge, the origin itself --
        # available to the next item. Keeping those points alive is what lets the exact
        # collision test below actually decide something; pruning them first would mean the
        # engine could describe an interlocking pack it could never propose.
        if shape is None:
            x1, y1, z1, x2, y2, z2 = bound
            retired = [key for key in self.points
                       if x1 <= key[0] < x2 and y1 <= key[1] < y2 and z1 <= key[2] < z2]
            for key in retired:
                del self.points[key]
            # `ordered_points` holds exactly the points of `points`, so the same test
            # retires the same set there without rebuilding a key per point.
            if retired:
                self.ordered_points = [point for point in self.ordered_points
                                       if not (x1 <= point.x < x2 and y1 <= point.y < y2 and z1 <= point.z < z2)]
        self._absorb(self._exposed_points(box))

    def add_direct(self, placement: Placement) -> None:
        """Record a placement when a solver will never query extreme points.

        GridSolver constructs a proven non-overlapping lattice directly. Updating the
        general solver's occupied-box and projected-point indexes for that path turned
        an otherwise linear placement loop into quadratic work.
        """
        box = placement.envelope_box
        compression_sensitive = (
            self.compression_sensitive
            or placement.instance.item.shape_type is ShapeType.COMPRESSIBLE
        )
        if compression_sensitive:
            self.used_volume_ticks = _used_volume_with_current_loads((*self.placements, placement))
        else:
            self.used_volume_ticks += used_volume_delta(self.placements, placement)
        self.placements.append(placement)
        self.payload_ticks += placement.instance.weight.ticks
        item = placement.instance.item
        self.stack_sensitive = self.stack_sensitive or is_stack_sensitive(item)
        self.route_sensitive = self.route_sensitive or item.stop_index is not None
        self.compression_sensitive = compression_sensitive
        if box.z2 > self.max_z:
            self.max_z = box.z2

    def _absorb(self, points: Iterable[Point]) -> None:
        dims = self.container.inner_dimensions
        length, width, height = dims.length.ticks, dims.width.ticks, dims.height.ticks
        # A solid containing a point is registered in that point's cell, so only that
        # bucket has to be checked rather than every bound in the container.
        index, bounds = self.index, self.bounds
        cell_x, cell_y, cell_z, cells = index.cell_x, index.cell_y, index.cell_z, index.cells
        for point in points:
            x, y, z = point.x, point.y, point.z
            if x >= length or y >= width or z >= height: continue
            key = (x, y, z)
            if key in self.points: continue
            bucket = cells.get((x // cell_x, y // cell_y, z // cell_z), ())
            if any(bx1 <= x < bx2 and by1 <= y < by2 and bz1 <= z < bz2
                   for bx1, by1, bz1, bx2, by2, bz2 in map(bounds.__getitem__, bucket)): continue
            self.points[key] = point
            point_key = (z, y, x)
            low, high = 0, len(self.ordered_points)
            while low < high:
                middle = (low + high) // 2
                current = self.ordered_points[middle]
                if (current.z, current.y, current.x) <= point_key: low = middle + 1
                else: high = middle
            self.ordered_points.insert(low, point)

    def _exposed_points(self, box: AxisAlignedBox) -> list[Point]:
        origin = box.origin
        corners = [
            Point(box.x2, origin.y, origin.z), Point(origin.x, box.y2, origin.z), Point(origin.x, origin.y, box.z2),
            Point(box.x2, box.y2, origin.z), Point(box.x2, origin.y, box.z2), Point(origin.x, box.y2, box.z2),
        ]
        projections = [
            Point(box.x2, origin.y, self._surface_z(box.x2, origin.y, origin.z)),
            Point(box.x2, self._surface_y(box.x2, origin.z, origin.y), origin.z),
            Point(origin.x, box.y2, self._surface_z(origin.x, box.y2, origin.z)),
            Point(self._surface_x(box.y2, origin.z, origin.x), box.y2, origin.z),
            Point(origin.x, self._surface_y(origin.x, box.z2, origin.y), box.z2),
            Point(self._surface_x(origin.y, box.z2, origin.x), origin.y, box.z2),
        ]
        return [*corners, *projections]

    # Each projection is the highest face at or below a ceiling among the solids whose
    # footprint covers the point on the other two axes. Such a solid ends inside the
    # ceiling, so it is registered in one of the cells of the ray below it: walking that
    # ray visits every solid that can contribute, and a maximum is indifferent to seeing
    # one twice.
    def _surface_z(self, x: int, y: int, ceiling: int) -> int:
        index, bounds = self.index, self.bounds
        ix, iy, cells = x // index.cell_x, y // index.cell_y, index.cells
        best = 0
        for iz in range(-(-ceiling // index.cell_z)):
            for position in cells.get((ix, iy, iz), ()):
                b = bounds[position]
                if best < b[5] <= ceiling and b[0] <= x < b[3] and b[1] <= y < b[4]: best = b[5]
        return best

    def _surface_y(self, x: int, z: int, ceiling: int) -> int:
        index, bounds = self.index, self.bounds
        ix, iz, cells = x // index.cell_x, z // index.cell_z, index.cells
        best = 0
        for iy in range(-(-ceiling // index.cell_y)):
            for position in cells.get((ix, iy, iz), ()):
                b = bounds[position]
                if best < b[4] <= ceiling and b[0] <= x < b[3] and b[2] <= z < b[5]: best = b[4]
        return best

    def _surface_x(self, y: int, z: int, ceiling: int) -> int:
        index, bounds = self.index, self.bounds
        iy, iz, cells = y // index.cell_y, z // index.cell_z, index.cells
        best = 0
        for ix in range(-(-ceiling // index.cell_x)):
            for position in cells.get((ix, iy, iz), ()):
                b = bounds[position]
                if best < b[3] <= ceiling and b[1] <= y < b[4] and b[2] <= z < b[5]: best = b[3]
        return best


@dataclass(frozen=True, slots=True)
class SingleContainerSolution:
    state: ContainerState
    unpacked: tuple[ItemInstance, ...]
    exhaustive: bool = False
    time_limit_reached: bool = False
    # True only when `GridSolver` actually built this container via its regular-lattice
    # arithmetic. `GridSolver.pack_one` silently delegates to `ExtremePointSolver` when a
    # specific container disqualifies the lattice (obstacles, tags, axles, stack density,
    # ground-contact rules, mixed item types when the caller names "grid" explicitly via
    # `config.solvers`) -- the delegated result is an ordinary beam-search answer, not the
    # provably-dominant one 's portfolio short-circuit assumes.
    dominant_lattice: bool = False


class SingleContainerSolver(Protocol):
    name: str
    def pack_one(self, container: Container, sequence: int, items: Sequence[ItemInstance], config: PackingConfig, stats: SearchStats, deadline: Deadline) -> SingleContainerSolution: ...


def default_constraints(config: PackingConfig, custom: Sequence[PlacementConstraint] = ()) -> tuple[PlacementConstraint, ...]:
    return (FloorConstraint(), ContainerEligibilityConstraint(), CompatibilityConstraint(),
            TagCountConstraint(), SupportConstraint(config.minimum_support_ratio), TopLoadConstraint(),
            RouteOrderConstraint(), StopAccessibilityConstraint(config.access_directions),
            AxleLoadConstraint(), *custom)


def _candidate_score(state: ContainerState, point: Point, dims: Dimensions) -> tuple[int, ...]:
    inner = state.container.inner_dimensions
    new_height = max(state.max_z, point.z + dims.height.ticks)
    rx = inner.length.ticks - (point.x + dims.length.ticks)
    ry = inner.width.ticks - (point.y + dims.width.ticks)
    rz = inner.height.ticks - (point.z + dims.height.ticks)
    return (point.z, new_height, rx * ry + rz, point.y, point.x)


def _axle_balanced_points(state: ContainerState, item: ItemInstance, forms: Sequence[tuple], limit_x: int) -> list[Point]:
    """Extra floor-level candidates that seat this item on an axle's own limit.

    Every other candidate point is flush against a wall or another placed box, which
    is complete for plain volume packing but not once axle limits are in play: the
    only feasible spot for an item can be floating in open floor space, away from
    every wall and every other box, purely to keep that item's own moment off one
    axle's limit (see `axle_balanced_origins`). Floor level only (z=0), since that is
    the one place `_support_ratio` grants full support with no lateral contact.
    """
    container = state.container
    other_units = load_units(tuple(state.placements))
    tare_ticks = container.tare_weight.ticks
    tare_doubled_x = container.inner_dimensions.length.ticks
    floor_ys = sorted({point.y for point in state.points.values() if point.z == 0}) or [0]
    points: list[Point] = []
    for _, _, _, dx, _, _, _ in forms:
        for x1 in axle_balanced_origins(container.axles, other_units, tare_ticks, tare_doubled_x, item.weight.ticks, dx):
            if 0 <= x1 <= limit_x - dx:
                points.extend(Point(x1, y, 0) for y in floor_ys)
    return points


def _nesting_points(state: ContainerState, item: ItemInstance) -> list[Point]:
    """Origins where ``item`` sinks into an existing identical footprint.

    These origins intentionally lie inside the existing envelope and therefore cannot
    live in the ordinary free-point set. At most one point is derived per placement;
    ordering and deduplication stay deterministic.
    """
    depth = item.item.nesting_height
    if depth is None:
        return []
    points = {
        (placement.envelope_origin.x, placement.envelope_origin.y,
         placement.envelope_box.z2 - depth.ticks)
        for placement in state.placements
        if placement.instance.item.id == item.item.id
    }
    return [Point(x, y, z) for x, y, z in sorted(points, key=lambda point: (point[2], point[1], point[0]))
            if z >= 0]


def find_candidates(state: ContainerState, item: ItemInstance, config: PackingConfig, constraints: Sequence[PlacementConstraint], stats: SearchStats, deadline: Deadline, max_candidates: int | None = None, points: Sequence[Point] | None = None) -> list[Candidate]:
    container = state.container
    if container.max_items is not None and len(state.placements) >= container.max_items: return []
    if container.max_payload is not None and state.payload_ticks + item.weight.ticks > container.max_payload.ticks: return []
    compression_sensitive = (
        state.compression_sensitive or item.item.shape_type is ShapeType.COMPRESSIBLE
    )
    reserve_needs_candidate = (
        item.item.nesting_height is not None
        or item.item.shape_type is ShapeType.CONVEX_HULL
        or compression_sensitive
    )
    if container.void_fill_reserve_ratio > 0 and not reserve_needs_candidate:
        if state.used_volume_ticks + item.dimensions.volume > usable_volume(container): return []
    inner = container.inner_dimensions
    limit_x, limit_y, limit_z = inner.length.ticks, inner.width.ticks, inner.height.ticks
    clearance = config.clearance.ticks
    # Envelope and extents depend only on the rotation, so they are built once instead
    # of once per (point, rotation) pair.
    forms = []
    # A hull is not the same solid under two rotations that happen to give the same box, so
    # `unique_rotations` -- which keys on the box -- would silently drop orientations that
    # differ. Cuboids keep the deduplication they have always had.
    is_hull = item.item.shape_type is ShapeType.CONVEX_HULL
    exact_hull = hull_collision_is_exact(item.item, not clearance)
    rotation_forms = (
        tuple((rotation, item.dimensions.rotated(rotation)) for rotation in item.item.allowed_rotations)
        if is_hull else item.dimensions.unique_rotations(item.item.allowed_rotations)
    )
    for rotation, physical in rotation_forms:
        envelope = physical.expand(config.clearance) if clearance else physical
        # A clearance margin around a hull is not a hull, so the refined test is dropped and
        # the envelope stands -- over-reserving, which is the safe direction.
        shape = (hull.shape_for(item.item.hull_vertices, rotation.value)
                 if exact_hull else None)
        forms.append((rotation, physical, envelope, envelope.length.ticks, envelope.width.ticks,
                      envelope.height.ticks, shape))
    placed = tuple(state.placements)
    stack_sensitive = (state.stack_sensitive
                       or is_stack_sensitive(item.item)
                       or container.max_stack_density is not None)
    route_sensitive = (item.item.stop_index is not None
                       or state.route_sensitive)
    bounds = state.bounds
    hull_shapes = state.hull_shapes
    index = state.index
    nesting = item.item.nesting_height is not None
    placement_offset = len(bounds) - len(placed)
    reserve_check = container.void_fill_reserve_ratio > 0 and reserve_needs_candidate
    usable = usable_volume(container) if reserve_check else 0
    # Everything that decides whether a rule can fire is fixed for this call, so the chain
    # is pruned once here rather than answered "allow" once per position. The support rule
    # stays in regardless: `support_checks` counts every time the chain reaches it.
    active = [(constraint, isinstance(constraint, SupportConstraint))
              for constraint in active_constraints(constraints, container, item, stack_sensitive, route_sensitive)]
    tracing = trace.active()
    if points is None:
        if item.item.nesting_height is None:
            ordered = state.ordered_points[:config.max_candidate_points]
        else:
            merged = {(point.x, point.y, point.z): point
                      for point in (*state.ordered_points, *_nesting_points(state, item))}
            ordered = sorted(merged.values(), key=lambda point: (point.z, point.y, point.x))[:config.max_candidate_points]
        if container.axles is not None:
            ordered = [*ordered, *_axle_balanced_points(state, item, forms, limit_x)]
    else:
        ordered = sorted(points, key=lambda p: (p.z, p.y, p.x))
    candidates: list[Candidate] = []
    best: Candidate | None = None
    retained: list[tuple[tuple[int, ...], int, Candidate]] = []
    bounded = max_candidates is not None and max_candidates > 1
    for point in ordered:
        deadline.check()
        stats.candidate_points_considered += 1
        x1, y1, z1 = point.x, point.y, point.z
        for rotation, physical, envelope, dx, dy, dz, shape in forms:
            stats.placements_attempted += 1
            x2, y2, z2 = x1 + dx, y1 + dy, z1 + dz
            if x2 > limit_x or y2 > limit_y or z2 > limit_z:
                if tracing:
                    trace.emit({"type": "filter", "item_id": item.id, "point": {"x": x1, "y": y1, "z": z1}, "rotation": rotation.value, "reason": "boundary"})
                continue
            tentative = None
            if nesting:
                position = Point(x1 + clearance, y1 + clearance, z1 + clearance)
                tentative = Placement(item, position, rotation, physical, point, envelope)
            blocked = False
            for candidate_index in index.query(x1, y1, z1, x2, y2, z2):
                bx1, by1, bz1, bx2, by2, bz2 = bounds[candidate_index]
                stats.collision_checks += 1
                if x1 < bx2 and bx1 < x2 and y1 < by2 and by1 < y2 and z1 < bz2 and bz1 < z2:
                    placement_index = candidate_index - placement_offset
                    if tentative is not None and placement_index >= 0 and is_valid_nesting(placed[placement_index], tentative):
                        continue
                    blocker = hull_shapes[candidate_index]
                    # The axis-aligned test is the broad phase and stays mandatory. Only when
                    # a hull is one of the two solids does the exact test get to overrule it,
                    # so a request of ordinary boxes never reaches this branch at all.
                    if (shape is not None or blocker is not None) and not _solids_collide(
                            shape, (x1, y1, z1), (dx, dy, dz),
                            blocker, (bx1, by1, bz1), (bx2 - bx1, by2 - by1, bz2 - bz1)):
                        stats.hull_refinements += 1
                        continue
                    blocked = True
                    break
            if blocked:
                if tracing:
                    trace.emit({"type": "filter", "item_id": item.id, "point": {"x": x1, "y": y1, "z": z1}, "rotation": rotation.value, "reason": "collision"})
                continue
            context = ConstraintContext(container, placed, item, point, rotation, physical, envelope, stack_sensitive, route_sensitive)
            rejected = False
            for constraint, counts_support in active:
                if counts_support:
                    stats.support_checks += 1
                result = constraint.evaluate(context)
                if not result.allowed:
                    if tracing:
                        trace.emit({"type": "placement_rejection", "item_id": item.id, "point": {"x": x1, "y": y1, "z": z1}, "rotation": rotation.value, "constraint": type(constraint).__name__, "code": result.code})
                    rejected = True
                    break
            if rejected: continue
            score = _candidate_score(state, point, envelope)
            position = tentative.position if tentative is not None else None
            if reserve_check:
                if position is None:
                    position = Point(x1 + clearance, y1 + clearance, z1 + clearance)
                reserve_placement = tentative or Placement(
                    item, position, rotation, physical, point, envelope
                )
                if compression_sensitive:
                    # Zero load gives the candidate its largest possible physical volume,
                    # while appending it can only compress existing supports. Therefore this
                    # is a safe upper bound: when it fits, the exact support-graph refresh
                    # cannot turn the candidate into a reserve violation. Only candidates
                    # close to the boundary pay the non-local calculation.
                    upper_bound = state.used_volume_ticks + occupied_volume(reserve_placement)
                    projected_volume = (
                        upper_bound
                        if upper_bound <= usable
                        else _used_volume_with_current_loads((*placed, reserve_placement))
                    )
                else:
                    projected_volume = state.used_volume_ticks + used_volume_delta(
                        placed, reserve_placement
                    )
                if projected_volume > usable:
                    continue
            stats.candidates_evaluated += 1
            if tracing:
                trace.emit({"type": "score", "item_id": item.id, "point": {"x": x1, "y": y1, "z": z1}, "rotation": rotation.value, "score": list(score)})
            # Keep every check, counter and trace event, but materialize only candidates
            # that survive selection. Equal scores keep the earlier enumeration entry.
            if max_candidates == 1 and best is not None and score >= best.score:
                continue
            if bounded and len(retained) == max_candidates and score >= retained[0][2].score:
                continue
            if position is None:
                position = Point(x1 + clearance, y1 + clearance, z1 + clearance)
            candidate = Candidate(point, position, rotation, physical, envelope, score)
            if max_candidates == 1:
                best = candidate
            elif bounded:
                # Negate the full integer key and ordinal so heapq's root is the worst
                # retained entry, including the latest entry in a stable-sort tie.
                entry = (tuple(-value for value in score), -stats.candidates_evaluated, candidate)
                if len(retained) < max_candidates:
                    heapq.heappush(retained, entry)
                else:
                    heapq.heapreplace(retained, entry)
            else: candidates.append(candidate)
    if max_candidates == 1: return [] if best is None else [best]
    if bounded:
        return [entry[2] for entry in sorted(retained, reverse=True)]
    if max_candidates is not None: return []
    candidates.sort(key=lambda c: c.score)
    return candidates


def _support_ratio(state: ContainerState, item: ItemInstance, candidate: Candidate) -> float:
    if candidate.point.z == 0: return 1.0
    box = AxisAlignedBox(candidate.point, candidate.envelope_dimensions)
    support = direct_support_view(state.placements, item, box)
    return support.supporting_area / candidate.envelope_dimensions.base_area


def _extended(state: ContainerState, item: ItemInstance, candidate: Candidate) -> ContainerState:
    child = state.copy()
    child.add(Placement(item, candidate.position, candidate.rotation, candidate.dimensions,
                        candidate.point, candidate.envelope_dimensions, _support_ratio(state, item, candidate)))
    return child


def group_batches(items: Sequence[ItemInstance]) -> list[tuple[ItemInstance, ...]]:
    """Items in a group share a container, so they are offered to a solver atomically.

    A batch that does not fit is rejected as a whole and leaves the rest of the order
    untouched -- an impossible group must never strand unrelated items.
    """
    batches: list[tuple[ItemInstance, ...] | list[ItemInstance]] = []
    groups: dict[str, list[ItemInstance]] = {}
    for item in items:
        group = item.item.group
        if group is None:
            batches.append((item,))
            continue
        members = groups.get(group)
        if members is None:
            members = [item]
            groups[group] = members
            batches.append(members)
        else:
            members.append(item)
    # Output slots follow first appearance, and members follow input order. Freeze
    # grouped buckets in place, releasing the lookup before allocating the tuples.
    if groups:
        groups.clear()
        for position, batch in enumerate(batches):
            if isinstance(batch, list):
                batches[position] = tuple(batch)
    return cast(list[tuple[ItemInstance, ...]], batches)


def _place_batch(state: ContainerState, batch: Sequence[ItemInstance], config: PackingConfig, constraints: Sequence[PlacementConstraint], stats: SearchStats, deadline: Deadline, width: int | None):
    """States reachable by placing every member of `batch`, best first.

    Yields lazily: the branch-and-bound search abandons most branches after the first
    child, and copying a state per candidate up front dominates its runtime.
    """
    if len(batch) == 1:
        for candidate in find_candidates(state, batch[0], config, constraints, stats, deadline, width):
            yield _extended(state, batch[0], candidate)
        return
    working = state
    for item in batch:
        candidates = find_candidates(working, item, config, constraints, stats, deadline, 1)
        if not candidates: return
        working = _extended(working, item, candidates[0])
    yield working


def _maximum_count_with_capacity(costs: Sequence[int], capacity: int) -> int:
    """Optimistic cardinality bound for one additive resource.

    Choosing the cheapest remaining units first can only *overestimate* how many
    geometrically feasible items fit.  It is therefore safe for ordering/pruning a
    beam that minimises unpacked-item count; geometry, support and incompatibility can
    make the real answer worse, never better.
    """
    used = 0
    count = 0
    for cost in sorted(costs):
        if used + cost > capacity:
            break
        used += cost
        count += 1
    return count


def _unpacked_lower_bound(
    state: ContainerState,
    unplaced: Sequence[ItemInstance],
    future: Sequence[ItemInstance],
) -> int:
    """Admissible volume/weight lower bound on final unpacked cardinality.

    The bound deliberately ignores geometry and treats every future item as freely
    selectable.  For nesting, nominal volumes are not additive, so the volume half is
    disabled rather than making an unsafe claim; weight remains additive.  This weak
    but sound bound is enough to keep a deliberate large-item skip alive when it can
    make room for several smaller future items.
    """
    if not future:
        return len(unplaced)
    possible = len(future)
    if not any(item.item.nesting_height is not None for item in future):
        free_volume = max(0, state.container.inner_dimensions.volume - state.used_volume_ticks)
        possible = min(
            possible,
            _maximum_count_with_capacity(
                [item.dimensions.volume for item in future], free_volume
            ),
        )
    if state.container.max_payload is not None:
        free_weight = max(0, state.container.max_payload.ticks - state.payload_ticks)
        possible = min(
            possible,
            _maximum_count_with_capacity(
                [item.weight.ticks for item in future], free_weight
            ),
        )
    return len(unplaced) + len(future) - possible


def _node_key(
    node: tuple[ContainerState, tuple[ItemInstance, ...]],
    future: Sequence[ItemInstance] = (),
) -> tuple:
    state, unplaced = node
    used = state.used_volume_ticks
    signature = "|".join(f"{p.instance.id}@{p.envelope_origin.x},{p.envelope_origin.y},{p.envelope_origin.z}"
                         for p in state.placements)
    return (_unpacked_lower_bound(state, unplaced, future), len(unplaced),
            -len(state.placements), state.max_z, -used, signature)


def beam_pack(container: Container, sequence: int, items: Sequence[ItemInstance], config: PackingConfig, constraints: Sequence[PlacementConstraint], stats: SearchStats, deadline: Deadline) -> SingleContainerSolution:
    """Greedy placement widened into a beam by `max_candidates_per_item`.

    A width of one is plain best-fit greedy; wider settings keep that many partial
    packings alive so an early locally-best corner cannot dictate the whole layout.
    """
    branch_width = max(1, config.max_candidates_per_item)
    width = max(1, config.container_plan_beam_width)
    if width == 1:
        # Preserve the historical fast/balanced path exactly.  The quality-only
        # anytime seed below intentionally performs a second rollout so a bounded
        # wider beam always has a complete incumbent; doing that at width one merely
        # doubled candidate work and trace events without adding a reachable state.
        batches = group_batches(items)
        beam: list[tuple[ContainerState, tuple[ItemInstance, ...]]] = [
            (ContainerState(container, sequence), ())
        ]
        for position, batch in enumerate(batches):
            expansions: list[tuple[ContainerState, tuple[ItemInstance, ...]]] = []
            exhausted = False
            for state, unplaced in beam:
                stats.search_nodes_expanded += 1
                try:
                    children = list(
                        _place_batch(
                            state, batch, config, constraints, stats, deadline,
                            branch_width,
                        )
                    )
                except TimeLimitReached:
                    exhausted = True
                    children = []
                if children:
                    expansions.extend((child, unplaced) for child in children)
                else:
                    expansions.append((state, (*unplaced, *batch)))
            expansions.sort(key=_node_key)
            beam = expansions[:branch_width]
            if exhausted:
                state, unplaced = beam[0]
                pending = tuple(
                    item for later in batches[position + 1:] for item in later
                )
                return SingleContainerSolution(
                    state, (*unplaced, *pending), False, True
                )
        state, unplaced = beam[0]
        return SingleContainerSolution(state, unplaced)

    node_limit = config.container_plan_node_limit if width > 1 else None
    nodes = 0
    batches = group_batches(items)
    initial = ContainerState(container, sequence)
    beam: list[tuple[ContainerState, tuple[ItemInstance, ...]]] = [(initial, ())]
    # Seed the anytime search with a complete greedy rollout.  A deadline/node bound
    # can then stop improvement work without ever returning an unfinished beam prefix.
    greedy = ContainerState(container, sequence)
    greedy_unplaced: tuple[ItemInstance, ...] = ()
    for greedy_position, batch in enumerate(batches):
        try:
            children = list(
                _place_batch(greedy, batch, config, constraints, stats, deadline, 1)
            )
        except TimeLimitReached:
            pending = tuple(item for later in batches[greedy_position:] for item in later)
            greedy_unplaced = (*greedy_unplaced, *pending)
            break
        if children:
            greedy = children[0]
        else:
            greedy_unplaced = (*greedy_unplaced, *batch)
    incumbent: tuple[ContainerState, tuple[ItemInstance, ...]] = (greedy, greedy_unplaced)
    incumbent_key = _node_key(incumbent)
    for position, batch in enumerate(batches):
        expansions: list[tuple[ContainerState, tuple[ItemInstance, ...]]] = []
        exhausted = False
        for state, unplaced in beam:
            if node_limit is not None and nodes >= node_limit:
                exhausted = True
                break
            nodes += 1
            stats.search_nodes_expanded += 1
            try:
                children = list(_place_batch(state, batch, config, constraints, stats, deadline, branch_width))
            except TimeLimitReached:
                exhausted = True
                children = []
            if children:
                expansions.extend((child, unplaced) for child in children)
                # A feasible locally-best placement can still be globally harmful: an
                # awkward large item may exclude several smaller later items.  Keeping
                # an explicit skip child turns the candidate beam into an anytime
                # knapsack search instead of a collection of irrevocable greedy paths.
                expansions.append((state, (*unplaced, *batch)))
            else:
                expansions.append((state, (*unplaced, *batch)))
        future = tuple(item for later in batches[position + 1:] for item in later)
        for state, unplaced in expansions:
            candidate = (state, (*unplaced, *future))
            candidate_key = _node_key(candidate)
            if candidate_key < incumbent_key:
                incumbent = candidate
                incumbent_key = candidate_key
        if not expansions:
            break
        expansions.sort(key=lambda node: _node_key(node, future))
        beam = expansions[:width]
        if exhausted:
            # Return the best complete interpretation seen at any earlier prefix, not
            # whichever partial node happened to lead the beam when the bound fired.
            # Consequently a larger counted node limit replays the same prefix and can
            # only retain or improve this incumbent.
            state, unplaced = incumbent
            return SingleContainerSolution(state, unplaced, False, deadline.expired)
    completed = min(beam, key=_node_key) if beam else incumbent
    state, unplaced = completed if _node_key(completed) <= incumbent_key else incumbent
    return SingleContainerSolution(state, unplaced)


class ExtremePointSolver:
    name = "extreme_points"
    def __init__(self, constraints: Sequence[PlacementConstraint] = ()): self.constraints = tuple(constraints)
    def pack_one(self, container, sequence, items, config, stats, deadline):
        return beam_pack(container, sequence, items, config, default_constraints(config, self.constraints), stats, deadline)


def ordering_lead(instance, config) -> tuple:
    """The keys every built-in item ordering starts with.

    Priority is a preference, not a guarantee: it leads so a caller can bias the search,
    but ties (the default, priority 0 for all items) fall through to the strategy's own
    key unchanged. Under `maximum_value` the objective's second key is the value left
    behind (docs/OBJECTIVE.md), so the most valuable item takes first refusal on the
    space -- behind an explicit priority bias, ahead of the strategy's own key. An
    undeclared value is zero, so a request that never sets `value` orders exactly as it
    did before this key existed.
    """
    item = instance.item
    # A route is unloaded last-in-first-out, so the later a stop, the earlier its items
    # must be loaded to end up underneath. Without this, `RouteOrderConstraint` refuses
    # the second item of a two-stop column and the answer costs a container it did not
    # need. Items with no stop ride the whole route and go to the bottom with the last
    # stop. Feasibility leads the objective's own key: an item that cannot be placed at
    # all loses key 0, which dominates every objective's key 1.
    stop = -RIDES_THE_WHOLE_ROUTE if item.stop_index is None else -float(item.stop_index)
    if config.objective != "maximum_value":
        return (-item.priority, stop)
    return (-item.priority, stop, -(item.value or 0))


class LayerSolver:
    name = "layer"
    # Every layer start re-sorts by a total key ending in item id, discarding whatever
    # order it is handed, so repeating it per ordering only burns portfolio budget.
    order_insensitive = True
    def __init__(self, constraints: Sequence[PlacementConstraint] = ()): self.constraints = tuple(constraints)
    def pack_one(self, container, sequence, items, config, stats, deadline):
        ordered = sorted(items, key=lambda i: (*ordering_lead(i, config), -i.dimensions.base_area, -i.dimensions.volume, -i.weight.ticks, i.id))
        return beam_pack(container, sequence, ordered, config, default_constraints(config, self.constraints), stats, deadline)


def _lattice_profile(item) -> tuple:
    """Everything about `item` that determines its behaviour in a regular lattice.

    Two declared item types are fungible for `GridSolver` purposes when this key
    matches -- e.g. one item's dimensions rotated 90 degrees from another's, with
    full rotation freedom making every achievable physical footprint identical
. Keyed by declared id historically, which meant twelve
    geometrically identical items split across two declared types (a natural
    catalog pattern -- the same carton under two SKUs) never reached the exact
    lattice tiling a single declared type would have found, and needed a second
    container the layout never actually required.
    """
    return (
        frozenset(
            (physical.length.ticks, physical.width.ticks, physical.height.ticks)
            for _, physical in item.dimensions.unique_rotations(item.allowed_rotations)
        ),
        item.weight.ticks,
        item.stackable,
        item.must_be_on_floor,
        None if item.max_top_load is None else item.max_top_load.ticks,
        item.tags,
        item.incompatible_tags,
        item.eligible_container_tags,
        item.ground_contact_rule,
        # Nesting overlap is valid only between instances of the same declared item
        # type. Keep ordinary, non-nesting catalog aliases fungible, but never let a
        # mixed-SKU lattice borrow the prototype's overlap allowance.
        item.id if item.nesting_height is not None else None,
        None if item.nesting_height is None else item.nesting_height.ticks,
        item.group,
        # Two items are only interchangeable in one lattice if they agree about how tall
        # a column may be and when they come off the truck. Fusing a limit of 1 with an
        # unlimited item, or stop 0 with stop 1, produces a tiling neither one allows.
        item.max_stacked_items,
        item.stop_index,
    )


class GridSolver:
    """Regular lattice for a single, geometrically fungible item profile.

    The lattice cannot express per-item physical rules, so every rule that a regular
    stack could break is folded into the layer count and capacity before any placement
    is made. Anything the lattice cannot honour falls back to the general solver.
    """

    name = "grid"
    order_insensitive = True

    def supports(self, items: Sequence[ItemInstance]) -> bool:
        if not items: return False
        prototype = items[0].item
        profile = _lattice_profile(prototype)
        previous = prototype
        for instance in items:
            item = instance.item
            # Quantities reuse an immutable Item; only a different object can bring
            # a different profile. Keep one profile instead of materializing a set.
            if item is not prototype and item is not previous and _lattice_profile(item) != profile:
                return False
            previous = item
        return True

    def pack_one(self, container, sequence, items, config, stats, deadline):
        if container.obstacles or not items or not self.supports(items):
            # A mixed-type list would otherwise place every item using the first
            # item's dimensions -- silently wrong geometry, not merely suboptimal.
            return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        prototype = items[0].item
        if prototype.tags & prototype.incompatible_tags:
            return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        if prototype.eligible_container_tags and not (prototype.eligible_container_tags & container.tags):
            return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        if container.void_fill_reserve_ratio > 0:
            return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        if container.max_stack_density is not None:
            # The single-layer heuristic below only ever reasons about one item
            # resting on one other, never the cumulative load a tall column presses
            # through its own footprint -- defer to the general solver, which checks
            # the whole stack per candidate via stack_density_exceeded.
            return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        if container.axles is not None:
            # A regular lattice computes capacity without evaluating the longitudinal
            # reaction after each added item. The general solver already applies the
            # exact gross axle constraint per candidate, so use it instead of ever
            # constructing a packing that only the post-validator can reject.
            return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        if prototype.shape_type is not ShapeType.RIGID_CUBOID:
            # The lattice is closed-form over boxes: it counts cells from envelope extents
            # and caps a column from `max_top_load` arithmetic alone. Neither step can see a
            # hull -- it would tile bounding boxes and call the result exact -- and neither
            # can see pressure, so a compressible column would be sized without ever asking
            # whether its bottom item survives. The general solver checks both per candidate
            #.
            return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        state = ContainerState(container, sequence)
        # Every non-floor lattice cell has one full-area direct supporter, including
        # an exact nesting predecessor. That satisfies ratio=1, covered and single;
        # multiple is satisfiable only on the floor and therefore caps the lattice to
        # one layer instead of delegating the otherwise exact grid.
        single_layer = (prototype.must_be_on_floor or not prototype.stackable
                        or prototype.ground_contact_rule == "multiple")
        # How much less than the full height each layer above the first consumes:
        # zero (the ordinary lattice) unless the item declares it sinks into an
        # identical one beneath it. Reduces to the original nz/z formulas
        # exactly when nesting_height is unset.
        nesting_ticks = 0 if prototype.nesting_height is None else prototype.nesting_height.ticks
        best = None
        for rotation, physical in prototype.dimensions.unique_rotations(prototype.allowed_rotations):
            envelope = physical.expand(config.clearance) if config.clearance.ticks else physical
            nx = container.inner_dimensions.length.ticks // envelope.length.ticks
            ny = container.inner_dimensions.width.ticks // envelope.width.ticks
            layer_step = envelope.height.ticks - nesting_ticks
            inner_height = container.inner_dimensions.height.ticks
            nz = 0 if inner_height < envelope.height.ticks else (inner_height - envelope.height.ticks) // layer_step + 1
            if single_layer: nz = min(nz, 1)
            if prototype.max_top_load is not None and prototype.weight.ticks > 0:
                # In a uniform column the bottom item bears every item above it, so
                # ``nz - 1`` item weights must fit its top-load allowance. This is
                # closed-form O(1) work per rotation and also caps the compact lattice.
                nz = min(nz, prototype.max_top_load.ticks // prototype.weight.ticks + 1)
            if prototype.max_stacked_items is not None:
                # A uniform column of `nz` identical items leaves `nz - 1` resting above
                # its bottom one, which is exactly the count `max_stacked_items` bounds.
                # The lattice can express this in its layer count, so it caps rather
                # than falling back to the general solver.
                nz = min(nz, prototype.max_stacked_items + 1)
            capacity = nx * ny * nz
            if container.max_items is not None: capacity = min(capacity, container.max_items)
            if container.max_payload is not None and prototype.weight.ticks:
                capacity = min(capacity, container.max_payload.ticks // prototype.weight.ticks)
            for tag in prototype.tags & container.tag_limits.keys():
                capacity = min(capacity, container.tag_limits[tag])
            score = (-capacity, envelope.volume, rotation.value)
            if best is None or score < best[0]: best = (score, rotation, physical, envelope, nx, ny, layer_step, capacity)
        _, rotation, physical, envelope, nx, ny, layer_step, capacity = best
        total = min(capacity, len(items))
        # A group is all-or-nothing: a lattice that cannot hold every member holds none.
        if prototype.group is not None and total < len(items):
            return SingleContainerSolution(state, tuple(items), dominant_lattice=True)
        clearance = config.clearance.ticks
        if (
            total > 0 and not config.require_placement_coordinates and nesting_ticks == 0
            and len({i.item.id for i in items[:total]}) == 1
        ):
            # Quantity-compression fast path: the lattice parameters above
            # already fully determine every coordinate in O(r); skip the O(n) loop
            # that would otherwise build one `Placement` per instance purely to fill
            # them in. `nesting_ticks == 0` keeps this to the case
            # `used_volume`/`centre_of_mass` reduce to a plain per-item sum -- see
            # `LatticeSummary`. `LatticeSummary.item_type` is a single string, so this
            # path additionally requires every instance to share one *declared* id
            # (the fungibility rule widened `supports()` to admit geometrically fungible but
            # differently-declared types too; those still take the per-item loop
            # below, which tags each `Placement` from its own `ItemInstance`).
            summary = LatticeSummary(prototype.id, rotation, physical, envelope, nx, ny, layer_step, clearance, total, prototype.weight.ticks)
            state.add_lattice(summary, tuple(items[:total]))
            stats.search_nodes_expanded += 1
            stats.candidate_points_considered += total
            stats.candidates_evaluated += total
            stats.placements_attempted += total
            return SingleContainerSolution(state, tuple(items[total:]))
        target_footprint = (physical.length.ticks, physical.width.ticks, physical.height.ticks)
        placed = 0
        for index in range(total):
            if deadline.expired: break
            stats.search_nodes_expanded += 1
            stats.candidate_points_considered += 1
            x, y, z = index % nx, (index // nx) % ny, index // (nx * ny)
            point = Point(x * envelope.length.ticks, y * envelope.width.ticks, z * layer_step)
            position = Point(point.x + clearance, point.y + clearance, point.z + clearance)
            # A mixed-declared-type lattice group is only reached when every
            # item shares the same *achievable footprint set*, not necessarily the same
            # rotation labels reaching it -- item A's own "LWH" and item B's own "WLH"
            # can both be `physical` while meaning different things relative to each
            # item's own declared base dimensions, so the rotation stored on each
            # `Placement` must be looked up per item, never copied from `prototype`.
            own_rotation = physical_rotation = None
            for candidate_rotation, candidate_dims in items[index].item.dimensions.unique_rotations(
                items[index].item.allowed_rotations
            ):
                if (
                    candidate_dims.length.ticks,
                    candidate_dims.width.ticks,
                    candidate_dims.height.ticks,
                ) == target_footprint:
                    own_rotation, physical_rotation = candidate_rotation, candidate_dims
                    break
            assert own_rotation is not None, "supports() guarantees a matching rotation exists"
            state.add_direct(Placement(items[index], position, own_rotation, physical_rotation, point, envelope, 1.0))
            placed += 1
            stats.candidates_evaluated += 1
            stats.placements_attempted += 1
        if prototype.group is not None and placed < len(items):
            return SingleContainerSolution(ContainerState(container, sequence), tuple(items), False, deadline.expired, dominant_lattice=True)
        return SingleContainerSolution(state, tuple(items[placed:]), False, placed < total and deadline.expired, dominant_lattice=True)


@dataclass(frozen=True, slots=True)
class Space:
    origin: Point
    dimensions: Dimensions

    @property
    def box(self) -> AxisAlignedBox: return AxisAlignedBox(self.origin, self.dimensions)


def _subtract(space: Space, box: AxisAlignedBox) -> list[Space]:
    """The six maximal slabs of `space` that survive carving `box` out of it."""
    source = space.box
    if not source.intersects(box): return [space]
    out: list[Space] = []

    def push(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> None:
        if x2 > x1 and y2 > y1 and z2 > z1:
            out.append(Space(Point(x1, y1, z1), Dimensions(Length(x2 - x1), Length(y2 - y1), Length(z2 - z1))))

    push(source.origin.x, source.origin.y, source.origin.z, min(source.x2, box.origin.x), source.y2, source.z2)
    push(max(source.origin.x, box.x2), source.origin.y, source.origin.z, source.x2, source.y2, source.z2)
    push(source.origin.x, source.origin.y, source.origin.z, source.x2, min(source.y2, box.origin.y), source.z2)
    push(source.origin.x, max(source.origin.y, box.y2), source.origin.z, source.x2, source.y2, source.z2)
    push(source.origin.x, source.origin.y, source.origin.z, source.x2, source.y2, min(source.z2, box.origin.z))
    push(source.origin.x, source.origin.y, max(source.origin.z, box.z2), source.x2, source.y2, source.z2)
    return out


#: Adversarial item mixes can carve far more surviving maximal spaces than any real
#: request needs -- growth is roughly linear in placement count once size variety
#: defeats the dominance filter below, and every extra space multiplies the per-item
#: choose() scan, the carve and the O(new * s) dominance check. Past this many
#: surviving spaces, the smallest (least likely to fit a future item) are dropped
#: first; well under any request this library's tests or fixtures exercise.
MAX_MAXIMAL_SPACES = 256


def subtract_all(
    spaces: Sequence[Space],
    boxes: Sequence[AxisAlignedBox],
    stats: SearchStats | None = None,
) -> list[Space]:
    # Precondition: `spaces` is containment-free -- every call site passes either a
    # single whole space or this function's own output. A survivor untouched by every
    # carve can therefore neither contain nor be contained by another survivor, and a
    # new slab cannot contain it either (the slab's own parent could not), so only
    # new slabs need the dominance check below: O(new * s) instead of O(s^2).
    current: list[tuple[Space, bool]] = [(space, False) for space in spaces]
    for box in boxes:
        if stats is not None:
            stats.space_partitions += len(current)
        carved: list[tuple[Space, bool]] = []
        for space, is_new in current:
            if space.box.intersects(box):
                carved.extend((part, True) for part in _subtract(space, box))
            else:
                carved.append((space, is_new))
        current = carved
    unique: dict[tuple[int, ...], tuple[Space, bool]] = {}
    for space, is_new in current:
        unique[(space.origin.x, space.origin.y, space.origin.z, space.dimensions.length.ticks,
                space.dimensions.width.ticks, space.dimensions.height.ticks)] = (space, is_new)
    kept = sorted(unique.values(),
                  key=lambda pair: (pair[0].origin.z, pair[0].origin.y, pair[0].origin.x,
                                    -pair[0].dimensions.volume))
    # Without this filter every placement multiplies the space list and the solver
    # degenerates into an exponential scan of overlapping duplicates.
    extents = [(space.origin.x, space.origin.y, space.origin.z,
                space.origin.x + space.dimensions.length.ticks,
                space.origin.y + space.dimensions.width.ticks,
                space.origin.z + space.dimensions.height.ticks)
               for space, _ in kept]
    result: list[Space] = []
    for index, (space, is_new) in enumerate(kept):
        if is_new:
            x1, y1, z1, x2, y2, z2 = extents[index]
            if any(other[0] <= x1 and other[1] <= y1 and other[2] <= z1
                   and x2 <= other[3] and y2 <= other[4] and z2 <= other[5]
                   for position, other in enumerate(extents) if position != index):
                continue
        result.append(space)
    if len(result) > MAX_MAXIMAL_SPACES:
        result = sorted(result, key=lambda s: -s.dimensions.volume)[:MAX_MAXIMAL_SPACES]
        result.sort(key=lambda s: (s.origin.z, s.origin.y, s.origin.x))
    return result


class MaximalSpaceSolver:
    """Maximal free spaces plus the common feasibility engine.

    Candidates are evaluated *at* each space origin. Asking the shared finder for its
    globally best point and then testing whether it happens to equal a space origin
    would reject almost every space, which is why the point set is passed explicitly.
    """

    name = "maximal_spaces"

    def __init__(self, constraints: Sequence[PlacementConstraint] = ()): self.constraints = tuple(constraints)

    def pack_one(self, container, sequence, items, config, stats, deadline):
        state = ContainerState(container, sequence)
        constraints = default_constraints(config, self.constraints)
        spaces = subtract_all(
            [Space(Point(0, 0, 0), container.inner_dimensions)],
            [box for obstacle in container.obstacles for box in obstacle.boxes],
            stats,
        )
        unplaced: list[ItemInstance] = []
        batches = group_batches(items)
        for position, batch in enumerate(batches):
            # A failed single-item batch never mutates `state`/`spaces` in place --
            # `_choose` only reads them and a rejection is decided before `_extended`/
            # `subtract_all` ever rebind those names -- so only a multi-item batch
            # that can be left half-applied needs the pre-batch snapshot.
            snapshot, saved = (state.copy(), list(spaces)) if len(batch) > 1 else (state, spaces)
            placed = True
            try:
                for item in batch:
                    stats.search_nodes_expanded += 1
                    chosen = self._choose(state, spaces, item, config, constraints, stats, deadline)
                    if chosen is None:
                        placed = False
                        break
                    state = _extended(state, item, chosen)
                    spaces = subtract_all(
                        spaces,
                        [AxisAlignedBox(chosen.point, chosen.envelope_dimensions)],
                        stats,
                    )
            except TimeLimitReached:
                pending = tuple(item for later in batches[position:] for item in later)
                return SingleContainerSolution(snapshot, (*unplaced, *pending), False, True)
            if not placed:
                state, spaces = snapshot, saved
                unplaced.extend(batch)
        return SingleContainerSolution(state, tuple(unplaced))

    @staticmethod
    def _choose(state, spaces, item, config, constraints, stats, deadline) -> Candidate | None:
        for space in spaces:
            deadline.check()
            for candidate in find_candidates(state, item, config, constraints, stats, deadline, None, [space.origin]):
                if candidate.envelope_dimensions.fits_inside(space.dimensions): return candidate
        return None


@dataclass(frozen=True, slots=True)
class _HomogeneousBlock:
    """A completely filled rectangular lattice of one declared item type."""

    space: Space
    item_id: str
    rotation: Rotation
    physical: Dimensions
    envelope: Dimensions
    nx: int
    ny: int
    nz: int
    count: int
    used_volume: int
    score: tuple

    @property
    def box(self) -> AxisAlignedBox:
        return AxisAlignedBox(
            self.space.origin,
            Dimensions(
                Length(self.nx * self.envelope.length.ticks),
                Length(self.ny * self.envelope.width.ticks),
                Length(self.nz * self.envelope.height.ticks),
            ),
        )


class HomogeneousBlockSolver:
    """Pack deterministic, solid single-type blocks into maximal free spaces.

    The ordinary extreme-point solver makes one irrevocable geometric decision per
    instance.  That is deliberately general, but it is a poor search unit for large
    unrestricted carton-loading requests: hundreds of equivalent copies of one SKU
    describe a regular block much more compactly than hundreds of independent nodes.
    This strategy enumerates bounded rectangular lattices, commits the best solid
    block, subtracts its bounding box from the free-space set, and repeats.  It runs
    both cardinality-first and occupied-volume-first orderings and returns the better
    canonical objective vector.

    A block is used only for the plain-box subset for which every generated member has
    full support and no per-placement business rule can distinguish it.  Requests
    outside that subset fall back to ``ExtremePointSolver`` rather than weakening a
    constraint.  Candidate enumeration is O(B*S*T*R*X*Y) time and O(S+n) space, where
    B is the number of committed blocks, S the capped maximal-space count, T item
    types, R unique rotations, and X/Y the grid extents.  Both the wall-clock/effort
    deadline and ``container_plan_node_limit`` bound the product explicitly.
    """

    name = "homogeneous_blocks"
    order_insensitive = True

    def __init__(self, constraints: Sequence[PlacementConstraint] = ()):
        self.constraints = tuple(constraints)

    def pack_one(self, container, sequence, items, config, stats, deadline):
        if not self._supports(container, items, config):
            return ExtremePointSolver(self.constraints).pack_one(
                container, sequence, items, config, stats, deadline
            )
        best: SingleContainerSolution | None = None
        reached = False
        for mode in ("count", "volume"):
            if deadline.expired:
                reached = True
                break
            solution = self._pack_mode(
                container, sequence, items, config, stats, deadline, mode
            )
            reached = reached or solution.time_limit_reached
            if best is None or self._solution_key(solution) < self._solution_key(best):
                best = solution
        if best is None:
            return SingleContainerSolution(
                ContainerState(container, sequence), tuple(items), False, reached
            )
        return SingleContainerSolution(best.state, best.unpacked, False, reached)

    def _supports(self, container, items, config) -> bool:
        if self.constraints or container.obstacles or container.axles is not None:
            return False
        if (
            container.tag_limits
            or container.max_stack_density is not None
            or container.void_fill_reserve_ratio > 0
        ):
            return False
        return all(
            item.item.group is None
            and not item.item.tags
            and not item.item.incompatible_tags
            and not item.item.eligible_container_tags
            and item.item.stackable
            and not item.item.must_be_on_floor
            and item.item.max_top_load is None
            and item.item.max_stacked_items is None
            and item.item.minimum_support_ratio == 0
            and item.item.ground_contact_rule in (None, "free")
            and item.item.nesting_height is None
            and item.item.stop_index is None
            for item in items
        )

    @staticmethod
    def _solution_key(solution: SingleContainerSolution) -> tuple:
        state = solution.state
        signature = tuple(
            (p.instance.id, p.envelope_origin.x, p.envelope_origin.y, p.envelope_origin.z)
            for p in state.placements
        )
        return (len(solution.unpacked), -state.used_volume_ticks, state.max_z, signature)

    def _pack_mode(self, container, sequence, items, config, stats, deadline, mode):
        state = ContainerState(container, sequence)
        spaces = [Space(Point(0, 0, 0), container.inner_dimensions)]
        remaining: dict[str, list[ItemInstance]] = {}
        for instance in items:
            remaining.setdefault(instance.item.id, []).append(instance)
        nodes = 0
        reached = False
        while spaces and any(remaining.values()):
            best: _HomogeneousBlock | None = None
            for space in spaces:
                space_volume = space.dimensions.volume
                for item_id in sorted(remaining):
                    available = remaining[item_id]
                    if not available:
                        continue
                    prototype = available[0]
                    capacity = len(available)
                    if container.max_items is not None:
                        capacity = min(capacity, container.max_items - state.placement_count)
                    if container.max_payload is not None and prototype.weight.ticks > 0:
                        capacity = min(
                            capacity,
                            (container.max_payload.ticks - state.payload_ticks)
                            // prototype.weight.ticks,
                        )
                    if capacity <= 0:
                        continue
                    for rotation, physical in prototype.dimensions.unique_rotations(
                        prototype.item.allowed_rotations
                    ):
                        envelope = (
                            physical.expand(config.clearance)
                            if config.clearance.ticks
                            else physical
                        )
                        maximum_x = space.dimensions.length.ticks // envelope.length.ticks
                        maximum_y = space.dimensions.width.ticks // envelope.width.ticks
                        maximum_z = space.dimensions.height.ticks // envelope.height.ticks
                        for nx in range(1, min(maximum_x, capacity) + 1):
                            maximum_ny = min(maximum_y, capacity // nx)
                            for ny in range(1, maximum_ny + 1):
                                if (
                                    nodes >= config.container_plan_node_limit
                                    or deadline.expired
                                ):
                                    reached = True
                                    break
                                nodes += 1
                                stats.search_nodes_expanded += 1
                                nz = min(maximum_z, capacity // (nx * ny))
                                if nz <= 0:
                                    continue
                                count = nx * ny * nz
                                used_volume = count * physical.volume
                                space_fill_ppm = used_volume * 1_000_000 // space_volume
                                if mode == "count":
                                    lead = (-count, -space_fill_ppm, -used_volume)
                                else:
                                    lead = (-used_volume, -space_fill_ppm, -count)
                                score = (
                                    *lead,
                                    space.origin.z,
                                    space.origin.y,
                                    space.origin.x,
                                    item_id,
                                    rotation.value,
                                    nx,
                                    ny,
                                    nz,
                                )
                                candidate = _HomogeneousBlock(
                                    space, item_id, rotation, physical, envelope,
                                    nx, ny, nz, count, used_volume, score,
                                )
                                if best is None or candidate.score < best.score:
                                    best = candidate
                            if reached:
                                break
                        if reached:
                            break
                    if reached:
                        break
                if reached:
                    break
            if best is None or reached:
                break
            chosen = remaining[best.item_id][:best.count]
            del remaining[best.item_id][:best.count]
            clearance = config.clearance.ticks
            index = 0
            for z in range(best.nz):
                for y in range(best.ny):
                    for x in range(best.nx):
                        point = Point(
                            best.space.origin.x + x * best.envelope.length.ticks,
                            best.space.origin.y + y * best.envelope.width.ticks,
                            best.space.origin.z + z * best.envelope.height.ticks,
                        )
                        position = Point(
                            point.x + clearance,
                            point.y + clearance,
                            point.z + clearance,
                        )
                        state.add_direct(
                            Placement(
                                chosen[index], position, best.rotation, best.physical,
                                point, best.envelope, 1.0,
                            )
                        )
                        index += 1
                        stats.candidates_evaluated += 1
                        stats.placements_attempted += 1
            spaces = subtract_all(spaces, [best.box], stats)
        unpacked = tuple(
            instance
            for item_id in sorted(remaining)
            for instance in remaining[item_id]
        )
        return SingleContainerSolution(state, unpacked, False, reached or deadline.expired)


def _placed_volume(state: ContainerState) -> int:
    return sum(p.dimensions.volume for p in state.placements)


def _is_better_state_values(
    candidate_count: int,
    candidate_volume: int,
    incumbent_count: int,
    incumbent_volume: int,
) -> bool:
    if candidate_count != incumbent_count:
        return candidate_count > incumbent_count
    return candidate_volume > incumbent_volume


def is_better_container_state(candidate: ContainerState, incumbent: ContainerState) -> bool:
    """True if `candidate` should replace `incumbent` as a best-so-far.

    More items placed always wins; among equal counts, higher placed volume wins.
    Without the second key, `ExactSmallSolver`'s branch-and-bound never looked past item
    count at all, so a longer-running search reaching an equal-count-but-worse-arranged
    state later in traversal order would silently replace a better-arranged one found
    earlier -- how more search time made the chosen packing *worse* under the outer
    objective despite placing no fewer items. The two extra keys the outer
    scorer cares about (unused volume, stack height) were never considered here at all.
    """
    return _is_better_state_values(
        len(candidate.placements),
        _placed_volume(candidate),
        len(incumbent.placements),
        _placed_volume(incumbent),
    )


class ExactSmallSolver:
    """Depth-first branch and bound over whole group batches.

    The search is exact only for the discrete candidate model and item-count objective;
    it deliberately does not publish a global optimality claim.
    """

    name = "exact_small"

    def __init__(self, constraints: Sequence[PlacementConstraint] = ()): self.constraints = tuple(constraints)

    def pack_one(self, container, sequence, items, config, stats, deadline):
        if len(items) > config.exact_item_limit: raise ValueError("exact-small item limit exceeded")
        constraints = default_constraints(config, self.constraints)
        batches = group_batches(items)
        batch_volumes = tuple(
            sum(item.dimensions.volume for item in batch) for batch in batches
        )
        suffix_volumes = [0] * (len(batches) + 1)
        for index in range(len(batches) - 1, -1, -1):
            suffix_volumes[index] = batch_volumes[index] + suffix_volumes[index + 1]
        best = ContainerState(container, sequence)
        best_volume = 0

        def state_rank(state: ContainerState, state_volume: int) -> tuple[int, ...]:
            """The exact solver's incumbent order for this one container.

            Landed cost precedes unused volume in the public objective. A promotional
            bracket may make a heavier equal-count subset cheaper, so the historical
            count/volume rank was wrong for this objective (second review).
            """
            count = len(state.placements)
            if config.objective != "lowest_landed_cost":
                return (-count, -state_volume)
            dimensions = container.outer_dimensions or container.inner_dimensions
            dim_weight = dimensional_weight(
                dimensions,
                config.dimensional_weight_divisor,
                config.dimensional_weight_length_unit,
                config.dimensional_weight_weight_unit,
            )
            billed_ticks = max(container.tare_weight.ticks + state.payload_ticks, dim_weight.ticks)
            table = container.rate_table
            charge = None if table is None else table.charge_minor_or_none(_grams(billed_ticks))
            return (-count, UNPRICEABLE_MINOR if charge is None else charge, -state_volume)

        best_rank = state_rank(best, best_volume)

        def dfs(index: int, state: ContainerState, reachable: int, state_volume: int) -> None:
            nonlocal best, best_volume, best_rank
            if deadline.expired:
                return
            stats.search_nodes_expanded += 1
            state_count = len(state.placements)
            best_count = len(best.placements)
            candidate_rank = state_rank(state, state_volume)
            if candidate_rank < best_rank:
                best = state
                best_volume = state_volume
                best_rank = candidate_rank
            if index >= len(batches): return
            potential_count = state_count + reachable
            best_count = len(best.placements)
            if potential_count < best_count:
                return
            if (config.objective != "lowest_landed_cost"
                    and potential_count == best_count
                    and state_volume + suffix_volumes[index] <= best_volume):
                return
            batch = batches[index]
            remaining = reachable - len(batch)
            for child in _place_batch(state, batch, config, constraints, stats, deadline, None):
                dfs(index + 1, child, remaining, state_volume + batch_volumes[index])
                if len(best.placements) == len(items) or deadline.expired: return
            dfs(index + 1, state, remaining, state_volume)

        try:
            dfs(0, ContainerState(container, sequence), len(items), 0)
        except TimeLimitReached:
            # `dfs` itself only ever backs out gracefully via `deadline.expired`, but
            # `_place_batch` -> `find_candidates` still checks the same deadline with
            # the raising `deadline.check()`, and nothing between there and here caught
            # it. Left unguarded, a deadline that expires mid-scan discards `best`
            # entirely (a local variable, lost with the unwound stack) rather than
            # keeping whatever was already found -- the exact mechanism the lattice proof gate
            # describes, just one call deeper than the `is_better_container_state` fix
            # above addresses: a *longer* budget that reaches this point instead of
            # completing first returns nothing, which reads as the search getting
            # worse with more time. Swallowing it here restores the invariant that more
            # search time can only add to `best`, never erase it.
            pass
        packed = {p.instance.id for p in best.placements}
        # This search proves only the maximum number of items reachable through the
        # generated discrete candidate points. It stops after the first full packing
        # and does not minimize the remaining keys of the public objective vector, so
        # it must not advertise global optimality.
        return SingleContainerSolution(
            best,
            tuple(i for i in items if i.id not in packed),
            False,
            deadline.expired,
        )


@dataclass(frozen=True, slots=True)
class RawSolution:
    solver_name: str
    containers: tuple[PackedContainer, ...]
    unpacked: tuple[UnpackedItem, ...]
    stats: SearchStats
    time_limit_reached: bool = False
    effort_limit_reached: bool = False
    exhaustive: bool = False


@dataclass(frozen=True, slots=True)
class PortfolioRun:
    solutions: tuple[RawSolution, ...]
    starts: tuple[StartRecord, ...]

    def __iter__(self):
        return iter(self.solutions)

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, index):
        return self.solutions[index]


class DeterministicRandom:
    """Seeded linear congruential generator.

    Only the high bits are published: the low bits of a power-of-two modulus have
    tiny periods, and `next_int` is called with very small bounds by the shuffle.
    Draws outside the largest whole multiple of `upper` are rejected so the result
    carries no modulo bias.
    """

    MODULUS = 1 << 31
    MULTIPLIER = 1_103_515_245
    INCREMENT = 12_345
    DRAW_LIMIT = 1 << 32

    def __init__(self, seed: int): self.state = seed % self.MODULUS

    def _bits(self) -> int:
        self.state = (self.MULTIPLIER * self.state + self.INCREMENT) % self.MODULUS
        return self.state >> 15

    def next_int(self, upper: int) -> int:
        if upper <= 1: return 0
        bound = self.DRAW_LIMIT - (self.DRAW_LIMIT % upper)
        while True:
            draw = self._bits() * 65536 + self._bits()
            if draw < bound: return draw % upper

    def shuffled(self, values):
        out = list(values)
        for index in range(len(out) - 1, 0, -1):
            other = self.next_int(index + 1)
            out[index], out[other] = out[other], out[index]
        return out


def _with_top_loads(placements: Sequence[Placement]) -> tuple[Placement, ...]:
    loads = top_loads(load_units(placements))
    return tuple(
        Placement(p.instance, p.position, p.rotation, p.dimensions, p.envelope_origin,
                  p.envelope_dimensions, p.support_ratio, Weight(load))
        for p, load in zip(placements, loads)
    )


def _used_volume_with_current_loads(placements: Sequence[Placement]) -> int:
    """Physical volume after this scene's support loads have been propagated.

    Compression makes appending one placement non-local: a new upper item changes the
    occupied height of existing supports. The rigid-item path keeps its O(1) delta; this
    composite path rebuilds the support graph and nesting total in
    O(n log n + q + e) time and O(n + e) graph space, where q is broad-phase work and e
    is the contact-edge count; both become O(n^2) in a physically dense worst case. This
    is the bound of one refresh, not one solve. The whole-solve sum over candidates and
    search nodes is stated in docs/ALGORITHMS-AND-COMPLEXITY.md.
    """
    return nesting_used_volume(_with_top_loads(placements))


class DefaultContainerSelector:
    """Prefers the container that holds the most items, then the cheapest, then the tightest."""

    def score(self, container: Container, solution: SingleContainerSolution) -> tuple:
        state = solution.state
        used = state.used_volume_ticks
        return (-state.placement_count, container.cost_minor, container.inner_dimensions.volume - used, container.id)


@dataclass(slots=True)
class _ContainerPlan:
    packed: tuple[PackedContainer, ...]
    remaining: tuple[ItemInstance, ...]
    inventory: dict[str, int | None]
    sequences: dict[str, int]
    exhaustive: bool = True
    dominant_lattice: bool = True


#: Solver names a caller may pass in `PackingConfig.solvers`, in the order they will run.
SOLVER_FACTORIES = {
    "grid": lambda constraints: GridSolver(),
    "extreme_points": lambda constraints: ExtremePointSolver(constraints),
    "homogeneous_blocks": lambda constraints: HomogeneousBlockSolver(constraints),
    "layer": lambda constraints: LayerSolver(constraints),
    "maximal_spaces": lambda constraints: MaximalSpaceSolver(constraints),
    "exact_small": lambda constraints: ExactSmallSolver(constraints),
}


class UnknownSolverError(ValueError):
    """`PackingConfig.solvers` named a solver this library does not implement."""


def _run_concurrent_start(solver, order, containers, config, container_selector, effort_budget, absolute_deadline_ns):
    """One portfolio start's worker body, run in its own process.

    Module-level, not a bound method, so `ProcessPoolExecutor` can pickle it
    without pulling in the whole orchestrator (custom constraints/orders/solvers
    are never on this path -- see `SolverOrchestrator._eligible_for_concurrent_execution`).
    A throwaway `SolverOrchestrator` carrying only the (picklable, stateless by
    construction here) container selector is enough to reach `_across_containers`;
    everything else this start needs -- its own `SearchStats`, its own
    `ContainerState`s, its own view of the shared deadline -- is built fresh
    inside this process and never touches any other worker's state.
    """
    stats = SearchStats()
    budget = Deadline.until(absolute_deadline_ns)
    if effort_budget is not None:
        budget = budget.with_effort(effort_budget, stats)
    orchestrator = SolverOrchestrator(container_selector=container_selector)
    packed, unpacked, exhaustive, reached, dominant_lattice = orchestrator._across_containers(
        solver, order, containers, config, stats, budget
    )
    return packed, unpacked, exhaustive, reached, dominant_lattice, stats


class SolverOrchestrator:
    def __init__(self, custom_constraints=(), custom_orders=(), custom_solvers=(), container_selector=None):
        self.custom_constraints = tuple(custom_constraints); self.custom_orders = tuple(custom_orders); self.custom_solvers = tuple(custom_solvers)
        self.container_selector = container_selector or DefaultContainerSelector()

    def _solvers(self, items, config):
        if config.solvers:
            unknown = [name for name in config.solvers if name not in SOLVER_FACTORIES]
            if unknown:
                raise UnknownSolverError(
                    f"unknown solver name(s) {unknown}; expected one of {sorted(SOLVER_FACTORIES)}"
                )
            base = [SOLVER_FACTORIES[name](self.custom_constraints) for name in config.solvers]
            return [*self.custom_solvers, *base]
        grid = GridSolver()
        if grid.supports(items) and not self.custom_constraints:
            base = [grid] if config.profile.value == "fast" else [grid, ExtremePointSolver()]
        elif config.profile.value == "fast": base = [LayerSolver(self.custom_constraints)]
        elif config.profile.value == "exact_small" and len(items) <= config.exact_item_limit: base = [ExactSmallSolver(self.custom_constraints), ExtremePointSolver(self.custom_constraints)]
        elif config.profile.value == "quality":
            base = [
                HomogeneousBlockSolver(self.custom_constraints),
                ExtremePointSolver(self.custom_constraints),
                MaximalSpaceSolver(self.custom_constraints),
                LayerSolver(self.custom_constraints),
            ]
            if len(items) <= config.exact_item_limit: base.insert(0, ExactSmallSolver(self.custom_constraints))
        else: base = [ExtremePointSolver(self.custom_constraints), LayerSolver(self.custom_constraints)]
        return [*self.custom_solvers, *base]

    def _orders(self, items, config):
        # Every strategy is keyed behind `ordering_lead`, which carries priority and,
        # under `maximum_value`, the item's own value.
        strategies = []
        if config.profile.value == "quality":
            strategies.extend([
                ("small_edge", lambda i: (i.dimensions.max_edge, i.dimensions.volume, i.id)),
                ("small_volume", lambda i: (i.dimensions.volume, i.dimensions.max_edge, i.id)),
            ])
        strategies.extend([
            ("volume", lambda i: (-i.dimensions.volume, -i.dimensions.max_edge, -i.weight.ticks, i.id)),
            ("base_area", lambda i: (-i.dimensions.base_area, -i.dimensions.volume, i.id)),
            ("weight", lambda i: (-i.weight.ticks, -i.dimensions.volume, i.id)),
            ("constrained", lambda i: (len(i.item.allowed_rotations), -i.item.minimum_support_ratio, -i.dimensions.volume, i.id)),
        ])
        limit = 1 if config.profile.value == "fast" else config.multi_start_orders
        out = []; seen = set()
        for name, key in strategies:
            order = tuple(sorted(items, key=lambda i, key=key: (*ordering_lead(i, config), *key(i))))
            signature = tuple(i.id for i in order)
            if signature not in seen: out.append((name, order)); seen.add(signature)
            if len(out) >= limit: return out
        for strategy in self.custom_orders:
            order = tuple(strategy.order(items, config.seed)); signature = tuple(i.id for i in order)
            if signature not in seen: out.append((strategy.name, order)); seen.add(signature)
            if len(out) >= limit: return out
        rng = DeterministicRandom(config.seed); attempts = 0
        while len(out) < limit and attempts < max(32, limit * 16):
            attempts += 1; order = tuple(rng.shuffled(items)); signature = tuple(i.id for i in order)
            if signature not in seen: out.append((f"seeded_{len(out)}", order)); seen.add(signature)
        return out

    def _starts(self, items, config):
        orders = self._orders(items, config)
        starts = []
        for solver in self._solvers(items, config):
            # A lattice ignores item order, so repeating it per ordering only burns budget.
            chosen = orders[:1] if getattr(solver, "order_insensitive", False) else orders
            starts.extend((solver, name, order) for name, order in chosen)
        if config.container_plan_beam_width <= 1 or config.profile.value != "quality":
            return starts
        greedy = [
            (solver, name, order)
            if getattr(solver, "name", "") == "homogeneous_blocks"
            else (solver, f"{name}:greedy", order)
            for solver, name, order in starts
        ]
        extreme = next(
            (start for start in starts if getattr(start[0], "name", "") == "extreme_points"),
            None,
        )
        if extreme is None:
            return greedy
        item_types = sorted({item.item.id for item in items})
        neighborhoods = [
            (extreme[0], f"neighborhood:{item_type}", extreme[2])
            for item_type in item_types
        ]
        refinement = (extreme[0], "best:beam", extreme[2])
        return [*greedy, *neighborhoods, refinement]

    def solve(self, items, containers, config, deadline):
        starts = self._starts(items, config)
        effort_budget = config.effort_budget
        if effort_budget is not None and effort_budget.max_restarts is not None:
            starts = starts[:effort_budget.max_restarts]
        if self._eligible_for_concurrent_execution(deadline, config):
            return self._solve_concurrent(items, containers, config, deadline, starts, effort_budget)
        return self._solve_sequential(items, containers, config, deadline, starts, effort_budget)

    def _eligible_for_concurrent_execution(self, deadline, config) -> bool:
        """Gate for the worker-process path.

        Concurrency is opt-in (`parallel_starts > 1`, default `1`) and only
        engages when every input crossing the process boundary is known-safe to
        pickle and re-run in a fresh process: the real system clock (an injected
        test/simulation clock has no cross-process meaning, see
        `Deadline.uses_real_clock`) and no caller-supplied extensions (custom
        constraints/orders/solvers/container selector), which may legitimately
        close over unpicklable state. Ineligible calls fall back to the exact
        original sequential path -- silently, since correctness never depends on
        this gate, only whether this particular call gets the parallel speed-up.
        """
        return (
            config.parallel_starts > 1
            and config.container_plan_beam_width == 1
            and deadline.uses_real_clock
            and not self.custom_constraints
            and not self.custom_orders
            and not self.custom_solvers
            and isinstance(self.container_selector, DefaultContainerSelector)
        )

    def _solve_sequential(self, items, containers, config, deadline, starts, effort_budget):
        results = []
        completed_orders: dict[str, tuple] = {}
        records = [
            StartRecord(f"{solver.name}:{order_name}", False, False, False)
            for solver, order_name, _ in starts
        ]
        for position, (solver, order_name, order) in enumerate(starts):
            if deadline.expired: break
            if order_name.startswith("neighborhood:") or order_name.endswith(":beam"):
                from .extensions import resolve_objective_scorer
                scorer = resolve_objective_scorer(config.objective, config)
                eligible = [
                    result for result in results
                    if isinstance(completed_orders[result.solver_name][0], ExtremePointSolver)
                    and (
                        not order_name.startswith("neighborhood:")
                        or result.solver_name.endswith(":greedy")
                    )
                ]
                if not eligible:
                    continue
                ranked = sorted(
                    eligible,
                    key=lambda result: (scorer.score(result), result.solver_name),
                )
                best_result = ranked[0]
                solver, base_name, order = completed_orders[best_result.solver_name]
                if order_name.startswith("neighborhood:"):
                    item_type = order_name.split(":", 1)[1]
                    order = tuple(
                        [item for item in order if item.item.id != item_type]
                        + [item for item in order if item.item.id == item_type]
                    )
                    order_name = f"{base_name}:late_{item_type}"
                else:
                    order_name = f"{base_name}:beam"
            stats = SearchStats()
            budget = deadline.slice(len(starts) - position)
            if effort_budget is not None:
                budget = budget.with_effort(effort_budget, stats)
            start_config = config
            if order_name.endswith(":greedy") or ":late_" in order_name:
                start_config = replace(
                    config,
                    max_candidates_per_item=1,
                    container_plan_beam_width=1,
                    container_plan_node_limit=1,
                )
            packed, unpacked, exhaustive, reached, dominant_lattice = self._across_containers(solver, order, containers, start_config, stats, budget)
            effort_reached = budget.effort_exceeded
            wall_reached = reached and budget.remaining_ns <= 0
            if effort_reached:
                unpacked = tuple(
                    UnpackedItem(item.instance, "effort_limit", item.details)
                    if item.reason == "time_limit" else item
                    for item in unpacked
                )
            start_id = f"{solver.name}:{order_name}"
            results.append(RawSolution(
                start_id, packed, unpacked, stats, wall_reached, effort_reached, exhaustive
            ))
            completed_orders[start_id] = (solver, order_name, order)
            records[position] = StartRecord(start_id, True, not reached, reached)
            # A single item type's densest possible packing is the regular lattice grid
            # just tried: no other solver in this portfolio can place it in fewer
            # containers or less unused volume/height once it has placed everything,
            # so remaining starts -- typically a slow per-candidate search that treats
            # every instance separately -- can only spend what is left of the deadline
            # on an answer that is provably no better.
            #
            # That claim only holds when the lattice arithmetic was actually used:
            # `GridSolver.pack_one` silently delegates to `ExtremePointSolver` per
            # container when the lattice cannot express that container's rules (or,
            # for a mixed-type request, when a caller names "grid" explicitly via
            # `config.solvers` and bypasses the orchestrator's own homogeneity check).
            # A delegated result is an ordinary beam-search answer, not a proven
            # optimum -- short-circuiting on it discarded a genuinely better start
            # whenever the delegate happened to finish (which, being time-bound, was
            # itself budget-dependent), producing a *worse* chosen packing from a
            # *larger* time budget. `dominant_lattice` is true only when
            # every container this start packed actually went through the lattice.
            if dominant_lattice and not unpacked:
                break
        return self._finish_portfolio(items, containers, deadline, results, records)

    def _solve_concurrent(self, items, containers, config, deadline, starts, effort_budget):
        """Runs every start after the first one concurrently, in separate processes.

        The first start always runs in-process, exactly as `_solve_sequential`
        would run it: `_solvers` always places the order-insensitive lattice
        first when it is eligible at all, so the grid short-circuit below
        can only ever fire on this one start, and running it in-process first
        keeps that short-circuit -- and every sequential-mode test that depends
        on it -- byte-for-byte unchanged.

        Every remaining start is submitted to a `ProcessPoolExecutor` bound to
        one shared absolute deadline (`Deadline.until`) computed once, right
        here, from however much of the portfolio's own deadline is left. Because
        `CLOCK_MONOTONIC` is machine-wide rather than per-process, every worker
        can independently compare its own fresh `monotonic_ns()` reading against
        that same instant -- the combined budget is honoured (no worker can run
        past it) without dividing it among workers the way sequential slicing
        does, since these workers genuinely run at the same time rather than
        waiting on one another.

        Each worker owns a private `SearchStats` and constructs its own
        `ContainerState` from scratch; nothing is shared between workers or with
        the parent process while a start is running. A start's outcome is
        therefore a pure function of (solver, item order, containers, config,
        effort budget, shared deadline instant) alone -- never of which worker
        the OS scheduled first or when its result happened to arrive back in the
        parent. Results are placed into `results`/`records` by each start's fixed
        position in `starts`, not by completion order, so the final list handed
        to the (order-independent, full-key) ranking in `Packer.pack` is
        identical no matter how the pool interleaves the work.
        """
        if not starts or deadline.expired:
            return self._finish_portfolio(items, containers, deadline, [], [
                StartRecord(f"{solver.name}:{order_name}", False, False, False)
                for solver, order_name, _ in starts
            ])
        # `ProcessPoolExecutor` checks the host's semaphore limits in its
        # constructor. Sandboxed and otherwise restricted runtimes may deny that
        # query even though importing multiprocessing succeeds. Concurrency is an
        # optimisation, so detect that failure before doing any start-specific
        # work and preserve the exact sequential result and deadline semantics.
        # The probe does not submit work (and therefore starts no child process);
        # it only exercises the platform capability check used by the real pool.
        try:
            probe = ProcessPoolExecutor(max_workers=1)
        except (NotImplementedError, OSError):
            return self._solve_sequential(
                items, containers, config, deadline, starts, effort_budget
            )
        else:
            probe.shutdown(wait=True)
        records = [
            StartRecord(f"{solver.name}:{order_name}", False, False, False)
            for solver, order_name, _ in starts
        ]
        results = [None] * len(starts)

        solver0, order_name0, order0 = starts[0]
        stats0 = SearchStats()
        budget0 = deadline.slice(len(starts))
        if effort_budget is not None:
            budget0 = budget0.with_effort(effort_budget, stats0)
        packed0, unpacked0, exhaustive0, reached0, dominant_lattice0 = self._across_containers(solver0, order0, containers, config, stats0, budget0)
        effort_reached0 = budget0.effort_exceeded
        wall_reached0 = reached0 and budget0.remaining_ns <= 0
        if effort_reached0:
            unpacked0 = tuple(
                UnpackedItem(item.instance, "effort_limit", item.details)
                if item.reason == "time_limit" else item
                for item in unpacked0
            )
        start_id0 = f"{solver0.name}:{order_name0}"
        results[0] = RawSolution(start_id0, packed0, unpacked0, stats0, wall_reached0, effort_reached0, exhaustive0)
        records[0] = StartRecord(start_id0, True, not reached0, reached0)

        remaining = list(enumerate(starts))[1:]
        # Same correction as the sequential path: only a genuine lattice
        # construction is provably no worse than what remains, never a GridSolver
        # delegate result -- see `_solve_sequential`'s identical check for the full
        # rationale.
        short_circuit = dominant_lattice0 and not unpacked0
        if remaining and not short_circuit and not deadline.expired:
            absolute_deadline_ns = monotonic_ns() + deadline.remaining_ns
            container_selector = self.container_selector
            worker_count = min(config.parallel_starts, len(remaining))
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _run_concurrent_start, solver, order, containers, config,
                        container_selector, effort_budget, absolute_deadline_ns,
                    ): position
                    for position, (solver, order_name, order) in remaining
                }
                outcomes = {futures[future]: future.result() for future in futures}
            for position, (solver, order_name, order) in remaining:
                packed, unpacked, exhaustive, reached, dominant_lattice, stats = outcomes[position]
                effort_reached = effort_budget is not None and effort_budget.exceeded(stats)
                wall_reached = reached and (absolute_deadline_ns - monotonic_ns()) <= 0
                if effort_reached:
                    unpacked = tuple(
                        UnpackedItem(item.instance, "effort_limit", item.details)
                        if item.reason == "time_limit" else item
                        for item in unpacked
                    )
                start_id = f"{solver.name}:{order_name}"
                results[position] = RawSolution(start_id, packed, unpacked, stats, wall_reached, effort_reached, exhaustive)
                records[position] = StartRecord(start_id, True, not reached, reached)

        results = [result for result in results if result is not None]
        return self._finish_portfolio(items, containers, deadline, results, records)

    def _finish_portfolio(self, items, containers, deadline, results, records):
        global_deadline = deadline.expired
        records = [
            replace(record, global_deadline_reached=global_deadline)
            for record in records
        ]
        if not results:
            fallback_id = "portfolio:fallback"
            results = [RawSolution(
                fallback_id,
                (),
                tuple(UnpackedItem(i, *self._unpacked_reason(i, containers, True)) for i in items),
                SearchStats(),
                True,
            )]
            records.append(StartRecord(
                fallback_id,
                True,
                True,
                False,
                global_deadline_reached=global_deadline,
            ))
        return PortfolioRun(tuple(results), tuple(records))

    def _unpacked_reason(self, instance, containers, reached):
        dimension_fit = any(
            any(
                dimensions.fits_inside(container.inner_dimensions)
                for _, dimensions in instance.dimensions.unique_rotations(
                    instance.item.allowed_rotations
                )
            )
            for container in containers
        )
        # Same geometric check with every physical orientation allowed, not only the
        # item's own restricted set: distinguishes "genuinely too big in any rotation"
        # from "this exact rotation restriction, and only it, rules every container
        # out" -- both are pure geometry, so both are provable without a complete
        # search, but they are different facts a caller needs different information to
        # act on.
        fits_with_any_rotation = any(
            any(
                dimensions.fits_inside(container.inner_dimensions)
                for _, dimensions in instance.dimensions.unique_rotations(Rotation.all())
            )
            for container in containers
        )
        weight_fit = any(
            container.max_payload is None
            or instance.weight.ticks <= container.max_payload.ticks
            for container in containers
        )
        eligible_container = (
            not instance.item.eligible_container_tags
            or any(instance.item.eligible_container_tags & container.tags for container in containers)
        )
        # A registered constraint that can prove this item fits nowhere gets to say so,
        # and to name the rule responsible. Ranked with the other structural proofs and
        # after them: a request that is geometrically impossible is impossible whatever
        # a policy says, and reporting the policy first would send a caller to fix the
        # wrong thing.
        proven_by_constraint = next(
            (
                proof
                for constraint in self.custom_constraints
                if hasattr(constraint, "proves_unplaceable")
                for proof in (constraint.proves_unplaceable(instance, containers),)
                if proof is not None
            ),
            None,
        )

        # Structural bounds remain proofs even if the search deadline was reached.
        if not fits_with_any_rotation:
            return "no_compatible_container_dimensions", ()
        if not dimension_fit:
            return "rotation_restricted", ()
        if not weight_fit:
            return "payload_exceeded", ()
        if not eligible_container:
            return "no_eligible_container", ()
        if proven_by_constraint is not None:
            return proven_by_constraint[0], (proven_by_constraint[1],)
        if reached:
            return "time_limit", ()
        if instance.item.group:
            return "group_cannot_fit_together", ()
        return "no_feasible_placement", ()

    def _support_is_the_blocker(self, instance, packed, config):
        """Post-hoc diagnosis for the `no_feasible_placement` fallback:
        true when `instance` could be placed in at least one already-packed
        container's *final* layout once `SupportConstraint` alone is set aside, but
        cannot with it in place.

        Deliberately not folded into `_unpacked_reason`'s structural checks: those
        are proofs independent of search completeness (pure geometry across every
        container), while this replays the actual layout the search produced and
        asks a real question of it via the same `find_candidates` every solver uses
        -- an *observed* fact about that one layout, not a claim that no layout
        anywhere would have worked. A different start order could, in principle,
        have produced a different final layout where this doesn't hold; see
        `ReasonProof.for_reason`'s `observed` level and `docs/PUBLIC-API.md`'s
        rejection proof section.
        """
        constraints = default_constraints(config, self.custom_constraints)
        support_constraints = tuple(c for c in constraints if isinstance(c, SupportConstraint))
        if not support_constraints:
            return False
        # The built-in support specification is a strict no-op for this item when it
        # has neither a contact rule nor an effective minimum ratio. Avoid rebuilding
        # every placement and running two candidate searches merely to rediscover that
        # it cannot be the blocker. Custom subclasses remain conservative: an override
        # may reject for semantics not represented by ``applies_to``.
        if all(type(c) is SupportConstraint and not c.applies_to(instance)
               for c in support_constraints):
            return False
        without_support = tuple(c for c in constraints if not isinstance(c, SupportConstraint))
        # A small, fixed local budget -- independent of the caller's own deadline,
        # which may already be exhausted -- so this diagnostic can never itself hang
        # the solve; it is a best-effort explanation, not a required step.
        budget = Deadline(2_000)
        for packed_container in packed:
            if budget.expired:
                return False
            state = ContainerState(packed_container.container, packed_container.sequence)
            for placement in packed_container.placements:
                state.add(placement)
            if not find_candidates(state, instance, config, without_support, SearchStats(), budget, 1):
                continue
            if not find_candidates(state, instance, config, constraints, SearchStats(), budget, 1):
                return True
        return False

    def _across_containers(self, solver, items, containers, config, stats, deadline):
        beam = (
            config.container_plan_beam_width > 1
            and isinstance(self.container_selector, DefaultContainerSelector)
        )
        self._record_root_bound(solver, beam, items, containers, config, stats)
        if beam:
            return self._across_containers_beam(
                solver, items, containers, config, stats, deadline
            )
        return self._across_containers_greedy(
            solver, items, containers, config, stats, deadline
        )

    @staticmethod
    def _packed_state(state: ContainerState) -> tuple[PackedContainer, set[str]]:
        container = state.container
        if state.lattice_summary is not None:
            packed = PackedContainer(
                container, state.sequence, (), state.lattice_summary, state.lattice_items
            )
            ids = {item.id for item in state.lattice_items}
        else:
            packed = PackedContainer(
                container, state.sequence, _with_top_loads(state.placements)
            )
            ids = {placement.instance.id for placement in state.placements}
        return packed, ids

    @staticmethod
    def _score_plan(packed, remaining, config) -> tuple[int, ...]:
        # Late import avoids making the solver layer depend on the extension registry
        # at module import time; objective scorers themselves depend only on domain VOs.
        from .extensions import resolve_objective_scorer

        unpacked = tuple(UnpackedItem(item, "search_exhausted") for item in remaining)
        scorer = resolve_objective_scorer(config.objective, config)
        return tuple(scorer.score_containers(packed, unpacked))

    @staticmethod
    def _additional_container_lower_bound(plan, containers) -> int:
        """Admissible volume/weight lower bound for unopened containers.

        The largest still-available capacity is granted to every future container,
        which can only underestimate how many are required.  Nominal volume is not
        additive for nesting, so that half is disabled in that case; payload remains
        exact and additive.
        """
        available = [
            container for container in containers
            if plan.inventory[container.id] is None or plan.inventory[container.id] > 0
        ]
        if not plan.remaining or not available:
            return 0
        lower = 0
        if not any(item.item.nesting_height is not None for item in plan.remaining):
            maximum_volume = max(c.inner_dimensions.volume for c in available)
            if maximum_volume > 0:
                total_volume = sum(item.dimensions.volume for item in plan.remaining)
                lower = max(lower, (total_volume + maximum_volume - 1) // maximum_volume)
        finite_payloads = [
            c.max_payload.ticks for c in available if c.max_payload is not None
        ]
        if len(finite_payloads) == len(available) and finite_payloads:
            maximum_payload = max(finite_payloads)
            if maximum_payload > 0:
                total_weight = sum(item.weight.ticks for item in plan.remaining)
                lower = max(lower, (total_weight + maximum_payload - 1) // maximum_payload)
        return lower

    def _plan_bound_key(self, plan, containers, config) -> tuple:
        partial = list(self._score_plan(plan.packed, (), config))
        count_index = 1 if config.objective == "default" else 2
        partial[count_index] += self._additional_container_lower_bound(plan, containers)
        remaining_signature = tuple(item.id for item in plan.remaining)
        packing_signature = tuple(container.id for container in plan.packed)
        return (*partial, remaining_signature, packing_signature)

    @staticmethod
    def _record_root_bound(solver, beam, items, containers, config, stats) -> None:
        """Compute the request-level lower bound once, at the root.

        The plan beam already computes the same relaxation *per node* to prune with -- the
        request-level bound is that formula with the state empty, which is why this costs one
        sort rather than a second search. `exact_small` gets it for the opposite reason: it
        exhausts its candidate set and can say the incumbent is optimal over that set, and a
        bound is what turns "I stopped looking" into a statement about the request.

        Only for the default objective. `lowest_cost` and `maximum_value` order their score
        keys differently, so a bound vector compared against them would line up cost against
        container count -- not a weaker claim, a meaningless one.

        Nothing downstream reads this yet, and that is deliberate: the number is visible to
        the engines and to nothing else until a contract freeze decides whether a caller ever
        sees a gap.
        """
        if stats.objective_lower_bound is not None:
            return
        if not (beam or getattr(solver, "name", None) == "exact_small"):
            return
        if config.objective != "default":
            return
        try:
            stats.objective_lower_bound = bounds.compute(items, containers).as_tuple()
        except bounds.BoundOverflowError:
            # The bound refuses past its declared ceiling. Nothing reads it yet, so
            # a refusal leaves it unset rather than failing a pack that is otherwise fine --
            # the refusal is the bound declining to answer, not the request being invalid.
            # When a caller-facing gap field exists, that field carries the refusal instead.
            stats.objective_lower_bound = None


    def _across_containers_beam(self, solver, items, containers, config, stats, deadline):
        """Bounded deterministic beam over partial multi-container plans.

        Each child commits one independently trial-packed container.  Closed
        containers never affect future geometry, so plans with identical remaining
        instances and inventory have identical continuations; retaining only the
        lexicographically best partial objective is therefore exact dominance pruning.
        """
        ordered = sorted(containers, key=lambda c: (c.cost_minor, c.inner_dimensions.volume, c.id))
        maximum = config.max_containers if config.max_containers is not None else sum(
            c.quantity if c.quantity is not None else len(items) for c in containers
        )
        initial = _ContainerPlan(
            (), tuple(items), {c.id: c.quantity for c in containers},
            {c.id: 0 for c in containers},
        )
        beam = [initial]
        incumbent = initial
        plan_nodes = 0
        reached = False

        while beam and plan_nodes < config.container_plan_node_limit:
            expansions: list[_ContainerPlan] = []
            for plan in beam:
                if not plan.remaining or len(plan.packed) >= maximum:
                    if self._score_plan(plan.packed, plan.remaining, config) < self._score_plan(
                        incumbent.packed, incumbent.remaining, config
                    ):
                        incumbent = plan
                    continue
                for container in ordered:
                    if plan_nodes >= config.container_plan_node_limit:
                        break
                    if deadline.expired:
                        reached = True
                        break
                    available = plan.inventory[container.id]
                    if available is not None and available <= 0:
                        continue
                    plan_nodes += 1
                    try:
                        one = solver.pack_one(
                            container, plan.sequences[container.id] + 1,
                            plan.remaining, config, stats, deadline,
                        )
                    except TimeLimitReached:
                        reached = True
                        break
                    reached = reached or one.time_limit_reached
                    if one.state.placement_count == 0:
                        continue
                    packed_container, ids = self._packed_state(one.state)
                    inventory = dict(plan.inventory)
                    if inventory[container.id] is not None:
                        inventory[container.id] -= 1
                    sequences = dict(plan.sequences)
                    sequences[container.id] += 1
                    child = _ContainerPlan(
                        (*plan.packed, packed_container),
                        tuple(item for item in plan.remaining if item.id not in ids),
                        inventory,
                        sequences,
                        plan.exhaustive and one.exhaustive,
                        plan.dominant_lattice and one.dominant_lattice,
                    )
                    expansions.append(child)
                    if self._score_plan(child.packed, child.remaining, config) < self._score_plan(
                        incumbent.packed, incumbent.remaining, config
                    ):
                        incumbent = child
                    if reached:
                        break
                if reached:
                    break
            if reached or not expansions:
                break

            # Exact dominance: same remaining instances plus the same inventory means
            # the continuation set is identical.  A worse closed prefix can never catch
            # up because every objective component is additive and non-negative.
            dominant: dict[tuple, _ContainerPlan] = {}
            for plan in expansions:
                signature = (
                    tuple(item.id for item in plan.remaining),
                    tuple((key, plan.inventory[key]) for key in sorted(plan.inventory)),
                )
                previous = dominant.get(signature)
                if previous is None or self._score_plan(plan.packed, (), config) < self._score_plan(
                    previous.packed, (), config
                ):
                    dominant[signature] = plan
            beam = sorted(
                dominant.values(),
                key=lambda plan: self._plan_bound_key(plan, containers, config),
            )[:config.container_plan_beam_width]

        unpacked = []
        for instance in incumbent.remaining:
            reason, details = self._unpacked_reason(instance, containers, reached)
            if reason == "no_feasible_placement" and self._support_is_the_blocker(
                instance, incumbent.packed, config
            ):
                reason = "insufficient_support"
            unpacked.append(UnpackedItem(instance, reason, details))
        proven = (
            incumbent.exhaustive and not reached and not unpacked
            and len(incumbent.packed) == 1
        )
        return (
            incumbent.packed, tuple(unpacked), proven, reached,
            incumbent.dominant_lattice,
        )

    def _across_containers_greedy(self, solver, items, containers, config, stats, deadline):
        remaining = list(items); packed = []; reached = False; exhaustive = True; dominant_lattice = True
        inventory = {c.id: c.quantity for c in containers}; sequences = {c.id: 0 for c in containers}
        maximum = config.max_containers if config.max_containers is not None else sum(c.quantity if c.quantity is not None else len(items) for c in containers)
        ordered_containers = sorted(containers, key=lambda c: (c.cost_minor, c.inner_dimensions.volume, c.id))
        while remaining and len(packed) < maximum:
            if deadline.expired: reached = True; break
            # `remaining` is immutable across one selection round, so the tuple the
            # per-template trials receive is built once per round, not once per
            # template; the winner is tracked as a running first-minimum instead of
            # retaining every trial's full container state until the round ends.
            round_items = tuple(remaining)
            best_score = best = None
            for container in ordered_containers:
                if inventory[container.id] is not None and inventory[container.id] <= 0: continue
                try:
                    one = solver.pack_one(container, sequences[container.id] + 1, round_items, config, stats, deadline)
                except TimeLimitReached:
                    reached = True
                    break
                if one.time_limit_reached:
                    reached = True
                if one.state.placement_count == 0:
                    if reached:
                        break
                    continue
                if (
                    config.objective in ("shipping_cost", "lowest_landed_cost")
                    and config.dimensional_weight_divisor is not None
                ):
                    dimensions = container.outer_dimensions or container.inner_dimensions
                    dim_weight = dimensional_weight(
                        dimensions,
                        config.dimensional_weight_divisor,
                        config.dimensional_weight_length_unit,
                        config.dimensional_weight_weight_unit,
                    ).ticks
                    # `payload_ticks` and `placement_count` are lattice-aware: the
                    # compact path carries no per-item placements, so summing
                    # `state.placements` here priced a quantity-compressed trial as
                    # tare alone and let an unpriceable container win the round.
                    gross = container.tare_weight.ticks + one.state.payload_ticks
                    billed = max(gross, dim_weight)
                    placed = one.state.placement_count
                    if config.objective == "shipping_cost":
                        score = (-placed, billed, container.id)
                    else:
                        # Landed cost ranks the round by the money the finished score will
                        # charge, not by the grams it is derived from. This loop commits
                        # the trial verbatim, so its billed weight is already final and the
                        # tariff can be read now; a bracket step or a minimum charge makes
                        # the cheaper shipment the heavier one, and a trial the tariff
                        # cannot price is unshippable rather than merely dear, so it sorts
                        # behind every priceable alternative. Keys after the charge
                        # mirror Rust's, so the two engines pick the same container.
                        charge = (
                            container.rate_table.charge_minor_or_none(_grams(billed))
                            if container.rate_table is not None
                            else None
                        )
                        containers_needed = -(-len(round_items) // max(placed, 1))
                        score = (
                            UNPRICEABLE_MINOR if charge is None else charge,
                            containers_needed,
                            -placed,
                            container.id,
                        )
                else:
                    score = self.container_selector.score(container, one)
                if best_score is None or score < best_score:
                    best_score, best = score, one
                if reached:
                    break
            if best is None: break
            c = best.state.container; sequences[c.id] += 1
            if trace.active():
                trace.emit({"type": "container_selection", "container_id": c.id, "sequence": sequences[c.id], "cost_minor": c.cost_minor, "reason": "cheapest_eligible"})
            exhaustive = exhaustive and best.exhaustive
            dominant_lattice = dominant_lattice and best.dominant_lattice
            if inventory[c.id] is not None: inventory[c.id] -= 1
            committed, ids = self._packed_state(best.state)
            packed.append(committed)
            remaining = [i for i in remaining if i.id not in ids]
            if reached: break
        unpacked = []
        for instance in remaining:
            reason, details = self._unpacked_reason(instance, containers, reached)
            if reason == "no_feasible_placement" and self._support_is_the_blocker(instance, packed, config):
                reason = "insufficient_support"
            unpacked.append(UnpackedItem(instance, reason, details))
        # Optimality may only be claimed when an exhaustive search filled a single
        # container -- the cheapest one, since containers are tried in cost order.
        proven = exhaustive and not reached and not unpacked and len(packed) == 1
        return tuple(packed), tuple(unpacked), proven, reached, dominant_lattice
