"""Packing dependency graphs and safe-order replay validation.

Loading and unloading are distinct directed graphs over placement indices, not one
graph read two ways -- the reopen reason for both tasks was exactly that conflation:

- **Unloading** (`UnloadingDependencyGraph`, `safe_removal_order`): replay begins with
  the full, final scene. An item depends on its *children* (whatever rests on top of
  it) -- those must come off first -- and is blocked in a given direction if another
  still-present item's envelope lies between it and the outside of the container.
- **Loading** (`LoadingDependencyGraph`, `safe_loading_order`): replay begins with an
  empty container. An item depends on its *supporters* (whatever it rests on) -- those
  must already be present -- and is blocked in a given direction if inserting it along
  that direction would have to pass through an item already loaded.

Both share one accessibility primitive. `direction` names which container wall's
region a box's sweep occupies (its own face to that wall), not a direction of travel:
the region between a box and its `+x` wall is the same region whether the box is
*leaving* through it (unloading, sliding toward `+x`) or *arriving* through it
(loading, coming from outside and sliding toward the box's own resting `-x` side).
`_blocked` answers "is this region clear of every other present item" identically for
both; the two graphs differ only in which items count as present at each step and
which structural dependency (children vs. supporters) gates the step at all.

Deliberately scoped to geometry: contact-derived support and axis-aligned accessibility
sweeps, the same primitives `ContactGraph` and the solver's own collision
check already share. Re-validating an item's own physical rules (`minimum_support_ratio`
as a fraction, `max_stacked_items`, `max_top_load`) at each loading step would need
`Item`/`Placement` objects threaded through in place of bare `AxisAlignedBox`, a
materially larger change than the loading/unloading gap this task was reopened over;
those rules are already checked once, for the finished scene, by whichever process
produced it (the solver's own placement constraints, or `IndependentSolutionValidator`
for a hand-built scene) -- not re-verified here.
"""

from __future__ import annotations

from ._compat import dataclass
from typing import Sequence

from .constraints import (_touches_corners, direct_support_view, load_units, overloaded,
                          stack_density_exceeded, stack_limit_exceeded, stacked_counts)
from .contact import ContactGraph
from .geometry import AxisAlignedBox, Dimensions
from .models import Container, Placement

ALL_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")


class InvalidDirectionError(ValueError):
    """A direction outside the six-value vocabulary was supplied. Rejected rather than
    silently treated as one of the six -- `-z` in particular, since that was this
    module's own previous (wrong) default for anything unrecognised."""

    code = "invalid_direction"

    def __init__(self, direction: str):
        self.direction = direction
        super().__init__(f"unknown movement direction {direction!r}; expected one of {ALL_DIRECTIONS}")

    def to_dict(self) -> dict:
        return {"code": self.code, "direction": self.direction}


def _validated(directions: Sequence[str]) -> Sequence[str]:
    for direction in directions:
        if direction not in ALL_DIRECTIONS:
            raise InvalidDirectionError(direction)
    return directions


class SequenceError(Exception):
    """No safe order exists: every remaining placement is blocked in every allowed
    direction, for whichever graph raised this (loading or unloading). Carries the
    indices still stuck when the search stopped, not just the first one, since a real
    deadlock is usually a mutual one."""

    code = "sequence_stuck"

    def __init__(self, stuck: frozenset[int]):
        self.stuck = stuck
        super().__init__(f"no safe order exists: placements {sorted(stuck)} are mutually blocking")

    def to_dict(self) -> dict:
        return {"code": self.code, "stuck": sorted(self.stuck)}


class RouteSequenceError(Exception):
    """No safe order exists that fully empties `stop` before the route moves on to the
    next one -- every placement still due there is blocked, whether by another
    placement due at the same stop, one due at a later stop that has not left yet, or
    one with no stop assigned at all. Distinct from `SequenceError`: that one
    means no *unconstrained* order exists at all; this means the constrained,
    stop-by-stop order the route itself demands does not."""

    def __init__(self, stop: int, stuck: frozenset[int]):
        self.stop = stop
        self.stuck = stuck
        super().__init__(f"stop {stop}: placements {sorted(stuck)} cannot be unloaded there")


def _swept_volume(box: AxisAlignedBox, container: Dimensions, direction: str) -> tuple[int, int, int, int, int, int]:
    """The region between `box`'s own face and the matching container wall along
    `direction` -- identical whether a box leaves through that wall (unloading) or
    arrives through it (loading)."""
    x1, y1, z1, x2, y2, z2 = box.origin.x, box.origin.y, box.origin.z, box.x2, box.y2, box.z2
    if direction == "+x":
        x1 = x2
        x2 = container.length.ticks
    elif direction == "-x":
        x2 = x1
        x1 = 0
    elif direction == "+y":
        y1 = y2
        y2 = container.width.ticks
    elif direction == "-y":
        y2 = y1
        y1 = 0
    elif direction == "+z":
        z1 = z2
        z2 = container.height.ticks
    elif direction == "-z":
        z2 = z1
        z1 = 0
    else:
        raise InvalidDirectionError(direction)
    return x1, y1, z1, x2, y2, z2


def _blocking_indices(index: int, box: AxisAlignedBox, boxes: Sequence[AxisAlignedBox],
                       present: frozenset[int], container: Dimensions, direction: str) -> frozenset[int]:
    """Every other currently-present box whose envelope intersects `box`'s `direction`
    sweep -- the evidence `_blocked` reduces to a bare boolean."""
    sx1, sy1, sz1, sx2, sy2, sz2 = _swept_volume(box, container, direction)
    return frozenset(
        other_index for other_index in present
        if other_index != index
        and sx1 < boxes[other_index].x2 and boxes[other_index].origin.x < sx2
        and sy1 < boxes[other_index].y2 and boxes[other_index].origin.y < sy2
        and sz1 < boxes[other_index].z2 and boxes[other_index].origin.z < sz2
    )


def _blocked(index: int, box: AxisAlignedBox, boxes: Sequence[AxisAlignedBox],
             present: frozenset[int], container: Dimensions, direction: str) -> bool:
    """Would `box`'s `direction` sweep collide with any other currently-present box?

    Same predicate as `_blocking_indices`, but stopping at the first blocker: the
    boolean callers (the safe-order search and reachability sweeps) ask it O(n*d)
    times per step and never read the set, which only the evidence paths need.
    """
    sx1, sy1, sz1, sx2, sy2, sz2 = _swept_volume(box, container, direction)
    return any(
        other_index != index
        and sx1 < boxes[other_index].x2 and boxes[other_index].origin.x < sx2
        and sy1 < boxes[other_index].y2 and boxes[other_index].origin.y < sy2
        and sz1 < boxes[other_index].z2 and boxes[other_index].origin.z < sz2
        for other_index in present
    )


def _clear_direction(index: int, box: AxisAlignedBox, boxes: Sequence[AxisAlignedBox],
                      present: frozenset[int], container: Dimensions,
                      directions: Sequence[str]) -> str | None:
    """The first allowed direction (in the order given) whose sweep is clear, or None
    if every one is blocked. Returned rather than a bare bool so a caller building step
    evidence knows *which* direction the step actually used."""
    for direction in directions:
        if not _blocked(index, box, boxes, present, container, direction):
            return direction
    return None


def _validate_box_at_step(index: int, step: int, boxes: Sequence[AxisAlignedBox],
                          present: frozenset[int], container: Dimensions) -> None:
    """Independently enforce the two final-scene invariants a sweep alone cannot
    prove: containment and non-overlap with placements already present."""
    box = boxes[index]
    if (box.origin.x < 0 or box.origin.y < 0 or box.origin.z < 0
            or box.x2 > container.length.ticks
            or box.y2 > container.width.ticks
            or box.z2 > container.height.ticks):
        raise SequenceReplayError(index, step, "placement is outside the container")
    if any(other != index and box.intersects(boxes[other]) for other in present):
        raise SequenceReplayError(
            index, step, "placement collides with an already present placement"
        )


def _acyclic(depends_on: tuple[frozenset[int], ...]) -> bool:
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> bool:
        if node in visited:
            return True
        if node in visiting:
            return False
        visiting.add(node)
        for dependency in depends_on[node]:
            if not visit(dependency):
                return False
        visiting.discard(node)
        visited.add(node)
        return True

    return all(visit(node) for node in range(len(depends_on)))


@dataclass(frozen=True, slots=True)
class UnloadingDependencyGraph:
    """`depends_on[i]` is the set of placement indices that must be removed before `i`
    -- `i`'s children (`ContactGraph`): whatever rests on `i` sits strictly
    above it, so following edges strictly increases height and a finite set cannot
    cycle back on itself (checked, not assumed, by `is_acyclic`)."""

    depends_on: tuple[frozenset[int], ...]

    @classmethod
    def build(cls, boxes: Sequence[AxisAlignedBox]) -> "UnloadingDependencyGraph":
        contacts = ContactGraph(boxes)
        return cls(tuple(frozenset(contacts.children(index)) for index in range(len(boxes))))

    def is_acyclic(self) -> bool:
        return _acyclic(self.depends_on)


@dataclass(frozen=True, slots=True)
class LoadingDependencyGraph:
    """`depends_on[i]` is the set of placement indices that must already be present
    before `i` can be loaded -- `i`'s supporters (`ContactGraph`): whatever
    `i` rests on sits strictly below it, so following edges strictly *decreases*
    height and a finite set cannot cycle back on itself either, by the same argument
    run in reverse."""

    depends_on: tuple[frozenset[int], ...]

    @classmethod
    def build(cls, boxes: Sequence[AxisAlignedBox]) -> "LoadingDependencyGraph":
        contacts = ContactGraph(boxes)
        return cls(tuple(
            frozenset(edge.index for edge in contacts.supporters(index))
            for index in range(len(boxes))
        ))

    def is_acyclic(self) -> bool:
        return _acyclic(self.depends_on)


def safe_removal_order(boxes: Sequence[AxisAlignedBox], container: Dimensions,
                       directions: Sequence[str] = ALL_DIRECTIONS) -> list[int]:
    """A concrete order placements could be removed in, starting from the full scene,
    without ever needing to move a still-present placement out of the way first.
    Raises `SequenceError` if no such order exists for the given `directions`, and
    `InvalidDirectionError` if `directions` names anything outside the six-value
    vocabulary."""
    _validated(directions)
    graph = UnloadingDependencyGraph.build(boxes)
    present = frozenset(range(len(boxes)))
    order: list[int] = []
    while present:
        removable = [
            index for index in present
            if not (graph.depends_on[index] & present)
            and _clear_direction(index, boxes[index], boxes, present, container, directions) is not None
        ]
        if not removable:
            raise SequenceError(present)
        # Deterministic: the lowest index among this step's candidates, not
        # arbitrary set iteration order.
        chosen = min(removable)
        order.append(chosen)
        present = present - {chosen}
    replay_removal_order(boxes, container, order, directions)
    return order


def safe_route_removal_order(boxes: Sequence[AxisAlignedBox], stops: Sequence[int | None],
                             container: Dimensions,
                             directions: Sequence[str] = ALL_DIRECTIONS) -> list[int]:
    """A concrete unloading order that fully empties each stop, in ascending stop order,
    before the next one starts -- a route check and a
    restriction to axis-aligned horizontal/vertical exits both fall directly out of
    `safe_removal_order`'s existing machinery, since every `directions` sweep here is
    already exactly one of those two kinds; the only thing missing was the route itself.

    `stops[i]` is the stop index placement `i` is due at; `None` means placement `i` is
    not on the route at all -- it is never scheduled for removal and stays present for
    every step of the replay, exactly like a fixture that rides the whole route, but it
    still counts as a potential blocker for anything scheduled around it. A request
    with every `stops[i]` set to `None` schedules nothing and returns `[]`, leaving
    single-stop callers (which never populate this at all) completely unaffected.

    Raises `RouteSequenceError` at the first stop that cannot be fully unloaded --
    carrying that stop and the placements still stuck there, not just the first one,
    since a real deadlock is usually mutual -- and `InvalidDirectionError` if
    `directions` names anything outside the six-value vocabulary.
    """
    _validated(directions)
    graph = UnloadingDependencyGraph.build(boxes)
    remaining = frozenset(range(len(boxes)))
    order: list[int] = []
    for stop in sorted({stop for stop in stops if stop is not None}):
        due = frozenset(index for index in remaining if stops[index] == stop)
        while due:
            removable = [
                index for index in due
                if not (graph.depends_on[index] & remaining)
                and _clear_direction(index, boxes[index], boxes, remaining, container, directions) is not None
            ]
            if not removable:
                raise RouteSequenceError(stop, due)
            # Deterministic: the lowest index among this step's candidates, not
            # arbitrary set iteration order.
            chosen = min(removable)
            order.append(chosen)
            remaining = remaining - {chosen}
            due = due - {chosen}
    return order


def safe_loading_order(boxes: Sequence[AxisAlignedBox], container: Dimensions,
                       directions: Sequence[str] = ALL_DIRECTIONS) -> list[int]:
    """A concrete order placements could be loaded in, starting from an empty
    container, so that at every step the item being added already has every one of its
    supporters present and a collision-free insertion sweep in some allowed direction
    against only what has been loaded so far. Raises `SequenceError` if no such order
    exists for the given `directions`, and `InvalidDirectionError` if `directions`
    names anything outside the six-value vocabulary.

    Computed as the exact reverse of a valid unloading order for the same scene and
    `directions`, not a second, independent forward search -- and this is a proof, not
    a shortcut. `safe_removal_order`'s greedy choice is trap-free because unloading
    only ever *shrinks* the present set, so a step that is clear now stays clear
    forever; a naive forward loading search does not have that property, since loading
    only ever *grows* the present set, and an early greedy choice can permanently
    block a later item's only clear direction (measured directly: two side-by-side
    items with only one exit allowed can deadlock a naive greedy loader that placed the
    easy one first). Reversal sidesteps that trap entirely: `UnloadingDependencyGraph`
    only ever depends on a box's *children*, so `s` supporting `r` puts `r` in `s`'s
    children and forces `r` out before `s` in any valid unloading order `R` -- meaning
    `s` precedes `r` in `reversed(R)`, exactly `LoadingDependencyGraph`'s own
    supporters-first requirement. The accessibility sweep is symmetric by construction
    (`_swept_volume`/`_blocked` measure the same region whether a box is leaving
    through a wall or arriving through it), so a step clear for removal at position `k`
    of `R` is clear for loading at `reversed(R)`'s matching position against the
    identical present set. `replay_loading_order` re-derives and checks this
    independently against `LoadingDependencyGraph` and a real forward simulation --
    it does not call this function or trust the reversal, so a mistake in this
    reasoning could not hide behind it.
    """
    _validated(directions)
    order = list(reversed(safe_removal_order(boxes, container, directions)))
    replay_loading_order(boxes, container, order, directions)
    return order


def safe_loading_order_for_placements(
    placements: Sequence[Placement],
    container: Container,
    directions: Sequence[str] = ALL_DIRECTIONS,
) -> list[int]:
    """Return one loading order that is safe geometrically *and* under every
    placement/container business rule.

    The older :func:`safe_loading_order` remains the explicitly geometry-only API for
    callers that have bare boxes. Callers with real packing-domain objects should use
    this composed entry point so forgetting a second rule-replay call is impossible.
    """
    boxes = [placement.envelope_box for placement in placements]
    order = safe_loading_order(boxes, container.inner_dimensions, directions)
    verify_loading_prefix_business_rules(placements, order, container)
    return order


class SequenceReplayError(Exception):
    """A given order includes a step that is not actually feasible against only the
    placements present at that point in the replay (loading: already loaded;
    unloading: not yet removed). Distinct from `SequenceError` (which means *no* order
    exists at all): this means the specific `order` supplied is wrong, whether or not
    some other order would have worked."""

    code = "sequence_replay"

    def __init__(self, index: int, step: int, reason: str):
        self.index = index
        self.step = step
        self.reason = reason
        super().__init__(f"step {step}: placement {index} is not safe there ({reason})")

    def to_dict(self) -> dict:
        return {"code": self.code, "index": self.index, "step": self.step, "reason": self.reason}


def replay_removal_order(boxes: Sequence[AxisAlignedBox], container: Dimensions, order: Sequence[int],
                         directions: Sequence[str] = ALL_DIRECTIONS) -> None:
    """Independently replay an unloading `order` (full scene to empty) and raise
    `SequenceReplayError` at the first step that is not actually feasible given only
    what remains present at that point. Reuses the same geometric primitives
    `safe_removal_order` is built from -- there is no second, possibly-disagreeing
    notion of "blocked" to maintain -- but none of its search or tie-break logic, so it
    can catch a generator bug, or a hand-authored or externally-supplied order, that
    produces a plausible-looking but wrong sequence."""
    _validated(directions)
    if sorted(order) != list(range(len(boxes))):
        raise SequenceReplayError(-1, -1, "order is not a permutation of every placement index exactly once")
    graph = UnloadingDependencyGraph.build(boxes)
    present = frozenset(range(len(boxes)))
    for step, index in enumerate(order):
        _validate_box_at_step(index, step, boxes, present, container)
        if graph.depends_on[index] & present:
            raise SequenceReplayError(index, step, "something still resting on it has not been removed yet")
        if _clear_direction(index, boxes[index], boxes, present, container, directions) is None:
            raise SequenceReplayError(index, step, "no allowed direction is clear of the remaining placements")
        present = present - {index}


def replay_loading_order(boxes: Sequence[AxisAlignedBox], container: Dimensions, order: Sequence[int],
                         directions: Sequence[str] = ALL_DIRECTIONS) -> None:
    """Independently replay a loading `order` (empty container to full scene) and
    raise `SequenceReplayError` at the first step that is not actually feasible given
    only what has already been loaded at that point. The loading counterpart to
    `replay_removal_order`, sharing the same primitives and the same independence from
    `safe_loading_order`'s own search logic."""
    _validated(directions)
    if sorted(order) != list(range(len(boxes))):
        raise SequenceReplayError(-1, -1, "order is not a permutation of every placement index exactly once")
    graph = LoadingDependencyGraph.build(boxes)
    present: frozenset[int] = frozenset()
    for step, index in enumerate(order):
        _validate_box_at_step(index, step, boxes, present, container)
        if not (graph.depends_on[index] <= present):
            raise SequenceReplayError(index, step, "a supporter has not been loaded yet")
        if _clear_direction(index, boxes[index], boxes, present, container, directions) is None:
            raise SequenceReplayError(index, step, "no allowed direction is clear of what has already been loaded")
        present = present | {index}


@dataclass(frozen=True, slots=True)
class SequenceStep:
    """One step of a generated order, carrying the evidence the acceptance calls for
    alongside the bare index: which other placements this step structurally depended
    on (support), and which specific direction its accessibility check actually used.
    A companion to the plain `list[int]` the base generators return -- not a
    replacement for it, since most callers (ordering a scene, driving playback) only
    ever need the index sequence, and `SequenceStep`'s evidence is re-derived here by
    literally replaying the already-computed order through the same independent
    primitives `replay_removal_order`/`replay_loading_order` use, not threaded through
    the generator's own search."""

    index: int
    direction: str
    depends_on: frozenset[int]

    def to_dict(self) -> dict:
        """Canonical JSON shape shared with the PHP, Rust and JavaScript
        implementations: `index`, `direction` and `depends_on` as a
        sorted list, matching `conformance/scene/sequence-fixtures.json`."""
        return {
            "index": self.index,
            "direction": self.direction,
            "depends_on": sorted(self.depends_on),
        }

@dataclass(frozen=True, slots=True)
class SequenceWarning:
    """Non-fatal, localization-ready instruction warning."""

    code: str
    index: int
    message_key: str
    arguments: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "index": self.index,
            "message_key": self.message_key,
            "arguments": dict(sorted(self.arguments.items())),
        }


def safe_removal_order_with_evidence(boxes: Sequence[AxisAlignedBox], container: Dimensions,
                                     directions: Sequence[str] = ALL_DIRECTIONS) -> list[SequenceStep]:
    """`safe_removal_order`, with each step's support dependency and the accessibility
    direction it actually used attached as evidence."""
    order = safe_removal_order(boxes, container, directions)
    graph = UnloadingDependencyGraph.build(boxes)
    present = frozenset(range(len(boxes)))
    steps = []
    for index in order:
        direction = _clear_direction(index, boxes[index], boxes, present, container, directions)
        assert direction is not None  # safe_removal_order already proved this step feasible
        steps.append(SequenceStep(index, direction, graph.depends_on[index]))
        present = present - {index}
    return steps


def safe_loading_order_with_evidence(boxes: Sequence[AxisAlignedBox], container: Dimensions,
                                     directions: Sequence[str] = ALL_DIRECTIONS) -> list[SequenceStep]:
    """`safe_loading_order`, with each step's support dependency and the accessibility
    direction it actually used attached as evidence."""
    order = safe_loading_order(boxes, container, directions)
    graph = LoadingDependencyGraph.build(boxes)
    present: frozenset[int] = frozenset()
    steps = []
    for index in order:
        direction = _clear_direction(index, boxes[index], boxes, present, container, directions)
        assert direction is not None  # safe_loading_order already proved this step feasible
        steps.append(SequenceStep(index, direction, graph.depends_on[index]))
        present = present | {index}
    return steps


def verify_loading_prefix_business_rules(
    placements: Sequence[Placement], order: Sequence[int], container: Container,
) -> None:
    """Independently reuse the exact constraint calculations already proven for a
    finished scene (`constraints.load_units`/`overloaded`/`stack_limit_exceeded`/
    `stack_density_exceeded`/`stacked_counts`) against every loading *prefix*, not
    only the final state.

    The module docstring's original reopen-scope note said re-checking `max_top_load`,
    `max_stacked_items`, `stackable` and a density limit "would need `Item`/`Placement`
    objects threaded through in place of bare `AxisAlignedBox`" -- this is that
    follow-up, additive rather than a change to the existing bare-geometry functions
    above (`safe_loading_order`, `replay_loading_order`, ...), which every current
    caller keeps using unmodified.

    Raises `SequenceReplayError` at the first step whose prefix violates a limit --
    pinned to that step even though these particular rules only ever accumulate as
    loading proceeds (so a violation present at step `k` is also present in the final
    scene): identifying *which* addition first broke a limit is strictly more useful
    to a caller than "the finished scene is invalid" alone, and is the reason this
    walks the prefix sequence instead of checking only the last step.

    `ground_contact_rule`'s FREE/COVERED/SINGLE/MULTIPLE semantics are re-derived here
    directly from `present` (boxes whose top face meets the new item's base, exactly
    `SupportConstraint.evaluate`'s own definition of a supporting surface) rather than
    through a spatial index: at prefix-replay time `present` is already the complete
    set of what a spatial index would be asked for, so building one would only add
    indirection, not precision.
    """
    if sorted(order) != list(range(len(placements))):
        raise SequenceReplayError(-1, -1, "order is not a permutation of every placement index exactly once")
    max_density_ticks = None if container.max_stack_density is None else container.max_stack_density.ticks
    present: list[Placement] = []
    for step, index in enumerate(order):
        present.append(placements[index])
        units = load_units(present)
        problem = overloaded(units) or stack_limit_exceeded(units) or stack_density_exceeded(units, max_density_ticks)
        if problem is None:
            counts = stacked_counts(units)
            for unit_index, unit in enumerate(units):
                item = present[unit_index].instance.item
                if not item.stackable and counts[unit_index] > 0:
                    problem = ("non_stackable_item_has_load", unit.label)
                    break
        if problem is None:
            candidate = present[-1]
            rule = candidate.instance.item.ground_contact_rule
            if rule is not None and rule != "free" and candidate.envelope_box.origin.z != 0:
                candidate_box = candidate.envelope_box
                support = direct_support_view(present[:-1], candidate.instance, candidate_box)
                surfaces = support.surfaces
                supporters = [entry.box for entry in support.supporters]
                violated = (
                    (rule == "covered" and not _touches_corners(candidate_box, surfaces))
                    or (rule == "single" and len(supporters) != 1)
                    or (rule == "multiple" and len(supporters) < 2)
                )
                if violated:
                    problem = ("ground_contact_violation", candidate.instance.id)
        if problem is not None:
            code, label = problem
            raise SequenceReplayError(index, step, f"{code}: {label}")


@dataclass(frozen=True, slots=True)
class Reachability:
    """Whether placement `index` could be reached and removed *right now*, given the
    full scene as it stands, its contact-graph dependents (`ContactGraph`)
    and the declared unloading route -- independent of whether a complete
    order exists for the *whole* load (`safe_removal_order`/`safe_route_removal_order`
    answer that separately, and can require moving other items first). A snapshot
    query: "if the door opened right now, what could actually come out first."

    A geometrically valid packing (passes placement/collision/support checks) can
    still box an item in on every exposed side, or place it behind an earlier-route
    stop's cargo -- `reachable` is `False` for exactly those placements, and each
    `blocked_by_*` set names which other placements are responsible, not just
    whether one is.

    `reachable` is true only when none of the three block it: nothing from
    `blocked_by_support` (still-present children resting on top) is
    present, every allowed exit sweep is not simultaneously blocked
    (`blocked_by_neighbors` is only populated when *every* allowed direction is
    blocked -- a single clear direction is enough to leave), and nothing from
    `blocked_by_route` (present placements due at an earlier stop) remains.
    """

    index: int
    reachable: bool
    blocked_by_support: frozenset[int]
    blocked_by_neighbors: frozenset[int]
    blocked_by_route: frozenset[int]

    def to_dict(self) -> dict:
        """Canonical cross-language JSON representation."""
        return {
            "index": self.index,
            "reachable": self.reachable,
            "blocked_by_support": sorted(self.blocked_by_support),
            "blocked_by_neighbors": sorted(self.blocked_by_neighbors),
            "blocked_by_route": sorted(self.blocked_by_route),
        }


def placement_reachability(boxes: Sequence[AxisAlignedBox], container: Dimensions,
                           stops: Sequence[int | None] | None = None,
                           directions: Sequence[str] = ALL_DIRECTIONS) -> tuple[Reachability, ...]:
    """Per-placement reachability for the full scene as it stands, one
    entry per box in `boxes` order.

    Reuses `UnloadingDependencyGraph` and the same accessibility sweep
    `safe_removal_order` is built from -- there is no second, possibly-disagreeing
    notion of "blocked" introduced here. Distinct from that generator: this answers
    a snapshot question ("what is reachable with nothing yet moved") rather than
    searching for a complete order, so it never raises `SequenceError` -- an item
    that is not reachable right now is simply reported as such, evidence attached.

    `stops` is optional and behaves exactly like `safe_route_removal_order`'s: `None`
    (the default, and every entry `None`) leaves `blocked_by_route` empty for every
    placement, so a single-stop caller that never populates stops is unaffected.
    When stops are present, a placement due at a later stop than the earliest stop
    still outstanding is blocked by every present placement due at that earlier
    stop, exactly the ordering `safe_route_removal_order` enforces for a *complete*
    unloading order -- reported here per placement instead of only surfacing the
    first stop a full replay gets stuck at.

    O(n^2) for `n` placements: one `_clear_direction`/`_blocking_indices` pass
    (`O(n)` per allowed direction, at most the fixed six-value vocabulary) per
    placement, the same bound `safe_removal_order`'s own search already pays.
    """
    _validated(directions)
    graph = UnloadingDependencyGraph.build(boxes)
    present = frozenset(range(len(boxes)))
    if stops is None:
        stops = tuple(None for _ in boxes)
    elif len(stops) != len(boxes):
        raise ValueError("stops must contain exactly one entry per placement")
    due_stops = {stops[index] for index in present if stops[index] is not None}
    earliest_due_stop = min(due_stops) if due_stops else None

    results = []
    for index in range(len(boxes)):
        box = boxes[index]
        blocked_by_support = graph.depends_on[index] & present
        blocked_by_route: frozenset[int] = frozenset()
        if (stops[index] is not None and earliest_due_stop is not None
                and stops[index] != earliest_due_stop):
            blocked_by_route = frozenset(
                other for other in present
                if other != index and stops[other] is not None and stops[other] < stops[index]
            )
        clear = _clear_direction(index, box, boxes, present, container, directions)
        if clear is None and directions:
            blocked_by_neighbors = frozenset[int]().union(*(
                _blocking_indices(index, box, boxes, present, container, direction)
                for direction in directions
            ))
        else:
            blocked_by_neighbors = frozenset()
        reachable = not blocked_by_support and not blocked_by_route and clear is not None
        results.append(Reachability(index, reachable, blocked_by_support, blocked_by_neighbors, blocked_by_route))
    return tuple(results)
