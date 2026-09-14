"""Operator locks, expressed as request-derived constraints.

`docs/EXECUTION-PLAN.md` is the contract. A lock is an operator saying *this item stays in
this box, in this place*, and the whole design problem is that honouring one must not become
a way around the solver or the validator.

It is not. A lock is a `PlacementConstraint` supplied through the existing
`ExtensionRegistry`, and everything after that is the ordinary path: the same portfolio, the
same constraints, the same independent validation. There is no lock-aware solver, no relaxed
validator and no special case anywhere in the engine -- which is why a lock cannot produce a
placement the engine would otherwise refuse. The strongest thing a lock can do is *forbid*
alternatives to itself.

Four properties, and the first is what makes the rest safe.

**The approved plan is never mutated.** `resolve_with_locks` returns a `LockedResolve`
carrying a *separate* result. The original is untouched and citable, so "what was approved"
and "what was proposed after the lock" are two artifacts rather than two states of one.

**A physically invalid lock is rejected, not accommodated.** The lock constraint refuses
every candidate that is not the locked one; it never asserts that the locked one is legal.
Support, top-load, overlap and every other rule still run, so a lock that would float an item
simply leaves it unplaced -- and the result says so with the engine's own vocabulary.

**Infeasibility is an answer, not an exception.** A lock set the solver cannot satisfy comes
back as `preserved=False` with the ordinary `unpacked_items[].proof`, not as a raised error.
A well-formed request the model cannot answer has always been a result with a status here.

**Preservation is verified, not assumed.** Forbidding alternatives does not make the solver
try the locked point; it only stops it succeeding anywhere else. Whether the lock actually
held is therefore checked against the returned result, and reported.

## A lock reserves a slot; it does not demand priority

The reference `docs/EXECUTION-PLAN.md` derives is `(container_index, item_type,
position_ticks, orientation)` -- deliberately not `item_id`, because an id is an instance
counter four engines need not agree on. That address names a *placement*, and the constraint
has to mean the same thing: locking one of eight identical cubes must leave the other seven
free to go wherever they fit.

Two earlier readings of that failed, and both failed by emptying a container that had been
full. Read as a rule about the *item type* -- no cube may be anywhere but the locked point --
a container that held eight came back holding one. Read as a rule about *order* -- the locked
slot must be filled before any other instance -- a lock on the far corner came back holding
nothing, because candidate points are extreme points derived from what is already placed and
an empty container offers only the origin. A lock the search cannot reach yet is not an
infeasible lock.

So the rule reserves rather than forces. While a lock is outstanding, its box is off limits
to every other candidate of that type; the candidate that *is* the lock is admitted, and
every candidate that does not touch the reserved volume is untouched. The locked slot is
therefore still empty when the search finally reaches it, which is what makes the lock hold
without any part of the engine knowing a lock exists.

Nothing here forces the slot to be filled, and that is the honest shape: a reservation the
solve never used comes back as a `missing` lock rather than as a load with seven items
deleted from it.

## What a lock costs, and what it does not

Locking is not free, and the cost is not in this module. Registering *any* caller constraint
takes a request off `GridSolver` (`solvers.py`), because the grid places by formula and never
evaluates the constraint chain -- a grid solver that ignored a lock would be the bypass
property 4 forbids. On an exactly tiling request that is a real jump in candidates evaluated,
3.5x at 8 items and 26x at 125, and a do-nothing constraint pays every bit of it.

The lock's own cost, measured against that baseline in `benchmarks/lock_scaling.py`, is a
ratio of 1.00 at every size: the same candidates are evaluated, some with a different verdict.
Locking changes which solver answers, not how the answer scales.

`context.placements` is per-container search state, and the search is deterministic, so the
outstanding set is a deterministic function of the request and the lock set.

The reservation covers other instances of the locked type, which is what the address makes
computable: a lock names an `item_type`, and the box it occupies follows from that type's
dimensions under the locked orientation. A *different* type taking the reserved volume is
not refused, because the constraint has no way to size a box it is not currently evaluating
-- it is caught after the solve, as a lock reported `missing`.

## The one thing a constraint cannot see

A lock names a container index; a constraint runs inside a search that has no index yet --
`container_index` is a position in the *result's* sorted container order, which does not
exist until the solve finishes. The constraint therefore applies the outstanding rule in
every container the locked box could physically occupy, and `resolve_with_locks` verifies
preservation at the exact index afterwards. Over-applying can only *forbid*, never place,
so the failure mode is a reported `missing` lock and never a placement the engine would have
refused.
"""

from __future__ import annotations

from ._compat import dataclass
from typing import Any, Mapping, Optional, Sequence

from .constraints import ConstraintContext, ConstraintResult
from .extensions import ExtensionRegistry
from .geometry import AxisAlignedBox, Point, Rotation

__all__ = [
    "LOCK_VIOLATED",
    "LockSetError",
    "LockedResolve",
    "PlacementLock",
    "lock_registry",
    "locks_from_plan",
    "resolve_with_locks",
]

#: The rejection code a locked item's non-locked candidate carries. Named like every other
#: constraint code so that a trace, a diagnostic and an `unpacked_items[].details` entry
#: read the same way whether the refusal came from physics or from an operator.
LOCK_VIOLATED = "operator_lock"

_Slot = tuple[str, tuple[int, int, int]]


class LockSetError(ValueError):
    """A lock set that no request could satisfy, raised before any solve.

    The distinction this draws is the one that decides what an operator is told. A lock the
    *solve* could not honour is an answer -- `preserved=False`, with the lock named in
    `missing` -- because whether it fits is a question about a request. A lock set that
    contradicts *itself* is not a question at all: two locks whose boxes overlap cannot both
    hold against any request, in any container, under any solver. Answering it with a solve
    would return an emptied container and call it a result.

    This is the same line `packvium.execution` draws when it refuses a loading order that is
    not a permutation.
    """


@dataclass(frozen=True, slots=True)
class PlacementLock:
    """One operator lock: an item type pinned to an exact position and orientation.

    Addressed the way the execution plan addresses a placement -- by `container_index`,
    `item_type`, exact integer `position_ticks` and `orientation`, never by `item_id`. The
    reason is the one `docs/EXECUTION-PLAN.md` gives: an id is an instance counter that four
    engines need not agree on, and a lock that meant different boxes in different runtimes
    would be worse than no lock.
    """

    container_index: int
    item_type: str
    orientation: str
    position_ticks: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.container_index < 0:
            raise ValueError("a lock must name a container index")
        if not self.item_type:
            raise ValueError("a lock must name an item type")
        if self.orientation not in Rotation.__members__:
            raise ValueError(f"{self.orientation!r} is not an orientation this engine emits")
        if len(self.position_ticks) != 3:
            raise ValueError("a lock's position must have three axes")
        if any(not isinstance(t, int) or isinstance(t, bool) for t in self.position_ticks):
            raise ValueError("a lock's position must be exact integer ticks")
        if any(t < 0 for t in self.position_ticks):
            raise ValueError("a lock's position cannot be negative")

    @property
    def slot(self) -> _Slot:
        return (self.orientation, tuple(self.position_ticks))


def locks_from_plan(plan: Mapping[str, Any], *, container_index: int,
                    item_types: Sequence[str]) -> tuple[PlacementLock, ...]:
    """Read locks out of an execution plan's own placement references.

    The plan is where an operator sees the placements, so it is where they point at one.
    This reads the references the plan already emits rather than inventing a second address
    format, which is the only reason the two can be trusted to mean the same box.
    """
    wanted = set(item_types)
    locks = []
    for container in plan.get("containers") or ():
        if container.get("container_index") != container_index:
            continue
        for step in container.get("steps") or ():
            reference = step.get("placement") or {}
            if reference.get("item_type") not in wanted:
                continue
            ticks = reference.get("position_ticks") or {}
            locks.append(PlacementLock(
                container_index=container_index,
                item_type=str(reference["item_type"]),
                orientation=str(reference["orientation"]),
                position_ticks=(int(ticks["x"]), int(ticks["y"]), int(ticks["z"])),
            ))
    return tuple(locks)


@dataclass(frozen=True, slots=True)
class _LockConstraint:
    """Refuses a locked type's candidates while that type still owes a locked slot.

    Deliberately one-directional. It can say *no*, and it can say nothing else: there is no
    branch in which it returns `allow()` for a placement the other constraints would have
    refused, because it never runs instead of them -- it runs alongside them, and every
    constraint must allow a candidate for it to be placed.

    Three narrowings keep a lock from costing more than it asks for. An item type with no
    lock is untouched. A lock whose box cannot fit this container is not applied here at all.
    And a candidate that does not touch a reserved volume is allowed wherever it lands, which
    is what leaves the seven unlocked cubes free.

    Complexity: O(L) per candidate for L locks naming the item's type, plus one walk of the
    container's placements when such a lock exists -- the same O(P) walk the support and
    overlap rules already make, so `docs/ALGORITHMS-AND-COMPLEXITY.md`'s per-candidate bound
    is unchanged. A request with no locks never constructs this constraint at all.

    Measured, in `benchmarks/lock_scaling.py`: against a caller-constraint baseline, a lock
    evaluates exactly the same number of candidates at every size -- ratio 1.00 from 8 items
    to 125 -- which is the constant factor the bound requires.
    """

    locks: tuple[PlacementLock, ...]

    def _applicable(self, lock: PlacementLock, context: ConstraintContext) -> bool:
        """Whether this container could hold the locked box at all.

        A constraint cannot know which container index it is packing, so without this a lock
        read from one container would steer the first instance in *every* container -- and in
        a container too small for the locked origin, would refuse every candidate and leave
        the type unpacked. The test is the physical box rather than the clearance envelope:
        it exists to drop locks that plainly cannot apply here, and a lock this admits is
        still judged by the ordinary geometry rules.
        """
        box = context.item.dimensions.rotated(Rotation[lock.orientation])
        inner = context.container.inner_dimensions
        x, y, z = lock.position_ticks
        return (x + box.length.ticks <= inner.length.ticks
                and y + box.width.ticks <= inner.width.ticks
                and z + box.height.ticks <= inner.height.ticks)

    def _reserved_box(self, lock: PlacementLock,
                      context: ConstraintContext) -> AxisAlignedBox:
        """The volume the locked item will occupy, sized from its own type.

        The physical box rather than the clearance envelope: a clearance is a margin the
        ordinary clearance rule already enforces around whatever ends up here, and reserving
        it twice would refuse neighbours the engine is willing to place.
        """
        dimensions = context.item.dimensions.rotated(Rotation[lock.orientation])
        return AxisAlignedBox(Point(*lock.position_ticks), dimensions)

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        item_type = context.item.item.id
        relevant = [lock for lock in self.locks
                    if lock.item_type == item_type and self._applicable(lock, context)]
        if not relevant:
            return ConstraintResult.allow()

        filled = {
            (placement.rotation.name,
             (placement.position.x, placement.position.y, placement.position.z))
            for placement in context.placements
            if placement.instance.item.id == item_type
        }
        outstanding = [lock for lock in relevant if lock.slot not in filled]
        if not outstanding:
            return ConstraintResult.allow()

        candidate_slot = (context.rotation.name,
                          (context.point.x, context.point.y, context.point.z))
        candidate_box = AxisAlignedBox(context.point, context.dimensions)
        for lock in outstanding:
            if lock.slot == candidate_slot:
                continue
            if candidate_box.intersects(self._reserved_box(lock, context)):
                return ConstraintResult.reject(
                    LOCK_VIOLATED,
                    f"{item_type} is locked to {lock.orientation}@{lock.position_ticks}",
                )
        return ConstraintResult.allow()


def lock_registry(locks: Sequence[PlacementLock],
                  base: Optional[ExtensionRegistry] = None) -> ExtensionRegistry:
    """The caller's extensions plus the lock constraint.

    A registry rather than a new solver argument, because that is the extension point this
    engine already has and using it means the locked run is the ordinary run. `base` is
    preserved so an application's own constraints are not silently dropped by locking.
    """
    registry = base or ExtensionRegistry()
    if not locks:
        return registry
    constraint = _LockConstraint(tuple(locks))
    return ExtensionRegistry(
        placement_constraints=(*registry.placement_constraints, constraint),
        item_order_strategies=registry.item_order_strategies,
        solvers=registry.solvers,
        container_selector=registry.container_selector,
    )


@dataclass(frozen=True, slots=True)
class LockedResolve:
    """A re-solve under a lock set, beside the plan it came from -- never replacing it.

    `preserved` is measured against `result`, not promised by the constraint: forbidding
    alternatives stops the solver succeeding elsewhere, it does not make it try the locked
    point. `missing` names the locks the result does not contain, which is what a caller
    shows an operator when their lock could not be honoured.
    """

    locks: tuple[PlacementLock, ...]
    result: Mapping[str, Any]
    preserved: bool
    missing: tuple[PlacementLock, ...] = ()

    def __post_init__(self) -> None:
        if self.preserved and self.missing:
            raise ValueError("a preserved resolve cannot be missing a lock")
        if not self.preserved and not self.missing:
            raise ValueError("an unpreserved resolve must name the locks it could not honour")


def _placed(result: Mapping[str, Any]) -> set[tuple[int, str, str, tuple[int, int, int]]]:
    """Every placement in the result, keyed the way a lock addresses one.

    `container_index` is the container's position in the emitted list, which is the same
    order `packvium.execution` indexes and the order the plan's references carry.
    """
    placed = set()
    for index, container in enumerate(result.get("containers") or ()):
        for placement in container.get("placements") or ():
            position = placement.get("position") or {}
            try:
                ticks = tuple(int(position[axis]["ticks"]) for axis in ("x", "y", "z"))
            except (KeyError, TypeError):
                continue
            placed.add((index, str(placement.get("item_type")),
                        str(placement.get("orientation")), ticks))
    return placed


def _refuse_a_contradictory_set(locks: Sequence[PlacementLock],
                               items: Mapping[str, Any]) -> None:
    """Every way a lock set can be wrong without a request being consulted.

    Cheap and exhaustive: O(L^2) over a set an operator typed, against a solve that is the
    expensive thing here. Locks in different containers are never compared -- they describe
    different boxes and cannot contradict one another.
    """
    boxes = []
    for lock in locks:
        item = items.get(lock.item_type)
        if item is None:
            raise LockSetError(f"the request has no item type {lock.item_type!r} to lock")
        dimensions = item.dimensions.rotated(Rotation[lock.orientation])
        boxes.append((lock, AxisAlignedBox(Point(*lock.position_ticks), dimensions)))

    for index, (lock, box) in enumerate(boxes):
        for other, other_box in boxes[index + 1:]:
            if lock.container_index != other.container_index:
                continue
            if lock.slot == other.slot and lock.item_type == other.item_type:
                raise LockSetError(f"the same placement is locked twice: {lock.slot}")
            if box.intersects(other_box):
                raise LockSetError(
                    f"two locks claim overlapping space in container {lock.container_index}: "
                    f"{lock.item_type} at {lock.orientation}@{lock.position_ticks} and "
                    f"{other.item_type} at {other.orientation}@{other.position_ticks}")


def resolve_with_locks(request: Mapping[str, Any], locks: Sequence[PlacementLock],
                       *, base: Optional[ExtensionRegistry] = None) -> LockedResolve:
    """Re-solve `request` with `locks` applied, through the ordinary packing path.

    Imported here rather than at module scope so that this module holds no import-time
    dependency on the solver: the lock layer is a caller of the engine, not a part of it,
    and `docs/EXECUTION-PLAN.md` makes the direction of that dependency a rule.
    """
    from .serialization import _item, pack_from_dict

    locks = tuple(locks)
    unit = (request.get("units") or {}).get("length", "mm")
    _refuse_a_contradictory_set(locks, {
        raw["id"]: _item(raw, unit) for raw in request.get("items") or ()
    })
    result = pack_from_dict(dict(request), extensions=lock_registry(locks, base))
    placed = _placed(result)
    missing = tuple(
        lock for lock in locks
        if (lock.container_index, lock.item_type, lock.orientation,
            tuple(lock.position_ticks)) not in placed
    )
    return LockedResolve(locks=locks, result=result, preserved=not missing, missing=missing)
