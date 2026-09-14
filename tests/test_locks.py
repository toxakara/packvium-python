"""Operator locks.

`docs/EXECUTION-PLAN.md` names four properties, and each one is a thing the layer must
*not* do. So this suite is written against the failure modes rather than the happy path:

  * a lock must not mutate the approved plan;
  * a lock must not place anything the engine would refuse, in any language of the word --
    it can forbid alternatives to itself and nothing else;
  * an unsatisfiable lock must come back as a result with a status, not an exception;
  * preservation must be *measured*, because forbidding alternatives does not make the
    solver try the locked point.

A fifth property is not in the design document because it was found here: **a lock must not
cost the load anything it did not ask for.** Two implementations failed it, each by emptying
a container that had been full -- one refused by `item_type`, so eight cubes became one; the
other demanded the locked slot be filled first, so a lock on the far corner produced nothing
at all, because an empty container offers only the origin as a candidate point. Both are the
accommodation this layer exists to prevent, wearing the opposite costume, and both are
regression tests here.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from packvium.constraints import ConstraintContext
from packvium.extensions import ExtensionRegistry
from packvium.geometry import Dimensions, Point, Rotation
from packvium.locks import (
    LOCK_VIOLATED,
    LockSetError,
    LockedResolve,
    PlacementLock,
    _LockConstraint,
    lock_registry,
    locks_from_plan,
    resolve_with_locks,
)
from packvium.models import Container, Item, ItemInstance
from packvium.serialization import pack_from_dict

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "conformance" / "fixtures"

#: Millimetres in the fixed-point ticks the schema's `exactScalar` carries.
MM = 16_000
MM100 = 100 * MM


def eight_cubes() -> dict:
    """Eight 100mm cubes that exactly fill a 200mm box -- the corpus's simplest tiling."""
    # A cross-language fixture kept one level above this package; a published copy does not
    # carry it, and the lock tests that build their own scenes still run.
    fixture = FIXTURES / "exact-fit.json"
    if not fixture.is_file():
        pytest.skip("the shared cross-language fixture corpus is not part of this package")
    return json.loads(fixture.read_text())


def placed(result: dict) -> set[tuple[int, str, str, tuple[int, int, int]]]:
    return {
        (index, placement["item_type"], placement["orientation"],
         tuple(int(placement["position"][axis]["ticks"]) for axis in ("x", "y", "z")))
        for index, container in enumerate(result["containers"])
        for placement in container["placements"]
    }


def count(result: dict) -> int:
    return sum(len(container["placements"]) for container in result["containers"])


# ----------------------------------------------------- a lock binds one instance


def test_locking_one_cube_leaves_the_other_seven_free():
    """The regression this suite exists for.

    A lock addresses a placement -- `(container_index, item_type, position, orientation)`.
    Reading it as a rule about the *type* turns "keep this cube here" into "no cube may be
    anywhere else", which silently drops seven items from a feasible load.
    """
    request = eight_cubes()
    baseline = pack_from_dict(request)
    assert count(baseline) == 8, "the fixture no longer tiles; the rest of this test is void"

    lock = PlacementLock(0, "cube", "LWH", (0, 0, 0))
    resolve = resolve_with_locks(request, [lock])

    assert resolve.preserved
    assert count(resolve.result) == 8


@pytest.mark.parametrize("position", [
    (0, 0, 0),
    (MM100, 0, 0),
    (MM100, MM100, MM100),
])
def test_the_locked_slot_is_the_one_the_operator_named(position):
    """The far corner is the second regression.

    Candidate points are extreme points derived from what is already placed, so an empty
    container offers only the origin. An implementation that required the locked slot to be
    filled before any other instance could not reach `(100, 100, 100)` at all and returned an
    empty container -- a lock the search has not reached yet is not an infeasible lock.
    """
    request = eight_cubes()
    resolve = resolve_with_locks(request, [PlacementLock(0, "cube", "LWH", position)])

    assert resolve.preserved
    assert (0, "cube", "LWH", position) in placed(resolve.result)
    assert count(resolve.result) == 8, "the lock cost the load items it did not ask for"


def test_two_locks_on_one_type_are_both_honoured():
    request = eight_cubes()
    locks = [PlacementLock(0, "cube", "LWH", (0, 0, 0)),
             PlacementLock(0, "cube", "LWH", (MM100, MM100, 0))]
    resolve = resolve_with_locks(request, locks)

    assert resolve.preserved, resolve.missing
    assert count(resolve.result) == 8


# ---------------------------------------------------- the approved plan survives


def test_the_request_is_not_mutated():
    request = eight_cubes()
    before = copy.deepcopy(request)
    resolve_with_locks(request, [PlacementLock(0, "cube", "LWH", (0, 0, 0))])
    assert request == before


def test_the_original_result_is_a_separate_artifact():
    """Property 1. The lock produces a candidate beside the approved plan, never over it."""
    request = eight_cubes()
    approved = pack_from_dict(request)
    snapshot = copy.deepcopy(approved)

    resolve = resolve_with_locks(request, [PlacementLock(0, "cube", "LWH", (0, 0, 0))])

    assert approved == snapshot
    assert resolve.result is not approved


# ------------------------------- a self-contradictory set is refused before any solve


def test_two_locks_claiming_the_same_space_are_refused_without_solving():
    """The line between the two diagnostics this layer produces.

    Whether a lock *fits* is a question about a request, and its answer is `preserved=False`
    with the lock named. Whether a lock set contradicts *itself* is not a question about a
    request at all: two overlapping boxes cannot both hold under any request, container or
    solver. Solving it anyway returns an emptied container -- measured, before this guard
    existed: the two reservations blocked every cell of a lattice that had packed eight
    cubes -- and calling that a result would be the accommodation in reverse.
    """
    with pytest.raises(LockSetError, match="overlapping space"):
        resolve_with_locks(eight_cubes(), [
            PlacementLock(0, "cube", "LWH", (0, 0, 0)),
            PlacementLock(0, "cube", "LWH", (50 * MM, 0, 0)),
        ])


def test_the_same_placement_locked_twice_is_refused():
    with pytest.raises(LockSetError, match="locked twice"):
        resolve_with_locks(eight_cubes(), [PlacementLock(0, "cube", "LWH", (0, 0, 0))] * 2)


def test_a_lock_on_an_item_the_request_does_not_contain_is_refused():
    """An operator naming an item that is not in the shipment is a typo, not an
    infeasibility. Reported as one, it would arrive as a quietly unpreserved lock."""
    with pytest.raises(LockSetError, match="no item type 'wedge'"):
        resolve_with_locks(eight_cubes(), [PlacementLock(0, "wedge", "LWH", (0, 0, 0))])


def test_locks_in_different_containers_never_contradict_each_other():
    """They describe different boxes, so identical coordinates in each are ordinary."""
    request = eight_cubes()
    request["containers"][0]["quantity"] = 2
    resolve_with_locks(request, [PlacementLock(0, "cube", "LWH", (0, 0, 0)),
                                 PlacementLock(1, "cube", "LWH", (0, 0, 0))])


# ------------------------------------------- an unsatisfiable lock is an answer


def test_a_lock_the_solver_cannot_satisfy_returns_a_result_and_not_an_exception():
    """Property 3, and the reason it matters: an operator gets diagnostics, not a stack trace.

    The lock reserves a box straddling the centre of a container that tiles exactly, so the
    reserved volume meets every cell of the lattice and no cube has anywhere legal to go.
    Nothing in the lock layer decides that: the reservation refuses the overlapping
    candidates and the ordinary search reports what it could not place, in its own
    vocabulary.
    """
    request = eight_cubes()
    off_grid = (50 * MM, 50 * MM, 50 * MM)

    resolve = resolve_with_locks(request, [PlacementLock(0, "cube", "LWH", off_grid)])

    assert not resolve.preserved
    assert resolve.missing[0].position_ticks == off_grid
    assert resolve.result["status"] in {"best_found", "infeasible"}
    assert resolve.result["unpacked_items"], "an unhonoured lock must leave a trail"
    assert resolve.result["unpacked_items"][0]["proof"]["level"] in {
        "proven", "observed", "inferred", "unknown_due_to_limit"
    }


def test_an_unhonoured_lock_is_never_softened_into_a_success():
    request = eight_cubes()
    resolve = resolve_with_locks(
        request, [PlacementLock(0, "cube", "LWH", (50 * MM, 50 * MM, 50 * MM))])

    with pytest.raises(ValueError, match="cannot be missing a lock"):
        LockedResolve(locks=resolve.locks, result=resolve.result, preserved=True,
                      missing=resolve.missing)


def test_an_unpreserved_resolve_must_say_which_lock_failed():
    with pytest.raises(ValueError, match="must name the locks"):
        LockedResolve(locks=(), result={}, preserved=False, missing=())


# ------------------------------------- a lock outside the container is inert, not fatal


def test_a_lock_that_cannot_apply_to_this_container_does_not_empty_it():
    """The narrowing in `_applicable`, from the operator's side.

    A constraint cannot know which container index it is packing, so a lock read from one
    container is offered to every container in the solve. Without the reach test, a lock
    whose box does not fit here would refuse every candidate and the container would come
    back empty. It is reported as missing instead -- the load is untouched and the operator
    is told their lock did not hold.
    """
    request = eight_cubes()
    resolve = resolve_with_locks(request, [PlacementLock(0, "cube", "LWH", (2 * MM100, 0, 0))])

    assert not resolve.preserved
    assert count(resolve.result) == 8
    assert resolve.result["status"] == "feasible"


# -------------------------------------------------------- the constraint itself


CUBE = Item.create("cube", Dimensions.mm(100, 100, 100), 0)
BOX = Container.create("box", Dimensions.mm(200, 200, 200))


def context(x: int, y: int, z: int, rotation: Rotation = Rotation.LWH,
            placements=()) -> ConstraintContext:
    dimensions = CUBE.dimensions.rotated(rotation)
    return ConstraintContext(BOX, tuple(placements), ItemInstance(CUBE, 1),
                             Point(x, y, z), rotation, dimensions, dimensions)


def test_the_constraint_can_only_ever_refuse():
    """Property 4, as the cheapest possible test of it.

    `ConstraintResult` has two shapes and a chain is an AND, so a constraint that returns
    `allow()` has said nothing -- it cannot overrule the rules that refuse. Whatever this
    constraint decides, it cannot be the reason a placement exists.
    """
    constraint = _LockConstraint((PlacementLock(0, "cube", "LWH", (0, 0, 0)),))
    verdicts = {constraint.evaluate(context(x, 0, 0)).allowed for x in (50 * MM, MM100)}
    assert verdicts == {True, False}


def test_a_candidate_that_misses_the_reserved_volume_is_untouched():
    """What leaves the seven unlocked cubes free: a reservation is about one box, not about
    an item type and not about the order the search fills things in."""
    constraint = _LockConstraint((PlacementLock(0, "cube", "LWH", (0, 0, 0)),))
    for position in ((MM100, 0, 0), (0, MM100, 0), (MM100, MM100, MM100)):
        assert constraint.evaluate(context(*position)).allowed, position


def test_a_candidate_overlapping_the_reserved_volume_is_refused():
    constraint = _LockConstraint((PlacementLock(0, "cube", "LWH", (0, 0, 0)),))
    assert not constraint.evaluate(context(50 * MM, 0, 0)).allowed


def test_the_candidate_that_is_the_lock_is_admitted():
    constraint = _LockConstraint((PlacementLock(0, "cube", "LWH", (0, 0, 0)),))
    assert constraint.evaluate(context(0, 0, 0)).allowed


def test_the_refusal_names_the_lock_and_carries_the_operator_code():
    constraint = _LockConstraint((PlacementLock(0, "cube", "LWH", (0, 0, 0)),))
    verdict = constraint.evaluate(context(50 * MM, 0, 0))

    assert not verdict.allowed
    assert verdict.code == LOCK_VIOLATED
    assert "LWH@(0, 0, 0)" in verdict.detail


def test_an_unlocked_item_type_is_untouched():
    constraint = _LockConstraint((PlacementLock(0, "wedge", "LWH", (0, 0, 0)),))
    assert constraint.evaluate(context(0, 0, 0)).allowed


def test_a_satisfied_lock_stops_reserving():
    """The reservation lasts exactly as long as the lock is outstanding.

    Once the container holds the locked slot, the volume is defended by the ordinary overlap
    rule and this constraint has nothing left to say -- which is what stops a lock set from
    accumulating cost as the container fills.
    """
    from packvium.models import Placement

    constraint = _LockConstraint((PlacementLock(0, "cube", "LWH", (0, 0, 0)),))
    assert not constraint.evaluate(context(50 * MM, 0, 0)).allowed

    occupied = Placement(ItemInstance(CUBE, 1), Point(0, 0, 0), Rotation.LWH,
                         CUBE.dimensions, Point(0, 0, 0), CUBE.dimensions)
    assert constraint.evaluate(context(50 * MM, 0, 0, placements=(occupied,))).allowed


def test_the_orientation_is_part_of_the_lock_and_not_decoration():
    """Same origin, different rotation: not the locked placement, and it takes the reserved
    volume -- so it is refused rather than accepted as close enough."""
    constraint = _LockConstraint((PlacementLock(0, "cube", "LWH", (0, 0, 0)),))
    assert not constraint.evaluate(context(0, 0, 0, Rotation.WLH)).allowed


# ------------------------------------------------------------ addressing a lock


@pytest.mark.parametrize("kwargs, message", [
    (dict(container_index=-1), "container index"),
    (dict(item_type=""), "item type"),
    (dict(orientation="sideways"), "orientation this engine emits"),
    (dict(position_ticks=(0, 0)), "three axes"),
    (dict(position_ticks=(0, 0, 1.5)), "exact integer ticks"),
    (dict(position_ticks=(0, 0, True)), "exact integer ticks"),
    (dict(position_ticks=(0, 0, -1)), "cannot be negative"),
])
def test_an_address_the_engine_could_not_have_emitted_is_refused(kwargs, message):
    address = dict(container_index=0, item_type="cube", orientation="LWH",
                   position_ticks=(0, 0, 0))
    with pytest.raises(ValueError, match=message):
        PlacementLock(**{**address, **kwargs})


def test_a_lock_is_read_from_the_plans_own_reference():
    """The plan is where an operator sees a placement, so it is where they point at one.

    Reading the plan's emitted reference rather than a second address format is the only
    reason the two can be trusted to name the same box.
    """
    from packvium.execution import build_execution_plan

    request = eight_cubes()
    result = pack_from_dict(request)
    plan = build_execution_plan(request, result)

    locks = locks_from_plan(plan, container_index=0, item_types=["cube"])

    assert len(locks) == 8
    assert {lock.slot for lock in locks} == {
        (placement["orientation"],
         tuple(int(placement["position"][axis]["ticks"]) for axis in ("x", "y", "z")))
        for placement in result["containers"][0]["placements"]
    }


def test_locking_a_whole_container_reproduces_it():
    """Every placement locked is the strongest form of the first property: the load the
    operator approved is the load that comes back."""
    from packvium.execution import build_execution_plan

    request = eight_cubes()
    approved = pack_from_dict(request)
    locks = locks_from_plan(build_execution_plan(request, approved),
                            container_index=0, item_types=["cube"])

    resolve = resolve_with_locks(request, locks)

    assert resolve.preserved, resolve.missing
    assert placed(resolve.result) == placed(approved)


def test_a_lock_read_from_another_container_index_is_not_read():
    from packvium.execution import build_execution_plan

    request = eight_cubes()
    plan = build_execution_plan(request, pack_from_dict(request))
    assert locks_from_plan(plan, container_index=1, item_types=["cube"]) == ()


def test_only_the_named_item_types_become_locks():
    from packvium.execution import build_execution_plan

    request = eight_cubes()
    plan = build_execution_plan(request, pack_from_dict(request))
    assert locks_from_plan(plan, container_index=0, item_types=["wedge"]) == ()


# ---------------------------------------------------------------- the registry


def test_an_empty_lock_set_returns_the_callers_registry_unchanged():
    base = ExtensionRegistry()
    assert lock_registry([], base) is base


def test_locking_does_not_drop_the_callers_own_constraints():
    """An application that locks a placement must not lose its own rules by doing so."""
    class Refuses:
        def evaluate(self, context):
            raise AssertionError("not evaluated in this test")

    mine = Refuses()
    registry = lock_registry([PlacementLock(0, "cube", "LWH", (0, 0, 0))],
                             ExtensionRegistry(placement_constraints=(mine,)))

    assert registry.placement_constraints[0] is mine
    assert isinstance(registry.placement_constraints[1], _LockConstraint)


#: The instant the policy is evaluated at. A policy carries one because reading a clock
#: here would make the same request pack differently on different days.
AS_OF = 1_704_067_200_000

SEGREGATED = {
    "configuration": {"solver_profile": "fast", "time_limit_ms": 300000},
    "policy": {
        "as_of": AS_OF,
        "rules": [{
            "id": "hazmat-food-segregation", "version": 1, "effective_at": AS_OF,
            "priority": 100,
            "separate_tags": {"tag": "hazmat", "from_tag": "food"},
        }],
    },
    "items": [
        {"id": "drum", "quantity": 1, "tags": ["hazmat"],
         "dimensions": {"length": "100", "width": "100", "height": "100"}},
        {"id": "crate", "quantity": 1, "tags": ["food"],
         "dimensions": {"length": "100", "width": "100", "height": "100"}},
    ],
    "containers": [{"id": "box", "quantity": 2,
                    "inner_dimensions": {"length": "200", "width": "200", "height": "200"}}],
}


def test_locking_does_not_switch_off_the_policy_the_request_carries():
    """`pack_from_dict` compiles `policy` into placement constraints, and a lock arrives
    through the same door. Replacing that set instead of adding to it would make locking
    anything a way to turn a segregation rule off -- a lock silently loosening a rule is the
    exact failure property 4 forbids, reached through the serialization layer instead of the
    solver."""
    request = copy.deepcopy(SEGREGATED)
    unlocked = pack_from_dict(request)
    assert len(unlocked["containers"]) == 2, "the rule no longer separates; the test is void"

    resolve = resolve_with_locks(request, [PlacementLock(0, "drum", "LWH", (0, 0, 0))])

    assert len(resolve.result["containers"]) == 2
    for container in resolve.result["containers"]:
        tags = {placement["item_type"] for placement in container["placements"]}
        assert tags != {"drum", "crate"}, "the lock let the segregation rule lapse"


# ------------------------------------------------------- the dependency direction


def test_the_lock_layer_is_a_caller_of_the_engine_and_not_a_part_of_it():
    """`docs/EXECUTION-PLAN.md` makes this a rule, and a grep is the cheapest test of it.

    A solver that imported the lock layer could grow a lock-aware path, which is exactly the
    special case property 4 forbids.
    """
    package = pathlib.Path(__file__).resolve().parents[1] / "src" / "packvium"
    for module in ("packer.py", "solvers.py", "constraints.py", "validation.py"):
        source = (package / module).read_text()
        assert "locks" not in source.replace("blocks", "").replace("interlock", ""), module


def test_a_locked_result_is_still_a_validated_result():
    """The acceptance asks that independent validation cover lock preservation *and* every
    existing physical constraint. It does so by construction -- `Packer` runs
    `IndependentSolutionValidator` over every solve and the locked solve is an ordinary one
    -- and this re-derives the cheapest of those guarantees from the placements alone, so
    the claim is not resting entirely on the call graph.
    """
    request = eight_cubes()
    resolve = resolve_with_locks(request, [PlacementLock(0, "cube", "LWH", (MM100, 0, 0))])

    assert resolve.result["feasibility"]["code"] == "feasible"
    boxes = [
        (tuple(int(p["position"][a]["ticks"]) for a in ("x", "y", "z")),
         tuple(int(p["dimensions"][d]["ticks"]) for d in ("length", "width", "height")))
        for c in resolve.result["containers"] for p in c["placements"]
    ]
    for index, (origin, size) in enumerate(boxes):
        for other_origin, other_size in boxes[index + 1:]:
            overlap = all(
                origin[axis] < other_origin[axis] + other_size[axis]
                and other_origin[axis] < origin[axis] + size[axis]
                for axis in range(3)
            )
            assert not overlap, "a locked solve returned overlapping placements"
