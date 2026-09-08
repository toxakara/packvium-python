"""The root lower bound is actually computed where the task says it is.

The cross-implementation agreement and corpus soundness live in
`conformance/tests/test_optimality_bounds_engine.py`. What is asserted here is the wiring:
that an `exact_small` or global-beam solve really does record a bound, that it stays out of
the serialised result, and that the number it records is sound against the packing that
solve produced.

That last part is why this is not a mock test. `hull_refinements` set the precedent for an
internal counter and nothing ever asserted it fired, so "internal" quietly became
"unobserved". A bound nobody checks is worse than none: it will be believed later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packvium-python" / "src"))

import packvium  # noqa: E402
from packvium import bounds, solvers  # noqa: E402

#: Four differently-shaped items. Identical items reach `GridSolver`'s quantity-compression
#: fast path, which answers before either target solver is consulted -- a request that never
#: exercises the code under test while looking like it does.
ITEMS = [
    {
        "id": f"i{index}",
        "quantity": 1,
        "dimensions": {
            "length": str(30 + 7 * index),
            "width": str(20 + 3 * index),
            "height": str(15 + 5 * index),
        },
    }
    for index in range(4)
]
CONTAINERS = [{"id": "c", "inner_dimensions": {"length": "100", "width": "100", "height": "100"}}]


def _pack(monkeypatch, configuration):
    """Pack, and return the result together with every bound the solve computed."""
    recorded: list[tuple] = []
    original = bounds.compute

    def spy(instances, containers):
        computed = original(instances, containers)
        recorded.append(computed.as_tuple())
        return computed

    monkeypatch.setattr(solvers.bounds, "compute", spy)
    result = packvium.pack_from_dict({
        "containers": CONTAINERS,
        "items": ITEMS,
        "configuration": dict(configuration, time_limit_ms=20000),
    })
    return result, recorded


@pytest.mark.parametrize("configuration,expected_solver", [
    ({"solver_profile": "exact_small"}, "exact_small"),
    ({"container_plan_beam_width": 4}, None),
])
def test_a_bound_is_recorded_for_the_solvers_the_task_names(monkeypatch, configuration,
                                                            expected_solver):
    """`exact_small` and the global container-set beam each reach the root bound."""
    result, recorded = _pack(monkeypatch, configuration)
    assert recorded, f"no lower bound was computed for {configuration}"
    if expected_solver is not None:
        assert result["algorithm"]["solver"].startswith(expected_solver)


@pytest.mark.parametrize("configuration", [
    {"solver_profile": "exact_small"},
    {"container_plan_beam_width": 4},
    {"solver_profile": "exact_small", "container_plan_beam_width": 4},
])
def test_the_recorded_bound_is_sound_against_the_packing_it_bounded(monkeypatch,
                                                                    configuration):
    """The bound and the arrangement come from the same solve, so they must agree.

    `gap` raises on a score below its bound rather than returning a negative number, so
    calling it is the assertion.
    """
    result, recorded = _pack(monkeypatch, configuration)
    score = [int(component) for component in result["score"]]
    for bound in recorded:
        bounds.gap(score, bounds.Bounds(*bound))


def test_the_bound_does_not_reach_the_serialised_result(monkeypatch):
    """Reporting a gap to a caller is a new public result field, and this project reserves
    and rejects such a field before a contract freeze rather than adding it mid-line.

    Also the practical half: `algorithm.metrics` is serialised into every result, so a new
    key there changes the bytes of every committed golden and would have to land in all four
    engines at once.
    """
    result, recorded = _pack(monkeypatch, {"solver_profile": "exact_small"})
    assert recorded, "the guard below would pass vacuously if nothing was computed"
    metrics = result["algorithm"]["metrics"]
    leaked = [key for key in metrics if "bound" in key or "gap" in key]
    assert not leaked, f"the internal bound leaked into algorithm.metrics as {leaked}"


def test_an_ordinary_solve_computes_no_bound(monkeypatch):
    """The constructive heuristics have nothing to add to it and do not pay for it.

    A capacity relaxation says the same thing whoever asks, so computing it for `grid` or
    `layer` would be arithmetic nobody reads -- and this stays true until a caller-facing
    field exists to read it.
    """
    _result, recorded = _pack(monkeypatch, {"solver_profile": "fast"})
    assert not recorded


def test_a_non_default_objective_records_no_bound(monkeypatch):
    """`lowest_cost` orders its score keys differently, so a bound vector compared against
    it would line up cost against container count.

    Recording nothing is the honest answer; recording a vector that cannot be compared is
    how a wrong gap gets published later.
    """
    _result, recorded = _pack(
        monkeypatch, {"solver_profile": "exact_small", "objective": "lowest_cost"})
    assert not recorded


def test_the_bound_survives_into_the_stats_object():
    """The field exists and carries the vector, which is what "internally available" means.

    Asserted directly rather than through a solve so that a rename of the field fails here
    with a clear message instead of silently making every spy-based test above vacuous.
    """
    stats = solvers.SearchStats()
    assert stats.objective_lower_bound is None
    stats.objective_lower_bound = (0, 1, 2, 3, 4)
    assert stats.objective_lower_bound == (0, 1, 2, 3, 4)
    assert "objective_lower_bound" not in stats.to_metrics().to_dict()


# ----------------------------------- the degenerate inputs the guards exist for
#
# Every branch below returns a *bound*, and a bound that is wrong on a degenerate request is
# wrong in the direction that matters: it claims work is unavoidable when it is not, or --
# worse -- claims a score is impossible that a solver then achieves. The helpers are called
# directly because that is the only honest way to reach a guard whose whole purpose is to
# never be reached through the front door.

def test_a_capacity_nobody_declared_is_infinite_rather_than_zero():
    """`None` means unbounded on the way in and on the way out.

    A payload limit nobody declared cannot be summed into a number; treating it as zero
    would make every item unplaceable and the unpacked bound maximal.
    """
    assert bounds._capacity_total([None, 5], [1, 1], unbounded_when_value_infinite=True) is None
    # Volume is the exception: a container with no usable volume contributes nothing however
    # many of it exist, so an absent value is skipped instead of poisoning the total.
    assert bounds._capacity_total([None, 5], [1, 2], unbounded_when_value_infinite=False) == 10


def test_unlimited_inventory_is_unbounded_only_when_the_type_carries_something():
    """An unlimited supply of zero capacity is still zero capacity."""
    assert bounds._capacity_total([7], [None], unbounded_when_value_infinite=True) is None
    assert bounds._capacity_total([0], [None], unbounded_when_value_infinite=True) == 0


def test_fitting_stops_at_the_item_that_exceeds_the_capacity():
    assert bounds._fit([2, 3, 4], 5) == 2
    assert bounds._fit([2, 3, 4], 100) == 3
    # No declared capacity fits everything rather than nothing.
    assert bounds._fit([2, 3, 4], None) == 3


def test_a_sum_past_the_ceiling_is_refused_by_the_guard_itself():
    assert bounds._guard(bounds.MAX_BOUND_SUM, "x") == bounds.MAX_BOUND_SUM
    with pytest.raises(bounds.BoundOverflowError):
        bounds._guard(bounds.MAX_BOUND_SUM + 1, "x")


def test_a_bound_that_cannot_cross_every_binding_exactly_is_refused():
    """Unlimited inventory must not hide a selected-cost overflow from the precheck."""
    instances = [
        packvium.ItemInstance(
            packvium.Item.create(str(index), packvium.Dimensions.mm(1, 1, 1), 1), index
        )
        for index in range(2)
    ]
    container = packvium.Container.create(
        "c", packvium.Dimensions.mm(1, 1, 1), max_items=1,
        cost_minor=bounds.MAX_BOUND_VALUE,
    )

    with pytest.raises(bounds.BoundOverflowError, match="exact portable result ceiling"):
        bounds.compute(instances, (container,))


def test_no_container_and_nothing_placed_bound_nothing():
    """Each key must degrade to zero rather than divide by a capacity that is not there."""
    assert bounds._container_bound([], [], False, 0, [], [], [], []) == 0
    assert bounds._cost_bound([5], [1], 0) == 0
    assert bounds._unused_volume_bound([], False, 0, [], 0) == 0
    assert bounds._stack_height_bound([], False, 0, [], [], 0) == 0


def test_a_container_with_no_room_bounds_nothing_rather_than_dividing_by_it():
    """Zero inner volume, zero base area and zero height each reach their own guard."""
    assert bounds._unused_volume_bound([1], False, 1, [0], 1) == 0
    assert bounds._stack_height_bound([1], False, 1, [0], [10], 1) == 0
    assert bounds._stack_height_bound([1], False, 1, [10], [0], 1) == 0


def test_an_item_count_limit_binds_the_unpacked_bound():
    """`max_items` is a capacity like volume and weight, and the worst of the three wins.

    The inventory has to be finite for it to bind: one container holding two of five items
    strands three. Left unlimited, an unbounded supply of two-item containers strands
    nothing and the count falls to `container_count` instead -- which is the same arithmetic
    answering a different question.
    """
    container = packvium.Container.create(
        "c", packvium.Dimensions.mm(100, 100, 100), max_items=2, quantity=1)
    item = packvium.Item.create("i", packvium.Dimensions.mm(1, 1, 1), 1, quantity=5)
    instances = packvium.PackingRequest((item,), (container,)).instances
    assert bounds.compute(instances, (container,)).unpacked_count == 3

    unlimited = packvium.Container.create(
        "u", packvium.Dimensions.mm(100, 100, 100), max_items=2)
    strands_nothing = bounds.compute(instances, (unlimited,))
    assert strands_nothing.unpacked_count == 0 and strands_nothing.container_count == 3


def test_a_slot_limit_raises_the_container_bound():
    """Five items into containers holding two each need three containers, by counting alone."""
    assert bounds._container_bound([1] * 5, [0] * 5, False, 5, [object()], [1000], [None], [2]) == 3


def test_a_nesting_item_occupies_less_than_its_box():
    nesting = packvium.Item.create(
        "n", packvium.Dimensions.mm(10, 10, 10), 1, nesting_height=packvium.Length.mm(5))
    plain = packvium.Item.create("p", packvium.Dimensions.mm(10, 10, 10), 1)
    assert bounds._occupies_less_than_its_box(nesting)
    assert not bounds._occupies_less_than_its_box(plain)


def test_a_score_below_its_bound_is_refused_and_an_attained_one_reports_no_gap():
    bound = bounds.Bounds(1, 1, 0, 0, 0)
    with pytest.raises(bounds.UnsoundBoundError):
        bounds.gap([0, 1, 0, 0, 0], bound)
    attained = bounds.gap([1, 1, 0, 0, 0], bound)
    assert attained.attained and attained.key is None and attained.absolute == 0
    missed = bounds.gap([2, 1, 0, 0, 0], bound)
    assert not missed.attained and missed.key == 0 and missed.relative == (1, 1)


def test_a_nesting_request_drops_the_volume_argument_end_to_end():
    """The branch found unsound, exercised through `compute` rather than the helper.

    Five 10mm cubes cannot fit one 10mm container by volume, and the bound says four are
    stranded. Declare a nesting height on the same items and the volume argument is dropped
    entirely -- nominal volumes stop summing to anything a solution must carry -- so the
    bound stops claiming anything is stranded at all.
    """
    container = packvium.Container.create("c", packvium.Dimensions.mm(10, 10, 10), quantity=1)
    solid = packvium.Item.create("s", packvium.Dimensions.mm(10, 10, 10), 1, quantity=5)
    nesting = packvium.Item.create("n", packvium.Dimensions.mm(10, 10, 10), 1, quantity=5,
                                   nesting_height=packvium.Length.mm(5))
    of = lambda item: packvium.PackingRequest((item,), (container,)).instances
    assert bounds.compute(of(solid), (container,)).unpacked_count == 4
    nested = bounds.compute(of(nesting), (container,))
    assert nested.unpacked_count == 0
    # And every key that rests on the same argument goes with it rather than half-applying.
    assert nested.unused_volume_ppm == 0 and nested.stack_height_ppm == 0


def test_a_container_with_no_usable_volume_still_bounds_the_count_by_its_other_limits():
    """`max(usable) == 0` must skip the volume term, not divide by it.

    A container whose usable volume is zero still carries a payload limit, and the count
    bound has to come from that instead of from a division nobody can perform.
    """
    assert bounds._container_bound([1, 1], [5, 5], False, 2, [object()], [0], [4], [None]) == 3
