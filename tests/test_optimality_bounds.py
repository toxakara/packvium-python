"""The request-level bound is computed where the design says, once per pack.

The cross-implementation agreement and corpus soundness live in
`conformance/tests/test_optimality_bounds_engine.py`. What is asserted here is the wiring:
that the packer computes the bound exactly once per pack, before the first solver, for every
solver and every covered objective, from every instance of the request; that the result
states it as OPTIMALITY-GAP.md § 1.1 says; and that the number is sound against
the packing that pack produced.

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
from packvium.extensions import DefaultSolutionScorer  # noqa: E402

#: Four differently-shaped items. Identical items reach `GridSolver`'s quantity-compression
#: fast path, which answers before most solvers are consulted -- a request that would make
#: "for every solver" true of only one of them.
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

#: Every built-in profile, and the global container-set beam, which the old per-start bound
#: was confined to together with `exact_small`.
SOLVES = [
    {"solver_profile": "fast"},
    {"solver_profile": "balanced"},
    {"solver_profile": "quality"},
    {"solver_profile": "exact_small"},
    {"container_plan_beam_width": 4},
]


#: Taken once, so that a second `_pack` in one test wraps the real functions rather than the
#: first call's spies.
_FOR_REQUEST = bounds.for_request
_SOLVE = solvers.SolverOrchestrator.solve


def _spy(monkeypatch):
    """Record every request-level bound, and its order against the solves."""
    recorded: list = []
    events: list[str] = []

    def spy_bound(instances, containers, objective):
        computed = _FOR_REQUEST(instances, containers, objective)
        recorded.append(computed)
        events.append("bound")
        return computed

    def spy_solve(self, *arguments, **keywords):
        events.append("solve")
        return _SOLVE(self, *arguments, **keywords)

    monkeypatch.setattr(bounds, "for_request", spy_bound)
    monkeypatch.setattr(solvers.SolverOrchestrator, "solve", spy_solve)
    return recorded, events


def _pack(monkeypatch, configuration, request=None):
    """Pack, and return the result with every request-level bound and when it was taken.

    `events` interleaves "bound" with "solve" so that "before the first solver" is read from
    the order of calls rather than assumed.
    """
    recorded, events = _spy(monkeypatch)
    document = request or {"containers": CONTAINERS, "items": ITEMS}
    document = dict(document, configuration={"time_limit_ms": 20000, **configuration})
    return packvium.pack_from_dict(document), recorded, events


@pytest.mark.parametrize("configuration", SOLVES, ids=lambda c: str(next(iter(c.values()))))
def test_one_bound_per_pack_before_the_first_solver_for_every_solver(monkeypatch, configuration):
    """F3: the old bound was taken per start, and only on the beam or `exact_small`."""
    result, recorded, events = _pack(monkeypatch, configuration)
    assert events[0] == "bound" and events.count("bound") == 1, events
    assert recorded[0] is not None
    assert tuple(int(c) for c in result["score"]) >= recorded[0]


def test_the_bound_does_not_depend_on_the_deadline(monkeypatch):
    """A start the deadline never reaches used to record nothing, so presence followed the
    clock. Computed from the request alone, the bound is the same under any budget."""
    _, generous, _ = _pack(monkeypatch, {"solver_profile": "quality"})
    _, starved, _ = _pack(monkeypatch, {"solver_profile": "quality", "time_limit_ms": 1})
    assert generous == starved and generous[0] is not None


def test_lowest_cost_gets_the_same_bound_in_its_own_order(monkeypatch):
    """`lowest_cost` scores `(count, cost, containers, unused, height)`; the bound is the same
    five numbers moved to those positions, and still never above the packing."""
    _, default, _ = _pack(monkeypatch, {"solver_profile": "exact_small"})
    result, cheapest, _ = _pack(
        monkeypatch, {"solver_profile": "exact_small", "objective": "lowest_cost"})
    unpacked, count, cost, unused, height = default[0]
    assert cheapest[0] == (unpacked, cost, count, unused, height)
    assert tuple(int(c) for c in result["score"]) >= cheapest[0]


@pytest.mark.parametrize("objective", ["maximum_value", "maximum_volume",
                                       "open_dimension_height"])
def test_an_objective_without_a_bound_gets_none(monkeypatch, objective):
    """Their score positions rank other quantities, so a vector compared against them would
    line up different things -- recording nothing is the honest answer."""
    _, recorded, _ = _pack(monkeypatch, {"solver_profile": "exact_small", "objective": objective})
    assert recorded == [None]


def test_a_custom_solution_scorer_gets_no_bound():
    """A caller-defined score's positions are not the bound's quantities."""
    request = packvium.PackingRequest(
        tuple(packvium.Item.create(f"i{n}", packvium.Dimensions.mm(10, 10, 10)) for n in range(2)),
        (packvium.Container.create("c", packvium.Dimensions.mm(100, 100, 100)),))
    assert packvium.Packer()._objective_bound(request) is not None
    custom = packvium.Packer(solution_scorer=DefaultSolutionScorer())
    assert custom._objective_bound(request) is None


def test_a_request_past_the_ceiling_withholds_the_bound_and_is_still_answered(monkeypatch):
    """F5's Python half: the refusal declines to state a bound; it never refuses the pack."""
    recorded, _ = _spy(monkeypatch)
    heavy = packvium.Item.create("heavy", packvium.Dimensions.mm(10, 10, 10), weight=10 ** 31)
    result = packvium.Packer(packvium.PackingConfig.quality(time_limit_ms=20000)).pack(
        [heavy], [packvium.Container.create("c", packvium.Dimensions.mm(100, 100, 100))])
    assert recorded == [None]
    assert result.complete


#: Finding F1, reproduced end to end. A 100 mm cube holds a fixed 100 x 100 x 50 mm slab and
#: one free slab of the same size, so the only packing fills it exactly.
FIXED_SLAB = {
    "items": [{"id": "slab", "quantity": 2,
               "dimensions": {"length": "100", "width": "100", "height": "50"}}],
    "containers": [{"id": "cube", "quantity": 1,
                    "inner_dimensions": {"length": "100", "width": "100", "height": "100"}}],
    "fixed_placements": [{"item_type": "slab", "container_type": "cube", "container_instance": 1,
                          "position": {"x": "0", "y": "0", "z": "0"}, "orientation": "LWH"}],
}


def test_a_fixed_placement_is_counted_by_the_bound(monkeypatch):
    """F1: the bound used to see only the free slab and claim half the cube must stay empty.

    That was 500000 ppm on key 3 against a packing scoring 0 -- a lower bound above a packing
    that exists. Over every instance the bound is the packing's own score: attained, and so a
    proof.
    """
    result, recorded, _ = _pack(monkeypatch, {"solver_profile": "quality"}, FIXED_SLAB)
    score = tuple(int(c) for c in result["score"])
    assert score == (0, 1, 0, 0, 1_000_000)
    assert recorded == [score]
    assert bounds.gap(score, bounds.Bounds(*recorded[0])).attained

    slab = packvium.Item.create("slab", packvium.Dimensions.mm(100, 100, 50), quantity=2)
    cube = packvium.Container.create("cube", packvium.Dimensions.mm(100, 100, 100), quantity=1)
    free_only = bounds.compute(slab.instances()[1:], (cube,))
    with pytest.raises(bounds.UnsoundBoundError):
        bounds.gap(score, free_only)


def test_the_result_states_the_bound_it_was_given(monkeypatch):
    """The one bound per pack is what the result states, prefix-truncated at the
    first position its score misses, with the gap there (OPTIMALITY-GAP.md § 1.1).

    Also the practical half: `algorithm.metrics` is serialised into every result, so the
    bound belongs in `optimality` and nowhere else.
    """
    result, recorded, _ = _pack(monkeypatch, {"solver_profile": "exact_small"})
    assert recorded[0] is not None, "the assertions below would pass vacuously"
    leaked = [key for key in result["algorithm"]["metrics"] if "bound" in key or "gap" in key]
    assert not leaked, f"the bound leaked into algorithm.metrics as {leaked}"
    score = tuple(int(c) for c in result["score"])
    stated = bounds.statement(score, recorded[0])
    assert {key: value for key, value in result["optimality"].items() if key != "code"} == stated
    for alternative in result["alternatives"]:
        assert {key: value for key, value in alternative["optimality"].items()
                if key != "code"} == bounds.statement(
                    tuple(int(c) for c in alternative["score"]), recorded[0])


def test_exhausting_a_search_is_no_longer_a_proof():
    """Q1: `exact_small` exhausting its candidates on one container used to set `optimal`.
    Its candidates are only the points it generated, so that is not a proof; attainment is.
    Four unlike items cannot fill the cube, so the bound is missed and the result says so."""
    result = packvium.pack_from_dict({"containers": CONTAINERS, "items": ITEMS,
                                      "configuration": {"solver_profile": "exact_small",
                                                        "time_limit_ms": 20000}})
    assert result["complete"] and len(result["containers"]) == 1
    assert result["status"] == "feasible"
    assert result["optimality"]["code"] == "not_proven" and "gap_key" in result["optimality"]


def test_a_complete_result_that_attains_its_bound_is_proven_optimal():
    """Q1: attainment sets `optimal` and `proven_optimal`, and the field carries the whole
    bound with no gap: absence of a gap, next to a bound, is the proof."""
    result = packvium.pack_from_dict(FIXED_SLAB)
    assert result["status"] == "optimal"
    assert result["optimality"] == {"code": "proven_optimal", "lower_bound": [0, 1, 0, 0, 1_000_000]}


def test_an_incomplete_result_that_attains_its_bound_keeps_best_found():
    """§ 1.3: `code` is defined by completeness for an incomplete result; the proof is still
    readable from `lower_bound` without `gap_key`."""
    result = packvium.pack_from_dict({
        "items": [{"id": "i", "quantity": 3,
                   "dimensions": {"length": "100", "width": "100", "height": "100"}}],
        "containers": [{"id": "c", "quantity": 1,
                        "inner_dimensions": {"length": "100", "width": "100", "height": "100"}}],
    })
    assert result["status"] == "best_found" and not result["complete"]
    assert result["optimality"] == {"code": "best_found", "lower_bound": [2, 1, 0, 0, 1_000_000]}


def test_a_statement_stops_at_the_first_missed_position():
    """The prefix, the unreduced relative gap, its absence on a zero bound, and nothing at all
    for a score below its bound, which shows the bound wrong."""
    bound = (0, 2, 10, 500, 7)
    assert bounds.statement(bound, bound) == {"lower_bound": list(bound)}
    assert bounds.statement((0, 2, 14, 0, 0), bound) == {
        "lower_bound": [0, 2, 10], "gap_key": 2, "absolute_gap": 4,
        "relative_gap": {"numerator": 4, "denominator": 10}}
    assert bounds.statement((1, 0, 0, 0, 0), bound) == {
        "lower_bound": [0], "gap_key": 0, "absolute_gap": 1}
    assert bounds.statement((0, 1, 99, 0, 0), bound) is None


def test_an_uncovered_objective_states_nothing():
    """§ 1.5: absence is "no statement", never "attained"."""
    result = packvium.pack_from_dict({"containers": CONTAINERS, "items": ITEMS,
                                      "configuration": {"objective": "maximum_value"}})
    assert set(result["optimality"]) == {"code"}


def test_the_statement_does_not_depend_on_the_deadline():
    """§ 6.3: under an effort budget the clock is only a fuse, so two different deadlines
    give the same answer, and the field with it."""
    def under(time_limit_ms):
        return packvium.pack_from_dict({
            "containers": CONTAINERS, "items": ITEMS,
            "configuration": {"solver_profile": "quality", "time_limit_ms": time_limit_ms,
                              "effort_budget": {"max_restarts": 4,
                                                "max_placement_attempts": 20000}}})
    first, second = under(60_000), under(120_000)
    assert "lower_bound" in first["optimality"]
    assert first["optimality"] == second["optimality"] and first["status"] == second["status"]


def test_the_solvers_no_longer_compute_a_bound_per_start():
    """The bound lives in the packer; a per-start copy would bring F3 back."""
    assert not hasattr(solvers.SolverOrchestrator, "_record_root_bound")
    assert not hasattr(solvers.SearchStats(), "objective_lower_bound")


def test_the_cost_bound_takes_the_cheapest_types_without_expanding_them():
    """F6: the same sum as the multiset of per-unit costs, from one sort of the types."""
    assert bounds._cost_bound([9, 5, 7], [3, 1, None], 4) == 5 + 7 * 3
    assert bounds._cost_bound([9, 5], [1, 1], 4) == 14
    assert bounds._cost_bound([3], [None], 10 ** 12) == 3 * 10 ** 12


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


def test_a_score_below_its_bound_withholds_the_statement_rather_than_publishing_it():
    """§ 1.5 condition 5: the bound is then the suspect, and an engine must not publish a
    claim it has just refuted. Unreachable through a sound bound, so stood in for here."""
    from packvium.packer import _stated_optimality
    from packvium.result import PackingStatus

    assert _stated_optimality(PackingStatus.FEASIBLE, (0, 1, 0, 0, 0), (0, 2, 0, 0, 0), ()) == (
        PackingStatus.FEASIBLE, None)


@pytest.mark.parametrize('cheap_cost', [0, 1])
def test_a_gap_past_the_portable_ceiling_withholds_only_the_statement(cheap_cost):
    request = {
        'items': [{'id': 'i', 'quantity': 2, 'dimensions': dict.fromkeys(('length', 'width', 'height'), '10')}],
        'containers': [
            {'id': 'cheap', 'inner_dimensions': dict.fromkeys(('length', 'width', 'height'), '1'),
             'cost_minor': cheap_cost, 'max_items': 1},
            {'id': 'big', 'inner_dimensions': dict.fromkeys(('length', 'width', 'height'), '10'),
             'cost_minor': bounds.MAX_BOUND_VALUE, 'max_items': 1},
        ],
    }
    result = packvium.pack_from_dict(request)
    assert result['score'][2] == 2 * bounds.MAX_BOUND_VALUE
    assert result['optimality'] == {'code': 'not_proven'}
    assert packvium.validate(request, result) == []
    result['optimality'].update(lower_bound=[0, 2, 2 * cheap_cost], gap_key=2,
                                absolute_gap=result['score'][2] - 2 * cheap_cost)
    if cheap_cost:
        result['optimality']['relative_gap'] = {
            'numerator': result['optimality']['absolute_gap'], 'denominator': 2 * cheap_cost}
    assert 'optimality_gap_mismatch' in [issue.code for issue in packvium.validate(request, result)]


@pytest.mark.parametrize('least', [0, 1])
def test_the_portable_gap_ceiling_is_inclusive(least):
    bound = (0, 1, least, 0, 0)
    statement = bounds.statement((0, 1, least + bounds.MAX_BOUND_VALUE, 0, 0), bound)
    assert statement['absolute_gap'] == bounds.MAX_BOUND_VALUE
    assert bounds.statement((0, 1, least + bounds.MAX_BOUND_VALUE + 1, 0, 0), bound) is None
