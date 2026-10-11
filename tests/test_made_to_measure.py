"""Made-to-measure containers: arithmetic, rungs, the trial chain, flow, admission
and validation, pinned to the worked numbers in docs/MADE-TO-MEASURE-CONTAINERS.md."""

from __future__ import annotations

import copy
import random
from dataclasses import replace

import pytest

from packvium import (Container, Dimensions, EffortBudget, InvalidRequestError, Item,
                      MadeToMeasure, PackingConfig, Point, Rotation, UnsupportedFeatureError,
                      pack_from_dict)
from packvium.made_to_measure import (MAX_EXACT_INTEGER, SQUARE_METRE_TICKS2, box_cost_minor,
                                      fabricate, icbrt_ceil, instance_edges, per_square_metre,
                                      solving_envelopes)
from packvium.models import PackedContainer, PackingRequest, Placement
from packvium.solvers import (ContainerState, Deadline, ExtremePointSolver, SearchStats,
                              SingleContainerSolution, SolverOrchestrator, TimeLimitReached)
from packvium.units import Length
from packvium.validation import IndependentSolutionValidator

MM = Length.TICKS_PER_MM


def machine(minimum: int = 60, increment: int = 25, **fields) -> MadeToMeasure:
    return MadeToMeasure((minimum * MM,) * 3, (increment * MM,) * 3, **fields)


def ticks(*sides_mm: int) -> tuple[int, ...]:
    return tuple(side * MM for side in sides_mm)


def mm(length, width, height) -> dict:
    return {"length": str(length), "width": str(width), "height": str(height)}


def request(**container_fields) -> dict:
    container = {"id": "machine", "cost_minor": 40, "inner_dimensions": mm(1510, 1510, 1510),
                 "made_to_measure": {"minimum": mm(60, 60, 60), "increment": mm(25, 25, 25),
                                     "cost_per_square_metre_minor": 1000,
                                     "outer_allowance": mm(3, 3, 3)}}
    container.update(container_fields)
    return {
        "units": {"length": "mm"},
        "configuration": {"solver_profile": "fast", "minimum_support_ratio": 1,
                          "time_limit_ms": 300000,
                          "effort_budget": {"max_search_nodes": 1_000_000}},
        "items": [{"id": "cube", "quantity": 8, "weight": "100", "dimensions": mm(100, 100, 100)}],
        "containers": [container],
    }


# ------------------------------------------------------------------------------ arithmetic


def test_grid_rounds_up_onto_the_machine_grid_and_never_below_the_minimum():
    spec = machine()
    assert spec.grid(0, 200 * MM) == 210 * MM
    assert spec.grid(0, 210 * MM) == 210 * MM
    assert spec.grid(0, 1) == 60 * MM
    assert spec.clamp(0, 2000 * MM, 1510 * MM) == 1510 * MM
    assert spec.clamp(0, 30 * MM, 1510 * MM) == 60 * MM


def test_the_split_cost_equals_the_direct_ceiling():
    generator = random.Random(2702)
    for _ in range(200):
        rate = generator.randrange(0, 2**53)
        area = generator.randrange(0, 2**110)
        assert per_square_metre(rate, area) == -(-rate * area // SQUARE_METRE_TICKS2)


def test_cost_rounds_a_sliver_of_material_up_to_a_whole_minor_unit():
    assert per_square_metre(1, 1) == 1
    assert per_square_metre(1000, 0) == 0


def test_the_worked_box_costs_its_fee_plus_its_material():
    assert box_cost_minor(machine(cost_per_square_metre_minor=1000), 40, ticks(210, 210, 210)) == 305


@pytest.mark.parametrize("value, root", [(0, 0), (1, 1), (7, 2), (8, 2), (9, 3),
                                         (10**21, 10**7), (10**21 + 1, 10**7 + 1)])
def test_the_integer_cube_root_is_the_least_cube_at_or_above(value, root):
    assert icbrt_ceil(value) == root


def test_a_machine_off_its_own_grid_is_refused():
    with pytest.raises(ValueError, match="does not divide"):
        Container("odd", Dimensions.mm(1500, 1510, 1510), made_to_measure=machine())
    with pytest.raises(ValueError, match="exceeds inner_dimensions"):
        Container("small", Dimensions.mm(50, 1510, 1510), made_to_measure=machine())
    with pytest.raises(ValueError, match="minimum must be positive"):
        MadeToMeasure((0, 1, 1), (1, 1, 1))
    with pytest.raises(ValueError, match="increment must be positive"):
        MadeToMeasure((1, 1, 1), (1, 0, 1))
    with pytest.raises(ValueError, match="prices cannot be negative"):
        MadeToMeasure((1, 1, 1), (1, 1, 1), tare_per_square_metre=-1)
    with pytest.raises(ValueError, match="outer_allowance cannot be negative"):
        MadeToMeasure((1, 1, 1), (1, 1, 1), outer_allowance=(0, -1, 0))


# ------------------------------------------------------------------------------ rungs


def cubes(count: int, side: int = 100) -> list[tuple[int, int, int]]:
    return [ticks(side, side, side)] * count


def test_eight_cubes_in_the_worked_machine_try_260_310_410_then_1510():
    rungs = solving_envelopes(machine(), ticks(1510, 1510, 1510), cubes(8))
    assert rungs == tuple(ticks(side, side, side) for side in (260, 310, 410, 1510))


def test_a_pinned_cross_section_keeps_only_the_maximum_rung():
    spec = MadeToMeasure(ticks(200, 100, 100), ticks(70, 10, 10))
    assert solving_envelopes(spec, ticks(1950, 100, 100), cubes(5)) == (ticks(1950, 100, 100),)


def test_three_cubes_in_the_small_machine_try_190_220_290_then_300():
    spec = machine(50, 10)
    rungs = solving_envelopes(spec, ticks(300, 300, 300), cubes(3))
    assert rungs == tuple(ticks(side, side, side) for side in (190, 220, 290, 300))


def test_a_rod_floors_the_length_only():
    edges = [ticks(1000, 50, 50), ticks(100, 100, 100)]
    first = solving_envelopes(machine(), ticks(1510, 1510, 1510), edges)[0]
    assert first[0] >= 1000 * MM
    assert first[1] < 1000 * MM and first[2] < 1000 * MM


def test_rungs_ignore_the_order_of_the_instances():
    edges = [ticks(300, 20, 40), ticks(100, 100, 100), ticks(50, 60, 70)]
    assert (solving_envelopes(machine(), ticks(1510, 1510, 1510), edges)
            == solving_envelopes(machine(), ticks(1510, 1510, 1510), list(reversed(edges))))


def test_clearance_grows_every_instance_edge_twice():
    assert instance_edges([Dimensions.mm(10, 20, 30)], 2 * MM) == [ticks(14, 24, 34)]


# ------------------------------------------------------------------------------ the chain


class ScriptedSolver:
    """Places a scripted number of items per rung, along x at the scripted pitch."""

    name = "scripted"

    def __init__(self, script: list[tuple[int, int]]):
        self.script = script
        self.envelopes: list[Dimensions] = []

    def pack_one(self, container, sequence, items, config, stats, deadline):
        count, pitch = self.script[len(self.envelopes)]
        self.envelopes.append(container.inner_dimensions)
        stats.search_nodes_expanded += 1
        state = ContainerState(container, sequence)
        for index, instance in enumerate(items[:count]):
            origin = Point(index * pitch * MM, 0, 0)
            state.add_direct(Placement(instance, origin, Rotation.LWH, instance.dimensions,
                                       origin, instance.dimensions, 1.0))
        return SingleContainerSolution(state, tuple(items[count:]))


def chain(solver, deadline=None, count: int = 3):
    declared = Container("machine", Dimensions.mm(1510, 1510, 1510), cost_minor=40,
                         made_to_measure=machine(cost_per_square_metre_minor=1000))
    items = Item.create("cube", Dimensions.mm(100, 100, 100), 0, quantity=count).instances()
    return SolverOrchestrator()._made_to_measure_trial(
        solver, declared, 1, items, PackingConfig.fast(), SearchStats(),
        deadline or Deadline(60_000))


def test_the_first_complete_rung_is_kept_without_trying_a_larger_one():
    solver = ScriptedSolver([(3, 100)] * 4)
    one, trials = chain(solver)
    assert trials == 1 and len(solver.envelopes) == 1
    assert one.state.container.inner_dimensions == Dimensions.mm(310, 110, 110)
    assert not one.exhaustive


def test_an_incomplete_chain_keeps_the_round_key_choice_and_the_smaller_rung_on_a_tie():
    # Rungs 2 and 4 place as many and cut the same 310 mm box; rung 2's placements win.
    solver = ScriptedSolver([(1, 100), (2, 100), (1, 100), (2, 105)])
    one, trials = chain(solver)
    assert trials == 4
    assert [p.position.x for p in one.state.placements] == [0, 100 * MM]


def test_a_chain_that_places_nothing_stays_a_zero_placement_trial():
    one, trials = chain(ScriptedSolver([(0, 100)] * 4))
    assert trials == 4 and one.state.placement_count == 0


def test_a_counted_limit_reached_mid_chain_stops_it_with_the_best_so_far():
    stats = SearchStats()
    deadline = Deadline(60_000).with_effort(EffortBudget(max_search_nodes=1), stats)
    solver = ScriptedSolver([(1, 100)] * 4)

    def counted(container, sequence, items, config, _stats, budget):
        return ScriptedSolver.pack_one(solver, container, sequence, items, config, stats, budget)

    solver.pack_one = counted
    one, trials = chain(solver, deadline)
    assert trials == 1 and one.time_limit_reached
    assert one.state.placement_count == 1


class InterruptedSolver(ScriptedSolver):
    """Runs out of time on the scripted rung, as a deadline polled mid-trial does."""

    def __init__(self, script, interrupted_at: int):
        super().__init__(script)
        self.interrupted_at = interrupted_at

    def pack_one(self, container, sequence, items, config, stats, deadline):
        if len(self.envelopes) == self.interrupted_at:
            raise TimeLimitReached("packing time limit reached")
        return super().pack_one(container, sequence, items, config, stats, deadline)


def test_a_deadline_inside_a_later_rung_keeps_the_best_rung_so_far():
    one, trials = chain(InterruptedSolver([(1, 100)] * 4, interrupted_at=1))
    assert trials == 1 and one.time_limit_reached and one.state.placement_count == 1


def test_a_deadline_inside_the_first_rung_reaches_the_round_as_it_would_from_any_trial():
    with pytest.raises(TimeLimitReached):
        chain(InterruptedSolver([(1, 100)] * 4, interrupted_at=0))


# ------------------------------------------------------------------------------ flow


def test_the_worked_order_comes_back_as_a_210_mm_box_not_an_800_mm_row():
    result = pack_from_dict(request())
    packed = result["containers"][0]
    assert {side: packed["inner_dimensions"][side]["value"] for side in ("length", "width", "height")} \
        == {"length": "210", "width": "210", "height": "210"}
    assert packed["outer_dimensions"]["length"]["value"] == "213"
    assert packed["achieved_dimensions"]["height"]["value"] == "200"
    assert packed["made_to_measure"]["cost_minor"] == 305
    assert result["score"][2] == 305
    assert result["optimality"]["code"] == "not_proven"


def test_the_board_weight_grows_the_tare_and_the_gross_weight():
    data = request()
    data["containers"][0]["made_to_measure"]["tare_per_square_metre"] = "500"
    packed = pack_from_dict(data)["containers"][0]
    assert packed["made_to_measure"]["tare_weight"]["value"] == "132.3"
    assert packed["gross_weight"]["value"] == "932.3"


def test_the_round_key_reads_the_fabricated_price():
    data = request()
    data["containers"].append({"id": "stock", "cost_minor": 300, "inner_dimensions": mm(210, 210, 210)})
    for profile in ("fast", "quality"):
        data["configuration"]["solver_profile"] = profile
        assert pack_from_dict(copy.deepcopy(data))["containers"][0]["container_type"] == "stock"


def test_the_compact_lattice_form_stands_down():
    data = request()
    data["configuration"]["require_placement_coordinates"] = False
    packed = pack_from_dict(data)["containers"][0]
    assert "lattice_summary" not in packed and len(packed["placements"]) == 8


def test_an_exact_search_is_never_reported_optimal_and_sets_no_root_bound():
    data = request()
    data["configuration"]["solver_profile"] = "exact_small"
    data["items"][0]["quantity"] = 2
    assert pack_from_dict(data)["optimality"]["code"] == "not_proven"
    stats = SearchStats()
    declared = Container("machine", Dimensions.mm(1510, 1510, 1510), made_to_measure=machine())
    SolverOrchestrator._record_root_bound(ExtremePointSolver(), True, (), (declared,),
                                          PackingConfig.quality(), stats)
    assert stats.objective_lower_bound is None


def test_a_load_larger_than_the_machine_opens_a_second_compact_box():
    data = request(inner_dimensions=mm(300, 300, 300),
                   made_to_measure={"minimum": mm(50, 50, 50), "increment": mm(10, 10, 10),
                                    "cost_per_square_metre_minor": 1000})
    data["items"][0]["quantity"] = 30
    costs = [packed["made_to_measure"]["cost_minor"] for packed in pack_from_dict(data)["containers"]]
    assert costs == [580, 200]


# ------------------------------------------------------------------------------ admission


@pytest.mark.parametrize("fields, name", [
    ({"axles": [{"position": "0"}, {"position": "1510"}]}, "container.made_to_measure+axles"),
    ({"obstacles": [{"id": "post", "dimensions": mm(10, 10, 10)}]}, "container.made_to_measure+obstacles"),
    ({"void_fill_reserve_ratio": 0.5}, "container.made_to_measure+void_fill_reserve_ratio"),
])
def test_a_field_that_reads_the_declared_frame_is_refused(fields, name):
    with pytest.raises(UnsupportedFeatureError, match=name.replace("+", r"\+")):
        pack_from_dict(request(**fields))


def test_an_empty_obstacle_list_and_a_zero_reserve_are_admitted():
    pack_from_dict(request(obstacles=[], void_fill_reserve_ratio=0))


def test_a_malformed_container_id_beside_a_machine_is_left_to_the_request_rules():
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(request(id=7))
    assert (caught.value.field, caught.value.detail) == ("/containers/0/id", "must be a string")


def test_a_fixed_placement_in_a_made_to_measure_container_is_refused():
    data = request()
    data["fixed_placements"] = [{"item_type": "cube", "container_type": "machine", "orientation": "LWH"}]
    with pytest.raises(UnsupportedFeatureError, match=r"fixed_placements\+container\.made_to_measure"):
        pack_from_dict(data)


def edit_machine(**fields):
    def edit(data):
        data["containers"][0]["made_to_measure"].update(fields)
    return edit


@pytest.mark.parametrize("edit, reason, field, detail", [
    (lambda d: d["containers"][0].update(made_to_measure=[]), "wrong_type",
     "/containers/0/made_to_measure", "must be an object"),
    (edit_machine(maximum=mm(1, 1, 1), extra=1), "not_allowed",
     "/containers/0/made_to_measure/extra", "is not a known field"),
    (lambda d: d["containers"][0]["made_to_measure"].pop("minimum"), "missing_field",
     "/containers/0/made_to_measure/minimum", "is required"),
    (lambda d: d["containers"][0]["made_to_measure"].pop("increment"), "missing_field",
     "/containers/0/made_to_measure/increment", "is required"),
    (edit_machine(minimum=mm(60, "-1", 60)), "negative_measure",
     "/containers/0/made_to_measure/minimum/width", "cannot be negative"),
    (edit_machine(outer_allowance=mm(3, 3, "x")), "wrong_type",
     "/containers/0/made_to_measure/outer_allowance/height", "must be a measure"),
    (edit_machine(cost_per_square_metre_minor=-1), "below_minimum",
     "/containers/0/made_to_measure/cost_per_square_metre_minor", "must be at least 0"),
    (edit_machine(tare_per_square_metre="-1"), "negative_measure",
     "/containers/0/made_to_measure/tare_per_square_metre", "cannot be negative"),
    (lambda d: d["containers"][0].update(outer_dimensions=mm(1600, 1600, 1600)), "not_allowed",
     "/containers/0/outer_dimensions", "cannot be combined with made_to_measure"),
    (edit_machine(minimum=mm(60, 0, 60)), "invalid_value",
     "/containers/0/made_to_measure/minimum/width", "must be positive"),
    (edit_machine(increment=mm(25, 25, 0)), "invalid_value",
     "/containers/0/made_to_measure/increment/height", "must be positive"),
    (edit_machine(minimum=mm(1600, 60, 60)), "invalid_value",
     "/containers/0/made_to_measure/minimum/length", "exceeds inner_dimensions"),
    (edit_machine(increment=mm(25, 30, 25)), "invalid_value",
     "/containers/0/made_to_measure/increment/width", "does not divide inner_dimensions minus minimum"),
    (edit_machine(cost_per_square_metre_minor=2**53 - 1), "invalid_value",
     "/containers/0/made_to_measure/cost_per_square_metre_minor",
     "prices the largest box above 9007199254740991"),
])
def test_each_request_error_names_its_rule(edit, reason, field, detail):
    data = request()
    edit(data)
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(data)
    assert (caught.value.reason, caught.value.field, caught.value.detail) == (reason, field, detail)


def test_the_price_ceiling_admits_the_largest_box_exactly_at_the_limit():
    data = request(inner_dimensions=mm(1000, 1000, 1000), cost_minor=0,
                   made_to_measure={"minimum": mm(1000, 1000, 1000), "increment": mm(1, 1, 1)})
    # Six square metres: the largest price whose box stays within 2^53 - 1.
    data["containers"][0]["made_to_measure"]["cost_per_square_metre_minor"] = MAX_EXACT_INTEGER // 6
    pack_from_dict(copy.deepcopy(data))
    data["containers"][0]["made_to_measure"]["cost_per_square_metre_minor"] = MAX_EXACT_INTEGER // 6 + 1
    with pytest.raises(InvalidRequestError, match="prices the largest box"):
        pack_from_dict(data)


# ------------------------------------------------------------------------------ validation


def test_the_engine_validator_recuts_the_box_from_the_declared_machine():
    declared = Container("machine", Dimensions.mm(1510, 1510, 1510), cost_minor=40,
                         made_to_measure=machine(cost_per_square_metre_minor=1000))
    items = (Item.create("cube", Dimensions.mm(100, 100, 100), 0),)
    instance = items[0].instances()[0]
    placement = Placement(instance, Point(0, 0, 0), Rotation.LWH, instance.dimensions,
                          Point(0, 0, 0), instance.dimensions, 1.0)
    good = fabricate(declared, ticks(100, 100, 100))
    request_ = PackingRequest(items, (declared,))
    validator = IndependentSolutionValidator()
    assert validator.validate(request_, (PackedContainer(good, 1, (placement,)),)).valid
    stale = replace(good, cost_minor=40)
    codes = [issue.code for issue in
             validator.validate(request_, (PackedContainer(stale, 1, (placement,)),)).issues]
    assert codes == ["made_to_measure_cost_mismatch"]
