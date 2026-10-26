"""An effort budget names a stop, never a counter that merely reached its limit.

A start that finishes on exactly its budget was not stopped, so it is reported complete; a
counted step past a limit is refused, so no counter ends above its limit; and under an effort
budget the clock is the request's fuse, never sliced per start.
"""
from __future__ import annotations

import json

import pytest

from packvium import Container, Dimensions, Item, PackingConfig
from packvium.effort import EffortBudget
from packvium.serialization import pack_from_dict
from packvium.solvers import (ContainerState, Deadline, MaximalSpaceSolver, SearchStats,
                              TimeLimitReached, _start_budget, beam_pack, default_constraints,
                              find_candidates)


def request(budget: dict | None, *, solvers: list[str] | None = None, profile: str = "fast",
            items: list[dict] | None = None) -> dict:
    configuration: dict = {"solver_profile": profile, "time_limit_ms": 60_000}
    if budget is not None:
        configuration["effort_budget"] = budget
    if solvers is not None:
        configuration["solvers"] = solvers
    return {
        "units": {"length": "mm"},
        "configuration": configuration,
        "items": items or [
            {"id": "a", "quantity": 3, "dimensions": {"length": "40", "width": "30", "height": "20"}},
            {"id": "b", "quantity": 3, "dimensions": {"length": "25", "width": "25", "height": "35"}},
        ],
        "containers": [{"id": "c", "inner_dimensions": {"length": "100", "width": "80", "height": "90"}}],
    }


def pack(payload: dict) -> dict:
    result = pack_from_dict(payload)
    return result if isinstance(result, dict) else result.to_dict()


CUBES = [{"id": "cube", "quantity": 60, "dimensions": {"length": "10", "width": "10", "height": "10"}}]


def test_a_start_that_finishes_on_exactly_its_budget_is_not_stopped():
    unlimited = pack(request(None))
    used = unlimited["algorithm"]["placements_attempted"]

    exact = pack(request({"max_placement_attempts": used}))

    assert exact["algorithm"]["effort_limit_reached"] is False
    assert exact["termination"]["code"] == "complete"
    assert exact["termination"]["winning_start_truncated"] is False
    assert exact["score"] == unlimited["score"]
    assert exact["containers"] == unlimited["containers"]


def test_a_budget_one_short_stops_the_start_on_its_limit():
    unlimited = pack(request(None))
    used = unlimited["algorithm"]["placements_attempted"]

    short = pack(request({"max_placement_attempts": used - 1}))

    assert short["termination"]["any_start_truncated"] is True
    assert short["algorithm"]["placements_attempted"] <= used - 1
    truncated = [start for start in short["termination"]["starts"] if start["truncated"]]
    assert truncated and all(not start["completed"] for start in truncated)


def test_a_block_commit_places_only_what_the_budget_has_room_for():
    result = pack(request({"max_placement_attempts": 7}, solvers=["homogeneous_blocks"],
                          profile="quality", items=CUBES))

    assert result["algorithm"]["placements_attempted"] == 7
    assert result["algorithm"]["candidates_evaluated"] == 7
    assert result["summary"]["packed_item_count"] == 7
    assert result["algorithm"]["effort_limit_reached"] is True
    assert result["termination"]["code"] == "effort_limit"


def test_a_lattice_too_large_for_the_budget_is_placed_one_counted_item_at_a_time():
    payload = request({"max_candidates_evaluated": 7}, solvers=["grid"], items=CUBES)
    payload["configuration"]["require_placement_coordinates"] = False

    result = pack(payload)

    assert result["algorithm"]["candidates_evaluated"] == 7
    assert result["summary"]["packed_item_count"] == 7
    assert result["algorithm"]["effort_limit_reached"] is True


def test_an_effort_budget_gives_every_start_the_whole_remaining_clock():
    deadline = Deadline(1_000, clock=lambda: 0)

    budgeted = _start_budget(deadline, EffortBudget(max_search_nodes=5), SearchStats(), 8)
    unbudgeted = _start_budget(deadline, None, SearchStats(), 8)

    assert budgeted.remaining_ns == 1_000_000_000
    assert unbudgeted.remaining_ns == 1_000_000_000 // 8


def test_a_poll_on_a_spent_budget_records_the_stop_and_a_later_read_does_not():
    stats = SearchStats()
    deadline = Deadline(60_000).with_effort(EffortBudget(max_search_nodes=3), stats)
    stats.search_nodes_expanded = 3

    assert deadline.stopped is False
    assert deadline.effort_stopped is False
    assert deadline.expired is True
    assert deadline.effort_stopped is True
    assert deadline.stopped is True


# The m1 cart (conformance's regression-effort-budget-truncated-loser), with a budget that
# never binds.
M1_CART = {
    "units": {"length": "cm"},
    "configuration": {
        "objective": "lowest_landed_cost",
        "effort_budget": {"max_candidates_evaluated": 1_000_000, "max_placement_attempts": 1_000_000,
                          "max_search_nodes": 1_000_000},
        "time_limit_ms": 300_000, "dimensional_weight_length_unit": "cm",
        "dimensional_weight_weight_unit": "kg", "dimensional_weight_divisor": 5000,
    },
    "items": [
        {"id": "product-21", "quantity": 10, "dimensions": {"length": "22", "width": "18", "height": "25"},
         "weight": "1.4 kg"},
        {"id": "product-22", "quantity": 20, "dimensions": {"length": "9", "width": "9", "height": "10"},
         "weight": "0.25 kg"},
        {"id": "product-11", "quantity": 20, "dimensions": {"length": "10", "width": "8", "height": "9"},
         "weight": "0.35 kg"},
    ],
    "containers": [
        {"id": "small", "inner_dimensions": {"length": "25", "width": "20", "height": "15"}, "cost_minor": 40,
         "rate_table": {"weight_brackets_g": [1000, 2000, 5000, 10000], "prices_minor": [495, 695, 995, 1495]}},
        {"id": "large", "inner_dimensions": {"length": "40", "width": "30", "height": "30"},
         "outer_dimensions": {"length": "41", "width": "31", "height": "31"}, "cost_minor": 90,
         "rate_table": {"weight_brackets_g": [3000, 20000], "prices_minor": [1200, 1900],
                        "minimum_charge_minor": 1000, "fuel_surcharge_permille": 85}},
    ],
}


def test_a_budget_that_never_binds_gives_one_answer_or_says_the_clock_fired():
    """(b): with the clock not sliced per start, a run either finishes every start and
    gives the one answer, or the fuse fired as a whole and `time_limit_reached` says so."""
    payload = json.loads(json.dumps(M1_CART))
    reference = pack(payload)
    payload["configuration"]["time_limit_ms"] = 1_500

    for _ in range(3):
        result = pack(payload)
        assert result["algorithm"]["time_limit_reached"] or (
            result["score"], result["containers"]) == (reference["score"], reference["containers"])


def budgeted(stats: SearchStats, **limits: int) -> Deadline:
    return Deadline(60_000).with_effort(EffortBudget(**limits), stats)


def cubes(quantity: int):
    return Item.create("cube", Dimensions.mm(30, 20, 10), quantity=quantity).instances()


BOX = Container.create("c", Dimensions.mm(100, 100, 100))


def test_a_feasible_candidate_past_its_limit_stops_the_scan_inside_a_point():
    config = PackingConfig.balanced()
    stats = SearchStats()
    deadline = budgeted(stats, max_candidates_evaluated=1)

    with pytest.raises(TimeLimitReached):
        find_candidates(ContainerState(BOX, 1), cubes(1)[0], config, default_constraints(config),
                        stats, deadline, 1)

    assert stats.candidates_evaluated == 1
    assert stats.candidate_points_considered == 1
    assert deadline.effort_stopped


@pytest.mark.parametrize("config", [PackingConfig.balanced(), PackingConfig.quality()],
                         ids=["greedy", "beam"])
def test_a_beam_node_reached_with_a_spent_budget_is_refused_and_not_counted(config):
    stats = SearchStats(search_nodes_expanded=2)
    deadline = budgeted(stats, max_search_nodes=2)

    solution = beam_pack(BOX, 1, cubes(3), config, default_constraints(config), stats, deadline)

    assert stats.search_nodes_expanded == 2
    assert solution.time_limit_reached
    assert deadline.effort_stopped


def test_a_maximal_space_item_reached_with_a_spent_budget_is_refused_and_not_counted():
    config = PackingConfig.quality()
    stats = SearchStats(search_nodes_expanded=2)
    deadline = budgeted(stats, max_search_nodes=2)

    solution = MaximalSpaceSolver().pack_one(BOX, 1, cubes(3), config, stats, deadline)

    assert stats.search_nodes_expanded == 2
    assert len(solution.unpacked) == 3 and solution.time_limit_reached
    assert deadline.effort_stopped


def test_a_wide_beam_stops_on_every_placement_limit_short_of_its_use_and_only_there():
    config = PackingConfig.quality()
    full = SearchStats()
    beam_pack(BOX, 1, cubes(3), config, default_constraints(config), full, Deadline(60_000))
    for limit in range(1, full.placements_attempted + 1):
        stats = SearchStats()
        deadline = budgeted(stats, max_placement_attempts=limit)
        beam_pack(BOX, 1, cubes(3), config, default_constraints(config), stats, deadline)
        assert stats.placements_attempted <= limit
        assert deadline.effort_stopped == (limit < full.placements_attempted)
