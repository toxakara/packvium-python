"""The dictionary API — the wire contract every language binding shares.

`pack_from_dict` is what the CLI, the conformance runner and every non-Python caller
go through, so its shape is a compatibility surface: a missing key or a value that
stops being a string is a breaking change for somebody.
"""

from __future__ import annotations

import json

import pytest

from packvium import (
    AlgorithmReport,
    Length,
    PackingResult,
    PackingStatus,
    ReasonProof,
    ResultFact,
    StartRecord,
    aggregate_termination,
    pack_from_dict,
)


def mm(length, width, height) -> dict:
    return {"length": str(length), "width": str(width), "height": str(height)}


def request(items, containers, **rest) -> dict:
    return {"units": {"length": "mm"}, "items": items, "containers": containers, **rest}


def cube(id: str, size=100, **rest) -> dict:
    return {"id": id, "dimensions": mm(size, size, size), **rest}


def box(id: str, length=200, width=200, height=200, **rest) -> dict:
    return {"id": id, "inner_dimensions": mm(length, width, height), **rest}


# ---------------------------------------------------------------------- shape

def test_the_result_is_plain_json():
    """No custom types leak out: every caller can hand this straight to a serializer."""
    result = pack_from_dict(request([cube("a", 50)], [box("c")]))
    assert json.loads(json.dumps(result)) == result


def test_the_documented_top_level_keys_are_all_present():
    result = pack_from_dict(request([cube("a", 50)], [box("c")]))
    assert set(result) == {
        "status", "feasibility", "termination", "optimality", "complete", "objective",
        "algorithm", "summary", "score", "containers", "unpacked_items",
        "catalog_versions_used", "warnings", "alternatives",
    }


def test_catalog_versions_are_pinned_in_result_and_duplicates_are_rejected():
    references = [
        {"catalog_id": "items", "version": 7, "effective_at": 10, "resolved_at": 20},
        {"catalog_id": "cartons", "version": 3, "effective_at": 11, "resolved_at": 20},
    ]
    payload = request([cube("a", 50)], [box("c")], catalog_versions_used=references)
    assert pack_from_dict(payload)["catalog_versions_used"] == references
    payload["catalog_versions_used"].append(dict(references[0]))
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        pack_from_dict(payload)


def test_the_three_result_axes_are_independent_of_the_legacy_status():
    result = PackingResult(
        PackingStatus.FEASIBLE,
        (),
        (),
        AlgorithmReport("quality", "test", 1, 42, time_limit_reached=True),
        (0,),
    ).to_dict()
    assert result["status"] == "feasible"
    assert result["feasibility"] == {"code": "feasible"}
    assert result["termination"]["code"] == "time_limit"
    assert result["termination"]["winning_start_truncated"] is True
    assert result["optimality"] == {"code": "not_proven"}


def test_an_unknown_future_fact_code_round_trips_verbatim():
    fact = ResultFact.from_dict({"code": "node_limit", "limit": 10_000})
    assert fact.to_dict() == {"code": "node_limit", "limit": 10_000}


def test_a_finished_winner_is_distinct_from_a_truncated_loser():
    fact = aggregate_termination((
        StartRecord("winner", True, True, False, selected=True),
        StartRecord("loser", True, False, True),
    ))
    assert fact.to_dict() == {
        "code": "complete",
        "any_start_truncated": True,
        "all_required_starts_completed": False,
        "winning_start_truncated": False,
        "global_deadline_reached": False,
        "starts": [
            {
                "id": "winner", "started": True, "completed": True,
                "truncated": False, "selected": True,
                "global_deadline_reached": False,
            },
            {
                "id": "loser", "started": True, "completed": False,
                "truncated": True, "selected": False,
                "global_deadline_reached": False,
            },
        ],
    }


def test_a_truncated_winner_affects_the_returned_answer():
    fact = aggregate_termination((
        StartRecord("winner", True, False, True, selected=True),
        StartRecord("loser", True, True, False),
    ))
    assert fact.code == "time_limit"
    assert fact.attributes["any_start_truncated"] is True
    assert fact.attributes["all_required_starts_completed"] is False
    assert fact.attributes["winning_start_truncated"] is True
    assert fact.attributes["global_deadline_reached"] is False


def test_a_normal_pack_reports_verifiable_per_start_termination():
    result = pack_from_dict(request([cube("a", 50)], [box("c")]))
    termination = result["termination"]
    starts = termination["starts"]
    assert sum(start["selected"] for start in starts) == 1
    assert termination["any_start_truncated"] == any(start["truncated"] for start in starts)
    assert termination["all_required_starts_completed"] == all(
        start["completed"] for start in starts
    )
    assert termination["winning_start_truncated"] == next(
        start["truncated"] for start in starts if start["selected"]
    )
    assert termination["global_deadline_reached"] == any(
        start["global_deadline_reached"] for start in starts
    )


def test_a_placement_reports_everything_a_caller_needs_to_reproduce_it():
    result = pack_from_dict(request([cube("a", 50)], [box("c")]))
    placement, = result["containers"][0]["placements"]
    assert set(placement) == {"item_id", "item_type", "position", "dimensions", "orientation",
                              "support_ratio", "top_load"}
    assert placement["item_id"] == "a#1"
    assert placement["item_type"] == "a"
    assert placement["orientation"] in {"LWH", "LHW", "WLH", "WHL", "HLW", "HWL"}


def test_a_container_reports_its_own_totals():
    result = pack_from_dict(request([cube("a", 50, quantity=2, weight="1 kg")], [box("c")]))
    packed = result["containers"][0]
    assert packed["id"] == "c#1"
    assert packed["container_type"] == "c"
    assert packed["payload_weight"]["ticks"] == 16_000_000_000
    assert int(packed["used_volume_ticks3"]) == 2 * (800_000 ** 3)


def test_a_container_reports_its_centre_of_mass_offset():
    result = pack_from_dict(request([cube("a", 50, quantity=2, weight="1 kg")], [box("c")]))
    packed = result["containers"][0]
    assert isinstance(packed["centre_of_mass_offset_ppm"], int)
    assert 0 <= packed["centre_of_mass_offset_ppm"] <= 1_000_000


def test_nesting_height_is_plumbed_through_and_fits_an_extra_layer():
    # A 100mm item nesting 40mm into the one below advances the stack by 60mm per
    # layer instead of 100mm, fitting a third into a 220mm-tall container.
    result = pack_from_dict(request(
        [{"id": "crate", "dimensions": mm(100, 100, 100), "quantity": 10, "nesting_height": {"value": "40", "unit": "mm"}}],
        [box("c", 100, 100, 220)],
        configuration={"solvers": ["grid"], "max_containers": 1},
    ))
    placed = sum(len(c["placements"]) for c in result["containers"])
    assert placed == 3


def test_the_score_is_a_list_of_exact_integers():
    """Serialized as JSON integers, never strings and never floats — a double cannot
    hold a container volume in cubic ticks exactly."""
    score = pack_from_dict(request([cube("a", 50)], [box("c")]))["score"]
    assert len(score) == 5
    assert all(isinstance(key, int) for key in score)


# --------------------------------------------------------------------- units

def test_dimensions_may_mix_notations_within_one_request():
    result = pack_from_dict(request(
        [{"id": "a", "quantity": 2,
          "dimensions": {"length": {"value": "4", "unit": "in"}, "width": "100", "height": "100"}}],
        [box("b", 210, 100, 100)],
    ))
    assert result["complete"]


def test_the_default_length_unit_is_millimetres():
    without = pack_from_dict({"items": [cube("a", 50)], "containers": [box("c")]})
    with_units = pack_from_dict(request([cube("a", 50)], [box("c")]))
    assert without["containers"][0]["inner_dimensions"] == with_units["containers"][0]["inner_dimensions"]


def test_the_request_unit_applies_to_every_measurement():
    result = pack_from_dict({"units": {"length": "in"}, "items": [cube("a", 1)],
                             "containers": [box("c", 10, 10, 10)]})
    assert result["containers"][0]["inner_dimensions"]["length"]["ticks"] == 10 * 406_400


def test_output_units_are_chosen_independently_of_the_input():
    result = pack_from_dict(request(
        [cube("a", 50, weight="1 kg")], [box("c")], output={"length_unit": "in", "weight_unit": "kg"},
    ))
    assert result["containers"][0]["inner_dimensions"]["length"]["unit"] == "in"
    assert result["containers"][0]["payload_weight"] == {"ticks": 8_000_000_000, "value": "1", "unit": "kg"}


# ------------------------------------------------------------- configuration

def test_the_solver_profile_and_seed_are_plumbed_through():
    result = pack_from_dict(request(
        [cube("a", 50)], [box("c")], configuration={"solver_profile": "quality", "seed": 7},
    ))
    assert result["algorithm"]["profile"] == "quality"
    assert result["algorithm"]["seed"] == 7


def test_an_unknown_profile_is_rejected_rather_than_silently_defaulted():
    with pytest.raises(ValueError):
        pack_from_dict(request([cube("a", 50)], [box("c")], configuration={"solver_profile": "magic"}))


def test_explicit_solvers_are_plumbed_through_and_drive_the_answer():
    result = pack_from_dict(request(
        [cube("a", 40, quantity=4)], [box("c")], configuration={"solvers": ["grid"]},
    ))
    assert result["algorithm"]["solver"].startswith("grid")


def test_explicit_solver_order_is_preserved_in_start_records():
    result = pack_from_dict(request(
        [cube("a", 40)], [box("c")],
        configuration={"solvers": ["grid", "layer", "extreme_points"]},
    ))
    assert [start["id"].split(":")[0] for start in result["termination"]["starts"]] == [
        "grid", "layer", "extreme_points",
    ]


def test_an_unknown_solver_name_is_rejected_rather_than_silently_defaulted():
    with pytest.raises(ValueError):
        pack_from_dict(request([cube("a", 50)], [box("c")], configuration={"solvers": ["brute_force"]}))


def test_a_container_stack_density_limit_is_plumbed_through_and_prevents_a_crush():
    # 550 kg stacked onto a 1 square-metre footprint crushes a 500 kg/m^2 limit even
    # though it is nowhere near the container's own payload capacity.
    result = pack_from_dict(request(
        [cube("base", 1000, weight="400 kg"), cube("load", 1000, weight="150 kg")],
        [box("dense", 1000, 1000, 200, max_stack_density="500 kg")],
        configuration={"solvers": ["extreme_points"]},
    ))
    assert result["status"] != "invalid_result"
    placed_ids = {p["item_id"] for c in result["containers"] for p in c["placements"]}
    assert placed_ids != {"base#1", "load#1"}


def test_axles_are_plumbed_through_and_prevent_an_overload():
    # 800 kg centred over a 1000mm span with axles at 100mm/900mm puts 400 kg on
    # each -- a 399 kg front limit rejects the single-item placement outright, so
    # the item can never be packed at all.
    result = pack_from_dict(request(
        [{"id": "heavy", "dimensions": mm(1000, 100, 100), "weight": "800 kg"}],
        [box("truck", 1000, 100, 100, axles=[
            {"position": {"value": "100", "unit": "mm"}, "max_load": {"value": "399", "unit": "kg"}},
            {"position": {"value": "900", "unit": "mm"}, "max_load": {"value": "500", "unit": "kg"}},
        ])],
        configuration={"solvers": ["extreme_points"]},
    ))
    assert result["status"] != "invalid_result"
    assert result["containers"] == []
    assert {u["item_id"] for u in result["unpacked_items"]} == {"heavy#1"}


def test_gross_axle_reactions_include_tare_and_are_serialized_as_exact_fractions():
    result = pack_from_dict(request(
        [{"id": "empty", "dimensions": mm(10, 10, 10)}],
        [box("truck", 20, 20, 20, tare_weight="100 kg", axles=[
            {"position": "5", "max_load": "50 kg"},
            {"position": "15", "max_load": "50 kg"},
        ])],
    ))
    reaction = result["containers"][0]["axle_reactions"]
    assert reaction["basis"] == "gross"
    assert reaction["front_numerator"] == reaction["rear_numerator"]


def test_an_effort_budget_is_plumbed_through_and_actually_bounds_the_search():
    result = pack_from_dict(request(
        [cube("cube", 20, quantity=20)], [box("c", 200, 200, 200)],
        configuration={
            "solvers": ["grid"],
            "time_limit_ms": 60_000,
            "effort_budget": {"max_search_nodes": 5},
        },
    ))
    placed = sum(len(c["placements"]) for c in result["containers"])
    assert placed == 5
    assert result["termination"]["code"] == "effort_limit"
    assert result["algorithm"]["time_limit_reached"] is False
    assert result["algorithm"]["effort_limit_reached"] is True


def test_the_container_budget_is_plumbed_through():
    result = pack_from_dict(request(
        [cube("a", 90, quantity=4)], [box("c", 100, 100, 100, quantity=10)],
        configuration={"max_containers": 2},
    ))
    assert result["summary"]["container_count"] == 2


def test_the_alternatives_count_is_plumbed_through():
    result = pack_from_dict(request(
        [cube("a", 40, quantity=6)], [box("c", 150, 150, 150, quantity=2)],
        configuration={"solver_profile": "quality", "time_limit_ms": 2000, "alternatives": 2},
    ))
    assert len(result["alternatives"]) <= 1


def test_clearance_is_plumbed_through_in_the_request_unit():
    result = pack_from_dict(request(
        [cube("a", 40, quantity=2)], [box("c")], configuration={"clearance": "2"},
    ))
    positions = [p["position"]["x"]["ticks"] for p in result["containers"][0]["placements"]]
    assert 2 * 16_000 in positions, "the first placement sits one clearance inside its envelope"


def test_a_support_requirement_is_plumbed_through():
    strict = pack_from_dict(request(
        [cube("a", 60, quantity=4)], [box("c", 100, 100, 400)],
        configuration={"minimum_support_ratio": 1.0},
    ))
    assert strict["status"] != "invalid_result"


def test_the_candidate_point_budget_is_supported_by_the_portable_api():
    result = pack_from_dict(request(
        [cube("a", 40, quantity=2)], [box("c")],
        configuration={"max_candidate_points": 16},
    ))
    assert result["complete"]


def test_the_candidate_point_budget_rejects_values_below_the_contract_minimum():
    with pytest.raises(ValueError, match="max_candidate_points"):
        pack_from_dict(request(
            [cube("a", 40)], [box("c")],
            configuration={"max_candidate_points": 15},
        ))


# ------------------------------------------------------------------ contents

def test_obstacles_survive_the_round_trip():
    result = pack_from_dict(request([cube("a", 40)], [box(
        "b", 100, 100, 100,
        obstacles=[{"id": "post", "origin": {"x": "0", "y": "0", "z": "0"},
                    "dimensions": mm(50, 50, 100)}],
    )]))
    assert result["complete"]


def test_a_non_rectangular_obstacle_is_expressed_as_a_union_of_boxes():
    """A tapered container corner: approximated by two boxes, not a
    single rectangle, and an item is correctly routed around both, not just the
    first one it happened to be constructed with."""
    result = pack_from_dict(request(
        [cube("filler", 20)],
        [box("tapered", 100, 100, 100, obstacles=[{
            "id": "taper", "origin": {"x": "0", "y": "0", "z": "0"}, "dimensions": mm(40, 100, 100),
            "additional_boxes": [{"origin": {"x": "60", "y": "0", "z": "0"}, "dimensions": mm(40, 100, 100)}],
        }])],
        configuration={"solvers": ["extreme_points"]},
    ))
    assert result["complete"]
    placement, = result["containers"][0]["placements"]
    x = placement["position"]["x"]["ticks"]
    assert Length.mm(40).ticks <= x <= Length.mm(60).ticks


def test_an_unplaceable_item_is_reported_with_its_reason():
    result = pack_from_dict(request([cube("a", 200)], [box("b", 100, 100, 100)]))
    unpacked, = result["unpacked_items"]
    assert unpacked == {"item_id": "a#1", "item_type": "a", "reason": "no_compatible_container_dimensions",
                        "details": [], "proof": {
                            "level": "proven",
                            "observations": [{
                                "code": "no_compatible_container_dimensions",
                                "count": 1,
                                "details": [],
                            }],
                        }}


@pytest.mark.parametrize(("reason", "level"), [
    ("no_compatible_container_dimensions", "proven"),
    ("payload_exceeded", "proven"),
    ("no_feasible_placement", "observed"),
    ("search_exhausted", "observed"),
    ("group_cannot_fit_together", "inferred"),
    ("time_limit", "unknown_due_to_limit"),
])
def test_reason_proof_levels_are_stable(reason, level):
    proof = ReasonProof.for_reason(reason)
    assert proof.level == level
    assert proof.observations[0].code == reason


def test_item_rules_survive_the_round_trip():
    result = pack_from_dict(request(
        [dict(cube("floor", 40, quantity=4), must_be_on_floor=True),
         dict(cube("kit", 30, quantity=2), group="kit")],
        [box("c", 100, 100, 100, quantity=4)],
    ))
    on_floor = [p for c in result["containers"] for p in c["placements"] if p["item_type"] == "floor"]
    assert all(p["position"]["z"]["ticks"] == 0 for p in on_floor)


def test_allowed_rotations_survive_the_round_trip():
    result = pack_from_dict(request(
        [{"id": "a", "dimensions": mm(120, 40, 60), "allowed_rotations": ["LWH"]}],
        [box("c", 60, 120, 40)],
    ))
    assert not result["complete"]


def test_alternatives_do_not_nest_indefinitely():
    """Each alternative is rendered without its own alternatives, so the payload stays
    bounded no matter how many starts ran."""
    result = pack_from_dict(request(
        [cube("a", 40, quantity=6)], [box("c", 150, 150, 150, quantity=2)],
        configuration={"solver_profile": "quality", "time_limit_ms": 2000},
    ))
    assert all(alternative["alternatives"] == [] for alternative in result["alternatives"])
