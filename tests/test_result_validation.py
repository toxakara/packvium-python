"""`packvium.validate(request, result)`, the reference validator, shipped in the package.

The validator checks the engines, so its worth is that it shares no code with them. These tests
hold the module to importing nothing else from `packvium`, and pin the public entry point.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import pytest

import packvium
from packvium import MalformedResult, pack_from_dict, validate

MODULE = Path(packvium.__file__).with_name("result_validation.py")

REQUEST = {
    "items": [{"id": "cube", "quantity": 2, "dimensions": {"length": "10", "width": "10", "height": "10"}}],
    "containers": [{"id": "box", "inner_dimensions": {"length": "20", "width": "10", "height": "10"}}],
}


#: The Python-version dataclass shim every package model routes through; it holds no engine code.
ALLOWED = {"_compat"}


def test_the_validator_imports_nothing_from_the_engine() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    from_packvium = [
        ast.dump(node) for node in imported
        if isinstance(node, ast.ImportFrom)
        and (node.level > 0 or (node.module or "").startswith("packvium"))
        and (node.module or "").removeprefix("packvium.") not in ALLOWED
        or isinstance(node, ast.Import) and any(alias.name.startswith("packvium") for alias in node.names)
    ]
    assert from_packvium == []


def test_a_result_the_engine_packed_is_valid() -> None:
    assert validate(REQUEST, pack_from_dict(REQUEST)) == []


def test_two_items_in_one_place_are_a_collision() -> None:
    result = pack_from_dict(REQUEST)
    placements = result["containers"][0]["placements"]
    placements[1]["position"] = copy.deepcopy(placements[0]["position"])
    # One place is one id (1.10): the moved item restates it, and shares it.
    placements[1]["placement_id"] = placements[0]["placement_id"]
    assert [issue.code for issue in validate(REQUEST, result)] == ["collision", "duplicate_placement_id"]


def test_a_result_of_the_wrong_shape_is_refused_rather_than_judged() -> None:
    with pytest.raises(MalformedResult):
        validate(REQUEST, {"containers": "none"})


@pytest.mark.parametrize("units, key", [({"length": "mm", "weight": "kg"}, "weight"),
                                         ({"zeta": None, "alpha": "x", "length": "mm"}, "alpha")])
def test_a_units_key_other_than_length_is_a_contract_break(units, key) -> None:
    """The first unknown key in code-point order is named, as every engine's refusal does."""
    result = pack_from_dict(REQUEST)
    with pytest.raises(MalformedResult, match=rf"^units\.{key} is not a known field$"):
        validate({**REQUEST, "units": units}, result)


def test_the_conformance_suite_runs_this_same_module() -> None:
    import sys

    conformance = Path(__file__).resolve().parents[2] / "conformance"
    if not (conformance / "validator.py").is_file():
        pytest.skip("the conformance suite lives in the workspace only")
    sys.path.insert(0, str(conformance))
    try:
        import validator
    finally:
        sys.path.remove(str(conformance))
    assert validator.validate is validate


def test_an_issue_reads_as_its_code_and_detail() -> None:
    from packvium.result_validation import Issue

    assert str(Issue("collision", "a#1 with a#2")) == "collision: a#1 with a#2"


def test_the_corridor_replay_unloads_the_near_box_first_and_refuses_a_later_one_in_the_way() -> None:
    """Two boxes in a row with the only door at x = 0: the near one must leave first."""
    from packvium.result_validation import Box, corridor_replay_succeeds

    boxes = {"near": Box(0, 0, 0, 10, 10, 10), "far": Box(10, 0, 0, 10, 10, 10)}
    inner = (20, 10, 10)
    assert corridor_replay_succeeds(boxes, {"near": 0, "far": 0}, inner, ("-x",))
    assert corridor_replay_succeeds(boxes, {"near": 0, "far": 1}, inner, ("-x",))
    assert not corridor_replay_succeeds(boxes, {"near": 1, "far": 0}, inner, ("-x",))


# ------------------------------------------------- the optimality bound and the gap

#: Exactly filled by the packing, so every bound position is attained: (0, 1, 0, 0, 1000000).
ATTAINED = REQUEST

#: One cube in a box for two: the height key misses its bound, 1000000 against 500000 (the
#: unused-volume key attains its 500000).
GAP = {
    "items": [{"id": "cube", "quantity": 1, "dimensions": {"length": "10", "width": "10", "height": "10"}}],
    "containers": [{"id": "box", "inner_dimensions": {"length": "20", "width": "10", "height": "10"}}],
}

WORKSPACE = Path(__file__).resolve().parents[2]


def _emitting(request: dict) -> dict:
    """The engine's result carrying the fields of docs/OPTIMALITY-GAP.md § 1.1 and § 1.3, as an
    engine emits them once lands; written here from the document, not the validator."""
    from packvium.result_validation import objective_lower_bound

    result = pack_from_dict(request)
    bound = objective_lower_bound(request)
    assert bound is not None
    score = result["score"]
    missed = next((key for key, (achieved, floor) in enumerate(zip(score, bound)) if achieved != floor), None)
    fact = {"code": result["optimality"]["code"]}
    if missed is None:
        fact["lower_bound"] = list(bound)
        result["status"], fact["code"] = "optimal", "proven_optimal"
    else:
        fact.update(lower_bound=list(bound[:missed + 1]), gap_key=missed,
                    absolute_gap=score[missed] - bound[missed])
        if bound[missed]:
            fact["relative_gap"] = {"numerator": score[missed] - bound[missed], "denominator": bound[missed]}
    result["optimality"] = fact
    return result


def _as_an_older_producer(result: dict) -> dict:
    """The same result as a producer from before the field states it: no bound and no gap,
    and the status its completeness alone gives."""
    for name in ("lower_bound", "gap_key", "absolute_gap", "relative_gap"):
        result["optimality"].pop(name, None)
    if result["status"] == "optimal":
        result["status"], result["optimality"]["code"] = "feasible", "not_proven"
    return result


def _codes(request: dict, result: dict) -> list[str]:
    return [issue.code for issue in validate(request, result)]


def test_a_result_stating_an_attained_bound_is_valid() -> None:
    result = _emitting(ATTAINED)
    assert result["optimality"] == {"code": "proven_optimal", "lower_bound": [0, 1, 0, 0, 1000000]}
    assert validate(ATTAINED, result) == []


def test_a_result_stating_a_gap_is_valid() -> None:
    result = _emitting(GAP)
    assert result["optimality"] == {
        "code": "not_proven", "lower_bound": [0, 1, 0, 500000, 500000], "gap_key": 4,
        "absolute_gap": 500000, "relative_gap": {"numerator": 500000, "denominator": 500000},
    }
    assert validate(GAP, result) == []


def test_the_engine_emits_what_the_document_says() -> None:
    """The engine's own fields, against the ones written here from the document."""
    for request in (ATTAINED, GAP):
        result = pack_from_dict(request)
        expected = _emitting(request)
        assert (result["status"], result["optimality"]) == (expected["status"], expected["optimality"])


def test_absence_is_never_an_issue() -> None:
    for request in (ATTAINED, GAP):
        result = _as_an_older_producer(pack_from_dict(request))
        assert "lower_bound" not in result["optimality"]
        assert validate(request, result) == []


def test_a_wrong_bound_value_is_an_optimality_bound_mismatch() -> None:
    result = _emitting(GAP)
    result["optimality"]["lower_bound"][1] = 2
    assert _codes(GAP, result) == ["optimality_bound_mismatch"]


def test_a_bound_where_none_is_stated_is_an_optimality_bound_mismatch() -> None:
    request = {**GAP, "configuration": {
        "objective": "shipping_cost", "dimensional_weight_divisor": 5000,
        "dimensional_weight_length_unit": "cm", "dimensional_weight_weight_unit": "kg"}}
    result = pack_from_dict(request)
    result["optimality"]["lower_bound"] = [0]
    assert [(issue.code, issue.detail) for issue in validate(request, result)] == [(
        "optimality_bound_mismatch",
        "lower_bound is reported where no bound is stated: objective 'shipping_cost' has no bound",
    )]


def test_a_wrong_absolute_gap_is_an_optimality_gap_mismatch() -> None:
    result = _emitting(GAP)
    result["optimality"]["absolute_gap"] += 1
    assert [(issue.code, issue.detail) for issue in validate(GAP, result)] == [
        ("optimality_gap_mismatch", "absolute_gap should be 500000"),
    ]


def test_a_reduced_relative_gap_and_a_short_prefix_are_optimality_gap_mismatches() -> None:
    reduced = _emitting(GAP)
    reduced["optimality"]["relative_gap"] = {"numerator": 1, "denominator": 1}
    assert _codes(GAP, reduced) == ["optimality_gap_mismatch"]
    short = _emitting(GAP)
    short["optimality"]["lower_bound"].pop()
    assert [(issue.code, issue.detail) for issue in validate(GAP, short)] == [
        ("optimality_gap_mismatch", "lower_bound has 4 entries, expected 5"),
    ]


def test_a_score_below_the_recomputed_bound_is_reported_with_or_without_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    from packvium import result_validation

    # A bound above what a valid packing scores can only be a wrong bound: stand one in.
    monkeypatch.setattr(result_validation, "lower_bound", lambda *inputs: (0, 2, 0, 0, 1000000))
    stating = pack_from_dict(ATTAINED)
    result = _as_an_older_producer(pack_from_dict(ATTAINED))
    assert [(issue.code, issue.detail) for issue in validate(ATTAINED, result)] == [
        ("score_below_bound", "score[1] is 1, below its lower bound 2"),
    ]
    assert _codes(ATTAINED, stating)[:2] == ["score_below_bound", "optimality_bound_mismatch"]


def test_a_misreported_score_below_the_bound_is_both_a_score_mismatch_and_below_it() -> None:
    result = _as_an_older_producer(pack_from_dict(GAP))
    result["score"][1] = 0
    assert _codes(GAP, result) == ["score_mismatch", "score_below_bound"]


def test_proven_optimal_without_an_attained_lower_bound_is_an_unproven_claim() -> None:
    result = _emitting(ATTAINED)
    del result["optimality"]["lower_bound"]
    assert _codes(ATTAINED, result) == ["unproven_optimality_claim"]


def test_optimal_is_accepted_exactly_when_the_bound_is_attained_and_the_result_complete() -> None:
    attained_but_feasible = _emitting(ATTAINED)
    attained_but_feasible["status"] = "feasible"
    attained_but_feasible["optimality"]["code"] = "not_proven"
    assert _codes(ATTAINED, attained_but_feasible) == ["result_fact_mismatch"]

    missed_but_optimal = _emitting(GAP)
    missed_but_optimal["status"] = "optimal"
    missed_but_optimal["optimality"]["code"] = "proven_optimal"
    assert _codes(GAP, missed_but_optimal) == ["result_fact_mismatch", "unproven_optimality_claim"]


def test_lowest_cost_reads_the_same_bound_in_its_own_order() -> None:
    from packvium.result_validation import objective_lower_bound

    priced = {**GAP, "containers": [{**GAP["containers"][0], "cost_minor": 7}]}
    assert objective_lower_bound(priced) == (0, 1, 7, 500000, 500000)
    assert objective_lower_bound({**priced, "configuration": {"objective": "lowest_cost"}}) == (0, 7, 1, 500000, 500000)


def test_each_objective_key_of_its_own_is_recomputed() -> None:
    """The oracle's first worked example, as a request -- three 600-volume items, one
    slot; the two cheapest values forgone, one item's 6 mm (96000 ticks), 800 of 1800 left out."""
    from packvium.result_validation import objective_lower_bound

    request = {"items": [{"id": f"i{value}", "value": value,
                          "dimensions": {"length": "10", "width": "10", "height": "6"}}
                         for value in (50, 10, 90)],
               "containers": [{"id": "c", "quantity": 1,
                               "inner_dimensions": {"length": "10", "width": "10", "height": "10"}}]}
    expected = {"maximum_value": (2, 60, 1, 0, 400000), "open_dimension_height": (2, 96000, 1, 0, 400000),
                "maximum_volume": (444445, 2, 1, 0, 400000)}
    for objective, vector in expected.items():
        assert objective_lower_bound({**request, "configuration": {"objective": objective}}) == vector


def test_an_objective_key_of_its_own_abstains_or_withholds_alone() -> None:
    """Nesting drops both volume terms; unlimited stock forces no volume out; a value sum past
    2^53 - 1 withholds `maximum_value` and leaves every other objective stated."""
    from packvium.result_validation import objective_lower_bound

    def under(objective: str, request: dict) -> Any:
        return objective_lower_bound({**request, "configuration": {"objective": objective}})

    box = {"length": "10", "width": "10", "height": "6"}
    one_slot = [{"id": "c", "quantity": 1, "inner_dimensions": {"length": "10", "width": "10", "height": "10"}}]
    nesting = {"items": [{"id": "cup", "quantity": 3, "dimensions": box, "nesting_height": "1"}],
               "containers": one_slot}
    assert under("maximum_volume", nesting)[0] == 0
    assert under("open_dimension_height", nesting)[1] == 0
    unlimited = {"items": [{"id": "i", "quantity": 3, "dimensions": box}],
                 "containers": [{**one_slot[0], "quantity": None}]}
    assert under("maximum_volume", unlimited)[0] == 0
    precious = {"items": [{"id": "i", "quantity": 3, "value": 2 ** 53 - 1, "dimensions": box}],
                "containers": one_slot}
    assert under("maximum_value", precious) is None
    assert under("open_dimension_height", precious) == (2, 96000, 1, 0, 400000)


def test_no_bound_is_stated_past_the_declared_ceiling_or_for_a_made_to_measure_container() -> None:
    from packvium.result_validation import objective_lower_bound

    huge = {"items": [{"id": "slab", "dimensions": {"length": "700", "width": "700", "height": "700"}}],
            "units": {"length": "m"},
            "containers": [{"id": "hall", "inner_dimensions": {"length": "800", "width": "800", "height": "800"}}]}
    assert objective_lower_bound(huge) is None
    cut = {**GAP, "containers": [{**GAP["containers"][0], "made_to_measure": {"increment": {
        "length": "1", "width": "1", "height": "1"}}}]}
    assert objective_lower_bound(cut) is None


def _oracle() -> Any:
    import importlib.util
    import sys

    path = WORKSPACE / "scripts" / "optimality_bounds.py"
    if not path.is_file():
        pytest.skip("the oracle lives in the workspace only")
    spec = importlib.util.spec_from_file_location("optimality_bounds_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `dataclass` resolves the module through `sys.modules` while the class body runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def _validator_bound(instances: Any, containers: Any) -> "tuple[int, ...] | None":
    """The validator's formulas over the oracle's own dataclasses."""
    from packvium.result_validation import BoundContainer, _runs, lower_bound

    return lower_bound(
        _runs([(instance.volume, 1) for instance in instances]),
        _runs([(instance.weight, 1) for instance in instances]),
        any(instance.nests for instance in instances),
        [BoundContainer(c.inner_volume, c.base_area, c.height, c.usable_volume, c.cost_minor,
                        c.payload, c.max_items, c.quantity) for c in containers],
    )


def test_the_validator_bound_matches_the_oracle_on_every_worked_and_refusal_example() -> None:
    oracle = _oracle()
    for example in (*oracle.WORKED_EXAMPLES, oracle.PERMUTATION_EXAMPLE, *oracle.REFUSAL_EXAMPLES):
        refused = oracle.refusal(example["instances"], example["containers"])
        expected = None if refused is not None else oracle.bounds(
            example["instances"], example["containers"]).as_vector()
        assert _validator_bound(example["instances"], example["containers"]) == expected, example["name"]


def test_the_validator_bound_matches_the_shared_scene_and_reads_each_fixture_as_it_does() -> None:
    import json

    from packvium.result_validation import (BoundContainer, _bound_inputs, _parse_containers, _parse_items,
                                            _runs, lower_bound)

    path = WORKSPACE / "conformance" / "scene" / "objective-bounds.json"
    if not path.is_file():
        pytest.skip("the shared scene lives in the workspace only")
    scene = json.loads(path.read_text(encoding="utf-8"))
    for case in (*scene["cases"], *scene["edges"]):
        instances = case["instances"]
        computed = lower_bound(
            _runs([(int(instance["volume"]), 1) for instance in instances]),
            _runs([(int(instance["weight"]), 1) for instance in instances]),
            any(instance["shrinks"] for instance in instances),
            [BoundContainer(int(c["inner"]), c["base_area"], c["height"], int(c["usable"]), c["cost_minor"],
                            c["payload"], c["max_items"], c["quantity"]) for c in case["containers"]],
        )
        assert (None if computed is None else list(computed)) == case["bounds"], case.get("name", case.get("fixture"))
    read = 0
    for case in scene["cases"]:
        if case["truncated"]:
            continue
        request = json.loads((WORKSPACE / "conformance" / "fixtures" / case["fixture"]).read_text(encoding="utf-8"))
        unit = request.get("units", {}).get("length", "mm")
        inputs = _bound_inputs(_parse_items(request, unit), _parse_containers(request, unit))
        assert lower_bound(*inputs) == tuple(case["bounds"]), case["fixture"]
        read += 1
    assert read > 400


def test_degenerate_capacities_are_read_as_every_port_reads_them() -> None:
    from packvium.result_validation import BoundContainer, lower_bound

    reserved_past_full = BoundContainer(1000, 100, 10, -1, 3, 5, None, 1)
    assert lower_bound(((0, 2),), ((1, 2),), False, [reserved_past_full]) == (2, 0, 0, 0, 0)
    unlimited = BoundContainer(1000, 100, 10, -1, 3, 5, None, None)
    assert lower_bound(((0, 2),), ((1, 2),), False, [unlimited]) == (0, 1, 3, 1000000, 0)
    holds_nothing = BoundContainer(1000, 100, 10, 1000, 0, 0, 0, None)
    assert lower_bound(((1, 1),), ((0, 1),), False, [holds_nothing]) == (1, 0, 0, 0, 0)
    assert lower_bound(((1, 1),), ((0, 1),), False, []) == (1, 0, 0, 0, 0)
    weightless = BoundContainer(1000, 100, 10, 1000, 0, 0, None, None)
    assert lower_bound(((1, 1),), ((0, 1),), False, [weightless]) == (0, 1, 0, 999000, 100000)


def test_no_bound_is_stated_for_an_invalid_result() -> None:
    result = pack_from_dict(GAP)
    result["status"] = "invalid_result"
    result["optimality"] = {"code": "not_proven", "lower_bound": [0]}
    assert ("lower_bound is reported where no bound is stated: status is 'invalid_result'"
            in [issue.detail for issue in validate(GAP, result)])


#: Counted work decides these packs; the wall clock is only a fuse.
COUNTED = {"time_limit_ms": 60_000, "effort_budget": {"max_candidates_evaluated": 100_000}}


def _cups(container: dict, configuration: dict) -> tuple[dict, dict]:
    """Two nesting cups of one type, packed by the engine."""
    request = {
        "items": [{"id": "cup", "quantity": 2, "nesting_height": "2",
                   "dimensions": {"length": "10", "width": "10", "height": "10"}}],
        "containers": [{"id": "c", "inner_dimensions": container}],
        "configuration": {**COUNTED, **configuration},
    }
    return request, pack_from_dict(request)


def test_nesting_cups_side_by_side_are_two_whole_volumes() -> None:
    """Same type, same nesting height -- but different footprints, so they do not nest."""
    request, result = _cups({"length": "20", "width": "10", "height": "10"}, {})
    packed, = result["containers"]
    assert {p["position"]["z"]["ticks"] for p in packed["placements"]} == {0}
    assert int(packed["used_volume_ticks3"]) == 2 * 160_000 ** 3
    assert validate(request, result) == []


def test_a_nesting_overlap_the_clearance_uses_up_deducts_no_volume() -> None:
    """The envelopes sink by the nesting height, but a 1 mm clearance on each side of a
    2 mm nest leaves the items themselves no overlap, so both count whole."""
    request, result = _cups({"length": "12", "width": "12", "height": "40"}, {"clearance": "1"})
    packed, = result["containers"]
    lower, upper = sorted(p["position"]["z"]["ticks"] for p in packed["placements"])
    assert upper - lower == 160_000
    assert int(packed["used_volume_ticks3"]) == 2 * 160_000 ** 3
    assert validate(request, result) == []

    packed["used_volume_ticks3"] = str(2 * 160_000 ** 3 - 1)
    assert [issue.code for issue in validate(request, result)] == ["used_volume_mismatch"]


def test_a_covered_item_resting_on_a_full_top_is_accepted() -> None:
    """Every base corner of the lid lies on the box beneath it, so `covered` holds."""
    side = {"length": "10", "width": "10"}
    request = {
        "items": [{"id": "base", "dimensions": {**side, "height": "10"}, "must_be_on_floor": True},
                  {"id": "lid", "dimensions": {**side, "height": "5"}, "ground_contact_rule": "covered"}],
        "containers": [{"id": "c", "inner_dimensions": {**side, "height": "15"}}],
        "configuration": COUNTED,
    }
    result = pack_from_dict(request)
    lid, = [p for p in result["containers"][0]["placements"] if p["item_type"] == "lid"]
    assert lid["position"]["z"]["ticks"] == 160_000
    assert validate(request, result) == []


def test_a_container_without_volume_counts_but_adds_no_unused_or_height_share() -> None:
    """A flat container cannot be divided by. It still counts as opened, and the score is
    recomputed without it in the ratio terms rather than failing on a zero divisor."""
    packed = {**REQUEST, "configuration": COUNTED}
    request = copy.deepcopy(packed)
    request["containers"].append({"id": "flat", "inner_dimensions": {"length": "20", "width": "10", "height": "0"}})
    result = pack_from_dict(packed)
    result["containers"].append({**copy.deepcopy(result["containers"][0]), "id": "flat#1",
                                 "container_type": "flat", "placements": [], "used_volume_ticks3": "0"})
    assert [str(issue) for issue in validate(request, result) if issue.code == "score_mismatch"] == [
        "score_mismatch: reported [0, 1, 0, 0, 1000000], recomputed [0, 2, 0, 0, 1000000]"]
