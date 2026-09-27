"""Structured request errors: one type, a closed reason, and the pointer of the bad value."""

from __future__ import annotations

import copy

import pytest

from packvium import EffortBudget, FixedPlacementError, InvalidRequestError, PackingConfig, pack_from_dict
from packvium.units import Length, Weight
from packvium.request_errors import REASONS, check_request, pointer


def request() -> dict:
    return {
        "items": [{"id": "cube", "quantity": 2, "weight": "1000",
                   "dimensions": {"length": "100", "width": "100", "height": "100"}}],
        "containers": [{"id": "box", "quantity": 2,
                        "inner_dimensions": {"length": "200", "width": "100", "height": "200"}}],
        "configuration": {"solver_profile": "fast"},
    }


def refusal(edit) -> InvalidRequestError:
    data = request()
    edit(data)
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(data)
    return caught.value


@pytest.mark.parametrize("edit, reason, field, detail", [
    (lambda d: d["items"][0].update(quantity=0), "below_minimum", "/items/0/quantity", "must be at least 1"),
    (lambda d: d["items"][0].update(quantity="2"), "wrong_type", "/items/0/quantity", "must be an integer"),
    (lambda d: d["items"][0].update(quantity=2**53), "above_maximum", "/items/0/quantity",
     "must be at most 9007199254740991"),
    (lambda d: d["items"][0].update(quantity=-2**60), "below_minimum", "/items/0/quantity", "must be at least 1"),
    (lambda d: d["items"][0]["dimensions"].update(width="-1"), "negative_measure",
     "/items/0/dimensions/width", "cannot be negative"),
    (lambda d: d["items"][0]["dimensions"].update(width={"value": "1", "unit": "furlong"}), "invalid_unit",
     "/items/0/dimensions/width", 'has an unknown unit "furlong"'),
    (lambda d: d["items"][0]["dimensions"].update(width={"unit": "mm"}), "wrong_type",
     "/items/0/dimensions/width", "must be a measure"),
    (lambda d: d["items"][0]["dimensions"].update(width=1.5), "wrong_type",
     "/items/0/dimensions/width", "must be a measure"),
    (lambda d: d["items"][0]["dimensions"].pop("height"), "missing_field", "/items/0/dimensions/height",
     "is required"),
    (lambda d: d["items"].append(copy.deepcopy(d["items"][0])), "duplicate_id", "/items/1/id",
     'repeats the id "cube"'),
    (lambda d: d["items"][0].update(minimum_support_ratio=2), "above_maximum",
     "/items/0/minimum_support_ratio", "must be at most 1"),
    (lambda d: d["items"][0].update(minimum_support_ratio="x"), "wrong_type",
     "/items/0/minimum_support_ratio", "must be a number"),
    (lambda d: d["items"][0].update(compression_ratio=-1), "below_minimum",
     "/items/0/compression_ratio", "must be at least 0"),
    (lambda d: d["containers"][0].update(tag_limits={"a/b~c": 0}), "below_minimum",
     "/containers/0/tag_limits/a~1b~0c", "must be at least 1"),
    (lambda d: d["containers"][0].update(rate_table={"prices_minor": [1, -2]}), "below_minimum",
     "/containers/0/rate_table/prices_minor/1", "must be at least 0"),
    (lambda d: d["containers"][0].update(obstacles=[{"id": "o", "origin": {"z": "-1"},
                                                     "dimensions": {"length": "1", "width": "1", "height": "1"}}]),
     "negative_measure", "/containers/0/obstacles/0/origin/z", "cannot be negative"),
    (lambda d: d.update(units={"length": 5}), "invalid_unit", "/units/length", "has an unknown unit 5"),
    (lambda d: d.update(items={"cube": 1}), "wrong_type", "/items", "must be a list"),
    (lambda d: d["items"].append("slab"), "wrong_type", "/items/1", "must be an object"),
    (lambda d: d["items"][0].update(id=7), "wrong_type", "/items/0/id", "must be a string"),
    (lambda d: d["items"][0]["dimensions"].update(width="ten"), "wrong_type",
     "/items/0/dimensions/width", "must be a measure"),
    (lambda d: d["items"][0]["dimensions"].update(width={"value": [1]}), "wrong_type",
     "/items/0/dimensions/width", "must be a measure"),
    (lambda d: d["items"][0]["dimensions"].update(width={"value": "1", "unit": 3}), "invalid_unit",
     "/items/0/dimensions/width", "has an unknown unit 3"),
    (lambda d: d["containers"][0].update(rate_table={"weight_brackets_g": 5}), "wrong_type",
     "/containers/0/rate_table/weight_brackets_g", "must be a list"),
    (lambda d: d["containers"][0].update(rate_table={"minimum_charge_minor": -1}), "below_minimum",
     "/containers/0/rate_table/minimum_charge_minor", "must be at least 0"),
    (lambda d: d["containers"][0].update(obstacles=[{"id": "o", "dimensions": {"length": "1", "width": "1"}}]),
     "missing_field", "/containers/0/obstacles/0/dimensions/height", "is required"),
    (lambda d: d["configuration"].update(solver_profile="fastest"), "not_allowed",
     "/configuration/solver_profile", 'must be one of ["fast","balanced","quality","exact_small"]'),
    (lambda d: d["configuration"].update(effort_budget={"max_restarts": 0}), "below_minimum",
     "/configuration/effort_budget/max_restarts", "must be at least 1"),
])
def test_a_malformed_request_names_the_value_and_the_rule(edit, reason, field, detail):
    error = refusal(edit)
    assert (error.code, error.reason, error.field, error.detail) == ("invalid_request", reason, field, detail)
    assert str(error) == f"invalid_request: {field}: {detail}"
    assert error.reason in REASONS


def test_the_error_is_still_a_value_error_for_existing_callers():
    assert isinstance(refusal(lambda d: d["items"][0].update(quantity=0)), ValueError)


def test_a_request_that_is_not_an_object_is_refused_as_a_whole():
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict([request()])
    assert (caught.value.field, str(caught.value)) == ("", "invalid_request: must be an object")


def test_what_the_rules_do_not_name_still_arrives_as_a_request_error():
    error = refusal(lambda d: d["items"][0].update(shape_type="blob"))
    assert (error.code, error.reason, error.field) == ("invalid_request", "invalid_value", "")


def test_a_fixed_placement_refusal_is_a_request_error_with_its_own_code():
    data = request()
    data["fixed_placements"] = [{"item_type": "cube", "container_type": "box", "orientation": "LWH",
                                 "position": {"x": "-5"}}]
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(data)
    error = caught.value
    assert isinstance(error, FixedPlacementError)
    assert (error.code, error.reason, error.field) == (
        "invalid_fixed_placement", "malformed", "/fixed_placements/0/position/x")
    assert str(error) == "invalid_fixed_placement: fixed_placements[0].position.x cannot be negative"


def test_a_fixed_set_that_cannot_hold_is_reported_against_the_whole_list():
    data = request()
    entry = {"item_type": "cube", "container_type": "box", "orientation": "LWH", "position": {"x": "0"}}
    data["fixed_placements"] = [entry, dict(entry)]
    with pytest.raises(FixedPlacementError) as caught:
        pack_from_dict(data)
    assert (caught.value.reason, caught.value.field) == ("cannot_hold", "/fixed_placements")


def test_null_optionals_are_absent_and_a_valid_request_passes():
    data = request()
    data["items"][0].update(max_stacked_items=None, stop_index=None)
    data["units"] = None
    data["configuration"]["effort_budget"] = None
    check_request(data)


def test_a_pointer_escapes_tilde_and_slash():
    assert pointer("containers", 0, "tag_limits", "a/b~c") == "/containers/0/tag_limits/a~1b~0c"


def test_a_units_object_without_a_length_keeps_millimetres():
    data = request()
    data["units"] = {}
    check_request(data)


def test_an_unparsable_fixed_coordinate_is_named():
    data = request()
    data["fixed_placements"] = [{"item_type": "cube", "container_type": "box", "orientation": "LWH",
                                 "position": {"y": {"value": [1]}}}]
    with pytest.raises(FixedPlacementError) as caught:
        pack_from_dict(data)
    assert str(caught.value) == "invalid_fixed_placement: fixed_placements[0].position.y is a measure"


@pytest.mark.parametrize("build, message", [
    (lambda: PackingConfig(max_containers=0), "max_containers must be at least 1"),
    (lambda: PackingConfig(max_candidate_points=15), "max_candidate_points must be at least 16"),
    (lambda: PackingConfig(minimum_support_ratio=2), "minimum_support_ratio must be between 0 and 1"),
    (lambda: PackingConfig(dimensional_weight_divisor=0), "dimensional_weight_divisor must be positive"),
    (lambda: PackingConfig(time_limit_ms=0), "positive configuration values required"),
    (lambda: EffortBudget(max_search_nodes=0), "effort_budget.max_search_nodes must be at least 1"),
])
def test_the_typed_api_keeps_its_own_guards(build, message):
    """A caller who builds the model in code never passes the JSON rule table."""
    with pytest.raises(ValueError, match=message):
        build()


def test_an_integral_float_is_a_whole_measure_and_a_fraction_is_refused():
    assert Length.parse(100.0).ticks == Length.parse(100).ticks
    assert Weight.parse(5.0).ticks == Weight.parse(5).ticks
    with pytest.raises(TypeError):
        Length.parse(1.5)
    with pytest.raises(TypeError):
        Weight.parse(1.5)
