"""Structured request errors: one type, a closed reason, and the pointer of the bad value."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from packvium import EffortBudget, FixedPlacementError, InvalidRequestError, PackingConfig, pack_from_dict
from packvium.units import Length, Weight
from packvium.request_errors import CONFIGURATION_FIELDS, REASONS, check_request, pointer


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
    (lambda d: d["configuration"].update(top_k=2, profile="balanced"), "not_allowed",
     "/configuration/profile", "is not a known field"),
    (lambda d: d["configuration"].update(effort_budget={"max_nodes": 1}), "not_allowed",
     "/configuration/effort_budget/max_nodes", "is not a known field"),
    (lambda d: d["configuration"].update(effort_budget={"max_restarts": 0}), "below_minimum",
     "/configuration/effort_budget/max_restarts", "must be at least 1"),
    (lambda d: d["items"][0].update(stacking_group=7), "wrong_type",
     "/items/0/stacking_group", "must be a string"),
    (lambda d: d["items"][0].update(stacking_group=""), "invalid_value",
     "/items/0/stacking_group", "must not be empty"),
    (lambda d: d["items"][0].update(accepts_stacking_groups="A"), "wrong_type",
     "/items/0/accepts_stacking_groups", "must be a list"),
    (lambda d: d["items"][0].update(accepts_stacking_groups=["A", 3]), "wrong_type",
     "/items/0/accepts_stacking_groups/1", "must be a string"),
    (lambda d: d["items"][0].update(accepts_stacking_groups=["A", ""]), "invalid_value",
     "/items/0/accepts_stacking_groups/1", "must not be empty"),
    (lambda d: d["items"][0].update(stacking_group="", accepts_stacking_groups=[""]), "invalid_value",
     "/items/0/stacking_group", "must not be empty"),
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
    # A model rule the schema does not state: an item needs at least one rotation. (An unknown
    # shape type was this case until made it a `not_allowed` rule.)
    error = refusal(lambda d: d["items"][0].update(allowed_rotations=[]))
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


def test_what_stacking_groups_do_not_refuse_is_accepted():
    data = request()
    data["items"][0].update(stacking_group="A", accepts_stacking_groups=["B", "B", "unknown"],
                            stackable=False)
    check_request(data)


def test_null_optionals_are_absent_and_a_valid_request_passes():
    data = request()
    data["items"][0].update(max_stacked_items=None, stop_index=None, stacking_group=None,
                            accepts_stacking_groups=None)
    data["units"] = None
    data["configuration"]["effort_budget"] = None
    check_request(data)


def test_a_request_parses_each_spelling_once(monkeypatch):
    """The check hands its parses to deserialisation. A value already parsed passes
    through `parse` untouched and is not counted."""
    parsed = []

    def spy(kind, parse):
        def counted(cls, value, *rest):
            if not isinstance(value, kind):
                parsed.append((kind.__name__, value))
            return parse(cls, value, *rest)
        return classmethod(counted)

    monkeypatch.setattr(Length, "parse", spy(Length, Length.parse.__func__))
    monkeypatch.setattr(Weight, "parse", spy(Weight, Weight.parse.__func__))
    pack_from_dict(request())
    assert sorted(parsed, key=str) == sorted([("Length", "100"), ("Length", "200"), ("Length", 0),
                                              ("Weight", "1000"), ("Weight", 0)], key=str)


def exact_fit_with_object_measures() -> tuple[dict, list[dict]]:
    """The exact-fit scene (eight 100 mm cubes in a 200 mm box) with its six sides written as `{value, unit}`."""
    def sides(value: str) -> dict:
        return {side: {"value": value, "unit": "mm"} for side in ("length", "width", "height")}
    data = {"configuration": {"solver_profile": "fast"},
            "items": [{"id": "cube", "quantity": 8, "dimensions": sides("100")}],
            "containers": [{"id": "box", "inner_dimensions": sides("200")}]}
    objects = [*data["items"][0]["dimensions"].values(), *data["containers"][0]["inner_dimensions"].values()]
    return data, objects


def counted_length_parses(monkeypatch) -> list:
    parsed: list = []
    parse = Length.parse.__func__

    def counted(cls, value, *rest):
        parsed.append(value)
        return parse(cls, value, *rest)
    monkeypatch.setattr(Length, "parse", classmethod(counted))
    return parsed


def test_a_request_parses_each_object_measure_once(monkeypatch):
    """An object spelling is parsed once per request, as a string is; before, the check
    and deserialisation parsed each of these six objects once each. Equal objects share it."""
    data, objects = exact_fit_with_object_measures()
    parsed = counted_length_parses(monkeypatch)
    answer = pack_from_dict(data)
    assert answer["status"] == "optimal"
    by_object = [sum(value is spelt for value in parsed) for spelt in objects]
    assert by_object == [1, 0, 0, 1, 0, 0]
    assert [value for value in parsed if isinstance(value, dict)] == [objects[0], objects[3]]


def test_a_refused_object_measure_is_refused_with_the_same_error_every_time():
    data, _ = exact_fit_with_object_measures()
    data["containers"][0]["inner_dimensions"]["width"] = {"value": "-1", "unit": "mm"}
    data["items"][0]["dimensions"]["width"] = {"value": "-1", "unit": "mm"}
    for _ in range(2):
        with pytest.raises(InvalidRequestError) as caught:
            pack_from_dict(data)
        assert (caught.value.reason, caught.value.field, caught.value.detail) == (
            "negative_measure", "/items/0/dimensions/width", "cannot be negative")


def test_string_and_object_spellings_of_one_value_are_parsed_once_each(monkeypatch):
    data, objects = exact_fit_with_object_measures()
    data["items"][0]["dimensions"]["width"] = "100"
    data["containers"][0]["inner_dimensions"].update(width="200 mm", height={"value": 200})
    parsed = counted_length_parses(monkeypatch)
    answer = pack_from_dict(data)
    assert answer["status"] == "optimal"
    spellings = [value if isinstance(value, str) else (value["value"], value.get("unit"))
                 for value in parsed if not isinstance(value, (int, Length))]
    assert sorted(spellings, key=str) == sorted(
        ["100", "200 mm", ("100", "mm"), ("200", "mm"), (200, None)], key=str)


def test_a_pointer_escapes_tilde_and_slash():
    assert pointer("containers", 0, "tag_limits", "a/b~c") == "/containers/0/tag_limits/a~1b~0c"


def test_a_units_object_without_a_length_keeps_millimetres():
    data = request()
    data["units"] = {}
    check_request(data)


@pytest.mark.parametrize("units, output", [(None, None), ({"length": None}, None),
                                           ({}, {"length_unit": None, "weight_unit": None})])
def test_a_null_unit_is_the_default_when_packing(units, output):
    """Null is the schema's default in `units` and `output` too: before, the rule table admitted
    it and deserialisation read it as a unit code, an AttributeError."""
    data = request()
    data.update(units=units, output=output)
    placed = pack_from_dict(data)["containers"][0]["placements"][0]
    assert placed["position"]["x"]["unit"] == "mm"


@pytest.mark.parametrize("length", [25, True, {"unit": "mm"}, ["mm"], "", "furlong"])
def test_a_units_length_that_is_no_known_code_is_an_unknown_unit(length):
    """The schema types `units.length` by its enum alone, so a number or an object is an unknown
    unit, as at every measure's own `unit`, not `wrong_type`."""
    data = request()
    data["units"] = {"length": length}
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(data)
    assert (caught.value.reason, caught.value.field) == ("invalid_unit", "/units/length")


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


def test_an_empty_tag_limits_object_is_accepted():
    data = request()
    data["containers"][0]["tag_limits"] = {}
    check_request(data)


def test_the_known_configuration_fields_are_the_schema_s():
    path = Path(__file__).resolve().parents[2] / "conformance" / "schema" / "packing-request.schema.json"
    # The schema sits beside this package in the workspace; a published copy does not carry it.
    if not path.is_file():
        pytest.skip("the request schema is not part of this package")
    schema = json.loads(path.read_text())
    configuration = schema["properties"]["configuration"]
    assert configuration["additionalProperties"] is False
    assert set(CONFIGURATION_FIELDS) == set(configuration["properties"])


@pytest.mark.parametrize("request_unit, output_unit", [
    ("cm", "cm"), ("MM", "mm"), ("IN", "in"), ("Inches", "in"), ("millimeter", "mm"), ("tick", "mm"),
])
def test_without_an_output_block_the_result_is_in_the_request_unit_spelt_as_its_code(request_unit, output_unit):
    """The output default is always a unit the schema lists for output."""
    data = request()
    data["units"] = {"length": request_unit}
    result = pack_from_dict(data)
    assert result["containers"][0]["inner_dimensions"]["length"]["unit"] == output_unit


@pytest.mark.parametrize("nesting, detail", [
    ("100", "must be less than the item's height"),
    ({"value": "11", "unit": "cm"}, "must be less than the item's height"),
])
def test_a_nesting_height_not_below_the_height_is_refused_at_its_pointer(nesting, detail):
    error = refusal(lambda d: d["items"][0].update(nesting_height=nesting))
    assert (error.reason, error.field, error.detail) == ("invalid_value", "/items/0/nesting_height", detail)
