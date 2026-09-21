"""The operational artifact and its exports.

`docs/OPERATIONAL-ARTIFACTS.md` is the contract. The assertions follow what the document
promises the artifact will not do: re-derive the plan or its order, address a placement by
`item_id`, let wall-clock time in, claim an exact replay it cannot give, or re-render a value
the result already rendered.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from packvium.artifact_exports import CSV_COLUMNS, export_csv, export_json, export_work_order_html
from packvium.artifacts import (
    FORMAT,
    SUITE_VERSION,
    OperationalArtifactError,
    build_operational_artifact,
    canonical_artifact_json,
)
from packvium.execution import build_execution_plan, canonical_plan_json

TICKS_PER_MM = 16000


def _scalar(ticks: int, unit: str = "mm") -> dict:
    per_unit = TICKS_PER_MM if unit == "mm" else 8000
    return {"ticks": ticks, "unit": unit, "value": str(ticks // per_unit)}


def _placement(item_type: str, x_mm: int, length_mm: int = 10, orientation: str = "LWH") -> dict:
    return {
        "item_id": f"{item_type}#{x_mm}",
        "item_type": item_type,
        "orientation": orientation,
        "position": {axis: _scalar((x_mm if axis == "x" else 0) * TICKS_PER_MM) for axis in ("x", "y", "z")},
        "dimensions": {axis: _scalar((length_mm if axis == "length" else 10) * TICKS_PER_MM)
                       for axis in ("length", "width", "height")},
        "support_ratio": "1.000000",
        "top_load": _scalar(0, "g"),
    }


def _result(*, algorithm: object = "default", placements=None, unpacked=None) -> dict:
    result = {
        "status": "feasible",
        "objective": "default",
        "score": [1, 0, 250],
        "feasibility": {"code": "all_items_packed"},
        "optimality": None,
        "containers": [{
            "id": "crate#1",
            "container_type": "crate",
            "inner_dimensions": {axis: _scalar(100 * TICKS_PER_MM) for axis in ("length", "width", "height")},
            "payload_weight": _scalar(8000, "g"),
            "gross_weight": _scalar(16000, "g"),
            "volume_utilization": "0.002000",
            "placements": placements if placements is not None else [_placement("box", 0), _placement("tin", 10, 5)],
        }],
        "unpacked_items": unpacked or [],
        "catalog_versions_used": [{"catalog_id": "cartons", "version": 3, "effective_at": 10, "resolved_at": 11}],
    }
    if algorithm == "default":
        result["algorithm"] = {"profile": "balanced", "solver": "extreme_point", "seed": 7, "duration_ms": 41,
                               "time_limit_reached": False, "effort_limit_reached": False}
    elif algorithm is not None:
        result["algorithm"] = algorithm
    return result


REQUEST = {"items": [{"id": "box", "dimensions": {"length": 10, "width": 10, "height": 10},
                      "minimum_support_ratio": 0.25}],
           "containers": [{"id": "crate", "inner_dimensions": {"length": 100, "width": 100, "height": 100}}]}


def _code(callable_, *arguments, **options) -> str:
    with pytest.raises(OperationalArtifactError) as refused:
        callable_(*arguments, **options)
    return refused.value.code


# --------------------------------------------------------------------------- the document


def test_the_plan_inside_is_exactly_the_plan_builder_output():
    orders = {0: [1, 0]}
    artifact = build_operational_artifact(REQUEST, _result(), loading_orders=orders)
    assert artifact["format"] == FORMAT
    assert artifact["suite_version"] == SUITE_VERSION
    assert canonical_plan_json(artifact["plan"]) == canonical_plan_json(
        build_execution_plan(REQUEST, _result(), loading_orders=orders))


def test_the_request_is_embedded_as_given_with_its_fractional_numbers():
    artifact = build_operational_artifact(REQUEST, _result())
    assert artifact["provenance"]["request"] == REQUEST
    assert '"minimum_support_ratio":0.25' in canonical_artifact_json(artifact)


def test_geometry_is_in_result_order_in_tick_strings_addressed_by_placement_reference():
    container = build_operational_artifact(REQUEST, _result())["geometry"]["containers"][0]
    assert container["inner_dimensions"] == {"length": "1600000", "width": "1600000", "height": "1600000"}
    assert [entry["placement"]["item_type"] for entry in container["placements"]] == ["box", "tin"]
    assert container["placements"][1]["dimensions"]["length"] == "80000"
    assert "item_id" not in json.dumps(container)


def test_work_order_lines_follow_the_plan_steps_and_copy_rendered_values():
    artifact = build_operational_artifact(REQUEST, _result(), loading_orders={0: [1, 0]})
    work_order = artifact["work_order"]
    container = work_order["containers"][0]
    assert (work_order["length_unit"], work_order["weight_unit"]) == ("mm", "g")
    assert (container["payload_weight"], container["gross_weight"]) == ("1", "2")
    assert [(line["sequence"], line["placement"]["item_type"]) for line in container["lines"]] == [(1, "tin"), (2, "box")]
    assert container["lines"][0]["position"] == {"x": "10", "y": "0", "z": "0"}
    assert [step["placement"] for step in artifact["plan"]["containers"][0]["steps"]] == [
        line["placement"] for line in container["lines"]]


def test_without_a_loading_order_no_line_is_numbered():
    lines = build_operational_artifact(REQUEST, _result())["work_order"]["containers"][0]["lines"]
    assert all("sequence" not in line for line in lines)


def test_a_result_without_containers_has_no_units_and_no_lines():
    result = _result()
    result["containers"] = []
    work_order = build_operational_artifact(REQUEST, result)["work_order"]
    assert work_order == {"length_unit": None, "weight_unit": None, "containers": []}


# ------------------------------------------------------------------------------ provenance


def test_the_solver_is_the_deterministic_part_of_the_algorithm_and_wall_clock_never_enters():
    artifact = build_operational_artifact(REQUEST, _result())
    assert artifact["provenance"]["solver"] == {"profile": "balanced", "solver": "extreme_point", "seed": 7,
                                                "time_limit_reached": False, "effort_limit_reached": False}
    assert artifact["provenance"]["replay"] == {"level": "exact", "because": None}
    assert artifact["provenance"]["catalog_versions_used"][0]["catalog_id"] == "cartons"
    assert "duration_ms" not in canonical_artifact_json(artifact)


def test_a_search_stopped_by_the_clock_is_not_promised_an_exact_replay():
    algorithm = dict(_result()["algorithm"], time_limit_reached=True)
    replay = build_operational_artifact(REQUEST, _result(algorithm=algorithm))["provenance"]["replay"]
    assert replay == {"level": "not_guaranteed", "because": "provenance.solver.time_limit_reached"}


def test_a_result_that_does_not_say_how_it_was_solved_is_not_promised_one_either():
    provenance = build_operational_artifact(REQUEST, _result(algorithm=None))["provenance"]
    assert provenance["solver"] is None
    assert provenance["replay"] == {"level": "not_guaranteed", "because": "provenance.solver"}


# --------------------------------------------------------------------------------- refusals


def test_a_request_that_is_not_an_object_is_refused():
    assert _code(build_operational_artifact, [], _result()) == "invalid_request"


def test_a_loading_order_that_is_not_a_permutation_is_refused_by_the_plan():
    assert _code(build_operational_artifact, REQUEST, _result(), loading_orders={0: [0, 0]}) == "invalid_plan_input"


def test_values_in_two_units_are_refused_rather_than_printed_as_one():
    result = _result()
    result["containers"][0]["placements"][0]["position"]["x"]["unit"] = "cm"
    assert _code(build_operational_artifact, REQUEST, result) == "mixed_units"


def test_two_placements_with_one_reference_are_refused():
    result = _result(placements=[_placement("box", 0), _placement("box", 0)])
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


def test_a_container_without_inner_dimensions_is_refused():
    result = _result()
    del result["containers"][0]["inner_dimensions"]
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


@pytest.mark.parametrize("algorithm", [{"profile": "fast"}, dict(_result()["algorithm"], time_limit_reached="no"), "fast"])
def test_an_algorithm_record_that_is_incomplete_or_mistyped_is_refused(algorithm):
    assert _code(build_operational_artifact, REQUEST, _result(algorithm=algorithm)) == "invalid_result"


def test_a_request_number_javascript_cannot_hold_is_refused():
    request = dict(REQUEST, metadata={"order": 2**53})
    assert _code(build_operational_artifact, request, _result()) == "number_out_of_range"


def test_a_container_that_is_not_an_object_is_refused_by_name_before_anything_reads_it():
    result = _result()
    result["containers"] = ["crate"]
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


def test_geometry_ticks_javascript_cannot_hold_are_refused():
    """Geometry writes ticks as strings, so the canonical writer never sees them as numbers."""
    result = _result()
    result["containers"][0]["placements"][0]["dimensions"]["length"]["ticks"] = 2**53
    assert _code(build_operational_artifact, REQUEST, result) == "number_out_of_range"


def test_a_catalog_reference_of_the_wrong_type_is_refused():
    result = _result()
    result["catalog_versions_used"][0]["version"] = 1.5
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


def test_a_result_that_is_not_an_object_is_refused():
    assert _code(build_operational_artifact, REQUEST, []) == "invalid_result"


def test_a_catalog_id_that_is_not_a_string_is_refused():
    result = _result()
    result["catalog_versions_used"][0]["catalog_id"] = 7
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


@pytest.mark.parametrize("ticks", [1.5, True, "160000"])
def test_geometry_ticks_that_are_not_an_integer_are_refused(ticks):
    """A boolean is an `int` to Python; JSON `true` is not a tick count in any engine."""
    result = _result()
    result["containers"][0]["placements"][0]["dimensions"]["length"]["ticks"] = ticks
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


def test_a_rendered_value_that_is_not_a_string_is_refused():
    result = _result()
    result["containers"][0]["placements"][0]["position"]["x"]["value"] = 0
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


def test_a_placement_that_is_not_an_object_is_refused_before_the_plan_reads_it():
    result = _result(placements=[_placement("box", 0), "tin"])
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


@pytest.mark.parametrize("field", ["containers", "unpacked_items", "catalog_versions_used"])
def test_a_list_field_that_is_not_a_list_is_refused(field):
    result = _result()
    result[field] = {"not": "a list"}
    assert _code(build_operational_artifact, REQUEST, result) == "invalid_result"


def test_the_builder_imports_no_solver_validator_renderer_or_clock():
    allowed = {"__future__", "typing", "decimal", "math", "json", "_canonical_json", "execution", "artifacts"}
    for name in ("artifacts.py", "artifact_exports.py", "_canonical_json.py"):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "src" / "packvium" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[-1] in allowed, f"{name} imports {node.module}"
            elif isinstance(node, ast.Import):
                assert all(alias.name in allowed for alias in node.names), f"{name} imports {node.names[0].name}"


# ----------------------------------------------------------------------------------- exports


def _artifact(**options) -> dict:
    return build_operational_artifact(REQUEST, _result(**options), loading_orders={0: [1, 0]})


def test_the_json_export_is_the_canonical_artifact():
    artifact = _artifact()
    assert export_json(artifact) == canonical_artifact_json(artifact)
    assert json.loads(export_json(artifact)) == json.loads(json.dumps(artifact))


@pytest.mark.parametrize("export", [export_json, export_csv, export_work_order_html])
def test_every_export_refuses_a_document_it_cannot_read(export):
    assert _code(export, dict(_artifact(), format="packvium-operational-artifact/v2")) == "unknown_format"
    assert _code(export, ["not", "an", "artifact"]) == "unknown_format"


@pytest.mark.parametrize("export", [export_csv, export_work_order_html])
def test_a_value_with_no_single_rendering_is_refused_rather_than_printed_four_ways(export):
    """Python would print `True`, PHP `1`, JavaScript `true`: none of them is the answer."""
    result = _result()
    result["containers"][0]["container_type"] = True
    assert _code(export, build_operational_artifact(REQUEST, result)) == "invalid_value"


def test_an_export_depends_on_the_canonical_bytes_not_on_how_python_holds_them():
    """A `1.0` seed in memory is the `1` every other engine reads back from the same bytes."""
    in_memory = _artifact()
    in_memory["provenance"]["solver"]["seed"] = 7.0
    read_back = json.loads(canonical_artifact_json(in_memory))
    for export in (export_json, export_csv, export_work_order_html):
        assert export(in_memory) == export(read_back)


@pytest.mark.parametrize("index", [0.5, True])
def test_a_hand_edited_container_index_is_refused_rather_than_numbered(index):
    artifact = _artifact()
    artifact["work_order"]["containers"][0]["container_index"] = index
    assert _code(export_work_order_html, artifact) == "invalid_value"


def test_the_csv_has_a_header_one_row_per_step_in_order_and_one_per_unplaced_item():
    unpacked = [{"item_id": "jack#1", "item_type": "jack", "reason": "no_compatible_container_dimensions",
                 "details": [], "proof": {"level": "proven"}}]
    rows = export_csv(_artifact(unpacked=unpacked)).split("\r\n")
    assert rows[0] == ",".join(CSV_COLUMNS)
    assert rows[1] == "step,0,crate,1,tin,LWH,160000,0,0,10,0,0,5,10,10,mm,,"
    assert rows[2] == "step,0,crate,2,box,LWH,0,0,0,0,0,0,10,10,10,mm,,"
    assert rows[3] == "unplaced,,,,jack,,,,,,,,,,,,no_compatible_container_dimensions,proven"
    assert rows[4] == "" and len(rows) == 5


def test_a_csv_field_is_quoted_only_when_it_must_be_and_never_rewritten():
    result = _result(placements=[_placement('a,"b"', 0), _placement("=SUM(A1)", 10), _placement("two\nlines", 20)])
    text = export_csv(build_operational_artifact(REQUEST, result))
    assert ',"a,""b""",' in text
    assert ",=SUM(A1)," in text
    assert ',"two\nlines",' in text


def test_the_work_order_is_self_contained_escaped_and_lists_every_step():
    result = _result(placements=[_placement("<b>&'\"", 0), _placement("tin", 10, 5)])
    html = export_work_order_html(build_operational_artifact(REQUEST, result, loading_orders={0: [1, 0]}))
    assert html.startswith("<!DOCTYPE html>\n") and html.endswith("</html>\n")
    assert "<script" not in html and "http" not in html and " src=" not in html
    assert "&lt;b&gt;&amp;&#39;&quot;" in html and "<b>&'" not in html
    assert html.count("&#9744;") == 2
    assert "Order: loading." in html and "extreme_point" in html and "cartons v3" in html
    assert html.isascii()


def test_the_work_order_says_when_there_is_no_order_and_when_everything_was_packed():
    html = export_work_order_html(build_operational_artifact(REQUEST, _result()))
    assert "Order: unavailable." in html
    assert "<p>Every item was packed.</p>" in html


def test_exports_are_deterministic_and_leave_the_artifact_untouched():
    artifact = _artifact()
    before = canonical_artifact_json(artifact)
    for export in (export_json, export_csv, export_work_order_html):
        assert export(artifact) == export(artifact)
    assert canonical_artifact_json(artifact) == before


# ------------------------------------------------------------------------------ the corpus

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "conformance" / "schema"


@pytest.mark.skipif(not (ROOT / "conformance" / "golden").is_dir(), reason="the corpus lives in the workspace only")
def test_every_golden_result_with_its_request_builds_a_schema_valid_artifact():
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    schemas = {name: json.loads((SCHEMAS / name).read_text())
               for name in ("operational-artifact.schema.json", "execution-plan.schema.json")}
    registry = referencing.Registry().with_resources(
        (schema["$id"], referencing.Resource.from_contents(schema)) for schema in schemas.values())
    validator = jsonschema.Draft202012Validator(schemas["operational-artifact.schema.json"], registry=registry)
    built = 0
    for golden in sorted((ROOT / "conformance" / "golden").glob("*.json")):
        result = json.loads(golden.read_text())
        if not isinstance(result, dict) or "status" not in result:
            continue
        request = json.loads((ROOT / "conformance" / "fixtures" / golden.name).read_text())
        orders = {index: list(reversed(range(len(container.get("placements") or []))))
                  for index, container in enumerate(result.get("containers") or [])}
        artifact = build_operational_artifact(request, result, loading_orders=orders)
        errors = list(validator.iter_errors(json.loads(canonical_artifact_json(artifact))))
        assert errors == [], f"{golden.name}: {errors[0].message if errors else ''}"
        export_csv(artifact)
        export_work_order_html(artifact)
        built += 1
    assert built >= 398
