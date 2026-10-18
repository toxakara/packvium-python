"""`packvium.validate(request, result)`, the reference validator, shipped in the package.

The validator checks the engines, so its worth is that it shares no code with them. These tests
hold the module to importing nothing else from `packvium`, and pin the public entry point.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

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
    assert [issue.code for issue in validate(REQUEST, result)] == ["collision"]


def test_a_result_of_the_wrong_shape_is_refused_rather_than_judged() -> None:
    with pytest.raises(MalformedResult):
        validate(REQUEST, {"containers": "none"})


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
