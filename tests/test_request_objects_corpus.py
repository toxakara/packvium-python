"""Every malformed request of the request-errors corpus is refused on the object path
with the refusal the JSON path gives.

Each case of `conformance/request_errors/corpus.json` (written from the harness by
`scripts/generate_request_error_corpus.py`) is built as objects with the deserializer's own
builders, but without the JSON rules, and passed to `Packer.pack`. A refusal while an object is
built must carry the JSON path's exception type, code and reason with an empty field; a refusal
from `Packer.pack` must carry its field and message too (docs/ERRORS.md, "A request built from
objects"). A case that has no object form, or whose difference is kept on purpose, is listed
below with why; a listed case that starts matching fails, so it has to leave the list.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

from packvium import InvalidRequestError, Packer, pack_from_dict
from packvium.geometry import Point, Rotation
from packvium.models import FixedPlacement
from packvium.serialization import (UnsupportedFeatureError, _config, _container, _item,
                                    _point_ticks, _without_null_optionals)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "conformance" / "request_errors" / "corpus.json"
GENERATOR = ROOT / "scripts" / "generate_request_error_corpus.py"

pytestmark = pytest.mark.skipif(not CORPUS.is_file(), reason="the corpus lives in the workspace only")

_UNKNOWN_KEY = "an object has no place for a key its class does not declare"
_COERCED = "the deserializer coerces it, so the objects it builds are a different, valid request"
_CONSTRUCTOR_FIRST = ("two faults: a constructor refuses its own value as the object is built, "
                      "before Packer.pack walks the request in the JSON order")
_NEGATIVE_POSITION = ("a Point cannot hold a negative coordinate; the measure it is read with "
                      "refuses it as negative_measure, where the JSON path names the placement")

_AS_LETTERS = "the deserializer reads text as the sequence of its letters, so the objects hold other values"
_OUTPUT = "the output units are arguments of PackingResult.to_dict, not part of the objects packed"
_UNITS = "the request's length unit is an argument of each measure, not an object packed"
_UNIT_KEY = ("the deserializer's parse cache is keyed by the unit, which an object or a list cannot key; "
             "Length.of refuses one as invalid_unit (test_units.py)")

#: Cases with no object form: the value cannot be held, or the deserializer changes it.
NOT_OBJECTS: dict[str, str] = {
    "request-not-object": "a request that is not an object has no objects",
    "items-missing": "Packer.pack takes the items as an argument",
    "items-not-list": "Packer.pack takes the items as a sequence of Item",
    "item-not-object": "an item that is not an object is not an Item",
    "containers-missing": "Packer.pack takes the containers as an argument",
    "configuration-not-object": "a configuration that is not an object is not a PackingConfig",
    "effort-not-object": "an effort budget that is not an object is not an EffortBudget",
    "container-made-to-measure-not-object": "a machine that is not an object is not a MadeToMeasure",
    "container-deformable-boundary-not-object": "the object holds the bag's kind, not a block",
    "container-deformable-boundary-kind-missing": "the object holds the bag's kind, not a block",
    "item-id-missing": "Item requires an id",
    "item-dimensions-missing": "Item requires dimensions",
    "item-side-missing": "Dimensions requires all three sides",
    "container-made-to-measure-minimum-missing": "MadeToMeasure requires a minimum",
    "container-made-to-measure-increment-missing": "MadeToMeasure requires an increment",
    "rate-brackets-not-list": "RateTable holds its brackets as a sequence",
    "solver-profile-unknown": "SolverProfile has no such member",
    "configuration-key-unknown-before-bad-value": "SolverProfile has no such member, and " + _UNKNOWN_KEY,
    "configuration-key-unknown": _UNKNOWN_KEY,
    "configuration-key-unknown-null": _UNKNOWN_KEY,
    "configuration-key-unknown-escaped": _UNKNOWN_KEY,
    "configuration-keys-unknown-first-in-code-point-order": _UNKNOWN_KEY,
    "effort-key-unknown": _UNKNOWN_KEY,
    "container-deformable-boundary-key-unknown": _UNKNOWN_KEY,
    "container-made-to-measure-key-unknown": _UNKNOWN_KEY,
    "unknown-configuration-key-before-its-values": _UNKNOWN_KEY,
    "unsupported-before-units": "an unsupported field has no object form",
    "unsupported-before-an-item": "an unsupported field has no object form",
    "container-flat-bag-with-access-directions": "an empty tuple is how an object leaves the field out",
    "item-id-number": _COERCED,
    "item-quantity-text": _COERCED,
    "item-accepts-stacking-groups-not-list": _COERCED,
    "container-access-directions-not-list": _COERCED,
    "fixed-position-unparsable": "a Point holds integer ticks, not text",
    **{f"fixed_placements.position.{axis}={value}": _NEGATIVE_POSITION
       for axis in ("x", "y", "z") for value in ('"-1"', "-1")},
    # A closed vocabulary the object holds as an enum member cannot hold a stranger.
    "item-allowed-rotations-not-list": _AS_LETTERS,
    "item-allowed-rotation-unknown": "Rotation has no such member",
    "item-allowed-rotation-lowercase": "Rotation has no such member",
    "item-shape-type-unknown": "ShapeType has no such member",
    "fixed-orientation-unknown": "Rotation has no such member",
    "configuration-solvers-not-list": _AS_LETTERS,
    "output-not-object": _OUTPUT,
    "output-length-unit-unknown": _OUTPUT,
    "output-weight-unit-unknown": _OUTPUT,
    "output-before-configuration": _OUTPUT,
    # The request's length unit is an argument of each measure, not an object of its own.
    "units-not-object": _UNITS,
    "units-list": _UNITS,
    "units-length-object": _UNIT_KEY,
    "units-length-list": _UNIT_KEY,
}

#: Cases the object path answers differently on purpose (docs/ERRORS.md).
DIFFERENT_ON_PURPOSE: dict[str, str] = {
    "configuration-objective-unknown": "an unknown objective keeps UnknownObjectiveError",
    "configuration-solver-unknown": "an unknown solver keeps UnknownSolverError",
    "container-access-direction-unknown": "an unknown direction keeps InvalidDirectionError",
    "item-accepts-stacking-groups-entry-empty": "a set's index counts in code-point order",
    "an-item-id-before-its-quantity": _CONSTRUCTOR_FIRST,
    "item-ids-before-containers": _CONSTRUCTOR_FIRST,
    "container-ids-before-fixed-placements": _CONSTRUCTOR_FIRST,
    "configuration-boolean-before-effort-budget": ("two faults: the EffortBudget is built, and refuses "
                                                   "its own value, before the PackingConfig that holds it"),
}

#: Pinned, so the skip set cannot grow without this line changing too.
SKIPPED = 62


@dataclass(frozen=True)
class Refusal:
    error: Optional[Exception]
    stage: str

    def describe(self) -> str:
        if self.error is None:
            return f"{self.stage}: accepted"
        fields = (getattr(self.error, "code", None), getattr(self.error, "reason", None),
                  getattr(self.error, "field", None), str(self.error))
        return f"{self.stage}: {type(self.error).__name__}{fields}"


def corpus() -> list[tuple[str, Any]]:
    # Parametrisation reads the corpus at collection, before `pytestmark` skips the module.
    if not CORPUS.is_file():
        return []
    return [(case["name"], case["request"]) for case in json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]]


def json_refusal(request: Any) -> Exception:
    with pytest.raises((InvalidRequestError, UnsupportedFeatureError)) as caught:
        pack_from_dict(copy.deepcopy(request))
    return caught.value


def fixed_placement(raw: dict[str, Any], unit: str) -> FixedPlacement:
    """`serialization._fixed_placements` without the JSON shape rules in front of it."""
    return FixedPlacement(item_id=raw["item_type"], container_id=raw["container_type"],
                          position=Point(*_point_ticks(raw.get("position", {}), unit)),
                          rotation=Rotation(raw["orientation"]),
                          container_instance=raw.get("container_instance", 1))


def object_refusal(request: Any) -> Refusal:
    try:
        data = _without_null_optionals(copy.deepcopy(request))
        unit = (data.get("units") or {}).get("length", "mm")
        config = _config(data.get("configuration") or {}, unit)
        items = [_item(raw, unit) for raw in data["items"]]
        containers = [_container(raw, unit) for raw in data["containers"]]
        fixed = [fixed_placement(raw, unit) for raw in data.get("fixed_placements") or []]
    except Exception as error:  # what the objects could not be built from is the finding
        return Refusal(error, "constructor")
    try:
        Packer(config).pack(items, containers, fixed)
    except Exception as error:
        return Refusal(error, "pack")
    return Refusal(None, "pack")


def same_refusal(from_json: Exception, from_objects: Refusal) -> bool:
    error = from_objects.error
    if not isinstance(error, InvalidRequestError) or not isinstance(error, type(from_json)):
        return False
    if (error.code, error.reason) != (from_json.code, from_json.reason):
        return False
    if from_objects.stage == "constructor":
        return error.field == ""
    return (error.field, str(error)) == (from_json.field, str(from_json))


def covered() -> list[tuple[str, Any]]:
    return [(name, request) for name, request in corpus() if name not in NOT_OBJECTS | DIFFERENT_ON_PURPOSE]


def skipped() -> list[tuple[str, Any]]:
    return [(name, request) for name, request in corpus() if name in NOT_OBJECTS | DIFFERENT_ON_PURPOSE]


def test_the_corpus_is_the_harness_corpus() -> None:
    spec = importlib.util.spec_from_file_location("generate_request_error_corpus", GENERATOR)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert CORPUS.read_text(encoding="utf-8") == generator.render(generator.build()), (
        "conformance/request_errors/corpus.json has drifted from the harness; run "
        "scripts/generate_request_error_corpus.py --write")


def test_the_skip_set_is_pinned_and_names_only_corpus_cases() -> None:
    names = {name for name, _ in corpus()}
    assert set(NOT_OBJECTS).isdisjoint(DIFFERENT_ON_PURPOSE)
    assert set(NOT_OBJECTS) | set(DIFFERENT_ON_PURPOSE) <= names
    assert len(NOT_OBJECTS) + len(DIFFERENT_ON_PURPOSE) == SKIPPED
    assert len(covered()) == len(names) - SKIPPED


@pytest.mark.parametrize("name, request_", covered(), ids=[name for name, _ in covered()])
def test_the_object_path_refuses_as_the_json_path_does(name: str, request_: Any) -> None:
    from_json, from_objects = json_refusal(request_), object_refusal(request_)
    assert same_refusal(from_json, from_objects), (
        f"json: {Refusal(from_json, 'json').describe()}; objects: {from_objects.describe()}")


@pytest.mark.parametrize("name, request_", skipped(), ids=[name for name, _ in skipped()])
def test_a_skipped_case_still_differs(name: str, request_: Any) -> None:
    """A skip that starts matching has been fixed, and must leave its list."""
    assert not same_refusal(json_refusal(request_), object_refusal(request_))
