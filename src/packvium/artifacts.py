"""The portable operational artifact.

`docs/OPERATIONAL-ARTIFACTS.md` is the contract. An execution plan tells an operator what to
do first; it does not say how big the boxes are, what the load weighs or which request and
solver produced it. The artifact is the one document that can be drawn, printed and traced
offline, and it is built so that it adds nothing a solver decided.

It reads the request, the result and the optional loading orders -- the plan's own inputs --
and calls no solver, validator, renderer or clock. The plan inside it is exactly what
`packvium.execution.build_execution_plan` emits for the same inputs. Placements are found by
the plan's `placement_ref`, never by `item_id`, so there is one address format to drift.
Four builders are held to byte-identical canonical output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._canonical_json import MAX_EXACT_MAGNITUDE, CanonicalJsonError, canonical_json
from .execution import ExecutionPlanError, build_execution_plan, placement_reference

__all__ = [
    "FORMAT",
    "SUITE_VERSION",
    "OperationalArtifactError",
    "build_operational_artifact",
    "canonical_artifact_json",
]

FORMAT = "packvium-operational-artifact/v1"

#: The suite version of this builder, the same string in all four engines of one release.
#: The engine's own name is deliberately not recorded: four correct builders naming themselves
#: would emit four different documents. `make version-set` rewrites this line.
SUITE_VERSION = "1.3.0"

#: The deterministic part of `result.algorithm`. `duration_ms` is wall-clock time and never
#: enters an artifact.
SOLVER_FIELDS = ("profile", "solver", "seed", "time_limit_reached", "effort_limit_reached")

DIMENSION_AXES = ("length", "width", "height")
POSITION_AXES = ("x", "y", "z")

PlacementKey = Tuple[Any, ...]


class OperationalArtifactError(ValueError):
    """The builder was handed something an artifact cannot carry.

    `code` is one of a closed set shared by all four engines: `invalid_request`,
    `invalid_result`, `invalid_plan_input`, `mixed_units`, `number_out_of_range`,
    `invalid_string`, `invalid_value`, `unknown_format`, `invalid_json`.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_operational_artifact(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    loading_orders: Optional[Mapping[int, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Build the artifact for one validated result. O(R + P) for a request of size R and P
    placements, plus key sorting when it is serialized."""
    if not isinstance(request, Mapping):
        raise OperationalArtifactError("invalid_request", "a request is a JSON object")
    if not isinstance(result, Mapping):
        raise OperationalArtifactError("invalid_result", "a result is a JSON object")
    _require_objects(result)
    try:
        plan = build_execution_plan(request, result, loading_orders=loading_orders)
    except ExecutionPlanError as error:
        raise OperationalArtifactError("invalid_plan_input", str(error)) from error

    containers = _list(result.get("containers"), "result.containers")
    length_unit, weight_unit = _units(containers)
    artifact = {
        "format": FORMAT,
        "suite_version": SUITE_VERSION,
        "provenance": _provenance(request, result),
        "plan": plan,
        "geometry": {"containers": [_geometry(index, container) for index, container in enumerate(containers)]},
        "work_order": {
            "length_unit": length_unit,
            "weight_unit": weight_unit,
            "containers": [
                _work_order_container(index, container, plan_container, length_unit, weight_unit)
                for index, (container, plan_container) in enumerate(zip(containers, plan["containers"]))
            ],
        },
    }
    # Refused here rather than when someone serializes it: an artifact that exists must have
    # one spelling in every engine.
    canonical_artifact_json(artifact)
    return artifact


def canonical_artifact_json(artifact: Mapping[str, Any]) -> str:
    """The artifact's RFC 8785 canonical form, the bytes four engines are compared on."""
    try:
        return canonical_json(artifact)
    except CanonicalJsonError as error:
        raise OperationalArtifactError(error.code, str(error)) from error


# ------------------------------------------------------------------------------ provenance


def _provenance(request: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    solver = _solver(result.get("algorithm"))
    catalogs = _list(result.get("catalog_versions_used"), "result.catalog_versions_used")
    for catalog in catalogs:
        _catalog(catalog)
    return {
        # Embedded, not digested: only the request itself lets someone replay the artifact
        # without a lookup.
        "request": request,
        "catalog_versions_used": catalogs,
        "solver": solver,
        "replay": _replay(solver),
    }


def _catalog(catalog: Mapping[str, Any]) -> None:
    """The result schema's closed catalog reference. The work order prints these fields, so a
    mistyped one would print differently in every engine."""
    if not isinstance(_field(catalog, "catalog_id", "catalog_versions_used[]"), str):
        raise OperationalArtifactError("invalid_result", "catalog_versions_used[].catalog_id is not a string")
    for name in ("version", "effective_at", "resolved_at"):
        value = _field(catalog, name, "catalog_versions_used[]")
        if isinstance(value, bool) or not isinstance(value, int):
            raise OperationalArtifactError("invalid_result", f"catalog_versions_used[].{name} is not an integer")


def _solver(algorithm: Any) -> Optional[Dict[str, Any]]:
    if algorithm is None:
        return None
    if not isinstance(algorithm, Mapping):
        raise OperationalArtifactError("invalid_result", "result.algorithm is not an object")
    missing = [field for field in SOLVER_FIELDS if field not in algorithm]
    if missing:
        raise OperationalArtifactError("invalid_result", f"result.algorithm has no {', '.join(missing)}")
    for flag in ("time_limit_reached", "effort_limit_reached"):
        if not isinstance(algorithm[flag], bool):
            raise OperationalArtifactError("invalid_result", f"result.algorithm.{flag} is not a boolean")
    return {field: algorithm[field] for field in SOLVER_FIELDS}


def _replay(solver: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """`exact` only when a replay must reproduce the result. A search stopped by wall-clock
    time cannot be reproduced, and a result that does not say how it was solved cannot be
    promised to; claiming otherwise would be softening a proof level by another name."""
    if solver is None:
        return {"level": "not_guaranteed", "because": "provenance.solver"}
    if solver["time_limit_reached"]:
        return {"level": "not_guaranteed", "because": "provenance.solver.time_limit_reached"}
    return {"level": "exact", "because": None}


# -------------------------------------------------------------------------------- geometry


def _geometry(index: int, container: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "container_index": index,
        "inner_dimensions": _tick_dimensions(_field(container, "inner_dimensions", f"containers[{index}]")),
        "placements": [
            {
                "placement": placement_reference(index, placement),
                "dimensions": _tick_dimensions(_field(placement, "dimensions", f"containers[{index}].placements[]")),
            }
            for placement in _placements(index, container)
        ],
    }


def _tick_dimensions(dimensions: Any) -> Dict[str, str]:
    """Lengths as decimal strings of ticks: the scene contract's spelling, and one no engine
    has to hold as a native number."""
    return {axis: _ticks(_field(dimensions, axis, "dimensions")) for axis in DIMENSION_AXES}


def _ticks(scalar: Any) -> str:
    ticks = _field(scalar, "ticks", "exact scalar")
    if isinstance(ticks, bool) or not isinstance(ticks, int):
        raise OperationalArtifactError("invalid_result", f"ticks {ticks!r} is not an integer")
    # Written as a string, so the canonical writer never sees it as a number; the range is
    # checked here instead, because JavaScript has already rounded such a value while parsing.
    if abs(ticks) > MAX_EXACT_MAGNITUDE:
        raise OperationalArtifactError("number_out_of_range", f"ticks {ticks} is beyond what every engine holds exactly")
    return str(ticks)


# ------------------------------------------------------------------------------ work order


def _units(containers: Sequence[Mapping[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """The display units, read from the result rather than re-derived from request defaults:
    the result already rendered every value in them."""
    if not containers:
        return None, None
    first = containers[0]
    length = _field(_field(_field(first, "inner_dimensions", "containers[0]"), "length", "inner_dimensions"), "unit", "length")
    weight = _field(_field(first, "payload_weight", "containers[0]"), "unit", "payload_weight")
    return length, weight


def _work_order_container(index: int, container: Mapping[str, Any], plan_container: Mapping[str, Any],
                          length_unit: Optional[str], weight_unit: Optional[str]) -> Dict[str, Any]:
    by_reference = _placements_by_reference(index, container)
    lines = []
    # One line per plan step, in plan step order: the order is the plan's, looked up, never
    # derived a second time.
    for step in plan_container["steps"]:
        placement = by_reference[_reference_key(step["placement"])]
        line: Dict[str, Any] = {}
        if "sequence" in step:
            line["sequence"] = step["sequence"]
        position = _field(placement, "position", "placement")
        dimensions = _field(placement, "dimensions", "placement")
        line["placement"] = step["placement"]
        line["position"] = {axis: _value(_field(position, axis, "position"), length_unit) for axis in POSITION_AXES}
        line["dimensions"] = {axis: _value(_field(dimensions, axis, "dimensions"), length_unit) for axis in DIMENSION_AXES}
        lines.append(line)
    return {
        "container_index": index,
        "container_type": container.get("container_type"),
        "payload_weight": _value(_field(container, "payload_weight", f"containers[{index}]"), weight_unit),
        "gross_weight": _value(_field(container, "gross_weight", f"containers[{index}]"), weight_unit),
        "lines": lines,
    }


def _placements_by_reference(index: int, container: Mapping[str, Any]) -> Dict[PlacementKey, Mapping[str, Any]]:
    placements = _placements(index, container)
    found = {_reference_key(placement_reference(index, placement)): placement for placement in placements}
    if len(found) != len(placements):
        raise OperationalArtifactError(
            "invalid_result", f"two placements in container {index} share an origin, type and orientation")
    return found


def _reference_key(reference: Mapping[str, Any]) -> PlacementKey:
    ticks = reference["position_ticks"]
    return (reference["item_type"], reference["orientation"], ticks["x"], ticks["y"], ticks["z"])


def _value(scalar: Any, unit: Optional[str]) -> str:
    """The result's rendered value, copied and never re-rendered."""
    if _field(scalar, "unit", "exact scalar") != unit:
        raise OperationalArtifactError("mixed_units", f"a value in {scalar.get('unit')!r} where the result uses {unit!r}")
    value = _field(scalar, "value", "exact scalar")
    if not isinstance(value, str):
        raise OperationalArtifactError("invalid_result", f"value {value!r} is not a string")
    return value


# --------------------------------------------------------------------------------- helpers


def _require_objects(result: Mapping[str, Any]) -> None:
    """Every list entry the artifact and its plan read is an object, checked before either
    reads one, so a malformed result is refused by name in every engine instead of failing
    wherever each language first touches it."""
    for name in ("containers", "unpacked_items", "alternatives", "catalog_versions_used"):
        for entry in _list(result.get(name), f"result.{name}"):
            if not isinstance(entry, Mapping):
                raise OperationalArtifactError("invalid_result", f"result.{name} holds a non-object entry")
    for index, container in enumerate(_list(result.get("containers"), "result.containers")):
        for placement in _placements(index, container):
            if not isinstance(placement, Mapping):
                raise OperationalArtifactError("invalid_result", f"containers[{index}].placements holds a non-object entry")


def _placements(index: int, container: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return _list(container.get("placements"), f"containers[{index}].placements")


def _list(value: Any, where: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise OperationalArtifactError("invalid_result", f"{where} is not a list")
    return list(value)


def _field(mapping: Any, name: str, where: str) -> Any:
    if not isinstance(mapping, Mapping) or name not in mapping:
        raise OperationalArtifactError("invalid_result", f"{where} has no {name}")
    return mapping[name]
