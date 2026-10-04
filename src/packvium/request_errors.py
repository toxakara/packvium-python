"""Structured request errors: one type, a closed reason, and the JSON Pointer of the bad value.

A caller who sends a malformed request needs to know which value is wrong and why, in a form a
program can branch on. A bare `ValueError("item quantity must be positive")` answers neither
question without parsing prose, and four engines phrase the prose four ways. So every request
error is an `InvalidRequestError` with:

- `code`: `invalid_request` (subclasses such as `FixedPlacementError` name their own);
- `reason`: one of `REASONS`, a closed set shared by all four engines;
- `field`: an RFC 6901 JSON Pointer to the offending value (`/items/0/quantity`), or `""` for the
  request as a whole;
- `detail`: the fixed text for the reason, so the message is identical in every engine.

`check_request` walks the schema's rules over the raw JSON before any model is built, and the
first violation wins, in a fixed order every engine follows: units, configuration, items,
containers. The floors are the ones the request schema declares, and a cross-engine suite
holds all four engines to the same reason, field and message.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Set, Tuple, Type

from ._canonical_json import MAX_EXACT_MAGNITUDE, json_integer, json_spelling
from .units import Length, Weight

REASONS = ("missing_field", "wrong_type", "below_minimum", "above_maximum", "negative_measure",
           "invalid_unit", "duplicate_id", "not_allowed", "invalid_value")

SOLVER_PROFILES = ("fast", "balanced", "quality", "exact_small")
OBJECTIVES = ("default", "lowest_cost", "shipping_cost", "lowest_landed_cost", "open_dimension_height", "maximum_value")
ACCESS_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")


class InvalidRequestError(ValueError):
    """The request is not one any engine may answer. Nothing was solved."""

    code = "invalid_request"

    def __init__(self, reason: str, field: str, detail: str) -> None:
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(self._message())

    def _message(self) -> str:
        return f"{self.code}: {self.field}: {self.detail}" if self.field else f"{self.code}: {self.detail}"


def pointer(*parts: Any) -> str:
    """The RFC 6901 pointer to a value, escaping `~` and `/` in keys."""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


# ------------------------------------------------------------------------------ primitives


def _integer(value: Any, field: str, minimum: int) -> None:
    number = json_integer(value)
    if number is None:
        _refuse_non_integer(value, field, minimum)
    if number < minimum:
        raise InvalidRequestError("below_minimum", field, f"must be at least {minimum}")


def _refuse_non_integer(value: Any, field: str, minimum: int) -> None:
    """A whole number past 2^53 - 1 is out of range, not mistyped: say which way."""
    whole = not isinstance(value, bool) and (
        isinstance(value, int) or (isinstance(value, float) and value.is_integer()))
    if whole and value < minimum:
        raise InvalidRequestError("below_minimum", field, f"must be at least {minimum}")
    if whole:
        raise InvalidRequestError("above_maximum", field, f"must be at most {MAX_EXACT_MAGNITUDE}")
    raise InvalidRequestError("wrong_type", field, "must be an integer")


def _ratio(value: Any, field: str, maximum: Optional[int] = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRequestError("wrong_type", field, "must be a number")
    if value < 0:
        raise InvalidRequestError("below_minimum", field, "must be at least 0")
    if maximum is not None and value > maximum:
        raise InvalidRequestError("above_maximum", field, f"must be at most {maximum}")


def _one_of(value: Any, field: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise InvalidRequestError("not_allowed", field, f"must be one of {json_spelling(list(allowed))}")


def _known_fields(value: Mapping[str, Any], field: str, known: Iterable[str]) -> None:
    """The schema closes this object: a key it does not name is refused, never ignored. The
    first unknown key in code-point order is named, the order every engine can share."""
    unknown = sorted(set(value) - set(known))
    if unknown:
        raise InvalidRequestError("not_allowed", pointer_join(field, unknown[0]), "is not a known field")


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidRequestError("wrong_type", field, "must be an object")
    return value


def _list(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise InvalidRequestError("wrong_type", field, "must be a list")
    return value


def _measure(value: Any, field: str, kind: Type[Length] | Type[Weight], unit: str) -> None:
    """A measure is an integer, a string or `{value, unit}`, and never negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Mapping)) or (
            isinstance(value, float) and json_integer(value) is None):
        raise InvalidRequestError("wrong_type", field, "must be a measure")
    if isinstance(value, Mapping):
        if "value" not in value:
            raise InvalidRequestError("wrong_type", field, "must be a measure")
        stated = value.get("unit", unit)
        if not isinstance(stated, str) or stated.lower().strip() not in kind._MULTIPLIERS:
            raise InvalidRequestError("invalid_unit", field, f"has an unknown unit {json_spelling(stated)}")
    try:
        kind.parse(value, unit)
    except (ValueError, TypeError, ArithmeticError) as error:
        if "cannot be negative" in str(error):
            raise InvalidRequestError("negative_measure", field, "cannot be negative") from error
        raise InvalidRequestError("wrong_type", field, "must be a measure") from error


# ------------------------------------------------------------------------------ the rules


_CONFIGURATION_INTEGERS = (("time_limit_ms", 1), ("alternatives", 1), ("max_containers", 1),
                           ("exact_item_limit", 1), ("multi_start_orders", 1),
                           ("max_candidates_per_item", 1), ("max_candidate_points", 16),
                           ("container_plan_beam_width", 1), ("container_plan_node_limit", 1),
                           ("dimensional_weight_divisor", 1))
_EFFORT_LIMITS = ("max_candidates_evaluated", "max_placement_attempts", "max_search_nodes",
                  "max_restarts")
#: Every key the request schema's `configuration` declares; it sets `additionalProperties: false`.
CONFIGURATION_FIELDS = ("alternatives", "clearance", "container_plan_beam_width",
                        "container_plan_node_limit", "dimensional_weight_divisor",
                        "dimensional_weight_length_unit", "dimensional_weight_weight_unit",
                        "effort_budget", "exact_item_limit", "max_candidate_points",
                        "max_candidates_per_item", "max_containers", "minimum_support_ratio",
                        "multi_start_orders", "objective", "require_placement_coordinates", "seed",
                        "solver_profile", "solvers", "time_limit_ms")
_SIDES = ("length", "width", "height")


def check_request(data: Any) -> None:
    """Refuse the request with its first violation, or return. O(size of the request)."""
    request = _object(data, "")
    unit = _check_units(request.get("units"))
    _check_configuration(request.get("configuration"), unit)
    items = _required_list(request, "items", "")
    for index, raw in enumerate(items):
        _check_item(raw, pointer("items", index), unit)
    _require_unique_ids(items, "items")
    containers = _required_list(request, "containers", "")
    for index, raw in enumerate(containers):
        _check_container(raw, pointer("containers", index), unit)
    _require_unique_ids(containers, "containers")


def _check_units(raw: Any) -> str:
    if raw is None:
        return "mm"
    units = _object(raw, "/units")
    length = units.get("length")
    if length is None:
        return "mm"
    if not isinstance(length, str) or length.lower().strip() not in Length._MULTIPLIERS:
        raise InvalidRequestError("invalid_unit", "/units/length",
                                  f"has an unknown unit {json_spelling(length)}")
    return length


def _check_configuration(raw: Any, unit: str) -> None:
    if raw is None:
        return
    configuration = _object(raw, "/configuration")
    _known_fields(configuration, "/configuration", CONFIGURATION_FIELDS)
    _optional(configuration, "solver_profile", "/configuration",
              lambda v, f: _one_of(v, f, SOLVER_PROFILES))
    _optional(configuration, "objective", "/configuration",
              lambda v, f: _one_of(v, f, OBJECTIVES))
    for name, minimum in _CONFIGURATION_INTEGERS:
        _optional(configuration, name, "/configuration", lambda v, f, m=minimum: _integer(v, f, m))
    _optional(configuration, "minimum_support_ratio", "/configuration", _ratio)
    _optional(configuration, "clearance", "/configuration", lambda v, f: _measure(v, f, Length, unit))
    effort = configuration.get("effort_budget")
    if effort is not None:
        budget = _object(effort, "/configuration/effort_budget")
        _known_fields(budget, "/configuration/effort_budget", _EFFORT_LIMITS)
        for name in _EFFORT_LIMITS:
            _optional(budget, name, "/configuration/effort_budget", lambda v, f: _integer(v, f, 1))


def _check_item(raw: Any, where: str, unit: str) -> None:
    item = _object(raw, where)
    _required_string(item, "id", where)
    _optional(item, "quantity", where, lambda v, f: _integer(v, f, 1))
    _dimensions(_required(item, "dimensions", where), where + "/dimensions", unit)
    _optional(item, "weight", where, lambda v, f: _measure(v, f, Weight, "g"))
    _optional(item, "max_top_load", where, lambda v, f: _measure(v, f, Weight, "g"))
    _optional(item, "nesting_height", where, lambda v, f: _measure(v, f, Length, unit))
    _optional(item, "max_stacked_items", where, lambda v, f: _integer(v, f, 1))
    _optional(item, "stop_index", where, lambda v, f: _integer(v, f, 0))
    _optional(item, "value", where, lambda v, f: _integer(v, f, 0))
    _optional(item, "max_compression_pressure_kpa", where, lambda v, f: _integer(v, f, 0))
    _optional(item, "minimum_support_ratio", where, _ratio)
    _optional(item, "compression_ratio", where, lambda v, f: _ratio(v, f, None))


def _check_container(raw: Any, where: str, unit: str) -> None:
    container = _object(raw, where)
    _required_string(container, "id", where)
    _optional(container, "quantity", where, lambda v, f: _integer(v, f, 1))
    _dimensions(_required(container, "inner_dimensions", where), where + "/inner_dimensions", unit)
    _optional(container, "outer_dimensions", where, lambda v, f: _dimensions(v, f, unit))
    for name in ("tare_weight", "max_payload", "max_stack_density"):
        _optional(container, name, where, lambda v, f: _measure(v, f, Weight, "g"))
    _optional(container, "max_items", where, lambda v, f: _integer(v, f, 1))
    _optional(container, "cost_minor", where, lambda v, f: _integer(v, f, 0))
    _optional(container, "void_fill_reserve_ratio", where, _ratio)
    _optional(container, "access_directions", where, _access_directions)
    _optional(container, "tag_limits", where, _tag_limits)
    _optional(container, "rate_table", where, _rate_table)
    _optional(container, "obstacles", where, lambda v, f: _obstacles(v, f, unit))


def _access_directions(raw: Any, where: str) -> None:
    values = _list(raw, where)
    for index, value in enumerate(values):
        _one_of(value, pointer_join(where, index), ACCESS_DIRECTIONS)


def _dimensions(raw: Any, where: str, unit: str) -> None:
    sides = _object(raw, where)
    for side in _SIDES:
        _measure(_required(sides, side, where), pointer_join(where, side), Length, unit)


def _tag_limits(raw: Any, where: str) -> None:
    # Code-point order, as every engine walks them: a JavaScript object puts integer-like keys
    # first, so insertion order would name a different bad tag there.
    limits = _object(raw, where)
    for tag in sorted(limits):
        _integer(limits[tag], pointer_join(where, tag), 1)


def _rate_table(raw: Any, where: str) -> None:
    table = _object(raw, where)
    for name, minimum in (("weight_brackets_g", 1), ("prices_minor", 0)):
        values = table.get(name)
        if values is not None:
            for index, value in enumerate(_list(values, pointer_join(where, name))):
                _integer(value, pointer_join(where, name, index), minimum)
    _optional(table, "minimum_charge_minor", where, lambda v, f: _integer(v, f, 0))
    _optional(table, "fuel_surcharge_permille", where, lambda v, f: _integer(v, f, 0))


def _obstacles(raw: Any, where: str, unit: str) -> None:
    for index, entry in enumerate(_list(raw, where)):
        at = pointer_join(where, index)
        obstacle = _object(entry, at)
        origin = obstacle.get("origin")
        if origin is not None:
            point = _object(origin, at + "/origin")
            for axis in ("x", "y", "z"):
                _optional(point, axis, at + "/origin", lambda v, f: _measure(v, f, Length, unit))
        _dimensions(_required(obstacle, "dimensions", at), at + "/dimensions", unit)


# ------------------------------------------------------------------------------ plumbing


def pointer_join(base: str, *parts: Any) -> str:
    return base + pointer(*parts)


def _optional(container: Mapping[str, Any], name: str, where: str,
              check: Callable[[Any, str], None]) -> None:
    """An optional field: absent and null are the default; anything else must pass."""
    value = container.get(name)
    if value is not None:
        check(value, pointer_join(where, name))


def _required(container: Mapping[str, Any], name: str, where: str) -> Any:
    if container.get(name) is None:
        raise InvalidRequestError("missing_field", pointer_join(where, name), "is required")
    return container[name]


def _required_list(container: Mapping[str, Any], name: str, where: str) -> Sequence[Any]:
    return _list(_required(container, name, where), pointer_join(where, name))


def _required_string(container: Mapping[str, Any], name: str, where: str) -> None:
    if not isinstance(_required(container, name, where), str):
        raise InvalidRequestError("wrong_type", pointer_join(where, name), "must be a string")


def _require_unique_ids(entries: Iterable[Mapping[str, Any]], key: str) -> None:
    """Checked once every entry is well formed, so the later of two equal ids is named."""
    seen: Set[str] = set()
    for index, entry in enumerate(entries):
        identifier = entry["id"]
        if identifier in seen:
            raise InvalidRequestError("duplicate_id", pointer(key, index, "id"),
                                      f"repeats the id {json_spelling(identifier)}")
        seen.add(identifier)
