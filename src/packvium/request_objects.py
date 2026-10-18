"""The request schema's rules, applied to a request built from objects.

`check_request` walks a JSON request; `Packer.pack` takes objects, whose own constructors already
refuse most of the same values but cannot know where in a request they will sit. This walks the
objects in the JSON rules' order -- configuration, items, item ids, containers, container ids,
fixed placements -- with the same primitives, so a value the JSON path refuses is refused here with the same reason
and the same RFC 6901 field. Only rules `check_request` also applies are checked, so on the JSON
path, which runs `check_request` first, this never refuses anything.

Unordered sets (accepted stacking groups, tag limits) are walked in code-point order: a set has
no order of its own, and hash order must not decide which value an error names. O(size of the
request).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from ._canonical_json import MAX_EXACT_MAGNITUDE, json_integer, json_spelling
from .config import PackingConfig
from .fixed_placements import FixedPlacementError, orientation_rule
from .flat_bag import DEFORMABLE_BOUNDARY_KINDS
from .geometry import Point, Rotation
from .made_to_measure import MadeToMeasure, as_triple, box_cost_minor_for
from .models import Container, FixedPlacement, Item, RateTable
from .request_errors import (BAG_EXCLUSIONS, InvalidRequestError, _integer, _one_of,
                             _ratio, _set_beside_a_bag, _stacking_group_name, pointer,
                             pointer_join)

#: (JSON name, PackingConfig attribute, floor), in `check_request`'s order.
_CONFIGURATION_INTEGERS = (("time_limit_ms", "time_limit_ms", 1), ("alternatives", "top_k", 1),
                           ("max_containers", "max_containers", 1),
                           ("exact_item_limit", "exact_item_limit", 1),
                           ("multi_start_orders", "multi_start_orders", 1),
                           ("max_candidates_per_item", "max_candidates_per_item", 1),
                           ("max_candidate_points", "max_candidate_points", 16),
                           ("container_plan_beam_width", "container_plan_beam_width", 1),
                           ("container_plan_node_limit", "container_plan_node_limit", 1),
                           ("dimensional_weight_divisor", "dimensional_weight_divisor", 1))
_EFFORT_LIMITS = ("max_candidates_evaluated", "max_placement_attempts", "max_search_nodes",
                  "max_restarts")


def check_objects(items: Sequence[Item], containers: Sequence[Container], config: PackingConfig,
                  fixed_placements: Sequence[FixedPlacement] = ()) -> None:
    """Refuse the request with its first violation, or return."""
    _check_configuration(config)
    for index, item in enumerate(items):
        _check_item(item, pointer("items", index))
    _require_unique_ids(items, "items")
    for index, container in enumerate(containers):
        _check_container(container, pointer("containers", index))
    _require_unique_ids(containers, "containers")
    for index, placement in enumerate(fixed_placements):
        _check_fixed_placement(placement, f"fixed_placements[{index}]", pointer("fixed_placements", index))


def _check_configuration(config: PackingConfig) -> None:
    # `objective` is not checked here: an unknown one keeps its own, older error on this path,
    # `UnknownObjectiveError`, raised when the objective is resolved.
    where = "/configuration"
    for name, attribute, minimum in _CONFIGURATION_INTEGERS:
        _optional(getattr(config, attribute), pointer_join(where, name), minimum)
    _ratio(config.minimum_support_ratio, pointer_join(where, "minimum_support_ratio"))
    if config.effort_budget is not None:
        budget = pointer_join(where, "effort_budget")
        for name in _EFFORT_LIMITS:
            _optional(getattr(config.effort_budget, name), pointer_join(budget, name), 1)


def _check_item(item: Item, where: str) -> None:
    _string_id(item.id, where)
    _integer(item.quantity, pointer_join(where, "quantity"), 1)
    _optional(item.max_stacked_items, pointer_join(where, "max_stacked_items"), 1)
    _optional(item.stop_index, pointer_join(where, "stop_index"), 0)
    _optional(item.value, pointer_join(where, "value"), 0)
    _optional(item.max_compression_pressure_kpa, pointer_join(where, "max_compression_pressure_kpa"), 0)
    _ratio(item.minimum_support_ratio, pointer_join(where, "minimum_support_ratio"))
    if item.stacking_group is not None:
        _stacking_group_name(item.stacking_group, pointer_join(where, "stacking_group"))
    accepted = pointer_join(where, "accepts_stacking_groups")
    for index, name in enumerate(_code_point_order(item.accepts_stacking_groups)):
        _stacking_group_name(name, pointer_join(accepted, index))


def _check_container(container: Container, where: str) -> None:
    _string_id(container.id, where)
    _optional(container.quantity, pointer_join(where, "quantity"), 1)
    _optional(container.max_items, pointer_join(where, "max_items"), 1)
    _integer(container.cost_minor, pointer_join(where, "cost_minor"), 0)
    _ratio(container.void_fill_reserve_ratio, pointer_join(where, "void_fill_reserve_ratio"))
    limits = pointer_join(where, "tag_limits")
    for tag in _code_point_order(container.tag_limits):
        _integer(container.tag_limits[tag], pointer_join(limits, tag), 1)
    if container.rate_table is not None:
        _check_rate_table(container.rate_table, pointer_join(where, "rate_table"))
    if container.deformable_boundary is not None:
        _check_flat_bag(container, where)
    if container.made_to_measure is not None:
        _check_made_to_measure(container, container.made_to_measure, where)


def _check_rate_table(table: RateTable, where: str) -> None:
    for name, minimum in (("weight_brackets_g", 1), ("prices_minor", 0)):
        for index, value in enumerate(getattr(table, name)):
            _integer(value, pointer_join(where, name, index), minimum)
    _integer(table.minimum_charge_minor, pointer_join(where, "minimum_charge_minor"), 0)
    _integer(table.fuel_surcharge_permille, pointer_join(where, "fuel_surcharge_permille"), 0)


def _check_flat_bag(container: Container, where: str) -> None:
    _one_of(container.deformable_boundary, pointer_join(where, "deformable_boundary", "kind"),
            DEFORMABLE_BOUNDARY_KINDS)
    for name in BAG_EXCLUSIONS:
        if _set_beside_a_bag(name, _as_json_value(getattr(container, name))):
            raise InvalidRequestError("not_allowed", pointer_join(where, name),
                                      "cannot be combined with deformable_boundary")


def _check_made_to_measure(container: Container, machine: MadeToMeasure, where: str) -> None:
    field = pointer_join(where, "made_to_measure")
    price = pointer_join(field, "cost_per_square_metre_minor")
    _integer(machine.cost_per_square_metre_minor, price, 0)
    if container.outer_dimensions is not None:
        raise InvalidRequestError("not_allowed", pointer_join(where, "outer_dimensions"),
                                  "cannot be combined with made_to_measure")
    largest = box_cost_minor_for(machine.cost_per_square_metre_minor, container.cost_minor,
                                 as_triple(container.inner_dimensions))
    if largest > MAX_EXACT_MAGNITUDE:
        raise InvalidRequestError("invalid_value", price, f"prices the largest box above {MAX_EXACT_MAGNITUDE}")


def _check_fixed_placement(placement: FixedPlacement, where: str, field: str) -> None:
    """`fixed_placements.require_fixed_placement_shapes`' rules that an object can break, in its
    order and with its refusal; whether the set can hold is `admit_fixed_placements`' to say."""
    for name, value in (("item_type", placement.item_id), ("container_type", placement.container_id)):
        if not isinstance(value, str) or not value:
            raise _malformed_placement(f"{where}.{name} is a non-empty string", pointer_join(field, name))
    if not isinstance(placement.rotation, Rotation):
        raise _malformed_placement(f"{where}.orientation {orientation_rule()}", pointer_join(field, "orientation"))
    instance = json_integer(placement.container_instance)
    if instance is None or instance < 1:
        raise _malformed_placement(f"{where}.container_instance counts from 1",
                                   pointer_join(field, "container_instance"))
    position = pointer_join(field, "position")
    if not isinstance(placement.position, Point):
        raise _malformed_placement(f"{where}.position is a point object", position)
    for axis in ("x", "y", "z"):
        # A point holds ticks, so its coordinate is a measure only as an integer; `Point`
        # itself already refuses a negative one.
        value = getattr(placement.position, axis)
        if isinstance(value, bool) or not isinstance(value, int):
            raise _malformed_placement(f"{where}.position.{axis} is a measure", pointer_join(position, axis))


def _malformed_placement(detail: str, field: str) -> FixedPlacementError:
    return FixedPlacementError(detail, "malformed", field)


def _string_id(identifier: Any, where: str) -> None:
    if not isinstance(identifier, str):
        raise InvalidRequestError("wrong_type", pointer_join(where, "id"), "must be a string")


def _require_unique_ids(entries: Sequence[Item] | Sequence[Container], key: str) -> None:
    """The later of two equal ids is named, as on the JSON path."""
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if entry.id in seen:
            raise InvalidRequestError("duplicate_id", pointer(key, index, "id"),
                                      f"repeats the id {json_spelling(entry.id)}")
        seen.add(entry.id)


def _optional(value: Optional[int], field: str, minimum: int) -> None:
    if value is not None:
        _integer(value, field, minimum)


def _code_point_order(values: Iterable[Any]) -> list[Any]:
    """Strings in code-point order; anything mistyped after them, so it is still checked and
    sorting a mixed set cannot fail."""
    return sorted(values, key=lambda value: (not isinstance(value, str), value if isinstance(value, str) else repr(value)))


def _as_json_value(value: Any) -> Any:
    """An empty tuple is how an object leaves a list field out; JSON leaves it absent."""
    return None if value == () else value
