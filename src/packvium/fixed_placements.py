"""Admission of a request's fixed placements (docs/PLAN-REVISIONS.md).

A fixed placement is an item already in a known place: loaded, or locked there by an
operator. It enters the solve as a real `Placement` -- weight, support, top load and all --
seeded into the one container instance it names, so every rule the engine already enforces
holds for it with no rule of its own.

What this module adds is the refusal. A fixed set that is not a valid packing on its own is
refused before any search, with `invalid_fixed_placement`, rather than exempted: an exempt
item would need a second validator, and four engines would have to agree on which rules it
skips. The check is the ordinary `IndependentSolutionValidator` run over the fixed set alone,
with item accounting left out because the free items have not been placed yet.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, List, Mapping

from ._canonical_json import json_integer, json_spelling
from ._compat import dataclass
from .request_errors import InvalidRequestError, pointer
from .constraints import direct_support_view, load_units, top_loads
from .geometry import AxisAlignedBox, Point
from .models import (Container, FixedPlacement, Item, ItemInstance, PackedContainer,
                     PackingRequest, Placement)
from .units import Length, Weight
from .validation import IndependentSolutionValidator


class FixedPlacementError(InvalidRequestError):
    """The request's fixed placements are malformed (`reason` `malformed`, `field` the bad value)
    or cannot all hold (`cannot_hold`, `field` `/fixed_placements`), so no search is attempted."""

    code = "invalid_fixed_placement"

    def __init__(self, detail: str, reason: str = "cannot_hold", field: str = "/fixed_placements"):
        super().__init__(reason, field, detail)

    def _message(self) -> str:
        return f"{self.code}: {self.detail}"


ORIENTATIONS = ("LWH", "LHW", "WLH", "WHL", "HLW", "HWL")
_REQUIRED = ("item_type", "container_type", "orientation")
_FIELDS = (*_REQUIRED, "container_instance", "position")
_AXES = ("x", "y", "z")


def require_fixed_placement_shapes(raw: Any, unit: str = "mm") -> List[Mapping[str, Any]]:
    """The request's `fixed_placements` as JSON gave them, or a refusal naming the first entry
    that is not the schema's shape. Nothing is coerced: `"1"` and `true` are not instances, and a
    list is not a position. An absent or null field is no fixed placements."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _malformed("fixed_placements is a list", "/fixed_placements")
    for index, entry in enumerate(raw):
        _require_entry_shape(entry, f"fixed_placements[{index}]", pointer("fixed_placements", index), unit)
    return raw


def _malformed(detail: str, field: str) -> FixedPlacementError:
    return FixedPlacementError(detail, "malformed", field)


def _require_entry_shape(entry: Any, where: str, field: str, unit: str) -> None:
    if not isinstance(entry, Mapping):
        raise _malformed(f"{where} is an object", field)
    unknown = sorted(set(entry) - set(_FIELDS))
    if unknown:
        raise _malformed(f"{where} cannot carry {json_spelling(unknown)}", field)
    missing = [name for name in _REQUIRED if name not in entry]
    if missing:
        raise _malformed(f"{where} needs {json_spelling(missing)}", field)
    for name in ("item_type", "container_type"):
        if not isinstance(entry[name], str) or not entry[name]:
            raise _malformed(f"{where}.{name} is a non-empty string", f"{field}/{name}")
    if entry["orientation"] not in ORIENTATIONS:
        raise _malformed(f"{where}.orientation is one of the six codes", f"{field}/orientation")
    instance = json_integer(entry.get("container_instance", 1))
    if instance is None or instance < 1:
        raise _malformed(f"{where}.container_instance counts from 1", f"{field}/container_instance")
    position = entry.get("position", {})
    require_point_shape(position, f"{where}.position", f"{field}/position", _malformed)
    for axis in _AXES:
        if axis in position:
            _require_coordinate(position[axis], f"{where}.position.{axis}", f"{field}/position/{axis}", unit)


def _require_coordinate(value: Any, where: str, field: str, unit: str) -> None:
    try:
        Length.parse(value, unit)
    except (ValueError, TypeError, ArithmeticError, KeyError, AttributeError) as error:
        if "cannot be negative" in str(error):
            raise _malformed(f"{where} cannot be negative", field) from error
        raise _malformed(f"{where} is a measure", field) from error


def require_point_shape(point: Any, where: str, field: str,
                        error: Callable[[str, str], Exception]) -> None:
    """A point is an object of `x`, `y` and `z` measures; units are the length parser's.
    `error` builds the refusal from its message and the pointer of the bad value."""
    if not isinstance(point, Mapping):
        raise error(f"{where} is a point object", field)
    unknown = sorted(set(point) - set(_AXES))
    if unknown:
        raise error(f"{where} cannot carry {json_spelling(unknown)}", field)
    for axis in _AXES:
        if axis in point and (point[axis] is None or isinstance(point[axis], (bool, list))):
            raise error(f"{where}.{axis} is a measure", f"{field}/{axis}")


@dataclass(frozen=True, slots=True)
class FixedLoad:
    """What the solve starts from: the fixed containers, and the items still to place.

    `containers` are in opening order -- request container order, then instance -- each
    holding only its fixed placements. `free` is every instance a fixed placement did not
    take, in request order.
    """
    containers: tuple[PackedContainer, ...]
    free: tuple[ItemInstance, ...]


def admit_fixed_placements(
    request: PackingRequest,
    minimum_support_ratio: float,
    clearance: Length,
    max_containers: int | None,
) -> FixedLoad:
    """Resolve `request.fixed_placements`, or refuse the request.

    O(f log f) to resolve f entries, plus one validator pass over them, which is O(f^2) in
    the worst case of its pairwise sweep. It runs once, before search.
    """
    if not request.fixed_placements:
        return FixedLoad((), request.instances)
    items = {item.id: item for item in request.items}
    containers = {container.id: container for container in request.containers}
    for entry in request.fixed_placements:
        _require_known(entry, items, containers)
    instances = _assign_instances(request.fixed_placements, items)
    grouped = _group_by_container(request, instances, clearance)
    _require_contiguous_instances(grouped, containers, max_containers)
    packed = tuple(
        PackedContainer(containers[container_id], sequence, _with_support_and_loads(placements))
        for (container_id, sequence), placements in grouped
    )
    _require_valid_alone(request, packed, minimum_support_ratio, clearance)
    taken = {placement.instance.id for container in packed for placement in container.placements}
    free = tuple(instance for instance in request.instances if instance.id not in taken)
    return FixedLoad(packed, free)


def _require_known(entry: FixedPlacement, items: dict[str, Item],
                   containers: dict[str, Container]) -> None:
    item = items.get(entry.item_id)
    if item is None:
        raise FixedPlacementError(f"unknown item type {json_spelling(entry.item_id)}")
    if entry.container_id not in containers:
        raise FixedPlacementError(f"unknown container type {json_spelling(entry.container_id)}")
    if entry.rotation not in item.allowed_rotations:
        raise FixedPlacementError(
            f"{entry.item_id} may not be placed in orientation {entry.rotation.value}"
        )


def _assign_instances(entries: tuple[FixedPlacement, ...],
                      items: dict[str, Item]) -> tuple[ItemInstance, ...]:
    """Fixed items take the first instances of their type, in the order they are listed."""
    taken: dict[str, int] = defaultdict(int)
    instances = []
    for entry in entries:
        item = items[entry.item_id]
        taken[item.id] += 1
        if taken[item.id] > item.quantity:
            raise FixedPlacementError(
                f"{taken[item.id]} {item.id} fixed, {item.quantity} requested"
            )
        instances.append(ItemInstance(item, taken[item.id]))
    return tuple(instances)


def _group_by_container(
    request: PackingRequest,
    instances: tuple[ItemInstance, ...],
    clearance: Length,
) -> list[tuple[tuple[str, int], list[Placement]]]:
    grouped: dict[tuple[str, int], list[Placement]] = defaultdict(list)
    for entry, instance in zip(request.fixed_placements, instances):
        grouped[(entry.container_id, entry.container_instance)].append(
            _placement(entry, instance, clearance)
        )
    order = {container.id: index for index, container in enumerate(request.containers)}
    return sorted(grouped.items(), key=lambda pair: (order[pair[0][0]], pair[0][1]))


def _placement(entry: FixedPlacement, instance: ItemInstance, clearance: Length) -> Placement:
    margin = clearance.ticks
    position = entry.position
    # The clearance envelope must be inside the container, as it must for any placement; an
    # item flush against a wall has an envelope that starts before it.
    if min(position.x, position.y, position.z) < margin:
        raise FixedPlacementError(f"outside_container: {instance.id}")
    dimensions = instance.dimensions.rotated(entry.rotation)
    envelope_origin = Point(position.x - margin, position.y - margin, position.z - margin)
    envelope = dimensions.expand(clearance) if margin else dimensions
    return Placement(instance, position, entry.rotation, dimensions, envelope_origin,
                     envelope, fixed=True)


def _require_contiguous_instances(
    grouped: list[tuple[tuple[str, int], list[Placement]]],
    containers: dict[str, Container],
    max_containers: int | None,
) -> None:
    named: dict[str, list[int]] = defaultdict(list)
    for (container_id, sequence), _ in grouped:
        named[container_id].append(sequence)
    for container_id, sequences in named.items():
        if sequences != list(range(1, len(sequences) + 1)):
            raise FixedPlacementError(
                f"{container_id} instances {json_spelling(sequences)} are not numbered 1..{len(sequences)}"
            )
        quantity = containers[container_id].quantity
        if quantity is not None and len(sequences) > quantity:
            raise FixedPlacementError(
                f"{len(sequences)} {container_id} named, {quantity} available"
            )
    if max_containers is not None and len(grouped) > max_containers:
        raise FixedPlacementError(
            f"{len(grouped)} containers hold fixed items, max_containers is {max_containers}"
        )


def _with_support_and_loads(placements: list[Placement]) -> tuple[Placement, ...]:
    """Each fixed item's support ratio from the fixed items under it, and its top load.

    Support is measured against every other fixed item in the container, not only the ones
    listed before it: the set is one arrangement, and listing order is not physics.
    """
    supported = []
    for index, placement in enumerate(placements):
        others = placements[:index] + placements[index + 1:]
        supported.append(Placement(
            placement.instance, placement.position, placement.rotation, placement.dimensions,
            placement.envelope_origin, placement.envelope_dimensions,
            _support_ratio(others, placement), fixed=True,
        ))
    loads = top_loads(load_units(supported))
    return tuple(
        Placement(p.instance, p.position, p.rotation, p.dimensions, p.envelope_origin,
                  p.envelope_dimensions, p.support_ratio, Weight(load), fixed=True)
        for p, load in zip(supported, loads)
    )


def _support_ratio(others: list[Placement], placement: Placement) -> float:
    if placement.envelope_origin.z == 0:
        return 1.0
    box = AxisAlignedBox(placement.envelope_origin, placement.envelope_dimensions)
    support = direct_support_view(others, placement.instance, box)
    return support.supporting_area / placement.envelope_dimensions.base_area


def _require_valid_alone(
    request: PackingRequest,
    packed: tuple[PackedContainer, ...],
    minimum_support_ratio: float,
    clearance: Length,
) -> None:
    report = IndependentSolutionValidator().validate(
        request, packed, minimum_support_ratio, clearance, None
    )
    if not report.valid:
        issue = report.issues[0]
        raise FixedPlacementError(f"{issue.code}: {issue.detail}")
