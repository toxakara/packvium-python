"""Plan revisions: an append-only chain of exceptions against approved plans.

`docs/PLAN-REVISIONS.md` is the contract. A revision records what differed from an approved
plan -- a missing item, a substituted carton, a lock, a verified placement -- and derives the
request the next plan solves. It is a pure function of its parent, the approved artifact and
its events: it calls no solver, validator or clock, so four builders emit the same bytes.

A replan is then an ordinary solve of the derived request. Locks and verified placements
become `fixed_placements`; nothing here knows how a solver works.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._canonical_json import CanonicalJsonError, canonical_json, json_integer, json_spelling
from ._compat import dataclass
from .artifacts import FORMAT as ARTIFACT_FORMAT
from .artifacts import SUITE_VERSION
from .fixed_placements import hull_item_types, require_point_shape
from .made_to_measure import per_square_metre, surface_area
from .request_errors import refuse_orientation
from .units import Length, ParsedMeasures, Weight

__all__ = [
    "FORMAT",
    "EVENT_TYPES",
    "PlanRevisionError",
    "RevisionIssue",
    "apply_events",
    "canonical_revision_json",
    "derive_revision",
    "document_digest",
    "root_revision",
    "stock_cut_boxes",
    "verify_revision_chain",
]

FORMAT = "packvium-plan-revision/v1"

EVENT_TYPES = ("item_missing", "container_substituted", "placement_locked", "placement_verified")

_EVENT_FIELDS = {
    "item_missing": ("item_type", "quantity"),
    "container_substituted": ("container_type", "replacement"),
    "placement_locked": ("placement",),
    "placement_verified": ("placement",),
}

_PLACEMENT_FIELDS = ("item_type", "container_type", "container_instance", "position", "orientation",
                     "placement_id")


class PlanRevisionError(ValueError):
    """A revision could not be built. `code` is one of a closed set shared by four engines:
    `invalid_revision`, `invalid_event`, `event_conflict`, `invalid_artifact`, and the
    canonical-form codes `number_out_of_range`, `invalid_string`, `invalid_value`. A message
    quotes a value by its canonical JSON, so every engine writes the same text."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class RevisionIssue:
    """One way a chain fails its audit, anchored on the revision where it shows."""
    code: str
    revision: int
    detail: str


def canonical_revision_json(document: Mapping[str, Any]) -> str:
    """The RFC 8785 canonical form, the bytes a digest and four engines are compared on."""
    try:
        return canonical_json(document)
    except CanonicalJsonError as error:
        raise PlanRevisionError(error.code, str(error)) from error


def document_digest(document: Mapping[str, Any]) -> str:
    """`sha256:` and the hex SHA-256 of a document's canonical bytes."""
    encoded = canonical_revision_json(document).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def root_revision(request: Mapping[str, Any]) -> Dict[str, Any]:
    """Revision 0: the request as first approved, with nothing recorded against it."""
    if not isinstance(request, Mapping):
        raise PlanRevisionError("invalid_revision", "a request is a JSON object")
    document = _document(0, None, None, [], copy.deepcopy(dict(request)))
    canonical_revision_json(document)
    return document


def derive_revision(
    parent: Mapping[str, Any],
    approved_artifact: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The next revision: `events`, observed against `approved_artifact`, applied to `parent`.

    Copies and hashes the request/artifact in their serialized size. Placement-only event
    replay uses one canonical key per fixed placement; missing-item and carton events still
    scan their request lists, so the whole operation is not unconditionally O(R + E).
    """
    _require_revision(parent)
    _require_artifact(approved_artifact, parent["request"])
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence) or not events:
        raise PlanRevisionError("invalid_event", "a revision records at least one event")
    first = _last_sequence(parent) + 1
    hulls = _hull_types(parent["request"])
    recorded = [_event(event, first + offset, hulls) for offset, event in enumerate(events)]
    number = json_integer(parent["revision"]) + 1
    request = stock_cut_boxes(apply_events(parent["request"], recorded), approved_artifact, number)
    approved = {
        "artifact": document_digest(approved_artifact),
        "replay": copy.deepcopy(dict(approved_artifact["provenance"]["replay"])),
    }
    document = _document(number, document_digest(parent), approved, recorded, request)
    canonical_revision_json(document)
    return document


def apply_events(request: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The request these events derive from `request`. Refuses a contradiction, never physics:
    whether the new fixed set can hold is the engine's admission check when it is solved."""
    if not isinstance(request, Mapping):
        raise PlanRevisionError("invalid_revision", "a request is a JSON object")
    if isinstance(events, (str, bytes, Mapping)) or not isinstance(events, Sequence):
        raise PlanRevisionError("invalid_event", "events are a JSON array")
    derived = copy.deepcopy(dict(request))
    hulls = _hull_types(request)
    placement_index = _PlacementIndex()
    apply = {**_APPLY, "placement_locked": placement_index.apply,
             "placement_verified": placement_index.apply}
    for event in events:
        _require_shape(event, hulls)
        if json_integer(event.get("sequence")) is None:
            raise PlanRevisionError(
                "invalid_event", f"event sequence {json_spelling(event.get('sequence'))} is not an integer")
        apply[event["type"]](derived, event)
    return derived


def stock_cut_boxes(request: Mapping[str, Any], approved_artifact: Mapping[str, Any],
                    revision: int) -> Dict[str, Any]:
    """`request` with every made-to-measure box that holds a fixed placement made a stock
    container. The box was cut for the approved plan, so a replan cannot recut it:
    instance `n` of made-to-measure type `T` -- the `n`-th container of type `T` in the
    artifact, in result order -- becomes container `T@<revision>#<n>`, a copy of `T` without
    `made_to_measure`, quantity 1, at the cut inner and outer size, price and tare, each in
    ticks; it follows `T` in the container list. Its fixed placements move to its instance 1
    and `T`'s own `quantity`, if stated, loses the boxes taken (`T` is dropped at zero). New
    boxes are still cut to measure. Pure; O(C + F + A) for containers, fixed placements and
    the artifact's containers.
    """
    derived = copy.deepcopy(dict(request))
    fixed = derived.get("fixed_placements")
    containers = derived.get("containers")
    if not isinstance(fixed, list) or not isinstance(containers, list):
        return derived
    machines = {entry["id"]: entry for entry in containers
                if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
                and entry.get("made_to_measure") is not None}
    stocked: Dict[str, List[Dict[str, Any]]] = {machine_id: [] for machine_id in machines}
    named: Dict[str, Dict[str, Any]] = {}
    cuts = None
    for entry in fixed:
        if not isinstance(entry, dict) or entry.get("container_type") not in machines:
            continue
        machine_id = entry["container_type"]
        instance = json_integer(entry.get("container_instance", 1))
        stock_id = f"{machine_id}@{revision}#{instance}"
        if stock_id not in named:
            if cuts is None:
                cuts = _cut_boxes(approved_artifact)
            named[stock_id] = _stock_box(derived, machines[machine_id], stock_id,
                                         _cut_inner(cuts, machine_id, instance))
            stocked[machine_id].append(named[stock_id])
        entry["container_type"] = stock_id
        entry["container_instance"] = 1
        # A stated id named the cut box's instance of the machine, which this request no
        # longer has; admission would refuse it as a mismatch.
        entry.pop("placement_id", None)
    if not named:
        return derived
    clash = min(set(named) & {c.get("id") for c in containers if isinstance(c, Mapping)}, default=None)
    if clash is not None:
        raise PlanRevisionError("event_conflict", f"a container is already named {json_spelling(clash)}")
    rebuilt: List[Any] = []
    for container in containers:
        boxes = stocked.get(container.get("id")) if isinstance(container, Mapping) else None
        if not boxes:
            rebuilt.append(container)
            continue
        quantity = json_integer(container.get("quantity")) if container.get("quantity") is not None else None
        if quantity is None or quantity > len(boxes):
            if quantity is not None:
                container["quantity"] = quantity - len(boxes)
            rebuilt.append(container)
        rebuilt.extend(boxes)
    derived["containers"] = rebuilt
    return derived


def _cut_boxes(artifact: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """Index raw shapes in artifact order; unselected dimensions are never parsed. O(A)."""
    work = artifact.get("work_order") if isinstance(artifact, Mapping) else None
    geometry = artifact.get("geometry") if isinstance(artifact, Mapping) else None
    orders = work.get("containers") if isinstance(work, Mapping) else None
    shapes = geometry.get("containers") if isinstance(geometry, Mapping) else None
    cuts: Dict[str, List[Any]] = {}
    if isinstance(orders, list) and isinstance(shapes, list):
        for order, shape in zip(orders, shapes):
            machine_id = order.get("container_type") if isinstance(order, Mapping) else None
            if isinstance(machine_id, str):
                cuts.setdefault(machine_id, []).append(shape)
    return cuts


def _cut_inner(cuts: Dict[str, List[Any]], machine_id: str, instance: Optional[int]) -> List[int]:
    """The selected box's ticks, read only when it is stocked. Expected O(1)."""
    shapes = cuts.get(machine_id, ())
    if instance is not None and 1 <= instance <= len(shapes):
        return [_artifact_ticks(shapes[instance - 1], axis) for axis in ("length", "width", "height")]
    raise PlanRevisionError("invalid_artifact",
                            f"the approved plan holds no {machine_id} instance {json_spelling(instance)}")


def _artifact_ticks(shape: Any, axis: str) -> int:
    sides = shape.get("inner_dimensions") if isinstance(shape, Mapping) else None
    value = sides.get(axis) if isinstance(sides, Mapping) else None
    ticks = json_integer(int(value)) if isinstance(value, str) and value.isascii() and value.isdigit() else None
    if ticks is None:
        raise PlanRevisionError("invalid_artifact", f"a cut box's {axis} is not a tick count")
    return ticks


def _stock_box(request: Mapping[str, Any], machine: Mapping[str, Any], stock_id: str,
               inner: List[int]) -> Dict[str, Any]:
    """The stock container a cut box becomes, priced as `made_to_measure.fabricate` prices it."""
    units = _stock_object(request.get("units"), "units", optional=True)
    unit = units.get("length") if units.get("length") is not None else "mm"
    if not isinstance(unit, str):
        raise PlanRevisionError("invalid_revision", "a made-to-measure container's units.length is not a string")
    measures = ParsedMeasures()
    spec = _stock_object(machine["made_to_measure"], "made_to_measure")

    def read(kind: type, value: Any, default_unit: Any, name: str) -> int:
        try:
            return measures.read(kind, value, default_unit).ticks
        except (ValueError, TypeError, KeyError, ArithmeticError) as error:
            raise PlanRevisionError("invalid_revision",
                                    f"a made-to-measure container's {name} is not a measure") from error

    def length(value: Any) -> int:
        return read(Length, value, unit, "outer_allowance")

    def weight(value: Any, name: str) -> int:
        return read(Weight, value, "g", name)

    allowance = _stock_object(spec.get("outer_allowance"), "outer_allowance", optional=True)
    outer = [inner[index] + (length(allowance[axis]) if allowance.get(axis) is not None else 0)
             for index, axis in enumerate(("length", "width", "height"))]
    area = surface_area(tuple(inner))
    price = _stock_integer(spec.get("cost_per_square_metre_minor"), "cost_per_square_metre_minor")
    tare_rate = (0 if spec.get("tare_per_square_metre") is None
                 else weight(spec["tare_per_square_metre"], "tare_per_square_metre"))
    declared_tare = 0 if machine.get("tare_weight") is None else weight(machine["tare_weight"], "tare_weight")
    box = {key: copy.deepcopy(value) for key, value in machine.items() if key != "made_to_measure"}
    box.update({
        "id": stock_id,
        "quantity": 1,
        "inner_dimensions": _tick_box(inner),
        "outer_dimensions": _tick_box(outer),
        "cost_minor": _stock_integer(machine.get("cost_minor"), "cost_minor") + per_square_metre(price, area),
        "tare_weight": {"value": str(declared_tare + per_square_metre(tare_rate, area)), "unit": "ticks"},
    })
    return box


def _stock_object(value: Any, name: str, optional: bool = False) -> Mapping[str, Any]:
    if value is None and optional:
        return {}
    if not isinstance(value, Mapping):
        raise PlanRevisionError("invalid_revision", f"a made-to-measure container's {name} is not an object")
    return value


def _stock_integer(value: Any, name: str) -> int:
    integer = json_integer(0 if value is None else value)
    if integer is None:
        raise PlanRevisionError("invalid_revision", f"a made-to-measure container's {name} is not an integer")
    return integer


def _tick_box(sides: List[int]) -> Dict[str, Dict[str, str]]:
    return {axis: {"value": str(side), "unit": "ticks"} for axis, side in zip(("length", "width", "height"), sides)}


def _holds_cut_boxes(request: Mapping[str, Any]) -> bool:
    """Whether a fixed placement names a made-to-measure container type."""
    containers = request.get("containers")
    fixed = request.get("fixed_placements")
    if not isinstance(containers, list) or not isinstance(fixed, list):
        return False
    machines = {entry.get("id") for entry in containers
                if isinstance(entry, Mapping) and entry.get("made_to_measure") is not None}
    return any(isinstance(entry, Mapping) and entry.get("container_type") in machines for entry in fixed)


def verify_revision_chain(
    revisions: Sequence[Mapping[str, Any]],
    artifacts: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
) -> List[RevisionIssue]:
    """Every way the chain fails to be what its root and events derive, without stopping at
    the first. `artifacts[k]`, when given, is the artifact revision `k` names as approved."""
    if isinstance(revisions, (str, bytes, Mapping)) or not isinstance(revisions, Sequence):
        raise PlanRevisionError("invalid_revision", "a chain is a JSON array")
    if artifacts is not None and (isinstance(artifacts, (str, bytes, Mapping))
                                  or not isinstance(artifacts, Sequence)):
        raise PlanRevisionError("invalid_artifact", "artifacts is a JSON array")
    issues: List[RevisionIssue] = []
    last_sequence = 0
    for position, revision in enumerate(revisions):
        try:
            _require_revision(revision)
        except PlanRevisionError as error:
            issues.append(RevisionIssue("invalid_revision", position, str(error)))
            return issues
        number = json_integer(revision["revision"])
        if number != position:
            issues.append(RevisionIssue("revision_number", position,
                                        f"revision {number} at position {position}"))
        parent = revisions[position - 1] if position else None
        expected_parent = None if parent is None else document_digest(parent)
        recorded_parent = revision.get("parent")
        if recorded_parent != expected_parent:
            issues.append(RevisionIssue("parent_mismatch", position,
                                        f"parent {_plain(recorded_parent)}, expected {_plain(expected_parent)}"))
        last_sequence = _check_sequences(revision, position, last_sequence, issues)
        artifact = None if artifacts is None or position >= len(artifacts) else artifacts[position]
        if parent is not None:
            _check_request(revision, parent, artifact, position, issues)
        if artifact is not None and parent is not None:
            _check_artifact(revision, parent, artifact, position, issues)
    return issues


# ---------------------------------------------------------------------------- building


def _document(number: int, parent: Optional[str], approved: Optional[Dict[str, Any]],
              events: List[Dict[str, Any]], request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": FORMAT,
        "suite_version": SUITE_VERSION,
        "revision": number,
        "parent": parent,
        "approved": approved,
        "events": events,
        "request": request,
    }


def _require_revision(document: Any) -> None:
    if not isinstance(document, Mapping) or document.get("format") != FORMAT:
        raise PlanRevisionError("invalid_revision", f"not a {FORMAT} document")
    number = json_integer(document.get("revision"))
    if number is None or number < 0:
        raise PlanRevisionError("invalid_revision", "revision is a non-negative integer")
    if not isinstance(document.get("request"), Mapping):
        raise PlanRevisionError("invalid_revision", "a revision carries its request")
    events = document.get("events")
    if not isinstance(events, list):
        raise PlanRevisionError("invalid_revision", "a revision carries its events")
    if not all(isinstance(event, Mapping) and json_integer(event.get("sequence")) is not None
               for event in events):
        raise PlanRevisionError("invalid_revision", "a revision's events each carry an integer sequence")


def _require_artifact(artifact: Any, request: Mapping[str, Any]) -> None:
    if not isinstance(artifact, Mapping) or artifact.get("format") != ARTIFACT_FORMAT:
        raise PlanRevisionError("invalid_artifact", f"not a {ARTIFACT_FORMAT} document")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get("replay"), Mapping):
        raise PlanRevisionError("invalid_artifact", "the artifact carries no provenance.replay")
    if not _same(provenance.get("request"), request):
        raise PlanRevisionError("invalid_artifact",
                                "the artifact was built from a different request than the parent's")


def _last_sequence(revision: Mapping[str, Any]) -> int:
    events = revision["events"]
    return json_integer(events[-1]["sequence"]) if events else _chain_start(revision)


def _chain_start(revision: Mapping[str, Any]) -> int:
    if json_integer(revision["revision"]) != 0:
        raise PlanRevisionError("invalid_revision", "only the root revision records no events")
    return 0


def _event(raw: Any, sequence: int, hulls: frozenset[str]) -> Dict[str, Any]:
    _require_shape(raw, hulls)
    if json_integer(raw.get("sequence")) != sequence:
        raise PlanRevisionError(
            "invalid_event",
            f"event sequence {json_spelling(raw.get('sequence'))} does not continue the chain at {sequence}",
        )
    return copy.deepcopy(dict(raw))


def _hull_types(request: Mapping[str, Any]) -> frozenset[str]:
    """The request's hull item types, which alone may be placed in a signed orientation (1.10)."""
    return hull_item_types(request.get("items")) if isinstance(request, Mapping) else frozenset()


def _require_shape(raw: Any, hulls: frozenset[str]) -> None:
    if not isinstance(raw, Mapping):
        raise PlanRevisionError("invalid_event", "an event is a JSON object")
    kind = raw.get("type")
    if not isinstance(kind, str) or kind not in _EVENT_FIELDS:
        raise PlanRevisionError("invalid_event", f"unknown event type {json_spelling(kind)}")
    allowed = {"sequence", "type", *_EVENT_FIELDS[kind]}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise PlanRevisionError("invalid_event", f"{kind} does not carry {json_spelling(unknown)}")
    missing = [field for field in _EVENT_FIELDS[kind] if field not in raw]
    if missing:
        raise PlanRevisionError("invalid_event", f"{kind} needs {json_spelling(missing)}")
    _VALIDATE[kind](raw, hulls)


def _validate_item_missing(event: Mapping[str, Any], _hulls: frozenset[str]) -> None:
    _require_name(event["item_type"], "item_type")
    quantity = json_integer(event["quantity"])
    if quantity is None or quantity < 1:
        raise PlanRevisionError("invalid_event", "item_missing.quantity is a positive integer")


def _validate_container_substituted(event: Mapping[str, Any], _hulls: frozenset[str]) -> None:
    _require_name(event["container_type"], "container_type")
    replacement = event["replacement"]
    if not isinstance(replacement, Mapping):
        raise PlanRevisionError("invalid_event", "a replacement is a container object")
    _require_name(replacement.get("id"), "replacement.id")


def _validate_placement(event: Mapping[str, Any], hulls: frozenset[str]) -> None:
    placement = event["placement"]
    if not isinstance(placement, Mapping):
        raise PlanRevisionError("invalid_event", "a placement is a fixed-placement object")
    unknown = sorted(set(placement) - set(_PLACEMENT_FIELDS))
    if unknown:
        raise PlanRevisionError("invalid_event", f"a placement does not carry {json_spelling(unknown)}")
    _require_name(placement.get("item_type"), "placement.item_type")
    _require_name(placement.get("container_type"), "placement.container_type")
    refuse_orientation(placement.get("orientation"), placement.get("item_type") in hulls, "",
                       lambda _reason, _field, detail: PlanRevisionError("invalid_event", detail),
                       "placement.orientation")
    instance = json_integer(placement.get("container_instance", 1))
    if instance is None or instance < 1:
        raise PlanRevisionError("invalid_event", "placement.container_instance counts from 1")
    require_point_shape(placement.get("position", {}), "placement.position", "", _invalid_event)
    # Only its type is the event's to judge: whether it names this placement needs the
    # request's unit, and is the engine's admission rule when the revision is solved (1.10).
    identifier = placement.get("placement_id")
    if identifier is not None and not isinstance(identifier, str):
        raise PlanRevisionError("invalid_event", "placement.placement_id is a string")


def _invalid_event(message: str, _field: str) -> PlanRevisionError:
    return PlanRevisionError("invalid_event", message)


def _require_name(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise PlanRevisionError("invalid_event", f"{field} is a non-empty string")


#: Each event type's own rule, given the request's hull item types (only a placement reads them).
_VALIDATE = {
    "item_missing": _validate_item_missing,
    "container_substituted": _validate_container_substituted,
    "placement_locked": _validate_placement,
    "placement_verified": _validate_placement,
}


# ---------------------------------------------------------------------------- applying


def _apply_item_missing(request: Dict[str, Any], event: Mapping[str, Any]) -> None:
    items = _entries(request, "items")
    index = _index_of(items, event["item_type"], "item")
    quantity = json_integer(items[index].get("quantity", 1))
    if quantity is None:
        raise PlanRevisionError("invalid_revision", f"request.items[{index}].quantity is an integer")
    remaining = quantity - json_integer(event["quantity"])
    fixed = sum(1 for entry in _fixed_entries(request)
                if entry.get("item_type") == event["item_type"])
    if remaining < fixed:
        raise PlanRevisionError(
            "event_conflict",
            f"event {_sequence(event)}: {fixed} {event['item_type']} are fixed, "
            f"{max(remaining, 0)} would remain",
        )
    if remaining > 0:
        items[index]["quantity"] = remaining
        return
    if len(items) == 1:
        raise PlanRevisionError("event_conflict",
                                f"event {_sequence(event)}: no item would remain to pack")
    del items[index]


def _apply_container_substituted(request: Dict[str, Any], event: Mapping[str, Any]) -> None:
    containers = _entries(request, "containers")
    index = _index_of(containers, event["container_type"], "container")
    if any(entry.get("container_type") == event["container_type"]
           for entry in _fixed_entries(request)):
        raise PlanRevisionError(
            "event_conflict",
            f"event {_sequence(event)}: fixed placements are in {event['container_type']}",
        )
    replacement_id = event["replacement"]["id"]
    if any(position != index and container.get("id") == replacement_id
           for position, container in enumerate(containers)):
        raise PlanRevisionError(
            "event_conflict",
            f"event {_sequence(event)}: another container is already {replacement_id}",
        )
    containers[index] = copy.deepcopy(dict(event["replacement"]))


class _PlacementIndex:
    """Per-replay index; the request's first-seen fixed-placement order stays untouched."""

    def __init__(self) -> None:
        self.fixed: Optional[List[Dict[str, Any]]] = None
        self.keys: Optional[set[str]] = None
        self.indexable = True

    def apply(self, request: Dict[str, Any], event: Mapping[str, Any]) -> None:
        if self.fixed is None:
            self.fixed = _fixed_entries(request)
            request["fixed_placements"] = self.fixed
            try:
                self.keys = {canonical_revision_json(_identity(entry)) for entry in self.fixed}
            except PlanRevisionError:
                self.indexable = False
        placement = _identity(copy.deepcopy(dict(event["placement"])))
        if self.indexable:
            try:
                key = canonical_revision_json(placement)
            except PlanRevisionError:
                self.indexable = False
            else:
                if key in self.keys:
                    return
                self.fixed.append(dict(event["placement"]))
                self.keys.add(key)
                return
        # An uncanonicalizable value may occur in a direct apply_events call. Preserve the
        # old first-match/error order rather than making index construction an earlier error.
        if any(_same(_identity(entry), placement) for entry in self.fixed):
            return
        self.fixed.append(dict(event["placement"]))


def _identity(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """What makes two fixed entries one box: everything but `placement_id`, which only restates
    the container and position (1.10), with the default instance spelled out."""
    return {"container_instance": 1, **{key: value for key, value in entry.items() if key != "placement_id"}}


_APPLY = {
    "item_missing": _apply_item_missing,
    "container_substituted": _apply_container_substituted,
}


def _entries(request: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    entries = request.get(key)
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise PlanRevisionError("invalid_revision", f"request.{key} is a list of objects")
    return entries


def _fixed_entries(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The request's fixed placements; absent or null is none, anything else not a list of
    objects is a request no event can safely edit."""
    if request.get("fixed_placements") is None:
        return []
    return _entries(request, "fixed_placements")


def _index_of(entries: List[Dict[str, Any]], identifier: str, kind: str) -> int:
    for index, entry in enumerate(entries):
        if entry.get("id") == identifier:
            return index
    raise PlanRevisionError("event_conflict", f"the request has no {kind} {json_spelling(identifier)}")


def _sequence(event: Mapping[str, Any]) -> int:
    return json_integer(event["sequence"])


# ------------------------------------------------------------------------------ audit


def _check_sequences(revision: Mapping[str, Any], position: int, last: int,
                     issues: List[RevisionIssue]) -> int:
    events = revision["events"]
    if position == 0 and events:
        issues.append(RevisionIssue("sequence_gap", 0, "the root revision records no events"))
    if position > 0 and not events:
        issues.append(RevisionIssue("sequence_gap", position, "a revision records at least one event"))
    for event in events:
        sequence = _sequence(event)
        if sequence != last + 1:
            issues.append(RevisionIssue("sequence_gap", position,
                                        f"event {sequence} where {last + 1} was next"))
        last = sequence
    return last


def _check_request(revision: Mapping[str, Any], parent: Mapping[str, Any],
                   artifact: Optional[Mapping[str, Any]], position: int,
                   issues: List[RevisionIssue]) -> None:
    try:
        derived = apply_events(parent["request"], revision["events"])
        if _holds_cut_boxes(derived):
            # The cut sizes live only in the approved artifact.
            if artifact is None:
                issues.append(RevisionIssue("artifact_required", position,
                                            "fixed items in a made-to-measure box are checked against the approved artifact"))
                return
            derived = stock_cut_boxes(derived, artifact, json_integer(revision["revision"]))
    except PlanRevisionError as error:
        issues.append(RevisionIssue("request_mismatch", position, f"the events do not apply: {error}"))
        return
    if not _same(derived, revision["request"]):
        issues.append(RevisionIssue("request_mismatch", position,
                                    "the request is not what the parent's request and these events derive"))


def _check_artifact(revision: Mapping[str, Any], parent: Mapping[str, Any],
                    artifact: Mapping[str, Any], position: int, issues: List[RevisionIssue]) -> None:
    approved = revision.get("approved")
    digest = _digest_or_none(artifact)
    if digest is None or not isinstance(approved, Mapping) or approved.get("artifact") != digest:
        issues.append(RevisionIssue("artifact_mismatch", position,
                                    "the artifact's digest is not the one this revision approved"))
        return
    provenance = artifact.get("provenance") if isinstance(artifact, Mapping) else None
    if not isinstance(provenance, Mapping) or not _same(provenance.get("request"), parent["request"]):
        issues.append(RevisionIssue("artifact_mismatch", position,
                                    "the artifact was built from a different request than the parent's"))
    elif not _same(provenance.get("replay"), approved.get("replay")):
        issues.append(RevisionIssue("artifact_mismatch", position,
                                    "approved.replay is not the artifact's provenance.replay"))


def _same(left: Any, right: Any) -> bool:
    """Equality as the canonical form sees it, so `1.0` and `1` are one value, as in every engine."""
    return canonical_revision_json(left) == canonical_revision_json(right)


def _digest_or_none(document: Any) -> Optional[str]:
    """An audited artifact's digest; one with no canonical form matches no approval."""
    try:
        return document_digest(document)
    except PlanRevisionError:
        return None


def _plain(value: Any) -> str:
    """A digest as itself, and anything a tampered chain puts in its place as JSON."""
    return value if isinstance(value, str) else json_spelling(value)
