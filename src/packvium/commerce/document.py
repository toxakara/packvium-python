"""Parse the canonical commerce document into the three registries.

The document format is specified in docs/COMMERCE-API.md. This module does no
commercial arithmetic whatsoever: it validates shape, then hands every field to
`packvium.commerce.rating`, `.policy` and `.catalog` -- the same `CarrierRegistry`,
`PolicyRegistry` and `CatalogRegistry` the workspace application modules publish into.
A version's number is its 1-based position in its `versions` list, which is exactly how
those registries already number a `publish()`.

Parsing is strict in both directions: a missing required key and an unrecognised extra
key are both `CommerceInputError`. A field this contract does not define cannot be
silently ignored, for the same reason the packing engines refuse an unknown request
field rather than dropping it.

Complexity: one pass over the payload, `O(size of the document)` time and space.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .._compat import dataclass
from .catalog import (
    CartonMaster,
    CatalogError,
    CatalogRegistry,
    CatalogSnapshot,
    ExclusionRule,
    ExclusionScope,
    FacilityOverride,
    ItemMaster,
    PalletMaster,
)
from .errors import CommerceInputError
from .policy import (
    PolicyAction,
    PolicyOperator,
    PolicyPredicate,
    PolicyRegistry,
    PolicyScope,
    UnsupportedPredicateError,
)
from .rating import AccessorialCharge, CarrierRegistry


# ------------------------------------------------------------------- shape primitives

def _fail(path: str, message: str) -> None:
    raise CommerceInputError("{0}: {1}".format(path, message))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(path, "expected a list")
    return value


def _integer(value: Any, path: str) -> int:
    # `bool` is a subclass of `int`; a boolean where an exact integer belongs is a
    # caller mistake, not a zero or a one.
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an exact integer")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    return value


def _default(fields: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """An omitted optional field and an explicit JSON `null` mean the same thing.

    Every other implementation of this contract collapses the two -- PHP through `??`,
    Rust and JavaScript through their own optional lookups -- so reading `.get(key,
    default)` here, which only collapses the first, would make Python the one language
    that rejects `{"minimum_charge_minor": null}`.
    """
    value = fields.get(key)
    return fallback if value is None else value


def _keys(value: Mapping[str, Any], path: str, required: Iterable[str], optional: Iterable[str] = ()) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    if missing:
        _fail(path, "missing required key(s) {0}".format(missing))
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail(path, "unrecognised key(s) {0}".format(unknown))


def _model(path: str, build):
    """Run a model constructor, reporting its own validation as an input error.

    The models validate their own invariants (`__post_init__`); this keeps that one
    definition of "valid" and only re-labels the failure for the API boundary.
    """
    try:
        return build()
    except (ValueError, TypeError, UnsupportedPredicateError, CatalogError) as error:
        raise CommerceInputError("{0}: {1}".format(path, error)) from error


def _dimensions(value: Any, path: str, axes: int) -> Tuple[int, ...]:
    entries = _sequence(value, path)
    if len(entries) != axes:
        _fail(path, "expected exactly {0} axes".format(axes))
    return tuple(_integer(entry, "{0}[{1}]".format(path, index)) for index, entry in enumerate(entries))


# ------------------------------------------------------------------------- the document

@dataclass(frozen=True, slots=True)
class CommerceDocument:
    """The three append-only histories a request is answered against."""

    carriers: CarrierRegistry
    policies: PolicyRegistry
    catalogs: Mapping[str, CatalogRegistry]


def load_document(document: Any) -> CommerceDocument:
    """Build the registries described by one canonical commerce document."""
    payload = _mapping(document, "document")
    _keys(payload, "document", (), ("tariffs", "policy_rules", "catalogs"))
    return CommerceDocument(
        carriers=_load_tariffs(payload.get("tariffs", ())),
        policies=_load_policy_rules(payload.get("policy_rules", ())),
        catalogs=_load_catalogs(payload.get("catalogs", ())),
    )


# --------------------------------------------------------------------------- tariffs

def _load_accessorials(value: Any, path: str) -> Dict[str, AccessorialCharge]:
    charges: Dict[str, AccessorialCharge] = {}
    for index, entry in enumerate(_sequence(value, path)):
        entry_path = "{0}[{1}]".format(path, index)
        fields = _mapping(entry, entry_path)
        _keys(fields, entry_path, ("accessorial_id",), ("flat_charge_minor", "permille_of_base"))
        accessorial_id = _text(fields["accessorial_id"], entry_path + ".accessorial_id")
        if accessorial_id in charges:
            _fail(entry_path, "duplicate accessorial_id {0!r}".format(accessorial_id))
        flat = fields.get("flat_charge_minor")
        permille = fields.get("permille_of_base")
        charges[accessorial_id] = _model(entry_path, lambda: AccessorialCharge(
            accessorial_id=accessorial_id,
            flat_charge_minor=None if flat is None else _integer(flat, entry_path + ".flat_charge_minor"),
            permille_of_base=None if permille is None else _integer(permille, entry_path + ".permille_of_base"),
        ))
    return charges


def _load_tariffs(value: Any) -> CarrierRegistry:
    registry = CarrierRegistry()
    seen: set = set()
    for index, entry in enumerate(_sequence(value, "document.tariffs")):
        path = "document.tariffs[{0}]".format(index)
        fields = _mapping(entry, path)
        _keys(fields, path, ("carrier_id", "service_id", "versions"))
        carrier_id = _text(fields["carrier_id"], path + ".carrier_id")
        service_id = _text(fields["service_id"], path + ".service_id")
        if (carrier_id, service_id) in seen:
            _fail(path, "duplicate tariff history for {0}/{1}".format(carrier_id, service_id))
        seen.add((carrier_id, service_id))
        _publish_tariff_versions(registry, carrier_id, service_id, fields["versions"], path)
    return registry


def _publish_tariff_versions(
    registry: CarrierRegistry, carrier_id: str, service_id: str, value: Any, parent: str,
) -> None:
    versions = _sequence(value, parent + ".versions")
    if not versions:
        _fail(parent + ".versions", "a tariff history needs at least one version")
    for index, entry in enumerate(versions):
        path = "{0}.versions[{1}]".format(parent, index)
        fields = _mapping(entry, path)
        _keys(
            fields, path,
            ("effective_at", "dimensional_weight_divisor", "cost_per_dimensional_kg_minor"),
            ("minimum_charge_minor", "fuel_surcharge_permille", "accessorials"),
        )
        zones = _mapping(fields["cost_per_dimensional_kg_minor"], path + ".cost_per_dimensional_kg_minor")
        costs = {
            _text(zone, path + ".cost_per_dimensional_kg_minor key"):
                _integer(cost, "{0}.cost_per_dimensional_kg_minor[{1!r}]".format(path, zone))
            for zone, cost in zones.items()
        }
        accessorials = _load_accessorials(_default(fields, "accessorials", ()), path + ".accessorials")
        _model(path, lambda: registry.publish(
            carrier_id, service_id,
            effective_at=_integer(fields["effective_at"], path + ".effective_at"),
            dimensional_weight_divisor=_integer(
                fields["dimensional_weight_divisor"], path + ".dimensional_weight_divisor"),
            cost_per_dimensional_kg_minor=costs,
            minimum_charge_minor=_integer(
                _default(fields, "minimum_charge_minor", 0), path + ".minimum_charge_minor"),
            fuel_surcharge_permille=_integer(
                _default(fields, "fuel_surcharge_permille", 0), path + ".fuel_surcharge_permille"),
            accessorials=accessorials,
        ))


# ----------------------------------------------------------------------- policy rules

def _enum(kind, value: Any, path: str):
    try:
        return kind(_text(value, path))
    except ValueError:
        _fail(path, "unsupported {0} {1!r}".format(kind.__name__, value))


def _load_predicates(value: Any, path: str) -> List[PolicyPredicate]:
    predicates = []
    for index, entry in enumerate(_sequence(value, path)):
        entry_path = "{0}[{1}]".format(path, index)
        fields = _mapping(entry, entry_path)
        _keys(fields, entry_path, ("scope", "field", "operator"), ("value",))
        predicates.append(_model(entry_path, lambda: PolicyPredicate(
            scope=_enum(PolicyScope, fields["scope"], entry_path + ".scope"),
            field=_text(fields["field"], entry_path + ".field"),
            operator=_enum(PolicyOperator, fields["operator"], entry_path + ".operator"),
            value=fields.get("value"),
        )))
    return predicates


def _load_policy_rules(value: Any) -> PolicyRegistry:
    registry = PolicyRegistry()
    seen: set = set()
    for index, entry in enumerate(_sequence(value, "document.policy_rules")):
        path = "document.policy_rules[{0}]".format(index)
        fields = _mapping(entry, path)
        _keys(fields, path, ("rule_id", "versions"))
        rule_id = _text(fields["rule_id"], path + ".rule_id")
        if rule_id in seen:
            _fail(path, "duplicate rule history for {0!r}".format(rule_id))
        seen.add(rule_id)
        _publish_rule_versions(registry, rule_id, fields["versions"], path)
    return registry


def _publish_rule_versions(registry: PolicyRegistry, rule_id: str, value: Any, parent: str) -> None:
    versions = _sequence(value, parent + ".versions")
    if not versions:
        _fail(parent + ".versions", "a rule history needs at least one version")
    for index, entry in enumerate(versions):
        path = "{0}.versions[{1}]".format(parent, index)
        fields = _mapping(entry, path)
        _keys(fields, path, ("scope", "action", "predicates", "priority", "effective_at"), ("reason",))
        _model(path, lambda: registry.publish(
            rule_id,
            scope=_enum(PolicyScope, fields["scope"], path + ".scope"),
            action=_enum(PolicyAction, fields["action"], path + ".action"),
            predicates=_load_predicates(fields["predicates"], path + ".predicates"),
            priority=_integer(fields["priority"], path + ".priority"),
            effective_at=_integer(fields["effective_at"], path + ".effective_at"),
            reason=_text(_default(fields, "reason", ""), path + ".reason"),
        ))


# --------------------------------------------------------------------------- catalogs

def _load_item(fields: Mapping[str, Any], path: str) -> ItemMaster:
    _keys(fields, path, ("id", "dimensions_mm", "weight_g"), ("description",))
    return _model(path, lambda: ItemMaster(
        id=_text(fields["id"], path + ".id"),
        dimensions_mm=_dimensions(fields["dimensions_mm"], path + ".dimensions_mm", 3),
        weight_g=_integer(fields["weight_g"], path + ".weight_g"),
        description=_text(_default(fields, "description", ""), path + ".description"),
    ))


def _load_carton(fields: Mapping[str, Any], path: str) -> CartonMaster:
    _keys(fields, path, ("id", "inner_dimensions_mm", "max_payload_g"), ("cost_minor",))
    return _model(path, lambda: CartonMaster(
        id=_text(fields["id"], path + ".id"),
        inner_dimensions_mm=_dimensions(fields["inner_dimensions_mm"], path + ".inner_dimensions_mm", 3),
        max_payload_g=_integer(fields["max_payload_g"], path + ".max_payload_g"),
        cost_minor=_integer(_default(fields, "cost_minor", 0), path + ".cost_minor"),
    ))


def _load_pallet(fields: Mapping[str, Any], path: str) -> PalletMaster:
    _keys(fields, path, ("id", "deck_dimensions_mm", "max_payload_g"), ("max_stack_height_mm",))
    height = fields.get("max_stack_height_mm")
    return _model(path, lambda: PalletMaster(
        id=_text(fields["id"], path + ".id"),
        deck_dimensions_mm=_dimensions(fields["deck_dimensions_mm"], path + ".deck_dimensions_mm", 2),
        max_payload_g=_integer(fields["max_payload_g"], path + ".max_payload_g"),
        max_stack_height_mm=None if height is None else _integer(height, path + ".max_stack_height_mm"),
    ))


_ENTRY_LOADERS = {"item": _load_item, "carton": _load_carton, "pallet": _load_pallet}


def _load_exclusion(fields: Mapping[str, Any], path: str) -> ExclusionRule:
    _keys(fields, path, ("id", "scope", "subject_id", "excluded_id"), ("reason",))
    return _model(path, lambda: ExclusionRule(
        id=_text(fields["id"], path + ".id"),
        scope=_enum(ExclusionScope, fields["scope"], path + ".scope"),
        subject_id=_text(fields["subject_id"], path + ".subject_id"),
        excluded_id=_text(fields["excluded_id"], path + ".excluded_id"),
        reason=_text(_default(fields, "reason", ""), path + ".reason"),
    ))


def _load_override(fields: Mapping[str, Any], path: str) -> FacilityOverride:
    _keys(fields, path, ("id", "facility_id", "entry_id", "kind", "override"))
    kind = _text(fields["kind"], path + ".kind")
    if kind not in _ENTRY_LOADERS:
        _fail(path + ".kind", "expected one of {0}".format(sorted(_ENTRY_LOADERS)))
    override = _ENTRY_LOADERS[kind](_mapping(fields["override"], path + ".override"), path + ".override")
    return _model(path, lambda: FacilityOverride(
        id=_text(fields["id"], path + ".id"),
        facility_id=_text(fields["facility_id"], path + ".facility_id"),
        entry_id=_text(fields["entry_id"], path + ".entry_id"),
        override=override,
    ))


def _load_snapshot(value: Any, path: str) -> CatalogSnapshot:
    fields = _mapping(value, path)
    _keys(fields, path, (), ("items", "cartons", "pallets", "exclusions", "overrides"))

    def entries(key: str, load):
        collected = []
        for index, entry in enumerate(_sequence(_default(fields, key, ()), "{0}.{1}".format(path, key))):
            entry_path = "{0}.{1}[{2}]".format(path, key, index)
            collected.append(load(_mapping(entry, entry_path), entry_path))
        return tuple(collected)

    return _model(path, lambda: CatalogSnapshot(
        items=entries("items", _load_item),
        cartons=entries("cartons", _load_carton),
        pallets=entries("pallets", _load_pallet),
        exclusions=entries("exclusions", _load_exclusion),
        overrides=entries("overrides", _load_override),
    ))


def _load_catalogs(value: Any) -> Dict[str, CatalogRegistry]:
    catalogs: Dict[str, CatalogRegistry] = {}
    for index, entry in enumerate(_sequence(value, "document.catalogs")):
        path = "document.catalogs[{0}]".format(index)
        fields = _mapping(entry, path)
        _keys(fields, path, ("catalog_id", "versions"))
        catalog_id = _text(fields["catalog_id"], path + ".catalog_id")
        if catalog_id in catalogs:
            _fail(path, "duplicate catalog history for {0!r}".format(catalog_id))
        registry = _model(path, lambda: CatalogRegistry(catalog_id))
        _publish_catalog_versions(registry, fields["versions"], path)
        catalogs[catalog_id] = registry
    return catalogs


def _publish_catalog_versions(registry: CatalogRegistry, value: Any, parent: str) -> None:
    versions = _sequence(value, parent + ".versions")
    if not versions:
        _fail(parent + ".versions", "a catalog history needs at least one version")
    for index, entry in enumerate(versions):
        path = "{0}.versions[{1}]".format(parent, index)
        fields = _mapping(entry, path)
        if "rollback_to" in fields:
            _publish_rollback(registry, fields, path)
            continue
        _keys(fields, path, ("effective_at", "published_at", "snapshot"), ("note",))
        _model(path, lambda: registry.publish(
            _load_snapshot(fields["snapshot"], path + ".snapshot"),
            effective_at=_integer(fields["effective_at"], path + ".effective_at"),
            published_at=_integer(fields["published_at"], path + ".published_at"),
            note=_text(_default(fields, "note", ""), path + ".note"),
        ))


def _publish_rollback(registry: CatalogRegistry, fields: Mapping[str, Any], path: str) -> None:
    _keys(fields, path, ("rollback_to", "published_at"), ("effective_at", "note"))
    effective_at: Optional[int] = None
    if fields.get("effective_at") is not None:
        effective_at = _integer(fields["effective_at"], path + ".effective_at")
    _model(path, lambda: registry.rollback(
        _integer(fields["rollback_to"], path + ".rollback_to"),
        published_at=_integer(fields["published_at"], path + ".published_at"),
        effective_at=effective_at,
        note=_text(_default(fields, "note", ""), path + ".note"),
    ))
