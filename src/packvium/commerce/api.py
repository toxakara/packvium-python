"""The three exported commercial and control-plane functions.

Each one loads the caller's commerce document into the registries defined by
`packvium.commerce.rating`, `.policy` and `.catalog`, asks those registries the
question, and serializes the answer. No price, decision or version resolution is
computed here -- every number in a result comes back out of the same objects
`commerce/rating/objective.py` and `integration/product/` already use, which is what
keeps the exported answer and the answer a packing request optimises against from ever
diverging.

The contract -- request shapes, result shapes and the closed set of rejection codes --
is docs/COMMERCE-API.md. Complexity is stated there too; the wrapper itself adds one
linear parse of the payload and nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .catalog import (
    AmbiguousCatalogReferenceError,
    CatalogVersionNotFoundError,
    NoEffectiveCatalogVersionError,
    ResolvedCatalog,
)
from .document import CommerceDocument, load_document
from .errors import CommerceInputError, _Rejection
from .policy import (
    PolicyDecision,
    PolicyRegistry,
    PolicyRule,
    PolicyRuleNotFoundError,
    PolicyScope,
    PolicyVersionNotFoundError,
    decide,
)
from .rating import RateBreakdown, RatingRequest, TariffNotFoundError, UnavailableServiceError, rate_tariff

API_VERSION = 1

#: The closed set of rejection codes, in the order docs/COMMERCE-API.md tabulates them.
REJECTION_CODES = (
    "tariff_not_found",
    "no_effective_tariff",
    "unavailable_zone",
    "unavailable_accessorial",
    "policy_rule_not_found",
    "policy_version_not_found",
    "catalog_not_found",
    "catalog_version_not_found",
    "no_effective_catalog_version",
    "ambiguous_catalog_reference",
)


# ------------------------------------------------------------------ request primitives

def _fail(path: str, message: str) -> None:
    raise CommerceInputError("{0}: {1}".format(path, message))


def _request(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("request", "expected an object")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an exact integer")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    return value


def _keys(value: Mapping[str, Any], path: str, required, optional=()) -> None:
    required_set = set(required)
    missing = sorted(required_set - set(value))
    if missing:
        _fail(path, "missing required key(s) {0}".format(missing))
    unknown = sorted(set(value) - (required_set | set(optional)))
    if unknown:
        _fail(path, "unrecognised key(s) {0}".format(unknown))


def _exactly_one(value: Mapping[str, Any], path: str, names: Sequence[str]) -> str:
    present = [name for name in names if value.get(name) is not None]
    if len(present) != 1:
        _fail(path, "expected exactly one of {0}".format(list(names)))
    return present[0]


def _ok(key: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {"api_version": API_VERSION, "status": "ok", key: dict(payload)}


def canonical_json(result: Mapping[str, Any]) -> str:
    """The one byte-comparable spelling of a result document.

    Cross-language equality is asserted on this string, not on a parsed object, so key
    order and whitespace cannot make two identical answers look different -- or two
    different answers look identical.
    """
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _rejected(rejection: _Rejection) -> Dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "status": "rejected",
        "error": {"code": rejection.code, "fields": dict(rejection.fields)},
    }


# -------------------------------------------------------------------------------- quote

def quote(document: Any, request: Any) -> Dict[str, Any]:
    """Price one shipment against one pinned or effective-dated tariff version.

    `document` is a canonical commerce document, `request` names the carrier service,
    the version pin (`tariff_version`) or the instant (`as_of`), and the shipment.
    Returns the `RateBreakdown` `commerce/rating/model.py` produces, field for field.
    """
    loaded = load_document(document)
    fields = _request(request)
    _keys(
        fields, "request",
        ("carrier_id", "service_id", "zone", "actual_weight_g", "volume_mm3"),
        ("tariff_version", "as_of", "requested_accessorials"),
    )
    pin = _exactly_one(fields, "request", ("tariff_version", "as_of"))
    carrier_id = _text(fields["carrier_id"], "request.carrier_id")
    service_id = _text(fields["service_id"], "request.service_id")
    accessorials = _accessorial_ids(fields.get("requested_accessorials"))
    rating_request = _build_rating_request(fields, accessorials)

    try:
        tariff = _resolve_tariff(loaded, carrier_id, service_id, fields, pin)
        breakdown = _rate(tariff, rating_request, carrier_id, service_id)
    except _Rejection as rejection:
        return _rejected(rejection)
    return _ok("quote", _quote_payload(breakdown))


def _accessorial_ids(value: Any) -> Tuple[str, ...]:
    """The requested accessorials, as a list of strings and nothing else.

    A bare string and a mapping are both iterable, so iterating whatever arrives would
    quietly turn `"liftgate"` into eight one-character ids and `{"liftgate": 1}` into a
    one-element list -- two wrong answers where the other implementations report a
    malformed request.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        _fail("request.requested_accessorials", "expected a list")
    return tuple(
        _text(entry, "request.requested_accessorials[{0}]".format(index))
        for index, entry in enumerate(value)
    )


def _build_rating_request(fields: Mapping[str, Any], accessorials: Tuple[str, ...]) -> RatingRequest:
    try:
        return RatingRequest(
            zone=_text(fields["zone"], "request.zone"),
            actual_weight_g=_integer(fields["actual_weight_g"], "request.actual_weight_g"),
            volume_mm3=_integer(fields["volume_mm3"], "request.volume_mm3"),
            requested_accessorials=accessorials,
        )
    except ValueError as error:
        raise CommerceInputError("request: {0}".format(error)) from error


def _resolve_tariff(loaded: CommerceDocument, carrier_id: str, service_id: str,
                    fields: Mapping[str, Any], pin: str):
    identity = {"carrier_id": carrier_id, "service_id": service_id}
    if pin == "tariff_version":
        version = _integer(fields["tariff_version"], "request.tariff_version")
        try:
            return loaded.carriers.tariff(carrier_id, service_id, version)
        except TariffNotFoundError:
            raise _Rejection("tariff_not_found", dict(identity, tariff_version=version))
    as_of = _integer(fields["as_of"], "request.as_of")
    try:
        loaded.carriers.versions(carrier_id, service_id)
    except TariffNotFoundError:
        raise _Rejection("tariff_not_found", dict(identity))
    try:
        return loaded.carriers.effective_tariff(carrier_id, service_id, as_of=as_of)
    except TariffNotFoundError:
        raise _Rejection("no_effective_tariff", dict(identity, as_of=as_of))


def _rate(tariff, rating_request: RatingRequest, carrier_id: str, service_id: str) -> RateBreakdown:
    try:
        return rate_tariff(tariff, rating_request)
    except UnavailableServiceError as error:
        identity = {
            "carrier_id": carrier_id, "service_id": service_id, "tariff_version": tariff.version,
        }
        if error.zone is not None:
            raise _Rejection("unavailable_zone", dict(identity, zone=error.zone))
        raise _Rejection(
            "unavailable_accessorial", dict(identity, accessorial_ids=list(error.accessorial_ids)),
        )


def _quote_payload(breakdown: RateBreakdown) -> Dict[str, Any]:
    return {
        "carrier_id": breakdown.carrier_id,
        "service_id": breakdown.service_id,
        "tariff_version": breakdown.tariff_version,
        "zone": breakdown.zone,
        "actual_weight_g": breakdown.actual_weight_g,
        "dimensional_weight_g": breakdown.dimensional_weight_g,
        "billed_weight_g": breakdown.billed_weight_g,
        "base_charge_minor": breakdown.base_charge_minor,
        "minimum_charge_applied": breakdown.minimum_charge_applied,
        "fuel_surcharge_minor": breakdown.fuel_surcharge_minor,
        "accessorial_charges_minor": [
            [accessorial_id, amount] for accessorial_id, amount in breakdown.accessorial_charges_minor
        ],
        "total_minor": breakdown.total_minor,
    }


# ----------------------------------------------------------------------- evaluate_policy

def evaluate_policy(document: Any, request: Any) -> Dict[str, Any]:
    """Decide one eligibility question against a pinned or effective-dated rule set."""
    loaded = load_document(document)
    fields = _request(request)
    _keys(fields, "request", ("scope", "context"), ("as_of", "rule_versions"))
    pin = _exactly_one(fields, "request", ("as_of", "rule_versions"))
    scope = _policy_scope(fields["scope"])
    context = fields["context"]
    if not isinstance(context, Mapping):
        _fail("request.context", "expected an object")

    try:
        decision = _decide(loaded.policies, scope, context, fields, pin)
    except _Rejection as rejection:
        return _rejected(rejection)
    return _ok("decision", _decision_payload(decision))


def _policy_scope(value: Any) -> PolicyScope:
    try:
        return PolicyScope(_text(value, "request.scope"))
    except ValueError:
        _fail("request.scope", "unsupported policy scope {0!r}".format(value))


def _decide(policies: PolicyRegistry, scope: PolicyScope, context: Mapping[str, Any],
            fields: Mapping[str, Any], pin: str) -> PolicyDecision:
    if pin == "as_of":
        return policies.evaluate(scope, context, as_of=_integer(fields["as_of"], "request.as_of"))
    return decide(_pinned_rules(policies, fields["rule_versions"]), scope, context)


def _pinned_rules(policies: PolicyRegistry, value: Any) -> Sequence[PolicyRule]:
    pins: List[Tuple[str, int]] = []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("request.rule_versions", "expected a list")
    for index, entry in enumerate(value):
        path = "request.rule_versions[{0}]".format(index)
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            _fail(path, "expected a [rule_id, version] pair")
        pins.append((_text(entry[0], path + "[0]"), _integer(entry[1], path + "[1]")))
    try:
        return policies.resolve_versions(pins)
    except ValueError as error:
        raise CommerceInputError("request.rule_versions: {0}".format(error)) from error
    except PolicyVersionNotFoundError as error:
        raise _Rejection(
            "policy_version_not_found", {"rule_id": error.rule_id, "version": error.version},
        )
    except PolicyRuleNotFoundError as error:
        raise _Rejection("policy_rule_not_found", {"rule_id": error.rule_id})


def _decision_payload(decision: PolicyDecision) -> Dict[str, Any]:
    citation = decision.citation
    return {
        "scope": decision.scope.value,
        "allowed": decision.allowed,
        "citation": None if citation is None else {
            "rule_id": citation.rule_id,
            "version": citation.version,
            "action": citation.action.value,
            "priority": citation.priority,
            "reason": citation.reason,
        },
    }


# ------------------------------------------------------------------ catalog_version_info

def catalog_version_info(document: Any, request: Any) -> Dict[str, Any]:
    """Report which catalog version a reference resolves to, and what it contains."""
    loaded = load_document(document)
    fields = _request(request)
    _keys(fields, "request", ("catalog_id", "resolved_at"), ("version", "as_of"))
    catalog_id = _text(fields["catalog_id"], "request.catalog_id")
    resolved_at = _integer(fields["resolved_at"], "request.resolved_at")
    version = None if fields.get("version") is None else _integer(fields["version"], "request.version")
    as_of = None if fields.get("as_of") is None else _integer(fields["as_of"], "request.as_of")
    if version is not None and as_of is not None:
        _fail("request", "expected at most one of ['version', 'as_of']")

    try:
        resolved = _resolve_catalog(loaded, catalog_id, resolved_at, version, as_of)
    except _Rejection as rejection:
        return _rejected(rejection)
    return _ok("catalog", _catalog_payload(loaded, resolved))


def _resolve_catalog(loaded: CommerceDocument, catalog_id: str, resolved_at: int,
                     version: Optional[int], as_of: Optional[int]) -> ResolvedCatalog:
    registry = loaded.catalogs.get(catalog_id)
    if registry is None:
        raise _Rejection("catalog_not_found", {"catalog_id": catalog_id})
    selector: Dict[str, Any] = {"catalog_id": catalog_id}
    if version is not None:
        selector["version"] = version
    if as_of is not None:
        selector["as_of"] = as_of
    try:
        return registry.resolve(resolved_at=resolved_at, version=version, as_of=as_of)
    except CatalogVersionNotFoundError:
        raise _Rejection("catalog_version_not_found", selector)
    except NoEffectiveCatalogVersionError:
        raise _Rejection("no_effective_catalog_version", selector)
    except AmbiguousCatalogReferenceError:
        raise _Rejection("ambiguous_catalog_reference", selector)


def _catalog_payload(loaded: CommerceDocument, resolved: ResolvedCatalog) -> Dict[str, Any]:
    reference = resolved.reference
    snapshot = resolved.snapshot
    published = loaded.catalogs[reference.catalog_id].versions[reference.version - 1]
    return {
        "catalog_id": reference.catalog_id,
        "version": reference.version,
        "effective_at": reference.effective_at,
        "published_at": published.published_at,
        "resolved_at": reference.resolved_at,
        "rolled_back_from": published.rolled_back_from,
        "note": published.note,
        "entry_counts": {
            "items": len(snapshot.items),
            "cartons": len(snapshot.cartons),
            "pallets": len(snapshot.pallets),
            "exclusions": len(snapshot.exclusions),
            "overrides": len(snapshot.overrides),
        },
        "item_ids": sorted(entry.id for entry in snapshot.items),
        "carton_ids": sorted(entry.id for entry in snapshot.cartons),
        "pallet_ids": sorted(entry.id for entry in snapshot.pallets),
    }
