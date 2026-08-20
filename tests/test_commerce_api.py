"""The exported commercial and control-plane API.

Two things are checked here, and they are different things:

  * the wrapper agrees with the model. Every success case computes the same answer a
    second time by driving `CarrierRegistry` / `PolicyRegistry` / `CatalogRegistry`
    directly, and asserts the exported document reports exactly that. This is the guard
    against the wrapper growing a second implementation.
  * the workspace paths still resolve to the same objects. `commerce/rating/model.py`,
    `domain/policy/model.py` and `domain/catalog/model.py` are re-export shims after
    the export consolidation; the identity assertions below fail the moment one starts carrying
    its own copy.

docs/COMMERCE-API.md is the contract under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from packvium.commerce import (
    REJECTION_CODES,
    CommerceInputError,
    canonical_json,
    catalog_version_info,
    evaluate_policy,
    quote,
)
from packvium.commerce.catalog import CatalogRegistry, CatalogSnapshot, CartonMaster, ItemMaster
from packvium.commerce.policy import (
    PolicyAction,
    PolicyOperator,
    PolicyPredicate,
    PolicyRegistry,
    PolicyScope,
)
from packvium.commerce.rating import AccessorialCharge, CarrierRegistry, RatingRequest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------- fixture data

TARIFF_VERSIONS = [
    {
        "effective_at": 0,
        "dimensional_weight_divisor": 5000,
        "cost_per_dimensional_kg_minor": {"zone-a": 450, "zone-b": 610},
        "minimum_charge_minor": 900,
        "fuel_surcharge_permille": 120,
        "accessorials": [
            {"accessorial_id": "liftgate", "flat_charge_minor": 250},
            {"accessorial_id": "residential", "permille_of_base": 75},
        ],
    },
    {
        "effective_at": 1000,
        "dimensional_weight_divisor": 4000,
        "cost_per_dimensional_kg_minor": {"zone-a": 480},
        "minimum_charge_minor": 950,
        "fuel_surcharge_permille": 140,
        "accessorials": [{"accessorial_id": "liftgate", "flat_charge_minor": 275}],
    },
]

DOCUMENT = {
    "tariffs": [{"carrier_id": "acme", "service_id": "ground", "versions": TARIFF_VERSIONS}],
    "policy_rules": [
        {
            "rule_id": "no-hazmat-air",
            "versions": [
                {
                    "scope": "hazmat", "action": "reject", "priority": 10, "effective_at": 0,
                    "reason": "class 1.4 is not accepted on air services",
                    "predicates": [
                        {"scope": "hazmat", "field": "un_class", "operator": "equals", "value": "1.4"},
                    ],
                },
            ],
        },
        {
            "rule_id": "allow-known-shippers",
            "versions": [
                {
                    "scope": "hazmat", "action": "allow", "priority": 5, "effective_at": 0,
                    "reason": "vetted shipper",
                    "predicates": [
                        {"scope": "hazmat", "field": "shipper", "operator": "in", "value": ["vetted"]},
                    ],
                },
            ],
        },
    ],
    "catalogs": [
        {
            "catalog_id": "dc-12",
            "versions": [
                {
                    "effective_at": 0, "published_at": 0, "note": "initial",
                    "snapshot": {
                        "items": [{"id": "sku-1", "dimensions_mm": [100, 200, 300], "weight_g": 1200}],
                        "cartons": [{
                            "id": "box-m", "inner_dimensions_mm": [320, 240, 180],
                            "max_payload_g": 15000, "cost_minor": 85,
                        }],
                    },
                },
                {"rollback_to": 1, "published_at": 900, "effective_at": 900, "note": "revert"},
            ],
        },
    ],
}


def build_carrier_registry() -> CarrierRegistry:
    """The same two tariff versions the document declares, published by hand."""
    registry = CarrierRegistry()
    for version in TARIFF_VERSIONS:
        registry.publish(
            "acme", "ground",
            effective_at=version["effective_at"],
            dimensional_weight_divisor=version["dimensional_weight_divisor"],
            cost_per_dimensional_kg_minor=version["cost_per_dimensional_kg_minor"],
            minimum_charge_minor=version["minimum_charge_minor"],
            fuel_surcharge_permille=version["fuel_surcharge_permille"],
            accessorials={
                entry["accessorial_id"]: AccessorialCharge(
                    entry["accessorial_id"],
                    flat_charge_minor=entry.get("flat_charge_minor"),
                    permille_of_base=entry.get("permille_of_base"),
                )
                for entry in version["accessorials"]
            },
        )
    return registry


def build_policy_registry() -> PolicyRegistry:
    registry = PolicyRegistry()
    registry.publish(
        "no-hazmat-air", scope=PolicyScope.HAZMAT, action=PolicyAction.REJECT, priority=10,
        effective_at=0, reason="class 1.4 is not accepted on air services",
        predicates=[PolicyPredicate(
            scope=PolicyScope.HAZMAT, field="un_class", operator=PolicyOperator.EQUALS, value="1.4",
        )],
    )
    registry.publish(
        "allow-known-shippers", scope=PolicyScope.HAZMAT, action=PolicyAction.ALLOW, priority=5,
        effective_at=0, reason="vetted shipper",
        predicates=[PolicyPredicate(
            scope=PolicyScope.HAZMAT, field="shipper", operator=PolicyOperator.IN, value=["vetted"],
        )],
    )
    return registry


def build_catalog_registry() -> CatalogRegistry:
    registry = CatalogRegistry("dc-12")
    registry.publish(
        CatalogSnapshot(
            items=(ItemMaster("sku-1", (100, 200, 300), 1200),),
            cartons=(CartonMaster("box-m", (320, 240, 180), 15000, 85),),
        ),
        effective_at=0, published_at=0, note="initial",
    )
    registry.rollback(1, published_at=900, effective_at=900, note="revert")
    return registry


def shipment(**overrides):
    request = {
        "carrier_id": "acme", "service_id": "ground", "tariff_version": 1,
        "zone": "zone-a", "actual_weight_g": 1200, "volume_mm3": 6_000_000,
        "requested_accessorials": ["liftgate"],
    }
    request.update(overrides)
    return request


# ------------------------------------------------------- the wrapper agrees with the model

class TestQuoteMatchesTheModel:
    @pytest.mark.parametrize("pin", [
        {"tariff_version": 1, "as_of": None},
        {"tariff_version": None, "as_of": 0},
    ])
    def test_every_breakdown_field_comes_from_the_rating_model(self, pin):
        result = quote(DOCUMENT, shipment(**pin))

        expected = build_carrier_registry().rate_with_version(
            "acme", "ground", 1,
            RatingRequest(zone="zone-a", actual_weight_g=1200, volume_mm3=6_000_000,
                          requested_accessorials=("liftgate",)),
        )
        assert result["status"] == "ok"
        assert result["quote"] == {
            "carrier_id": expected.carrier_id,
            "service_id": expected.service_id,
            "tariff_version": expected.tariff_version,
            "zone": expected.zone,
            "actual_weight_g": expected.actual_weight_g,
            "dimensional_weight_g": expected.dimensional_weight_g,
            "billed_weight_g": expected.billed_weight_g,
            "base_charge_minor": expected.base_charge_minor,
            "minimum_charge_applied": expected.minimum_charge_applied,
            "fuel_surcharge_minor": expected.fuel_surcharge_minor,
            "accessorial_charges_minor": [list(pair) for pair in expected.accessorial_charges_minor],
            "total_minor": expected.total_minor,
        }

    def test_as_of_resolves_the_later_version_the_registry_resolves(self):
        result = quote(DOCUMENT, shipment(as_of=1500, tariff_version=None,
                                          requested_accessorials=["liftgate"]))

        expected = build_carrier_registry().rate(
            "acme", "ground",
            RatingRequest(zone="zone-a", actual_weight_g=1200, volume_mm3=6_000_000,
                          requested_accessorials=("liftgate",)),
            as_of=1500,
        )
        assert result["quote"]["tariff_version"] == expected.tariff_version == 2
        assert result["quote"]["total_minor"] == expected.total_minor

    def test_accessorial_charges_keep_the_requested_order(self):
        result = quote(DOCUMENT, shipment(requested_accessorials=["residential", "liftgate"]))

        assert [pair[0] for pair in result["quote"]["accessorial_charges_minor"]] == [
            "residential", "liftgate",
        ]

    def test_a_shipment_with_no_accessorials_needs_no_accessorial_key(self):
        request = shipment()
        request.pop("requested_accessorials")

        assert quote(DOCUMENT, request)["quote"]["accessorial_charges_minor"] == []


class TestPolicyMatchesTheModel:
    def test_an_as_of_decision_matches_the_registry(self):
        result = evaluate_policy(DOCUMENT, {"scope": "hazmat", "context": {"un_class": "1.4"}, "as_of": 0})

        expected = build_policy_registry().evaluate(PolicyScope.HAZMAT, {"un_class": "1.4"}, as_of=0)
        assert result["decision"]["allowed"] is expected.allowed is False
        assert result["decision"]["citation"]["rule_id"] == expected.citation.rule_id
        assert result["decision"]["citation"]["version"] == expected.citation.version

    def test_deny_still_outranks_allow_through_the_wrapper(self):
        context = {"un_class": "1.4", "shipper": "vetted"}

        result = evaluate_policy(DOCUMENT, {"scope": "hazmat", "context": context, "as_of": 0})

        assert result["decision"]["allowed"] is False
        assert result["decision"]["citation"]["action"] == "reject"

    def test_nothing_matching_is_allowed_without_a_citation(self):
        result = evaluate_policy(DOCUMENT, {"scope": "hazmat", "context": {"un_class": "9"}, "as_of": 0})

        assert result["decision"] == {"scope": "hazmat", "allowed": True, "citation": None}

    def test_pinned_rule_versions_are_order_independent(self):
        pins = [["no-hazmat-air", 1], ["allow-known-shippers", 1]]
        request = {"scope": "hazmat", "context": {"un_class": "1.4"}, "rule_versions": pins}

        forward = evaluate_policy(DOCUMENT, request)
        reversed_pins = dict(request, rule_versions=list(reversed(pins)))

        assert canonical_json(forward) == canonical_json(evaluate_policy(DOCUMENT, reversed_pins))


class TestCatalogMatchesTheModel:
    def test_metadata_matches_the_registry_resolution(self):
        result = catalog_version_info(DOCUMENT, {"catalog_id": "dc-12", "version": 2, "resolved_at": 1700})

        registry = build_catalog_registry()
        expected = registry.resolve(resolved_at=1700, version=2)
        assert result["catalog"]["effective_at"] == expected.reference.effective_at
        assert result["catalog"]["resolved_at"] == 1700
        assert result["catalog"]["rolled_back_from"] == 1
        assert result["catalog"]["entry_counts"] == {
            "items": 1, "cartons": 1, "pallets": 0, "exclusions": 0, "overrides": 0,
        }
        assert result["catalog"]["item_ids"] == ["sku-1"]

    def test_an_as_of_lookup_selects_the_same_version_the_registry_does(self):
        result = catalog_version_info(DOCUMENT, {"catalog_id": "dc-12", "as_of": 500, "resolved_at": 600})

        expected = build_catalog_registry().resolve(resolved_at=600, as_of=500)
        assert result["catalog"]["version"] == expected.reference.version == 1


# ---------------------------------------------------------------------------- rejections

class TestRejections:
    def test_every_documented_code_is_reachable_and_none_other_is(self):
        produced = {
            quote(DOCUMENT, shipment(carrier_id="nobody"))["error"]["code"],
            quote(DOCUMENT, shipment(tariff_version=99))["error"]["code"],
            quote(DOCUMENT, shipment(tariff_version=None, as_of=-1))["error"]["code"],
            quote(DOCUMENT, shipment(zone="zone-z"))["error"]["code"],
            quote(DOCUMENT, shipment(requested_accessorials=["helicopter"]))["error"]["code"],
            evaluate_policy(DOCUMENT, {
                "scope": "hazmat", "context": {}, "rule_versions": [["ghost", 1]],
            })["error"]["code"],
            evaluate_policy(DOCUMENT, {
                "scope": "hazmat", "context": {}, "rule_versions": [["no-hazmat-air", 7]],
            })["error"]["code"],
            catalog_version_info(DOCUMENT, {"catalog_id": "nope", "resolved_at": 1})["error"]["code"],
            catalog_version_info(DOCUMENT, {"catalog_id": "dc-12", "version": 9, "resolved_at": 1})["error"]["code"],
            catalog_version_info(DOCUMENT, {"catalog_id": "dc-12", "as_of": -1, "resolved_at": 1})["error"]["code"],
            catalog_version_info(DOCUMENT, {"catalog_id": "dc-12", "resolved_at": 1})["error"]["code"],
        }

        assert produced == set(REJECTION_CODES)

    def test_an_unpriceable_zone_names_the_zone_and_the_resolved_version(self):
        result = quote(DOCUMENT, shipment(zone="zone-z"))

        assert result == {
            "api_version": 1, "status": "rejected",
            "error": {"code": "unavailable_zone", "fields": {
                "carrier_id": "acme", "service_id": "ground", "tariff_version": 1, "zone": "zone-z",
            }},
        }

    def test_missing_accessorials_are_reported_together_and_sorted(self):
        result = quote(DOCUMENT, shipment(requested_accessorials=["zeppelin", "helicopter"]))

        assert result["error"]["code"] == "unavailable_accessorial"
        assert result["error"]["fields"]["accessorial_ids"] == ["helicopter", "zeppelin"]

    def test_a_rejection_carries_no_prose(self):
        rendered = canonical_json(quote(DOCUMENT, shipment(zone="zone-z")))

        assert "has no rate for" not in rendered


# -------------------------------------------------------------------------- input errors

class TestInputErrors:
    @pytest.mark.parametrize("request_overrides", [
        {"tariff_version": None},                       # neither pin
        {"as_of": 0},                                   # both pins
        {"actual_weight_g": -1},
        {"actual_weight_g": True},
        {"requested_accessorials": ["liftgate", "liftgate"]},
        {"zone": 7},
    ])
    def test_a_malformed_request_raises_rather_than_rejecting(self, request_overrides):
        with pytest.raises(CommerceInputError):
            quote(DOCUMENT, shipment(**request_overrides))

    def test_an_unrecognised_request_key_is_refused_not_ignored(self):
        with pytest.raises(CommerceInputError, match="unrecognised key"):
            quote(DOCUMENT, shipment(discount_code="FREE"))

    def test_an_unrecognised_document_key_is_refused_not_ignored(self):
        with pytest.raises(CommerceInputError, match="unrecognised key"):
            quote(dict(DOCUMENT, surcharges=[]), shipment())

    def test_an_unsupported_policy_operator_fails_admission(self):
        document = {"policy_rules": [{"rule_id": "r", "versions": [{
            "scope": "hazmat", "action": "reject", "priority": 1, "effective_at": 0,
            "predicates": [{"scope": "hazmat", "field": "x", "operator": "contains", "value": 1}],
        }]}]}

        with pytest.raises(CommerceInputError, match="unsupported"):
            evaluate_policy(document, {"scope": "hazmat", "context": {}, "as_of": 0})

    def test_a_duplicate_accessorial_id_is_refused(self):
        versions = [dict(TARIFF_VERSIONS[0], accessorials=[
            {"accessorial_id": "liftgate", "flat_charge_minor": 1},
            {"accessorial_id": "liftgate", "flat_charge_minor": 2},
        ])]
        document = {"tariffs": [{"carrier_id": "acme", "service_id": "ground", "versions": versions}]}

        with pytest.raises(CommerceInputError, match="duplicate accessorial_id"):
            quote(document, shipment())


# ------------------------------------------------------------------------ canonical form

class TestCanonicalForm:
    def test_key_order_in_the_input_cannot_change_the_output_bytes(self):
        shuffled = dict(reversed(list(shipment().items())))

        assert canonical_json(quote(DOCUMENT, shipment())) == canonical_json(quote(DOCUMENT, shuffled))

    def test_the_canonical_form_is_compact_sorted_json(self):
        rendered = canonical_json(quote(DOCUMENT, shipment()))

        assert rendered == json.dumps(json.loads(rendered), sort_keys=True, separators=(",", ":"))
        assert rendered.startswith('{"api_version":1,')


# --------------------------------------------------------- the workspace paths still hold

@pytest.mark.skipif(
    not (WORKSPACE_ROOT / "commerce" / "rating" / "model.py").exists(),
    reason="workspace tree is not present next to an installed package",
)
class TestWorkspaceShimsReExportTheSameObjects:
    @staticmethod
    def _workspace_module(dotted: str):
        if str(WORKSPACE_ROOT) not in sys.path:
            sys.path.insert(0, str(WORKSPACE_ROOT))
        return __import__(dotted, fromlist=["*"])

    def test_rating_shim_is_not_a_second_implementation(self):
        import packvium.commerce.rating as canonical

        shim = self._workspace_module("commerce.rating.model")
        assert shim.rate_tariff is canonical.rate_tariff
        assert shim.CarrierRegistry is canonical.CarrierRegistry

    def test_policy_shim_is_not_a_second_implementation(self):
        import packvium.commerce.policy as canonical

        shim = self._workspace_module("domain.policy.model")
        assert shim.decide is canonical.decide
        assert shim.PolicyRegistry is canonical.PolicyRegistry

    def test_catalog_shim_is_not_a_second_implementation(self):
        import packvium.commerce.catalog as canonical

        shim = self._workspace_module("domain.catalog.model")
        assert shim.CatalogRegistry is canonical.CatalogRegistry
        assert shim.CatalogSnapshot is canonical.CatalogSnapshot
