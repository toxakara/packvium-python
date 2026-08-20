"""Every way to hand the commerce API something it should refuse, or something legal
that looks like it should not be.

`test_commerce_api.py` proves the wrapper agrees with the model on the shapes a caller
is expected to send. This file is the other half: the shapes a caller is *not* expected
to send, and the handful that look wrong but are not.

The point is not coverage for its own sake. Every case here is a place where a plausible
implementation quietly does the wrong thing instead of failing — treats a JSON `true` as
the integer 1, accepts a float where the contract says exact integer, silently drops a
field it does not recognise, lets a rollback point at a version that does not exist yet,
sorts ids by something other than code point. A test that never sends malformed input
cannot tell a strict parser from a permissive one.
"""

from __future__ import annotations

import json

import pytest

from packvium.commerce import (
    CommerceInputError,
    canonical_json,
    catalog_version_info,
    evaluate_policy,
    load_document,
    quote,
)

TARIFF_VERSION = {
    "effective_at": 0,
    "dimensional_weight_divisor": 5000,
    "cost_per_dimensional_kg_minor": {"zone-a": 450},
    "minimum_charge_minor": 900,
    "fuel_surcharge_permille": 120,
    "accessorials": [{"accessorial_id": "liftgate", "flat_charge_minor": 250}],
}
DOCUMENT = {"tariffs": [{"carrier_id": "acme", "service_id": "ground",
                         "versions": [TARIFF_VERSION]}]}
SHIPMENT = {
    "carrier_id": "acme", "service_id": "ground", "tariff_version": 1,
    "zone": "zone-a", "actual_weight_g": 1200, "volume_mm3": 6_000_000,
}


def tariff_document(**overrides):
    return {"tariffs": [{"carrier_id": "acme", "service_id": "ground",
                         "versions": [dict(TARIFF_VERSION, **overrides)]}]}


def policy_document(**overrides):
    version = {
        "scope": "hazmat", "action": "reject", "priority": 1, "effective_at": 0,
        "predicates": [{"scope": "hazmat", "field": "un_class", "operator": "equals",
                        "value": "1.4"}],
    }
    version.update(overrides)
    return {"policy_rules": [{"rule_id": "r", "versions": [version]}]}


def catalog_document(*versions):
    return {"catalogs": [{"catalog_id": "c", "versions": list(versions)}]}


def catalog_request(**overrides):
    return dict({"catalog_id": "c", "resolved_at": 1}, **overrides)


# ------------------------------------------------------------------- the outer envelope

class TestTheDocumentItself:
    @pytest.mark.parametrize("document", [None, [], "{}", 7, True, 1.5, ()])
    def test_a_document_that_is_not_an_object_is_refused(self, document):
        with pytest.raises(CommerceInputError, match="document: expected an object"):
            quote(document, SHIPMENT)

    def test_an_empty_document_is_legal_and_prices_nothing(self):
        assert quote({}, SHIPMENT)["error"]["code"] == "tariff_not_found"

    @pytest.mark.parametrize("key", ["tariffs", "policy_rules", "catalogs"])
    def test_each_history_may_be_omitted_independently(self, key):
        document = dict(DOCUMENT)
        document.pop(key, None)

        assert load_document(document) is not None

    @pytest.mark.parametrize("collection", ["tariffs", "policy_rules", "catalogs"])
    def test_a_history_collection_must_be_a_list(self, collection):
        with pytest.raises(CommerceInputError, match="expected a list"):
            load_document({collection: {"carrier_id": "acme"}})

    @pytest.mark.parametrize("collection", ["tariffs", "policy_rules", "catalogs"])
    def test_an_empty_history_collection_is_legal(self, collection):
        assert load_document({collection: []}) is not None

    @pytest.mark.parametrize("collection,entry", [
        ("tariffs", {"carrier_id": "a", "service_id": "b", "versions": []}),
        ("policy_rules", {"rule_id": "r", "versions": []}),
        ("catalogs", {"catalog_id": "c", "versions": []}),
    ])
    def test_a_history_with_no_versions_is_refused(self, collection, entry):
        with pytest.raises(CommerceInputError, match="at least one version"):
            load_document({collection: [entry]})

    @pytest.mark.parametrize("collection,entry", [
        ("tariffs", {"carrier_id": "a", "service_id": "b", "versions": [TARIFF_VERSION]}),
        ("policy_rules", {"rule_id": "r", "versions": [
            {"scope": "hazmat", "action": "allow", "priority": 0, "effective_at": 0,
             "predicates": [{"scope": "hazmat", "field": "f", "operator": "exists"}]}]}),
        ("catalogs", {"catalog_id": "c", "versions": [
            {"effective_at": 0, "published_at": 0, "snapshot": {}}]}),
    ])
    def test_two_histories_with_the_same_identity_are_refused(self, collection, entry):
        with pytest.raises(CommerceInputError, match="duplicate"):
            load_document({collection: [entry, entry]})

    def test_a_history_entry_must_be_an_object(self):
        with pytest.raises(CommerceInputError, match="expected an object"):
            load_document({"tariffs": ["acme"]})

    def test_an_unrecognised_top_level_key_is_refused_not_ignored(self):
        with pytest.raises(CommerceInputError, match=r"unrecognised key\(s\) \['discounts'\]"):
            load_document({"discounts": []})

    def test_a_missing_identity_key_names_what_is_missing(self):
        with pytest.raises(CommerceInputError, match=r"missing required key\(s\) \['service_id'\]"):
            load_document({"tariffs": [{"carrier_id": "acme", "versions": [TARIFF_VERSION]}]})


# --------------------------------------------------------------------------- scalar types

class TestScalarTypes:
    @pytest.mark.parametrize("value", [True, False, 1.0, 0.5, "1", None, [], {}])
    def test_only_an_exact_integer_is_an_integer(self, value):
        with pytest.raises(CommerceInputError, match="expected an exact integer"):
            load_document(tariff_document(effective_at=value))

    def test_a_float_that_happens_to_be_whole_is_still_not_an_integer(self):
        with pytest.raises(CommerceInputError, match="expected an exact integer"):
            load_document(tariff_document(minimum_charge_minor=900.0))

    @pytest.mark.parametrize("value", [7, True, None, [], {}])
    def test_only_a_string_is_a_string(self, value):
        with pytest.raises(CommerceInputError, match="expected a string"):
            load_document({"tariffs": [{"carrier_id": value, "service_id": "g",
                                        "versions": [TARIFF_VERSION]}]})

    def test_an_explicit_null_optional_reads_as_absent(self):
        document = tariff_document(minimum_charge_minor=None, accessorials=None)

        assert quote(document, SHIPMENT)["quote"]["minimum_charge_applied"] is False


# --------------------------------------------------------------------------------- tariffs

class TestTariffAdmission:
    @pytest.mark.parametrize("overrides,message", [
        ({"dimensional_weight_divisor": 0}, "must be positive"),
        ({"dimensional_weight_divisor": -1}, "must be positive"),
        ({"effective_at": -1}, "cannot be negative"),
        ({"minimum_charge_minor": -1}, "cannot be negative"),
        ({"fuel_surcharge_permille": -1}, "cannot be negative"),
        ({"cost_per_dimensional_kg_minor": {"zone-a": -1}}, "cannot be negative"),
    ])
    def test_an_out_of_range_tariff_field_fails_admission(self, overrides, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(tariff_document(**overrides))

    def test_a_zone_map_must_be_an_object(self):
        with pytest.raises(CommerceInputError, match="expected an object"):
            load_document(tariff_document(cost_per_dimensional_kg_minor=[["zone-a", 450]]))

    def test_a_tariff_with_no_zones_at_all_is_admitted_and_prices_nothing(self):
        document = tariff_document(cost_per_dimensional_kg_minor={})

        assert quote(document, SHIPMENT)["error"]["code"] == "unavailable_zone"

    @pytest.mark.parametrize("accessorial,message", [
        ({"accessorial_id": "x"}, "exactly one"),
        ({"accessorial_id": "x", "flat_charge_minor": 1, "permille_of_base": 1}, "exactly one"),
        ({"accessorial_id": "x", "flat_charge_minor": -1}, "cannot be negative"),
        ({"accessorial_id": "x", "permille_of_base": -1}, "cannot be negative"),
        ({"accessorial_id": "", "flat_charge_minor": 1}, "required"),
    ])
    def test_a_malformed_accessorial_fails_admission(self, accessorial, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(tariff_document(accessorials=[accessorial]))

    def test_an_accessorial_charging_zero_is_legal(self):
        document = tariff_document(
            accessorials=[{"accessorial_id": "free", "flat_charge_minor": 0}])

        result = quote(document, dict(SHIPMENT, requested_accessorials=["free"]))

        assert result["quote"]["accessorial_charges_minor"] == [["free", 0]]

    def test_a_permille_accessorial_of_a_zero_base_charges_zero(self):
        document = tariff_document(
            cost_per_dimensional_kg_minor={"zone-a": 0}, minimum_charge_minor=0,
            accessorials=[{"accessorial_id": "pct", "permille_of_base": 999}])

        result = quote(document, dict(SHIPMENT, requested_accessorials=["pct"]))

        assert result["quote"]["total_minor"] == 0


# ------------------------------------------------------------------------------- requests

class TestQuoteRequestAdmission:
    @pytest.mark.parametrize("payload", [None, [], "x", 7])
    def test_a_request_that_is_not_an_object_is_refused(self, payload):
        with pytest.raises(CommerceInputError, match="request: expected an object"):
            quote(DOCUMENT, payload)

    def test_neither_pin_is_refused(self):
        request = dict(SHIPMENT)
        request.pop("tariff_version")

        with pytest.raises(CommerceInputError, match="exactly one"):
            quote(DOCUMENT, request)

    def test_both_pins_are_refused(self):
        with pytest.raises(CommerceInputError, match="exactly one"):
            quote(DOCUMENT, dict(SHIPMENT, as_of=0))

    @pytest.mark.parametrize("accessorials,message", [
        (["a", "a"], "unique"),
        ([""], "non-empty"),
        ([7], "expected a string"),
        ("liftgate", "expected a list"),
        ({"liftgate": 1}, "expected a list"),
    ])
    def test_a_malformed_accessorial_request_is_refused(self, accessorials, message):
        with pytest.raises(CommerceInputError, match=message):
            quote(DOCUMENT, dict(SHIPMENT, requested_accessorials=accessorials))

    def test_an_empty_zone_is_refused_rather_than_looked_up(self):
        with pytest.raises(CommerceInputError, match="zone is required"):
            quote(DOCUMENT, dict(SHIPMENT, zone=""))

    @pytest.mark.parametrize("field", ["actual_weight_g", "volume_mm3"])
    def test_a_negative_measurement_is_refused(self, field):
        with pytest.raises(CommerceInputError, match="cannot be negative"):
            quote(DOCUMENT, dict(SHIPMENT, **{field: -1}))

    def test_a_pinned_version_of_zero_or_below_is_simply_not_found(self):
        assert quote(DOCUMENT, dict(SHIPMENT, tariff_version=0))["error"] == {
            "code": "tariff_not_found",
            "fields": {"carrier_id": "acme", "service_id": "ground", "tariff_version": 0},
        }

    def test_a_negative_as_of_predates_every_version(self):
        request = dict(SHIPMENT, tariff_version=None, as_of=-5)

        assert quote(DOCUMENT, request)["error"]["code"] == "no_effective_tariff"

    def test_an_unknown_service_on_a_known_carrier_is_not_found(self):
        request = dict(SHIPMENT, service_id="hyperloop", tariff_version=None, as_of=0)

        assert quote(DOCUMENT, request)["error"] == {
            "code": "tariff_not_found", "fields": {"carrier_id": "acme", "service_id": "hyperloop"},
        }


# --------------------------------------------------------------------------------- policy

class TestPolicyAdmission:
    @pytest.mark.parametrize("overrides,message", [
        ({"scope": "warehouse"}, "unsupported"),
        ({"action": "maybe"}, "unsupported"),
        ({"predicates": []}, "at least one predicate"),
        ({"effective_at": -1}, "cannot be negative"),
    ])
    def test_a_malformed_rule_fails_admission(self, overrides, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(policy_document(**overrides))

    @pytest.mark.parametrize("predicate,message", [
        ({"scope": "hazmat", "field": "f", "operator": "contains", "value": 1}, "unsupported"),
        ({"scope": "warehouse", "field": "f", "operator": "equals", "value": 1}, "unsupported"),
        ({"scope": "hazmat", "field": "", "operator": "exists"}, "field is required"),
        ({"scope": "hazmat", "field": "f", "operator": "equals"}, "requires a value"),
        ({"scope": "customer", "field": "f", "operator": "exists"}, "share the rule's own scope"),
    ])
    def test_a_malformed_predicate_fails_admission(self, predicate, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(policy_document(predicates=[predicate]))

    def test_a_unary_predicate_may_carry_no_value(self):
        document = policy_document(
            predicates=[{"scope": "hazmat", "field": "un_class", "operator": "absent"}])

        assert evaluate_policy(document, {"scope": "hazmat", "context": {}, "as_of": 0})[
            "decision"]["allowed"] is False

    @pytest.mark.parametrize("payload,message", [
        ({"scope": "atlantis", "context": {}, "as_of": 0}, "unsupported policy scope"),
        ({"scope": "hazmat", "context": [], "as_of": 0}, "expected an object"),
        ({"scope": "hazmat", "context": {}}, "exactly one"),
        ({"scope": "hazmat", "context": {}, "as_of": 0, "rule_versions": []}, "exactly one"),
        ({"scope": "hazmat", "context": {}, "rule_versions": "r"}, "expected a list"),
        ({"scope": "hazmat", "context": {}, "rule_versions": [["r"]]}, "pair"),
        ({"scope": "hazmat", "context": {}, "rule_versions": [["r", 1, 2]]}, "pair"),
        ({"scope": "hazmat", "context": {}, "rule_versions": [["r", 1], ["r", 1]]}, "same rule id twice"),
    ])
    def test_a_malformed_policy_request_is_refused(self, payload, message):
        with pytest.raises(CommerceInputError, match=message):
            evaluate_policy(policy_document(), payload)

    def test_an_empty_pinned_snapshot_allows_everything(self):
        request = {"scope": "hazmat", "context": {"un_class": "1.4"}, "rule_versions": []}

        assert evaluate_policy(policy_document(), request)["decision"] == {
            "scope": "hazmat", "allowed": True, "citation": None,
        }

    def test_a_rule_in_another_scope_never_decides_this_one(self):
        request = {"scope": "customer", "context": {"un_class": "1.4"}, "as_of": 0}

        assert evaluate_policy(policy_document(), request)["decision"]["allowed"] is True

    def test_an_ineffective_rule_does_not_participate(self):
        document = policy_document(effective_at=1000)
        request = {"scope": "hazmat", "context": {"un_class": "1.4"}, "as_of": 999}

        assert evaluate_policy(document, request)["decision"]["citation"] is None

    @pytest.mark.parametrize("operator,value,context,allowed", [
        ("equals", "1.4", {"un_class": "1.4"}, False),
        ("equals", "1.4", {"un_class": "1.5"}, True),
        ("not_equals", "1.4", {"un_class": "1.5"}, False),
        ("in", ["1.4", "1.5"], {"un_class": "1.5"}, False),
        ("not_in", ["1.4"], {"un_class": "9"}, False),
        ("exists", None, {"un_class": None}, False),
        ("absent", None, {}, False),
        ("absent", None, {"un_class": None}, True),
    ])
    def test_every_operator_decides_the_way_the_contract_says(
        self, operator, value, context, allowed,
    ):
        predicate = {"scope": "hazmat", "field": "un_class", "operator": operator}
        if value is not None:
            predicate["value"] = value
        document = policy_document(predicates=[predicate])

        result = evaluate_policy(document, {"scope": "hazmat", "context": context, "as_of": 0})

        assert result["decision"]["allowed"] is allowed

    def test_a_context_value_of_a_type_the_predicate_does_not_use_simply_does_not_match(self):
        document = policy_document(
            predicates=[{"scope": "hazmat", "field": "un_class", "operator": "equals",
                         "value": "1.4"}])
        request = {"scope": "hazmat", "context": {"un_class": ["1.4"]}, "as_of": 0}

        assert evaluate_policy(document, request)["decision"]["allowed"] is True


# -------------------------------------------------------------------------------- catalog

class TestCatalogAdmission:
    @pytest.mark.parametrize("entry,message", [
        ({"id": "i", "dimensions_mm": [1, 1], "weight_g": 1}, "exactly 3 axes"),
        ({"id": "i", "dimensions_mm": [1, 1, 1, 1], "weight_g": 1}, "exactly 3 axes"),
        ({"id": "i", "dimensions_mm": [0, 1, 1], "weight_g": 1}, "must be positive"),
        ({"id": "i", "dimensions_mm": [1, 1, 1], "weight_g": 0}, "must be positive"),
        ({"id": "", "dimensions_mm": [1, 1, 1], "weight_g": 1}, "required"),
        ({"id": "i", "dimensions_mm": "1x1x1", "weight_g": 1}, "expected a list"),
    ])
    def test_a_malformed_item_fails_admission(self, entry, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(catalog_document(
                {"effective_at": 0, "published_at": 0, "snapshot": {"items": [entry]}}))

    def test_a_pallet_carries_two_axes_and_an_optional_height(self):
        document = catalog_document({"effective_at": 0, "published_at": 0, "snapshot": {
            "pallets": [{"id": "p", "deck_dimensions_mm": [1200, 800], "max_payload_g": 1}]}})

        assert catalog_version_info(document, catalog_request())["catalog"]["pallet_ids"] == ["p"]

    @pytest.mark.parametrize("pallet,message", [
        ({"id": "p", "deck_dimensions_mm": [1, 1, 1], "max_payload_g": 1}, "exactly 2 axes"),
        ({"id": "p", "deck_dimensions_mm": [1, 1], "max_payload_g": 1,
          "max_stack_height_mm": 0}, "must be positive"),
        ({"id": "p", "deck_dimensions_mm": [1, 1], "max_payload_g": 0}, "must be positive"),
    ])
    def test_a_malformed_pallet_fails_admission(self, pallet, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(catalog_document(
                {"effective_at": 0, "published_at": 0, "snapshot": {"pallets": [pallet]}}))

    @pytest.mark.parametrize("exclusion,message", [
        ({"id": "x", "scope": "item_wheelbarrow", "subject_id": "a", "excluded_id": "b"},
         "unsupported"),
        ({"id": "x", "scope": "item_carton", "subject_id": "", "excluded_id": "b"},
         "must reference both"),
    ])
    def test_a_malformed_exclusion_fails_admission(self, exclusion, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(catalog_document(
                {"effective_at": 0, "published_at": 0, "snapshot": {"exclusions": [exclusion]}}))

    @pytest.mark.parametrize("kind,payload", [
        ("item", {"id": "e", "dimensions_mm": [1, 1, 1], "weight_g": 1}),
        ("carton", {"id": "e", "inner_dimensions_mm": [1, 1, 1], "max_payload_g": 1}),
        ("pallet", {"id": "e", "deck_dimensions_mm": [1, 1], "max_payload_g": 1}),
    ])
    def test_an_override_of_every_kind_is_admitted(self, kind, payload):
        override = {"id": "o", "facility_id": "F", "entry_id": "e",
                    "kind": kind, "override": payload}
        document = catalog_document(
            {"effective_at": 0, "published_at": 0, "snapshot": {"overrides": [override]}})

        counts = catalog_version_info(document, catalog_request())["catalog"]["entry_counts"]

        assert counts["overrides"] == 1

    @pytest.mark.parametrize("override,message", [
        ({"id": "o", "facility_id": "F", "entry_id": "e", "kind": "crate",
          "override": {"id": "e", "dimensions_mm": [1, 1, 1], "weight_g": 1}}, "expected one of"),
        ({"id": "o", "facility_id": "F", "entry_id": "other", "kind": "item",
          "override": {"id": "e", "dimensions_mm": [1, 1, 1], "weight_g": 1}}, "must match"),
        ({"id": "o", "facility_id": "", "entry_id": "e", "kind": "item",
          "override": {"id": "e", "dimensions_mm": [1, 1, 1], "weight_g": 1}}, "required"),
    ])
    def test_a_malformed_override_fails_admission(self, override, message):
        with pytest.raises(CommerceInputError, match=message):
            load_document(catalog_document(
                {"effective_at": 0, "published_at": 0, "snapshot": {"overrides": [override]}}))

    @pytest.mark.parametrize("collection,entry", [
        ("items", {"id": "d", "dimensions_mm": [1, 1, 1], "weight_g": 1}),
        ("cartons", {"id": "d", "inner_dimensions_mm": [1, 1, 1], "max_payload_g": 1}),
        ("pallets", {"id": "d", "deck_dimensions_mm": [1, 1], "max_payload_g": 1}),
    ])
    def test_two_entries_with_the_same_id_in_one_snapshot_are_refused(self, collection, entry):
        with pytest.raises(CommerceInputError, match="duplicate"):
            load_document(catalog_document({"effective_at": 0, "published_at": 0,
                                            "snapshot": {collection: [entry, entry]}}))

    def test_a_rollback_to_a_version_that_does_not_exist_yet_is_refused(self):
        with pytest.raises(CommerceInputError, match="no version 2"):
            load_document(catalog_document(
                {"effective_at": 0, "published_at": 0, "snapshot": {}},
                {"rollback_to": 2, "published_at": 1}))

    def test_a_rollback_to_itself_is_refused(self):
        with pytest.raises(CommerceInputError, match="no version 1"):
            load_document(catalog_document({"rollback_to": 1, "published_at": 1}))

    def test_a_rollback_defaults_its_effective_date_to_its_publication(self):
        document = catalog_document(
            {"effective_at": 0, "published_at": 0, "snapshot": {}},
            {"rollback_to": 1, "published_at": 42})

        catalog = catalog_version_info(document, catalog_request(version=2))["catalog"]

        assert catalog["effective_at"] == catalog["published_at"] == 42
        assert catalog["note"] == "rollback to version 1"

    def test_a_snapshot_and_a_rollback_in_one_version_takes_the_rollback_path(self):
        # `rollback_to` decides which form this is, so a stray `snapshot` beside it is an
        # unrecognised key rather than a silently-ignored one.
        with pytest.raises(CommerceInputError, match="unrecognised key"):
            load_document(catalog_document(
                {"effective_at": 0, "published_at": 0, "snapshot": {}},
                {"rollback_to": 1, "published_at": 1, "snapshot": {}}))

    @pytest.mark.parametrize("payload,message", [
        ({"catalog_id": "c"}, "missing required key"),
        ({"resolved_at": 1}, "missing required key"),
        ({"catalog_id": "c", "resolved_at": 1, "version": 1, "as_of": 1}, "at most one"),
        ({"catalog_id": "c", "resolved_at": 1, "extra": 1}, "unrecognised key"),
    ])
    def test_a_malformed_catalog_request_is_refused(self, payload, message):
        document = catalog_document({"effective_at": 0, "published_at": 0, "snapshot": {}})

        with pytest.raises(CommerceInputError, match=message):
            catalog_version_info(document, payload)


# ------------------------------------------------------------ legal but surprising inputs

class TestLegalButSurprising:
    def test_an_id_may_be_any_non_empty_string_including_punctuation_and_spaces(self):
        weird = ' \t"\\/kľúč 🙂 '
        document = catalog_document({"effective_at": 0, "published_at": 0, "snapshot": {
            "items": [{"id": weird, "dimensions_mm": [1, 1, 1], "weight_g": 1}]}})

        catalog = catalog_version_info(document, catalog_request())["catalog"]

        assert catalog["item_ids"] == [weird]
        assert json.loads(canonical_json({"catalog": catalog}))["catalog"]["item_ids"] == [weird]

    def test_a_zone_name_may_contain_anything_a_json_key_can(self):
        document = tariff_document(cost_per_dimensional_kg_minor={"zóna/1": 450})

        assert quote(document, dict(SHIPMENT, zone="zóna/1"))["status"] == "ok"

    def test_a_quote_far_beyond_a_64_bit_product_stays_exact(self):
        document = tariff_document(
            dimensional_weight_divisor=1,
            cost_per_dimensional_kg_minor={"zone-a": 10 ** 12},
            minimum_charge_minor=0, fuel_surcharge_permille=0, accessorials=[])

        result = quote(document, dict(SHIPMENT, actual_weight_g=0, volume_mm3=10 ** 12))

        assert result["quote"]["total_minor"] == 10 ** 21
        assert isinstance(result["quote"]["total_minor"], int)

    def test_a_thousand_versions_resolve_to_the_last_effective_one(self):
        versions = [dict(TARIFF_VERSION, effective_at=index) for index in range(1000)]
        document = {"tariffs": [{"carrier_id": "acme", "service_id": "ground",
                                 "versions": versions}]}

        request = dict(SHIPMENT, tariff_version=None, as_of=500)

        assert quote(document, request)["quote"]["tariff_version"] == 501

    def test_canonical_json_does_not_escape_non_ascii_or_slashes(self):
        rendered = canonical_json({"note": "zóna/1 🙂"})

        assert rendered == '{"note":"zóna/1 🙂"}'

    def test_two_documents_that_differ_only_in_key_order_price_identically(self):
        shuffled = {"tariffs": [{"versions": [dict(reversed(list(TARIFF_VERSION.items())))],
                                 "service_id": "ground", "carrier_id": "acme"}]}

        assert canonical_json(quote(DOCUMENT, SHIPMENT)) == canonical_json(quote(shuffled, SHIPMENT))
