"""The commerce models' own guards, reached by constructing them directly.

`packvium.commerce.rating`, `.policy` and `.catalog` are public: a caller can build a
`Tariff` or a `PolicyDecision` by hand instead of going through a document. That path
skips every check the document parser performs, so the models validate themselves — and
a guard nothing ever trips is a guard nobody knows still works.

Each test here constructs an object the parser would never produce, and asserts the
model refuses it rather than carrying the contradiction forward into an answer.
"""

from __future__ import annotations

import pytest

from packvium.commerce.catalog import (
    CartonMaster,
    CatalogEntryKind,
    CatalogEntryNotFoundError,
    CatalogError,
    CatalogReference,
    CatalogRegistry,
    CatalogSnapshot,
    CatalogVersion,
    CatalogVersionNotFoundError,
    ExclusionRule,
    ExclusionScope,
    FacilityOverride,
    ItemMaster,
    PalletMaster,
    ResolvedCatalog,
)
from packvium.commerce.policy import (
    PolicyAction,
    PolicyCitation,
    PolicyDecision,
    PolicyOperator,
    PolicyPredicate,
    PolicyRegistry,
    PolicyRule,
    PolicyScope,
    UnsupportedPredicateError,
)
from packvium.commerce.rating import AccessorialCharge, Tariff, _ceil_div


def item(id="i"):
    return ItemMaster(id, (10, 10, 10), 1)


def carton(id="c"):
    return CartonMaster(id, (10, 10, 10), 1)


def pallet(id="p"):
    return PalletMaster(id, (10, 10), 1)


def predicate(scope=PolicyScope.HAZMAT):
    return PolicyPredicate(scope=scope, field="f", operator=PolicyOperator.EXISTS)


# --------------------------------------------------------------------------------- rating

class TestRatingGuards:
    @pytest.mark.parametrize("denominator", [0, -1, -1000])
    def test_ceil_div_refuses_a_non_positive_denominator(self, denominator):
        # Unreachable through a parsed document -- the divisor is validated positive
        # before it gets here -- so this is the only place the guard can be exercised.
        with pytest.raises(ValueError, match="denominator must be positive"):
            _ceil_div(1, denominator)

    @pytest.mark.parametrize("numerator,denominator,expected", [
        (0, 7, 0), (1, 7, 1), (7, 7, 1), (8, 7, 2), (13, 7, 2), (14, 7, 2),
    ])
    def test_ceil_div_always_rounds_up(self, numerator, denominator, expected):
        assert _ceil_div(numerator, denominator) == expected

    def test_a_tariff_needs_a_carrier_id(self):
        with pytest.raises(ValueError, match="carrier_id is required"):
            self.tariff(carrier_id="")

    def test_a_tariff_needs_a_service_id(self):
        with pytest.raises(ValueError, match="service_id is required"):
            self.tariff(service_id="")

    @pytest.mark.parametrize("version", [0, -1])
    def test_a_tariff_version_number_is_positive(self, version):
        with pytest.raises(ValueError, match="version must be positive"):
            self.tariff(version=version)

    def test_an_accessorial_map_key_must_match_its_own_id(self):
        charge = AccessorialCharge("liftgate", flat_charge_minor=1)

        with pytest.raises(ValueError, match="must match its own accessorial_id"):
            self.tariff(accessorials={"lift-gate": charge})

    @staticmethod
    def tariff(**overrides):
        fields = {
            "carrier_id": "acme", "service_id": "ground", "version": 1, "effective_at": 0,
            "dimensional_weight_divisor": 1, "cost_per_dimensional_kg_minor": {"z": 1},
            "minimum_charge_minor": 0, "fuel_surcharge_permille": 0, "accessorials": {},
        }
        fields.update(overrides)
        return Tariff(**fields)


# --------------------------------------------------------------------------------- policy

class TestPolicyGuards:
    def test_a_rule_needs_a_rule_id(self):
        with pytest.raises(ValueError, match="rule_id is required"):
            self.rule(rule_id="")

    @pytest.mark.parametrize("version", [0, -3])
    def test_a_rule_version_number_is_positive(self, version):
        with pytest.raises(ValueError, match="version must be positive"):
            self.rule(version=version)

    def test_a_rejection_without_a_citation_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="must carry a citation"):
            PolicyDecision(scope=PolicyScope.HAZMAT, allowed=False, citation=None)

    def test_an_allow_decision_may_carry_no_citation(self):
        decision = PolicyDecision(scope=PolicyScope.HAZMAT, allowed=True)

        assert decision.citation is None

    def test_a_rejection_with_a_citation_is_fine(self):
        citation = PolicyCitation(rule_id="r", version=1, action=PolicyAction.REJECT,
                                  priority=0, reason="")

        assert PolicyDecision(PolicyScope.HAZMAT, False, citation).citation is citation

    @staticmethod
    def rule(**overrides):
        fields = {
            "rule_id": "r", "version": 1, "scope": PolicyScope.HAZMAT,
            "action": PolicyAction.REJECT, "predicates": (predicate(),),
            "priority": 0, "effective_at": 0,
        }
        fields.update(overrides)
        return PolicyRule(**fields)


# -------------------------------------------------------------------------------- catalog

class TestCatalogGuards:
    def test_a_carton_needs_exactly_three_axes(self):
        with pytest.raises(ValueError, match="exactly three axes"):
            CartonMaster("c", (10, 10), 1)

    def test_an_exclusion_rule_between_two_real_ids_is_accepted(self):
        rule = ExclusionRule("x", ExclusionScope.ITEM_PALLET, "i", "p", reason="hazmat")

        assert rule.scope is ExclusionScope.ITEM_PALLET

    @pytest.mark.parametrize("override,kind", [
        (item("e"), CatalogEntryKind.ITEM),
        (carton("e"), CatalogEntryKind.CARTON),
        (pallet("e"), CatalogEntryKind.PALLET),
    ])
    def test_an_override_derives_its_kind_from_what_it_overrides(self, override, kind):
        facility_override = FacilityOverride("o", "F", "e", override)

        assert facility_override.entry_kind is kind

    @pytest.mark.parametrize("fields,message", [
        ({"number": 0}, "version number must be positive"),
        ({"effective_at": -1}, "effective_at cannot be negative"),
        ({"published_at": -1}, "published_at cannot be negative"),
        ({"rolled_back_from": 0}, "rolled_back_from must reference a positive"),
    ])
    def test_a_malformed_version_cannot_be_constructed(self, fields, message):
        base = {"number": 1, "snapshot": CatalogSnapshot(), "effective_at": 0,
                "published_at": 0}
        base.update(fields)

        with pytest.raises(ValueError, match=message):
            CatalogVersion(**base)

    @pytest.mark.parametrize("fields,message", [
        ({"catalog_id": ""}, "catalog_id is required"),
        ({"version": 0}, "version must be positive"),
        ({"effective_at": -1}, "effective_at cannot be negative"),
        ({"resolved_at": -1}, "resolved_at cannot be negative"),
    ])
    def test_a_malformed_reference_cannot_be_constructed(self, fields, message):
        base = {"catalog_id": "c", "version": 1, "effective_at": 0, "resolved_at": 0}
        base.update(fields)

        with pytest.raises(ValueError, match=message):
            CatalogReference(**base)

    def test_a_reference_serializes_to_its_wire_shape(self):
        reference = CatalogReference("c", 2, 10, 20)

        assert reference.as_dict() == {
            "catalog_id": "c", "version": 2, "effective_at": 10, "resolved_at": 20,
        }

    def test_a_registry_needs_a_catalog_id_and_reports_it(self):
        with pytest.raises(ValueError, match="catalog_id is required"):
            CatalogRegistry("")

        assert CatalogRegistry("dc-1").catalog_id == "dc-1"

    def test_every_entry_kind_is_reachable_through_a_resolved_catalog(self):
        snapshot = CatalogSnapshot(items=(item(),), cartons=(carton(),), pallets=(pallet(),))
        registry = CatalogRegistry("dc-1")
        registry.publish(snapshot, effective_at=0, published_at=0)

        resolved = registry.resolve(resolved_at=1, version=1)

        assert resolved.item("i").id == "i"
        assert resolved.carton("c").id == "c"
        assert resolved.pallet("p").id == "p"

    @pytest.mark.parametrize("lookup", ["item", "carton", "pallet"])
    def test_an_absent_entry_is_a_structured_error_not_none(self, lookup):
        resolved = ResolvedCatalog(CatalogReference("c", 1, 0, 0), CatalogSnapshot())

        with pytest.raises(CatalogEntryNotFoundError, match=f"no {lookup} with id 'ghost'"):
            getattr(resolved, lookup)("ghost")


# ------------------------------------------------- guards the document parser bypasses

class TestGuardsOnlyDirectConstructionReaches:
    """Paths a parsed document never takes, because the parser checks first.

    The models are public, so a caller can reach these without a document at all. Each
    one is a place where the model — not the parser — is the last line of defence.
    """

    @pytest.mark.parametrize("dimensions", [(10, 10), (10, 10, 10, 10), ()])
    def test_an_item_needs_exactly_three_axes(self, dimensions):
        with pytest.raises(ValueError, match="exactly three axes"):
            ItemMaster("i", dimensions, 1)

    def test_a_pallet_needs_exactly_two_deck_axes(self):
        with pytest.raises(ValueError, match="exactly two axes"):
            PalletMaster("p", (10, 10, 10), 1)

    @pytest.mark.parametrize("fields,message", [
        ({"max_payload_g": 0}, "max_payload_g must be positive"),
        ({"cost_minor": -1}, "cost_minor cannot be negative"),
    ])
    def test_a_cartons_payload_and_cost_are_range_checked(self, fields, message):
        base = {"id": "c", "inner_dimensions_mm": (10, 10, 10), "max_payload_g": 1}
        base.update(fields)

        with pytest.raises(ValueError, match=message):
            CartonMaster(**base)

    @pytest.mark.parametrize("scope,operator,message", [
        ("warehouse", PolicyOperator.EQUALS, "unsupported policy scope"),
        (PolicyScope.HAZMAT, "contains", "unsupported policy operator"),
    ])
    def test_a_predicate_built_from_raw_strings_still_fails_admission(
        self, scope, operator, message,
    ):
        # A predicate deserialized straight off a wire payload arrives as strings; the
        # model coerces them to enum members and refuses anything outside the vocabulary.
        with pytest.raises(UnsupportedPredicateError, match=message):
            PolicyPredicate(scope=scope, field="f", operator=operator, value=1)

    def test_a_predicate_may_be_built_from_valid_raw_strings(self):
        built = PolicyPredicate(scope="hazmat", field="f", operator="equals", value=1)

        assert built.scope is PolicyScope.HAZMAT
        assert built.operator is PolicyOperator.EQUALS

    def test_a_predicate_with_a_hand_forced_operator_is_refused_at_evaluation(self):
        # `matches` re-checks rather than trusting construction, because a frozen
        # dataclass can still be bypassed with object.__setattr__.
        built = PolicyPredicate(scope=PolicyScope.HAZMAT, field="f",
                                operator=PolicyOperator.EQUALS, value=1)
        object.__setattr__(built, "operator", "contains")

        with pytest.raises(UnsupportedPredicateError, match="unsupported policy operator"):
            built.matches({"f": 1})


class TestHistoricalReplay:
    def test_a_recorded_reference_replays_to_the_data_it_was_made_against(self):
        registry = CatalogRegistry("dc-1")
        registry.publish(CatalogSnapshot(items=(item("old"),)), effective_at=0, published_at=0)
        reference = registry.resolve(resolved_at=5, version=1).reference
        registry.publish(CatalogSnapshot(items=(item("new"),)), effective_at=10, published_at=10)

        replayed = registry.resolve_reference(reference, resolved_at=99)

        assert [entry.id for entry in replayed.snapshot.items] == ["old"]
        assert replayed.reference.resolved_at == 99

    def test_a_reference_from_another_catalog_is_refused(self):
        registry = CatalogRegistry("dc-1")
        registry.publish(CatalogSnapshot(), effective_at=0, published_at=0)
        foreign = CatalogReference("dc-2", 1, 0, 0)

        with pytest.raises(CatalogError, match="reference is for catalog 'dc-2'"):
            registry.resolve_reference(foreign, resolved_at=1)

    def test_an_empty_catalog_has_no_version_to_resolve(self):
        with pytest.raises(CatalogVersionNotFoundError, match="no published versions"):
            CatalogRegistry("dc-1").resolve(resolved_at=1)


class TestLookupWalksPastNonMatches:
    """A snapshot lookup is a scan, not a dictionary. Every earlier test happened to ask
    for the first entry, which is the one arrangement that never exercises the skip."""

    @pytest.mark.parametrize(
        ("kind", "snapshot", "wanted"),
        [
            ("item", CatalogSnapshot(items=(item("a"), item("b"), item("c"))), "c"),
            ("carton", CatalogSnapshot(cartons=(carton("a"), carton("b"))), "b"),
            ("pallet", CatalogSnapshot(pallets=(pallet("a"), pallet("b"))), "b"),
        ],
    )
    def test_the_last_entry_is_found_after_skipping_the_others(self, kind, snapshot, wanted):
        assert getattr(snapshot, kind)(wanted).id == wanted

    @pytest.mark.parametrize("kind", ["item", "carton", "pallet"])
    def test_a_full_scan_that_matches_nothing_names_the_kind(self, kind):
        snapshot = CatalogSnapshot(
            items=(item("a"), item("b")),
            cartons=(carton("a"), carton("b")),
            pallets=(pallet("a"), pallet("b")),
        )

        with pytest.raises(CatalogEntryNotFoundError, match=f"no {kind} with id 'z'"):
            getattr(snapshot, kind)("z")


class TestPublishRevalidatesSmuggledPredicates:
    """`PolicyPredicate.__post_init__` is the first line of defence, and `publish` is
    documented as a deliberate second one. A predicate whose enum was overwritten after
    construction is the only way to reach it -- and it is exactly what an unpickled or
    hand-patched object looks like."""

    def _smuggled(self, field, value):
        built = PolicyPredicate(
            scope=PolicyScope.HAZMAT, field="f", operator=PolicyOperator.EXISTS
        )
        object.__setattr__(built, field, value)
        return built

    @pytest.mark.parametrize(
        ("field", "value"),
        [("scope", "nowhere"), ("operator", "contains")],
    )
    def test_an_unsupported_scope_or_operator_is_refused_at_publish(self, field, value):
        registry = PolicyRegistry()

        with pytest.raises(UnsupportedPredicateError, match="unsupported scope/operator"):
            registry.publish(
                "r",
                scope=PolicyScope.HAZMAT,
                action=PolicyAction.REJECT,
                predicates=(self._smuggled(field, value),),
                priority=0,
                effective_at=0,
            )

    def test_a_well_formed_predicate_still_publishes(self):
        registry = PolicyRegistry()

        rule = registry.publish(
            "r",
            scope=PolicyScope.HAZMAT,
            action=PolicyAction.REJECT,
            predicates=(predicate(),),
            priority=0,
            effective_at=0,
        )

        assert rule.version == 1, "the guard must not reject the ordinary case"
