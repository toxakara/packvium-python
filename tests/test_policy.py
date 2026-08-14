"""Versioned eligibility rules compiled into the constraint pipeline.

Every assertion here is about a *packing*, not about a parsed object: a rule that is
resolved correctly and then never consulted is indistinguishable, from the outside, from
one that was ignored. Each test therefore packs the same request twice — once with the
rule participating and once without — and asserts the two answers differ in the way the
rule says they should.
"""

from __future__ import annotations

import copy

import pytest

from packvium.explain import explain_reason
from packvium.policy import PolicyError, PolicyRuleSet, ShipmentContext
from packvium.serialization import pack_from_dict

AS_OF = 1_704_067_200_000

REQUEST = {
    "units": {"length": "mm"},
    "policy": {
        "as_of": AS_OF,
        "shipment": {"facility": "SEA1", "carrier": "ups"},
        "rules": [
            {
                "id": "hazmat-food-segregation",
                "version": 1,
                "effective_at": AS_OF,
                "priority": 100,
                "separate_tags": {"tag": "hazmat", "from_tag": "food"},
            },
        ],
    },
    "items": [
        {"id": "drum", "quantity": 1, "dimensions": {"length": "100", "width": "100", "height": "100"},
         "tags": ["hazmat"]},
        {"id": "carton", "quantity": 1, "dimensions": {"length": "100", "width": "100", "height": "100"},
         "tags": ["food"]},
    ],
    "containers": [
        {"id": "pallet", "quantity": 4,
         "inner_dimensions": {"length": "300", "width": "300", "height": "300"}},
    ],
}


def containers_used(request: dict) -> int:
    return pack_from_dict(request)["summary"]["container_count"]


def without_policy(request: dict) -> dict:
    stripped = copy.deepcopy(request)
    stripped.pop("policy")
    return stripped


def test_a_segregation_rule_opens_a_container_the_geometry_did_not_need() -> None:
    # Both items fit in one pallet by geometry and weight alone, so a second container is
    # the rule's doing and nothing else's.
    assert containers_used(without_policy(REQUEST)) == 1
    assert containers_used(copy.deepcopy(REQUEST)) == 2


def test_a_container_tag_rule_leaves_an_item_behind_rather_than_routing_it_wrongly() -> None:
    request = copy.deepcopy(REQUEST)
    request["policy"]["rules"] = [{
        "id": "cold-chain", "version": 1, "effective_at": AS_OF, "priority": 10,
        "require_container_tag": {"item_tag": "food", "container_tag": "reefer"},
    }]
    result = pack_from_dict(request)
    assert [item["item_type"] for item in result["unpacked_items"]] == ["carton"]

    # The same request against a container that carries the tag packs everything.
    request["containers"][0]["tags"] = ["reefer"]
    assert pack_from_dict(request)["summary"]["unpacked_item_count"] == 0


def test_a_tag_cap_splits_a_container_the_cap_would_otherwise_overfill() -> None:
    request = copy.deepcopy(REQUEST)
    request["items"] = [{
        "id": "cell", "quantity": 3,
        "dimensions": {"length": "100", "width": "100", "height": "100"}, "tags": ["lithium"],
    }]
    request["policy"]["rules"] = [{
        "id": "lithium-cap", "version": 1, "effective_at": AS_OF, "priority": 10,
        "limit_tag_per_container": {"tag": "lithium", "max": 2},
    }]
    assert containers_used(without_policy(request)) == 1
    assert containers_used(request) == 2


def test_a_rule_that_is_not_yet_effective_does_not_participate() -> None:
    request = copy.deepcopy(REQUEST)
    request["policy"]["as_of"] = AS_OF - 1
    assert containers_used(request) == 1


def test_a_rule_scoped_to_another_shipment_does_not_participate() -> None:
    for fact, value in (("facility", "PDX9"), ("carrier", "fedex")):
        request = copy.deepcopy(REQUEST)
        request["policy"]["rules"][0]["applies_to"] = {fact: value}
        assert containers_used(request) == 1, fact
        # ...and the same rule scoped to the fact the shipment *does* declare applies.
        request["policy"]["rules"][0]["applies_to"] = {fact: request["policy"]["shipment"][fact]}
        assert containers_used(request) == 2, fact


def test_a_rule_scoped_to_a_fact_the_shipment_never_declared_does_not_participate() -> None:
    # An undeclared fact is not a wildcard. Treating it as one would let a rule written
    # for one customer silently apply to a shipment that never named a customer at all.
    request = copy.deepcopy(REQUEST)
    request["policy"]["rules"][0]["applies_to"] = {"customer": "acme"}
    assert containers_used(request) == 1


def test_the_highest_effective_version_of_one_id_wins() -> None:
    # Append-only per id: the later version replaces the earlier one rather than both
    # being enforced, which is what makes a published rule set replayable.
    request = copy.deepcopy(REQUEST)
    superseded = copy.deepcopy(request["policy"]["rules"][0])
    superseded["version"] = 2
    superseded["effective_at"] = AS_OF
    superseded["separate_tags"] = {"tag": "hazmat", "from_tag": "nothing-here"}
    request["policy"]["rules"].append(superseded)
    assert containers_used(request) == 1

    # And a superseding version that is not yet effective leaves the older one in force.
    request["policy"]["rules"][1]["effective_at"] = AS_OF + 1
    assert containers_used(request) == 2


def test_resolution_orders_rules_by_priority_then_id_never_by_declaration_order() -> None:
    rules = [
        {"id": "b", "version": 1, "effective_at": 0, "priority": 5,
         "separate_tags": {"tag": "x", "from_tag": "y"}},
        {"id": "a", "version": 1, "effective_at": 0, "priority": 5,
         "separate_tags": {"tag": "x", "from_tag": "y"}},
        {"id": "c", "version": 1, "effective_at": 0, "priority": 9,
         "separate_tags": {"tag": "x", "from_tag": "y"}},
    ]
    resolved = PolicyRuleSet.from_dict({"as_of": 0, "rules": rules}).rules
    assert [rule.id for rule in resolved] == ["c", "a", "b"]
    # Declaring them in the opposite order must not move them.
    reversed_order = PolicyRuleSet.from_dict({"as_of": 0, "rules": list(reversed(rules))}).rules
    assert [rule.id for rule in reversed_order] == ["c", "a", "b"]


def test_an_undeclared_shipment_fact_is_absent_rather_than_matching_anything() -> None:
    assert ShipmentContext(facility="SEA1").satisfied_by(ShipmentContext(facility="SEA1"))
    assert not ShipmentContext(facility="SEA1").satisfied_by(ShipmentContext())
    assert ShipmentContext().satisfied_by(ShipmentContext(facility="SEA1"))


@pytest.mark.parametrize("rule, expected", [
    ({"id": "", "version": 1, "effective_at": 0, "priority": 0,
      "separate_tags": {"tag": "a", "from_tag": "b"}}, "id must be a non-empty string"),
    ({"id": "r", "version": 0, "effective_at": 0, "priority": 0,
      "separate_tags": {"tag": "a", "from_tag": "b"}}, "version must be an integer >= 1"),
    ({"id": "r", "version": True, "effective_at": 0, "priority": 0,
      "separate_tags": {"tag": "a", "from_tag": "b"}}, "version must be an integer >= 1"),
    ({"id": "r", "version": 1, "effective_at": -1, "priority": 0,
      "separate_tags": {"tag": "a", "from_tag": "b"}}, "effective_at must be an integer >= 0"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0}, "exactly one rule form"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "separate_tags": {"tag": "a", "from_tag": "b"},
      "limit_tag_per_container": {"tag": "a", "max": 1}}, "exactly one rule form"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "separate_tags": {"tag": "a"}}, "is missing from_tag"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "separate_tags": {"tag": "a", "from_tag": "b", "extra": 1}}, "unknown keys: extra"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "limit_tag_per_container": {"tag": "a", "max": -1}}, "max must be an integer >= 0"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "applies_to": {"region": "eu"}, "separate_tags": {"tag": "a", "from_tag": "b"}},
     "unknown shipment facts: region"),
    # A tag is matched against item and container tags by identity, so a number or an
    # empty string would silently never match instead of failing loudly.
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "separate_tags": {"tag": 5, "from_tag": "b"}}, "tag must be a non-empty string"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "applies_to": {"facility": ""}, "separate_tags": {"tag": "a", "from_tag": "b"}},
     "facility must be a non-empty string"),
    # Wrong JSON shape rather than wrong value: each of these arrives from the wire, and
    # a string where an object belongs must name the place it was found.
    ("not-an-object", "must be an object"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "separate_tags": "a,b"}, "separate_tags must be an object"),
    ({"id": "r", "version": 1, "effective_at": 0, "priority": 0,
      "applies_to": "SEA1", "separate_tags": {"tag": "a", "from_tag": "b"}},
     "applies_to must be an object"),
])
def test_a_malformed_rule_fails_admission_rather_than_being_dropped(rule: dict, expected: str) -> None:
    # A rule quietly dropped for being malformed would let a request pack in a way its
    # own policy forbids -- the failure the whole contract exists to prevent.
    with pytest.raises(PolicyError, match=expected):
        PolicyRuleSet.from_dict({"as_of": 0, "rules": [rule]})


def test_rules_without_an_as_of_fail_admission() -> None:
    rule = {"id": "r", "version": 1, "effective_at": 0, "priority": 0,
            "separate_tags": {"tag": "a", "from_tag": "b"}}
    with pytest.raises(PolicyError, match="as_of is required"):
        PolicyRuleSet.from_dict({"rules": [rule]})
    # An empty rule set needs no instant, because nothing can be dated.
    assert PolicyRuleSet.from_dict({"rules": []}).rules == ()
    assert PolicyRuleSet.from_dict(None).rules == ()


def test_an_unknown_policy_key_fails_admission() -> None:
    with pytest.raises(PolicyError, match="unknown keys: effect"):
        PolicyRuleSet.from_dict({"as_of": 0, "rules": [], "effect": "deny"})


SHAPED_RULE = {"id": "r", "version": 1, "effective_at": 0, "priority": 0,
               "separate_tags": {"tag": "a", "from_tag": "b"}}


@pytest.mark.parametrize("raw, expected", [
    ("deny-everything", "policy must be an object"),
    ({"as_of": 0, "rules": {"id": "r"}}, "policy.rules must be an array"),
    ({"as_of": 0, "shipment": "SEA1", "rules": [SHAPED_RULE]}, "policy.shipment must be an object"),
])
def test_a_policy_block_of_the_wrong_shape_fails_admission(raw: object, expected: str) -> None:
    with pytest.raises(PolicyError, match=expected):
        PolicyRuleSet.from_dict(raw)


def test_an_empty_rule_set_does_not_reach_the_shipment_context() -> None:
    # Recorded because it is a real asymmetry, not an oversight to trip over later: with
    # no rules the block returns before `shipment` is parsed, so a malformed context is
    # accepted there. Harmless -- nothing can consult it -- but `policy`'s own unknown
    # keys *are* still checked first, so the two guards do not run at the same depth.
    assert PolicyRuleSet.from_dict({"as_of": 0, "shipment": "SEA1", "rules": []}).rules == ()
    with pytest.raises(PolicyError, match="unknown keys"):
        PolicyRuleSet.from_dict({"as_of": 0, "shipment": "SEA1", "rules": [], "effect": "deny"})


# ---------------------------------------------------------------- citation


def unpacked_by_policy(request: dict) -> dict:
    result = pack_from_dict(request)
    return next(item for item in result["unpacked_items"] if item["reason"] == "policy_rule")


def test_a_rejection_names_the_rule_and_version_that_caused_it() -> None:
    request = copy.deepcopy(REQUEST)
    request["policy"]["rules"] = [{
        "id": "cold-chain", "version": 7, "effective_at": AS_OF, "priority": 10,
        "require_container_tag": {"item_tag": "food", "container_tag": "reefer"},
    }]
    item = unpacked_by_policy(request)
    assert item["item_type"] == "carton"
    # Version, not just id: replaying a past decision needs to know which text was in
    # force, and two versions of one rule can forbid different things.
    assert item["details"] == [
        "cold-chain@7: requires a container tagged 'reefer', "
        "which none of the containers offered carries"
    ]
    # Proven rather than observed: no search outcome can make the item placeable, so
    # this is a fact about the request.
    assert item["proof"]["level"] == "proven"
    assert item["proof"]["observations"][0]["code"] == "policy_rule"
    assert "policy rule" in explain_reason("policy_rule")


def test_a_rule_the_search_worked_around_is_not_reported_as_proven() -> None:
    # Segregation left nothing behind here -- it opened a second container. Reporting a
    # policy rejection would name a rule that did not reject anything.
    result = pack_from_dict(copy.deepcopy(REQUEST))
    assert result["summary"]["unpacked_item_count"] == 0
    assert result["summary"]["container_count"] == 2


def test_a_cap_that_leaves_an_item_behind_is_observed_not_proven() -> None:
    # A per-container cap depends on what else was packed, so an item it leaves behind
    # was left behind by the search. Claiming `proven` would assert more than the engine
    # knows, so the generic observed reason stands and no rule is cited.
    request = copy.deepcopy(REQUEST)
    request["items"] = [{
        "id": "cell", "quantity": 3,
        "dimensions": {"length": "100", "width": "100", "height": "100"}, "tags": ["lithium"],
    }]
    request["containers"] = [{
        "id": "pallet", "quantity": 1,
        "inner_dimensions": {"length": "300", "width": "300", "height": "300"},
    }]
    request["policy"]["rules"] = [{
        "id": "lithium-cap", "version": 1, "effective_at": AS_OF, "priority": 10,
        "limit_tag_per_container": {"tag": "lithium", "max": 2},
    }]
    result = pack_from_dict(request)
    assert result["summary"]["unpacked_item_count"] == 1
    assert {item["reason"] for item in result["unpacked_items"]} != {"policy_rule"}


def test_a_routing_rule_the_item_does_not_match_cites_nothing() -> None:
    # The rule participates and is a routing rule, but this item does not carry its
    # item_tag, so it cannot be the reason anything was left behind. Skipping it rather
    # than citing it is what keeps a citation from naming an innocent policy.
    request = copy.deepcopy(REQUEST)
    request["items"] = [{
        "id": "oversized", "quantity": 1,
        "dimensions": {"length": "9000", "width": "9000", "height": "9000"}, "tags": ["dry"],
    }]
    request["policy"]["rules"] = [{
        "id": "cold-chain", "version": 1, "effective_at": AS_OF, "priority": 10,
        "require_container_tag": {"item_tag": "frozen", "container_tag": "reefer"},
    }]
    result = pack_from_dict(request)
    assert [item["reason"] for item in result["unpacked_items"]] == [
        "no_compatible_container_dimensions"
    ]
    assert all(item["details"] == [] for item in result["unpacked_items"])


def test_geometry_outranks_policy_in_the_reported_reason() -> None:
    # An item too big for every container is impossible whatever a policy says, and
    # reporting the policy first would send a caller to fix the wrong thing.
    request = copy.deepcopy(REQUEST)
    request["items"] = [{
        "id": "slab", "quantity": 1,
        "dimensions": {"length": "9000", "width": "9000", "height": "9000"}, "tags": ["food"],
    }]
    request["policy"]["rules"] = [{
        "id": "cold-chain", "version": 1, "effective_at": AS_OF, "priority": 10,
        "require_container_tag": {"item_tag": "food", "container_tag": "reefer"},
    }]
    result = pack_from_dict(request)
    assert [item["reason"] for item in result["unpacked_items"]] == [
        "no_compatible_container_dimensions"
    ]
