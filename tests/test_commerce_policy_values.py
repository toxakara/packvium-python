"""Policy values must produce the same business decision in every language."""

import pytest

from packvium.commerce import evaluate_policy


def decision(operator, value, context):
    document = {"policy_rules": [{"rule_id": "restriction", "versions": [{
        "scope": "facility", "action": "reject", "priority": 1, "effective_at": 0,
        "predicates": [{"scope": "facility", "field": "x", "operator": operator,
                        "value": value}],
    }]}]}
    return evaluate_policy(document, {"scope": "facility", "context": context, "as_of": 0})["decision"]


@pytest.mark.parametrize("actual", [1, True, None, [], {"a": 1}])
@pytest.mark.parametrize("operator,allowed", [("in", True), ("not_in", False)])
def test_non_string_context_values_are_not_substrings(actual, operator, allowed):
    assert decision(operator, "abc", {"x": actual})["allowed"] is allowed


@pytest.mark.parametrize("operator", ["equals", "not_equals", "in", "not_in"])
def test_missing_context_fields_do_not_match_binary_operators(operator):
    assert decision(operator, {"a": 1}, {})["citation"] is None


@pytest.mark.parametrize("operator,value,actual,allowed", [
    ("equals", {"a": [1, {"flag": True}], "b": None},
     {"b": None, "a": [True, {"flag": 1}]}, False),
    ("not_equals", {"a": None}, {"b": None}, False),
    ("equals", {"a": None}, {"b": None}, True),
    ("equals", {"a": [1, 2]}, {"a": [2, 1]}, True),
    ("in", [{"a": 1, "b": True}], {"b": 1, "a": True}, False),
    ("not_in", [{"a": 1}], {"a": 2}, False),
    ("in", {"a": None}, "a", False),
    ("in", {"a": "b"}, "b", True),
    ("not_in", {"a": 1}, "missing", False),
    ("in", {"a": 1}, ["a"], True),
    ("in", {"a": 1}, {"a": 1}, True),
])
def test_structured_values_follow_reference_semantics(operator, value, actual, allowed):
    result = decision(operator, value, {"x": actual})
    assert result["allowed"] is allowed
    assert (result["citation"] is None) is allowed
