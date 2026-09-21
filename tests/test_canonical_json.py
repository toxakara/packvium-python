"""RFC 8785 canonical JSON, the spelling the execution plan and the operational artifact share.

What is checked is where the language defaults disagree, because that is where a
cross-language byte comparison fails: integral floats, exponents, key order outside the
Basic Multilingual Plane, the characters JSON writers escape differently, and the values no
engine can carry exactly. Where Node is installed its own `JSON.stringify` is the oracle for
numbers, since RFC 8785 defines them as ECMAScript writes them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from packvium._canonical_json import CanonicalJsonError, canonical_json

NUMBERS = [
    (0.25, "0.25"), (1.0, "1"), (-0.0, "0"), (4.5, "4.5"), (0.1, "0.1"), (2e-3, "0.002"),
    (0.000001, "0.000001"), (1e-7, "1e-7"), (1.5e-7, "1.5e-7"), (1e-27, "1e-27"), (-1.5, "-1.5"),
    (123456789.5, "123456789.5"), (333333333.33333329, "333333333.3333333"),
    (9007199254740991.0, "9007199254740991"), (5e-324, "5e-324"), (100.0, "100"),
]


@pytest.mark.parametrize(("value", "spelled"), NUMBERS)
def test_numbers_are_written_as_ecmascript_writes_them(value, spelled):
    assert canonical_json(value) == spelled


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_number_matches_javascript_itself():
    values = [value for value, _ in NUMBERS]
    script = "process.stdout.write(JSON.stringify(JSON.parse(require('fs').readFileSync(0,'utf8'))))"
    completed = subprocess.run(["node", "-e", script], input=json.dumps(values), capture_output=True,
                               text=True, check=True)
    assert completed.stdout == canonical_json(values)


def test_integers_and_literals_keep_their_spelling():
    assert canonical_json([0, -7, 9007199254740991, True, False, None]) == "[0,-7,9007199254740991,true,false,null]"


def test_keys_are_sorted_by_utf16_code_units_not_code_points():
    """U+FFFD sorts after U+1F600 by code point and before it by UTF-16 code unit, whose
    first unit is the high surrogate U+D83D. JavaScript sorts the second way."""
    assert canonical_json({"�": 1, "\U0001F600": 2, "a": 0}) == '{"a":0,"\U0001F600":2,"�":1}'


def test_only_the_characters_rfc_8785_names_are_escaped():
    text = "\"\\\b\t\n\f\r\x00\x1f/\x7f  é"
    assert canonical_json(text) == '"\\"\\\\\\b\\t\\n\\f\\r\\u0000\\u001f/\x7f  é"'


def test_there_is_no_whitespace_and_nesting_is_preserved():
    assert canonical_json({"b": [1, {"d": [], "c": {}}], "a": "x"}) == '{"a":"x","b":[1,{"c":{},"d":[]}]}'


@pytest.mark.parametrize("value", [2**53, -(2**53), 9007199254740992.0, float("inf"), float("nan")])
def test_a_number_no_engine_holds_exactly_is_refused(value):
    with pytest.raises(CanonicalJsonError) as refused:
        canonical_json({"n": value})
    assert refused.value.code == "number_out_of_range"


def test_a_lone_surrogate_is_refused_in_a_value_and_in_a_key():
    for value in ("\ud800", {"\udc00": 1}):
        with pytest.raises(CanonicalJsonError) as refused:
            canonical_json(value)
        assert refused.value.code == "invalid_string"


@pytest.mark.parametrize("value", [{1: "a"}, {"a": {1, 2}}, b"bytes"])
def test_a_value_json_cannot_spell_is_refused(value):
    with pytest.raises(CanonicalJsonError) as refused:
        canonical_json(value)
    assert refused.value.code == "invalid_value"


GOLDEN = Path(__file__).resolve().parents[2] / "conformance" / "golden"


@pytest.mark.skipif(not GOLDEN.is_dir(), reason="the golden corpus lives in the workspace only")
def test_no_plan_an_engine_emits_changed_its_bytes_when_the_spelling_became_rfc_8785():
    """1.2.0 published `json.dumps(sort_keys=True)` plans. For every golden result the two
    spellings are the same bytes, so the switch fixes only inputs the adapters disagreed on."""
    from packvium.execution import build_execution_plan, canonical_plan_json

    for golden in sorted(GOLDEN.glob("*.json")):
        result = json.loads(golden.read_text())
        if not isinstance(result, dict) or "status" not in result:
            continue
        plan = build_execution_plan({}, result)
        assert canonical_plan_json(plan) == json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False), golden.name
