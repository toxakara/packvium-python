"""The reference validator gives its recorded verdict on every pair of the shared corpus.

`conformance/result_validation/corpus.json` (written by
`scripts/generate_result_validation_corpus.py`) is the subset of the cross-language
result-validation corpus that reaches every arc, issue code and refusal kind the full corpus
reaches in this module, each pair with the outcome `conformance/result_validation/run.py`
recorded for it. Reading the documents from their JSON text, as the harness does, this module
must answer each one the same way: the same issues (code and detail, in order), or the same
refusal. A change to a verdict is a change to the contract every port is held to, so the corpus
is regenerated with it, never edited by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from packvium import MalformedResult, validate

CORPUS = Path(__file__).resolve().parents[2] / "conformance" / "result_validation" / "corpus.json"

pytestmark = pytest.mark.skipif(not CORPUS.is_file(), reason="the corpus lives in the workspace only")


def _corpus() -> list[dict[str, Any]]:
    decoded = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert decoded["format"] == "packvium-result-validation-corpus/v1"
    cases: list[dict[str, Any]] = decoded["cases"]
    return cases


def _outcome(request_text: str, result_text: str) -> dict[str, Any]:
    """The harness's `reference_outcome`, so both read the corpus the same way."""
    try:
        request, result = json.loads(request_text), json.loads(result_text)
        return {"issues": [{"code": issue.code, "detail": str(issue.detail)} for issue in validate(request, result)]}
    except MalformedResult as refusal:
        return {"refusal": {"kind": "contract", "message": str(refusal)}}
    except Exception as failure:  # noqa: BLE001 -- any other failure is the reference refusing
        return {"refusal": {"kind": "shape", "message": f"{type(failure).__name__}: {failure}"}}


def test_the_corpus_reaches_every_issue_code_the_module_can_raise() -> None:
    module = Path(validate.__code__.co_filename).read_text(encoding="utf-8")
    raisable = set(re.findall(r'Issue\(\s*"([a-z_0-9]+)"', module))
    raised = {issue["code"] for case in _corpus() for issue in case["expected"].get("issues", ())}
    assert raisable - raised == set()


def test_every_pair_gets_its_recorded_verdict() -> None:
    disagreements = [
        (index, case["expected"], outcome)
        for index, case in enumerate(_corpus())
        if (outcome := _outcome(case["request"], case["result"])) != case["expected"]
    ]
    assert disagreements == []
