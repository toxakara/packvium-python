"""Approved revisions feed historical evaluation without touching a plan."""

from __future__ import annotations

import copy

import pytest

from packvium.holdout import ValidationVerdict, OrderEvaluationArtifact, evaluate_on_history
from packvium.outcomes import OutcomeEventType, OutcomeLedger
from packvium.recommendations import ExpectedDelta, Recommendation
from packvium.revision_outcomes import record_revision, revision_outcome_events
from packvium.revisions import PlanRevisionError, document_digest
from packvium.serialization import pack_from_dict
from packvium.simulation import OrderRunResult, ScenarioVersionPin

from test_fixed_placements import without_durations
from test_revisions import chain

BASELINE_PIN = ScenarioVersionPin(catalog_version=1)
TREATMENT_PIN = ScenarioVersionPin(catalog_version=2)


def test_each_exception_becomes_one_ledger_event_named_by_its_revision():
    revisions, _ = chain()
    events = revision_outcome_events(revisions[1], "order-1", 500)
    digest = document_digest(revisions[1])
    assert [event.event_id for event in events] == [f"order-1@{digest}#1", f"order-1@{digest}#2"]
    assert [event.event_type for event in events] == [OutcomeEventType.OPERATOR_OVERRIDE,
                                                       OutcomeEventType.ITEM_MISSING]
    assert events[1].payload == {"item_type": "cube", "quantity": 1}


def test_the_root_revision_records_nothing():
    revisions, _ = chain()
    assert revision_outcome_events(revisions[0], "order-1", 500) == ()


def test_a_document_that_is_not_a_revision_is_refused():
    with pytest.raises(PlanRevisionError):
        revision_outcome_events({"events": []}, "order-1", 500)


def test_recording_a_revision_twice_changes_nothing():
    revisions, _ = chain()
    ledger = OutcomeLedger()
    first = record_revision(ledger, revisions[2], "order-1", 500)
    again = record_revision(ledger, revisions[2], "order-1", 500)
    assert first == again
    assert len(ledger.events_for_decision("order-1")) == 1


def test_recording_never_changes_the_revision_or_its_plan():
    revisions, _ = chain()
    before = copy.deepcopy(revisions)
    record_revision(OutcomeLedger(), revisions[1], "order-1", 500)
    assert revisions == before


def test_a_locked_placement_is_reported_against_the_baseline_in_holdout():
    revisions, _ = chain()
    ledger = OutcomeLedger()
    record_revision(ledger, revisions[1], "trained-1", 100)
    record_revision(ledger, revisions[1], "held-1", 900)
    record_revision(ledger, revisions[2], "held-1", 950)

    def evaluate(decision_id, pin):
        request = revisions[-1]["request"]
        run = OrderRunResult(order_id=decision_id, succeeded=True,
                             metrics={"cost": 10 if pin == BASELINE_PIN else 8})
        return OrderEvaluationArtifact(run=run, request=request, result=pack_from_dict(request))

    recommendation = Recommendation(
        recommendation_id="rec-1", proposal="cheaper cartons",
        supporting_order_ids=("trained-1",),
        expected_deltas=(ExpectedDelta(metric="cost", baseline_mean=10.0, treatment_mean=8.0),),
        confidence=1.0, constraints=("no rejected placement",), rollback_plan="republish")
    evaluation = evaluate_on_history(
        recommendation, ledger, ["trained-1", "held-1"], BASELINE_PIN, TREATMENT_PIN, evaluate,
        lambda request, result: ValidationVerdict(valid=not result["warnings"]),
        {"cost": False}, split_at=800,
    )
    assert evaluation.training_decision_ids == ("trained-1",)
    (decision,) = evaluation.decisions
    assert decision.verdict == "improved"
    assert decision.baseline_realised_risk == ("operator_override",)


def test_the_ledger_cannot_change_a_packing():
    revisions, _ = chain()
    request = revisions[-1]["request"]
    before = pack_from_dict(request)
    record_revision(OutcomeLedger(), revisions[1], "order-1", 500)
    after = pack_from_dict(request)
    assert without_durations(before) == without_durations(after)

