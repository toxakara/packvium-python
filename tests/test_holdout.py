"""Replaying a recommendation against history it has not seen.

`tests/` ships inside the wheel, so these import `packvium.holdout` the way a consumer
does rather than through the workspace shim.

Two rules are the reason this module exists rather than a spreadsheet. A packing the
injected validator rejects is that arm's failure **at any cost** — its metrics never reach
the comparison. And a recommendation may not be scored on the evidence that produced it,
which is refused rather than warned about, because a holdout score is worth exactly as
much as that separation.
"""

from __future__ import annotations

import pytest

from packvium.holdout import (
    IMPROVED,
    REGRESSED,
    TRADED_OFF,
    UNPACKABLE,
    HoldoutError,
    OrderEvaluationArtifact,
    SupportingEvidenceInHoldoutError,
    ValidationVerdict,
    evaluate_on_history,
)
from packvium.outcomes import OutcomeEvent, OutcomeEventType, OutcomeLedger
from packvium.recommendations import ExpectedDelta, Recommendation
from packvium.simulation import OrderRunResult, ScenarioVersionPin

BASELINE_PIN = ScenarioVersionPin(catalog_version=1)
TREATMENT_PIN = ScenarioVersionPin(catalog_version=2)
DIRECTIONS = {"cost": False, "utilisation": True}


def recommendation(supporting=("trained-1",)):
    return Recommendation(
        recommendation_id="rec-1", proposal="cheaper cartons",
        supporting_order_ids=tuple(supporting),
        expected_deltas=(ExpectedDelta(metric="cost", baseline_mean=10.0,
                                       treatment_mean=8.0),),
        confidence=1.0, constraints=("no rejected placement",),
        rollback_plan="republish version 1")


def ledger_with(*events):
    """`events` are `(decision_id, recorded_at)` shipments, plus optional event types."""
    ledger = OutcomeLedger()
    for index, event in enumerate(events, start=1):
        decision_id, at = event[0], event[1]
        kind = event[2] if len(event) > 2 else OutcomeEventType.ACTUAL_CARTON
        payload = ({"carton_id": "box-a", "catalog_version": 1}
                   if kind is OutcomeEventType.ACTUAL_CARTON
                   else {"reason_code": "crushed_corner", "severity": "minor"})
        ledger.record(OutcomeEvent(event_id=f"e{index}", decision_id=decision_id,
                                   event_type=kind, payload=payload, recorded_at=at))
    return ledger


def replay(table, invalid=()):
    """An evaluator over a `{decision_id: {arm: (cost, utilisation)}}` table.

    `invalid` names `(decision_id, arm)` pairs the validator should reject.
    """
    def evaluate(decision_id: str, pin: ScenarioVersionPin) -> OrderEvaluationArtifact:
        arm = "baseline" if pin == BASELINE_PIN else "treatment"
        cost, utilisation = table[decision_id][arm]
        run = OrderRunResult(order_id=decision_id, succeeded=True,
                             metrics={"cost": cost, "utilisation": utilisation})
        return OrderEvaluationArtifact(run=run, request={"decision_id": decision_id},
                                       result={"arm": arm})

    def validator(request, result) -> ValidationVerdict:
        if (request["decision_id"], result["arm"]) in invalid:
            return ValidationVerdict(valid=False, codes=("unsupported_item",))
        return ValidationVerdict(valid=True)

    return evaluate, validator


def evaluate(table, *, ledger, invalid=(), decision_ids, split_at=500, rec=None):
    evaluator, validator = replay(table, invalid)
    return evaluate_on_history(
        rec or recommendation(), ledger, decision_ids=decision_ids,
        baseline_pin=BASELINE_PIN, treatment_pin=TREATMENT_PIN,
        evaluator=evaluator, validator=validator, higher_is_better=DIRECTIONS,
        split_at=split_at)


class TestTheSplitIsByRecordedTime:
    def test_decisions_recorded_before_the_split_are_training(self):
        result = evaluate({"trained-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.8)},
                           "held-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.8)}},
                          ledger=ledger_with(("trained-1", 100), ("held-1", 900)),
                          decision_ids=("trained-1", "held-1"))
        assert result.training_decision_ids == ("trained-1",)
        assert result.holdout_decision_ids == ("held-1",)

    def test_only_held_out_decisions_are_scored(self):
        result = evaluate({"trained-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.8)},
                           "held-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.8)}},
                          ledger=ledger_with(("trained-1", 100), ("held-1", 900)),
                          decision_ids=("trained-1", "held-1"))
        assert [d.decision_id for d in result.decisions] == ["held-1"]

    def test_the_split_uses_the_earliest_event_for_a_decision(self):
        # A later correction must not drag a decision across the split it was recorded on
        # the other side of.
        ledger = ledger_with(("held-1", 100), ("held-1", 900, OutcomeEventType.DAMAGE))
        result = evaluate({"held-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.8)}},
                          ledger=ledger, decision_ids=("held-1",), split_at=500,
                          rec=recommendation(supporting=("other",)))
        assert result.training_decision_ids == ("held-1",)
        assert result.decisions == ()

    def test_a_decision_with_no_recorded_events_is_refused(self):
        """It has no timestamp, so it cannot be placed on either side. Defaulting it
        would be this module guessing."""
        with pytest.raises(ValueError, match="no recorded events"):
            evaluate({}, ledger=OutcomeLedger(), decision_ids=("never-shipped",))

    def test_a_negative_split_is_refused(self):
        with pytest.raises(ValueError, match="split_at cannot be negative"):
            evaluate({}, ledger=ledger_with(("held-1", 900)), decision_ids=("held-1",),
                     split_at=-1)

    def test_an_empty_decision_list_is_refused(self):
        # An empty holdout would report zero regressions and read as a clean bill.
        with pytest.raises(ValueError, match="at least one decision id"):
            evaluate({}, ledger=OutcomeLedger(), decision_ids=())


class TestAProposalCannotBeScoredOnItsOwnEvidence:
    def test_a_cited_decision_in_the_holdout_is_refused_by_name(self):
        with pytest.raises(SupportingEvidenceInHoldoutError, match="trained-1"):
            evaluate({"trained-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.8)}},
                     ledger=ledger_with(("trained-1", 900)),
                     decision_ids=("trained-1",), split_at=500)

    def test_the_same_decision_on_the_training_side_is_fine(self):
        result = evaluate({"trained-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.8)}},
                          ledger=ledger_with(("trained-1", 100)),
                          decision_ids=("trained-1",), split_at=500)
        assert result.training_decision_ids == ("trained-1",)

    def test_the_refusal_is_a_holdout_error(self):
        assert issubclass(SupportingEvidenceInHoldoutError, HoldoutError)


class TestTheVerdictPerDecision:
    def _verdict(self, baseline, treatment, invalid=()):
        result = evaluate({"held-1": {"baseline": baseline, "treatment": treatment}},
                          ledger=ledger_with(("held-1", 900)), invalid=invalid,
                          decision_ids=("held-1",))
        return result.decisions[0]

    def test_better_on_every_axis_is_an_improvement(self):
        assert self._verdict((10.0, 0.7), (8.0, 0.9)).verdict == IMPROVED

    def test_worse_on_every_axis_is_a_regression(self):
        assert self._verdict((8.0, 0.9), (10.0, 0.7)).verdict == REGRESSED

    def test_cheaper_but_sparser_is_a_trade_off_rather_than_a_win(self):
        assert self._verdict((10.0, 0.9), (8.0, 0.7)).verdict == TRADED_OFF

    def test_identical_metrics_are_a_trade_off_rather_than_a_win(self):
        # Neither dominates, so neither is named; that is the report saying "no difference".
        assert self._verdict((10.0, 0.8), (10.0, 0.8)).verdict == TRADED_OFF

    def test_a_rejected_treatment_is_a_regression_however_cheap_it_was(self):
        """The rule the module exists to enforce. The treatment here is cheaper *and*
        denser, and the validator refused the placement, so its metrics are discarded
        rather than discounted."""
        outcome = self._verdict((16.0, 0.6), (10.0, 0.9), invalid=(("held-1", "treatment"),))
        assert outcome.verdict == REGRESSED
        assert outcome.treatment_valid is False
        assert outcome.treatment_codes == ("unsupported_item",)

    def test_a_rejected_baseline_leaves_a_valid_treatment_an_improvement(self):
        outcome = self._verdict((8.0, 0.9), (10.0, 0.7), invalid=(("held-1", "baseline"),))
        assert outcome.verdict == IMPROVED
        assert outcome.baseline_valid is False

    def test_both_arms_rejected_is_unpackable_rather_than_a_tie(self):
        outcome = self._verdict((10.0, 0.7), (8.0, 0.9),
                                invalid=(("held-1", "baseline"), ("held-1", "treatment")))
        assert outcome.verdict == UNPACKABLE


class TestRealisedRiskIsReportedAgainstTheArmThatShipped:
    def test_recorded_damage_is_attached_to_the_baseline(self):
        """Damage happened under the carton that actually shipped. Crediting the
        treatment with avoiding it would score a counterfactual nobody ran."""
        ledger = ledger_with(("held-1", 900), ("held-1", 950, OutcomeEventType.DAMAGE))
        result = evaluate({"held-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.9)}},
                          ledger=ledger, decision_ids=("held-1",))
        assert result.decisions[0].baseline_realised_risk == ("damage",)

    def test_a_clean_history_carries_no_realised_risk(self):
        result = evaluate({"held-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.9)}},
                          ledger=ledger_with(("held-1", 900)), decision_ids=("held-1",))
        assert result.decisions[0].baseline_realised_risk == ()


class TestTheTally:
    def test_each_verdict_is_counted_once(self):
        table = {
            "up": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.9)},
            "down": {"baseline": (8.0, 0.9), "treatment": (10.0, 0.7)},
            "sideways": {"baseline": (10.0, 0.9), "treatment": (8.0, 0.7)},
        }
        ledger = ledger_with(("up", 900), ("down", 910), ("sideways", 920))
        result = evaluate(table, ledger=ledger,
                          decision_ids=("up", "down", "sideways"))
        assert (result.improved, result.regressed,
                result.traded_off, result.unpackable) == (1, 1, 1, 0)

    def test_the_evaluation_records_both_pins_it_compared(self):
        result = evaluate({"held-1": {"baseline": (10.0, 0.7), "treatment": (8.0, 0.9)}},
                          ledger=ledger_with(("held-1", 900)), decision_ids=("held-1",))
        assert result.baseline_pin == BASELINE_PIN
        assert result.treatment_pin == TREATMENT_PIN
        assert result.split_at == 500


class TestTheArtifactCarriesWhatTheValidatorReads:
    def _run(self):
        return OrderRunResult(order_id="d1", succeeded=True, metrics={"cost": 1.0})

    def test_a_complete_artifact_is_accepted(self):
        artifact = OrderEvaluationArtifact(run=self._run(), request={"a": 1},
                                           result={"b": 2})
        assert artifact.run.order_id == "d1"

    @pytest.mark.parametrize("missing", ["request", "result"])
    def test_an_artifact_without_the_pair_is_refused(self, missing):
        """A run summary alone cannot be independently validated: the validator is handed
        the request/result pair precisely so that no score can influence it."""
        fields = {"run": self._run(), "request": {"a": 1}, "result": {"b": 2}}
        fields[missing] = None
        with pytest.raises(ValueError, match="request/result pair"):
            OrderEvaluationArtifact(**fields)
