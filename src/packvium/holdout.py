"""Historical replay and holdout evaluation for a recommendation.

`docs/INTELLIGENCE-API.md` is the contract. The question this answers is whether a
recommendation *would have* improved past decisions, scored on a slice it was not derived
from -- so a proposal can be argued about before it is ever applied forward.

Three rules shape every line below, and all three are the difference between a backtest
and a story told about one.

**The split is by time, and the training view is as of that time.** A decision is held out
when its earliest recorded event is at or after `split_at`. The training side folds only
events recorded strictly before it (`packvium.outcomes.OutcomeLedger.view_as_of`), so a
correction filed
later stays in the immutable ledger but does not reach backwards into what was knowable
when the recommendation was formed. Held-out decisions may accumulate later events,
because those are precisely the outcomes being evaluated. A recommendation whose
`supporting_order_ids` reach into the holdout is refused rather than scored: it would be
citing its own answer sheet.

**Most of the ledger cannot be scored counterfactually, and pretending otherwise is the
failure mode this module exists to avoid.** The ledger records what happened under the
carton that actually shipped. Re-running the treatment produces a different carton, so
only the deterministically recomputable part -- the packing decision and what the pinned
tariff prices from it -- can be compared. `DAMAGE`, `RETURN`, `REPACK` and
`OPERATOR_OVERRIDE` were realised under a box the treatment did not choose; nothing in the
ledger says what would have happened in another one. They are carried as negative evidence
about the baseline and never as a predicted benefit of the treatment. The obvious
implementation -- diff every recorded metric -- silently credits a recommendation with
avoiding damage it was never tested against.

**The validator runs unconditionally, before any metric can be compared, and its verdict
is not an input to a score.** An arm the validator rejects has failed for that decision,
whatever it cost. There is no threshold at which that changes, and `validator` is handed
no score, so nothing learned from history can reach it. That is the project's fourth
release gate expressed as control flow rather than as a promise.

The aggregate is a count of decisions, never a mean delta. A mean over a cohort is one
number that a few large orders can carry by themselves, and refusing that collapse is why
the Pareto comparator exists in the first place.
"""

from __future__ import annotations

from ._compat import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .outcomes import OutcomeEventType, OutcomeLedger
from .recommendations import Recommendation
from .simulation import (
    OrderRunResult,
    ScenarioResult,
    ScenarioVersionPin,
    compare_scenarios,
)

__all__ = [
    "DecisionOutcome",
    "HoldoutError",
    "HoldoutEvaluation",
    "OrderEvaluationArtifact",
    "SupportingEvidenceInHoldoutError",
    "ValidationVerdict",
    "evaluate_on_history",
]


#: Ledger facts that describe something a human or the world did to the carton that
#: actually shipped. Under a different carton they may not have happened at all, so they
#: are reported against the baseline and never credited to the treatment.
NOT_COUNTERFACTUAL = frozenset({
    OutcomeEventType.DAMAGE,
    OutcomeEventType.RETURN,
    OutcomeEventType.REPACK,
    OutcomeEventType.OPERATOR_OVERRIDE,
})

IMPROVED = "improved"
REGRESSED = "regressed"
TRADED_OFF = "traded_off"
UNPACKABLE = "unpackable"


class HoldoutError(Exception):
    """Base class for every error raised by this module."""


class SupportingEvidenceInHoldoutError(HoldoutError):
    """The recommendation cites a decision that the split holds out.

    Refused rather than scored: a recommendation evaluated on the evidence it was derived
    from measures how well it remembers, not whether it generalises.
    """


@dataclass(frozen=True, slots=True)
class ValidationVerdict:
    """An independent validator's answer about one packing.

    Deliberately carries no score and no metric. It is produced by a callback that is
    never handed one, which is what keeps a learned number from reaching the verdict.
    """

    valid: bool
    codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.valid and self.codes:
            raise ValueError("a valid verdict cannot carry rejection codes")
        if not self.valid and not self.codes:
            raise ValueError("a rejection must say which rules it failed")


@dataclass(frozen=True, slots=True)
class OrderEvaluationArtifact:
    """One arm's run of one decision, plus the pair an independent check needs.

    The forward path's `OrderEvaluationArtifact` retains score, rates and ids but not
    placement geometry, so claiming it can be independently validated would be false.
    This envelope requires the request and result explicitly for that reason -- the
    validator is given the packing, not a summary of it.
    """

    run: OrderRunResult
    request: Any
    result: Any

    def __post_init__(self) -> None:
        if self.request is None or self.result is None:
            raise ValueError(
                "a holdout artifact must carry the request/result pair the validator reads; "
                "a run summary alone cannot be independently validated"
            )


#: `(decision_id, pin) -> OrderEvaluationArtifact`.
HoldoutEvaluator = Callable[[str, ScenarioVersionPin], OrderEvaluationArtifact]

#: `(request, result) -> ValidationVerdict`. Separate from the evaluator on purpose: an
#: engine that graded its own homework would make the gate a comment.
IndependentValidator = Callable[[Any, Any], ValidationVerdict]


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """What the replay found for one held-out decision, retained rather than summarised.

    The release gate about performance claims requires raw artifacts, not just counts, so
    every verdict keeps the validator codes that produced it and the realised-risk events
    the baseline actually incurred.
    """

    decision_id: str
    verdict: str
    baseline_valid: bool
    treatment_valid: bool
    baseline_codes: tuple[str, ...] = ()
    treatment_codes: tuple[str, ...] = ()
    baseline_realised_risk: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("decision_id is required")
        if self.verdict not in (IMPROVED, REGRESSED, TRADED_OFF, UNPACKABLE):
            raise ValueError(f"unknown verdict {self.verdict!r}")
        if self.verdict == IMPROVED and not self.treatment_valid:
            raise ValueError(
                "a packing the validator rejected can never be recorded as an improvement"
            )


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """The complete record of one holdout evaluation.

    Retains `split_at`, both pins and every per-decision verdict -- not just the counts --
    so the run can be argued with rather than merely believed.
    """

    recommendation_id: str
    split_at: int
    baseline_pin: ScenarioVersionPin
    treatment_pin: ScenarioVersionPin
    training_decision_ids: tuple[str, ...]
    holdout_decision_ids: tuple[str, ...]
    decisions: tuple[DecisionOutcome, ...] = ()

    @property
    def improved(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == IMPROVED)

    @property
    def regressed(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == REGRESSED)

    @property
    def traded_off(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == TRADED_OFF)

    @property
    def unpackable(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == UNPACKABLE)


def _earliest_event_time(ledger: OutcomeLedger, decision_id: str) -> Optional[int]:
    events = ledger.events_for_decision(decision_id)
    if not events:
        return None
    return min(event.recorded_at for event in events)


def _realised_risk(ledger: OutcomeLedger, decision_id: str) -> tuple[str, ...]:
    """The baseline's realised bad outcomes, as recorded. Negative evidence only."""
    return tuple(sorted({
        event.event_type.value
        for event in ledger.current_view(decision_id)
        if event.event_type in NOT_COUNTERFACTUAL
    }))


def _arm(
    decision_id: str, pin: ScenarioVersionPin,
    evaluator: HoldoutEvaluator, validator: IndependentValidator,
) -> tuple[OrderRunResult, ValidationVerdict]:
    """Run one arm and validate it. The verdict is produced before any metric is read."""
    artifact = evaluator(decision_id, pin)
    verdict = validator(artifact.request, artifact.result)
    if not verdict.valid:
        # The arm failed for this decision. Its metrics are discarded rather than
        # down-weighted: there is no cost at which a rejected placement competes.
        return (
            OrderRunResult(
                order_id=decision_id, succeeded=False,
                failure_reason="independent validation rejected the placement: "
                               + ", ".join(verdict.codes),
            ),
            verdict,
        )
    return artifact.run, verdict


def _verdict_for(
    decision_id: str, baseline_run: OrderRunResult, treatment_run: OrderRunResult,
    baseline_pin: ScenarioVersionPin, treatment_pin: ScenarioVersionPin,
    higher_is_better: Mapping[str, bool],
) -> str:
    if not baseline_run.succeeded and not treatment_run.succeeded:
        return UNPACKABLE
    if not treatment_run.succeeded:
        return REGRESSED
    if not baseline_run.succeeded:
        # The treatment packed a decision the baseline could not, and its own packing
        # passed the validator. That is an improvement in the only sense available here.
        return IMPROVED

    # Both arms are valid, so the comparison is the same per-decision Pareto report the
    # forward path uses -- one profile, two engines, never a blended delta.
    reports = compare_scenarios(
        ScenarioResult(scenario_id="baseline", version_pin=baseline_pin,
                       order_ids=(decision_id,), runs=(baseline_run,)),
        ScenarioResult(scenario_id="treatment", version_pin=treatment_pin,
                       order_ids=(decision_id,), runs=(treatment_run,)),
        higher_is_better,
    )
    report = reports[0]
    if report.winner == "treatment":
        return IMPROVED
    if report.winner == "baseline":
        return REGRESSED
    return TRADED_OFF


def evaluate_on_history(
    recommendation: Recommendation,
    ledger: OutcomeLedger,
    decision_ids: Sequence[str],
    baseline_pin: ScenarioVersionPin,
    treatment_pin: ScenarioVersionPin,
    evaluator: HoldoutEvaluator,
    validator: IndependentValidator,
    higher_is_better: Mapping[str, bool],
    *,
    split_at: int,
) -> HoldoutEvaluation:
    """Score `recommendation` against the decisions held out by `split_at`.

    `split_at` is required and has no default. Any default would be this library making a
    claim about how much history is enough for a business it cannot see.

    Raises `SupportingEvidenceInHoldoutError` when the recommendation cites a held-out
    decision, and `ValueError` when a decision has no recorded events -- an unrecorded
    decision cannot be placed on either side of a split by time.
    """
    if split_at < 0:
        raise ValueError("split_at cannot be negative")
    if not decision_ids:
        raise ValueError("at least one decision id is required")

    training: list[str] = []
    holdout: list[str] = []
    for decision_id in decision_ids:
        earliest = _earliest_event_time(ledger, decision_id)
        if earliest is None:
            raise ValueError(
                f"decision {decision_id!r} has no recorded events, so it cannot be split by time"
            )
        (holdout if earliest >= split_at else training).append(decision_id)

    cited_in_holdout = sorted(set(recommendation.supporting_order_ids) & set(holdout))
    if cited_in_holdout:
        raise SupportingEvidenceInHoldoutError(
            f"recommendation {recommendation.recommendation_id!r} cites held-out "
            f"decision(s) {cited_in_holdout}; it may not be scored on its own evidence"
        )

    outcomes: list[DecisionOutcome] = []
    for decision_id in holdout:
        baseline_run, baseline_verdict = _arm(decision_id, baseline_pin, evaluator, validator)
        treatment_run, treatment_verdict = _arm(decision_id, treatment_pin, evaluator, validator)
        outcomes.append(DecisionOutcome(
            decision_id=decision_id,
            verdict=_verdict_for(decision_id, baseline_run, treatment_run,
                                 baseline_pin, treatment_pin, higher_is_better),
            baseline_valid=baseline_verdict.valid,
            treatment_valid=treatment_verdict.valid,
            baseline_codes=baseline_verdict.codes,
            treatment_codes=treatment_verdict.codes,
            baseline_realised_risk=_realised_risk(ledger, decision_id),
        ))

    return HoldoutEvaluation(
        recommendation_id=recommendation.recommendation_id,
        split_at=split_at,
        baseline_pin=baseline_pin,
        treatment_pin=treatment_pin,
        training_decision_ids=tuple(training),
        holdout_decision_ids=tuple(holdout),
        decisions=tuple(outcomes),
    )
