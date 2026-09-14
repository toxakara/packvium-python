"""Prove a carton change is better before you publish it.

Run it:

    PYTHONPATH=src python3 examples/intelligence.py

Every other example asks the engine to pack one shipment. This one asks a different
question, the one that comes up when you already pack well and want to change something:
*is the new carton set actually an improvement, and how would I know?*

Four functions answer it, and none of them returns a single blended number. A score that
folds cost and utilisation together can name a "winner" that is worse on the axis you
actually care about, so what you get back is a per-order Pareto report: which orders
improved, which regressed, and which genuinely traded one axis for another.

Nothing here reads the clock or the network. The scenarios are pinned to catalog versions
you supply, so the same comparison replays to the same answer next year.
"""

from __future__ import annotations

from packvium.holdout import (
    OrderEvaluationArtifact,
    SupportingEvidenceInHoldoutError,
    ValidationVerdict,
    evaluate_on_history,
)
from packvium.outcomes import OutcomeEvent, OutcomeEventType, OutcomeLedger
from packvium.recommendations import propose_recommendation
from packvium.simulation import (
    OrderRunResult,
    ScenarioVersionPin,
    compare_scenarios,
    run_scenario,
)

# --------------------------------------------------------------------------------------
# Two carton sets over the same five orders. You would measure these by packing each
# order twice; they are written out here so the example has nothing to hide.
#
# `cost_minor` is in cents and lower is better. `utilisation` is filled volume over
# container volume and higher is better. Order 5 is the interesting one: the proposed
# cartons cannot pack it at all.
# --------------------------------------------------------------------------------------
MEASURED = {
    "order-1": {"baseline": (1240, 0.71), "treatment": (1120, 0.76)},
    "order-2": {"baseline": (980, 0.68), "treatment": (910, 0.72)},
    "order-3": {"baseline": (1500, 0.80), "treatment": (1600, 0.83)},
    "order-4": {"baseline": (2100, 0.64), "treatment": (1890, 0.69)},
    "order-5": {"baseline": (1350, 0.70), "treatment": None},
}
ORDER_IDS = tuple(MEASURED)

#: Both directions are stated explicitly. There is no default, because guessing that
#: runtime or cost is "higher is better" would silently invert the whole report.
HIGHER_IS_BETTER = {"cost_minor": False, "utilisation": True}

#: A pin is what makes a comparison replayable: it names the catalog version each arm
#: was run against rather than whatever is current when you happen to run it.
BASELINE_PIN = ScenarioVersionPin(catalog_version=1)
TREATMENT_PIN = ScenarioVersionPin(catalog_version=2)


def arm(name: str):
    """An evaluator for one arm. In your code this calls `pack()`; here it replays the
    measurements above so the example teaches the comparison and not the packing."""

    def evaluate(order_id: str, version_pin: ScenarioVersionPin) -> OrderRunResult:
        measured = MEASURED[order_id][name]
        if measured is None:
            return OrderRunResult(
                order_id=order_id, succeeded=False,
                failure_reason="no carton in the proposed set fits this order",
            )
        cost, utilisation = measured
        return OrderRunResult(
            order_id=order_id, succeeded=True,
            metrics={"cost_minor": float(cost), "utilisation": utilisation},
        )

    return evaluate


print("=" * 78)
print("1. Two scenarios, compared order by order")
print("=" * 78)
print()

baseline = run_scenario("current-cartons", BASELINE_PIN, ORDER_IDS, arm("baseline"))
treatment = run_scenario("proposed-cartons", TREATMENT_PIN, ORDER_IDS, arm("treatment"))

for report in compare_scenarios(baseline, treatment, HIGHER_IS_BETTER):
    if report.winner is not None:
        print(f"  {report.profile}: {report.winner}")
    else:
        print(f"  {report.profile}: no winner — {' and '.join(report.pareto_optimal)} "
              f"are both Pareto-optimal")

print()
print("  order-3 has no winner because the trade-off is real: the proposed cartons cost")
print("  more and pack denser. A blended score would have picked one and hidden that.")
print()
print("  order-5 names `baseline` as winner for a duller reason — the proposed cartons")
print("  produced no answer there, so there was only one candidate to be optimal. That")
print("  is not a baseline win, and the next step is careful not to count it as one.")
print()

print("=" * 78)
print("2. A proposal, or an explicit refusal to make one")
print("=" * 78)
print()

# The paired cohort is four orders out of five, so confidence is 0.8. Ask for more than
# the evidence supports and you get `None` rather than a proposal with a caveat attached.
strict = propose_recommendation(
    "rec-cartons-2024", "adopt the proposed carton set", baseline, treatment,
    constraints=("no placement the validator would reject",),
    rollback_plan="republish catalog version 1",
    minimum_cohort_size=4, minimum_confidence=0.9,
)
print(f"  minimum_confidence=0.9 -> {strict}")
print("  Four of five orders are comparable, so confidence is 0.80 and the bar is not")
print("  met. `None` is the whole answer: there is no low-confidence proposal to weigh.")
print()

recommendation = propose_recommendation(
    "rec-cartons-2024", "adopt the proposed carton set", baseline, treatment,
    constraints=("no placement the validator would reject",),
    rollback_plan="republish catalog version 1",
    minimum_cohort_size=4, minimum_confidence=0.75,
)
assert recommendation is not None
print(f"  minimum_confidence=0.75 -> {recommendation.recommendation_id}")
print(f"  supported by: {', '.join(recommendation.supporting_order_ids)}")
print(f"  confidence:   {recommendation.confidence:.2f}")
for delta in recommendation.expected_deltas:
    print(f"    {delta.metric:<12} {delta.baseline_mean:>8.3f} -> {delta.treatment_mean:>8.3f}")
print()
print("  order-5 is absent from the supporting orders. A delta is a paired comparison,")
print("  so an order only one arm could pack contributes to neither mean.")
print()

print("=" * 78)
print("3. Replay against history the proposal has never seen")
print("=" * 78)
print()

# The ledger is append-only and records what actually happened to a shipment. Its
# timestamps are what a holdout split is made of.
#
# Each event type accepts a closed set of fields and nothing else -- `damage` takes a
# reason and a severity, not a carton id. A ledger that accepted arbitrary payloads would
# be a log; the point of this one is that a replay can rely on what it finds there.
SHIPPED = OutcomeEventType.ACTUAL_CARTON
ledger = OutcomeLedger()
HISTORY = (
    ("order-1", SHIPPED, 100, {"carton_id": "box-a", "catalog_version": 1}),
    ("order-2", SHIPPED, 120, {"carton_id": "box-a", "catalog_version": 1}),
    ("shipment-101", SHIPPED, 900, {"carton_id": "box-b", "catalog_version": 1}),
    ("shipment-102", SHIPPED, 910, {"carton_id": "box-a", "catalog_version": 1}),
    ("shipment-103", SHIPPED, 920, {"carton_id": "box-c", "catalog_version": 1}),
    ("shipment-104", SHIPPED, 930, {"carton_id": "box-b", "catalog_version": 1}),
    # A real bad outcome under the carton that actually shipped. It is reported against
    # the baseline and never credited to the treatment, because it happened.
    ("shipment-104", OutcomeEventType.DAMAGE, 940,
     {"reason_code": "crushed_corner", "severity": "minor"}),
)
for index, (decision_id, event_type, recorded_at, payload) in enumerate(HISTORY, start=1):
    ledger.record(OutcomeEvent(
        event_id=f"e{index}", decision_id=decision_id, event_type=event_type,
        payload=payload, recorded_at=recorded_at,
    ))

REPLAY = {
    "shipment-101": {"baseline": (1400, 0.66), "treatment": (1210, 0.73)},
    "shipment-102": {"baseline": (1150, 0.74), "treatment": (1180, 0.77)},
    # Cheaper under the proposed cartons -- and unstackable, which the validator catches.
    "shipment-103": {"baseline": (1600, 0.62), "treatment": (1050, 0.81)},
    "shipment-104": {"baseline": (990, 0.69), "treatment": (940, 0.71)},
}
INVALID = {("shipment-103", "treatment"): ("unsupported_item",)}


def replay(decision_id: str, version_pin: ScenarioVersionPin) -> OrderEvaluationArtifact:
    """Re-pack one historical decision under whichever carton set the pin names."""
    name = "baseline" if version_pin == BASELINE_PIN else "treatment"
    cost, utilisation = REPLAY[decision_id][name]
    run = OrderRunResult(
        order_id=decision_id, succeeded=True,
        metrics={"cost_minor": float(cost), "utilisation": utilisation},
    )
    # The request/result pair is what the validator is handed. It deliberately carries no
    # score, so nothing it decides can be influenced by how good the packing looked.
    return OrderEvaluationArtifact(run=run, request={"decision_id": decision_id},
                                   result={"arm": name})


def validator(request, result) -> ValidationVerdict:
    """Your own independent check, run over the packing rather than over its score."""
    codes = INVALID.get((request["decision_id"], result["arm"]), ())
    return ValidationVerdict(valid=not codes, codes=codes)


evaluation = evaluate_on_history(
    recommendation, ledger,
    decision_ids=("order-1", "order-2", "shipment-101", "shipment-102",
                  "shipment-103", "shipment-104"),
    baseline_pin=BASELINE_PIN, treatment_pin=TREATMENT_PIN,
    evaluator=replay, validator=validator, higher_is_better=HIGHER_IS_BETTER,
    split_at=500,
)

print("  split_at=500 — required, with no default, because how much history is enough")
print("  is a claim about your business that this library cannot make for you.")
print()
print(f"  training: {', '.join(evaluation.training_decision_ids)}")
print(f"  held out: {', '.join(evaluation.holdout_decision_ids)}")
print()
for decision in evaluation.decisions:
    note = ""
    if decision.treatment_codes:
        note = f"  (treatment rejected: {', '.join(decision.treatment_codes)})"
    elif decision.baseline_realised_risk:
        note = (f"  (baseline actually suffered: "
                f"{', '.join(decision.baseline_realised_risk)})")
    print(f"    {decision.decision_id:<14} {decision.verdict}{note}")
print()
print(f"  improved {evaluation.improved}, regressed {evaluation.regressed}, "
      f"traded off {evaluation.traded_off}, unpackable {evaluation.unpackable}")
print()
print("  shipment-104's note is negative evidence about the baseline, not a reason the")
print("  treatment won. Damage, returns, repacks and operator overrides happened under")
print("  the carton that actually shipped, so they are reported against that arm and")
print("  never credited to the one that was never tried.")
print()
print("  shipment-103 is the one to read twice. The proposed cartons packed it for 1050")
print("  against 1600 and denser besides — and the independent validator rejected the")
print("  placement, so it counts as a regression. There is no cost at which a packing")
print("  your validator refuses becomes an improvement; the metrics are discarded, not")
print("  discounted.")
print()

print("=" * 78)
print("4. A proposal cannot be scored on its own evidence")
print("=" * 78)
print()

# Move the split earlier and order-1 -- which the recommendation cites -- lands in the
# held-out set. That is marking your own homework, so it is refused rather than reported.
try:
    evaluate_on_history(
        recommendation, ledger,
        decision_ids=("order-1", "shipment-101"),
        baseline_pin=BASELINE_PIN, treatment_pin=TREATMENT_PIN,
        evaluator=replay, validator=validator, higher_is_better=HIGHER_IS_BETTER,
        split_at=50,
    )
except SupportingEvidenceInHoldoutError as refusal:
    print(f"  refused: {refusal}")
print()
print("  The refusal is the feature. A holdout score is worth exactly as much as the")
print("  separation between the evidence that produced the proposal and the evidence")
print("  that tests it, and nothing else in the call can enforce that separation.")
