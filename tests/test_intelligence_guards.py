"""Every refusal the intelligence surface makes, fired at least once.

Measured before this file existed: the six modules 1.2.0 exports sat at 93% statement
coverage, and **39 of the 41 uncovered statements were validation guards** -- the `raise`
inside a `__post_init__` or at the head of a function. Nothing anywhere fired them.

That is a worse gap than the number suggests. A guard is the one kind of code whose
absence looks exactly like its presence: delete the check, reorder `__post_init__` so it
runs before the field is set, invert a comparison, and every existing test still passes,
because every existing test supplies valid input. The construction that should have been
refused is simply accepted, and the first thing to notice is a caller holding a
`Recommendation` with a confidence of 4.0 or a ledger event stamped before the epoch.

So each test here builds one object that is wrong in exactly one way and asserts the
refusal. They are cheap, they are boring, and they are the reason a future edit to any of
these classes cannot quietly stop validating.
"""

from __future__ import annotations

import pytest

from packvium.holdout import (
    IMPROVED,
    REGRESSED,
    TRADED_OFF,
    UNPACKABLE,
    DecisionOutcome,
    HoldoutEvaluation,
    ValidationVerdict,
    evaluate_on_history,
)
from packvium.locks import _placed
from packvium.outcomes import (
    ALLOWED_FIELDS,
    OutcomeEvent,
    OutcomeEventType,
    OutcomeLedger,
    UnsupportedOutcomeFieldError,
)
from packvium.pareto import CandidateResult, ProfileReport
from packvium.recommendations import (
    ApprovalRecord,
    ExpectedDelta,
    MismatchedOrderCorpusError,
    Recommendation,
    propose_recommendation,
)
from packvium.simulation import (
    OrderRunResult,
    ScenarioResult,
    ScenarioVersionPin,
)

PIN = ScenarioVersionPin(catalog_version=1)


def _run(order_id: str, cost: float = 10.0, **overrides) -> OrderRunResult:
    fields = {"order_id": order_id, "succeeded": True, "metrics": {"cost": cost}}
    fields.update(overrides)
    return OrderRunResult(**fields)


def _scenario(scenario_id: str, runs: tuple[OrderRunResult, ...]) -> ScenarioResult:
    return ScenarioResult(scenario_id=scenario_id, version_pin=PIN,
                          order_ids=tuple(r.order_id for r in runs), runs=runs)


class TestAVersionPinCannotBeAmbiguous:
    """A pin exists so a scenario replays to the same answer. Every refusal here is a
    pin that would have replayed to a different one, or to none."""

    def test_a_version_number_below_one_is_refused(self):
        # Versions are positions in a published history starting at 1, so 0 is not an
        # "unset" sentinel -- `None` is. Accepting it would silently pin nothing.
        with pytest.raises(ValueError, match="version numbers must be positive"):
            ScenarioVersionPin(catalog_version=0)

    def test_a_tariff_version_below_one_is_refused(self):
        with pytest.raises(ValueError, match="version numbers must be positive"):
            ScenarioVersionPin(tariff_version=-1)

    def test_a_policy_pin_without_a_rule_id_is_refused(self):
        with pytest.raises(ValueError, match="policy version pins require"):
            ScenarioVersionPin(policy_versions=(("", 2),))

    def test_a_policy_pin_with_a_non_positive_version_is_refused(self):
        with pytest.raises(ValueError, match="policy version pins require"):
            ScenarioVersionPin(policy_versions=(("carrier-rules", 0),))

    def test_the_same_policy_rule_cannot_be_pinned_twice(self):
        # Two pins for one rule is not a merge conflict to resolve by last-wins: it is a
        # caller who does not know which version they meant.
        with pytest.raises(ValueError, match="same policy rule twice"):
            ScenarioVersionPin(policy_versions=(("carrier-rules", 1), ("carrier-rules", 2)))

    def test_an_empty_solver_version_is_refused_but_an_absent_one_is_not(self):
        with pytest.raises(ValueError, match="solver_version must be non-empty"):
            ScenarioVersionPin(catalog_version=1, solver_version="")
        assert ScenarioVersionPin(catalog_version=1,
                                  solver_version=None).solver_version is None

    def test_a_pin_that_pins_nothing_is_refused(self):
        """Not a coverage gap -- this one was already exercised -- but it is the guard
        that makes every other field optional without making the type meaningless."""
        with pytest.raises(ValueError, match="at least one of its fields"):
            ScenarioVersionPin()


class TestARunCannotCarryAnUnnamedNumber:
    def test_a_run_without_an_order_id_is_refused(self):
        with pytest.raises(ValueError, match="order_id is required"):
            OrderRunResult(order_id="", succeeded=True, metrics={"cost": 1.0})

    def test_a_metric_with_an_empty_name_is_refused(self):
        # An unnamed axis cannot be compared against the same axis in the other arm, so
        # it would drop out of every dominance check while still looking like evidence.
        with pytest.raises(ValueError, match="metric names must be non-empty"):
            OrderRunResult(order_id="order-1", succeeded=True, metrics={"": 1.0})


class TestAScenarioMustDescribeTheCorpusItRan:
    def test_a_scenario_without_an_id_is_refused(self):
        with pytest.raises(ValueError, match="scenario_id is required"):
            ScenarioResult(scenario_id="", version_pin=PIN, order_ids=("order-1",),
                           runs=(_run("order-1"),))

    def test_a_scenario_over_no_orders_is_refused(self):
        with pytest.raises(ValueError, match="order_ids cannot be empty"):
            ScenarioResult(scenario_id="s", version_pin=PIN, order_ids=(), runs=())

    def test_runs_that_do_not_line_up_with_the_orders_are_refused(self):
        # The pairing is positional everywhere downstream. A mismatch here would pair one
        # order's baseline against another order's treatment and report the difference.
        with pytest.raises(ValueError, match="1:1"):
            ScenarioResult(scenario_id="s", version_pin=PIN,
                           order_ids=("order-1", "order-2"), runs=(_run("order-1"),))

    def test_runs_in_a_different_order_than_the_ids_are_refused(self):
        with pytest.raises(ValueError, match="1:1"):
            ScenarioResult(scenario_id="s", version_pin=PIN,
                           order_ids=("order-1", "order-2"),
                           runs=(_run("order-2"), _run("order-1")))


class TestAParetoCandidateCannotBeAnonymous:
    def test_a_candidate_without_a_profile_is_refused(self):
        with pytest.raises(ValueError, match="profile is required"):
            CandidateResult(profile="", engine="python", metrics={"cost": 1.0})

    def test_a_candidate_without_an_engine_is_refused(self):
        with pytest.raises(ValueError, match="engine is required"):
            CandidateResult(profile="fast", engine="", metrics={"cost": 1.0})

    def test_a_candidate_with_no_metrics_is_refused(self):
        # Nothing dominates a candidate with no axes, so an empty one would arrive on the
        # frontier of every profile it was added to.
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            CandidateResult(profile="fast", engine="python", metrics={})


class TestAProfileReportCannotContradictItself:
    def test_a_report_without_a_profile_is_refused(self):
        with pytest.raises(ValueError, match="profile is required"):
            ProfileReport(profile="", pareto_optimal=("python",), dominated=())

    def test_a_winner_that_is_not_on_the_frontier_is_refused(self):
        with pytest.raises(ValueError, match="must be one of the Pareto-optimal"):
            ProfileReport(profile="fast", pareto_optimal=("python",), dominated=("php",),
                          winner="php")

    def test_a_winner_alongside_a_genuine_trade_off_is_refused(self):
        # The whole point of the type: when two candidates survive, naming one of them
        # winner is the single-number collapse the report exists to refuse.
        with pytest.raises(ValueError, match="exactly one candidate is Pareto-optimal"):
            ProfileReport(profile="fast", pareto_optimal=("php", "python"), dominated=(),
                          winner="python")


class TestARecommendationCannotBeUnaccountable:
    def _valid(self, **overrides):
        fields = {
            "recommendation_id": "rec-1", "proposal": "cheaper cartons",
            "supporting_order_ids": ("order-1",),
            "expected_deltas": (ExpectedDelta(metric="cost", baseline_mean=10.0,
                                              treatment_mean=8.0),),
            "confidence": 0.8, "constraints": ("no placement the validator rejects",),
            "rollback_plan": "republish v1",
        }
        fields.update(overrides)
        return Recommendation(**fields)

    def test_the_valid_shape_is_actually_valid(self):
        """Guards the negatives below: if the baseline were itself refused, every test in
        this class would pass for the wrong reason."""
        assert self._valid().recommendation_id == "rec-1"

    def test_a_delta_on_an_unnamed_metric_is_refused(self):
        with pytest.raises(ValueError, match="metric is required"):
            ExpectedDelta(metric="", baseline_mean=1.0, treatment_mean=2.0)

    def test_a_recommendation_without_an_id_is_refused(self):
        with pytest.raises(ValueError, match="recommendation_id is required"):
            self._valid(recommendation_id="")

    def test_a_recommendation_without_a_proposal_is_refused(self):
        with pytest.raises(ValueError, match="proposal is required"):
            self._valid(proposal="")

    def test_a_recommendation_citing_no_orders_is_refused(self):
        # Evidence-free advice is the one output this module must not be able to produce.
        with pytest.raises(ValueError, match="at least one supporting order"):
            self._valid(supporting_order_ids=())

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, 4.0])
    def test_a_confidence_outside_zero_to_one_is_refused(self, confidence):
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            self._valid(confidence=confidence)

    def test_a_recommendation_stating_no_constraint_is_refused(self):
        """Found by the canary above rather than by reading: the shared fixture passed an
        empty tuple and every negative test in this class was passing on the wrong
        exception."""
        with pytest.raises(ValueError, match="at least one non-empty constraint"):
            self._valid(constraints=())

    def test_a_recommendation_whose_only_constraint_is_blank_is_refused(self):
        with pytest.raises(ValueError, match="at least one non-empty constraint"):
            self._valid(constraints=("",))

    def test_a_recommendation_without_a_rollback_plan_is_refused(self):
        with pytest.raises(ValueError, match="must state its rollback plan"):
            self._valid(rollback_plan="")


class TestAnApprovalRecordCannotBeBackdatedOrUnversioned:
    def test_an_approval_without_a_recommendation_is_refused(self):
        with pytest.raises(ValueError, match="recommendation_id is required"):
            ApprovalRecord(recommendation_id="", approved_at=1, published_version=1)

    def test_an_approval_before_the_epoch_is_refused(self):
        with pytest.raises(ValueError, match="approved_at cannot be negative"):
            ApprovalRecord(recommendation_id="rec-1", approved_at=-1, published_version=1)

    def test_an_approval_naming_version_zero_is_refused(self):
        # Published versions are positions in a history starting at 1, so 0 names nothing.
        with pytest.raises(ValueError, match="published_version must be positive"):
            ApprovalRecord(recommendation_id="rec-1", approved_at=1, published_version=0)


class TestProposingFromEvidenceThatCannotSupportIt:
    def test_two_different_corpora_are_refused_rather_than_intersected(self):
        # Silently comparing the overlap would let cohort mix manufacture an improvement.
        with pytest.raises(MismatchedOrderCorpusError):
            propose_recommendation(
                "rec-1", "p", _scenario("b", (_run("order-1"),)),
                _scenario("t", (_run("order-2"),)),
                constraints=(), rollback_plan="back", minimum_cohort_size=1,
                minimum_confidence=0.0)

    def test_a_non_positive_cohort_floor_is_refused(self):
        with pytest.raises(ValueError, match="minimum_cohort_size must be positive"):
            propose_recommendation(
                "rec-1", "p", _scenario("b", (_run("order-1"),)),
                _scenario("t", (_run("order-1"),)),
                constraints=(), rollback_plan="back", minimum_cohort_size=0,
                minimum_confidence=0.0)

    @pytest.mark.parametrize("confidence", [-0.5, 1.5])
    def test_a_confidence_floor_outside_zero_to_one_is_refused(self, confidence):
        with pytest.raises(ValueError, match="minimum_confidence must be between 0 and 1"):
            propose_recommendation(
                "rec-1", "p", _scenario("b", (_run("order-1"),)),
                _scenario("t", (_run("order-1"),)),
                constraints=(), rollback_plan="back", minimum_cohort_size=1,
                minimum_confidence=confidence)

    def test_arms_that_share_no_metric_produce_no_recommendation(self):
        """Both arms succeeded on every order, so the cohort is full and confidence is
        1.0 -- and there is still nothing to recommend, because no axis appears on both
        sides. `None` rather than a `Recommendation` carrying an empty delta list."""
        baseline = _scenario("b", (OrderRunResult(order_id="order-1", succeeded=True,
                                                  metrics={"cost": 10.0}),))
        treatment = _scenario("t", (OrderRunResult(order_id="order-1", succeeded=True,
                                                   metrics={"utilisation": 0.8}),))
        assert propose_recommendation(
            "rec-1", "p", baseline, treatment, constraints=(), rollback_plan="back",
            minimum_cohort_size=1, minimum_confidence=0.0) is None


class TestALedgerEventCannotBeMalformed:
    def test_an_event_without_an_id_is_refused(self):
        with pytest.raises(ValueError, match="event_id is required"):
            OutcomeEvent(event_id="", decision_id="d1",
                         event_type=OutcomeEventType.MEASURED_WEIGHT,
                         payload={"weight_g": 100}, recorded_at=1)

    def test_an_event_recorded_before_the_epoch_is_refused(self):
        # Timestamps are what a holdout split is made of; a negative one would sort into
        # the training side of every split that could ever be chosen.
        with pytest.raises(ValueError, match="recorded_at cannot be negative"):
            OutcomeEvent(event_id="e1", decision_id="d1",
                         event_type=OutcomeEventType.MEASURED_WEIGHT,
                         payload={"weight_g": 100}, recorded_at=-1)

    def test_a_view_at_a_negative_instant_is_refused(self):
        with pytest.raises(ValueError, match="as-of time cannot be negative"):
            OutcomeLedger().view_as_of("d1", -1)

    @pytest.mark.parametrize("event_type,payload,rejected", [
        (OutcomeEventType.DAMAGE, {"carton_id": "box-a"}, "carton_id"),
        (OutcomeEventType.MEASURED_WEIGHT, {"weight_g": 1, "severity": "high"}, "severity"),
        (OutcomeEventType.RETURN, {"operator_id": "op-7"}, "operator_id"),
    ])
    def test_a_field_the_event_type_does_not_define_is_refused(self, event_type, payload,
                                                               rejected):
        """The closed set is what lets a replay rely on what it finds. An open payload
        would make the ledger a log."""
        with pytest.raises(UnsupportedOutcomeFieldError, match=rejected):
            OutcomeEvent(event_id="e1", decision_id="d1", event_type=event_type,
                         payload=payload, recorded_at=1)


class TestEveryEventTypeDeclaresItsFields:
    """The guard behind these two tests cannot fire through any legitimate call, and that
    is exactly why it is worth holding: it exists to catch a *future* edit that adds an
    `OutcomeEventType` member and forgets to say which fields it carries. Without a test,
    the first sign would be an event accepted with a payload nobody validated."""

    def test_no_event_type_is_missing_from_the_allowed_set(self):
        undeclared = [event_type.name for event_type in OutcomeEventType
                      if event_type not in ALLOWED_FIELDS]
        assert not undeclared, f"event type(s) with no declared fields: {undeclared}"

    def test_the_allowed_set_declares_nothing_that_is_not_an_event_type(self):
        # The other direction: a stale entry left behind by a removed member would make
        # the check above pass while the table quietly described a type that is gone.
        assert set(ALLOWED_FIELDS) == set(OutcomeEventType)

    def test_an_undeclared_event_type_is_refused_rather_than_waved_through(self):
        """`event_type` is not type-checked at runtime, so the refusal has to be real
        rather than implied by the annotation."""
        with pytest.raises(UnsupportedOutcomeFieldError, match="unsupported event type"):
            OutcomeEvent(event_id="e1", decision_id="d1", event_type="delivered",
                         payload={}, recorded_at=1)


class TestAValidationVerdictCannotBeSelfContradictory:
    def test_a_valid_verdict_carrying_rejection_codes_is_refused(self):
        with pytest.raises(ValueError, match="valid verdict cannot carry rejection codes"):
            ValidationVerdict(valid=True, codes=("unsupported_item",))

    def test_a_rejection_naming_no_rule_is_refused(self):
        # "Rejected, reason unavailable" is the shape an operator cannot act on.
        with pytest.raises(ValueError, match="must say which rules it failed"):
            ValidationVerdict(valid=False, codes=())


class TestADecisionOutcomeCannotMisreportItsArm:
    def test_a_decision_without_an_id_is_refused(self):
        with pytest.raises(ValueError, match="decision_id is required"):
            DecisionOutcome(decision_id="", verdict=IMPROVED, baseline_valid=True,
                            treatment_valid=True)

    def test_a_verdict_outside_the_four_is_refused(self):
        with pytest.raises(ValueError, match="unknown verdict"):
            DecisionOutcome(decision_id="d1", verdict="better", baseline_valid=True,
                            treatment_valid=True)

    def test_an_improvement_whose_treatment_was_rejected_is_refused(self):
        """The gate that matters most, asserted at the type rather than trusted to the
        control flow that also enforces it: there is no cost at which a packing the
        validator refused becomes an improvement."""
        with pytest.raises(ValueError):
            DecisionOutcome(decision_id="d1", verdict=IMPROVED, baseline_valid=True,
                            treatment_valid=False, treatment_codes=("unstable",))

    @pytest.mark.parametrize("verdict", [IMPROVED, REGRESSED, TRADED_OFF, UNPACKABLE])
    def test_each_of_the_four_verdicts_is_accepted(self, verdict):
        outcome = DecisionOutcome(decision_id="d1", verdict=verdict, baseline_valid=True,
                                  treatment_valid=True)
        assert outcome.verdict == verdict


class TestTheHoldoutTallyCountsEveryVerdict:
    def test_all_four_counters_report_their_own_verdict(self):
        """`unpackable` had no caller anywhere. A counter nothing reads is a counter that
        can be wrong for a whole release."""
        evaluation = HoldoutEvaluation(
            recommendation_id="rec-1", split_at=10, baseline_pin=PIN, treatment_pin=PIN,
            training_decision_ids=(), holdout_decision_ids=("a", "b", "c", "d"),
            decisions=tuple(
                DecisionOutcome(decision_id=decision_id, verdict=verdict,
                                baseline_valid=True, treatment_valid=True)
                for decision_id, verdict in (
                    ("a", IMPROVED), ("b", REGRESSED), ("c", TRADED_OFF), ("d", UNPACKABLE))
            ),
        )
        assert (evaluation.improved, evaluation.regressed,
                evaluation.traded_off, evaluation.unpackable) == (1, 1, 1, 1)


class TestReplayRefusesAnUnanswerableRequest:
    def _recommendation(self):
        return Recommendation(
            recommendation_id="rec-1", proposal="p", supporting_order_ids=("order-1",),
            expected_deltas=(ExpectedDelta(metric="cost", baseline_mean=10.0,
                                           treatment_mean=8.0),),
            confidence=1.0, constraints=("no placement the validator rejects",),
            rollback_plan="back")

    def _evaluate(self, **overrides):
        fields = {
            "recommendation": self._recommendation(), "ledger": OutcomeLedger(),
            "decision_ids": ("d1",), "baseline_pin": PIN, "treatment_pin": PIN,
            "evaluator": lambda decision_id, pin: None,
            "validator": lambda request, result: ValidationVerdict(valid=True),
            "higher_is_better": {"cost": False}, "split_at": 10,
        }
        fields.update(overrides)
        return evaluate_on_history(**fields)

    def test_a_negative_split_is_refused(self):
        with pytest.raises(ValueError, match="split_at cannot be negative"):
            self._evaluate(split_at=-1)

    def test_an_empty_decision_list_is_refused(self):
        # Vacuous success is the failure mode: an empty holdout would report zero
        # regressions and read as a clean bill of health.
        with pytest.raises(ValueError, match="at least one decision id is required"):
            self._evaluate(decision_ids=())

    def test_a_decision_with_no_recorded_events_is_refused(self):
        """An unrecorded decision has no timestamp, so it cannot be placed on either side
        of a split by time -- and defaulting it to one side would be this library
        guessing."""
        with pytest.raises(ValueError, match="no recorded events"):
            self._evaluate(decision_ids=("never-shipped",))


class TestAMalformedPlacementIsSkippedRatherThanCrashing:
    """`_placed` is private and reached through `resolve_with_locks`, but the branch it
    guards cannot be provoked through that path: the solver does not emit a placement
    without tick coordinates. Tested directly, because a defensive branch nothing
    exercises is a defensive branch that may already be broken."""

    def _container(self, *placements):
        return {"containers": [{"placements": list(placements)}]}

    def _placement(self, position):
        return {"item_type": "box", "orientation": "LWH", "position": position}

    def test_a_well_formed_placement_is_collected(self):
        found = _placed(self._container(self._placement(
            {"x": {"ticks": 0}, "y": {"ticks": 16000}, "z": {"ticks": 32000}})))
        assert found == {(0, "box", "LWH", (0, 16000, 32000))}

    def test_a_placement_missing_an_axis_is_skipped(self):
        assert _placed(self._container(self._placement(
            {"x": {"ticks": 0}, "y": {"ticks": 0}}))) == set()

    def test_a_placement_whose_axis_is_not_a_mapping_is_skipped(self):
        assert _placed(self._container(self._placement(
            {"x": 0, "y": 0, "z": 0}))) == set()

    def test_a_skipped_placement_does_not_hide_a_good_one(self):
        # The `continue` must skip one placement, not abandon the container.
        found = _placed(self._container(
            self._placement({"x": {"ticks": 0}}),
            self._placement({"x": {"ticks": 1}, "y": {"ticks": 2}, "z": {"ticks": 3}}),
        ))
        assert found == {(0, "box", "LWH", (1, 2, 3))}
