"""Proposing a change, and the single route from a proposal to a published one.

`tests/` ships inside the wheel, so these import `packvium.recommendations` the way a
consumer does.

Two properties carry the module. A proposal is built from a **paired** comparison, so an
order only one arm could pack contributes to neither mean and cannot manufacture an
improvement out of cohort mix. And a `Recommendation` is never itself a mutation: it is
handed no registry, and the only path to one is an explicit approval call.
"""

from __future__ import annotations

import pytest

from packvium.recommendations import (
    ApprovalRecord,
    CatalogRegistry,
    CatalogSnapshot,
    ExpectedDelta,
    MismatchedOrderCorpusError,
    PolicyAction,
    PolicyPredicate,
    PolicyRegistry,
    PolicyScope,
    Recommendation,
    approve,
    approve_catalog,
    approve_policy,
    propose_recommendation,
)
from packvium.commerce.policy import PolicyOperator
from packvium.simulation import OrderRunResult, ScenarioResult, ScenarioVersionPin

PIN = ScenarioVersionPin(catalog_version=1)
CONSTRAINTS = ("no placement the validator rejects",)
ROLLBACK = "republish catalog version 1"


def scenario(name: str, table: dict[str, tuple[float, float] | None]) -> ScenarioResult:
    runs = []
    for order_id, measured in table.items():
        if measured is None:
            runs.append(OrderRunResult(order_id=order_id, succeeded=False,
                                       failure_reason="no carton fits"))
        else:
            cost, utilisation = measured
            runs.append(OrderRunResult(order_id=order_id, succeeded=True,
                                       metrics={"cost": cost, "utilisation": utilisation}))
    return ScenarioResult(scenario_id=name, version_pin=PIN,
                          order_ids=tuple(table), runs=tuple(runs))


def propose(baseline, treatment, **overrides):
    fields = {"constraints": CONSTRAINTS, "rollback_plan": ROLLBACK,
              "minimum_cohort_size": 1, "minimum_confidence": 0.0}
    fields.update(overrides)
    return propose_recommendation("rec-1", "cheaper cartons", baseline, treatment, **fields)


class TestAProposalIsBuiltFromPairedEvidence:
    def test_the_deltas_are_the_means_of_the_paired_orders(self):
        proposal = propose(scenario("b", {"a": (10.0, 0.6), "b": (20.0, 0.8)}),
                           scenario("t", {"a": (8.0, 0.7), "b": (16.0, 0.9)}))
        assert proposal is not None
        deltas = {d.metric: (d.baseline_mean, d.treatment_mean)
                  for d in proposal.expected_deltas}
        assert deltas["cost"] == (15.0, 12.0)
        assert deltas["utilisation"] == (0.7, 0.8)

    def test_deltas_are_sorted_by_metric_name(self):
        # Report order cannot depend on dict insertion order, or two identical
        # comparisons would print differently.
        proposal = propose(scenario("b", {"a": (10.0, 0.6)}), scenario("t", {"a": (8.0, 0.7)}))
        assert [d.metric for d in proposal.expected_deltas] == ["cost", "utilisation"]

    def test_an_order_only_one_arm_packed_is_excluded_from_both_means(self):
        """The paired rule. Averaging each arm's survivors separately would let the
        difference in *who survived* look like a difference in performance."""
        proposal = propose(scenario("b", {"a": (10.0, 0.6), "b": (100.0, 0.1)}),
                           scenario("t", {"a": (8.0, 0.7), "b": None}))
        assert proposal.supporting_order_ids == ("a",)
        deltas = {d.metric: d.baseline_mean for d in proposal.expected_deltas}
        assert deltas["cost"] == 10.0

    def test_confidence_is_the_paired_share_of_the_whole_corpus(self):
        proposal = propose(
            scenario("b", {"a": (10.0, 0.6), "b": (10.0, 0.6), "c": (10.0, 0.6),
                           "d": (10.0, 0.6)}),
            scenario("t", {"a": (8.0, 0.7), "b": (8.0, 0.7), "c": (8.0, 0.7), "d": None}))
        assert proposal.confidence == 0.75

    def test_only_metrics_both_arms_reported_become_deltas(self):
        baseline = ScenarioResult(
            scenario_id="b", version_pin=PIN, order_ids=("a",),
            runs=(OrderRunResult(order_id="a", succeeded=True,
                                 metrics={"cost": 10.0, "weight": 5.0}),))
        treatment = ScenarioResult(
            scenario_id="t", version_pin=PIN, order_ids=("a",),
            runs=(OrderRunResult(order_id="a", succeeded=True,
                                 metrics={"cost": 8.0, "volume": 3.0}),))
        proposal = propose(baseline, treatment)
        assert [d.metric for d in proposal.expected_deltas] == ["cost"]


class TestAProposalIsWithheldRatherThanHedged:
    def test_a_cohort_below_the_floor_produces_nothing(self):
        assert propose(scenario("b", {"a": (10.0, 0.6)}), scenario("t", {"a": (8.0, 0.7)}),
                       minimum_cohort_size=2) is None

    def test_a_confidence_below_the_floor_produces_nothing(self):
        assert propose(scenario("b", {"a": (10.0, 0.6), "b": (10.0, 0.6)}),
                       scenario("t", {"a": (8.0, 0.7), "b": None}),
                       minimum_confidence=0.9) is None

    def test_arms_that_share_no_metric_produce_nothing(self):
        """Full cohort, full confidence, and still no axis both sides measured. `None`
        rather than a recommendation carrying an empty delta list."""
        baseline = ScenarioResult(
            scenario_id="b", version_pin=PIN, order_ids=("a",),
            runs=(OrderRunResult(order_id="a", succeeded=True, metrics={"cost": 10.0}),))
        treatment = ScenarioResult(
            scenario_id="t", version_pin=PIN, order_ids=("a",),
            runs=(OrderRunResult(order_id="a", succeeded=True,
                                 metrics={"utilisation": 0.9}),))
        assert propose(baseline, treatment) is None

    def test_two_different_corpora_are_refused(self):
        with pytest.raises(MismatchedOrderCorpusError):
            propose(scenario("b", {"a": (10.0, 0.6)}), scenario("t", {"z": (8.0, 0.7)}))


class TestPublishingIsASeparateExplicitAct:
    def _recommendation(self):
        return Recommendation(
            recommendation_id="rec-1", proposal="cheaper cartons",
            supporting_order_ids=("a",),
            expected_deltas=(ExpectedDelta(metric="cost", baseline_mean=10.0,
                                           treatment_mean=8.0),),
            confidence=1.0, constraints=CONSTRAINTS, rollback_plan=ROLLBACK)

    def test_approve_calls_the_injected_publisher_exactly_once(self):
        calls = []

        def publish() -> int:
            calls.append(1)
            return 7

        record = approve(self._recommendation(), at=1_000, publish=publish)
        assert calls == [1]
        assert record == ApprovalRecord(recommendation_id="rec-1", approved_at=1_000,
                                        published_version=7)

    def test_a_catalog_approval_publishes_through_the_real_registry(self):
        registry = CatalogRegistry(catalog_id="cartons")
        record = approve_catalog(self._recommendation(), registry, CatalogSnapshot(),
                                 approved_at=1_000, effective_at=2_000)
        assert record.published_version == 1
        assert record.recommendation_id == "rec-1"

    def test_a_second_catalog_approval_appends_a_new_version(self):
        # Append-only: publishing again never replaces what was published before.
        registry = CatalogRegistry(catalog_id="cartons")
        first = approve_catalog(self._recommendation(), registry, CatalogSnapshot(),
                                approved_at=1_000, effective_at=2_000)
        second = approve_catalog(self._recommendation(), registry, CatalogSnapshot(),
                                 approved_at=3_000, effective_at=4_000)
        assert (first.published_version, second.published_version) == (1, 2)

    def test_a_policy_approval_publishes_a_versioned_rule(self):
        registry = PolicyRegistry()
        record = approve_policy(
            self._recommendation(), registry, rule_id="no-hazmat-air",
            scope=PolicyScope.HAZMAT, action=PolicyAction.REJECT,
            predicates=(PolicyPredicate(scope=PolicyScope.HAZMAT, field="class",
                                        operator=PolicyOperator.EQUALS, value="1.4"),),
            priority=10, approved_at=1_000, effective_at=2_000)
        assert record.published_version == 1
        assert record.recommendation_id == "rec-1"

    def test_a_proposal_is_handed_no_registry_of_its_own(self):
        """Structural, not a convention: `propose_recommendation` has no registry
        parameter, so the proposal path cannot reach a mutation even by mistake."""
        import inspect
        parameters = inspect.signature(propose_recommendation).parameters
        assert not [name for name in parameters if "registry" in name.lower()]


class TestTheDeltaItself:
    def test_delta_is_treatment_minus_baseline(self):
        """Signed on purpose: whether a negative delta is good depends on the axis, and
        this type deliberately does not know."""
        cheaper = ExpectedDelta(metric="cost", baseline_mean=10.0, treatment_mean=8.0)
        denser = ExpectedDelta(metric="utilisation", baseline_mean=0.7, treatment_mean=0.9)
        assert cheaper.delta == -2.0
        assert denser.delta == pytest.approx(0.2)

    def test_a_recommendation_must_report_at_least_one_delta(self):
        # A proposal with nothing measured is advice without evidence.
        with pytest.raises(ValueError, match="at least one expected delta"):
            Recommendation(recommendation_id="rec-1", proposal="p",
                           supporting_order_ids=("a",), expected_deltas=(),
                           confidence=1.0, constraints=CONSTRAINTS,
                           rollback_plan=ROLLBACK)
