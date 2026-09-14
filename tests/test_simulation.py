"""Scenario runs and the order-by-order comparison built from them.

`tests/` ships inside the wheel, so these import `packvium.simulation` the way a consumer
does rather than reaching it through the workspace shim.

The comparison's one rule: a scenario is compared **per order**, never as a blended delta.
An aggregate would hide which orders improved and which paid for the improvement, and that
is exactly the information a caller needs before publishing a catalog change.
"""

from __future__ import annotations

import pytest

from packvium.simulation import (
    MismatchedOrderCorpusError,
    ScenarioError,
    OrderRunResult,
    ScenarioResult,
    ScenarioVersionPin,
    compare_scenarios,
    run_scenario,
)

PIN = ScenarioVersionPin(catalog_version=1)
DIRECTIONS = {"cost": False, "utilisation": True}


def arm(**by_order):
    """An evaluator that replays a table. `None` means the arm could not pack that order."""
    def evaluate(order_id: str, version_pin: ScenarioVersionPin) -> OrderRunResult:
        measured = by_order[order_id]
        if measured is None:
            return OrderRunResult(order_id=order_id, succeeded=False,
                                  failure_reason="no carton fits")
        cost, utilisation = measured
        return OrderRunResult(order_id=order_id, succeeded=True,
                              metrics={"cost": cost, "utilisation": utilisation})
    return evaluate


class TestRunningAScenario:
    def test_every_order_is_evaluated_in_the_given_sequence(self):
        seen = []

        def evaluate(order_id, version_pin):
            seen.append(order_id)
            return OrderRunResult(order_id=order_id, succeeded=True, metrics={"cost": 1.0})

        result = run_scenario("s", PIN, ("b", "a", "c"), evaluate)
        assert seen == ["b", "a", "c"]
        assert result.order_ids == ("b", "a", "c")

    def test_the_pin_reaches_the_evaluator_unchanged(self):
        """A scenario replays from stored versions, so the pin is what the evaluator is
        expected to resolve its catalog against."""
        seen = []

        def evaluate(order_id, version_pin):
            seen.append(version_pin)
            return OrderRunResult(order_id=order_id, succeeded=True, metrics={"cost": 1.0})

        run_scenario("s", PIN, ("a",), evaluate)
        assert seen == [PIN]

    def test_the_same_arguments_reproduce_an_equal_result(self):
        # Reproducibility is this function's own determinism, not a separate mechanism.
        first = run_scenario("s", PIN, ("a", "b"), arm(a=(10.0, 0.7), b=(12.0, 0.8)))
        second = run_scenario("s", PIN, ("a", "b"), arm(a=(10.0, 0.7), b=(12.0, 0.8)))
        assert first == second

    def test_a_failed_order_is_carried_rather_than_dropped(self):
        result = run_scenario("s", PIN, ("a", "b"), arm(a=(10.0, 0.7), b=None))
        assert [r.succeeded for r in result.runs] == [True, False]
        assert result.runs[1].failure_reason == "no carton fits"

    def test_metrics_are_read_only_once_the_run_exists(self):
        run = OrderRunResult(order_id="a", succeeded=True, metrics={"cost": 1.0})
        with pytest.raises(TypeError):
            run.metrics["cost"] = 2.0


class TestARunGuardsItsOwnClaims:
    def test_a_succeeded_run_must_report_a_metric(self):
        # "It worked" with nothing measured is not a comparable result.
        with pytest.raises(ValueError, match="at least one metric"):
            OrderRunResult(order_id="a", succeeded=True, metrics={})

    def test_a_failed_run_must_say_why(self):
        with pytest.raises(ValueError, match="failure_reason"):
            OrderRunResult(order_id="a", succeeded=False)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_metric_is_refused_at_the_run(self, value):
        """Stricter than the Pareto module, which accepts infinities: a *measurement* of
        infinite cost is a broken evaluator, not a caller's encoding of "unpriceable"."""
        with pytest.raises(ValueError, match="finite"):
            OrderRunResult(order_id="a", succeeded=True, metrics={"cost": value})

    def test_a_boolean_is_not_a_metric(self):
        # `True` is an `int` in Python, which is exactly why this needs its own guard.
        with pytest.raises(ValueError, match="finite"):
            OrderRunResult(order_id="a", succeeded=True, metrics={"cost": True})


class TestComparingTwoScenarios:
    def _pair(self, baseline_table, treatment_table, orders=("a", "b")):
        return (run_scenario("base", PIN, orders, arm(**baseline_table)),
                run_scenario("treat", PIN, orders, arm(**treatment_table)))

    def test_one_report_per_order(self):
        base, treat = self._pair({"a": (10.0, 0.7), "b": (12.0, 0.8)},
                                 {"a": (8.0, 0.9), "b": (11.0, 0.85)})
        reports = compare_scenarios(base, treat, DIRECTIONS)
        assert [r.profile for r in reports] == ["a", "b"]

    def test_a_dominant_arm_is_named_per_order(self):
        base, treat = self._pair({"a": (10.0, 0.7), "b": (12.0, 0.8)},
                                 {"a": (8.0, 0.9), "b": (14.0, 0.6)})
        winners = {r.profile: r.winner for r in compare_scenarios(base, treat, DIRECTIONS)}
        assert winners == {"a": "treatment", "b": "baseline"}

    def test_a_real_trade_off_on_one_order_leaves_that_order_without_a_winner(self):
        """The reason there is no aggregate: order `b` genuinely trades cost for density,
        and no single number can report that without deciding it."""
        base, treat = self._pair({"a": (10.0, 0.7), "b": (12.0, 0.8)},
                                 {"a": (8.0, 0.9), "b": (14.0, 0.9)})
        reports = {r.profile: r for r in compare_scenarios(base, treat, DIRECTIONS)}
        assert reports["a"].winner == "treatment"
        assert reports["b"].winner is None
        assert reports["b"].pareto_optimal == ("baseline", "treatment")

    def test_an_order_only_one_arm_could_pack_reports_that_arm_alone(self):
        base, treat = self._pair({"a": (10.0, 0.7), "b": (12.0, 0.8)},
                                 {"a": (8.0, 0.9), "b": None})
        reports = {r.profile: r for r in compare_scenarios(base, treat, DIRECTIONS)}
        assert reports["b"].pareto_optimal == ("baseline",)
        # Named winner, and not a win: there was only one candidate to be optimal.
        assert reports["b"].winner == "baseline"

    def test_an_order_neither_arm_could_pack_produces_no_report(self):
        base, treat = self._pair({"a": (10.0, 0.7), "b": None}, {"a": (8.0, 0.9), "b": None})
        assert [r.profile for r in compare_scenarios(base, treat, DIRECTIONS)] == ["a"]

    def test_two_different_corpora_are_refused_rather_than_intersected(self):
        """Comparing the overlap would let cohort mix manufacture a difference that
        neither arm actually produced."""
        base = run_scenario("base", PIN, ("a", "b"), arm(a=(10.0, 0.7), b=(12.0, 0.8)))
        treat = run_scenario("treat", PIN, ("a", "c"), arm(a=(8.0, 0.9), c=(9.0, 0.9)))
        with pytest.raises(MismatchedOrderCorpusError, match="identical order corpus"):
            compare_scenarios(base, treat, DIRECTIONS)

    def test_the_same_orders_in_a_different_sequence_are_refused(self):
        base = run_scenario("base", PIN, ("a", "b"), arm(a=(10.0, 0.7), b=(12.0, 0.8)))
        treat = run_scenario("treat", PIN, ("b", "a"), arm(a=(8.0, 0.9), b=(11.0, 0.9)))
        with pytest.raises(MismatchedOrderCorpusError):
            compare_scenarios(base, treat, DIRECTIONS)

    def test_the_arms_may_be_pinned_to_different_catalog_versions(self):
        # That is the point of a comparison: one pin per arm, the same orders.
        treatment_pin = ScenarioVersionPin(catalog_version=2)
        base = run_scenario("base", PIN, ("a",), arm(a=(10.0, 0.7)))
        treat = ScenarioResult(scenario_id="treat", version_pin=treatment_pin,
                               order_ids=("a",),
                               runs=(OrderRunResult(order_id="a", succeeded=True,
                                                    metrics={"cost": 8.0,
                                                             "utilisation": 0.9}),))
        assert compare_scenarios(base, treat, DIRECTIONS)[0].winner == "treatment"


class TestAScenarioReportsItsOwnConfidence:
    """asks for confidence and invalid-run counts to be *visible*. They are
    properties of the result rather than something each caller re-derives, which is what
    keeps two callers from computing them differently."""

    def _mixed(self):
        return run_scenario("s", PIN, ("a", "b", "c", "d"),
                            arm(a=(10.0, 0.7), b=None, c=(12.0, 0.8), d=(9.0, 0.9)))

    def test_the_counts_partition_the_runs(self):
        result = self._mixed()
        assert (result.succeeded_count, result.failed_count) == (3, 1)
        assert result.succeeded_count + result.failed_count == len(result.runs)

    def test_confidence_is_the_usable_share(self):
        assert self._mixed().confidence == 0.75

    def test_a_wholly_successful_scenario_has_full_confidence(self):
        result = run_scenario("s", PIN, ("a",), arm(a=(10.0, 0.7)))
        assert result.confidence == 1.0
        assert result.failed_count == 0

    def test_a_wholly_failed_scenario_has_no_confidence(self):
        result = run_scenario("s", PIN, ("a",), arm(a=None))
        assert result.confidence == 0.0
        assert result.succeeded_count == 0


class TestReachingOneOrdersRawRun:
    def test_the_run_for_a_known_order_comes_back(self):
        result = run_scenario("s", PIN, ("a", "b"), arm(a=(10.0, 0.7), b=(12.0, 0.8)))
        assert result.raw_artifact("b").metrics["cost"] == 12.0

    def test_an_unknown_order_is_refused_by_name(self):
        # Returning `None` would push the mistake downstream into a metrics lookup.
        result = run_scenario("s", PIN, ("a",), arm(a=(10.0, 0.7)))
        with pytest.raises(ScenarioError, match="no run recorded for order 'zzz'"):
            result.raw_artifact("zzz")
