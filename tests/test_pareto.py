"""Dominance and the report built from it, through the package's own import path.

`packvium.pareto` is exercised in the workspace by the benchmark comparators, which reach
it through a re-export shim. That proves the shim works; it says nothing about the module
a consumer imports, and `tests/` ships inside the wheel — so a consumer running the
shipped suite exercised none of this until these tests existed.

What is asserted here is the module's one job: **never collapse a trade-off into a single
number.** A result dominates another only when it is no worse on every named axis and
strictly better on at least one, and when two results survive that, the report says so
instead of picking.
"""

from __future__ import annotations

import pytest

from packvium.pareto import (
    CandidateResult,
    InconsistentAxesError,
    NonFiniteMetricError,
    ParetoReportError,
    ProfileReport,
    dominates,
    generate_report,
)

#: Cost is cheaper-is-better, utilisation is higher-is-better. Stating both directions is
#: mandatory in this module, so every test carries them.
DIRECTIONS = {"cost": False, "utilisation": True}


def candidate(engine: str, cost: float, utilisation: float, profile: str = "balanced"):
    return CandidateResult(profile=profile, engine=engine,
                           metrics={"cost": cost, "utilisation": utilisation})


class TestDominance:
    def test_better_on_both_axes_dominates(self):
        assert dominates({"cost": 8.0, "utilisation": 0.9},
                         {"cost": 10.0, "utilisation": 0.7}, DIRECTIONS)

    def test_worse_on_both_axes_does_not(self):
        assert not dominates({"cost": 12.0, "utilisation": 0.6},
                             {"cost": 10.0, "utilisation": 0.7}, DIRECTIONS)

    def test_a_genuine_trade_off_dominates_in_neither_direction(self):
        """Cheaper but sparser. This is the case the whole module exists for: a blended
        score would rank these, and ranking them is a claim about the caller's priorities
        that nothing in the metrics supports."""
        cheap = {"cost": 8.0, "utilisation": 0.6}
        dense = {"cost": 12.0, "utilisation": 0.9}
        assert not dominates(cheap, dense, DIRECTIONS)
        assert not dominates(dense, cheap, DIRECTIONS)

    def test_identical_metrics_dominate_in_neither_direction(self):
        # "No worse on every axis" is satisfied; "strictly better somewhere" is not.
        same = {"cost": 10.0, "utilisation": 0.8}
        assert not dominates(same, dict(same), DIRECTIONS)

    def test_equal_on_one_axis_and_better_on_the_other_dominates(self):
        assert dominates({"cost": 10.0, "utilisation": 0.9},
                         {"cost": 10.0, "utilisation": 0.8}, DIRECTIONS)

    def test_direction_is_read_from_the_map_rather_than_guessed(self):
        """The same two candidates, with the meaning of `cost` inverted. Nothing about the
        numbers says which way is better, which is why the map has no default."""
        low, high = {"cost": 8.0}, {"cost": 12.0}
        assert dominates(low, high, {"cost": False})
        assert dominates(high, low, {"cost": True})

    def test_an_axis_missing_from_either_side_is_refused(self):
        with pytest.raises(InconsistentAxesError, match="utilisation"):
            dominates({"cost": 8.0}, {"cost": 10.0, "utilisation": 0.7}, DIRECTIONS)
        with pytest.raises(InconsistentAxesError, match="utilisation"):
            dominates({"cost": 8.0, "utilisation": 0.9}, {"cost": 10.0}, DIRECTIONS)

    def test_an_axis_absent_from_the_direction_map_is_simply_not_compared(self):
        # Extra metrics are not an error; they are data the caller did not ask to rank on.
        assert dominates({"cost": 8.0, "weight": 99.0}, {"cost": 10.0, "weight": 1.0},
                         {"cost": False})

    def test_a_nan_on_either_side_is_refused_rather_than_counted_equal(self):
        """`NaN > x` and `NaN < x` are both false, so an unguarded comparison silently
        reports the axis as a tie and lets garbage onto the frontier."""
        with pytest.raises(NonFiniteMetricError, match="left candidate.*cost"):
            dominates({"cost": float("nan")}, {"cost": 1.0}, {"cost": False})
        with pytest.raises(NonFiniteMetricError, match="right candidate.*cost"):
            dominates({"cost": 1.0}, {"cost": float("nan")}, {"cost": False})

    def test_infinities_compare_as_the_ends_of_the_number_line(self):
        """Deliberately allowed: a caller encoding "unpriceable" as an infinite cost gets
        the answer they mean."""
        assert dominates({"cost": 5.0}, {"cost": float("inf")}, {"cost": False})
        assert not dominates({"cost": float("inf")}, {"cost": 5.0}, {"cost": False})
        assert not dominates({"cost": float("inf")}, {"cost": float("inf")}, {"cost": False})


class TestTheReport:
    def test_one_dominant_candidate_becomes_the_named_winner(self):
        reports = generate_report(
            [candidate("python", 10.0, 0.7), candidate("php", 8.0, 0.9)], DIRECTIONS)
        assert len(reports) == 1
        assert reports[0].winner == "php"
        assert reports[0].pareto_optimal == ("php",)
        assert reports[0].dominated == ("python",)

    def test_a_trade_off_leaves_no_winner_and_lists_both(self):
        reports = generate_report(
            [candidate("python", 8.0, 0.6), candidate("php", 12.0, 0.9)], DIRECTIONS)
        assert reports[0].winner is None
        assert reports[0].pareto_optimal == ("php", "python")
        assert reports[0].dominated == ()

    def test_a_single_candidate_is_optimal_by_default(self):
        """Nothing can dominate it, so it is on the frontier. Worth pinning because it is
        easy to misread as evidence that it beat something."""
        reports = generate_report([candidate("python", 10.0, 0.7)], DIRECTIONS)
        assert reports[0].pareto_optimal == ("python",)
        assert reports[0].winner == "python"

    def test_three_candidates_split_into_frontier_and_dominated(self):
        reports = generate_report([
            candidate("cheap", 8.0, 0.6),
            candidate("dense", 12.0, 0.9),
            candidate("worse", 14.0, 0.5),
        ], DIRECTIONS)
        assert reports[0].pareto_optimal == ("cheap", "dense")
        assert reports[0].dominated == ("worse",)
        assert reports[0].winner is None

    def test_each_profile_gets_its_own_report_sorted_by_name(self):
        reports = generate_report([
            candidate("python", 10.0, 0.7, profile="quality"),
            candidate("php", 8.0, 0.9, profile="fast"),
        ], DIRECTIONS)
        assert [r.profile for r in reports] == ["fast", "quality"]

    def test_frontier_and_dominated_are_each_sorted(self):
        """Report order cannot depend on input order, or two runs of the same comparison
        would print differently."""
        forward = generate_report(
            [candidate("zeta", 8.0, 0.6), candidate("alpha", 12.0, 0.9)], DIRECTIONS)
        reverse = generate_report(
            [candidate("alpha", 12.0, 0.9), candidate("zeta", 8.0, 0.6)], DIRECTIONS)
        assert forward[0].pareto_optimal == reverse[0].pareto_optimal == ("alpha", "zeta")

    def test_no_candidates_produces_no_reports(self):
        assert generate_report([], DIRECTIONS) == ()

    def test_two_results_for_one_engine_in_one_profile_are_refused(self):
        with pytest.raises(ParetoReportError, match="more than one result"):
            generate_report([candidate("python", 10.0, 0.7), candidate("python", 8.0, 0.9)],
                            DIRECTIONS)

    def test_an_empty_direction_map_is_refused(self):
        # With no axes there is no dominance, so every candidate would be "optimal".
        with pytest.raises(ValueError, match="at least one metric axis"):
            generate_report([candidate("python", 10.0, 0.7)], {})

    def test_a_nan_candidate_cannot_reach_the_frontier(self):
        """Measured before the refusal existed: an all-`NaN` candidate came back optimal
        beside a clean one, because nothing could dominate it."""
        with pytest.raises(NonFiniteMetricError):
            generate_report([candidate("python", 10.0, 0.7),
                             candidate("broken", float("nan"), float("nan"))], DIRECTIONS)


class TestTheReportTypeGuardsItsOwnClaims:
    def test_a_winner_must_be_on_the_frontier(self):
        with pytest.raises(ValueError, match="must be one of the Pareto-optimal"):
            ProfileReport(profile="fast", pareto_optimal=("php",), dominated=("python",),
                          winner="python")

    def test_a_winner_cannot_be_named_when_two_survive(self):
        with pytest.raises(ValueError, match="exactly one candidate"):
            ProfileReport(profile="fast", pareto_optimal=("php", "python"), dominated=(),
                          winner="php")

    def test_no_winner_alongside_a_frontier_of_two_is_well_formed(self):
        report = ProfileReport(profile="fast", pareto_optimal=("php", "python"), dominated=())
        assert report.winner is None


def test_prepared_frontiers_match_independent_pairwise_comparison():
    import random

    rng = random.Random(914)
    values = [-float('inf'), -(10 ** 100), -1, -0.0, 0, 1, 10 ** 100, float('inf')]
    for dimension in (1, 2, 3, 5):
        for scene in range(30):
            directions = {f'axis{i}': bool(rng.randrange(2)) for i in range(dimension)}
            candidates = [CandidateResult('p', str(i), {axis: rng.choice(values) for axis in directions})
                          for i in range(2 + scene * 2)]
            expected = tuple(sorted(candidate.engine for candidate in candidates
                                    if not any(other is not candidate and dominates(other.metrics, candidate.metrics, directions)
                                               for other in candidates)))
            report, = generate_report(candidates, directions)
            assert report.pareto_optimal == expected
            assert report.dominated == tuple(sorted(set(c.engine for c in candidates) - set(expected)))
            assert generate_report(list(reversed(candidates)), directions) == (report,)
