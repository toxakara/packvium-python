"""The exported simulation and recommendations API.

`docs/INTELLIGENCE-API.md` is the contract. This task is packaging, not behaviour, so
what is checked here is packaging:

  * the modules are importable under their public names, and the surface the design
    document names is actually present;
  * the workspace paths resolve to the *same objects*. `simulation/scenario.py`,
    `recommendations/engine.py` and `benchmarks/comparator/pareto_report.py` are
    re-export shims after the move; the identity assertions fail the moment one starts
    carrying its own copy. The design document's objection to a duplicate public
    spelling -- "a second definition free to drift from the first" -- is what these
    guard.

Behaviour is not re-tested here. `simulation/tests/` and `recommendations/tests/` own
it and now exercise the package through the shims, so a second copy of those assertions
would be the very duplication this file exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _require_workspace_file(relative: str) -> None:
    # The shims live in the workspace around this package; a published copy has no
    # workspace, so there is nothing whose identity could drift and nothing to check.
    if not (WORKSPACE_ROOT / relative).is_file():
        pytest.skip("the workspace re-export shims are not part of this package")


def _workspace_module(dotted: str):
    _require_workspace_file(dotted.replace(".", "/") + ".py")
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    return __import__(dotted, fromlist=["*"])


class TestTheDesignedSurfaceIsExported:
    """Every name docs/INTELLIGENCE-API.md tables as exported is reachable."""

    def test_simulation_exports_the_scenario_surface(self):
        import packvium.simulation as simulation

        for name in ("ScenarioVersionPin", "OrderRunResult", "ScenarioResult",
                     "run_scenario", "compare_scenarios",
                     "ScenarioError", "MismatchedOrderCorpusError"):
            assert hasattr(simulation, name), name

    def test_recommendations_exports_the_proposal_surface(self):
        import packvium.recommendations as recommendations

        for name in ("ExpectedDelta", "Recommendation", "ApprovalRecord",
                     "propose_recommendation", "approve",
                     "approve_catalog", "approve_policy", "RecommendationError"):
            assert hasattr(recommendations, name), name

    def test_the_comparison_type_a_caller_receives_is_importable(self):
        # `compare_scenarios` returns `tuple[ProfileReport, ...]`, so the type is part of
        # a public signature. It could not stay in `benchmarks/`, which is never exported.
        from packvium.pareto import ProfileReport
        from packvium.simulation import ProfileReport as re_exported

        assert re_exported is ProfileReport


class TestWorkspaceShimsReExportTheSameObjects:
    def test_scenario_shim_is_not_a_second_implementation(self):
        import packvium.simulation as canonical

        shim = _workspace_module("simulation.scenario")
        assert shim.run_scenario is canonical.run_scenario
        assert shim.compare_scenarios is canonical.compare_scenarios
        assert shim.ScenarioResult is canonical.ScenarioResult

    def test_recommendations_shim_is_not_a_second_implementation(self):
        import packvium.recommendations as canonical

        shim = _workspace_module("recommendations.engine")
        assert shim.propose_recommendation is canonical.propose_recommendation
        assert shim.approve_catalog is canonical.approve_catalog
        assert shim.Recommendation is canonical.Recommendation

    def test_pareto_shim_is_not_a_second_implementation(self):
        import packvium.pareto as canonical

        # The benchmark comparators keep importing this path and have no reason to know
        # where the implementation moved; what they must not get is a second dominance
        # rule, which would make a benchmark score against something a caller never sees.
        _require_workspace_file("benchmarks/comparator/pareto_report.py")
        sys.path.insert(0, str(WORKSPACE_ROOT / "benchmarks" / "comparator"))
        import pareto_report as shim

        assert shim.generate_report is canonical.generate_report
        assert shim.dominates is canonical.dominates
        assert shim.ProfileReport is canonical.ProfileReport


class TestHistoricalReplayIsExportedWithItsLedger:
    """. `evaluate_on_history` could not ship until the ledger did: its central
    argument is an `OutcomeLedger`, and a function whose main argument a consumer cannot
    construct is worse than an unexported one."""

    def test_the_ledger_and_the_replay_are_both_importable(self):
        from packvium.holdout import HoldoutEvaluation, evaluate_on_history
        from packvium.outcomes import OutcomeEvent, OutcomeEventType, OutcomeLedger

        assert callable(evaluate_on_history)
        assert hasattr(HoldoutEvaluation, "improved")
        assert callable(OutcomeLedger().record)
        assert OutcomeEvent and OutcomeEventType

    def test_the_ledger_shim_is_not_a_second_implementation(self):
        import packvium.outcomes as canonical

        shim = _workspace_module("domain.outcomes.model")
        assert shim.OutcomeLedger is canonical.OutcomeLedger
        assert shim.OutcomeEvent is canonical.OutcomeEvent

    def test_the_replay_shim_is_not_a_second_implementation(self):
        import packvium.holdout as canonical

        shim = _workspace_module("recommendations.holdout")
        assert shim.evaluate_on_history is canonical.evaluate_on_history
        assert shim.ValidationVerdict is canonical.ValidationVerdict

    def test_the_ledger_public_surface_stays_closed(self):
        """The guard `domain/outcomes/tests` holds, re-asserted where the code now lives.

        A new method on the ledger is a decision to widen a published surface, and has to
        be made deliberately rather than by arriving.
        """
        from packvium.outcomes import OutcomeLedger

        assert {n for n in dir(OutcomeLedger) if not n.startswith("_")} == {
            "record", "events_for_decision", "current_view", "view_as_of",
        }

    def test_the_validator_gate_survives_the_move(self):
        """A rejected packing can never be recorded as an improvement, in the package too."""
        import pytest

        from packvium.holdout import IMPROVED, DecisionOutcome

        with pytest.raises(ValueError, match="never be recorded as an improvement"):
            DecisionOutcome(decision_id="d1", verdict=IMPROVED,
                            baseline_valid=True, treatment_valid=False,
                            treatment_codes=("unsupported_overhang",))


class TestTheRegistryRouteStaysStructural:
    def test_a_proposal_is_handed_no_registry(self):
        """`propose_recommendation` has no path to a registry, and that is structural.

        docs/INTELLIGENCE-API.md makes this the API's whole obligation: nothing exported
        may write to a registry except through `approve*`. A proposal that could reach
        one would make that a documented promise rather than an enforced one.
        """
        import inspect

        from packvium.recommendations import propose_recommendation

        parameters = inspect.signature(propose_recommendation).parameters
        assert not [p for p in parameters if "registry" in p.lower()], sorted(parameters)

    def test_a_recommendation_cannot_exist_without_a_rollback_plan(self):
        import inspect

        from packvium.recommendations import propose_recommendation

        rollback = inspect.signature(propose_recommendation).parameters["rollback_plan"]
        assert rollback.default is inspect.Parameter.empty
