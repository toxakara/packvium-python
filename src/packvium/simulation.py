"""Domain model for scenario simulation and what-if comparison.

A scenario runs a fixed order corpus against one *version pin* -- an explicit,
recorded set of catalog/tariff/policy versions (the same append-only version numbers
`packvium.commerce.catalog`, `packvium.commerce.rating` and `packvium.commerce.policy`
already hand out) -- and records what happened, order by order, without mutating
anything those registries hold. Two scenarios (baseline and treatment) compare only when
they ran the *identical* order corpus; the comparison itself is a Pareto report
(`packvium.pareto`), never a single blended number, for the same reason that module
rejects one.

This module deliberately does not itself price, validate, or pack anything: `run_scenario`
takes an injected `OrderEvaluator` callable -- the caller supplies a function that
actually calls the catalog/rating/policy registries and the solver, keeping simulation
orchestration decoupled from what it orchestrates. Given a deterministic evaluator, a
scenario is byte-for-byte reproducible purely from its own recorded `ScenarioVersionPin`
and `order_ids` -- proven directly by re-running the same evaluator twice, not assumed.
"""

from __future__ import annotations

from ._compat import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

from .pareto import CandidateResult, ProfileReport, generate_report


class ScenarioError(Exception):
    """Base class for every scenario-domain error raised by this module."""


class MismatchedOrderCorpusError(ScenarioError):
    """Baseline and treatment did not run the identical order corpus -- comparing them
    would not isolate the effect of the version pin, so this is refused rather than
    silently compared anyway."""


@dataclass(frozen=True, slots=True)
class ScenarioVersionPin:
    """The exact, recorded configuration one scenario ran against. Every field is an
    explicit version number (or `None` if that registry was not consulted) -- never
    "current", so a stored `ScenarioResult` can be replayed byte-for-byte regardless of
    what any registry's history has grown to since (the same reproducibility guarantee
    `commerce/rating/model.py`'s `rate_with_version` gives one rate lookup)."""

    catalog_version: Optional[int] = None
    tariff_version: Optional[int] = None
    policy_version: Optional[int] = None
    policy_versions: tuple[tuple[str, int], ...] = ()
    solver_version: Optional[str] = None

    def __post_init__(self) -> None:
        if all(v is None for v in (self.catalog_version, self.tariff_version, self.policy_version, self.solver_version)) \
                and not self.policy_versions:
            raise ValueError("a version pin must set at least one of its fields")
        numeric = (self.catalog_version, self.tariff_version, self.policy_version)
        if any(value is not None and value <= 0 for value in numeric):
            raise ValueError("version numbers must be positive")
        if any(not rule_id or version <= 0 for rule_id, version in self.policy_versions):
            raise ValueError("policy version pins require a rule id and positive version")
        if len({rule_id for rule_id, _ in self.policy_versions}) != len(self.policy_versions):
            raise ValueError("a version pin cannot name the same policy rule twice")
        object.__setattr__(self, "policy_versions", tuple(sorted(self.policy_versions)))
        if self.solver_version is not None and not self.solver_version:
            raise ValueError("solver_version must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class OrderRunResult:
    """One order's outcome under one version pin. `metrics` are Pareto axes (material
    cost, shipping cost, utilisation, a damage-risk proxy, runtime, ...) supplied by the
    caller's evaluator -- this module never computes or invents them."""

    order_id: str
    succeeded: bool
    metrics: Mapping[str, float] = ()
    failure_reason: Optional[str] = None
    artifact: Any = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id is required")
        if self.succeeded and not self.metrics:
            raise ValueError("a succeeded run must report at least one metric")
        if not self.succeeded and not self.failure_reason:
            raise ValueError("a failed run must record a failure_reason")
        metrics = dict(self.metrics)
        if any(not name for name in metrics):
            raise ValueError("metric names must be non-empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in metrics.values()
        ):
            raise ValueError("metric values must be finite")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


#: `(order_id, version_pin) -> OrderRunResult`. Supplied by the caller -- keeps pricing,
#: policy evaluation and packing out of this orchestration layer.
OrderEvaluator = Callable[[str, ScenarioVersionPin], OrderRunResult]


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """The complete, immutable record of one scenario run: which version pin, which
    orders (the corpus itself, recorded -- not just its length), and every order's raw
    outcome, retained in full (acceptance: "raw artifacts are retained")."""

    scenario_id: str
    version_pin: ScenarioVersionPin
    order_ids: tuple[str, ...]
    runs: tuple[OrderRunResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_ids", tuple(self.order_ids))
        object.__setattr__(self, "runs", tuple(self.runs))
        if not self.scenario_id:
            raise ValueError("scenario_id is required")
        if not self.order_ids:
            raise ValueError("order_ids cannot be empty")
        if tuple(run.order_id for run in self.runs) != self.order_ids:
            raise ValueError("runs must correspond 1:1 to order_ids, in the same order")

    @property
    def succeeded_count(self) -> int:
        return sum(1 for run in self.runs if run.succeeded)

    @property
    def failed_count(self) -> int:
        return sum(1 for run in self.runs if not run.succeeded)

    @property
    def confidence(self) -> float:
        """The fraction of orders that produced a usable outcome -- 's
        "confidence and invalid-run counts are visible", made a first-class,
        always-computed property rather than something a caller must derive."""
        return self.succeeded_count / len(self.runs)

    def raw_artifact(self, order_id: str) -> OrderRunResult:
        for run in self.runs:
            if run.order_id == order_id:
                return run
        raise ScenarioError(f"no run recorded for order {order_id!r}")


def run_scenario(
    scenario_id: str, version_pin: ScenarioVersionPin, order_ids: Sequence[str], evaluator: OrderEvaluator,
) -> ScenarioResult:
    """Run every order in `order_ids` through `evaluator` under `version_pin`, in
    order. Given a deterministic `evaluator`, calling this twice with the same
    arguments reproduces a byte-identical `ScenarioResult` -- 's "a scenario is
    fully reproducible from stored versions" is this function's own determinism, not a
    separate mechanism bolted on afterward."""
    runs = tuple(evaluator(order_id, version_pin) for order_id in order_ids)
    return ScenarioResult(scenario_id=scenario_id, version_pin=version_pin, order_ids=tuple(order_ids), runs=runs)


def compare_scenarios(
    baseline: ScenarioResult, treatment: ScenarioResult, higher_is_better: Mapping[str, bool],
) -> tuple[ProfileReport, ...]:
    """Compare two scenarios order by order via 's Pareto report -- never a
    single blended delta. Each order becomes its own "profile" (so a caller sees exactly
    which orders improved, regressed, or traded off, not just an aggregate), and
    "baseline"/"treatment" are the two "engines" compared within it. Only orders both
    scenarios actually succeeded on are compared; a failed run has no metrics to compare
    with and is excluded from that order's report rather than crashing the comparison.
    """
    if baseline.order_ids != treatment.order_ids:
        raise MismatchedOrderCorpusError(
            "baseline and treatment must have run the identical order corpus, in the same order"
        )
    candidates: list[CandidateResult] = []
    # ScenarioResult already proves that runs correspond 1:1 to order_ids in order, and
    # the corpus equality check above proves both arms have the same order. Pair them in
    # one pass instead of calling raw_artifact twice per order (which made this O(o^2)).
    # The orchestration outside generate_report is now O(o) time and O(o) candidates.
    for base_run, treat_run in zip(baseline.runs, treatment.runs):
        order_id = base_run.order_id
        if base_run.succeeded:
            candidates.append(CandidateResult(profile=order_id, engine="baseline", metrics=base_run.metrics))
        if treat_run.succeeded:
            candidates.append(CandidateResult(profile=order_id, engine="treatment", metrics=treat_run.metrics))
    return generate_report(candidates, higher_is_better)
