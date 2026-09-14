"""Domain model for carton-catalog and rule-change recommendations.

A recommendation proposes a carton or rule change, backed by a `packvium.simulation`
baseline-vs-treatment comparison over a real order cohort: it names exactly
which orders support it, what changed and by how much per metric, how confident that
evidence is, what constraints bound it, and how to undo it if approved. It is never an
automatic mutation of the production catalog -- `propose_recommendation` has no method
that could write to any registry; only a separate, explicit `approve()` call can, and
only through a caller-injected `publish` callback, the same dependency-injection shape
`packvium.simulation`'s `OrderEvaluator` uses.

"Recommendations with insufficient evidence are suppressed": `propose_recommendation`
returns `None` rather than a `Recommendation` when the paired successful cohort is too
small or too unreliable (its share of all attempted orders falls below an explicit
caller-supplied threshold) -- suppression
is a `None` return, not an exception a caller might mishandle, and not a `Recommendation`
object with a misleadingly low confidence value quietly attached.
"""

from __future__ import annotations

from ._compat import dataclass
from typing import Callable, Mapping, Optional, Sequence

from .commerce.catalog import CatalogRegistry, CatalogSnapshot
from .commerce.policy import (
    PolicyAction, PolicyPredicate, PolicyRegistry, PolicyRule, PolicyScope,
)
from .simulation import MismatchedOrderCorpusError, ScenarioResult, compare_scenarios


class RecommendationError(Exception):
    """Base class for every recommendation-domain error raised by this module."""


@dataclass(frozen=True, slots=True)
class ExpectedDelta:
    """One metric's change from baseline to treatment, averaged over every order both
    scenarios actually succeeded on (a failed order contributes to neither side's
    average, the same exclusion `packvium.simulation`'s `compare_scenarios` already
    applies per order)."""

    metric: str
    baseline_mean: float
    treatment_mean: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric is required")

    @property
    def delta(self) -> float:
        return self.treatment_mean - self.baseline_mean


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One reviewable proposal. Never mutates anything by existing -- see the module
    docstring."""

    recommendation_id: str
    proposal: str
    supporting_order_ids: tuple[str, ...]
    expected_deltas: tuple[ExpectedDelta, ...]
    confidence: float
    constraints: tuple[str, ...]
    rollback_plan: str

    def __post_init__(self) -> None:
        if not self.recommendation_id:
            raise ValueError("recommendation_id is required")
        if not self.proposal:
            raise ValueError("proposal is required")
        if not self.supporting_order_ids:
            raise ValueError("a recommendation must cite at least one supporting order")
        if not self.expected_deltas:
            raise ValueError("a recommendation must report at least one expected delta")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.constraints or any(not constraint for constraint in self.constraints):
            raise ValueError("a recommendation must state at least one non-empty constraint")
        if not self.rollback_plan:
            raise ValueError("a recommendation must state its rollback plan")


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """The immutable record of one approval: which recommendation, when, and which new
    registry version publishing it actually produced."""

    recommendation_id: str
    approved_at: int
    published_version: int

    def __post_init__(self) -> None:
        if not self.recommendation_id:
            raise ValueError("recommendation_id is required")
        if self.approved_at < 0:
            raise ValueError("approved_at cannot be negative")
        if self.published_version <= 0:
            raise ValueError("published_version must be positive")


def propose_recommendation(
    recommendation_id: str, proposal: str, baseline: ScenarioResult, treatment: ScenarioResult,
    *, constraints: tuple[str, ...], rollback_plan: str, minimum_cohort_size: int, minimum_confidence: float,
) -> Optional[Recommendation]:
    """Build a `Recommendation` from a baseline/treatment comparison, or return `None`
    if the evidence behind it is too thin. `minimum_cohort_size` and
    `minimum_confidence` are required explicitly -- there is no built-in default, since
    what counts as "enough evidence" is a product decision this module does not make on
    a caller's behalf.
    """
    if baseline.order_ids != treatment.order_ids:
        raise MismatchedOrderCorpusError(
            "baseline and treatment must have run the identical order corpus, in the same order"
        )
    if minimum_cohort_size <= 0:
        raise ValueError("minimum_cohort_size must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0 and 1")
    # A delta is a paired comparison, not the difference between two independently
    # filtered populations. If the arms fail on different orders, averaging each arm's
    # surviving orders separately can manufacture an improvement from cohort mix alone.
    # Build each metric from the exact same successful order pairs. This is O(o * k) time
    # and O(o * k) retained values for o orders and at most k common metrics per order.
    paired: dict[str, tuple[list[float], list[float]]] = {}
    supporting_order_ids: list[str] = []
    for baseline_run, treatment_run in zip(baseline.runs, treatment.runs):
        if not baseline_run.succeeded or not treatment_run.succeeded:
            continue
        supporting_order_ids.append(baseline_run.order_id)
        for metric in baseline_run.metrics.keys() & treatment_run.metrics.keys():
            baseline_values, treatment_values = paired.setdefault(metric, ([], []))
            baseline_values.append(baseline_run.metrics[metric])
            treatment_values.append(treatment_run.metrics[metric])

    # Evidence size is the paired cohort, not the number of orders attempted in either
    # arm. Otherwise two mostly-disjoint survivor sets could advertise a large cohort and
    # high confidence while a delta rested on one shared order.
    comparable_count = len(supporting_order_ids)
    confidence = comparable_count / len(baseline.order_ids)
    if comparable_count < minimum_cohort_size or confidence < minimum_confidence:
        return None

    deltas = []
    for metric in sorted(paired):
        baseline_values, treatment_values = paired[metric]
        deltas.append(ExpectedDelta(
            metric=metric,
            baseline_mean=sum(baseline_values) / len(baseline_values),
            treatment_mean=sum(treatment_values) / len(treatment_values),
        ))
    if not deltas:
        return None  # no metric both sides actually produced -- nothing to recommend from

    return Recommendation(
        recommendation_id=recommendation_id,
        proposal=proposal,
        supporting_order_ids=tuple(supporting_order_ids),
        expected_deltas=tuple(deltas),
        confidence=confidence,
        constraints=constraints,
        rollback_plan=rollback_plan,
    )


def approve(recommendation: Recommendation, *, at: int, publish: Callable[[], int]) -> ApprovalRecord:
    """The only path from a `Recommendation` to an actual registry change: calls the
    caller-injected `publish` exactly once and records the version it returns. Nothing
    in `propose_recommendation` above can reach this -- a recommendation existing is
    never itself a mutation."""
    published_version = publish()
    return ApprovalRecord(
        recommendation_id=recommendation.recommendation_id, approved_at=at, published_version=published_version,
    )


def approve_catalog(
    recommendation: Recommendation,
    registry: CatalogRegistry,
    snapshot: CatalogSnapshot,
    *,
    approved_at: int,
    effective_at: int,
) -> ApprovalRecord:
    """Publish an approved catalog proposal through the real append-only registry."""
    version = registry.publish(
        snapshot,
        published_at=approved_at,
        effective_at=effective_at,
        note=f"approved recommendation {recommendation.recommendation_id}",
    )
    return ApprovalRecord(recommendation.recommendation_id, approved_at, version.number)


def approve_policy(
    recommendation: Recommendation,
    registry: PolicyRegistry,
    *,
    rule_id: str,
    scope: PolicyScope,
    action: PolicyAction,
    predicates: Sequence[PolicyPredicate],
    priority: int,
    approved_at: int,
    effective_at: int,
) -> ApprovalRecord:
    """Publish an approved policy proposal through the real append-only registry."""
    rule: PolicyRule = registry.publish(
        rule_id,
        scope=scope,
        action=action,
        predicates=predicates,
        priority=priority,
        effective_at=effective_at,
        reason=f"approved recommendation {recommendation.recommendation_id}",
    )
    return ApprovalRecord(recommendation.recommendation_id, approved_at, rule.version)
