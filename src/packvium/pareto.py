"""A Pareto report over benchmark results, one per profile.

"A single blended score hides the trade-off" (this task's own description) -- a report
that turns utilisation, runtime and container count into one weighted number can name a
"winner" that is worse on the one axis a caller actually cares about. This module never
computes such a number. A result **dominates** another only if it is no worse on every
named axis and strictly better on at least one; the report per profile is the set of
results nothing else dominates (the Pareto frontier), not a ranking.

"The report names a winner per profile" is read literally but not force-fitted: when
exactly one result survives as non-dominated, that is the named winner. When more than
one survives -- a genuine trade-off, e.g. one engine faster and another denser, neither
strictly better than the other -- the report says so explicitly (`winner is None`,
`pareto_optimal` lists all of them) rather than breaking the tie with an arbitrary or
blended pick, which would be exactly the single-number collapse this task rejects.

Scope: this module is pure report generation over caller-supplied metrics -- it does not
itself run competitors or collect their results (still TODO) or filter by
's validator (a disqualified candidate must be excluded by the caller before it
ever reaches this module; `generate_report` assumes every candidate it receives already
passed validation).
"""

from __future__ import annotations

import math
from ._compat import dataclass
from typing import Mapping, Optional, Sequence


class ParetoReportError(Exception):
    """Base class for every error raised by this module."""


class InconsistentAxesError(ParetoReportError):
    """Two candidates being compared do not report the same set of metric axes -- an
    axis missing from one side would silently drop out of the dominance comparison
    rather than being caught."""


class NonFiniteMetricError(ParetoReportError):
    """A metric is `NaN`, which cannot take part in a dominance comparison.

    Dominance is decided by `>` and `<`, and **both are false for `NaN`** -- so an axis
    carrying one sets neither `better` nor `worse` and is silently counted as *equal*.
    Measured before this refusal existed: a candidate whose metrics were all `NaN` came back
    on the Pareto frontier beside a clean one, because nothing could dominate it. Presenting
    garbage as an optimal trade-off is the one output this module exists to prevent.

    This is the same judgement `InconsistentAxesError` already makes -- a silent partial
    comparison is worse than an explicit error -- applied to an axis that is present but
    uncomparable rather than absent.

    **Infinities are allowed, deliberately.** `inf` and `-inf` are ends of the number line,
    not holes in it: `inf > 5` is true, `5 > inf` is false, and `inf` against `inf` is
    correctly neither better nor worse. A caller encoding "unpriceable" or "timed out" as an
    infinite cost gets exactly the dominance answer they mean, so refusing those would break
    a reasonable use for the sake of a tidier rule. `NaN` is refused because it is not a
    value; infinities are kept because they are.
    """


def _refuse_nan(metrics: Mapping[str, float], where: str) -> None:
    """Guard for one candidate's metrics. O(axes), which is the size of `higher_is_better`.

    Named `where` rather than positional, because the error a caller sees has to say which
    candidate carried the bad value -- an evaluator producing `NaN` is a bug upstream of this
    module, and a message that does not point at it just moves the search.
    """
    for axis, value in metrics.items():
        if isinstance(value, float) and math.isnan(value):
            raise NonFiniteMetricError(f"{where}: metric {axis!r} is NaN and cannot be compared")


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """One engine's result on one profile, already validated by the caller."""

    profile: str
    engine: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile is required")
        if not self.engine:
            raise ValueError("engine is required")
        if not self.metrics:
            raise ValueError("metrics cannot be empty")
        # At construction rather than at comparison: an invalid candidate then cannot exist,
        # so `generate_report` and `_pareto_frontier` inherit the guarantee without paying
        # for it once per pair.
        _refuse_nan(self.metrics, f"{self.engine} on profile {self.profile}")


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """One profile's Pareto frontier. `winner` is set only when exactly one candidate
    survived as non-dominated; a genuine multi-way trade-off leaves it `None` and lists
    every surviving engine in `pareto_optimal` instead."""

    profile: str
    pareto_optimal: tuple[str, ...]
    dominated: tuple[str, ...]
    winner: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile is required")
        if self.winner is not None and self.winner not in self.pareto_optimal:
            raise ValueError("winner, when set, must be one of the Pareto-optimal engines")
        if self.winner is not None and len(self.pareto_optimal) != 1:
            raise ValueError("winner can only be set when exactly one candidate is Pareto-optimal")


def dominates(a: Mapping[str, float], b: Mapping[str, float], higher_is_better: Mapping[str, bool]) -> bool:
    """Whether `a` dominates `b`: no worse than `b` on every axis in `higher_is_better`,
    and strictly better on at least one. Raises `InconsistentAxesError` if either side
    is missing an axis `higher_is_better` names, and `NonFiniteMetricError` if either
    carries a `NaN` -- in both cases because a silent partial comparison is worse than an
    explicit error. Infinities are accepted and compare as the ends of the number line."""
    missing = set(higher_is_better) - set(a) | set(higher_is_better) - set(b)
    if missing:
        raise InconsistentAxesError(f"missing metric axis/axes: {sorted(missing)}")
    # `dominates` is public and takes bare mappings, so it cannot rely on `CandidateResult`
    # having already validated them.
    _refuse_nan(a, "left candidate")
    _refuse_nan(b, "right candidate")
    at_least_as_good = True
    strictly_better_somewhere = False
    for axis, higher_wins in higher_is_better.items():
        a_value, b_value = a[axis], b[axis]
        better = a_value > b_value if higher_wins else a_value < b_value
        worse = a_value < b_value if higher_wins else a_value > b_value
        if worse:
            at_least_as_good = False
        if better:
            strictly_better_somewhere = True
    return at_least_as_good and strictly_better_somewhere


def _pareto_frontier(candidates: Sequence[CandidateResult], higher_is_better: Mapping[str, bool]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    optimal: list[str] = []
    dominated: list[str] = []
    for candidate in candidates:
        is_dominated = any(
            other is not candidate and dominates(other.metrics, candidate.metrics, higher_is_better)
            for other in candidates
        )
        (dominated if is_dominated else optimal).append(candidate.engine)
    return tuple(sorted(optimal)), tuple(sorted(dominated))


def generate_report(candidates: Sequence[CandidateResult], higher_is_better: Mapping[str, bool]) -> tuple[ProfileReport, ...]:
    """One `ProfileReport` per distinct `profile` among `candidates`, sorted by profile
    name for a deterministic report order. `higher_is_better` must be supplied
    explicitly for every metric axis used -- there is no default direction, since
    guessing wrong (e.g. treating runtime as "higher is better") would silently invert
    the whole report."""
    if not higher_is_better:
        raise ValueError("higher_is_better must name at least one metric axis")
    by_profile: dict[str, list[CandidateResult]] = {}
    for candidate in candidates:
        by_profile.setdefault(candidate.profile, []).append(candidate)

    reports = []
    for profile in sorted(by_profile):
        group = by_profile[profile]
        engine_names = [c.engine for c in group]
        if len(set(engine_names)) != len(engine_names):
            raise ParetoReportError(f"profile {profile!r} has more than one result for the same engine")
        optimal, dominated = _pareto_frontier(group, higher_is_better)
        winner = optimal[0] if len(optimal) == 1 else None
        reports.append(ProfileReport(profile=profile, pareto_optimal=optimal, dominated=dominated, winner=winner))
    return tuple(reports)
