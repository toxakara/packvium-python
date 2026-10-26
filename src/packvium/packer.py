from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from . import bounds
from .config import PackingConfig
from .extensions import (ExtensionRegistry, SolutionScorer, UnknownObjectiveError,
                         resolve_objective_scorer, unpriceable_container)
from .fixed_placements import admit_fixed_placements
from .models import Container, Item, PackingRequest, UnratedWeightError
from .request_objects import check_objects
from .result import AlgorithmReport, PackingResult, PackingStatus, ResultFact
from .result import aggregate_termination, derive_result_facts, unsupported_placements_warning
from .solvers import Deadline, SolverOrchestrator
from contextlib import nullcontext

from .trace import TraceSink, use_trace
from .validation import IndependentSolutionValidator


class Packer:
    def __init__(
        self,
        config: PackingConfig | None = None,
        extensions: ExtensionRegistry | None = None,
        *,
        solution_scorer: SolutionScorer | None = None,
        clock: Callable[[], int] | None = None,
        trace: TraceSink | None = None,
    ):
        self.config = config or PackingConfig()
        self.extensions = extensions or ExtensionRegistry()
        self.solution_scorer = solution_scorer
        self.clock = clock
        self.trace = trace

    def _objective_bound(self, request: PackingRequest) -> "tuple[int, ...] | None":
        """The request's lower bound in the objective's score order, or `None`.

        Over every instance, fixed placements included. A custom solution scorer
        ranks by positions the bound knows nothing about, so it gets none.
        """
        if self.solution_scorer is not None:
            return None
        return bounds.for_request(request.instances, request.containers, self.config.objective)

    def pack(self, items, containers, fixed_placements=()) -> PackingResult:
        items, containers, fixed_placements = tuple(items), tuple(containers), tuple(fixed_placements)
        # The JSON path's rules, with the JSON path's reason and field: a request
        # built from objects is refused exactly where the same request as JSON would be.
        check_objects(items, containers, self.config, fixed_placements)
        request = PackingRequest(items, containers, fixed_placements)
        # Both weight objectives price the same billed weight, so both need the divisor
        # up front -- a wrong guess would silently misprice every shipment. And rating
        # some containers while others carry no tariff would rank a priced packing
        # against an unpriced one as though the unpriced were free: a missing rate table
        # is a static property of the request, unlike a billed weight past the last
        # bracket, which depends on how the search filled the box and loses a candidate
        # instead. Rust and the JavaScript fallback refuse both at admission with these
        # same sentences (review); the scorer's late checks stay as the backstop
        # for callers who bypass `Packer.pack`.
        if (
            self.config.objective in ("shipping_cost", "lowest_landed_cost")
            and self.config.dimensional_weight_divisor is None
        ):
            raise UnknownObjectiveError(
                f"the {self.config.objective} objective requires "
                f"configuration.dimensional_weight_divisor"
            )
        if self.config.objective == "lowest_landed_cost":
            unrated = next((c for c in request.containers if c.rate_table is None), None)
            if unrated is not None:
                raise UnknownObjectiveError(
                    f"the lowest_landed_cost objective requires a rate_table on every "
                    f"container; {unrated.id!r} has none"
                )
        fixed = admit_fixed_placements(
            request, self.config.minimum_support_ratio, self.config.clearance,
            self.config.max_containers,
        )
        # Once per pack, from the request alone and before the first solver runs, so that
        # computing the bound never depends on which solver won, how many starts ran or
        # how much of the deadline was left (OPTIMALITY-GAP.md F3).
        objective_bound = self._objective_bound(request)
        deadline = (
            Deadline(self.config.time_limit_ms, clock=self.clock)
            if self.clock is not None
            else Deadline(self.config.time_limit_ms)
        )
        orchestrator = SolverOrchestrator(self.extensions.placement_constraints, self.extensions.item_order_strategies, self.extensions.solvers, self.extensions.container_selector)
        # A trace passed to this Packer wins; otherwise leave any ambient
        # `with use_trace(...):` context from the caller untouched rather than
        # silently overriding it with `None`.
        scope = use_trace(self.trace) if self.trace is not None else nullcontext()
        with scope:
            portfolio = orchestrator.solve(fixed.free, request.containers, self.config, deadline,
                                           fixed.containers)
        raw_solutions = portfolio.solutions
        ranked = []
        validator = IndependentSolutionValidator()
        scorer = self.solution_scorer or resolve_objective_scorer(self.config.objective, self.config)
        for raw in raw_solutions:
            score = scorer.score(raw)
            if raw.unpacked and raw.time_limit_reached: status = PackingStatus.TIME_LIMIT
            elif raw.unpacked: status = PackingStatus.BEST_FOUND
            else: status = PackingStatus.FEASIBLE
            warnings = ()
            if self.config.validate_result:
                # Independent validation re-derives every guarantee from per-item
                # placements. A container built by GridSolver's compact fast path
                # carries a `lattice_summary` instead; expand it into the
                # identical placements the O(n) path would have built just for this
                # check -- `raw.containers` itself, and therefore the returned
                # result, stays compact.
                validation_containers = tuple(
                    replace(c, placements=c.expand_placements(), lattice_summary=None, lattice_items=())
                    if c.lattice_summary is not None else c
                    for c in raw.containers
                )
                report = validator.validate(
                    request,
                    validation_containers,
                    self.config.minimum_support_ratio,
                    self.config.clearance,
                    raw.unpacked,
                )
                if not report.valid:
                    status = PackingStatus.INVALID_RESULT; warnings = tuple(f"{i.code}: {i.detail}" for i in report.issues)
            unsupported = unsupported_placements_warning(raw.containers, self.config.minimum_support_ratio)
            if unsupported is not None:
                warnings = (*warnings, unsupported)
            starts = []
            winner_marked = False
            for record in portfolio.starts:
                selected_record = not winner_marked and record.id == raw.solver_name
                winner_marked = winner_marked or selected_record
                starts.append(type(record)(
                    record.id,
                    record.started,
                    record.completed,
                    record.truncated,
                    selected=selected_record,
                    global_deadline_reached=record.global_deadline_reached,
                ))
            starts = tuple(starts)
            termination = aggregate_termination(
                starts,
                error=status is PackingStatus.INVALID_RESULT,
            )
            if raw.effort_limit_reached and not raw.time_limit_reached:
                termination = type(termination)("effort_limit", termination.attributes)
            aggregate_timed_out = (
                termination.attributes["global_deadline_reached"]
                or raw.time_limit_reached
            )
            status, optimality = _stated_optimality(status, score, objective_bound, raw.unpacked)
            ranked.append(PackingResult(
                status,
                raw.containers,
                raw.unpacked,
                AlgorithmReport(
                    self.config.profile.value,
                    raw.solver_name,
                    deadline.elapsed_ms,
                    self.config.seed,
                    aggregate_timed_out,
                    raw.effort_limit_reached,
                    raw.stats.candidates_evaluated,
                    raw.stats.placements_attempted,
                    raw.stats.to_metrics(),
                ),
                score,
                warnings,
                termination=termination,
                optimality=optimality,
                objective=self.config.objective,
            ))
        if not ranked:
            return PackingResult(
                PackingStatus.INFEASIBLE, (), (),
                AlgorithmReport(self.config.profile.value, "none", deadline.elapsed_ms, self.config.seed),
                (0,), objective=self.config.objective,
            )
        ranked.sort(key=lambda r: (r.status is PackingStatus.INVALID_RESULT, r.score, r.algorithm.solver))
        valid_ranked = [result for result in ranked if result.status is not PackingStatus.INVALID_RESULT]
        selected = valid_ranked or ranked
        best = selected[0]
        # The search ranks an unpriceable packing worst so that any priceable alternative
        # beats it; reaching here with one still winning means no alternative existed.
        # Returning it would quote a number the carrier never published, so the run is
        # refused -- the same refusal the other three engines give, in the same words. The
        # refusal is deliberately here and not in the scorer: raising while *comparing*
        # candidates would abort runs that have a perfectly shippable answer.
        unpriceable = unpriceable_container(best.containers, self.config)
        if unpriceable is not None:
            container_id, grams, bound = unpriceable
            raise UnratedWeightError(
                f"container {container_id!r} bills at {grams} g, above its rate table's "
                f"last bracket ({bound} g); the shipment has no published price"
            )
        return PackingResult(
            best.status,
            best.containers,
            best.unpacked,
            best.algorithm,
            best.score,
            best.warnings,
            # The sentinel is a search device, never an answer -- alternatives included.
            # A runner-up the tariff cannot price is dropped before the slice, so up to
            # top_k-1 usable packings survive when priceable runners exist beyond an
            # unpriceable one (review).
            tuple(
                runner for runner in selected[1:]
                if unpriceable_container(runner.containers, self.config) is None
            )[: max(0, self.config.top_k - 1)],
            best.feasibility,
            best.termination,
            best.optimality,
            best.objective,
        )


def _stated_optimality(
    status: PackingStatus,
    score: "tuple[int, ...]",
    objective_bound: "tuple[int, ...] | None",
    unpacked: "tuple[object, ...]",
) -> "tuple[PackingStatus, ResultFact | None]":
    """The status and `optimality` fact one ranked result states (OPTIMALITY-GAP.md § 1.3).

    Attainment is the only route to `optimal`: a complete result whose score meets the bound
    on every position is proven, and nothing else is -- an exhausted candidate set covers only
    the points it generated. `None` keeps the fact derived from the status alone, which is
    every result the bound does not speak for (§ 1.5).
    """
    if objective_bound is None or status is PackingStatus.INVALID_RESULT:
        return status, None
    fields = bounds.statement(score, objective_bound)
    if fields is None:
        return status, None
    if not unpacked and "gap_key" not in fields:
        status = PackingStatus.OPTIMAL
    code = derive_result_facts(status, not unpacked, False)[2].code
    return status, ResultFact(code, fields)
