from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .config import PackingConfig
from .extensions import ExtensionRegistry, SolutionScorer, resolve_objective_scorer
from .models import Container, Item, PackingRequest
from .result import AlgorithmReport, PackingResult, PackingStatus
from .result import aggregate_termination
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

    def pack(self, items, containers) -> PackingResult:
        request = PackingRequest(tuple(items), tuple(containers))
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
            portfolio = orchestrator.solve(request.instances, request.containers, self.config, deadline)
        raw_solutions = portfolio.solutions
        ranked = []
        validator = IndependentSolutionValidator()
        scorer = self.solution_scorer or resolve_objective_scorer(self.config.objective, self.config)
        for raw in raw_solutions:
            score = scorer.score(raw)
            if raw.unpacked and raw.time_limit_reached: status = PackingStatus.TIME_LIMIT
            elif raw.unpacked: status = PackingStatus.BEST_FOUND
            elif raw.exhaustive: status = PackingStatus.OPTIMAL
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
        return PackingResult(
            best.status,
            best.containers,
            best.unpacked,
            best.algorithm,
            best.score,
            best.warnings,
            tuple(selected[1:self.config.top_k]),
            best.feasibility,
            best.termination,
            best.optimality,
            best.objective,
        )
