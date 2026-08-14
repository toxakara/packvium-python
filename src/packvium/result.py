from __future__ import annotations

from dataclasses import field

from ._compat import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .axle_load import axle_reactions
from .constraints import load_units, reserved_volume
from .models import PackedContainer, UnpackedItem
from .units import Length


class PackingStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    BEST_FOUND = "best_found"
    TIME_LIMIT = "time_limit"
    INFEASIBLE = "infeasible"
    INVALID_RESULT = "invalid_result"


@dataclass(frozen=True, slots=True)
class ResultFact:
    """One open, forward-compatible result axis.

    `code` is intentionally a string rather than an enum: a newer producer may add a
    termination value without making an older client unable to parse and preserve it.
    """

    code: str
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("result fact code must not be empty")

    def to_dict(self) -> dict:
        return {"code": self.code, **self.attributes}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ResultFact":
        code = raw.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError("result fact requires a non-empty string code")
        return cls(code, {key: value for key, value in raw.items() if key != "code"})


@dataclass(frozen=True, slots=True)
class StartRecord:
    id: str
    started: bool
    completed: bool
    truncated: bool
    selected: bool = False
    global_deadline_reached: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started": self.started,
            "completed": self.completed,
            "truncated": self.truncated,
            "selected": self.selected,
            "global_deadline_reached": self.global_deadline_reached,
        }


def aggregate_termination(
    starts: Iterable[StartRecord],
    *,
    error: bool = False,
) -> ResultFact:
    records = tuple(starts)
    if not records:
        raise ValueError("termination aggregation requires at least one start record")
    selected = tuple(record for record in records if record.selected)
    if len(selected) != 1:
        raise ValueError("termination aggregation requires exactly one selected start")
    any_truncated = any(record.truncated for record in records)
    all_completed = all(record.completed for record in records)
    winning_truncated = selected[0].truncated
    global_deadline = any(record.global_deadline_reached for record in records)
    code = "error" if error else "time_limit" if winning_truncated or global_deadline else "complete"
    return ResultFact(code, {
        "any_start_truncated": any_truncated,
        "all_required_starts_completed": all_completed,
        "winning_start_truncated": winning_truncated,
        "global_deadline_reached": global_deadline,
        "starts": [record.to_dict() for record in records],
    })


def derive_result_facts(
    status: PackingStatus,
    complete: bool,
    time_limit_reached: bool,
) -> tuple[ResultFact, ResultFact, ResultFact]:
    if status is PackingStatus.INFEASIBLE:
        feasibility = "infeasible"
    elif complete and status is not PackingStatus.INVALID_RESULT:
        feasibility = "feasible"
    else:
        feasibility = "unknown"

    termination = (
        "error" if status is PackingStatus.INVALID_RESULT
        else "time_limit" if time_limit_reached
        else "complete"
    )
    optimality = (
        "proven_optimal" if status is PackingStatus.OPTIMAL
        else "proven_infeasible" if status is PackingStatus.INFEASIBLE
        else "best_found" if not complete and status is not PackingStatus.INVALID_RESULT
        else "not_proven"
    )
    return ResultFact(feasibility), ResultFact(termination), ResultFact(optimality)


@dataclass(frozen=True, slots=True)
class SolverMetrics:
    candidate_points_considered: int = 0
    orientations_considered: int = 0
    feasible_candidates: int = 0
    collision_checks: int = 0
    support_checks: int = 0
    space_partitions: int = 0
    search_nodes_expanded: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_points_considered": self.candidate_points_considered,
            "orientations_considered": self.orientations_considered,
            "feasible_candidates": self.feasible_candidates,
            "collision_checks": self.collision_checks,
            "support_checks": self.support_checks,
            "space_partitions": self.space_partitions,
            "search_nodes_expanded": self.search_nodes_expanded,
        }


@dataclass(frozen=True, slots=True)
class AlgorithmReport:
    profile: str
    solver: str
    duration_ms: int
    seed: int
    time_limit_reached: bool = False
    effort_limit_reached: bool = False
    candidates_evaluated: int = 0
    placements_attempted: int = 0
    metrics: SolverMetrics = SolverMetrics()


@dataclass(frozen=True, slots=True)
class PackingResult:
    status: PackingStatus
    containers: tuple[PackedContainer, ...]
    unpacked: tuple[UnpackedItem, ...]
    algorithm: AlgorithmReport
    score: tuple[int | float, ...]
    warnings: tuple[str, ...] = ()
    alternatives: tuple["PackingResult", ...] = ()
    feasibility: ResultFact | None = None
    termination: ResultFact | None = None
    optimality: ResultFact | None = None
    objective: str = "default"
    catalog_versions_used: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        derived = derive_result_facts(self.status, not self.unpacked, self.algorithm.time_limit_reached)
        if self.feasibility is None:
            object.__setattr__(self, "feasibility", derived[0])
        if self.termination is None:
            object.__setattr__(self, "termination", aggregate_termination((
                StartRecord(
                    self.algorithm.solver or "unknown",
                    True,
                    not self.algorithm.time_limit_reached,
                    self.algorithm.time_limit_reached,
                    selected=True,
                    global_deadline_reached=self.algorithm.time_limit_reached,
                ),
            ), error=self.status is PackingStatus.INVALID_RESULT))
        if self.optimality is None:
            object.__setattr__(self, "optimality", derived[2])

    @property
    def complete(self) -> bool: return not self.unpacked
    @property
    def packed_item_count(self) -> int: return sum(c.placement_count for c in self.containers)

    def to_dict(self, length_unit: str = "mm", weight_unit: str = "g", include_alternatives: bool = True) -> dict:
        def placement_dict(p):
            return {
                "item_id": p.instance.id,
                "item_type": p.instance.item.id,
                "position": p.position.to_dict(length_unit),
                "dimensions": p.dimensions.to_dict(length_unit),
                "orientation": p.rotation.value,
                "support_ratio": f"{p.support_ratio:.6f}",
                "top_load": p.top_load.to_dict(weight_unit),
            }
        def lattice_summary_dict(summary):
            # Omitted entirely (not even a `null` key) when this container was not built
            # by the compact fast path, so a default (unset or `true`
            # `require_placement_coordinates`) request's canonical JSON is byte-for-byte
            # unchanged -- this is a strict, opt-in addition, not a shape change.
            if summary is None:
                return {}
            return {"lattice_summary": {
                "item_type": summary.item_type,
                "orientation": summary.rotation.value,
                "physical_dimensions": summary.physical.to_dict(length_unit),
                "envelope_dimensions": summary.envelope.to_dict(length_unit),
                "nx": summary.nx,
                "ny": summary.ny,
                "layers_used": summary.layers_used,
                "layer_step": Length(summary.layer_step).to_dict(length_unit),
                "count": summary.count,
            }}
        def axle_dict(container):
            if container.container.axles is None:
                return {}
            denominator, front, rear = axle_reactions(
                container.container.axles,
                load_units(container.placements),
                container.container.tare_weight.ticks,
                container.container.inner_dimensions.length.ticks,
            )
            return {"axle_reactions": {
                "basis": "gross",
                "denominator": str(denominator),
                "front_numerator": str(front),
                "rear_numerator": str(rear),
            }}
        return {
            "status": self.status.value,
            "feasibility": self.feasibility.to_dict(),
            "termination": self.termination.to_dict(),
            "optimality": self.optimality.to_dict(),
            "complete": self.complete,
            "objective": self.objective,
            "algorithm": {
                "profile": self.algorithm.profile, "solver": self.algorithm.solver,
                "duration_ms": self.algorithm.duration_ms, "seed": self.algorithm.seed,
                "time_limit_reached": self.algorithm.time_limit_reached,
                "effort_limit_reached": self.algorithm.effort_limit_reached,
                "candidates_evaluated": self.algorithm.candidates_evaluated,
                "placements_attempted": self.algorithm.placements_attempted,
                "metrics": self.algorithm.metrics.to_dict(),
            },
            "summary": {"container_count": len(self.containers), "packed_item_count": self.packed_item_count, "unpacked_item_count": len(self.unpacked)},
            "score": list(self.score),
            "containers": [{
                "id": c.id, "container_type": c.container.id,
                "inner_dimensions": c.container.inner_dimensions.to_dict(length_unit),
                "outer_dimensions": (c.container.outer_dimensions or c.container.inner_dimensions).to_dict(length_unit),
                "payload_weight": c.payload_weight.to_dict(weight_unit),
                "gross_weight": c.gross_weight.to_dict(weight_unit),
                "used_volume_ticks3": str(c.used_volume),
                "volume_utilization": f"{c.utilization:.6f}",
                "centre_of_mass_offset_ppm": c.centre_of_mass_offset_ppm,
                **axle_dict(c),
                "void_fill_reserve_ticks3": str(reserved_volume(c.container)),
                # Compact form: when `configuration.require_placement_coordinates`
                # was false and GridSolver's lattice fast path applied, `placements` stays
                # empty and an added `lattice_summary` key carries the same information in
                # O(1)/O(r) instead of one entry per instance.
                **lattice_summary_dict(c.lattice_summary),
                "placements": [placement_dict(p) for p in c.placements],
            } for c in self.containers],
            "unpacked_items": [{
                "item_id": u.instance.id,
                "item_type": u.instance.item.id,
                "reason": u.reason,
                "details": list(u.details),
                "proof": u.proof.to_dict(),
            } for u in self.unpacked],
            "catalog_versions_used": [dict(reference) for reference in self.catalog_versions_used],
            "warnings": list(self.warnings),
            "alternatives": [a.to_dict(length_unit, weight_unit, False) for a in self.alternatives] if include_alternatives else [],
        }
