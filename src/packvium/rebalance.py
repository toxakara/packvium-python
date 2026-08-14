from __future__ import annotations

from dataclasses import replace

from ._compat import dataclass

from .config import PackingConfig
from .constraints import direct_support_view, load_units, top_loads
from .geometry import AxisAlignedBox, Dimensions, Point
from .models import ItemInstance, PackedContainer, PackingRequest, Placement, UnpackedItem
from .solvers import (ContainerState, Deadline, SearchStats, TimeLimitReached, default_constraints,
                      find_candidates)
from .units import Weight
from .validation import IndependentSolutionValidator


@dataclass(frozen=True, slots=True)
class WeightMove:
    """One item relocated from one already-packed container to another."""
    item_id: str
    from_container_id: str
    to_container_id: str


@dataclass(frozen=True, slots=True)
class RebalanceResult:
    containers: tuple[PackedContainer, ...]
    moves: tuple[WeightMove, ...]

    @property
    def improved(self) -> bool:
        return bool(self.moves)


def _refresh_top_loads(placements: tuple[Placement, ...]) -> tuple[Placement, ...]:
    """Re-derive every placement's reported `top_load` from the current stack.

    Moving an item changes what rests on what, so a stale `top_load` carried
    forward from before the move would misreport the very quantity a caller
    checks bearing limits against.
    """
    loads = top_loads(load_units(placements))
    return tuple(replace(p, top_load=Weight(load)) for p, load in zip(placements, loads))


def _support_ratio(
    existing: tuple[Placement, ...], instance: ItemInstance, origin: Point,
    envelope_dimensions: Dimensions,
) -> float:
    """Fraction of the base resting on something already in the container.

    Deliberately re-derived here from public geometry rather than importing the
    solver's own internal candidate-scoring helper: this module has no business
    depending on `solvers.py`'s private placement-time bookkeeping, only on the
    same public contact-area rule it uses.
    """
    if origin.z == 0:
        return 1.0
    box = AxisAlignedBox(origin, envelope_dimensions)
    return direct_support_view(existing, instance, box).supporting_area / envelope_dimensions.base_area


def _attempt_move(
    request: PackingRequest,
    validator: IndependentSolutionValidator,
    containers: list[PackedContainer],
    unpacked: tuple[UnpackedItem, ...],
    source_index: int,
    dest_index: int,
    placement_index: int,
    config: PackingConfig,
    deadline: Deadline,
) -> list[PackedContainer] | None:
    """Try relocating one placement; return the new container list if it is sound, else None.

    Every candidate is proven, not assumed: the item must find a real,
    constraint-respecting spot in the destination (`find_candidates`, the same
    search a solve itself uses), and the *entire* resulting set of containers
    must then pass the independent validator before this reports success.
    Nothing is removed from its source ahead of that check -- both container
    tuples are only ever replaced together, in the same returned list, so there
    is no observable state in which the item belongs to neither its old
    container nor its new one, and no path that "finishes" a move without the
    item landing exactly once. A move that fails geometrically (no room in the
    destination) or physically (the validator rejects the result -- for
    example because something was resting on the item that just moved) simply
    is not made; the caller's container list is untouched.
    """
    source = containers[source_index]
    dest = containers[dest_index]
    moving = source.placements[placement_index]

    dest_state = ContainerState(dest.container, dest.sequence)
    for placement in dest.placements:
        dest_state.add(placement)
    constraints = default_constraints(config)
    stats = SearchStats()
    candidates = find_candidates(dest_state, moving.instance, config, constraints, stats, deadline, max_candidates=1)
    if not candidates:
        return None
    candidate = candidates[0]

    new_dest_placement = Placement(
        moving.instance, candidate.position, candidate.rotation, candidate.dimensions,
        candidate.point, candidate.envelope_dimensions,
        _support_ratio(dest.placements, moving.instance, candidate.point, candidate.envelope_dimensions),
    )
    new_source_placements = source.placements[:placement_index] + source.placements[placement_index + 1:]
    new_dest_placements = (*dest.placements, new_dest_placement)

    trial = list(containers)
    trial[source_index] = replace(source, placements=_refresh_top_loads(new_source_placements))
    trial[dest_index] = replace(dest, placements=_refresh_top_loads(new_dest_placements))

    report = validator.validate(request, tuple(trial), config.minimum_support_ratio, config.clearance, unpacked)
    if not report.valid:
        return None
    return trial


def rebalance_weight(
    request: PackingRequest,
    containers: tuple[PackedContainer, ...],
    unpacked: tuple[UnpackedItem, ...] = (),
    config: PackingConfig | None = None,
    *,
    max_moves: int = 64,
    time_limit_ms: int = 1000,
) -> RebalanceResult:
    """Redistribute payload weight across an already-solved multi-container packing.

    An explicit, opt-in post-processing pass: call it after `Packer.pack()`, the
    same way a post-pass weight redistributor is invoked separately
    from packing itself, not run automatically inside `Packer.pack`. `unpacked`
    should be the same tuple `Packer.pack()` returned (or `()` for an already
    complete packing) -- it is passed straight through to the validator so full
    accounting is checked against every requested instance, not only the ones
    currently sitting in a container.

    such a post-pass carries a well-known failure mode: it can
    silently drop an item, because its own bookkeeping only checks whether the
    box count is still 1, not whether every item it started with is still
    accounted for. This function never substitutes a partial signal like that
    for the real thing. Every candidate move is simulated in full and checked
    against the same `IndependentSolutionValidator` every solve in this library
    is checked against -- a fresh, independent re-derivation of the whole
    packing from its placements alone -- and only committed once that check
    passes. See `_attempt_move` for exactly what that guarantees.

    Greedy and best-effort, not globally optimal: each round moves at most one
    item, from whichever container is currently heaviest, trying its own
    placements heaviest-first against the other containers lightest-first, and
    commits the first combination that both fits and strictly reduces the gap
    between the heaviest and lightest container's payload weight. Requiring a
    strict reduction on every commit makes the payload spread a strictly
    decreasing sequence of non-negative integers, so the pass always
    terminates on its own (bounded again by `max_moves`) whether or not it
    reaches the smallest spread achievable by some other arrangement.
    """
    config = config or PackingConfig()
    validator = IndependentSolutionValidator()
    deadline = Deadline(time_limit_ms)
    working = list(containers)
    moves: list[WeightMove] = []

    for _ in range(max_moves):
        if len(working) < 2 or deadline.expired:
            break
        weights = [c.payload_weight.ticks for c in working]
        spread = max(weights) - min(weights)
        if spread <= 0:
            break
        source_index = max(range(len(working)), key=lambda i: weights[i])
        source = working[source_index]
        ranked_items = sorted(
            range(len(source.placements)),
            key=lambda i: -source.placements[i].instance.weight.ticks,
        )
        destinations = sorted(
            (i for i in range(len(working)) if i != source_index),
            key=lambda i: weights[i],
        )

        committed = None
        for placement_index in ranked_items:
            weight_ticks = source.placements[placement_index].instance.weight.ticks
            if weight_ticks <= 0:
                continue
            for dest_index in destinations:
                projected = list(weights)
                projected[source_index] -= weight_ticks
                projected[dest_index] += weight_ticks
                if max(projected) - min(projected) >= spread:
                    continue
                try:
                    trial = _attempt_move(
                        request, validator, working, unpacked,
                        source_index, dest_index, placement_index, config, deadline,
                    )
                except TimeLimitReached:
                    trial = None
                if trial is None:
                    continue
                committed = (trial, source.placements[placement_index].instance.id, source.id, working[dest_index].id)
                break
            if committed is not None:
                break
        if committed is None:
            break
        trial, item_id, from_id, to_id = committed
        moves.append(WeightMove(item_id, from_id, to_id))
        working = trial

    return RebalanceResult(tuple(working), tuple(moves))
