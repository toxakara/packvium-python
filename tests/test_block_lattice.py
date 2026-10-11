"""The block start's group-at-a-time enumeration against the node-by-node one it replaced.

`HomogeneousBlockSolver` used to expand every `(nx, ny)` lattice node of every (space,
type, rotation) group, polling the deadline and counting each one. It now summarises a
group with `_lattice` and charges its nodes in one step. Nothing a caller can see may
move: the blocks, the counters, and -- when a node limit or an effort budget stops the
search -- the exact node it stops on. The oracle below is the 1.6.0 enumeration, kept
verbatim so that every claim is checked against what shipped, not against a restatement.
"""

from __future__ import annotations

import random

import pytest

from packvium import Container, Dimensions, EffortBudget, Item, PackingConfig
from packvium.solvers import (Deadline, HomogeneousBlockSolver, SearchStats, _HomogeneousBlock,
                              _lattice)


class NodeByNodeBlockSolver(HomogeneousBlockSolver):
    """The 1.6.0 round: every lattice node polls the deadline and is counted alone."""

    def _next_block(self, container, state, spaces, remaining, forms, config, stats, deadline,
                    mode, lattices, nodes):
        node_limit = config.container_plan_node_limit
        reached = False
        best: _HomogeneousBlock | None = None
        for space in spaces:
            space_volume = space.dimensions.volume
            for item_id in sorted(remaining):
                available = remaining[item_id]
                if not available:
                    continue
                prototype = available[0]
                capacity = len(available)
                if container.max_items is not None:
                    capacity = min(capacity, container.max_items - state.placement_count)
                if container.max_payload is not None and prototype.weight.ticks > 0:
                    capacity = min(
                        capacity,
                        (container.max_payload.ticks - state.payload_ticks)
                        // prototype.weight.ticks,
                    )
                if capacity <= 0:
                    continue
                for rotation, physical in prototype.dimensions.unique_rotations(
                    prototype.item.allowed_rotations
                ):
                    envelope = (
                        physical.expand(config.clearance)
                        if config.clearance.ticks
                        else physical
                    )
                    maximum_x = space.dimensions.length.ticks // envelope.length.ticks
                    maximum_y = space.dimensions.width.ticks // envelope.width.ticks
                    maximum_z = space.dimensions.height.ticks // envelope.height.ticks
                    group_count = group_nx = group_ny = group_nz = 0
                    for nx in range(1, min(maximum_x, capacity) + 1):
                        maximum_ny = min(maximum_y, capacity // nx)
                        for ny in range(1, maximum_ny + 1):
                            if nodes >= node_limit or deadline.expired:
                                reached = True
                                break
                            nodes += 1
                            stats.search_nodes_expanded += 1
                            nz = min(maximum_z, capacity // (nx * ny))
                            if nz > 0 and nx * ny * nz > group_count:
                                group_count = nx * ny * nz
                                group_nx, group_ny, group_nz = nx, ny, nz
                        if reached:
                            break
                    if reached:
                        break
                    if group_count:
                        best = self._better_block(
                            best, mode, space, space_volume, item_id, rotation, physical,
                            envelope, group_nx, group_ny, group_nz, group_count,
                        )
                if reached:
                    break
            if reached:
                break
        return best, nodes, reached


def node_by_node_lattice(columns: int, rows: int, layers: int, capacity: int):
    nodes = count = best = 0
    shape = (0, 0, 0)
    for nx in range(1, columns + 1):
        for ny in range(1, min(rows, capacity // nx) + 1):
            nodes += 1
            nz = min(layers, capacity // (nx * ny))
            if nz > 0 and nx * ny * nz > best:
                best = nx * ny * nz
                shape = (nx, ny, nz)
    count = best
    return (nodes, count, *shape)


def test_a_group_summary_is_its_node_by_node_enumeration():
    for capacity in range(1, 26):
        for columns in range(0, min(capacity, 7) + 1):
            for rows in range(0, 7):
                for layers in range(0, 6):
                    assert _lattice(columns, rows, layers, capacity) == \
                        node_by_node_lattice(columns, rows, layers, capacity)


def scene(seed: int):
    rng = random.Random(seed)
    box = Container.create(
        "box", Dimensions.mm(rng.randint(150, 400), rng.randint(120, 300), rng.randint(100, 260)),
        max_items=rng.choice([None, None, rng.randint(5, 40)]),
        max_payload=rng.choice([None, None, f"{rng.randint(5, 60)}kg"]),
    )
    items = []
    for index in range(rng.randint(2, 4)):
        items.extend(Item.create(
            f"t{index}", Dimensions.mm(rng.randint(20, 90), rng.randint(20, 90), rng.randint(20, 90)),
            f"{rng.randint(1, 3)}kg", quantity=rng.randint(1, 30),
        ).instances())
    return box, tuple(items)


def outcome(solver, box, items, config, deadline_for):
    stats = SearchStats()
    solution = solver.pack_one(box, 1, items, config, stats, deadline_for(stats))
    placements = [(p.instance.id, p.position, p.rotation, p.support_ratio)
                  for p in solution.state.placements]
    return (placements, [i.id for i in solution.unpacked], solution.time_limit_reached, stats)


def unlimited(stats):
    return Deadline(600_000)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("modes", [("count",), ("volume",), ("count", "volume")])
def test_every_node_limit_stops_on_the_node_it_stopped_on_before(seed, modes):
    box, items = scene(seed)
    full = PackingConfig.quality(container_plan_node_limit=10**9)
    total = outcome(NodeByNodeBlockSolver(modes=modes), box, items, full, unlimited)[3]
    per_mode = total.search_nodes_expanded
    limits = sorted({1, 2, 3, *range(4, per_mode + 2, max(1, per_mode // 40)),
                     per_mode - 1, per_mode, per_mode + 1} - {0})
    for limit in limits:
        config = PackingConfig.quality(container_plan_node_limit=limit)
        assert outcome(HomogeneousBlockSolver(modes=modes), box, items, config, unlimited) == \
            outcome(NodeByNodeBlockSolver(modes=modes), box, items, config, unlimited)


@pytest.mark.parametrize("seed", range(12))
def test_every_effort_budget_stops_on_the_node_it_stopped_on_before(seed):
    box, items = scene(seed)
    modes = ("count", "volume")
    config = PackingConfig.quality(container_plan_node_limit=10**9)
    total = outcome(NodeByNodeBlockSolver(modes=modes), box, items, config, unlimited)[3]
    nodes = total.search_nodes_expanded
    for budget in sorted({1, 2, *range(3, nodes + 2, max(1, nodes // 40)), nodes - 1, nodes} - {0}):
        def bounded(stats, budget=budget):
            return Deadline(600_000).with_effort(EffortBudget(max_search_nodes=budget), stats)
        assert outcome(HomogeneousBlockSolver(modes=modes), box, items, config, bounded) == \
            outcome(NodeByNodeBlockSolver(modes=modes), box, items, config, bounded)
    for candidates in (1, 5, 20):
        def other(stats, candidates=candidates):
            return Deadline(600_000).with_effort(EffortBudget(max_candidates_evaluated=candidates), stats)
        assert outcome(HomogeneousBlockSolver(modes=modes), box, items, config, other) == \
            outcome(NodeByNodeBlockSolver(modes=modes), box, items, config, other)


def test_a_budget_counting_other_stats_never_stops_the_enumeration_midway():
    box, items = scene(3)
    config = PackingConfig.quality(container_plan_node_limit=10**9)
    elsewhere = SearchStats()
    def foreign(stats):
        return Deadline(600_000).with_effort(EffortBudget(max_search_nodes=1), elsewhere)
    assert outcome(HomogeneousBlockSolver(), box, items, config, foreign) == \
        outcome(NodeByNodeBlockSolver(), box, items, config, foreign)
    assert Deadline(1).with_effort(EffortBudget(max_search_nodes=5), elsewhere) \
        ._search_node_ceiling(SearchStats()) is None


def test_a_full_memo_only_recomputes(monkeypatch):
    from packvium import solvers

    box, items = scene(5)
    config = PackingConfig.quality(container_plan_node_limit=10**9)
    remembered = outcome(HomogeneousBlockSolver(), box, items, config, unlimited)
    monkeypatch.setattr(solvers, "_LATTICE_MEMO_LIMIT", 0)
    assert outcome(HomogeneousBlockSolver(), box, items, config, unlimited) == remembered == \
        outcome(NodeByNodeBlockSolver(), box, items, config, unlimited)
