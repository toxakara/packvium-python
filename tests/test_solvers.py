"""The search machinery: time budgets, seeded randomness, free-space bookkeeping
and each of the four single-container solvers.

Every solver is exercised through the same two questions — does it place what it
claims, and is what it placed physically sound — because a solver that reports a
placement it never verified is the failure mode that matters.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest

from packvium import (AxisAlignedBox, Container, Dimensions, EffortBudget, Item, Length,
                         Obstacle, PackedContainer, PackingConfig, Placement, Point, Rotation)
from packvium.nesting import used_volume as nesting_used_volume
from packvium.solvers import (MAX_MAXIMAL_SPACES, MINIMUM_SLICE_NS, ContainerState, Deadline,
                                 DeterministicRandom, ExactSmallSolver, ExtremePointSolver,
                                 GridSolver, HomogeneousBlockSolver, LayerSolver, MaximalSpaceSolver, SearchStats, Space,
                                 SolverOrchestrator, TimeLimitReached, UnknownSolverError,
                                 default_constraints, find_candidates, group_batches,
                                 subtract_all, beam_pack, is_better_container_state,
                                 _maximum_count_with_capacity)

def generous() -> Deadline:
    """A fresh, effectively-unlimited deadline, started now.

    Not a module-level constant: its real wall clock would start ticking at test
    *collection* time, shared across every test in this file, rather than when each
    test actually runs. Harmless when the whole suite finishes in a couple of seconds,
    but exactly the class of flake this targets -- under real load (a
    concurrent build, a slow machine, a long-running gate) collection-to-execution time
    can exceed even a generous shared budget, and every later test in the file starts
    failing with a genuine `TimeLimitReached` on trivially-placeable single items.
    Measured: reproduced with the previous shared `Deadline(60_000)` during a
    a full build sharing the machine with concurrent native compilation.
    """
    return Deadline(60_000)


def instances(id: str, length, width, height, quantity=1, **kwargs):
    return Item.create(id, Dimensions.mm(length, width, height), kwargs.pop("weight", 0),
                       quantity=quantity, **kwargs).instances()


def extend(state: ContainerState, item, config: PackingConfig | None = None) -> ContainerState:
    """Place `item` at the state's best candidate point, mutating the state in place."""
    config = config or PackingConfig.balanced()
    candidate, = find_candidates(state, item, config, default_constraints(config), SearchStats(), generous(), 1)
    state.add(Placement(item, candidate.position, candidate.rotation, candidate.dimensions,
                        candidate.point, candidate.envelope_dimensions))
    return state


def test_container_state_caches_exact_volume_and_rule_sensitivity_across_copies():
    box = Container.create("c", Dimensions.mm(100, 100, 220))
    item = Item.create("crate", Dimensions.mm(100, 100, 100), quantity=3,
                       nesting_height=Length.mm(40), max_stacked_items=3, stop_index=1)
    state = ContainerState(box, 1)
    for instance, z in zip(item.instances(), (0, 60, 120)):
        dims = instance.dimensions
        point = Point(0, 0, Length.mm(z).ticks)
        state.add_direct(Placement(instance, point, Rotation.LWH, dims, point, dims))

    assert state.used_volume_ticks == Length.mm(100).ticks ** 2 * Length.mm(220).ticks
    assert state.used_volume_ticks == nesting_used_volume(state.placements)
    assert state.stack_sensitive
    assert state.route_sensitive
    assert state.ordered_points == sorted(state.points.values(), key=lambda point: (point.z, point.y, point.x))
    copied = state.copy()
    assert copied.used_volume_ticks == state.used_volume_ticks
    assert copied.stack_sensitive and copied.route_sensitive
    assert copied.ordered_points == state.ordered_points


def assert_physically_sound(state: ContainerState) -> None:
    boundary = AxisAlignedBox(Point(0, 0, 0), state.container.inner_dimensions)
    boxes = [p.envelope_box for p in state.placements]
    for index, box in enumerate(boxes):
        assert boundary.contains(box), "a placement escaped the container"
        assert not any(box.intersects(o.box) for o in state.container.obstacles), "a placement hit an obstacle"
        for other in boxes[index + 1:]:
            assert not box.intersects(other), "two placements overlap"


# ------------------------------------------------------------------- time budget

class CheckBudgetClock:
    """Monotonic fake that advances one millisecond on every read."""

    def __init__(self):
        self.reads = 0

    def __call__(self) -> int:
        value = self.reads * 1_000_000
        self.reads += 1
        return value


class FastClock:
    """Monotonic fake that advances one microsecond on every read.

    Paired with `CheckBudgetClock` (1ms/read) to stand in for a heavily loaded
    machine versus a near-idle one, without an actual wall clock or real timing --
    the same injected-clock technique the rest of this file already uses to make
    deadline behaviour deterministic and free of real jitter.
    """

    def __init__(self):
        self.reads = 0

    def __call__(self) -> int:
        value = self.reads * 1_000
        self.reads += 1
        return value


def test_a_fresh_deadline_has_its_whole_budget():
    deadline = Deadline(1_000)
    assert 0 < deadline.remaining_ns <= 1_000_000_000
    assert not deadline.expired


def test_an_injected_clock_expires_before_the_first_placement_without_sleeping():
    clock = CheckBudgetClock()
    expired = Deadline(1, clock=clock)
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    order = instances("a", 20, 20, 20, quantity=2)

    solution = beam_pack(
        box, 1, order, PackingConfig.balanced(),
        default_constraints(PackingConfig.balanced()), SearchStats(), expired,
    )

    assert solution.time_limit_reached
    assert not solution.state.placements
    assert {item.id for item in solution.unpacked} == {item.id for item in order}


def test_an_injected_clock_expires_mid_search_without_sleeping():
    clock = CheckBudgetClock()
    deadline = Deadline(8, clock=clock)
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    order = instances("a", 20, 20, 20, quantity=8)

    solution = beam_pack(
        box, 1, order, PackingConfig.balanced(),
        default_constraints(PackingConfig.balanced()), SearchStats(), deadline,
    )

    assert solution.time_limit_reached
    assert 0 < len(solution.state.placements) < len(order)
    assert len(solution.state.placements) + len(solution.unpacked) == len(order)


def test_a_slice_keeps_the_injected_clock():
    clock = CheckBudgetClock()
    parent = Deadline(10, clock=clock)
    child = parent.slice(2)

    assert not child.expired
    while not child.expired:
        pass
    assert parent.remaining_ns < parent.limit_ns
    while not parent.expired:
        pass


def test_a_deadline_caught_inside_the_beam_is_preserved_in_the_solution():
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    order = instances("a", 20, 20, 20, quantity=2)
    expired = Deadline(0, limit_ns=-1)

    solution = beam_pack(
        box, 1, order, PackingConfig.balanced(),
        default_constraints(PackingConfig.balanced()), SearchStats(), expired,
    )

    assert solution.time_limit_reached
    assert {item.id for item in solution.unpacked} == {item.id for item in order}


def test_an_exhausted_deadline_raises_rather_than_returning_quietly():
    expired = Deadline(0)
    assert expired.expired
    with pytest.raises(TimeLimitReached):
        expired.check()


def test_an_expired_portfolio_preserves_a_structural_dimension_proof():
    oversized = instances("oversized", 200, 200, 200)
    searchable = instances("searchable", 10, 10, 10)
    containers = (Container.create("small", Dimensions.mm(100, 100, 100)),)

    solution, = SolverOrchestrator().solve(
        (*oversized, *searchable),
        containers,
        PackingConfig.fast(),
        Deadline(0, limit_ns=-1),
    )

    assert solution.time_limit_reached
    rejected = {item.instance.item.id: item for item in solution.unpacked}
    assert rejected["oversized"].reason == "no_compatible_container_dimensions"
    assert rejected["oversized"].proof.level == "proven"
    assert rejected["searchable"].reason == "time_limit"
    assert rejected["searchable"].proof.level == "unknown_due_to_limit"


def test_a_high_priority_item_leads_every_built_in_ordering():
    # Priority is a preference, not a guarantee: a container that can only hold one of
    # the two items should hold the high-priority one, even though it is far smaller.
    small = instances("small", 10, 10, 10, priority=5)
    big = instances("big", 100, 100, 100)
    containers = (Container.create("box", Dimensions.mm(100, 100, 100), quantity=1),)
    config = PackingConfig.balanced(multi_start_orders=1, time_limit_ms=2_000)

    run = SolverOrchestrator().solve((*big, *small), containers, config, Deadline(2_000))

    packed_ids = {p.instance.id for solution in run for c in solution.containers for p in c.placements}
    assert packed_ids == {"small#1"}


def test_a_lattice_solver_runs_once_rather_than_once_per_ordering():
    # GridSolver already declares this; LayerSolver re-sorts by a total key ending in
    # item id, discarding whatever order it is handed, so it must declare it too.
    items = (*instances("a", 20, 20, 20, quantity=3), *instances("b", 10, 10, 10, quantity=3))
    containers = (Container.create("box", Dimensions.mm(100, 100, 100)),)

    run = SolverOrchestrator().solve(items, containers, PackingConfig.balanced(), Deadline(2_000))

    layer_starts = [start for start in run.starts if start.id.startswith("layer:")]
    assert len(layer_starts) == 1


def test_the_portfolio_drops_from_sixteen_starts_to_nine_on_a_three_type_order():
    """The acceptance criterion, made executable rather than only claimed in prose.

    Three item types, each diverse enough on a different axis (volume, base area,
    weight, rotation freedom) that all four of the built-in named strategies produce
    a genuinely distinct ordering, and enough instances (two of each) that the
    remaining slots up to the default `multi_start_orders=8` are filled by seeded
    random shuffles distinct from those four and from each other -- so the naive
    portfolio genuinely is `extreme_points` and `layer`, 8 orders each, 16 starts, not
    something that happens to collapse below that on its own. Declaring `LayerSolver`
    order-insensitive collapses its eight down to one, leaving 8 + 1 = 9.
    """
    tall_thin = instances("tall_thin", 10, 50, 2, quantity=2, weight=300,
                          allowed_rotations=(Rotation.LWH,))
    cube = instances("cube", 20, 20, 20, quantity=2, weight=100)
    heavy_flat = instances("heavy_flat", 30, 10, 20, quantity=2, weight=900,
                           allowed_rotations=(Rotation.LWH, Rotation.WLH, Rotation.LHW))
    items = (*tall_thin, *cube, *heavy_flat)
    containers = (Container.create("box", Dimensions.mm(100, 100, 100)),)
    config = PackingConfig.balanced()
    assert config.multi_start_orders == 8  # the naive 16 and reduced 9 below assume this default

    orders = SolverOrchestrator()._orders(items, config)
    assert len({signature for _, order in orders for signature in [tuple(i.id for i in order)]}) == 8

    run = SolverOrchestrator().solve(items, containers, config, generous())

    assert len(run.starts) == 9
    starts_by_solver = Counter(start.id.split(":", 1)[0] for start in run.starts)
    assert starts_by_solver == {"extreme_points": 8, "layer": 1}


def test_a_layer_solver_produces_the_identical_packing_regardless_of_the_order_it_is_handed():
    """Declaring `order_insensitive = True` is only safe if it is actually true.
    `LayerSolver` re-sorts by its own total key before placing anything, so feeding it
    several genuinely different orders of the same items (the same eight orders the
    cardinality test above establishes are distinct) must produce the identical
    packing every time -- proven directly here, not assumed from the class's own
    docstring comment.
    """
    tall_thin = instances("tall_thin", 10, 50, 2, quantity=2, weight=300,
                          allowed_rotations=(Rotation.LWH,))
    cube = instances("cube", 20, 20, 20, quantity=2, weight=100)
    heavy_flat = instances("heavy_flat", 30, 10, 20, quantity=2, weight=900,
                           allowed_rotations=(Rotation.LWH, Rotation.WLH, Rotation.LHW))
    items = (*tall_thin, *cube, *heavy_flat)
    container = Container.create("box", Dimensions.mm(100, 100, 100))
    config = PackingConfig.balanced()
    solver = LayerSolver()

    def signature(order):
        solution = solver.pack_one(container, 0, order, config, SearchStats(), generous())
        return tuple(sorted(
            (p.instance.id, p.position, p.rotation, p.dimensions) for p in solution.state.placements
        ))

    orders = SolverOrchestrator()._orders(items, config)
    assert len(orders) == 8
    signatures = {signature(order) for _, order in orders}
    assert len(signatures) == 1


def test_layer_solver_produces_identical_packing_on_real_br1_data():
    """A third, previously-unattempted claim: measured BR1 utilisation does
    not regress. The corpus that claim needed did not exist in this workspace when
    this was last measured; the corpus has since been committed
    (`benchmarks/datasets/br/orlib-thpack/thpack1.txt`). This test uses that real published
    instance's own item dimensions and quantities (instance 1, hand-verified against
    the raw file in `benchmarks/tests/test_br_dataset.py`) rather than fabricating
    scene data, and applies the exact same technique as the companion test above:
    if `LayerSolver` produces a byte-identical packing regardless of order on this
    real BR1 scene, collapsing its eight redundant per-order starts down to one
    cannot have changed -- let alone regressed -- the achieved utilisation.
    """
    type_1 = instances("type-1", 108, 76, 30, quantity=40)
    type_2 = instances("type-2", 110, 43, 25, quantity=33)
    type_3 = instances("type-3", 92, 81, 55, quantity=39)
    items = (*type_1, *type_2, *type_3)
    container = Container.create("br-container", Dimensions.mm(587, 233, 220))
    config = PackingConfig.balanced()
    solver = LayerSolver()

    def signature(order):
        solution = solver.pack_one(container, 0, order, config, SearchStats(), generous())
        return tuple(sorted(
            (p.instance.id, p.position, p.rotation, p.dimensions) for p in solution.state.placements
        ))

    orders = SolverOrchestrator()._orders(items, config)
    assert len(orders) == 8
    signatures = {signature(order) for _, order in orders}
    assert len(signatures) == 1


def test_slicing_divides_the_remaining_budget():
    deadline = Deadline(1_000)
    whole, tenth = deadline.slice(1), deadline.slice(10)
    assert tenth.limit_ns * 10 <= whole.limit_ns * 1.05


def test_a_slice_is_never_too_small_to_place_anything():
    """A long multi-start list would otherwise hand out budgets no solver can use."""
    deadline = Deadline(0, limit_ns=500 * MINIMUM_SLICE_NS)
    assert deadline.slice(10_000).limit_ns >= MINIMUM_SLICE_NS


def test_a_slice_never_exceeds_what_is_left():
    deadline = Deadline(0, limit_ns=1_000)
    assert deadline.slice(4).limit_ns <= 1_000


# ------------------------------------------------------- effort budgets

def test_an_effort_budget_bounds_search_nodes_regardless_of_clock_speed():
    """The reproducibility the effort budget promises: a caller who bounds by effort alone
    gets the same stopping point under a slow clock and a fast one, because
    nothing in the stopping decision depends on elapsed time."""
    box = Container.create("c", Dimensions.mm(200, 200, 200))
    order = instances("cube", 20, 20, 20, quantity=20)
    budget = EffortBudget(max_search_nodes=5)
    config = PackingConfig.balanced(time_limit_ms=60_000, effort_budget=budget, multi_start_orders=1)

    def run(clock):
        return SolverOrchestrator().solve(order, (box,), config, Deadline(config.time_limit_ms, clock=clock))

    slow, fast = run(CheckBudgetClock()), run(FastClock())
    slow_placed = {(p.instance.id, p.position.x, p.position.y, p.position.z)
                  for c in slow.solutions[0].containers for p in c.placements}
    fast_placed = {(p.instance.id, p.position.x, p.position.y, p.position.z)
                  for c in fast.solutions[0].containers for p in c.placements}
    assert slow_placed == fast_placed
    assert len(slow_placed) == 5


def test_an_effort_budget_leaves_time_only_behaviour_unchanged_when_absent():
    stats = SearchStats()
    deadline = Deadline(0, limit_ns=1_000_000)
    assert not deadline.expired
    assert deadline.effort_exceeded is False


def test_an_effort_budget_does_not_override_a_tighter_wall_clock():
    """The wall clock stays a live safety cutoff even with an effort budget set."""
    stats = SearchStats()
    already_expired = Deadline(0, limit_ns=-1)
    bounded = already_expired.with_effort(EffortBudget(max_search_nodes=1_000_000), stats)
    assert bounded.expired


def test_an_effort_budget_search_node_count_actually_stops_the_search():
    stats = SearchStats()
    generous_time = Deadline(60_000)
    bounded = generous_time.with_effort(EffortBudget(max_search_nodes=3), stats)
    assert not bounded.expired
    stats.search_nodes_expanded = 3
    assert bounded.expired


def test_max_restarts_bounds_the_portfolio_start_count():
    items = (*instances("a", 20, 20, 20), *instances("b", 15, 15, 15), *instances("c", 10, 10, 10))
    box = Container.create("box", Dimensions.mm(200, 200, 200))
    config = PackingConfig.balanced(solvers=("extreme_points",), multi_start_orders=8,
                                    effort_budget=EffortBudget(max_restarts=2))

    run = SolverOrchestrator().solve(items, (box,), config, Deadline(5_000))

    assert len(run.starts) == 2


def test_a_complete_grid_start_stops_the_portfolio_before_a_slower_one_wastes_the_deadline():
    """A single item type's regular lattice is always at least as good as any other
    in-portfolio solver on every objective key once it has placed everything, so a
    slower start after it can only spend what remains of the deadline on an answer that
    is provably no better. 2,000 identical cubes used to take the full 60s
    limit under the default balanced portfolio, with extreme_points alone reaching only
    297/2000 -- grid's already-complete answer sat unused the whole time."""
    items = instances("cube", 20, 20, 20, quantity=2_000)
    box = Container.create("box", Dimensions.mm(400, 400, 400), quantity=20)
    config = PackingConfig.balanced(time_limit_ms=60_000)

    run = SolverOrchestrator().solve(items, (box,), config, Deadline(60_000))

    solution, = run
    assert solution.solver_name == "grid:volume"
    assert not solution.unpacked
    started = [start.id for start in run.starts if start.started]
    assert started == ["grid:volume"]


# --------------------------------------------------------------- solver metrics

def test_candidate_metrics_count_geometry_and_support_work():
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    base, upper = instances("a", 40, 40, 40, quantity=2)
    state = extend(ContainerState(box, 1), base)
    stats = SearchStats()
    config = PackingConfig.balanced(minimum_support_ratio=0.5)

    find_candidates(
        state, upper, config, default_constraints(config), stats, generous(), None,
    )

    metrics = stats.to_metrics()
    assert metrics.candidate_points_considered > 0
    assert metrics.orientations_considered == stats.placements_attempted > 0
    assert metrics.feasible_candidates == stats.candidates_evaluated
    assert metrics.collision_checks > 0
    assert metrics.support_checks > 0


def test_space_and_node_metrics_count_their_own_solver_work():
    stats = SearchStats()
    subtract_all(
        [space(0, 0, 0, 100, 100, 100)],
        [solid(20, 20, 20, 30, 30, 30)],
        stats,
    )
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    beam_pack(
        box, 1, instances("a", 20, 20, 20, quantity=2),
        PackingConfig.balanced(), default_constraints(PackingConfig.balanced()),
        stats, generous(),
    )

    assert stats.space_partitions > 0
    assert stats.search_nodes_expanded > 0


# ---------------------------------------------------------- seeded randomness

def test_the_same_seed_replays_the_same_sequence():
    """Reproducibility is a documented promise of the library, and the multi-start
    orderings are drawn from here."""
    first = [DeterministicRandom(42).next_int(1_000) for _ in range(1)]
    assert [DeterministicRandom(42).next_int(1_000) for _ in range(1)] == first


def test_shuffles_are_reproducible_and_are_permutations():
    values = list(range(20))
    once = DeterministicRandom(7).shuffled(values)
    assert once == DeterministicRandom(7).shuffled(values)
    assert sorted(once) == values
    assert once != values, "a shuffle that changes nothing is not a shuffle"


def test_different_seeds_diverge():
    values = list(range(20))
    assert DeterministicRandom(1).shuffled(values) != DeterministicRandom(2).shuffled(values)


def test_the_sequence_is_pinned_against_the_other_implementations():
    """Golden vectors. Python and PHP must draw identically or the seeded multi-start
    orderings diverge and the two languages stop agreeing on the chosen packing. The
    mirror of this test lives in packvium-php/tests/SolverTest.php."""
    rng = DeterministicRandom(42)
    assert [rng.next_int(100) for _ in range(5)] == [98, 91, 19, 44, 31]
    assert DeterministicRandom(42).shuffled([0, 1, 2]) == [2, 1, 0]
    assert DeterministicRandom(7).shuffled(list(range(6))) == [1, 2, 4, 3, 5, 0]


def test_degenerate_bounds_are_answered_without_drawing():
    rng = DeterministicRandom(3)
    assert rng.next_int(0) == 0
    assert rng.next_int(1) == 0


def test_draws_stay_in_range_and_reach_every_value():
    """Rejection sampling removes the modulo bias a plain `% upper` would leave, so a
    small bound must still visit all of its values."""
    rng = DeterministicRandom(11)
    draws = [rng.next_int(3) for _ in range(3_000)]
    assert set(draws) == {0, 1, 2}
    assert all(300 < draws.count(value) < 1_700 for value in (0, 1, 2))


# ------------------------------------------------------------------ group batches

def test_ungrouped_items_are_offered_one_at_a_time():
    items = instances("a", 10, 10, 10, quantity=3)
    assert group_batches(items) == [(items[0],), (items[1],), (items[2],)]


def test_a_group_is_collected_into_a_single_batch():
    """Group members share a container, so a solver is offered all of them at once and
    can reject them as a whole."""
    pair = instances("pair", 10, 10, 10, quantity=2, group="kit")
    loose = instances("loose", 10, 10, 10)
    batches = group_batches([pair[0], loose[0], pair[1]])
    assert batches == [pair, (loose[0],)]


def test_a_group_keeps_the_position_of_its_first_member():
    pair = instances("pair", 10, 10, 10, quantity=2, group="kit")
    loose = instances("loose", 10, 10, 10)
    assert group_batches([loose[0], pair[0], pair[1]]) == [(loose[0],), pair]


def test_two_groups_stay_separate():
    left = instances("l", 10, 10, 10, quantity=2, group="left")
    right = instances("r", 10, 10, 10, quantity=2, group="right")
    assert group_batches([*left, *right]) == [left, right]


# --------------------------------------------------------------- maximal spaces

def space(x, y, z, length, width, height) -> Space:
    return Space(Point(x, y, z), Dimensions(Length(length), Length(width), Length(height)))


def solid(x, y, z, length, width, height) -> AxisAlignedBox:
    return AxisAlignedBox(Point(x, y, z), Dimensions(Length(length), Length(width), Length(height)))


def test_carving_nothing_leaves_the_space_untouched():
    whole = space(0, 0, 0, 100, 100, 100)
    assert subtract_all([whole], []) == [whole]


def test_carving_the_whole_space_leaves_nothing():
    assert subtract_all([space(0, 0, 0, 100, 100, 100)], [solid(0, 0, 0, 100, 100, 100)]) == []


def test_a_disjoint_solid_does_not_carve():
    whole = space(0, 0, 0, 10, 10, 10)
    assert subtract_all([whole], [solid(50, 50, 50, 10, 10, 10)]) == [whole]


def test_no_remaining_space_overlaps_the_carved_solid():
    obstacle = solid(20, 20, 0, 30, 30, 100)
    for remaining in subtract_all([space(0, 0, 0, 100, 100, 100)], [obstacle]):
        assert not remaining.box.intersects(obstacle)


def test_no_remaining_space_is_contained_in_another():
    """Without this filter every placement multiplies the space list and the solver
    degenerates into an exponential scan of overlapping duplicates."""
    spaces = subtract_all([space(0, 0, 0, 100, 100, 100)], [solid(20, 20, 20, 30, 30, 30)])
    for index, one in enumerate(spaces):
        for position, other in enumerate(spaces):
            assert position == index or not other.box.contains(one.box)


def test_the_free_space_list_never_exceeds_the_budget():
    """Adversarial size variety defeats the containment-dominance filter above -- each
    carve keeps producing slabs that merely overlap rather than nest -- so the surviving
    list would otherwise grow roughly linearly with no bound. Sizes and
    offsets are a fixed deterministic pattern, not randomness, so a failure reproduces
    from the test alone."""
    spaces = [space(0, 0, 0, 3000, 3000, 3000)]
    for k in range(120):
        size = 20 + (k * 7) % 60
        x, y, z = (k * 131) % 2900, (k * 277) % 2900, (k * 419) % 2900
        spaces = subtract_all(spaces, [solid(x, y, z, size, size, size)])
        assert len(spaces) <= MAX_MAXIMAL_SPACES


def test_an_adversarial_item_mix_still_places_completely_within_the_space_budget():
    """The same growth pattern, driven through the real solver: varied item sizes that
    keep carving non-nested residual spaces. Without a budget this reaches several
    hundred surviving spaces for 200 items; capped, the solver still places every one
    soundly."""
    sizes = (8, 12, 16, 20, 24, 10, 14, 18, 22, 26)
    items = [
        instance
        for k in range(200)
        for instance in Item.create(
            f"i{k}",
            Dimensions.mm(sizes[k % len(sizes)], sizes[(k * 3) % len(sizes)], sizes[(k * 7) % len(sizes)]),
            0,
            quantity=1,
        ).instances()
    ]
    box = Container.create("c", Dimensions.mm(600, 600, 600))
    config = PackingConfig.balanced(time_limit_ms=60_000)

    solution = MaximalSpaceSolver().pack_one(box, 1, items, config, SearchStats(), Deadline(60_000))

    assert not solution.unpacked
    assert len(solution.state.placements) == 200
    assert_physically_sound(solution.state)


def test_every_free_point_is_still_covered_after_carving():
    """The decisive property of a maximal-space decomposition: carving may not lose
    room. Sampled on a lattice across the container."""
    obstacles = [solid(20, 20, 0, 30, 30, 100), solid(70, 0, 0, 30, 100, 40)]
    spaces = subtract_all([space(0, 0, 0, 100, 100, 100)], obstacles)
    for x in range(0, 100, 7):
        for y in range(0, 100, 7):
            for z in range(0, 100, 13):
                point = Point(x, y, z)
                if any(o.contains_point(point) for o in obstacles):
                    continue
                assert any(s.box.contains_point(point) for s in spaces), f"lost free point {point}"


# --------------------------------------------------------------- container state

def test_an_empty_container_offers_its_origin():
    state = ContainerState(Container.create("c", Dimensions.mm(100, 100, 100)), 1)
    assert list(state.points) == [(0, 0, 0)]


def test_an_obstacle_at_the_origin_replaces_it_with_its_own_corners():
    obstacle = AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(50, 50, 50))
    state = ContainerState(
        Container.create("c", Dimensions.mm(100, 100, 100), obstacles=(Obstacle("o", obstacle),)), 1
    )
    assert (0, 0, 0) not in state.points
    assert (obstacle.x2, 0, 0) in state.points


def test_a_multi_box_obstacle_blocks_every_one_of_its_boxes():
    """A wheel-arch-style union: both boxes must be avoided, not just the
    first one an obstacle happened to be constructed with."""
    near = AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(40, 100, 100))
    far = AxisAlignedBox(Point(Length.mm(60).ticks, 0, 0), Dimensions.mm(40, 100, 100))
    arch = Obstacle("arch", near, additional_boxes=(far,))
    state = ContainerState(Container.create("c", Dimensions.mm(100, 100, 100), obstacles=(arch,)), 1)
    assert (0, 0, 0) not in state.points
    assert (far.origin.x, 0, 0) not in state.points
    assert (near.x2, 0, 0) in state.points


def test_an_item_is_placed_in_the_gap_between_two_obstacle_boxes():
    near = AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(40, 100, 100))
    far = AxisAlignedBox(Point(Length.mm(60).ticks, 0, 0), Dimensions.mm(40, 100, 100))
    box = Container.create("c", Dimensions.mm(100, 100, 100), obstacles=(Obstacle("arch", near, additional_boxes=(far,)),))
    items = instances("gap-filler", 20, 100, 100)
    solution = ExtremePointSolver().pack_one(box, 1, items, PackingConfig.balanced(), SearchStats(), generous())
    assert len(solution.state.placements) == 1
    placed = solution.state.placements[0]
    assert not placed.envelope_box.intersects(near)
    assert not placed.envelope_box.intersects(far)


def test_points_outside_the_container_are_discarded():
    state = ContainerState(Container.create("c", Dimensions.mm(100, 100, 100)), 1)
    inner = state.container.inner_dimensions
    assert all(p.x < inner.length.ticks and p.y < inner.width.ticks and p.z < inner.height.ticks
               for p in state.points.values())


def test_a_copied_state_does_not_share_mutable_bookkeeping():
    """The beam and the branch-and-bound search both extend a state by copying it, so a
    child that wrote through to its parent would corrupt every sibling branch."""
    state = ContainerState(Container.create("c", Dimensions.mm(100, 100, 100)), 1)
    clone = state.copy()
    extend(state, next(iter(instances("a", 10, 10, 10))))

    assert len(state.placements) == 1
    assert clone.placements == []
    assert clone.bounds == []
    assert list(clone.points) == [(0, 0, 0)]


# ------------------------------------------------------------------- candidates

def config_and_constraints(**kwargs):
    config = PackingConfig.balanced(**kwargs)
    return config, default_constraints(config)


def test_the_single_best_candidate_is_the_minimum_of_the_full_list():
    config, constraints = config_and_constraints()
    state = ContainerState(Container.create("c", Dimensions.mm(100, 100, 100)), 1)
    item, = instances("a", 30, 20, 10)

    every = find_candidates(state, item, config, constraints, SearchStats(), generous(), None)
    best = find_candidates(state, item, config, constraints, SearchStats(), generous(), 1)
    assert len(best) == 1
    assert best[0].score == min(c.score for c in every)


def test_a_wider_beam_returns_a_sorted_prefix():
    config, constraints = config_and_constraints()
    state = ContainerState(Container.create("c", Dimensions.mm(100, 100, 100)), 1)
    item, = instances("a", 30, 20, 10)

    top = find_candidates(state, item, config, constraints, SearchStats(), generous(), 3)
    assert len(top) <= 3
    assert [c.score for c in top] == sorted(c.score for c in top)


def test_an_item_that_cannot_fit_yields_no_candidate():
    config, constraints = config_and_constraints()
    state = ContainerState(Container.create("c", Dimensions.mm(10, 10, 10)), 1)
    item, = instances("a", 100, 100, 100)
    assert find_candidates(state, item, config, constraints, SearchStats(), generous(), None) == []


def test_a_full_container_is_rejected_before_any_geometry_is_tried():
    config, constraints = config_and_constraints()
    box = Container.create("c", Dimensions.mm(100, 100, 100), max_items=1)
    first, second = instances("a", 10, 10, 10, quantity=2)
    state = extend(ContainerState(box, 1), first)
    assert find_candidates(state, second, config, constraints, SearchStats(), generous(), None) == []


def test_the_payload_limit_stops_further_candidates():
    config, constraints = config_and_constraints()
    box = Container.create("c", Dimensions.mm(100, 100, 100), max_payload="1 kg")
    state = ContainerState(box, 1)
    heavy, = instances("h", 10, 10, 10, weight="2 kg")
    assert find_candidates(state, heavy, config, constraints, SearchStats(), generous(), None) == []


# ---------------------------------------------------------------------- solvers

ALL_SOLVERS = [ExtremePointSolver(), LayerSolver(), GridSolver(), MaximalSpaceSolver(), ExactSmallSolver()]


@pytest.mark.parametrize("solver", ALL_SOLVERS, ids=lambda s: s.name)
def test_every_solver_fills_an_exactly_divisible_container(solver):
    box = Container.create("c", Dimensions.mm(200, 200, 200))
    items = instances("cube", 100, 100, 100, quantity=4)
    solution = solver.pack_one(box, 1, items, PackingConfig.balanced(), SearchStats(), generous())

    assert len(solution.state.placements) == 4
    assert solution.unpacked == ()
    assert_physically_sound(solution.state)


@pytest.mark.parametrize("solver", ALL_SOLVERS, ids=lambda s: s.name)
def test_every_solver_respects_an_obstacle(solver):
    post = Obstacle("post", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(50, 50, 100)))
    box = Container.create("c", Dimensions.mm(100, 100, 100), obstacles=(post,))
    items = instances("a", 40, 40, 40, quantity=2)
    solution = solver.pack_one(box, 1, items, PackingConfig.balanced(), SearchStats(), generous())
    assert_physically_sound(solution.state)


@pytest.mark.parametrize("solver", ALL_SOLVERS, ids=lambda s: s.name)
def test_every_solver_reports_what_it_could_not_place(solver):
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    items = instances("cube", 90, 90, 90, quantity=3)
    solution = solver.pack_one(box, 1, items, PackingConfig.balanced(), SearchStats(), generous())

    placed = {p.instance.id for p in solution.state.placements}
    assert placed | {i.id for i in solution.unpacked} == {i.id for i in items}
    assert not placed & {i.id for i in solution.unpacked}


# --------------------------------------------------------------- lattice solver

def test_the_lattice_applies_to_one_geometrically_fungible_profile():
    assert GridSolver().supports(instances("a", 10, 10, 10, quantity=4))
    assert not GridSolver().supports([*instances("a", 10, 10, 10), *instances("b", 20, 20, 20)])
    assert not GridSolver().supports([])


def test_the_lattice_admits_a_declared_type_that_is_a_rotation_of_another():
    """Two declared types whose dimensions are 90-degree rotations of
    each other, with full rotation freedom, are the same physical item for lattice
    purposes -- a real catalog pattern (the same carton listed under two SKUs)."""
    swapped = instances("b", 12, 6, 20)  # "a" is 6x12x20; this is length/width swapped.
    assert GridSolver().supports([*instances("a", 6, 12, 20, quantity=11), *swapped])


def test_the_lattice_keeps_different_declared_nesting_types_out_of_one_column():
    """Nesting permits physical overlap only between instances of the same item type.

    Treating equal nesting profiles as fungible used the first type's lattice step for
    both and produced the illegal z=0/60 overlap in this 160mm-high container.
    """
    box = Container.create("c", Dimensions.mm(100, 100, 160))
    nesting = {
        "nesting_height": Length.mm(40),
        "allowed_rotations": (Rotation.LWH,),
    }
    items = [*instances("a", 100, 100, 100, **nesting),
             *instances("b", 100, 100, 100, **nesting)]

    assert not GridSolver().supports(items)
    solution = GridSolver().pack_one(
        box, 1, items, PackingConfig.fast(), SearchStats(), generous()
    )

    assert len(solution.state.placements) == 1
    assert len(solution.unpacked) == 1
    assert_physically_sound(solution.state)


def test_twelve_rotation_equivalent_items_split_across_two_types_still_share_one_container():
    box = Container.create("c", Dimensions.mm(25, 36, 21))
    a = instances("a", 6, 12, 20, quantity=11)
    b = instances("b", 12, 6, 20)  # length/width swapped relative to "a"
    solution = GridSolver().pack_one(box, 1, [*a, *b], PackingConfig.balanced(), SearchStats(), generous())

    assert solution.unpacked == ()
    assert len(solution.state.placements) == 12
    for placement in solution.state.placements:
        # The invariant a mixed-declared-type lattice must never break: the stored
        # rotation, applied to *that instance's own* declared dimensions, must
        # reproduce the physical box actually placed -- not a rotation borrowed from
        # a different declared type sharing the same lattice.
        assert placement.dimensions == placement.instance.item.dimensions.rotated(placement.rotation)


def test_the_lattice_falls_back_rather_than_placing_mixed_types_at_the_wrong_size():
    """`pack_one` is not only ever reached through `_solvers`' pre-filter -- a caller
    naming solvers explicitly, or a future extension, could hand it a mixed-type list
    directly. Without its own guard it would place every item at the first item's
    envelope size, which is silently wrong geometry, not merely a missed placement."""
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    small = instances("small", 20, 20, 20, quantity=2)
    big = instances("big", 80, 80, 80)
    solution = GridSolver().pack_one(box, 1, [*small, *big], PackingConfig.balanced(), SearchStats(), generous())

    for placement in solution.state.placements:
        assert placement.dimensions == placement.instance.item.dimensions.rotated(placement.rotation)


def test_the_lattice_keeps_non_stackable_items_on_the_floor():
    """A regular stack cannot express a per-item rule, so the rule is folded into the
    layer count before any placement is made."""
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    items = instances("flat", 40, 40, 40, quantity=8, stackable=False)
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())

    assert all(p.envelope_origin.z == 0 for p in solution.state.placements)
    assert_physically_sound(solution.state)


def test_the_lattice_caps_its_layers_by_max_stacked_items():
    """A regular stack cannot express a per-item rule, so `max_stacked_items` is folded
    into the layer count rather than sending the scene to the general solver.

    Two items may rest above the bottom one, so a column is three tall -- not the five
    the 200mm box has room for. Asserting the *height* rather than the placed count is
    what makes this fail if the cap is dropped: without it the lattice fills all five
    layers and still reports a physically sound state."""
    # A cube, so no rotation can change the layer height and quietly cap `nz` first.
    box = Container.create("c", Dimensions.mm(40, 40, 200))
    items = instances("cube", 40, 40, 40, quantity=5, max_stacked_items=2)
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())

    tops = {p.envelope_origin.z for p in solution.state.placements}
    assert len(tops) == 3, f"a column of three, not {len(tops)}"
    assert len(solution.state.placements) == 3
    assert len(solution.unpacked) == 2
    assert solution.dominant_lattice
    assert_physically_sound(solution.state)


def test_the_lattice_honours_a_floor_only_item():
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    items = instances("f", 40, 40, 40, quantity=8, must_be_on_floor=True)
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())
    assert all(p.envelope_origin.z == 0 for p in solution.state.placements)


def test_the_lattice_is_all_or_nothing_for_a_group():
    """Half a group in a container is worse than none of it, so a lattice that cannot
    hold every member holds none."""
    box = Container.create("c", Dimensions.mm(100, 100, 100))
    items = instances("kit", 90, 90, 90, quantity=2, group="kit")
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())
    assert solution.state.placements == []
    assert len(solution.unpacked) == 2


def test_the_lattice_respects_a_payload_ceiling():
    box = Container.create("c", Dimensions.mm(200, 200, 200), max_payload="2 kg")
    items = instances("w", 100, 100, 100, quantity=8, weight="1 kg")
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())
    assert len(solution.state.placements) == 2


@pytest.mark.parametrize("require_coordinates", [True, False], ids=["materialized", "compact"])
def test_the_lattice_caps_cumulative_top_load_in_every_result_form(require_coordinates):
    """A 1.5 kg limit admits one 1 kg item above a base, never three above it."""
    box = Container.create("c", Dimensions.mm(100, 100, 200))
    items = instances(
        "crate", 100, 100, 50,
        quantity=4,
        weight="1 kg",
        max_top_load="1.5 kg",
        allowed_rotations=(Rotation.LWH,),
    )
    config = PackingConfig.fast(require_placement_coordinates=require_coordinates)

    solution = GridSolver().pack_one(box, 1, items, config, SearchStats(), generous())

    assert solution.state.placement_count == 2
    assert len(solution.unpacked) == 2
    if require_coordinates:
        assert solution.state.lattice_summary is None
        placements = tuple(solution.state.placements)
    else:
        assert solution.state.lattice_summary is not None
        assert solution.state.lattice_summary.layers_used == 2
        placements = solution.state.lattice_summary.expand(items)
    assert [placement.envelope_origin.z for placement in placements] == [0, Length.mm(50).ticks]


def test_the_lattice_defers_to_the_general_solver_when_obstacles_are_present():
    post = Obstacle("post", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(50, 50, 100)))
    box = Container.create("c", Dimensions.mm(100, 100, 100), obstacles=(post,))
    items = instances("a", 40, 40, 40, quantity=2)
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.balanced(), SearchStats(), generous())
    assert_physically_sound(solution.state)


def test_the_lattice_nests_identical_items_reducing_the_stacked_height():
    """A 100mm item nesting 40mm into the one below only advances the stack by 60mm
    per layer, fitting a third layer into a 220mm container that would otherwise
    only ever hold two at the full 100mm step."""
    box = Container.create("c", Dimensions.mm(100, 100, 220))
    items = instances("crate", 100, 100, 100, quantity=10, nesting_height=Length.mm(40))
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())
    assert len(solution.state.placements) == 3
    zs = sorted(p.envelope_origin.z for p in solution.state.placements)
    assert zs == [0, Length.mm(60).ticks, Length.mm(120).ticks]
    boundary = AxisAlignedBox(Point(0, 0, 0), box.inner_dimensions)
    for p in solution.state.placements:
        assert boundary.contains(p.envelope_box)
    # Each layer overlaps only its immediate neighbour, by exactly nesting_height.
    boxes = sorted((p.envelope_box for p in solution.state.placements), key=lambda b: b.origin.z)
    for lower, upper in zip(boxes, boxes[1:]):
        assert upper.origin.z < lower.z2 == lower.origin.z + Length.mm(100).ticks
        assert lower.z2 - upper.origin.z == Length.mm(40).ticks
    assert boxes[0].z2 < boxes[2].origin.z  # non-adjacent layers never overlap


def test_general_solver_uses_exact_nested_volume_at_the_reserve_boundary():
    box = Container.create("c", Dimensions.mm(100, 100, 200), void_fill_reserve_ratio=0.25)
    items = instances("crate", 100, 100, 100, quantity=2, nesting_height=Length.mm(50))

    # A reserve disables the closed-form lattice, so this exercises item-specific
    # inside-envelope points and exact candidate-volume admission in the general path.
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())

    assert len(solution.state.placements) == 2
    assert [placement.envelope_origin.z for placement in solution.state.placements] == [0, Length.mm(50).ticks]
    assert solution.state.used_volume_ticks == box.inner_dimensions.volume * 3 // 4


def test_nesting_height_reduces_to_the_ordinary_lattice_when_unset():
    box = Container.create("c", Dimensions.mm(100, 100, 220))
    items = instances("crate", 100, 100, 100, quantity=10)
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())
    assert len(solution.state.placements) == 2
    assert_physically_sound(solution.state)


def test_the_lattice_defers_to_the_general_solver_when_a_stack_density_limit_is_set():
    """Its single-layer heuristic only ever reasons about one item resting on one
    other, never the cumulative load a tall column presses through its own
    footprint, so it must hand off rather than silently ignore the limit."""
    box = Container.create("c", Dimensions.mm(1000, 1000, 300), max_stack_density="500 kg")
    items = instances("cube", 1000, 1000, 100, quantity=3, weight="400 kg")
    solution = GridSolver().pack_one(box, 1, items, PackingConfig.balanced(), SearchStats(), generous())
    assert len(solution.state.placements) == 1
    assert_physically_sound(solution.state)


# ------------------------------------------------ quantity compression

def test_the_lattice_ignores_require_placement_coordinates_by_default():
    """Unset (or explicit `True`) `require_placement_coordinates` must reproduce the
    exact prior per-item placement loop -- this is a strict opt-in addition."""
    box = Container.create("c", Dimensions.mm(1000, 1000, 1000))
    items = instances("cube", 100, 100, 100, quantity=50)
    default_solution = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())
    explicit_true = GridSolver().pack_one(
        box, 1, items, PackingConfig.fast(require_placement_coordinates=True), SearchStats(), generous(),
    )
    assert default_solution.state.lattice_summary is None
    assert explicit_true.state.lattice_summary is None
    assert len(default_solution.state.placements) == len(explicit_true.state.placements) == 50
    for a, b in zip(default_solution.state.placements, explicit_true.state.placements):
        assert a == b


def test_the_lattice_compact_fast_path_builds_no_per_item_placement():
    """The whole point of this path: opting out of coordinates must not construct one
    `Placement` per instance. Verified directly rather than only by timing, which
    would be flaky under load."""
    box = Container.create("c", Dimensions.mm(3000, 3000, 3000))
    items = instances("cube", 100, 100, 100, quantity=10_000)
    constructed = []
    original_init = Placement.__init__

    def counting_init(self, *a, **k):
        constructed.append(1)
        return original_init(self, *a, **k)

    try:
        Placement.__init__ = counting_init
        solution = GridSolver().pack_one(
            box, 1, items, PackingConfig.fast(require_placement_coordinates=False), SearchStats(), generous(),
        )
    finally:
        Placement.__init__ = original_init
    assert constructed == []
    assert solution.state.placements == []
    assert solution.state.lattice_summary is not None
    assert solution.state.lattice_summary.count == 10_000
    assert solution.unpacked == ()


def test_the_lattice_compact_fast_path_reconstructs_identical_placements():
    """`expand` must produce exactly what the O(n) loop would have, coordinate for
    coordinate -- the compact form is a deferral, not a loss of information."""
    box = Container.create("c", Dimensions.mm(370, 370, 370))
    items = instances("cube", 100, 100, 100, quantity=137)
    compact = GridSolver().pack_one(
        box, 1, items, PackingConfig.fast(require_placement_coordinates=False), SearchStats(), generous(),
    )
    expanded = GridSolver().pack_one(box, 1, items, PackingConfig.fast(), SearchStats(), generous())
    assert compact.state.lattice_summary.count == len(expanded.state.placements)
    rebuilt = compact.state.lattice_summary.expand(items)
    assert rebuilt == tuple(expanded.state.placements)


def test_the_lattice_compact_fast_path_never_engages_with_nesting_height():
    """Nesting overlap bookkeeping is not part of the closed-form aggregates, so the
    fast path must defer to the ordinary per-item loop regardless of the flag,
    exactly as GridSolver already defers for obstacles, axles, and stack density."""
    box = Container.create("c", Dimensions.mm(100, 100, 220))
    items = instances("crate", 100, 100, 100, quantity=10, nesting_height=Length.mm(40))
    solution = GridSolver().pack_one(
        box, 1, items, PackingConfig.fast(require_placement_coordinates=False), SearchStats(), generous(),
    )
    assert solution.state.lattice_summary is None
    assert len(solution.state.placements) == 3


@pytest.mark.parametrize("length, width, quantity", [
    (100, 100, 1), (200, 100, 7), (300, 200, 47), (170, 130, 900), (1000, 1000, 3000),
])
def test_lattice_summary_aggregates_match_brute_force_expansion(length, width, quantity):
    """Cross-checks the closed-form `LatticeSummary` aggregates -- weight, used
    volume, top height, centre of mass -- against the existing per-placement
    functions run over a full expansion, across enough shapes to exercise a partial
    top layer, a partial top row, and an exact fit."""
    from packvium.centre_of_mass import centre_of_mass_offset_ppm as brute_force_com
    from packvium.nesting import used_volume as brute_force_used_volume

    box = Container.create("c", Dimensions.mm(length, width, 5000))
    items = instances("cube", 100, 100, 50, quantity=quantity, weight="2 kg")
    compact = GridSolver().pack_one(
        box, 1, items, PackingConfig.fast(require_placement_coordinates=False), SearchStats(), generous(),
    )
    summary = compact.state.lattice_summary
    assert summary is not None
    expanded = summary.expand(items)

    assert summary.total_weight_ticks == sum(p.instance.weight.ticks for p in expanded)
    assert summary.used_volume_ticks == brute_force_used_volume(expanded)
    assert summary.max_z_ticks == max(p.envelope_box.z2 for p in expanded)
    inner = box.inner_dimensions
    assert (summary.centre_of_mass_offset_ppm(inner.length.ticks, inner.width.ticks)
            == brute_force_com(inner, expanded))


# ------------------------------------------------------------------- exact search

# ------------------------------------------------------- exact-search tie-break

def test_is_better_container_state_prefers_more_items_regardless_of_volume():
    box = Container.create("c", Dimensions.mm(200, 200, 200))
    big, = instances("big", 150, 150, 150)
    small_a, small_b = instances("small", 20, 20, 20, quantity=2)
    fewer_but_bigger = extend(ContainerState(box, 1), big)
    more_but_smaller = extend(extend(ContainerState(box, 2), small_a), small_b)
    assert is_better_container_state(more_but_smaller, fewer_but_bigger)
    assert not is_better_container_state(fewer_but_bigger, more_but_smaller)


def test_is_better_container_state_breaks_an_item_count_tie_by_placed_volume():
    """The bug this closes: a branch-and-bound search that only compared item counts
    could let more search time replace a well-arranged tie with a worse-arranged one,
    silently discarding the two objective keys (unused volume, stack height) it never
    looked at."""
    box = Container.create("c", Dimensions.mm(200, 200, 200))
    small, = instances("small", 20, 20, 20)
    big, = instances("big", 80, 80, 80)
    smaller_state = extend(ContainerState(box, 1), small)
    bigger_state = extend(ContainerState(box, 2), big)
    assert is_better_container_state(bigger_state, smaller_state)
    assert not is_better_container_state(smaller_state, bigger_state)


def test_is_better_container_state_does_not_replace_an_equal_state():
    box = Container.create("c", Dimensions.mm(200, 200, 200))
    item, = instances("a", 50, 50, 50)
    state = extend(ContainerState(box, 1), item)
    assert not is_better_container_state(state, state)


def test_exact_equal_count_bound_keeps_a_larger_remaining_item_reachable():
    box = Container.create("c", Dimensions.mm(10, 10, 10))
    small, = instances("small", 4, 10, 10)
    large, = instances("large", 7, 10, 10)

    solution = ExactSmallSolver().pack_one(
        box,
        1,
        (small, large),
        PackingConfig.exact_small(),
        SearchStats(),
        generous(),
    )

    assert [placement.instance.item.id for placement in solution.state.placements] == ["large"]
    assert [item.item.id for item in solution.unpacked] == ["small"]


def test_the_exact_solver_refuses_an_order_beyond_its_limit():
    box = Container.create("c", Dimensions.mm(1_000, 1_000, 1_000))
    items = instances("a", 10, 10, 10, quantity=9)
    with pytest.raises(ValueError):
        ExactSmallSolver().pack_one(box, 1, items, PackingConfig.exact_small(exact_item_limit=8),
                                    SearchStats(), generous())


def test_a_completed_discrete_exact_search_does_not_claim_global_exhaustiveness():
    """Candidate enumeration is not a certificate for the public objective vector."""
    box = Container.create("c", Dimensions.mm(200, 200, 200))
    items = instances("cube", 100, 100, 100, quantity=4)
    solution = ExactSmallSolver().pack_one(box, 1, items, PackingConfig.exact_small(), SearchStats(), generous())
    assert not solution.exhaustive


def test_a_greedy_group_batch_forfeits_the_exhaustiveness_claim():
    """A multi-item batch is placed greedily rather than enumerated, so the tree is no
    longer a complete search of the placement space."""
    box = Container.create("c", Dimensions.mm(200, 200, 200))
    items = instances("kit", 100, 100, 100, quantity=2, group="kit")
    solution = ExactSmallSolver().pack_one(box, 1, items, PackingConfig.exact_small(), SearchStats(), generous())
    assert not solution.exhaustive


# ------------------------------------------------------------------ orchestration

def test_the_orchestrator_picks_solvers_from_the_profile():
    items = [*instances("a", 10, 10, 10), *instances("b", 20, 20, 20)]
    names = {solver.name for solver in SolverOrchestrator()._solvers(items, PackingConfig.quality())}
    assert {"homogeneous_blocks", "extreme_points", "maximal_spaces", "layer"} <= names
    assert SolverOrchestrator()._solvers(items, PackingConfig.fast())[0].name == "layer"


def test_support_diagnostic_skips_state_reconstruction_for_an_inactive_rule(monkeypatch):
    box = Container.create("box", Dimensions.mm(100, 100, 100))
    base, candidate = (
        instances("base", 50, 50, 50)[0],
        instances("candidate", 50, 50, 50)[0],
    )
    origin = Point(0, 0, 0)
    packed = (
        PackedContainer(
            box,
            1,
            (Placement(base, origin, Rotation.LWH, base.dimensions, origin, base.dimensions),),
        ),
    )

    def unexpected_reconstruction(self, *args, **kwargs):
        raise AssertionError("inactive support diagnostic reconstructed container state")

    monkeypatch.setattr(ContainerState, "__init__", unexpected_reconstruction)

    assert not SolverOrchestrator()._support_is_the_blocker(
        candidate, packed, PackingConfig.balanced()
    )


def test_explicit_solver_selection_overrides_the_profile_in_the_named_order():
    items = [*instances("a", 10, 10, 10), *instances("b", 20, 20, 20)]
    config = PackingConfig.quality(solvers=("layer", "extreme_points"))
    names = [solver.name for solver in SolverOrchestrator()._solvers(items, config)]
    assert names == ["layer", "extreme_points"]


def test_an_unknown_solver_name_is_a_structured_error_not_a_silent_fallback():
    items = instances("a", 10, 10, 10)
    config = PackingConfig.balanced(solvers=("brute_force",))
    with pytest.raises(UnknownSolverError):
        SolverOrchestrator()._solvers(items, config)


def test_explicit_solver_selection_drives_which_solver_actually_wins():
    items = instances("a", 40, 40, 40, quantity=4)
    containers = (Container.create("box", Dimensions.mm(100, 100, 100), quantity=1),)
    config = PackingConfig.balanced(solvers=("grid",))
    result = SolverOrchestrator().solve(items, containers, config, Deadline(2_000))
    assert all(solution.solver_name.startswith("grid") for solution in result)


def test_a_lattice_start_is_not_repeated_for_every_ordering():
    """A lattice ignores item order, so repeating it per ordering only burns budget."""
    items = instances("a", 10, 10, 10, quantity=4)
    starts = SolverOrchestrator()._starts(items, PackingConfig.quality())
    assert sum(1 for solver, _, _ in starts if solver.name == "grid") == 1


def test_orderings_are_distinct():
    items = [*instances("a", 10, 10, 10, weight="1 kg"), *instances("b", 30, 20, 10),
             *instances("c", 20, 20, 20, weight="3 kg")]
    orders = SolverOrchestrator()._orders(items, PackingConfig.quality())
    signatures = {tuple(i.id for i in order) for _, order in orders}
    assert len(signatures) == len(orders)


def test_an_expired_budget_still_produces_a_reportable_answer():
    """Solvers may not let a deadline escape: whatever is already placed is a valid
    packing and is worth far more than an empty result."""
    items = instances("m", 90, 80, 70, quantity=40)
    containers = (Container.create("c", Dimensions.mm(1_000, 1_000, 1_000), quantity=3),)
    solutions = SolverOrchestrator().solve(items, containers, PackingConfig.balanced(time_limit_ms=15), Deadline(15))
    assert solutions
    for solution in solutions:
        packed = {p.instance.id for c in solution.containers for p in c.placements}
        assert packed | {u.instance.id for u in solution.unpacked} == {i.id for i in items}


# ------------------------------------------------ global/block quality search

def test_additive_cardinality_bound_is_an_admissible_relaxation():
    """Ignoring geometry can overestimate what fits, but must never underestimate it.

    For small sets enumerate every subset and prove the cheapest-first closed form is
    exactly the maximum cardinality admitted by the additive resource. Geometry and
    business constraints can only lower the real packing count from there.
    """
    for costs in ((2, 3, 5), (1, 4, 4, 9), (3, 3, 3, 7, 8)):
        for capacity in range(sum(costs) + 1):
            exact = max(
                (len(chosen) for size in range(len(costs) + 1)
                 for chosen in combinations(costs, size) if sum(chosen) <= capacity),
                default=0,
            )
            assert _maximum_count_with_capacity(costs, capacity) == exact


def test_more_block_search_effort_cannot_worsen_the_objective():
    box = Container.create("box", Dimensions.mm(300, 200, 200), quantity=1)
    order = (
        *instances("large", 100, 100, 100, quantity=8),
        *instances("small", 60, 50, 40, quantity=20),
    )
    solver = HomogeneousBlockSolver()
    low = solver.pack_one(
        box, 1, order,
        PackingConfig.quality(container_plan_node_limit=1),
        SearchStats(), generous(),
    )
    high = solver.pack_one(
        box, 1, order,
        PackingConfig.quality(container_plan_node_limit=100_000),
        SearchStats(), generous(),
    )
    low_key = (len(low.unpacked), -low.state.used_volume_ticks, low.state.max_z)
    high_key = (len(high.unpacked), -high.state.used_volume_ticks, high.state.max_z)
    assert high_key <= low_key
    assert_physically_sound(high.state)


def test_block_search_falls_back_when_a_business_rule_can_distinguish_placements():
    box = Container.create("box", Dimensions.mm(100, 100, 200), quantity=1)
    order = instances("fragile", 100, 100, 100, quantity=2, stackable=False)
    solution = HomogeneousBlockSolver().pack_one(
        box, 1, order, PackingConfig.quality(), SearchStats(), generous(),
    )
    assert len(solution.state.placements) == 1
    assert len(solution.unpacked) == 1
    assert_physically_sound(solution.state)
