"""The solver paths an ordinary request rarely walks: fallbacks, bounds and early exits.

Each test pins one of them through the smallest scene that reaches it -- a deadline that
expires mid-lattice, a container the block solver cannot model, a custom solver that runs
out of time or claims more than it proved. A path nobody exercises is a path whose answer
nobody has checked, and these are exactly the ones a busy host or an unusual catalog hits.

Every determinism-sensitive test uses a fixed deadline (a zero budget, a counted effort
budget or an injected clock), never the wall clock deciding the outcome.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from packvium import (AxisAlignedBox, Container, Dimensions, EffortBudget, ExtensionRegistry, FixedPlacement,
                      Item, Length, Packer, PackingConfig, PackingStatus, Placement, Point, Rotation,
                      pack_from_dict, use_trace, validate)
from packvium.geometry import ShapeType
from packvium.solvers import (ContainerState, Deadline, DefaultContainerSelector, DeterministicRandom,
                              ExtremePointSolver, GridSolver, HomogeneousBlockSolver, LayerSolver,
                              PortfolioRun, RawSolution, SearchStats, SingleContainerSolution,
                              SolverOrchestrator, Space, TimeLimitReached, _ContainerPlan,
                              _nesting_points, _run_concurrent_start, beam_pack,
                              default_constraints, find_candidates, subtract_all)
from packvium import bounds
from packvium.units import Weight
from support import assert_sound, container, item

GENEROUS_MS = 60_000


def generous() -> Deadline:
    return Deadline(GENEROUS_MS)


def expired() -> Deadline:
    """A zero budget is already spent on the first reading, whatever the clock says."""
    return Deadline(0)


def expiring_after(readings: int) -> Deadline:
    """A deadline whose injected clock stays at zero for `readings` reads, then jumps.

    The first read is the deadline's own start, so `expiring_after(2)` survives exactly one
    `expired` check before tripping.
    """
    remaining = iter([0] * readings)
    return Deadline(1, clock=lambda: next(remaining, 10**18))


def items_of(items):
    return tuple(instance for one in items for instance in one.instances())


def counted(config_factory=PackingConfig.balanced, **kwargs) -> PackingConfig:
    """A configuration bounded by counted work, with the wall clock only as a fuse."""
    return config_factory(time_limit_ms=GENEROUS_MS,
                          effort_budget=EffortBudget(max_candidates_evaluated=100_000), **kwargs)


def cushion(id: str, **kwargs) -> Item:
    return Item.create(id, Dimensions.mm(100, 100, 100), shape_type=ShapeType.COMPRESSIBLE,
                       compression_ratio_ppm=250_000, max_compression_pressure_kpa=100, **kwargs)


# ----------------------------------------------------------------- custom solvers


class OutOfTime:
    """A solver that always reports its budget spent before placing anything."""

    name = "out_of_time"

    def __init__(self):
        self.calls = 0

    def pack_one(self, container, sequence, items, config, stats, deadline):
        self.calls += 1
        raise TimeLimitReached("packing time limit reached")


class ClaimsExhaustive:
    """Delegates to the built-in search and then claims its answer was exhaustive."""

    name = "a_claims_exhaustive"

    def pack_one(self, container, sequence, items, config, stats, deadline):
        solution = ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)
        return replace(solution, exhaustive=True)


class StacksInOnePlace:
    """Puts every item at the origin: fewer containers than any honest packing."""

    name = "a_stacks_in_one_place"

    def pack_one(self, container, sequence, items, config, stats, deadline):
        state = ContainerState(container, sequence)
        origin = Point(0, 0, 0)
        for instance in items:
            state.add(Placement(instance, origin, Rotation.LWH, instance.dimensions, origin, instance.dimensions))
        return SingleContainerSolution(state, ())


class NamedLikeTheExtremePointSolver:
    """Shares the built-in name but is not that solver, so no refinement may build on it."""

    name = "extreme_points"

    def pack_one(self, container, sequence, items, config, stats, deadline):
        return LayerSolver().pack_one(container, sequence, items, config, stats, deadline)


def packer(config: PackingConfig, **extensions) -> Packer:
    return Packer(config, ExtensionRegistry(**extensions))


def test_a_custom_solver_that_runs_out_of_time_leaves_the_greedy_plan_to_the_others():
    solver = OutOfTime()
    items, containers = [item("a", 40, 40, 40, quantity=3)], [container("c", 100, 100, 100)]
    result = packer(counted(), solvers=(solver,)).pack(items, containers)
    assert solver.calls > 0
    assert result.complete
    assert not result.algorithm.solver.startswith("out_of_time")
    assert_sound(result, items, containers)


def test_a_custom_solver_that_runs_out_of_time_ends_its_container_plan_beam():
    solver = OutOfTime()
    items, containers = [item("a", 40, 40, 40, quantity=3)], [container("c", 100, 100, 100)]
    config = counted(PackingConfig.balanced, container_plan_beam_width=2, container_plan_node_limit=8)
    result = packer(config, solvers=(solver,)).pack(items, containers)
    assert solver.calls > 0
    assert result.complete
    assert_sound(result, items, containers)


def test_a_custom_solver_that_runs_out_of_time_on_a_preloaded_container_keeps_the_fixed_item():
    solver = OutOfTime()
    cube = Item.create("cube", Dimensions.mm(100, 100, 100), quantity=3)
    box = Container.create("box", Dimensions.mm(200, 100, 200))
    fixed = [FixedPlacement("cube", "box", Point(0, 0, 0), Rotation.LWH)]
    result = packer(counted(), solvers=(solver,)).pack([cube], [box], fixed)
    assert solver.calls > 0
    assert result.complete
    assert any(placement.fixed for packed in result.containers for placement in packed.placements)


def test_a_fixed_item_in_an_unlimited_container_type_opens_that_container_first():
    cube = Item.create("cube", Dimensions.mm(100, 100, 100), quantity=2)
    box = Container.create("box", Dimensions.mm(100, 100, 200))
    assert box.quantity is None
    fixed = [FixedPlacement("cube", "box", Point(0, 0, 0), Rotation.LWH)]
    result = Packer(counted()).pack([cube], [box], fixed)
    assert result.complete
    first = result.containers[0]
    assert [placement.fixed for placement in first.placements] == [True, False]


def test_a_custom_solver_claiming_its_search_exhaustive_proves_nothing_by_that_claim():
    """OPTIMALITY-GAP.md Q1: exhaustion covers only the candidates a search generated, so it
    is not a proof. The bound is: a perfect fill attains it and is optimal whatever the solver
    claimed, and an answer that misses it is not, however exhaustive."""
    config = counted(solvers=("extreme_points",))
    fills = packer(config, solvers=(ClaimsExhaustive(),)).pack(
        [item("a", 50, 100, 100, quantity=2)], [container("c", 100, 100, 100)])
    assert fills.algorithm.solver.startswith("a_claims_exhaustive")
    assert fills.status is PackingStatus.OPTIMAL
    assert fills.to_dict()["optimality"] == {"code": "proven_optimal",
                                             "lower_bound": [0, 1, 0, 0, 1000000]}
    misses = packer(config, solvers=(ClaimsExhaustive(),)).pack(
        [item("a", 30, 30, 30, quantity=2)], [container("c", 100, 100, 100)])
    assert misses.algorithm.solver.startswith("a_claims_exhaustive")
    assert misses.status is PackingStatus.FEASIBLE
    assert "gap_key" in misses.to_dict()["optimality"]


def test_a_custom_solution_the_validator_rejects_never_beats_a_valid_one():
    """Stacking both cubes in one spot would use one container instead of two -- a
    better score, and an impossible packing. The validator must keep it from winning."""
    items = [item("a", 100, 100, 100, quantity=2)]
    containers = [container("c", 100, 100, 100, quantity=2)]
    config = counted(solvers=("extreme_points",))
    result = packer(config, solvers=(StacksInOnePlace(),)).pack(items, containers)
    assert result.status is PackingStatus.FEASIBLE
    assert not result.algorithm.solver.startswith("a_stacks_in_one_place")
    assert len(result.containers) == 2
    assert_sound(result, items, containers)


def test_a_neighbourhood_start_is_skipped_when_no_extreme_point_answer_exists_to_refine():
    items, containers = [item("a", 40, 40, 40, quantity=2), item("b", 30, 30, 30)], [container("c", 100, 100, 100)]
    config = counted(PackingConfig.quality, solvers=("layer",), container_plan_beam_width=2)
    result = packer(config, solvers=(NamedLikeTheExtremePointSolver(),)).pack(items, containers)
    refinements = [start for start in result.to_dict()["termination"]["starts"]
                   if ":neighborhood:" in start["id"] or start["id"].endswith(":beam")]
    assert refinements
    assert not any(start["started"] for start in refinements)
    assert result.complete


# ---------------------------------------------------------------- portfolio starts


def test_a_quality_beam_without_the_extreme_point_solver_runs_only_greedy_starts():
    items = [item("a", 40, 40, 40, quantity=2), item("b", 30, 30, 30)]
    config = counted(PackingConfig.quality, solvers=("layer",), container_plan_beam_width=2)
    starts = SolverOrchestrator()._starts(items_of(items), config)
    assert starts
    assert all(name.endswith(":greedy") for _, name, _ in starts)


def test_a_custom_ordering_can_fill_the_last_multi_start_slot():
    class Reverse:
        name = "reverse"

        def order(self, items, seed):
            return tuple(reversed(items))

    instances = items_of([item("a", 10, 10, 10, quantity=2)])
    config = PackingConfig.balanced(multi_start_orders=2)
    orders = SolverOrchestrator(custom_orders=(Reverse(),))._orders(instances, config)
    assert [name for name, _ in orders] == ["volume", "reverse"]
    assert [instance.id for instance in orders[1][1]] == ["a#2", "a#1"]


def test_a_portfolio_run_indexes_its_solutions_in_order():
    first = RawSolution("first", (), (), SearchStats())
    second = RawSolution("second", (), (), SearchStats())
    run = PortfolioRun((first, second), ())
    assert run[0] is first and run[1] is second
    assert len(run) == 2 and list(run) == [first, second]


def test_the_concurrent_portfolio_with_no_time_left_returns_the_fallback_without_starting_workers():
    instances = items_of([item("a", 40, 40, 40, quantity=2)])
    config = PackingConfig.balanced(parallel_starts=2)
    run = SolverOrchestrator().solve(instances, (container("c", 100, 100, 100),), config, expired())
    solution, = run.solutions
    assert solution.solver_name == "portfolio:fallback"
    assert all(not record.started for record in run.starts[:-1])


def test_the_concurrent_portfolio_stops_after_a_lattice_that_placed_everything():
    items, containers = [item("a", 50, 50, 50, quantity=8)], [container("c", 100, 100, 100)]
    result = Packer(PackingConfig.balanced(time_limit_ms=GENEROUS_MS, parallel_starts=2)).pack(items, containers)
    assert result.algorithm.solver.startswith("grid")
    starts = result.to_dict()["termination"]["starts"]
    assert starts[0]["started"] and starts[0]["selected"]
    assert not any(start["started"] for start in starts[1:])


def test_a_worker_start_counts_its_own_effort_against_the_shared_budget():
    instances = items_of([item("a", 40, 40, 40, quantity=4)])
    budget = EffortBudget(max_candidates_evaluated=1)
    _, unpacked, _, reached, _, stats, effort_stopped = _run_concurrent_start(
        ExtremePointSolver(), instances, (container("c", 100, 100, 100),), PackingConfig.balanced(),
        DefaultContainerSelector(), budget, generous().started + GENEROUS_MS * 1_000_000,
    )
    assert reached
    assert effort_stopped
    assert stats.candidates_evaluated == 1
    assert len(unpacked) > 0


# ------------------------------------------------------------ container-plan beam


def plan(remaining, containers) -> _ContainerPlan:
    return _ContainerPlan((), tuple(remaining), {c.id: c.quantity for c in containers},
                          {c.id: 0 for c in containers})


def test_the_container_lower_bound_ignores_volume_for_nesting_items():
    crate = Item.create("crate", Dimensions.mm(100, 100, 100), quantity=3, nesting_height=Length.mm(40))
    box = container("c", 100, 100, 100)
    # By nominal volume three crates need three boxes; nested they need one.
    assert SolverOrchestrator._additional_container_lower_bound(plan(crate.instances(), [box]), [box]) == 0


def test_the_container_lower_bound_counts_payload_when_every_container_limits_it():
    heavy = Item.create("heavy", Dimensions.mm(10, 10, 10), weight="3kg", quantity=3)
    box = container("c", 100, 100, 100, max_payload="4kg")
    assert SolverOrchestrator._additional_container_lower_bound(plan(heavy.instances(), [box]), [box]) == 3


def test_the_container_lower_bound_ignores_payload_when_no_container_accepts_any():
    cube = Item.create("cube", Dimensions.mm(100, 100, 100), weight="1kg", quantity=2)
    box = container("c", 100, 100, 100, max_payload="0g")
    # Volume alone: two cubes, one cube's worth of space each.
    assert SolverOrchestrator._additional_container_lower_bound(plan(cube.instances(), [box]), [box]) == 2


def test_a_root_bound_past_its_ceiling_is_left_unset_and_the_pack_still_succeeds():
    # F5's counterpart here: past the declared ceiling the bound is withheld, never raised.
    # A weight cannot pass the sum ceiling since admission caps the total at 2^63 - 1 ticks
    #, so two cubes at the 10^30 cubic-tick volume ceiling pass it instead.
    vast = item("vast", 625_000, 625_000, 625_000, quantity=2)
    assert bounds.for_request(vast.instances(), (container("c", 625_000, 625_000, 625_000),),
                              "default") is None
    result = Packer(counted(solvers=("exact_small",))).pack([vast], [container("c", 625_000, 625_000, 625_000)])
    assert result.complete


def test_the_container_plan_beam_prefers_one_container_that_holds_everything():
    items = [item("a", 100, 100, 100, quantity=3)]
    containers = [container("small", 100, 100, 100, quantity=3), container("big", 100, 100, 300, quantity=1)]
    config = counted(solvers=("extreme_points",), container_plan_beam_width=4, container_plan_node_limit=64)
    result = Packer(config).pack(items, containers)
    assert result.complete
    assert [packed.container.id for packed in result.containers] == ["big"]
    assert_sound(result, items, containers)


@pytest.mark.parametrize(("node_limit", "packed_count"), [(1, 4), (4, 4), (5, 5)])
def test_the_container_plan_beam_stops_expanding_at_its_effective_node_limit(node_limit, packed_count):
    # One type makes every trial add exactly one container. The beam's four-trial
    # floor applies to small budgets; a larger declared budget still stops search.
    items = [item("a", 100, 100, 100, quantity=6)]
    containers = [container("x", 100, 100, 100, quantity=6)]
    config = counted(solvers=("extreme_points",), container_plan_beam_width=4,
                     container_plan_node_limit=node_limit)
    result = Packer(config).pack(items, containers)
    assert len(result.containers) == packed_count
    assert len(result.unpacked) == 6 - packed_count
    assert_sound(result, items, containers)


def test_the_container_plan_beam_never_opens_more_of_a_type_than_the_request_has():
    items = [item("a", 100, 100, 100, quantity=3)]
    containers = [container("x", 100, 100, 100, quantity=1), container("y", 100, 100, 100, quantity=3)]
    config = counted(solvers=("extreme_points",), container_plan_beam_width=4, container_plan_node_limit=64)
    result = Packer(config).pack(items, containers)
    assert result.complete
    assert sorted(packed.container.id for packed in result.containers) == ["x", "y", "y"]


def test_the_container_plan_beam_names_support_as_the_reason_an_item_was_left_behind():
    """Two floor-bound ledges can cover at most half the floor, so the lid can never be
    fully carried; loading both ledges beats loading the lid, which then stays behind --
    for support, not for want of room."""
    ledge = item("ledge", 50, 50, 50, quantity=2, must_be_on_floor=True)
    lid = item("lid", 100, 100, 20, minimum_support_ratio=1.0, allowed_rotations=(Rotation.LWH,))
    items, containers = [ledge, lid], [container("c", 100, 100, 100, quantity=1)]
    config = counted(solvers=("extreme_points",), container_plan_beam_width=2, container_plan_node_limit=8)
    result = Packer(config).pack(items, containers)
    reasons = {unpacked.instance.item.id: unpacked.reason for unpacked in result.unpacked}
    assert reasons == {"lid": "insufficient_support"}


# ----------------------------------------------------------- support diagnostics


def test_support_is_not_the_blocker_when_nothing_fits_even_without_it():
    top = Item.create("top", Dimensions.mm(100, 100, 100), minimum_support_ratio=1.0)
    box = container("c", 100, 100, 100)
    full = ContainerState(box, 1)
    filler, = Item.create("filler", Dimensions.mm(100, 100, 100)).instances()
    full.add(Placement(filler, Point(0, 0, 0), Rotation.LWH, filler.dimensions, Point(0, 0, 0), filler.dimensions))
    packed, _ = SolverOrchestrator._packed_state(full)
    instance, = top.instances()
    assert not SolverOrchestrator()._support_is_the_blocker(instance, (packed,), PackingConfig.balanced())


def test_support_is_not_the_blocker_when_the_item_fits_with_support_in_some_container():
    top = Item.create("top", Dimensions.mm(50, 50, 50), minimum_support_ratio=1.0)
    packed, _ = SolverOrchestrator._packed_state(ContainerState(container("c", 100, 100, 100), 1))
    instance, = top.instances()
    assert not SolverOrchestrator()._support_is_the_blocker(instance, (packed,), PackingConfig.balanced())


# --------------------------------------------------------------- grid lattice


def test_the_grid_defers_a_self_incompatible_item_to_the_general_solver():
    """An item tagged with a tag it refuses cannot share a container with itself; only
    the general solver evaluates that rule per candidate."""
    instances = items_of([item("odd", 50, 50, 50, quantity=2, tags=frozenset({"x"}),
                               incompatible_tags=frozenset({"x"}))])
    solution = GridSolver().pack_one(container("c", 100, 100, 100), 1, instances,
                                     PackingConfig.balanced(), SearchStats(), generous())
    assert len(solution.state.placements) == 1
    assert not solution.dominant_lattice


def test_a_group_the_grid_could_not_finish_before_the_deadline_is_placed_not_at_all():
    instances = items_of([item("kit", 50, 50, 50, quantity=3, group="kit")])
    solution = GridSolver().pack_one(container("c", 100, 100, 100), 1, instances,
                                     PackingConfig.balanced(), SearchStats(), expiring_after(2))
    assert list(solution.state.placements) == []
    assert len(solution.unpacked) == 3
    assert solution.dominant_lattice


# --------------------------------------------------------- homogeneous blocks


def blocks(box: Container, instances, deadline: Deadline | None = None, **config):
    return HomogeneousBlockSolver().pack_one(box, 1, instances, PackingConfig.quality(**config),
                                             SearchStats(), deadline or generous())


def greedy_blocks(box: Container, instances, **config):
    """The block solver without `quality`'s lookahead: one greedy pass per mode."""
    config.setdefault("container_plan_node_limit", 100_000)
    return HomogeneousBlockSolver().pack_one(box, 1, instances, PackingConfig.balanced(**config),
                                             SearchStats(), generous())


def test_the_block_solver_places_nothing_once_the_deadline_has_passed():
    instances = items_of([item("a", 50, 50, 50, quantity=2)])
    solution = blocks(container("c", 100, 100, 100), instances, expired())
    assert list(solution.state.placements) == []
    assert solution.unpacked == instances
    assert solution.time_limit_reached


@pytest.mark.parametrize("rules", [
    {"tag_limits": {"x": 1}}, {"max_stack_density": "1kg"}, {"void_fill_reserve_ratio": 0.1},
])
def test_the_block_solver_defers_container_rules_it_cannot_model(rules):
    instances = items_of([item("a", 50, 50, 50, quantity=2)])
    solution = blocks(container("c", 100, 100, 100, **rules), instances)
    assert len(solution.state.placements) == 2


def test_the_block_solver_caps_a_block_at_the_container_item_limit():
    instances = items_of([item("a", 50, 50, 50, quantity=4)])
    solution = blocks(container("c", 100, 100, 100, max_items=3), instances)
    assert len(solution.state.placements) == 3


def test_the_block_solver_caps_a_block_at_the_remaining_payload():
    instances = items_of([item("a", 50, 50, 50, quantity=4, weight="1kg")])
    solution = blocks(container("c", 100, 100, 100, max_payload="2500g"), instances)
    assert len(solution.state.placements) == 2


def test_the_block_solver_skips_an_item_too_heavy_for_any_block():
    instances = items_of([item("anvil", 50, 50, 50, weight="3kg"), item("feather", 50, 50, 50, weight="1g")])
    solution = blocks(container("c", 100, 100, 100, max_payload="2kg"), instances)
    assert [p.instance.item.id for p in solution.state.placements] == ["feather"]
    assert [instance.item.id for instance in solution.unpacked] == ["anvil"]


def test_a_block_of_compressible_items_tracks_their_volume_under_current_load():
    instances = items_of([cushion("soft", quantity=2)])
    solution = blocks(container("c", 100, 100, 200), instances)
    assert len(solution.state.placements) == 2
    assert solution.state.compression_sensitive
    assert solution.state.used_volume_ticks == 2 * Dimensions.mm(100, 100, 100).volume


# ---------------------------------------------------------- candidate search


def test_a_compressible_item_is_admitted_against_the_void_fill_reserve_by_its_largest_volume():
    box = container("c", 100, 100, 200, void_fill_reserve_ratio=0.25)
    instance, = cushion("soft").instances()
    config = PackingConfig.balanced()
    candidates = find_candidates(ContainerState(box, 1), instance, config, default_constraints(config),
                                 SearchStats(), generous(), 1)
    assert [candidate.point for candidate in candidates] == [Point(0, 0, 0)]


def test_a_compressible_item_that_would_eat_into_the_void_fill_reserve_is_refused():
    box = container("c", 100, 100, 100, void_fill_reserve_ratio=0.25)
    instance, = cushion("soft").instances()
    config = PackingConfig.balanced()
    assert find_candidates(ContainerState(box, 1), instance, config, default_constraints(config),
                           SearchStats(), generous(), 1) == []


def test_asking_for_zero_candidates_returns_none():
    instance, = Item.create("a", Dimensions.mm(10, 10, 10)).instances()
    config = PackingConfig.balanced()
    assert find_candidates(ContainerState(container("c", 100, 100, 100), 1), instance, config,
                           default_constraints(config), SearchStats(), generous(), 0) == []


def test_an_item_that_does_not_nest_offers_no_nesting_points():
    instance, = Item.create("a", Dimensions.mm(10, 10, 10)).instances()
    assert _nesting_points(ContainerState(container("c", 100, 100, 100), 1), instance) == []


def test_a_candidate_blocked_by_a_placed_box_is_traced_as_a_collision():
    events = []
    items = [item("a", 60, 20, 20, quantity=3), item("b", 30, 30, 70)]
    with use_trace(events.append):
        Packer(counted(solvers=("extreme_points",))).pack(items, [container("c", 100, 100, 100)])
    assert any(event["type"] == "filter" and event["reason"] == "collision" for event in events)


def test_the_candidate_beam_skips_the_volume_bound_while_nesting_items_are_still_to_come():
    crate = Item.create("crate", Dimensions.mm(100, 100, 100), quantity=3, nesting_height=Length.mm(40))
    lid = Item.create("lid", Dimensions.mm(100, 100, 10))
    instances = items_of([lid, crate])
    config = PackingConfig.quality(container_plan_beam_width=2, container_plan_node_limit=64)
    solution = beam_pack(container("c", 100, 100, 250), 1, instances, config,
                         default_constraints(config), SearchStats(), generous())
    assert len(solution.state.placements) == 4


# --------------------------------------------------------- free-space and rng


def test_carving_a_box_that_misses_the_space_leaves_the_space_whole():
    space = Space(Point(0, 0, 0), Dimensions.mm(10, 10, 10))
    far = AxisAlignedBox(Point(Length.mm(20).ticks, 0, 0), Dimensions.mm(10, 10, 10))
    assert subtract_all([space], [far]) == [space]


def test_a_draw_in_the_biased_tail_is_redrawn_rather_than_folded():
    upper = 2**31 + 1
    bound = (1 << 32) - ((1 << 32) % upper)
    rng, replay = DeterministicRandom(1), DeterministicRandom(1)
    first = replay._bits() * 65536 + replay._bits()
    second = replay._bits() * 65536 + replay._bits()
    assert first >= bound, "seed 1 must open with a rejected draw for this test to mean anything"
    assert rng.next_int(upper) == second % upper


def _a_base_and_three_tops():
    """One 60x60x40 base and three 40x60x50 tops in a 100 mm crate: the base ends on two tops
    and overhangs them."""
    return items_of([item("base", 60, 60, 40), item("top", 40, 60, 50, quantity=3)])


def _resting_share(placements, placement) -> float:
    """The share of a placement's base on the floor or on a top face, from geometry alone."""
    box = placement.envelope_box
    if box.origin.z == 0:
        return 1.0
    resting = sum(other.envelope_box.overlap_area_xy(box) for other in placements
                  if other.envelope_box.z2 == box.origin.z)
    return resting / placement.envelope_dimensions.base_area


def test_a_block_set_on_a_smaller_one_reports_the_support_it_really_has():
    # The greedy pass sets the base on two tops; `quality`'s lookahead finds a layout with
    # no overhang at all, so the greedy pass is the one that exercises the report.
    solution = greedy_blocks(container("crate", 100, 100, 100, quantity=1), _a_base_and_three_tops())
    base, = [p for p in solution.state.placements if p.instance.item.id == "base"]
    assert base.envelope_origin.z > 0
    assert base.support_ratio == pytest.approx(5 / 6)
    assert base.support_ratio == _resting_share(solution.state.placements, base)
    assert {p.support_ratio for p in solution.state.placements if p.envelope_origin.z == 0} == {1.0}


def test_the_quality_lookahead_finds_a_layout_with_nothing_overhanging():
    box = container("crate", 100, 100, 100, quantity=1)
    greedy = greedy_blocks(box, _a_base_and_three_tops())
    looked = blocks(box, _a_base_and_three_tops())
    assert not looked.unpacked and not looked.time_limit_reached
    assert {p.support_ratio for p in looked.state.placements} == {1.0}
    assert looked.state.max_z < greedy.state.max_z


def test_the_block_solver_leaves_a_request_that_asks_for_support_to_the_per_item_search():
    solution = blocks(container("crate", 100, 100, 100, quantity=1), _a_base_and_three_tops(),
                      minimum_support_ratio=1.0)
    assert solution.state.placements
    assert {_resting_share(solution.state.placements, p) for p in solution.state.placements} == {1.0}


# ------------------------------------------------- rarely-walked solver branches


def _cushion(id: str) -> Item:
    """A 50 mm compressible cube, of the same lattice profile as `item(id, 50, 50, 50)`."""
    return Item.create(id, Dimensions.mm(50, 50, 50), "1kg", shape_type=ShapeType.COMPRESSIBLE,
                       compression_ratio_ppm=500_000, max_compression_pressure_kpa=100)


def test_a_compressible_type_never_joins_a_rigid_lattice():
    """Same size and weight once made one lattice profile, so a cushion joined a brick's
    lattice and was stacked with no crush check: the packer returned an `invalid_result`
    where a valid plan existed. The shape is part of the profile now."""
    brick, cushion = item("brick", 50, 50, 50, weight="1kg"), _cushion("cushion")

    assert not GridSolver().supports(items_of((brick, cushion)))
    assert GridSolver().supports(items_of((brick, item("alias", 50, 50, 50, weight="1kg"))))


def test_a_cushion_that_would_be_crushed_is_left_out_rather_than_stacked():
    request = {
        "units": {"length": "mm"},
        "items": [
            {"id": "rigid", "dimensions": {"length": "50", "width": "50", "height": "50"},
             "weight": {"value": "1", "unit": "kg"}},
            {"id": "soft", "quantity": 2, "dimensions": {"length": "50", "width": "50", "height": "50"},
             "weight": {"value": "1", "unit": "kg"}, "shape_type": "compressible",
             "compression_ratio": 0.5, "max_compression_pressure_kpa": 1},
        ],
        "containers": [{"id": "c", "inner_dimensions": {"length": "50", "width": "50", "height": "150"}}],
        "configuration": {"max_containers": 1, "time_limit_ms": 60000, "effort_budget": {
            "max_candidates_evaluated": 1000000, "max_placement_attempts": 1000000,
            "max_search_nodes": 1000000}},
    }

    result = pack_from_dict(request)

    assert result["status"] == "best_found"
    assert [u["item_type"] for u in result["unpacked_items"]] == ["soft"]
    assert validate(request, result) == []


def test_a_wider_candidate_list_on_a_one_plan_beam_ranks_its_children():
    """`max_candidates_per_item` keeps several children of the single plan a width-one beam
    holds, ordered by the node key, and that is what recovers the third item here: the
    greedy first corner for the first item leaves no room for the last one."""
    instances = items_of([item("i0", 6, 3, 5), item("i1", 4, 6, 4), item("i2", 5, 4, 5)])
    box = container("c", 8, 9, 6)

    def solve(candidates: int) -> SingleContainerSolution:
        config = PackingConfig.balanced(max_candidates_per_item=candidates)
        assert config.container_plan_beam_width == 1
        return ExtremePointSolver().pack_one(box, 1, instances, config, SearchStats(), generous())

    assert [one.id for one in solve(1).unpacked] == ["i2#1"]
    wide = solve(3)
    assert wide.unpacked == ()
    layout = [(p.instance.id, p.position) for p in wide.state.placements]
    assert layout == [(p.instance.id, p.position) for p in solve(3).state.placements]


def test_an_item_no_single_container_admits_on_every_count_keeps_no_feasible_placement():
    """One container is big enough but too weak, the other strong enough but too small. Each
    structural count passes somewhere, so no proof applies, and no container limit explains
    it either: the reason stays `no_feasible_placement` instead of blaming the inventory."""
    heavy = item("heavy", 50, 50, 50, weight="5kg")
    weak = container("weak", 100, 100, 100, max_payload=Weight.parse("1kg"))
    tiny = container("tiny", 40, 40, 40)
    result = Packer(counted()).pack([heavy], [weak, tiny])
    assert [(u.instance.id, u.reason) for u in result.unpacked] == [("heavy#1", "no_feasible_placement")]


def test_a_plan_beam_out_of_nodes_mid_round_keeps_its_best_complete_plan():
    """Two plans of width two over two container types use the four-node floor after the
    first plan's trials, so the second plan stops before its first trial. The incumbent
    already holds every item, and a node limit is not reported as a time limit."""
    instances = items_of([item("a", 50, 50, 50, quantity=3)])
    containers = (container("small", 50, 50, 50, cost_minor=1),
                  container("tall", 50, 50, 100, cost_minor=3))
    config = PackingConfig.quality(time_limit_ms=GENEROUS_MS, container_plan_beam_width=2,
                                   container_plan_node_limit=3)
    packed, unpacked, proven, reached, _ = SolverOrchestrator()._across_containers(
        ExtremePointSolver(), instances, containers, config, SearchStats(), generous())
    assert sorted((one.container.id, len(one.placements)) for one in packed) == [("small", 1), ("tall", 2)]
    assert unpacked == ()
    assert (proven, reached) == (False, False)
