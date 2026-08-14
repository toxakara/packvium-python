"""End-to-end packing through the public `Packer`.

Every scenario is checked twice: once for what it was supposed to achieve, and once
against the independent validator, so a test cannot pass on a count while the layout
underneath it is physically impossible.
"""

from __future__ import annotations

import pytest

from packvium import (AxisAlignedBox, Dimensions, EffortBudget, Length, Obstacle,
                         PackingConfig, PackingStatus, Point, Rotation, Weight)
from support import assert_sound, container, item, pack

PROFILES = {
    "fast": PackingConfig.fast,
    "balanced": PackingConfig.balanced,
    "quality": lambda **kw: PackingConfig.quality(time_limit_ms=2_000, **kw),
    "exact_small": lambda **kw: PackingConfig.exact_small(time_limit_ms=2_000, **kw),
}

DETERMINISTIC_PROFILES = {
    "fast": lambda: PackingConfig.fast(
        time_limit_ms=60_000,
        effort_budget=EffortBudget(max_search_nodes=500, max_restarts=1),
    ),
    "balanced": lambda: PackingConfig.balanced(
        time_limit_ms=60_000,
        effort_budget=EffortBudget(max_search_nodes=500, max_restarts=9),
    ),
    "quality": lambda: PackingConfig.quality(
        time_limit_ms=60_000,
        effort_budget=EffortBudget(max_search_nodes=500, max_restarts=9),
    ),
    "exact_small": lambda: PackingConfig.exact_small(
        time_limit_ms=60_000,
        effort_budget=EffortBudget(max_search_nodes=500, max_restarts=9),
    ),
}


# ---------------------------------------------------------------- basic packing

@pytest.mark.parametrize("profile", PROFILES.values(), ids=PROFILES)
def test_every_profile_fills_an_exactly_divisible_container(profile):
    items = [item("cube", 100, 100, 100, quantity=4)]
    containers = [container("box", 200, 200, 100)]
    result = pack(items, containers, profile())

    assert result.complete
    assert result.packed_item_count == 4
    assert len(result.containers) == 1
    assert_sound(result, items, containers)


def test_an_item_larger_than_every_container_is_reported_with_a_reason():
    items = [item("slab", 200, 200, 200)]
    containers = [container("box", 100, 100, 100)]
    result = pack(items, containers)

    assert not result.complete
    assert [u.reason for u in result.unpacked] == ["no_compatible_container_dimensions"]
    assert result.containers == ()


def test_an_item_heavier_than_every_container_allows_is_reported_with_a_reason():
    items = [item("anvil", 10, 10, 10, weight="10 kg")]
    containers = [container("box", 100, 100, 100, max_payload="1 kg")]
    result = pack(items, containers)
    assert [u.reason for u in result.unpacked] == ["payload_exceeded"]


def test_rotation_is_used_when_the_upright_orientation_will_not_fit():
    items = [item("plank", 120, 40, 60)]
    containers = [container("box", 60, 120, 40)]
    result = pack(items, containers)

    assert result.complete
    assert_sound(result, items, containers)


def test_keep_upright_forbids_the_rotation_that_would_have_helped():
    items = [item("plank", 120, 40, 60, keep_upright=True)]
    containers = [container("box", 60, 120, 40)]
    result = pack(items, containers)
    assert not result.complete
    # A plain dimension mismatch and "this item's own rotation restriction, and only
    # it, rules every container out" are different facts: the container
    # would hold the plank in the HLW orientation, but keep_upright forbids it.
    unpacked, = result.unpacked
    assert unpacked.reason == "rotation_restricted"
    assert unpacked.proof.level == "proven"


def test_a_final_layout_with_only_partial_support_reports_insufficient_support():
    """`base` (high priority, forced onto the floor) leaves only half its
    footprint available to support `topper`, whose own footprint exactly matches the
    container and cannot fit anywhere else -- the only geometric candidate for it
    exists, and support alone is what rejects it."""
    base = item("base", 30, 60, 20, allowed_rotations=[Rotation.LWH], must_be_on_floor=True, priority=10)
    topper = item("topper", 60, 60, 10, allowed_rotations=[Rotation.LWH], minimum_support_ratio=1.0)
    containers = [container("box", 60, 60, 40, quantity=1)]

    result = pack([base, topper], containers, PackingConfig.balanced(multi_start_orders=1))

    assert not result.complete
    unpacked, = result.unpacked
    assert unpacked.instance.id == "topper#1"
    assert unpacked.reason == "insufficient_support"
    assert unpacked.proof.level == "observed"


# ----------------------------------------------------------------- determinism

#: Fields describing the effort spent rather than the answer reached. The time budget
#: is wall clock, so how far a start gets before its slice runs out depends on the
#: host, not on the request. See `test_the_effort_counters_are_only_diagnostics`.
EFFORT_FIELDS = ("duration_ms", "candidates_evaluated", "placements_attempted", "metrics")


def chosen_answer(report: dict) -> dict:
    """The answer a caller receives, without the effort diagnostics or the runners-up.

    Alternatives are excluded because a start cut off mid-search reports whatever it
    had placed by then; two runs of the same request may legitimately disagree about a
    truncated runner-up while agreeing exactly on the winner.
    """
    for field in EFFORT_FIELDS:
        report["algorithm"].pop(field, None)
    for field in (
        "any_start_truncated",
        "all_required_starts_completed",
        "global_deadline_reached",
        "starts",
    ):
        report["termination"].pop(field, None)
    report.pop("alternatives", None)
    return report


@pytest.mark.parametrize("profile", DETERMINISTIC_PROFILES.values(), ids=DETERMINISTIC_PROFILES)
def test_the_same_request_produces_the_same_answer(profile):
    """A documented promise: the library is deterministic for a given seed, so a caller
    can cache, diff and reproduce a result. Counted work, not scheduler-dependent wall
    time, defines where search stops; the generous clock remains only a safety cutoff."""
    items = [item("a", 40, 30, 20, quantity=5), item("b", 60, 50, 40, quantity=3)]
    containers = [container("c", 150, 150, 150, quantity=3)]

    assert (chosen_answer(pack(items, containers, profile()).to_dict())
            == chosen_answer(pack(items, containers, profile()).to_dict()))


def test_the_effort_counters_are_only_diagnostics():
    """They are reported, and they are not part of what reproducibility covers.

    Counted work, not wall time, must decide where search stops here: three runs of a
    quality-search profile against a real (loaded, shared) clock can each get a
    different distance into the anytime search within the same millisecond window and
    legitimately land on different tied answers."""
    items = [item("a", 40, 30, 20, quantity=5)]
    containers = [container("c", 150, 150, 150, quantity=3)]
    config = PackingConfig.balanced(
        time_limit_ms=60_000,
        effort_budget=EffortBudget(max_search_nodes=500, max_restarts=9),
    )
    reports = [pack(items, containers, config).to_dict() for _ in range(3)]

    assert all(set(report["algorithm"]) >= set(EFFORT_FIELDS) for report in reports)
    answers = [chosen_answer(report) for report in reports]
    assert all(answer == answers[0] for answer in answers)


def test_a_different_seed_may_reorder_the_search_but_not_break_it():
    items = [item("a", 40, 30, 20, quantity=6)]
    containers = [container("c", 150, 150, 150, quantity=2)]
    for seed in (1, 7, 99):
        result = pack(items, containers, PackingConfig.balanced(seed=seed))
        assert_sound(result, items, containers)
        assert result.algorithm.seed == seed


# ------------------------------------------------------------ container choice

def test_a_payload_ceiling_splits_the_order_across_containers():
    items = [item("a", 50, 50, 50, quantity=2, weight="1 kg")]
    containers = [container("box", 200, 200, 200, max_payload="1.5 kg")]
    result = pack(items, containers, PackingConfig.fast())

    assert result.complete
    assert len(result.containers) == 2
    assert_sound(result, items, containers)


def test_the_cheaper_container_is_preferred_when_both_would_do():
    items = [item("a", 50, 50, 50)]
    containers = [container("cheap", 100, 100, 100, cost_minor=100),
                  container("dear", 100, 100, 100, cost_minor=900)]
    result = pack(items, containers)
    assert result.containers[0].container.id == "cheap"


def test_an_eligible_container_is_chosen_over_a_cheaper_ineligible_one():
    items = [item("perishable", 50, 50, 50, eligible_container_tags=["refrigerated"])]
    containers = [container("dry-van", 100, 100, 100, cost_minor=100),
                  container("reefer", 100, 100, 100, cost_minor=900, tags=["refrigerated"])]
    result = pack(items, containers)

    assert result.complete
    assert result.containers[0].container.id == "reefer"
    assert_sound(result, items, containers)


def test_an_item_with_no_eligible_container_is_reported_with_a_reason():
    items = [item("perishable", 50, 50, 50, eligible_container_tags=["refrigerated"])]
    containers = [container("dry-van", 100, 100, 100)]
    result = pack(items, containers)

    assert not result.complete
    assert [u.reason for u in result.unpacked] == ["no_eligible_container"]
    assert result.unpacked[0].proof.level == "proven"


def test_a_void_fill_reserve_reduces_usable_volume_without_touching_geometry():
    # 8 cubes of 50mm exactly fill a 100mm container (1,000,000 mm^3 / 125,000 mm^3
    # each). Reserving half the volume for packing material leaves room for exactly 4.
    items = [item("cube", 50, 50, 50, quantity=8)]
    unreserved = pack(items, [container("box", 100, 100, 100, quantity=1)])
    reserved_containers = [container("box", 100, 100, 100, quantity=1, void_fill_reserve_ratio=0.5)]
    reserved = pack(items, reserved_containers)

    assert unreserved.complete
    assert unreserved.packed_item_count == 8
    assert not reserved.complete
    assert reserved.packed_item_count == 4
    assert reserved.containers[0].container.inner_dimensions.length.ticks == \
        unreserved.containers[0].container.inner_dimensions.length.ticks
    assert_sound(reserved, items, reserved_containers)


def test_the_void_fill_reserve_is_reported_in_the_result():
    items = [item("cube", 50, 50, 50, quantity=8)]
    containers = [container("box", 100, 100, 100, void_fill_reserve_ratio=0.25)]
    result = pack(items, containers)
    assert result.to_dict()["containers"][0]["void_fill_reserve_ticks3"] == str(1_600_000**3 // 4)


def test_a_tag_limit_leaves_the_rest_unpacked_rather_than_opening_another_container():
    items = [item("drum", 30, 30, 30, quantity=3, tags=["hazmat"])]
    containers = [container("box", 100, 100, 100, quantity=1, tag_limits={"hazmat": 2})]
    result = pack(items, containers)

    assert not result.complete
    assert result.packed_item_count == 2
    assert len(result.unpacked) == 1
    assert_sound(result, items, containers)


def test_a_tag_limit_does_not_constrain_an_unrelated_item():
    items = [item("drum", 30, 30, 30, quantity=2, tags=["hazmat"]), item("box", 30, 30, 30)]
    containers = [container("c", 100, 100, 100, quantity=1, tag_limits={"hazmat": 2})]
    result = pack(items, containers)

    assert result.complete
    assert result.packed_item_count == 3
    assert_sound(result, items, containers)


def test_container_stock_is_never_exceeded():
    items = [item("a", 90, 90, 90, quantity=4)]
    containers = [container("c", 100, 100, 100, quantity=2)]
    result = pack(items, containers)

    assert len(result.containers) <= 2
    assert len(result.unpacked) == 2
    assert_sound(result, items, containers)


def test_the_container_budget_caps_the_answer():
    items = [item("a", 90, 90, 90, quantity=4)]
    containers = [container("c", 100, 100, 100, quantity=10)]
    result = pack(items, containers, PackingConfig.balanced(max_containers=2))

    assert len(result.containers) == 2
    assert len(result.unpacked) == 2


def test_an_item_ceiling_is_respected():
    items = [item("a", 10, 10, 10, quantity=6)]
    containers = [container("c", 100, 100, 100, quantity=6, max_items=2)]
    result = pack(items, containers)

    assert all(len(c.placements) <= 2 for c in result.containers)
    assert_sound(result, items, containers)


def test_containers_that_hold_nothing_are_not_reported():
    items = [item("a", 10, 10, 10)]
    containers = [container("c", 100, 100, 100, quantity=5)]
    result = pack(items, containers)
    assert len(result.containers) == 1


# --------------------------------------------------------------------- physics

def test_clearance_keeps_a_gap_between_neighbours():
    """The envelope, not the item, is what may not overlap; the reported position sits
    inside its envelope by exactly the clearance."""
    gap = Length.mm(2)
    items = [item("a", 40, 40, 40, quantity=4)]
    containers = [container("c", 200, 200, 200)]
    result = pack(items, containers, PackingConfig.balanced(clearance=gap))

    assert_sound(result, items, containers, clearance=gap)
    for placement in result.containers[0].placements:
        assert placement.position.x - placement.envelope_origin.x == gap.ticks
        assert placement.envelope_dimensions == placement.dimensions.expand(gap)


def test_an_obstacle_is_worked_around():
    post = Obstacle("post", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(50, 50, 100)))
    items = [item("a", 40, 40, 40, quantity=2)]
    containers = [container("box", 100, 100, 100, obstacles=(post,))]
    result = pack(items, containers)

    assert result.complete
    assert all(not p.envelope_box.intersects(post.box) for p in result.containers[0].placements)
    assert_sound(result, items, containers)


def test_a_shelf_along_one_wall_still_leaves_a_usable_container():
    shelf = Obstacle("shelf", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(20, 100, 100)))
    items = [item("a", 40, 40, 40, quantity=2)]
    containers = [container("box", 100, 100, 100, obstacles=(shelf,))]
    result = pack(items, containers, PackingConfig.quality(time_limit_ms=2_000))

    assert result.complete
    assert_sound(result, items, containers)


def test_floor_only_items_never_leave_the_floor():
    items = [item("f", 40, 40, 40, quantity=8, must_be_on_floor=True)]
    containers = [container("c", 100, 100, 100, quantity=4)]
    result = pack(items, containers)

    assert all(p.envelope_origin.z == 0 for c in result.containers for p in c.placements)
    assert_sound(result, items, containers)


def test_nothing_is_stacked_on_a_non_stackable_item():
    items = [item("n", 40, 40, 40, quantity=8, stackable=False)]
    containers = [container("c", 100, 100, 100, quantity=4)]
    result = pack(items, containers)

    assert all(p.envelope_origin.z == 0 for c in result.containers for p in c.placements)
    assert_sound(result, items, containers)


def test_a_non_stackable_candidate_is_not_slid_under_an_existing_item():
    """The fast load shortcut must include the candidate's own stackable flag.

    The shelf forces the wide item above the floor and exposes a tempting empty point
    underneath it. Accepting the non-stackable item there creates an invalid result.
    """
    shelf = Obstacle("shelf", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(50, 100, 20)))
    items = [
        item("a-upper", 100, 100, 10, allowed_rotations=(Rotation.LWH,)),
        item("b-under", 50, 100, 20, stackable=False, allowed_rotations=(Rotation.LWH,)),
    ]
    containers = [container("box", 100, 100, 100, obstacles=(shelf,))]
    result = pack(items, containers, PackingConfig.fast(time_limit_ms=1_000))

    assert result.complete
    assert result.status is not PackingStatus.INVALID_RESULT
    assert_sound(result, items, containers)


def test_a_bearing_limit_is_honoured_across_the_whole_stack():
    """The base carries everything above it, not just its immediate neighbour."""
    items = [item("base", 100, 100, 50, weight="1 kg", max_top_load="2.5 kg"),
             item("light", 100, 100, 50, quantity=4, weight="1 kg")]
    containers = [container("c", 100, 100, 400, quantity=3)]
    result = pack(items, containers)

    assert_sound(result, items, containers)
    for packed in result.containers:
        for placement in packed.placements:
            limit = placement.instance.item.max_top_load
            if limit is not None:
                assert placement.top_load <= limit


@pytest.mark.parametrize(
    "profile",
    [PackingConfig.fast, PackingConfig.balanced],
    ids=["fast", "balanced"],
)
def test_lattice_profiles_cap_cumulative_top_load_before_short_circuiting(profile):
    items = [
        item(
            "crate", 100, 100, 50,
            quantity=4,
            weight="1 kg",
            max_top_load="1.5 kg",
            allowed_rotations=(Rotation.LWH,),
        )
    ]
    containers = [container("c", 100, 100, 200, quantity=2)]

    result = pack(items, containers, profile(time_limit_ms=5_000))

    assert result.complete
    assert result.status is PackingStatus.FEASIBLE
    assert [len(packed.placements) for packed in result.containers] == [2, 2]
    assert_sound(result, items, containers)


def test_the_reported_top_load_is_the_cumulative_one():
    items = [item("base", 100, 100, 50, weight="1 kg"), item("upper", 100, 100, 50, quantity=2, weight="1 kg")]
    containers = [container("c", 100, 100, 150)]
    result = pack(items, containers, PackingConfig.balanced(minimum_support_ratio=1.0))

    assert result.complete
    bottom = min(result.containers[0].placements, key=lambda p: p.envelope_origin.z)
    assert bottom.top_load == Weight.of(2, "kg")


def test_a_materialized_nested_grid_reports_its_cumulative_top_loads():
    items = [item(
        "crate", 100, 100, 50,
        quantity=3,
        weight="1 kg",
        max_top_load="2 kg",
        minimum_support_ratio=1.0,
        ground_contact_rule="single",
        nesting_height=Length.mm(25),
        allowed_rotations=(Rotation.LWH,),
    )]
    containers = [container("c", 100, 100, 100)]

    result = pack(
        items,
        containers,
        PackingConfig.fast(time_limit_ms=1_000, require_placement_coordinates=True),
    )

    assert result.complete
    ordered = sorted(result.containers[0].placements, key=lambda placement: placement.envelope_origin.z)
    assert [placement.envelope_origin.z for placement in ordered] == [
        0, Length.mm(25).ticks, Length.mm(50).ticks,
    ]
    assert [placement.top_load for placement in ordered] == [
        Weight.of(2, "kg"), Weight.of(1, "kg"), Weight(0),
    ]
    assert [placement.support_ratio for placement in ordered] == [1.0, 1.0, 1.0]
    assert_sound(result, items, containers)


def test_different_declared_nesting_types_do_not_share_a_lattice_column():
    nesting = {
        "nesting_height": Length.mm(40),
        "allowed_rotations": (Rotation.LWH,),
    }
    items = [item("a", 100, 100, 100, **nesting),
             item("b", 100, 100, 100, **nesting)]
    containers = [container("c", 100, 100, 160, quantity=1)]

    result = pack(items, containers, PackingConfig.fast(time_limit_ms=1_000))

    assert not result.complete
    assert result.status is not PackingStatus.INVALID_RESULT
    assert result.packed_item_count == 1
    assert len(result.unpacked) == 1
    assert_sound(result, items, containers)


def test_extreme_points_never_nests_onto_a_non_stackable_item():
    items = [item(
        "crate", 100, 100, 50, quantity=2,
        nesting_height=Length.mm(25), stackable=False,
        allowed_rotations=(Rotation.LWH,),
    )]
    containers = [container("c", 100, 100, 75, quantity=1)]
    config = PackingConfig.fast(time_limit_ms=1_000, solvers=("extreme_points",))

    result = pack(items, containers, config)

    assert result.status is not PackingStatus.INVALID_RESULT
    assert result.packed_item_count == 1
    assert len(result.unpacked) == 1
    assert_sound(result, items, containers)


def test_a_required_support_ratio_is_met_by_every_stacked_item():
    items = [item("a", 60, 60, 20, quantity=6, minimum_support_ratio=0.75)]
    containers = [container("c", 121, 121, 100, quantity=3)]
    result = pack(items, containers, PackingConfig.quality(time_limit_ms=2_000))

    assert_sound(result, items, containers)
    assert all(p.support_ratio >= 0.75 - 1e-9
               for c in result.containers for p in c.placements if p.envelope_origin.z > 0)


# ---------------------------------------------------------------------- groups

def test_a_group_travels_in_one_container():
    items = [item("kit", 60, 60, 60, quantity=2, group="kit"), item("loose", 20, 20, 20, quantity=4)]
    containers = [container("c", 130, 130, 130, quantity=3)]
    result = pack(items, containers)

    assert result.complete
    homes = {c.id for c in result.containers for p in c.placements if p.instance.item.group == "kit"}
    assert len(homes) == 1
    assert_sound(result, items, containers)


def test_an_impossible_group_does_not_strand_unrelated_items():
    """The regression this batching exists for: a group that cannot fit must be rejected
    as a whole and leave the rest of the order untouched."""
    items = [item("kit", 80, 80, 80, quantity=2, group="kit"), item("loose", 20, 20, 20, quantity=4)]
    containers = [container("c", 100, 100, 100)]
    result = pack(items, containers)

    assert result.packed_item_count == 4
    assert {u.instance.item.id for u in result.unpacked} == {"kit"}
    assert all(u.reason == "group_cannot_fit_together" for u in result.unpacked)
    assert_sound(result, items, containers)


# ------------------------------------------------------------------ separation

def test_incompatible_items_are_separated():
    items = [item("food", 40, 40, 40, tags=["food"]), item("bleach", 40, 40, 40, incompatible_tags=["food"])]
    containers = [container("c", 100, 100, 100, quantity=2)]
    result = pack(items, containers)

    assert result.complete
    assert len(result.containers) == 2
    assert_sound(result, items, containers)


# ------------------------------------------------------------------- reporting

def test_exact_small_does_not_claim_global_optimality_without_a_certificate():
    """The discrete search stops at its first full packing and has no lower-bound
    certificate for the remaining objective keys."""
    items = [item("a", 100, 100, 100, quantity=2), item("b", 100, 100, 100, quantity=2)]
    containers = [container("box", 200, 200, 100)]
    result = pack(items, containers, PackingConfig.exact_small(time_limit_ms=5_000))

    assert result.complete
    assert result.status is PackingStatus.FEASIBLE


def test_a_single_item_type_takes_the_lattice_path_and_claims_no_proof():
    """The lattice cannot prove optimality, so it must not be reported as optimal even
    when it happens to fill the container perfectly."""
    items = [item("cube", 100, 100, 100, quantity=4)]
    containers = [container("box", 200, 200, 100)]
    result = pack(items, containers, PackingConfig.exact_small(time_limit_ms=5_000))

    assert result.complete
    assert result.status is PackingStatus.FEASIBLE


def test_a_complete_heuristic_answer_reports_feasible_not_optimal():
    """A heuristic that filled the order still has not proved it could not do better."""
    items = [item("cube", 50, 50, 50, quantity=16)]
    containers = [container("box", 200, 200, 200)]
    result = pack(items, containers, PackingConfig.balanced())
    assert result.status is PackingStatus.FEASIBLE


def test_an_incomplete_answer_within_the_budget_reports_best_found():
    items = [item("a", 90, 90, 90, quantity=3)]
    containers = [container("c", 100, 100, 100, quantity=1)]
    result = pack(items, containers, PackingConfig.balanced())
    assert result.status is PackingStatus.BEST_FOUND


def test_alternatives_are_capped_by_top_k():
    items = [item("a", 40, 30, 20, quantity=6)]
    containers = [container("c", 150, 150, 150, quantity=2)]
    result = pack(items, containers, PackingConfig.quality(time_limit_ms=2_000, top_k=3))
    assert len(result.alternatives) <= 2


def test_the_algorithm_report_names_the_solver_and_ordering_that_won():
    items = [item("a", 40, 30, 20, quantity=4)]
    containers = [container("c", 150, 150, 150)]
    report = pack(items, containers).algorithm

    assert ":" in report.solver, "the solver name carries the multi-start ordering"
    assert report.profile == "balanced"
    assert report.placements_attempted >= report.candidates_evaluated > 0
    assert report.metrics.orientations_considered == report.placements_attempted
    assert report.metrics.feasible_candidates == report.candidates_evaluated
    assert report.metrics.candidate_points_considered > 0
    assert report.metrics.search_nodes_expanded > 0


def test_a_tight_budget_still_returns_the_work_already_done():
    """Partial work is a valid packing and is worth far more than an empty result."""
    items = [item("m", 90, 80, 70, quantity=54)]
    containers = [container("c", 1_000, 1_000, 1_000, quantity=3)]
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=15))

    assert_sound(result, items, containers)
    assert result.status in (PackingStatus.TIME_LIMIT, PackingStatus.BEST_FOUND, PackingStatus.FEASIBLE)


# ---------------------------------------------- quantity compression

def test_quantity_compression_packs_10000_identical_items_within_budget():
    """The acceptance case: `require_placement_coordinates=False` must let 10,000
    identical items pack without paying for 10,000 `Placement` objects.

    The wall-clock ceiling below is deliberately generous (two orders of magnitude
    above what this actually takes on a modestly fast development machine, well
    under 100ms) -- it is a regression guard against the fast path silently falling
    back to the O(n) loop, not a tight performance assertion that could flake under
    CI scheduling noise. `validate_result=False` is paired with the flag so
    independent validation (which must still expand to check placements pairwise,
    see `Packer.pack`) is not itself the O(n) cost this test would otherwise be
    measuring instead of the solver's own fast path.
    """
    import time

    items = [item("crate", 100, 100, 100, quantity=10_000)]
    containers = [container("bin", 3_000, 3_000, 3_000)]
    # A generous time_limit_ms: the fast path itself never consults the deadline (it
    # is O(r), not a per-item loop with per-item checks), but the portfolio loop
    # checks the deadline before even calling the solver, and a short budget could
    # otherwise make this test flaky under a loaded CI machine rather than actually
    # exercising the fast path.
    config = PackingConfig.fast(time_limit_ms=5_000, require_placement_coordinates=False, validate_result=False)

    started = time.perf_counter()
    result = pack(items, containers, config)
    elapsed = time.perf_counter() - started

    assert result.status is PackingStatus.FEASIBLE
    assert not result.unpacked
    assert result.packed_item_count == 10_000
    assert len(result.containers) == 1
    packed = result.containers[0]
    assert packed.placements == ()
    assert packed.lattice_summary is not None
    assert packed.lattice_summary.count == 10_000
    assert elapsed < 5.0, f"quantity-compression fast path took {elapsed:.3f}s, expected well under 5s"

    # Fully reconstructible: expanding the summary must reproduce a physically sound,
    # non-overlapping layout identical to what the O(n) path would have built.
    expanded = packed.expand_placements()
    assert len(expanded) == 10_000
    boundary = AxisAlignedBox(Point(0, 0, 0), packed.container.inner_dimensions)
    assert all(boundary.contains(p.envelope_box) for p in expanded)
    boxes = [p.envelope_box for p in expanded]
    assert not any(a.intersects(b) for i, a in enumerate(boxes) for b in boxes[i + 1:i + 2])


def test_quantity_compression_default_behaviour_is_unchanged():
    """Leaving `require_placement_coordinates` unset must reproduce the exact
    per-item output the library has always returned -- this is a strict opt-in
    addition, not a behaviour change."""
    items = [item("crate", 90, 90, 90, quantity=40)]
    containers = [container("bin", 400, 400, 400)]

    default_result = pack(items, containers, PackingConfig.fast())
    explicit_result = pack(items, containers, PackingConfig.fast(require_placement_coordinates=True))

    assert default_result.containers[0].lattice_summary is None
    assert len(default_result.containers[0].placements) == 40
    assert_sound(default_result, items, containers)

    default_dict = default_result.to_dict(include_alternatives=False)
    explicit_dict = explicit_result.to_dict(include_alternatives=False)
    # `algorithm.duration_ms` is a live wall-clock reading of that particular call,
    # not part of the packing contract -- SERIALIZATION.md excludes all of
    # `algorithm` from the cross-implementation content comparison for the same
    # reason. Everything else, including every placement's coordinates, must match
    # exactly.
    default_dict["algorithm"]["duration_ms"] = explicit_dict["algorithm"]["duration_ms"] = 0
    assert default_dict == explicit_dict
    assert "lattice_summary" not in default_dict["containers"][0]


def test_quantity_compression_still_validates_end_to_end_when_requested():
    """`validate_result` stays the default `True`: the compact fast path must not
    silently bypass the independent validator. `Packer.pack` expands the summary
    just for this check (see its `validation_containers` comment) without
    materializing it into the returned, still-compact result."""
    items = [item("crate", 100, 100, 100, quantity=200)]
    containers = [container("bin", 1_000, 1_000, 1_000)]
    config = PackingConfig.fast(require_placement_coordinates=False)
    assert config.validate_result is True

    result = pack(items, containers, config)

    assert result.status is PackingStatus.FEASIBLE
    assert result.warnings == ()
    assert result.containers[0].placements == ()
    assert result.containers[0].lattice_summary is not None
    assert result.containers[0].lattice_summary.count == 200

# --------------------------------------------------------------- budget monotonicity

class _FakeClock:
    """Monotonic fake that advances a fixed step on every read.

    A real wall clock would make this test flaky (whether a beam search finishes
    inside a slice depends on host speed and load); a deterministic clock makes the
    exact point of truncation reproducible, the same technique test_solvers.py's
    `CheckBudgetClock` already uses.
    """

    def __init__(self, step_ns: int = 1_000_000):
        self.reads = 0
        self.step = step_ns

    def __call__(self) -> int:
        value = self.reads * self.step
        self.reads += 1
        return value


def _budget_monotonicity_scene(variant: int = 0):
    """A mixed-type scene, packed with a portfolio that names "grid" explicitly.

    `GridSolver` only ever enters the default portfolio when every requested item
    shares one id (`GridSolver.supports`) -- but nothing stops a caller from naming
    "grid" directly through `PackingConfig.solvers` for a mixed-type request. Every
    container this scene offers disqualifies the lattice outright (mixed types), so
    `GridSolver.pack_one` delegates entirely to `ExtremePointSolver` while still
    reporting the start name "grid:volume".
    """
    items = [
        item("a", 30 + variant, 30, 20, quantity=8 + variant),
        item("b", 20, 20 + variant, 20, quantity=9 + variant),
        item("c", 15, 15, 15 + variant, quantity=10),
    ]
    containers = [container("box", 120, 120, 100)]
    config_kwargs = dict(solvers=("grid", "layer", "maximal_spaces", "extreme_points"))
    return items, containers, config_kwargs


@pytest.mark.parametrize("clock_step_ns", [1_000_000, 2_000_000])
@pytest.mark.parametrize("objective", ["default", "lowest_cost", "shipping_cost"])
@pytest.mark.parametrize("scene_variant", range(4))
def test_raising_the_time_limit_never_lowers_the_chosen_ranks(clock_step_ns, objective, scene_variant):
    """A larger time budget must never choose a worse-ranked solution.

    A longer budget lets `GridSolver`'s delegated (non-lattice) start finish before
    slower, genuinely better-arranged starts get a turn, and the portfolio's
    "a complete grid start beats everything else" short-circuit used to take
    that at face value regardless of whether the lattice was actually used --
    discarding a strictly better already-available answer purely because more time
    let the delegate finish. Same request, same seed, three budgets 250ms/5s/30s:
    the chosen solution's rank (`PackingResult.score`, lower is better) must be
    non-increasing as the budget grows.
    """
    items, containers, config_kwargs = _budget_monotonicity_scene(scene_variant)
    previous_score = None
    for time_limit_ms in (250, 5_000, 30_000):
        config = PackingConfig.balanced(
            time_limit_ms=time_limit_ms,
            objective=objective,
            dimensional_weight_divisor=139 if objective == "shipping_cost" else None,
            **config_kwargs,
        )
        result = pack(items, containers, config, clock=_FakeClock(clock_step_ns))
        assert_sound(result, items, containers)
        if previous_score is not None:
            assert result.score <= previous_score, (
                f"budget {time_limit_ms}ms ranked worse ({result.score}) than a "
                f"smaller budget ({previous_score})"
            )
        previous_score = result.score


def test_delegated_grid_does_not_short_circuit_the_concurrent_portfolio():
    """The real-clock concurrent branch must apply the same lattice proof gate."""
    items = [item("a", 30, 30, 20), item("b", 20, 20, 20), item("c", 15, 15, 15)]
    containers = [container("box", 120, 120, 100)]
    result = pack(
        items,
        containers,
        PackingConfig.balanced(
            time_limit_ms=5_000,
            solvers=("grid", "layer", "extreme_points"),
            parallel_starts=2,
        ),
    )
    starts = result.termination.attributes["starts"]
    assert sum(1 for start in starts if start["started"]) > 1
