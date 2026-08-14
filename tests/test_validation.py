"""The independent validator, tested by handing it deliberately broken solutions.

It shares no state with the solvers and re-derives every guarantee from the placements
alone, which is only worth anything if it actually catches a fault. So each case here
injects one specific corruption and asserts the matching code comes back. The codes
themselves are part of the cross-language contract — see docs/VALIDATION-CONTRACT.md.
"""

from __future__ import annotations

import random

import pytest

from packvium import (AxisAlignedBox, Axle, Container, Dimensions, IndependentSolutionValidator, Item, Length,
                         Obstacle, PackedContainer, Placement, Point, Rotation, UnpackedItem, Weight)
from packvium.nesting import is_valid_nesting as _is_valid_nesting
from support import issues_for

MM = 16_000


def _brute_force_collision_pairs(placements) -> list[tuple[int, int]]:
    """Independent, deliberately naive all-pairs check sharing no code with
    `_collision_pairs`'s z-bucketed sweep -- silently dropping or inventing a pair is
    exactly the risk that sweep carries and a shared implementation could not catch."""
    pairs = set()
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            if placements[i].envelope_box.intersects(placements[j].envelope_box) and not _is_valid_nesting(placements[i], placements[j]):
                pairs.add((i, j))
    return sorted(pairs)


def item(id: str, length=100, width=100, height=100, **kwargs) -> Item:
    return Item.create(id, Dimensions.mm(length, width, height), kwargs.pop("weight", 0), **kwargs)


def place(instance, x=0, y=0, z=0, rotation=Rotation.LWH, dimensions=None, clearance=0) -> Placement:
    dims = dimensions or instance.item.dimensions.rotated(rotation)
    envelope = dims.expand(Length(clearance)) if clearance else dims
    return Placement(instance, Point(x + clearance, y + clearance, z + clearance), rotation, dims,
                     Point(x, y, z), envelope)


def shelf(length=200, width=100, height=100, **kwargs) -> Container:
    return Container.create("shelf", Dimensions.mm(length, width, height), **kwargs)


# ------------------------------------------------------------------- clean input

def test_a_sound_solution_raises_nothing():
    cubes = item("cube", quantity=2)
    box = shelf()
    first, second = cubes.instances()
    packed = PackedContainer(box, 1, (place(first), place(second, x=100 * MM)))
    assert issues_for([cubes], [box], [packed]) == []


def test_an_empty_solution_raises_nothing():
    assert issues_for([item("a")], [shelf()], []) == []


# ------------------------------------------------------------------ geometry

def test_overlapping_placements_are_caught():
    cubes = item("cube", quantity=2)
    box = shelf()
    first, second = cubes.instances()
    codes = issues_for([cubes], [box], [PackedContainer(box, 1, (place(first), place(second, x=50 * MM)))])
    assert "collision" in codes


def test_a_nested_pair_of_identical_items_is_not_a_collision():
    crate = item("crate", 100, 100, 100, quantity=2, nesting_height=Length.mm(40))
    box = shelf(100, 100, 300)
    lower, upper = crate.instances()
    packed = PackedContainer(box, 1, (place(lower), place(upper, z=60 * MM)))
    assert issues_for([crate], [box], [packed]) == []


def test_nested_layers_are_full_support_to_the_independent_validator():
    crate = item(
        "crate", 100, 100, 100, quantity=3,
        nesting_height=Length.mm(40), minimum_support_ratio=1.0,
        ground_contact_rule="single",
    )
    box = shelf(100, 100, 300)
    bottom, middle, top = crate.instances()
    packed = PackedContainer(box, 1, (
        place(bottom), place(middle, z=60 * MM), place(top, z=120 * MM),
    ))

    assert issues_for([crate], [box], [packed], minimum_support_ratio=1.0) == []


def test_independent_validator_rejects_nesting_on_a_non_stackable_item():
    crate = item(
        "crate", 100, 100, 50, quantity=2,
        nesting_height=Length.mm(25), stackable=False,
    )
    box = shelf(100, 100, 100)
    lower, upper = crate.instances()
    packed = PackedContainer(box, 1, (
        place(lower), place(upper, z=25 * MM),
    ))

    assert "non_stackable" in issues_for([crate], [box], [packed])


def test_an_overlap_deeper_than_the_declared_nesting_is_still_a_collision():
    crate = item("crate", 100, 100, 100, quantity=2, nesting_height=Length.mm(40))
    box = shelf(100, 100, 300)
    lower, upper = crate.instances()
    packed = PackedContainer(box, 1, (place(lower), place(upper, z=50 * MM)))  # 50mm overlap, not the declared 40mm
    assert "collision" in issues_for([crate], [box], [packed])


def test_two_different_item_types_never_get_a_nesting_exemption():
    a = item("a", 100, 100, 100, nesting_height=Length.mm(40))
    b = item("b", 100, 100, 100, nesting_height=Length.mm(40))
    box = shelf(100, 100, 300)
    a_instance, = a.instances()
    b_instance, = b.instances()
    packed = PackedContainer(box, 1, (place(a_instance), place(b_instance, z=60 * MM)))
    assert "collision" in issues_for([a, b], [box], [packed])


def test_an_offset_footprint_cannot_claim_a_nesting_exemption():
    crate = item("crate", 100, 100, 100, quantity=2, nesting_height=Length.mm(40))
    box = shelf(200, 100, 300)
    lower, upper = crate.instances()
    packed = PackedContainer(box, 1, (place(lower), place(upper, x=50 * MM, z=60 * MM)))
    assert "collision" in issues_for([crate], [box], [packed])


@pytest.mark.parametrize("seed", range(10))
def test_collision_pairs_agrees_with_brute_force_on_random_scenes(seed):
    """Random placements with varied z-heights, so the z-bucketing's cell size (the
    max envelope-box height in the scene) varies from run to run."""
    rng = random.Random(2000 + seed)
    cube = item("cube", quantity=15)
    placements = []
    for instance in cube.instances():
        length, width, height = (rng.randrange(5, 60, 5) for _ in range(3))
        x, y, z = (rng.randrange(0, 200, 5) for _ in range(3))
        dims = Dimensions(Length(length), Length(width), Length(height))
        placements.append(place(instance, x=x, y=y, z=z, dimensions=dims))
    assert IndependentSolutionValidator._collision_pairs(placements) == _brute_force_collision_pairs(placements)


def test_collision_pairs_agrees_with_brute_force_on_a_dense_multi_layer_lattice():
    """`_collision_pairs`'s x-sweep only pruned its active set by x-overlap, so a
    lattice with many z-levels sharing similar x ranges kept that active set close to
    O(n) -- this rebuilds that shape directly: several z-levels, each a
    dense non-overlapping grid sharing the same x/y footprint (so every level's grid
    shares x ranges with every other level's), plus a handful of random extra
    placements that can collide with the grid or each other."""
    cube = item("cube", 10, 10, 10, quantity=700)
    instances = iter(cube.instances())
    placements = []
    for level in range(6):
        z = level * 10 * MM
        for gx in range(10):
            for gy in range(10):
                placements.append(place(next(instances), x=gx * 10 * MM, y=gy * 10 * MM, z=z))
    rng = random.Random(42)
    for _ in range(40):
        x, y, z = (rng.randrange(0, 100, 5) * MM for _ in range(3))
        placements.append(place(next(instances), x=x, y=y, z=z))
    assert IndependentSolutionValidator._collision_pairs(placements) == _brute_force_collision_pairs(placements)


def test_a_placement_reaching_past_a_wall_is_caught():
    cube = item("cube")
    box = shelf()
    instance, = cube.instances()
    codes = issues_for([cube], [box], [PackedContainer(box, 1, (place(instance, x=150 * MM),))])
    assert "outside_container" in codes


def test_a_placement_inside_an_obstacle_is_caught():
    post = Obstacle("post", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(50, 50, 100)))
    box = shelf(obstacles=(post,))
    cube = item("cube", 40, 40, 40)
    instance, = cube.instances()
    codes = issues_for([cube], [box], [PackedContainer(box, 1, (place(instance),))])
    assert "obstacle_collision" in codes


def test_a_placement_inside_the_second_box_of_a_union_obstacle_is_caught():
    """The union's second box must be checked, not only its first."""
    near = AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(20, 100, 100))
    far = AxisAlignedBox(Point(80 * MM, 0, 0), Dimensions.mm(20, 100, 100))
    arch = Obstacle("arch", near, additional_boxes=(far,))
    box = shelf(100, 100, 100, obstacles=(arch,))
    cube = item("cube", 20, 20, 20)
    instance, = cube.instances()
    codes = issues_for([cube], [box], [PackedContainer(box, 1, (place(instance, x=85 * MM),))])
    assert "obstacle_collision" in codes


# ------------------------------------------------------------------ bookkeeping

def test_the_same_instance_reported_twice_is_caught():
    cube = item("cube")
    box = shelf()
    instance, = cube.instances()
    codes = issues_for([cube], [box], [PackedContainer(box, 1, (place(instance), place(instance, x=100 * MM)))])
    assert "duplicate_item" in codes


def test_an_item_that_was_never_requested_is_caught():
    requested, smuggled = item("requested"), item("smuggled")
    box = shelf()
    instance, = smuggled.instances()
    codes = issues_for([requested], [box], [PackedContainer(box, 1, (place(instance),))])
    assert "unknown_item" in codes


def test_a_full_solution_that_loses_an_item_is_caught():
    cube = item("cube")
    box = shelf()
    assert issues_for([cube], [box], [], unpacked=()) == ["missing_item"]


def test_an_instance_cannot_be_both_packed_and_unpacked():
    cube = item("cube")
    box = shelf()
    instance, = cube.instances()
    packed = PackedContainer(box, 1, (place(instance),))
    codes = issues_for(
        [cube], [box], [packed], unpacked=(UnpackedItem(instance, "no_feasible_placement"),)
    )
    assert "duplicate_item" in codes


def test_every_unpacked_item_requires_a_reason():
    cube = item("cube")
    box = shelf()
    instance, = cube.instances()
    assert "missing_reason" in issues_for(
        [cube], [box], [], unpacked=(UnpackedItem(instance, ""),)
    )


def test_a_group_cannot_be_partly_packed_and_partly_unpacked():
    kit = item("kit", quantity=2, group="kit")
    box = shelf()
    first, second = kit.instances()
    packed = PackedContainer(box, 1, (place(first),))
    codes = issues_for(
        [kit], [box], [packed], unpacked=(UnpackedItem(second, "group_cannot_fit_together"),)
    )
    assert "group_partial" in codes


def test_using_more_containers_than_exist_is_caught():
    cubes = item("cube", quantity=2)
    box = shelf(quantity=1)
    first, second = cubes.instances()
    codes = issues_for([cubes], [box],
                       [PackedContainer(box, 1, (place(first),)), PackedContainer(box, 2, (place(second),))])
    assert "container_inventory_exceeded" in codes


def test_exceeding_the_item_ceiling_is_caught():
    cubes = item("cube", quantity=2)
    box = shelf(max_items=1)
    first, second = cubes.instances()
    codes = issues_for([cubes], [box], [PackedContainer(box, 1, (place(first), place(second, x=100 * MM)))])
    assert "max_items_exceeded" in codes


def test_exceeding_the_payload_ceiling_is_caught():
    heavy = item("heavy", quantity=2, weight="1 kg")
    box = shelf(max_payload="1.5 kg")
    first, second = heavy.instances()
    codes = issues_for([heavy], [box], [PackedContainer(box, 1, (place(first), place(second, x=100 * MM)))])
    assert "payload_exceeded" in codes


def test_a_group_spread_across_containers_is_caught():
    """All-or-nothing is the whole meaning of a group; half of one in each of two
    cartons is a worse answer than leaving both out."""
    kit = item("kit", quantity=2, group="kit")
    box = shelf(quantity=2)
    first, second = kit.instances()
    codes = issues_for([kit], [box],
                       [PackedContainer(box, 1, (place(first),)), PackedContainer(box, 2, (place(second),))])
    assert "group_split" in codes


# ----------------------------------------------------------- reported geometry

def test_a_rotation_the_item_forbids_is_caught():
    upright = item("upright", 100, 50, 50, allowed_rotations=(Rotation.LWH,))
    box = shelf()
    instance, = upright.instances()
    placement = place(instance, rotation=Rotation.WLH)
    codes = issues_for([upright], [box], [PackedContainer(box, 1, (placement,))])
    assert "forbidden_rotation" in codes


def test_dimensions_that_do_not_match_the_reported_rotation_are_caught():
    """A solver could otherwise claim a box occupies a smaller footprint than it does
    and every collision check downstream would agree with the lie."""
    brick = item("brick", 100, 50, 50)
    box = shelf()
    instance, = brick.instances()
    lying = place(instance, rotation=Rotation.LWH, dimensions=brick.dimensions.rotated(Rotation.WLH))
    codes = issues_for([brick], [box], [PackedContainer(box, 1, (lying,))])
    assert "dimension_mismatch" in codes


def test_an_envelope_that_ignores_the_configured_clearance_is_caught():
    cube = item("cube", 50, 50, 50)
    box = shelf()
    instance, = cube.instances()
    without_gap = PackedContainer(box, 1, (place(instance),))
    assert issues_for([cube], [box], [without_gap], clearance=Length.mm(2)) == ["clearance_mismatch"]
    with_gap = PackedContainer(box, 1, (place(instance, clearance=Length.mm(2).ticks),))
    assert issues_for([cube], [box], [with_gap], clearance=Length.mm(2)) == []


# --------------------------------------------------------------------- physics

def test_a_floating_placement_is_caught_when_support_is_required():
    cube = item("cube", 50, 50, 50)
    box = shelf()
    instance, = cube.instances()
    floating = PackedContainer(box, 1, (place(instance, z=50 * MM),))
    assert issues_for([cube], [box], [floating], minimum_support_ratio=0.5) == ["insufficient_support"]


def test_a_tipping_placement_is_caught_even_though_the_area_ratio_is_met():
    """Area ratio alone is not stability: a 40% overlap on one side clears
    a 0.3 requirement while leaving the candidate's own centroid unsupported."""
    base = item("base", 40, 100, 10)
    top = item("top", 100, 100, 10, minimum_support_ratio=0.3)
    box = shelf()
    base_instance, = base.instances()
    top_instance, = top.instances()
    packed = PackedContainer(box, 1, (place(base_instance), place(top_instance, z=10 * MM)))
    assert "centre_of_gravity_unsupported" in issues_for([base, top], [box], [packed])


def test_an_axle_overload_is_caught():
    box = shelf(1000, 100, 100, axles=(Axle(Length.mm(100), Weight.of(399, "kg")), Axle(Length.mm(900), Weight.of(500, "kg"))))
    heavy = item("heavy", 1000, 100, 100, weight="800 kg")
    instance, = heavy.instances()
    packed = PackedContainer(box, 1, (place(instance),))
    assert "axle_overloaded" in issues_for([heavy], [box], [packed])


def test_a_floor_only_item_lifted_off_the_floor_is_caught():
    grounded = item("grounded", 50, 50, 50, must_be_on_floor=True)
    box = shelf()
    instance, = grounded.instances()
    codes = issues_for([grounded], [box], [PackedContainer(box, 1, (place(instance, z=50 * MM),))])
    assert "must_be_on_floor" in codes


def test_a_crushed_base_is_caught():
    base = item("base", 100, 100, 50, weight="1 kg", max_top_load="1 kg")
    load = item("load", 100, 100, 50, weight="5 kg")
    box = shelf()
    base_instance, = base.instances()
    load_instance, = load.instances()
    packed = PackedContainer(box, 1, (place(base_instance), place(load_instance, z=50 * MM)))
    assert "top_load_exceeded" in issues_for([base, load], [box], [packed])


def test_a_transitive_three_high_stack_limit_violation_is_caught():
    """`test_constraints.py`'s `test_a_three_high_column_counts_transitively_not_just_
    the_neighbour` already proves `stacked_counts`/`stack_limit_exceeded` count the
    full supported column, not just the direct neighbour, when driven directly; this
    proves the independent validator does too, reconstructing the column from real
    `PackedContainer` placements -- a naive direct-neighbour count would
    wrongly allow this (`base` only directly touches `middle`, count 1, meeting its
    own limit of 1), while the correct transitive count of 2 (`middle` and `top`, both
    resting somewhere above `base`) exceeds it.
    """
    base = item("base", 100, 100, 10, max_stacked_items=1)
    middle = item("middle", 100, 100, 10)
    top = item("top", 100, 100, 10)
    box = shelf(100, 100, 40)
    base_instance, = base.instances()
    middle_instance, = middle.instances()
    top_instance, = top.instances()
    packed = PackedContainer(box, 1, (
        place(base_instance), place(middle_instance, z=10 * MM), place(top_instance, z=20 * MM),
    ))
    assert "stacked_item_limit_exceeded" in issues_for([base, middle, top], [box], [packed])


def test_a_stack_limit_met_by_the_direct_neighbour_alone_is_not_caught():
    """The same geometry as above with `base`'s limit merely unset -- proves the
    previous test's rejection comes from the stack limit specifically, not from the
    column's geometry itself."""
    base = item("base", 100, 100, 10)
    middle = item("middle", 100, 100, 10)
    top = item("top", 100, 100, 10)
    box = shelf(100, 100, 40)
    base_instance, = base.instances()
    middle_instance, = middle.instances()
    top_instance, = top.instances()
    packed = PackedContainer(box, 1, (
        place(base_instance), place(middle_instance, z=10 * MM), place(top_instance, z=20 * MM),
    ))
    assert issues_for([base, middle, top], [box], [packed]) == []


def test_a_crushing_floor_load_is_caught_by_density_even_within_a_flat_top_load():
    """A flat `max_top_load` alone cannot express this: 550 kg is comfortably under a
    1000 kg absolute limit, but crushing once concentrated onto a 1 square metre
    footprint below the container's 500 kg/m^2 floor loading."""
    dense_shelf = shelf(1000, 1000, 200, max_stack_density="500 kg")
    base = item("base", 1000, 1000, 50, weight="400 kg", max_top_load="1000 kg")
    load = item("load", 1000, 1000, 50, weight="150 kg")
    base_instance, = base.instances()
    load_instance, = load.instances()
    packed = PackedContainer(dense_shelf, 1, (place(base_instance), place(load_instance, z=50 * MM)))
    assert "stack_density_exceeded" in issues_for([base, load], [dense_shelf], [packed])


def test_a_floor_load_within_the_density_limit_is_not_caught():
    dense_shelf = shelf(1000, 1000, 200, max_stack_density="500 kg")
    base = item("base", 1000, 1000, 50, weight="400 kg")
    load = item("load", 1000, 1000, 50, weight="100 kg")
    base_instance, = base.instances()
    load_instance, = load.instances()
    packed = PackedContainer(dense_shelf, 1, (place(base_instance), place(load_instance, z=50 * MM)))
    assert issues_for([base, load], [dense_shelf], [packed]) == []


def test_stacking_onto_a_non_stackable_item_is_caught():
    base = item("base", 100, 100, 50, stackable=False)
    load = item("load", 100, 100, 50)
    box = shelf()
    base_instance, = base.instances()
    load_instance, = load.instances()
    packed = PackedContainer(box, 1, (place(base_instance), place(load_instance, z=50 * MM)))
    assert "non_stackable" in issues_for([base, load], [box], [packed])


def test_the_stacking_rule_is_checked_against_later_placements_too():
    """Support and stacking are geometric facts. Judging a placement only against the
    ones reported before it let a solver hide a violation by ordering its output."""
    base = item("base", 100, 100, 50, stackable=False)
    load = item("load", 100, 100, 50)
    box = shelf()
    base_instance, = base.instances()
    load_instance, = load.instances()
    reversed_order = PackedContainer(box, 1, (place(load_instance, z=50 * MM), place(base_instance)))
    assert "non_stackable" in issues_for([base, load], [box], [reversed_order])


# --------------------------------------------------------- ground contact

def test_a_ratio_that_would_pass_is_still_caught_by_the_corner_rule():
    """A ratio cannot express the corner rule: 64% coverage concentrated in the
    middle clears a 50% requirement but touches none of the four base corners.
    `test_constraints.py`'s `test_a_ratio_the_corner_rule_rejects_that_a_ratio_accepts`
    already proves `SupportConstraint` itself catches this when driven directly; this
    proves the independent validator does too, reconstructing supporters from real
    `PackedContainer` placements rather than a hand-built `ConstraintContext` (the constraint layer's
    own "reconstruct supporters independently from placements")."""
    plate = item("plate", 80, 80, 10)
    lid = item("lid", 100, 100, 10, minimum_support_ratio=0.5, ground_contact_rule="covered")
    box = shelf(100, 100, 30)
    plate_instance, = plate.instances()
    lid_instance, = lid.instances()
    packed = PackedContainer(box, 1, (place(plate_instance, x=10 * MM, y=10 * MM), place(lid_instance, z=10 * MM)))
    assert "ground_contact_violation" in issues_for([plate, lid], [box], [packed])


def test_the_same_placement_raises_nothing_without_the_corner_rule():
    """The exact same geometry as above, `ground_contact_rule` merely unset -- proves
    the previous test's rejection comes from the corner rule specifically, not from
    the ratio, an unrelated bookkeeping check, or the plate/lid geometry itself."""
    plate = item("plate", 80, 80, 10)
    lid = item("lid", 100, 100, 10, minimum_support_ratio=0.5)
    box = shelf(100, 100, 30)
    plate_instance, = plate.instances()
    lid_instance, = lid.instances()
    packed = PackedContainer(box, 1, (place(plate_instance, x=10 * MM, y=10 * MM), place(lid_instance, z=10 * MM)))
    assert issues_for([plate, lid], [box], [packed]) == []


def test_single_and_multiple_rules_are_also_caught_through_the_validator():
    """`covered` above exercises the corner check specifically; `single` and
    `multiple` share the same dispatch path in `IndependentSolutionValidator` and are
    checked here too so the whole rule vocabulary is proven through the validator,
    not just one member of it."""
    left = item("left", 50, 100, 10)
    right = item("right", 50, 100, 10)
    box = shelf(100, 100, 30)
    left_instance, = left.instances()
    right_instance, = right.instances()

    split_candidate = item("split", 100, 100, 10, ground_contact_rule="single")
    split_instance, = split_candidate.instances()
    split_packed = PackedContainer(box, 1, (
        place(left_instance, x=0), place(right_instance, x=800_000), place(split_instance, z=10 * MM),
    ))
    assert "ground_contact_violation" in issues_for([left, right, split_candidate], [box], [split_packed])

    single_support = item("base", 100, 100, 10)
    multiple_candidate = item("bridge", 100, 100, 10, ground_contact_rule="multiple")
    single_support_instance, = single_support.instances()
    multiple_instance, = multiple_candidate.instances()
    single_packed = PackedContainer(box, 1, (
        place(single_support_instance), place(multiple_instance, z=10 * MM),
    ))
    assert "ground_contact_violation" in issues_for([single_support, multiple_candidate], [box], [single_packed])


def test_free_and_floor_placements_raise_nothing_through_the_validator():
    """`free` never checks contact at all, and a floor placement (z=0) satisfies every
    rule by definition -- both need to stay true through the validator's own
    dispatch, not only in `SupportConstraint`'s own unit tests."""
    base = item("base", 100, 100, 10)
    box = shelf(100, 100, 30)
    for rule in ("free", "covered", "single", "multiple"):
        floor_item = item(f"floor-{rule}", 100, 100, 10, ground_contact_rule=rule)
        floor_instance, = floor_item.instances()
        packed = PackedContainer(box, 1, (place(floor_instance),))
        assert issues_for([floor_item], [box], [packed]) == [], rule

    airborne = item("airborne", 50, 50, 10, ground_contact_rule="free")
    airborne_instance, = airborne.instances()
    base_instance, = base.instances()
    packed = PackedContainer(box, 1, (place(base_instance, x=0, y=0), place(airborne_instance, x=0, y=0, z=10 * MM)))
    assert issues_for([base, airborne], [box], [packed]) == []


# ----------------------------------------------------- container eligibility

def test_a_placement_in_an_ineligible_container_is_caught():
    """`test_constraints.py`'s `test_an_ineligible_container_is_refused` already
    proves `ContainerEligibilityConstraint` itself rejects this when driven directly
    (and the solver's own `default_constraints` includes it, so an eligible-tag
    mismatch is refused before a placement is ever attempted, not merely reported
    after the fact); this proves the independent validator catches the same
    violation reconstructed from real `PackedContainer` placements, in case
    a solver bug or a hand-built scene ever produced one anyway.
    """
    perishable = item("perishable", 50, 50, 50, eligible_container_tags=["refrigerated"])
    ordinary = shelf(tags=[])
    perishable_instance, = perishable.instances()
    packed = PackedContainer(ordinary, 1, (place(perishable_instance),))
    assert "container_ineligible" in issues_for([perishable], [ordinary], [packed])


def test_a_placement_in_an_eligible_container_is_not_caught():
    perishable = item("perishable", 50, 50, 50, eligible_container_tags=["refrigerated"])
    refrigerated = shelf(tags=["refrigerated"])
    perishable_instance, = perishable.instances()
    packed = PackedContainer(refrigerated, 1, (place(perishable_instance),))
    assert issues_for([perishable], [refrigerated], [packed]) == []


def test_an_item_with_no_eligibility_tags_may_go_in_any_container():
    ordinary_item = item("box", 50, 50, 50)
    box = shelf(tags=[])
    instance, = ordinary_item.instances()
    packed = PackedContainer(box, 1, (place(instance),))
    assert issues_for([ordinary_item], [box], [packed]) == []


# --------------------------------------------------------------- tag counts

def test_a_tag_limit_exceeded_by_a_third_item_is_caught():
    """`test_constraints.py`'s `test_a_third_item_of_a_limited_tag_is_refused` already
    proves `TagCountConstraint` itself rejects this when driven directly; this proves
    the independent validator catches the same violation reconstructed from real
    `PackedContainer` placements, not a hand-built `ConstraintContext`. Every one of
    the three placements is checked against the other two (not just the ones placed
    before it, the same "later placements too" discipline the stacking rule already
    gets), so all three are reported, not merely whichever the search happened to add
    last."""
    a = item("a", 50, 50, 50, tags=["hazmat"])
    b = item("b", 50, 50, 50, tags=["hazmat"])
    c = item("c", 50, 50, 50, tags=["hazmat"])
    box = shelf(200, 100, 100, tag_limits={"hazmat": 2})
    a_instance, = a.instances()
    b_instance, = b.instances()
    c_instance, = c.instances()
    packed = PackedContainer(box, 1, (
        place(a_instance, x=0), place(b_instance, x=50 * MM), place(c_instance, x=100 * MM),
    ))
    codes = issues_for([a, b, c], [box], [packed])
    assert codes.count("tag_count_exceeded") == 3


def test_a_tag_count_at_exactly_the_limit_is_not_caught():
    a = item("a", 50, 50, 50, tags=["hazmat"])
    b = item("b", 50, 50, 50, tags=["hazmat"])
    box = shelf(200, 100, 100, tag_limits={"hazmat": 2})
    a_instance, = a.instances()
    b_instance, = b.instances()
    packed = PackedContainer(box, 1, (place(a_instance, x=0), place(b_instance, x=50 * MM)))
    assert issues_for([a, b], [box], [packed]) == []


def test_an_untagged_item_is_never_limited_through_the_validator():
    a = item("a", 50, 50, 50, tags=["hazmat"])
    untagged = item("plain", 50, 50, 50)
    box = shelf(200, 100, 100, tag_limits={"hazmat": 1})
    a_instance, = a.instances()
    plain_instance, = untagged.instances()
    packed = PackedContainer(box, 1, (place(a_instance, x=0), place(plain_instance, x=50 * MM)))
    assert issues_for([a, untagged], [box], [packed]) == []


def test_incompatible_neighbours_are_caught():
    food = item("food", 50, 50, 50, tags=["food"])
    chemical = item("chem", 50, 50, 50, incompatible_tags=["food"])
    box = shelf()
    food_instance, = food.instances()
    chemical_instance, = chemical.instances()
    packed = PackedContainer(box, 1, (place(food_instance), place(chemical_instance, x=50 * MM)))
    assert "incompatible_items" in issues_for([food, chemical], [box], [packed])


# --------------------------------------------------------- unloading order

def test_a_route_order_that_agrees_with_the_stacking_order_raises_nothing():
    """`top` (the contact graph: it must come off before `bottom` no matter what
    any route says) is also due at the earlier stop here, so nothing conflicts."""
    bottom = item("bottom", 100, 100, 100, stop_index=1)
    top = item("top", 100, 100, 100, stop_index=0)
    box = shelf(100, 100, 200)
    bottom_instance, = bottom.instances()
    top_instance, = top.instances()
    packed = PackedContainer(box, 1, (place(bottom_instance), place(top_instance, z=100 * MM)))
    assert issues_for([bottom, top], [box], [packed]) == []


def test_a_stop_index_that_cannot_be_unloaded_in_route_order_is_caught():
    """`bottom` is due at the earlier stop, but `top` -- structurally required off
    first regardless of any route, since it rests on `bottom` -- is not due until
    later. No order can satisfy both, so this must come back as its own, distinct
    reason code, not get folded into an unrelated one."""
    bottom = item("bottom", 100, 100, 100, stop_index=0)
    top = item("top", 100, 100, 100, stop_index=1)
    box = shelf(100, 100, 200)
    bottom_instance, = bottom.instances()
    top_instance, = top.instances()
    packed = PackedContainer(box, 1, (place(bottom_instance), place(top_instance, z=100 * MM)))
    assert "unloading_order_violation" in issues_for([bottom, top], [box], [packed])


def test_items_without_a_stop_index_are_not_route_checked_at_all():
    """Existing single-stop callers never set `stop_index` -- the exact same
    structurally-conflicting stack as the violation case above must raise nothing at
    all when neither item is on a route, not merely avoid the new code."""
    a = item("a", 100, 100, 100)
    b = item("b", 100, 100, 100)
    box = shelf(100, 100, 200)
    a_instance, = a.instances()
    b_instance, = b.instances()
    packed = PackedContainer(box, 1, (place(a_instance), place(b_instance, z=100 * MM)))
    assert issues_for([a, b], [box], [packed]) == []


# ------------------------------------------------------------------- reporting

def test_the_report_names_the_offending_item():
    cubes = item("cube", quantity=2)
    box = shelf()
    first, second = cubes.instances()
    from packvium import IndependentSolutionValidator, PackingRequest
    report = IndependentSolutionValidator().validate(
        PackingRequest((cubes,), (box,)),
        (PackedContainer(box, 1, (place(first), place(second, x=50 * MM))),),
    )
    assert not report.valid
    assert any("cube#1" in issue.detail and "cube#2" in issue.detail for issue in report.issues)
