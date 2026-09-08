"""Physical placement rules: floor, compatibility, support area and bearing load.

These decide feasibility, so every comparison here is exact integer arithmetic even
though the public API accepts support ratios as floats. A boundary that moved by one
tick between languages would make the same request pack differently, which is the
class of bug the fixed-point design exists to prevent.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from packvium import AxisAlignedBox, Axle, Container, Dimensions, Item, Length, Point, Rotation, Weight
from packvium import constraints
from packvium.constraints import (SQUARE_METRE_TICKS, SUPPORT_SCALE, AxleLoadConstraint,
                                     CompatibilityConstraint, ConstraintContext, ContainerEligibilityConstraint,
                                     FloorConstraint, LoadSupportGraph, LoadUnit, SupportConstraint, TagCountConstraint,
                                     RIDES_THE_WHOLE_ROUTE, RouteOrderConstraint, TopLoadConstraint,
                                     direct_support_view, load_units, overloaded, required_area,
                                     route_order_violated, scaled_ratio,
                                     stack_density_exceeded, stack_limit_exceeded, stacked_counts, top_loads)
from packvium.contact import ContactGraph
from packvium.models import Placement

BOX = Container.create("box", Dimensions.mm(100, 100, 100))


def unit(x, y, z, length, width, height, weight=0, limit=None, stack_limit=None, label="u",
         nesting_item_id=None, nesting_height_ticks=None) -> LoadUnit:
    """A load-bearing solid described directly in ticks."""
    box = AxisAlignedBox(Point(x, y, z), Dimensions(Length(length), Length(width), Length(height)))
    return LoadUnit(
        box, weight, limit, stack_limit, label, nesting_item_id, nesting_height_ticks
    )


def instance_of(id: str, length=10, width=10, height=10, **kwargs):
    one, = Item.create(id, Dimensions.mm(length, width, height), kwargs.pop("weight", 0), **kwargs).instances()
    return one


def placed(instance, x=0, y=0, z=0) -> Placement:
    dims = instance.item.dimensions
    return Placement(instance, Point(x, y, z), Rotation.LWH, dims, Point(x, y, z), dims)


def context(instance, x=0, y=0, z=0, placements=(), container=BOX, stack_sensitive=True,
            dimensions=None, route_sensitive=True) -> ConstraintContext:
    dims = dimensions or instance.item.dimensions
    return ConstraintContext(container, tuple(placements), instance, Point(x, y, z),
                             Rotation.LWH, dims, dims, stack_sensitive, route_sensitive)


# ------------------------------------------------------- scaled ratio arithmetic

def test_ratios_scale_to_parts_per_million():
    assert scaled_ratio(0.0) == 0
    assert scaled_ratio(0.75) == 750_000
    assert scaled_ratio(1.0) == SUPPORT_SCALE
    assert scaled_ratio(1 / 3) == 333_333


@pytest.mark.parametrize("base_area", [0, 1, 999_999, 1_000_000, 2_560_000_000_000, 10**18])
@pytest.mark.parametrize("ratio", [0, 1, 333_333, 500_000, 750_000, 1_000_000])
def test_required_area_is_an_exact_floor_division(base_area, ratio):
    """Computed without ever forming `base_area * ratio`, which would overflow a 64-bit
    integer in PHP for a container-sized footprint. The split-and-recombine must still
    agree with the exact rational value tick for tick."""
    assert required_area(base_area, ratio) == Fraction(base_area * ratio, SUPPORT_SCALE).__floor__()


def test_full_support_requires_the_whole_footprint():
    assert required_area(2_560_000_000_000, SUPPORT_SCALE) == 2_560_000_000_000


# ------------------------------------------------------------ load propagation

def test_a_tower_pushes_its_whole_weight_onto_the_base():
    """Not just the box directly underneath: a stack of light items must be able to
    crush a base rated for less than their sum."""
    tower = (
        unit(0, 0, 0, 10, 10, 10, weight=100, label="bottom"),
        unit(0, 0, 10, 10, 10, 10, weight=100, label="middle"),
        unit(0, 0, 20, 10, 10, 10, weight=100, label="top"),
    )
    assert top_loads(tower) == [200, 100, 0]


def test_a_three_layer_nested_column_propagates_load_and_stacked_count_directly():
    nested = (
        unit(0, 0, 0, 10, 10, 10, weight=100, limit=200, label="bottom",
             nesting_item_id="crate", nesting_height_ticks=5),
        unit(0, 0, 5, 10, 10, 10, weight=100, limit=75, label="middle",
             nesting_item_id="crate", nesting_height_ticks=5),
        unit(0, 0, 10, 10, 10, 10, weight=100, label="top",
             nesting_item_id="crate", nesting_height_ticks=5),
    )

    assert top_loads(nested) == [200, 100, 0]
    assert stacked_counts(nested) == [2, 1, 0]
    assert overloaded(nested) == ("top_load_exceeded", "middle")


def test_non_stackable_rejects_nested_candidates_above_and_below():
    crate = Item.create(
        "crate", Dimensions.mm(100, 100, 50), quantity=2,
        nesting_height=Length.mm(25), stackable=False,
    )
    lower, upper = crate.instances()
    constraint = TopLoadConstraint()

    above = constraint.evaluate(
        context(upper, z=Length.mm(25).ticks, placements=(placed(lower),))
    )
    below = constraint.evaluate(
        context(lower, z=0, placements=(placed(upper, z=Length.mm(25).ticks),))
    )

    assert not above.allowed and above.code == "non_stackable"
    assert not below.allowed and below.code == "non_stackable"


def test_a_nested_top_load_boundary_rejects_one_tick_below_the_exact_load():
    nested = (
        unit(0, 0, 0, 10, 10, 10, weight=100, limit=199, label="bottom",
             nesting_item_id="crate", nesting_height_ticks=4),
        unit(0, 0, 6, 10, 10, 10, weight=100, label="middle",
             nesting_item_id="crate", nesting_height_ticks=4),
        unit(0, 0, 12, 10, 10, 10, weight=100, label="top",
             nesting_item_id="crate", nesting_height_ticks=4),
    )

    assert overloaded(nested) == ("top_load_exceeded", "bottom")


@pytest.mark.parametrize(
    ("rule", "allowed"),
    [("single", True), ("covered", True), ("multiple", False)],
)
def test_nested_support_is_one_full_area_predecessor_without_a_shadow_face(rule, allowed):
    crate = Item.create(
        "crate", Dimensions.mm(10, 10, 10), quantity=3,
        nesting_height=Length.mm(5), minimum_support_ratio=1.0,
        ground_contact_rule=rule,
    )
    bottom, middle, top = crate.instances()
    placements = (placed(bottom), placed(middle, z=Length.mm(5).ticks))
    candidate = context(top, z=Length.mm(10).ticks, placements=placements)

    support = direct_support_view(placements, top, candidate.envelope_box)
    assert [entry.index for entry in support.supporters] == [1]
    assert support.supporting_area == top.dimensions.base_area
    result = SupportConstraint(0.0).evaluate(candidate)
    assert result.allowed is allowed
    if not allowed:
        assert result.code == "ground_contact_violation"


def test_equal_item_ids_with_different_nesting_depths_do_not_share_support():
    lower = instance_of("crate", nesting_height=Length.mm(4))
    upper = instance_of("crate", nesting_height=Length.mm(5), minimum_support_ratio=1.0)
    result = SupportConstraint().evaluate(
        context(upper, z=Length.mm(6).ticks, placements=(placed(lower),))
    )
    assert not result.allowed and result.code == "insufficient_support"

    units = (
        unit(0, 0, 0, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=4),
        unit(0, 0, 6, 10, 10, 10, weight=100,
             nesting_item_id="crate", nesting_height_ticks=5),
    )
    assert top_loads(units) == [0, 0]


def test_only_the_adjacent_nesting_layer_can_be_a_direct_supporter():
    crate = Item.create(
        "crate", Dimensions.mm(10, 10, 10), quantity=3,
        nesting_height=Length.mm(5), minimum_support_ratio=1.0,
    )
    lower, intervening, candidate = crate.instances()
    placements = (placed(lower), placed(intervening, z=Length.mm(2).ticks))

    result = SupportConstraint().evaluate(
        context(candidate, z=Length.mm(5).ticks, placements=placements)
    )

    assert not result.allowed and result.code == "insufficient_support"


def test_nested_support_replacement_restores_canonical_index_order():
    supporters = (
        unit(0, 0, 0, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=5),
        unit(0, 0, 5, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=5),
        unit(0, 0, 0, 5, 10, 5),
    )
    graph = LoadSupportGraph(supporters)
    assert [edge.index for edge in graph.supporters(1)] == [0, 2]

    children = (
        unit(0, 0, 5, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=5),
        unit(0, 0, 10, 5, 10, 5),
        unit(0, 0, 0, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=5),
    )
    graph = LoadSupportGraph(children)
    assert graph.children(2) == (0, 1)


def test_a_nesting_predecessor_takes_its_place_in_the_surface_order():
    """A face seen before the predecessor keeps its position, it does not move behind it.

    `direct_support_view` promises placement-index order, and the replacement inserts
    the predecessor rather than appending it. Ordering is what makes two engines agree
    on which supporter a rejection names, so it is a contract, not a detail.
    """
    crate = Item.create("crate", Dimensions.mm(10, 10, 10), quantity=2,
                        nesting_height=Length.mm(5), minimum_support_ratio=1.0)
    predecessor, candidate = crate.instances()
    beside, = Item.create("shelf", Dimensions.mm(10, 10, 5)).instances()
    placements = (placed(beside, x=Length.mm(20).ticks), placed(predecessor))
    resting = context(candidate, z=Length.mm(5).ticks, placements=placements)

    support = direct_support_view(placements, candidate, resting.envelope_box)

    assert [surface.origin.x for surface in support.surfaces] == [Length.mm(20).ticks, 0]
    assert [entry.index for entry in support.supporters] == [1]


def test_a_replaced_nesting_edge_lands_after_a_lower_indexed_face_supporter():
    """The same ordering contract in `LoadSupportGraph`, where the edge is inserted into
    a list that already holds a retained face edge from an earlier unit."""
    units = (
        unit(0, 0, 0, 5, 10, 5),
        unit(0, 0, 0, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=5),
        unit(0, 0, 5, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=5),
    )

    assert [edge.index for edge in LoadSupportGraph(units).supporters(2)] == [0, 1]


def test_the_same_crate_type_at_the_wrong_depth_is_not_a_nesting_pair():
    """Matching item type and footprint are not enough: the vertical gap must be exactly
    the declared nesting height. Treating an off-by-two column as nested would carry the
    upper crate's weight onto a lower one it is not actually resting in."""
    units = (
        unit(0, 0, 0, 10, 10, 10, nesting_item_id="crate", nesting_height_ticks=5),
        unit(0, 0, 7, 10, 10, 10, weight=100,
             nesting_item_id="crate", nesting_height_ticks=5),
    )

    assert top_loads(units) == [0, 0]


def test_load_splits_by_contact_area():
    supporters_and_load = (
        unit(0, 0, 0, 10, 10, 10, label="narrow"),
        unit(10, 0, 0, 30, 10, 10, label="wide"),
        unit(0, 0, 10, 40, 10, 10, weight=1_000, label="beam"),
    )
    assert top_loads(supporters_and_load) == [250, 750, 0]


def test_an_indivisible_load_is_conserved_exactly():
    """The integer remainder goes to the last supporter rather than being dropped, so
    the distributed total always equals what was pushed down."""
    units = (
        unit(0, 0, 0, 1, 10, 10, label="a"),
        unit(1, 0, 0, 2, 10, 10, label="b"),
        unit(0, 0, 10, 3, 10, 10, weight=1_000, label="beam"),
    )
    loads = top_loads(units)
    assert loads == [333, 667, 0]
    assert sum(loads) == 1_000


def test_a_box_hanging_clear_of_everything_loads_nothing():
    units = (unit(0, 0, 0, 10, 10, 10, label="floor"), unit(50, 50, 0, 10, 10, 10, weight=99, label="away"))
    assert top_loads(units) == [0, 0]


def test_boxes_that_only_touch_sideways_do_not_bear_each_other():
    units = (unit(0, 0, 0, 10, 10, 10, label="left"), unit(10, 0, 0, 10, 10, 10, weight=500, label="right"))
    assert top_loads(units) == [0, 0]


def test_no_bearing_limits_means_no_check_to_run():
    units = (unit(0, 0, 0, 10, 10, 10, weight=10**9, label="a"), unit(0, 0, 10, 10, 10, 10, weight=10**9, label="b"))
    assert overloaded(units) is None


def test_the_first_crushed_unit_is_reported_by_name():
    units = (
        unit(0, 0, 0, 10, 10, 10, weight=1, limit=150, label="base"),
        unit(0, 0, 10, 10, 10, 10, weight=100, label="middle"),
        unit(0, 0, 20, 10, 10, 10, weight=100, label="top"),
    )
    assert overloaded(units) == ("top_load_exceeded", "base")


def test_a_limit_met_exactly_is_not_exceeded():
    units = (
        unit(0, 0, 0, 10, 10, 10, limit=100, label="base"),
        unit(0, 0, 10, 10, 10, 10, weight=100, label="top"),
    )
    assert overloaded(units) is None


def test_load_units_are_built_from_placements_and_the_candidate():
    base = placed(instance_of("base", weight="1 kg", max_top_load="2 kg"))
    extra = unit(0, 0, 160_000, 160_000, 160_000, 160_000, weight=8_000_000, label="candidate")
    units = load_units((base,), extra)
    assert [u.label for u in units] == ["base#1", "candidate"]
    assert units[0].weight_ticks == Weight.of(1, "kg").ticks
    assert units[0].max_top_load_ticks == Weight.of(2, "kg").ticks


# --------------------------------------------------------------- contact graph

def _random_box(rng: random.Random) -> AxisAlignedBox:
    l, w, h = (rng.randrange(5, 60, 5) for _ in range(3))
    x, y, z = (rng.randrange(0, 200, 5) for _ in range(3))
    return AxisAlignedBox(Point(x, y, z), Dimensions(Length(l), Length(w), Length(h)))


def _brute_force_supporters(boxes, index: int) -> set[tuple[int, int]]:
    """Independent, deliberately naive all-pairs contact check sharing no code with
    `ContactGraph`'s by-plane index -- silently dropping or inventing an edge is
    exactly the risk that index carries and a shared implementation could not catch."""
    target = boxes[index]
    return {
        (other_index, other.overlap_area_xy(target))
        for other_index, other in enumerate(boxes)
        if other_index != index and other.z2 == target.origin.z and other.overlap_area_xy(target) > 0
    }


def _brute_force_children(boxes, index: int) -> set[int]:
    target = boxes[index]
    return {
        other_index
        for other_index, other in enumerate(boxes)
        if other_index != index and other.origin.z == target.z2 and target.overlap_area_xy(other) > 0
    }


@pytest.mark.parametrize("seed", range(40))
def test_contact_graph_agrees_with_a_brute_force_pairwise_scan(seed):
    rng = random.Random(seed)
    boxes = [_random_box(rng) for _ in range(rng.randint(2, 10))]
    graph = ContactGraph(boxes)
    for index in range(len(boxes)):
        assert {(edge.index, edge.area) for edge in graph.supporters(index)} == _brute_force_supporters(boxes, index)
        assert set(graph.children(index)) == _brute_force_children(boxes, index)


def test_a_box_with_no_neighbours_has_no_supporters_or_children():
    graph = ContactGraph([AxisAlignedBox(Point(0, 0, 0), Dimensions(Length(10), Length(10), Length(10)))])
    assert graph.supporters(0) == ()
    assert graph.children(0) == ()


@pytest.mark.parametrize("seed", range(10))
def test_contact_graph_agrees_with_brute_force_on_a_dense_shared_level(seed):
    """The 2-10 box property test above never puts more than a handful of boxes on
    one z-level, so it cannot exercise `_LevelIndex`'s multi-cell-per-box path at any
    real density. A regular lattice (`GridSolver`) routinely puts hundreds of boxes on
    one shared level -- this rebuilds that shape directly: a uniform grid on
    one level (guaranteeing many boxes share exactly one spatial-hash cell) plus a
    handful of random extra boxes at the same and other levels, so the dense and the
    sparse cases are both present in one scene."""
    rng = random.Random(1000 + seed)
    boxes = []
    for gx in range(12):
        for gy in range(12):
            boxes.append(AxisAlignedBox(Point(gx * 10, gy * 10, 0), Dimensions(Length(10), Length(10), Length(10))))
    for _ in range(30):
        boxes.append(_random_box(rng))
    graph = ContactGraph(boxes)
    for index in range(len(boxes)):
        assert {(edge.index, edge.area) for edge in graph.supporters(index)} == _brute_force_supporters(boxes, index)
        assert set(graph.children(index)) == _brute_force_children(boxes, index)


def test_supporters_stay_ordered_by_index_for_the_remainder_split():
    """`top_loads` (constraints.py) hands the rounding remainder to whichever edge is
    *last* in `supporters(index)` -- this must stay the same ascending-index order the
    original all-pairs scan produced regardless of which spatial-hash cells the
    optimized lookup happens to visit first."""
    beam = AxisAlignedBox(Point(0, 0, 10), Dimensions(Length(3), Length(10), Length(10)))
    below = [
        AxisAlignedBox(Point(2, 0, 0), Dimensions(Length(1), Length(10), Length(10))),
        AxisAlignedBox(Point(1, 0, 0), Dimensions(Length(1), Length(10), Length(10))),
        AxisAlignedBox(Point(0, 0, 0), Dimensions(Length(1), Length(10), Length(10))),
    ]
    graph = ContactGraph([beam, *below])
    assert [edge.index for edge in graph.supporters(0)] == [1, 2, 3]


# ------------------------------------------------- incremental append

def _touching_scene(rng: random.Random, count: int) -> list[AxisAlignedBox]:
    """A scene whose boxes actually touch each other.

    `_random_box` draws from a range wide enough that most of its scenes have no
    contact at all, which is fine for the brute-force agreement property above -- an
    empty edge set is still an edge set both implementations must agree on. It is not
    fine here: what is under test is that a delta reproduces edges, so a corpus where
    most scenes have no edges would pass with the delta returning nothing. Snapping
    every coordinate and extent to one coarse lattice makes shared planes the norm.
    """
    return [
        AxisAlignedBox(
            Point(rng.randrange(0, 60, 10), rng.randrange(0, 60, 10), rng.choice([0, 10, 20, 30])),
            Dimensions(*(Length(rng.choice([10, 20, 30])) for _ in range(3))),
        )
        for _ in range(count)
    ]


def _widest_footprint(boxes) -> int:
    return max(max(box.x2 - box.origin.x, box.y2 - box.origin.y) for box in boxes)


def _edges(graph, count: int):
    """Both edge directions as ordered tuples.

    Compared as sequences, never as sets: `top_loads` hands the integer rounding
    remainder to whichever supporter is *last*, so two graphs holding the same edges in
    a different order are two different answers.
    """
    return (
        [tuple((edge.index, edge.area) for edge in graph.supporters(i)) for i in range(count)],
        [tuple(graph.children(i)) for i in range(count)],
    )


def _count_full_builds(monkeypatch, target) -> list[int]:
    """Record every from-scratch build of `target`, so a test can tell the delta path
    from the fallback. Without this an assertion that the two graphs match is satisfied
    by a `with_box` that quietly rebuilds everything -- correct, and none of the point."""
    builds: list[int] = []
    original = target.__init__

    def counting(self, units, cell_hint=1):
        builds.append(len(units))
        original(self, units, cell_hint)

    monkeypatch.setattr(target, "__init__", counting)
    return builds


@pytest.mark.parametrize("seed", range(40))
def test_appending_a_box_matches_building_the_whole_scene_at_once(seed, monkeypatch):
    """The delta is required to be *identical* to the full build, not merely equivalent.

    The base is built with the widest footprint in the scene as its hint, which is what
    a solver knows before it starts placing: the candidate about to be appended may be
    larger than anything already placed, and sizing the spatial hash from the placed
    boxes alone would send every append into the fallback.
    """
    rng = random.Random(2000 + seed)
    boxes = _touching_scene(rng, rng.randint(2, 14))
    split = max(1, len(boxes) // 2)
    hint = _widest_footprint(boxes)

    builds = _count_full_builds(monkeypatch, ContactGraph)
    graph = ContactGraph(boxes[:split], cell_hint=hint)
    for box in boxes[split:]:
        graph = graph.with_box(box)

    assert builds == [split], "an append fell back to a full rebuild"
    assert _edges(graph, len(boxes)) == _edges(ContactGraph(boxes), len(boxes))


@pytest.mark.parametrize("coordinates,extents,hint_mode", [
    ((0, 1, 2, 5, 10), (1, 2, 3), "exact"),
    ((0, 1, 2, 5, 10), (1, 2, 3), "one"),
    ((0, 1, 2, 5, 10), (1, 5, 10, 40), "one"),
    ((0, 1, 2, 5, 10), (1, 5, 10, 40), "huge"),
    ((0, 10, 100, 10 ** 9), (1, 5, 10, 40), "exact"),
    ((0, 10, 100, 10 ** 9), (1, 5, 10, 40), "one"),
    ((0, 10, 100, 10 ** 9), (1, 2, 3), "huge"),
])
def test_the_delta_matches_a_rebuild_across_scene_shapes(coordinates, extents, hint_mode):
    """The same invariant as the property test above, over shapes it deliberately excludes.

    That test asserts *zero* fallbacks, because proving the delta ran was the thing at
    stake. The consequence is that nothing exercised a run where the fallback and the delta
    interleave -- and a hint of one forces exactly that, several times per scene.

    The three axes vary independently on purpose. Tight coordinates make shared planes and
    zero-area edge contacts the norm; coordinates at 10^9 push the spatial hash's cell
    arithmetic somewhere a lattice never goes; a huge hint collapses every box into one
    cell, which is the degenerate case the hash exists to avoid and therefore the one most
    likely to be wrong.

    Written after an unsound optimality bound in a neighbouring module survived 183 tests
    that all shared one shape. Coverage there was complete; variety was not.
    """
    rng = random.Random(hash((coordinates, extents, hint_mode)) & 0xFFFF)
    for _ in range(60):
        count = rng.randint(1, 10)
        boxes = [
            AxisAlignedBox(
                Point(rng.choice(coordinates), rng.choice(coordinates), rng.choice(coordinates)),
                Dimensions(*(Length(rng.choice(extents)) for _ in range(3))),
            )
            for _ in range(count)
        ]
        widest = _widest_footprint(boxes)
        hint = {"exact": widest, "one": 1, "huge": widest * 100}[hint_mode]
        split = max(1, count // 2)
        graph = ContactGraph(boxes[:split], cell_hint=hint)
        for box in boxes[split:]:
            graph = graph.with_box(box)
        assert _edges(graph, count) == _edges(ContactGraph(boxes), count)


def test_a_box_wider_than_the_hint_rebuilds_and_is_still_correct(monkeypatch):
    """The hint is an optimisation; being wrong about it may cost time, never an answer.

    `_LevelIndex` is only sound while its cell is at least the largest footprint it
    indexes or is queried with, so a box that exceeds the cell has to be met with a
    rebuild -- this asserts both halves: that the rebuild happens, and that the result
    is the one the full build gives.
    """
    small = [
        AxisAlignedBox(Point(0, 0, 0), Dimensions(Length(10), Length(10), Length(10))),
        AxisAlignedBox(Point(10, 0, 0), Dimensions(Length(10), Length(10), Length(10))),
    ]
    wide = AxisAlignedBox(Point(0, 0, 10), Dimensions(Length(40), Length(10), Length(10)))

    builds = _count_full_builds(monkeypatch, ContactGraph)
    graph = ContactGraph(small).with_box(wide)

    assert builds == [2, 3], "a box exceeding the cell must not use the delta path"
    assert [edge.index for edge in graph.supporters(2)] == [0, 1]
    assert _edges(graph, 3) == _edges(ContactGraph([*small, wide]), 3)


def test_an_appended_box_lands_last_in_the_tuples_it_joins():
    """The remainder-split contract, on the delta path specifically.

    The new box always takes the highest index, so appending it to an existing
    supporter tuple keeps that tuple ascending -- but only because it is appended and
    not inserted, which is the kind of detail a from-scratch comparison on random
    scenes can miss when no scene happens to produce the collision.
    """
    below = [
        AxisAlignedBox(Point(0, 0, 0), Dimensions(Length(10), Length(10), Length(10))),
        AxisAlignedBox(Point(10, 0, 0), Dimensions(Length(10), Length(10), Length(10))),
    ]
    resting = AxisAlignedBox(Point(0, 0, 10), Dimensions(Length(20), Length(10), Length(10)))
    graph = ContactGraph([below[0], resting, below[1]], cell_hint=20).with_box(
        AxisAlignedBox(Point(0, 0, 20), Dimensions(Length(10), Length(10), Length(10)))
    )
    assert [edge.index for edge in graph.supporters(1)] == [0, 2]
    assert graph.children(1) == (3,)


@pytest.mark.parametrize("seed", range(40))
def test_appending_a_unit_matches_building_the_support_graph_at_once(seed, monkeypatch):
    rng = random.Random(4000 + seed)
    boxes = _touching_scene(rng, rng.randint(2, 12))
    units = [LoadUnit(box, 100, None, None, f"u{i}") for i, box in enumerate(boxes)]
    hint = _widest_footprint(boxes)

    builds = _count_full_builds(monkeypatch, LoadSupportGraph)
    graph = LoadSupportGraph(units[:1], cell_hint=hint)
    for unit in units[1:]:
        graph = graph.with_unit(unit, cell_hint=hint)

    assert builds == [1], "an append fell back to a full rebuild"
    assert _edges(graph, len(units)) == _edges(LoadSupportGraph(units), len(units))


def test_the_adversarial_dense_scene_costs_only_the_edges_it_reports(monkeypatch):
    """The bound the delta is actually claimed to meet, on the shape that is worst for it.

    Two layers of thin strips laid at right angles -- k running along x below, k along y
    above -- so every upper strip crosses every lower one and the graph really holds
    k*k edges. It is an ordinary criss-crossed dunnage stack, not a contrivance, and it is
    the shape any exact contact representation is quadratic on: the edges are there, and
    reporting them is the work.

    So the delta is not claimed to be cheap here. It is claimed to cost the edges it
    reports and nothing else: appending one strip touches k boxes, and the probe count
    must stay within a constant factor of k rather than climbing towards the k*k a
    from-scratch build performs -- which is what a delta being local *means*.
    """
    k = 40
    span = k * 10
    lower = [
        AxisAlignedBox(Point(0, index * 10, 0), Dimensions(Length(span), Length(10), Length(10)))
        for index in range(k)
    ]
    upper = [
        AxisAlignedBox(Point(index * 10, 0, 10), Dimensions(Length(10), Length(span), Length(10)))
        for index in range(k - 1)
    ]
    arriving = AxisAlignedBox(
        Point((k - 1) * 10, 0, 10), Dimensions(Length(10), Length(span), Length(10)))

    base = ContactGraph(lower + upper, cell_hint=span)
    assert sum(len(base.supporters(i)) for i in range(len(lower) + len(upper))) == k * (k - 1), (
        "the scene is meant to be quadratically dense; if it is not, the bound is untested"
    )

    probes = 0
    original = AxisAlignedBox.overlap_area_xy

    def counting(self, other):
        nonlocal probes
        probes += 1
        return original(self, other)

    monkeypatch.setattr(AxisAlignedBox, "overlap_area_xy", counting)
    graph = base.with_box(arriving)

    assert len(graph.supporters(len(lower) + len(upper))) == k
    assert probes <= 4 * k, f"{probes} probes to report {k} edges is not a local delta"


@pytest.mark.parametrize("coordinates,extents,hint_mode", [
    ((0, 1, 2, 5, 10), (1, 2, 3), "exact"),
    ((0, 1, 2, 5, 10), (1, 2, 3), "one"),
    ((0, 1, 2, 5, 10), (1, 5, 10, 40), "one"),
    ((0, 10, 100, 10 ** 9), (1, 5, 10, 40), "exact"),
    ((0, 10, 100, 10 ** 9), (1, 2, 3), "huge"),
])
def test_appending_a_unit_matches_a_rebuild_across_scene_shapes(coordinates, extents, hint_mode):
    """The support graph gets the same treatment as the contact graph beneath it.

    `LoadSupportGraph.with_unit` reads its edges straight off the face graph, so in the
    non-nesting case it is only as correct as `ContactGraph.with_box` -- but "only as
    correct as" is an argument, and an argument is what the unsound optimality bound in a
    neighbouring module also had. The shapes are varied here too rather than reasoned about.
    """
    rng = random.Random(hash((coordinates, extents, hint_mode)) & 0xFFFF)
    for _ in range(40):
        count = rng.randint(1, 9)
        units = [
            LoadUnit(
                AxisAlignedBox(
                    Point(rng.choice(coordinates), rng.choice(coordinates), rng.choice(coordinates)),
                    Dimensions(*(Length(rng.choice(extents)) for _ in range(3))),
                ),
                rng.choice((0, 100, 5000)), None, None, f"u{index}",
            )
            for index in range(count)
        ]
        widest = _widest_footprint([unit.box for unit in units])
        hint = {"exact": widest, "one": 1, "huge": widest * 100}[hint_mode]
        graph = LoadSupportGraph(units[:1], cell_hint=hint)
        for unit in units[1:]:
            graph = graph.with_unit(unit, cell_hint=hint)
        assert _edges(graph, count) == _edges(LoadSupportGraph(units), count)


def test_a_nesting_unit_is_met_with_a_full_rebuild(monkeypatch):
    """Nesting is deliberately excluded from the delta, and the exclusion is load-bearing.

    A nesting predecessor replaces the face edges of its whole column, so one new unit
    can rewrite edges arbitrarily far from itself -- the locality the delta rests on is
    simply not there. The rebuild is the correct answer, so assert it is taken.
    """
    stack = (
        unit(0, 0, 0, 10, 10, 10, label="a", nesting_item_id="tray", nesting_height_ticks=4),
        unit(0, 0, 6, 10, 10, 10, label="b", nesting_item_id="tray", nesting_height_ticks=4),
    )
    arriving = unit(0, 0, 12, 10, 10, 10, label="c", nesting_item_id="tray", nesting_height_ticks=4)

    builds = _count_full_builds(monkeypatch, LoadSupportGraph)
    graph = LoadSupportGraph(stack[:1]).with_unit(stack[1]).with_unit(arriving)

    assert builds == [1, 2, 3]
    assert _edges(graph, 3) == _edges(LoadSupportGraph((*stack, arriving)), 3)


# --------------------------------------------------------- stacked-item counting

def test_a_three_high_column_counts_transitively_not_just_the_neighbour():
    column = (
        unit(0, 0, 0, 10, 10, 10, label="bottom"),
        unit(0, 0, 10, 10, 10, 10, label="middle"),
        unit(0, 0, 20, 10, 10, 10, label="top"),
    )
    assert stacked_counts(column) == [2, 1, 0]


def test_items_side_by_side_do_not_count_toward_each_other():
    side_by_side = (unit(0, 0, 0, 10, 10, 10, label="left"), unit(10, 0, 0, 10, 10, 10, label="right"))
    assert stacked_counts(side_by_side) == [0, 0]


def test_an_item_resting_on_two_supporters_counts_once_for_each():
    supporters_and_load = (
        unit(0, 0, 0, 10, 10, 10, label="narrow"),
        unit(10, 0, 0, 30, 10, 10, label="wide"),
        unit(0, 0, 10, 40, 10, 10, label="beam"),
    )
    assert stacked_counts(supporters_and_load) == [1, 1, 0]


def test_no_stack_limits_means_no_check_to_run():
    column = (unit(0, 0, 0, 10, 10, 10, label="a"), unit(0, 0, 10, 10, 10, 10, label="b"))
    assert stack_limit_exceeded(column) is None


def test_the_first_unit_over_its_stack_limit_is_reported_by_name():
    column = (
        unit(0, 0, 0, 10, 10, 10, stack_limit=1, label="base"),
        unit(0, 0, 10, 10, 10, 10, label="middle"),
        unit(0, 0, 20, 10, 10, 10, label="top"),
    )
    assert stack_limit_exceeded(column) == ("stacked_item_limit_exceeded", "base")


def test_a_stack_limit_met_exactly_is_not_exceeded():
    column = (
        unit(0, 0, 0, 10, 10, 10, stack_limit=2, label="base"),
        unit(0, 0, 10, 10, 10, 10, label="middle"),
        unit(0, 0, 20, 10, 10, 10, label="top"),
    )
    assert stack_limit_exceeded(column) is None


def test_a_third_item_over_the_stack_limit_is_refused():
    base = placed(instance_of("base", max_stacked_items=1))
    middle = placed(instance_of("middle"), z=160_000)
    top = instance_of("top")
    result = TopLoadConstraint().evaluate(context(top, z=320_000, placements=(base, middle)))
    assert not result.allowed and result.code == "stacked_item_limit_exceeded"


def test_within_the_stack_limit_is_allowed():
    base = placed(instance_of("base", max_stacked_items=2))
    middle = placed(instance_of("middle"), z=160_000)
    top = instance_of("top")
    assert TopLoadConstraint().evaluate(context(top, z=320_000, placements=(base, middle))).allowed


# --------------------------------------------------------- stack density

ONE_METRE_TICKS = Length.mm(1000).ticks


def test_square_metre_ticks_is_derived_from_the_length_scale():
    assert SQUARE_METRE_TICKS == ONE_METRE_TICKS ** 2


def test_no_density_limit_means_no_check_to_run():
    column = (unit(0, 0, 0, 10, 10, 10, weight=10**9, label="a"),)
    assert stack_density_exceeded(column, None) is None


def test_a_unit_bearing_exactly_its_density_limit_is_not_exceeded():
    # A one-square-metre footprint collapses the cross-multiplication to a plain
    # integer comparison, keeping the boundary tick easy to state and verify by hand.
    floor = unit(0, 0, 0, ONE_METRE_TICKS, ONE_METRE_TICKS, 10, weight=500, label="floor")
    assert stack_density_exceeded((floor,), 500) is None


def test_a_unit_one_tick_over_its_density_limit_is_reported_by_name():
    floor = unit(0, 0, 0, ONE_METRE_TICKS, ONE_METRE_TICKS, 10, weight=501, label="floor")
    assert stack_density_exceeded((floor,), 500) == ("stack_density_exceeded", "floor")


def test_stack_density_is_checked_cumulatively_not_just_the_direct_neighbour():
    base = unit(0, 0, 0, ONE_METRE_TICKS, ONE_METRE_TICKS, 10, weight=100, label="base")
    top = unit(0, 0, 10, ONE_METRE_TICKS, ONE_METRE_TICKS, 10, weight=450, label="top")
    assert stack_density_exceeded((base, top), 500) == ("stack_density_exceeded", "base")


def test_a_small_footprint_receiving_a_large_load_from_above_is_exceeded():
    """The whole point of a density limit over a flat per-item one: the same absolute
    load becomes crushing once concentrated onto less area. `leg`'s own weight is
    negligible; it is purely `block`'s weight, transmitted through a tenth of the
    footprint, that crushes it."""
    leg = unit(0, 0, 0, ONE_METRE_TICKS // 10, ONE_METRE_TICKS, 10, weight=1, label="leg")
    block = unit(0, 0, 10, ONE_METRE_TICKS, ONE_METRE_TICKS, 10, weight=100, label="block")
    assert stack_density_exceeded((leg, block), 500) == ("stack_density_exceeded", "leg")


def test_a_density_limited_container_refuses_a_crushing_candidate():
    dense_box = Container.create(
        "dense", Dimensions(Length(ONE_METRE_TICKS), Length(ONE_METRE_TICKS), Length(ONE_METRE_TICKS)),
        max_stack_density=Weight(500),
    )
    base = placed(instance_of("base", length=1000, width=1000, height=10, weight=Weight(400)))
    heavy = instance_of("heavy", length=1000, width=1000, height=10, weight=Weight(150))
    result = TopLoadConstraint().evaluate(context(heavy, z=160_000, placements=(base,), container=dense_box))
    assert not result.allowed and result.code == "stack_density_exceeded"


def test_a_density_limited_container_allows_a_candidate_within_the_limit():
    dense_box = Container.create(
        "dense", Dimensions(Length(ONE_METRE_TICKS), Length(ONE_METRE_TICKS), Length(ONE_METRE_TICKS)),
        max_stack_density=Weight(500),
    )
    base = placed(instance_of("base", length=1000, width=1000, height=10, weight=Weight(400)))
    light = instance_of("light", length=1000, width=1000, height=10, weight=Weight(100))
    result = TopLoadConstraint().evaluate(context(light, z=160_000, placements=(base,), container=dense_box))
    assert result.allowed


# ------------------------------------------------------------------------ floor

def test_floor_only_items_are_kept_on_the_floor():
    grounded = instance_of("g", must_be_on_floor=True)
    assert FloorConstraint().evaluate(context(grounded, z=0)).allowed
    result = FloorConstraint().evaluate(context(grounded, z=160_000))
    assert not result.allowed and result.code == "must_be_on_floor"


def test_ordinary_items_may_rest_at_any_height():
    assert FloorConstraint().evaluate(context(instance_of("a"), z=160_000)).allowed


# ----------------------------------------------------------------- compatibility

def test_incompatibility_is_refused_in_both_directions():
    """Declared on either side; a candidate must not have to repeat the neighbour's rule."""
    food = instance_of("food", tags=["food"])
    chemical = instance_of("chem", incompatible_tags=["food"])

    outward = CompatibilityConstraint().evaluate(context(chemical, placements=(placed(food),)))
    inward = CompatibilityConstraint().evaluate(context(food, placements=(placed(chemical),)))
    assert not outward.allowed and outward.code == "incompatible_items"
    assert not inward.allowed and inward.code == "incompatible_items"


def test_untagged_items_share_a_container_freely():
    assert CompatibilityConstraint().evaluate(
        context(instance_of("a"), placements=(placed(instance_of("b")),))
    ).allowed


def test_unrelated_tags_do_not_conflict():
    assert CompatibilityConstraint().evaluate(
        context(instance_of("a", tags=["dry"]), placements=(placed(instance_of("b", tags=["cold"])),))
    ).allowed


# ------------------------------------------------------------------- tag counts

def test_no_tag_limits_means_no_check():
    hazmat = Container.create("box", Dimensions.mm(100, 100, 100), tag_limits={})
    assert TagCountConstraint().evaluate(context(instance_of("a", tags=["hazmat"]), container=hazmat)).allowed


def test_a_third_item_of_a_limited_tag_is_refused():
    limited = Container.create("box", Dimensions.mm(100, 100, 100), tag_limits={"hazmat": 2})
    already_two = (placed(instance_of("a", tags=["hazmat"])), placed(instance_of("b", tags=["hazmat"])))
    result = TagCountConstraint().evaluate(context(instance_of("c", tags=["hazmat"]), container=limited, placements=already_two))
    assert not result.allowed and result.code == "tag_count_exceeded"


def test_the_second_item_of_a_limit_of_two_is_allowed():
    limited = Container.create("box", Dimensions.mm(100, 100, 100), tag_limits={"hazmat": 2})
    one_already = (placed(instance_of("a", tags=["hazmat"])),)
    assert TagCountConstraint().evaluate(context(instance_of("b", tags=["hazmat"]), container=limited, placements=one_already)).allowed


def test_an_untagged_item_is_never_limited():
    limited = Container.create("box", Dimensions.mm(100, 100, 100), tag_limits={"hazmat": 1})
    already_one = (placed(instance_of("a", tags=["hazmat"])),)
    assert TagCountConstraint().evaluate(context(instance_of("b"), container=limited, placements=already_one)).allowed


def test_the_limit_applies_only_to_the_tag_it_names():
    limited = Container.create("box", Dimensions.mm(100, 100, 100), tag_limits={"hazmat": 1})
    already_one = (placed(instance_of("a", tags=["hazmat"])),)
    assert TagCountConstraint().evaluate(
        context(instance_of("b", tags=["fragile"]), container=limited, placements=already_one)
    ).allowed


# ------------------------------------------------------- container eligibility

def test_an_item_with_no_eligibility_tags_may_go_anywhere():
    assert ContainerEligibilityConstraint().evaluate(context(instance_of("a"))).allowed


def test_an_ineligible_container_is_refused():
    refrigerated = Container.create("reefer", Dimensions.mm(100, 100, 100), tags=["refrigerated"])
    item = instance_of("perishable", eligible_container_tags=["refrigerated"])
    ordinary = Container.create("dry-van", Dimensions.mm(100, 100, 100))

    result = ContainerEligibilityConstraint().evaluate(context(item, container=ordinary))
    assert not result.allowed and result.code == "container_ineligible"
    assert ContainerEligibilityConstraint().evaluate(context(item, container=refrigerated)).allowed


# ---------------------------------------------------------------------- support

def test_anything_on_the_floor_is_fully_supported():
    assert SupportConstraint(1.0).evaluate(context(instance_of("a"), z=0)).allowed


def test_no_required_ratio_means_no_check():
    airborne = instance_of("a")
    assert SupportConstraint(0.0).evaluate(context(airborne, z=160_000)).allowed


def test_support_no_op_paths_never_walk_the_stack(monkeypatch):
    """Floor contact and inactive above-floor rules are O(1), independent of stack size.

    Guards the one call that makes the check O(stack): a patch aimed anywhere else
    passes whether or not the fast paths hold.
    """
    def unexpected_stack_walk(placements, item, box):
        raise AssertionError(f"inactive support check walked {len(placements)} placement(s)")

    monkeypatch.setattr(constraints, "direct_support_view", unexpected_stack_walk)
    floor_only = instance_of("floor", minimum_support_ratio=1.0, ground_contact_rule="multiple")
    unconstrained = instance_of("unconstrained")
    explicitly_free = instance_of("free", ground_contact_rule="free")

    assert SupportConstraint(1.0).evaluate(context(floor_only, z=0)).allowed
    assert SupportConstraint(0.0).evaluate(context(unconstrained, z=160_000)).allowed
    assert SupportConstraint(0.0).evaluate(context(explicitly_free, z=160_000)).allowed


@pytest.mark.parametrize(
    ("supporter_width_ticks", "allowed"),
    [(1_200_000, True), (1_199_999, False)],
)
def test_the_support_boundary_is_exact_to_the_tick(supporter_width_ticks, allowed):
    """A footprint of 75% is enough for a 0.75 requirement; one tick less is not."""
    shelf = Item.create("shelf", Dimensions(Length(supporter_width_ticks), Length(1_600_000), Length(160_000)))
    shelf_instance, = shelf.instances()
    candidate = instance_of("c", 100, 100, 10, minimum_support_ratio=0.75)

    result = SupportConstraint(0.0).evaluate(
        context(candidate, z=160_000, placements=(placed(shelf_instance),))
    )
    assert result.allowed is allowed
    if not allowed:
        assert result.code == "insufficient_support"


def test_the_stricter_of_the_global_and_per_item_ratio_wins():
    shelf, = Item.create("shelf", Dimensions.mm(60, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10)
    resting = context(candidate, z=160_000, placements=(placed(shelf),))

    assert SupportConstraint(0.5).evaluate(resting).allowed
    assert not SupportConstraint(0.75).evaluate(resting).allowed


def test_only_surfaces_at_the_resting_height_count_as_support():
    """A box two levels down carries the stack, but it is not what the candidate rests on."""
    low, = Item.create("low", Dimensions.mm(100, 100, 5)).instances()
    candidate = instance_of("c", 100, 100, 10, minimum_support_ratio=0.5)
    result = SupportConstraint(0.0).evaluate(context(candidate, z=160_000, placements=(placed(low),)))
    assert not result.allowed and result.code == "insufficient_support"


# --------------------------------------------------------- support polygon

def test_a_candidate_that_meets_the_ratio_but_overhangs_its_centroid_still_tips():
    """Area ratio alone is not stability: a 40% overlap on one side clears a 0.3
    ratio requirement while leaving the candidate's own centroid unsupported."""
    shelf, = Item.create("shelf", Dimensions.mm(40, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10, minimum_support_ratio=0.3)
    result = SupportConstraint(0.0).evaluate(context(candidate, z=160_000, placements=(placed(shelf),)))
    assert not result.allowed and result.code == "centre_of_gravity_unsupported"


def test_two_supporters_bracketing_the_centroid_are_allowed():
    """Neither rail alone reaches the middle, but together their hull spans across
    the candidate's centroid -- the union, not any single supporter, is what matters."""
    left, = Item.create("left", Dimensions.mm(10, 100, 10)).instances()
    right, = Item.create("right", Dimensions.mm(10, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10, minimum_support_ratio=0.15)
    result = SupportConstraint(0.0).evaluate(
        context(candidate, z=160_000, placements=(placed(left), placed(right, x=1_440_000)))
    )
    assert result.allowed


def test_no_tipping_check_runs_when_no_support_ratio_is_required():
    """No new rejection code appears for a caller who never asked for support
    checking -- the tipping check shares the ratio check's gate, not a new one."""
    shelf, = Item.create("shelf", Dimensions.mm(40, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10)
    result = SupportConstraint(0.0).evaluate(context(candidate, z=160_000, placements=(placed(shelf),)))
    assert result.allowed


# ------------------------------------------------------- ground-contact rules

def test_a_ratio_the_corner_rule_rejects_that_a_ratio_accepts():
    """A ratio cannot express the corner rule: 64% coverage concentrated in the
    middle clears a 50% requirement but touches none of the four base corners."""
    plate, = Item.create("plate", Dimensions.mm(80, 80, 10)).instances()

    ratio_only = instance_of("ratio-only", 100, 100, 10, minimum_support_ratio=0.5)
    assert SupportConstraint(0.0).evaluate(
        context(ratio_only, z=160_000, placements=(placed(plate, x=160_000, y=160_000),))
    ).allowed

    corner_checked = instance_of("corner-checked", 100, 100, 10, minimum_support_ratio=0.5, ground_contact_rule="covered")
    result = SupportConstraint(0.0).evaluate(
        context(corner_checked, z=160_000, placements=(placed(plate, x=160_000, y=160_000),))
    )
    assert not result.allowed and result.code == "ground_contact_violation"


def test_ground_contact_rejection_precedes_support_ratio_rejection():
    """Moving the no-op guard ahead of surface discovery must not reorder failures."""
    candidate = instance_of(
        "strict", 100, 100, 10,
        minimum_support_ratio=1.0,
        ground_contact_rule="covered",
    )

    result = SupportConstraint(0.0).evaluate(context(candidate, z=160_000))

    assert not result.allowed
    assert result.code == "ground_contact_violation"


def test_free_never_checks_ground_contact():
    airborne = instance_of("a", ground_contact_rule="free")
    assert SupportConstraint(0.0).evaluate(context(airborne, z=160_000)).allowed


def test_covered_is_satisfied_when_supporters_cover_all_four_corners():
    left, = Item.create("left", Dimensions.mm(50, 100, 10)).instances()
    right, = Item.create("right", Dimensions.mm(50, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10, ground_contact_rule="covered")
    resting = context(candidate, z=160_000, placements=(
        placed(left, x=0, y=0), placed(right, x=800_000, y=0),
    ))
    assert SupportConstraint(0.0).evaluate(resting).allowed


def test_single_rejects_a_candidate_split_across_two_supporters():
    left, = Item.create("left", Dimensions.mm(50, 100, 10)).instances()
    right, = Item.create("right", Dimensions.mm(50, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10, ground_contact_rule="single")
    resting = context(candidate, z=160_000, placements=(
        placed(left, x=0, y=0), placed(right, x=800_000, y=0),
    ))
    result = SupportConstraint(0.0).evaluate(resting)
    assert not result.allowed and result.code == "ground_contact_violation"


def test_single_accepts_a_candidate_resting_on_exactly_one_supporter():
    base, = Item.create("base", Dimensions.mm(100, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10, ground_contact_rule="single")
    assert SupportConstraint(0.0).evaluate(
        context(candidate, z=160_000, placements=(placed(base),))
    ).allowed


def test_multiple_rejects_a_candidate_resting_on_exactly_one_supporter():
    base, = Item.create("base", Dimensions.mm(100, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10, ground_contact_rule="multiple")
    result = SupportConstraint(0.0).evaluate(context(candidate, z=160_000, placements=(placed(base),)))
    assert not result.allowed and result.code == "ground_contact_violation"


def test_multiple_accepts_a_candidate_split_across_two_supporters():
    left, = Item.create("left", Dimensions.mm(50, 100, 10)).instances()
    right, = Item.create("right", Dimensions.mm(50, 100, 10)).instances()
    candidate = instance_of("c", 100, 100, 10, ground_contact_rule="multiple")
    resting = context(candidate, z=160_000, placements=(
        placed(left, x=0, y=0), placed(right, x=800_000, y=0),
    ))
    assert SupportConstraint(0.0).evaluate(resting).allowed


def test_a_floor_resting_item_satisfies_every_ground_contact_rule():
    for rule in ("free", "covered", "single", "multiple"):
        candidate = instance_of("c", 100, 100, 10, ground_contact_rule=rule)
        assert SupportConstraint(0.0).evaluate(context(candidate, z=0)).allowed, rule


# --------------------------------------------------------------------- bearing

def test_nothing_may_be_stacked_on_a_non_stackable_item():
    base = instance_of("base", stackable=False)
    result = TopLoadConstraint().evaluate(context(instance_of("c"), z=160_000, placements=(placed(base),)))
    assert not result.allowed and result.code == "non_stackable"


def test_a_non_stackable_item_may_not_be_slid_underneath_another():
    """Stacking is a relation, not a direction. Rejecting only the downward case let a
    solver reach the same forbidden arrangement by placing the two boxes in the other
    order."""
    above = placed(instance_of("above"), z=160_000)
    sliding_under = instance_of("under", 10, 10, 10, stackable=False)
    result = TopLoadConstraint().evaluate(context(sliding_under, z=0, placements=(above,)))
    assert not result.allowed and result.code == "non_stackable"


def test_a_non_stackable_item_beside_another_is_fine():
    beside = placed(instance_of("beside"), x=160_000)
    assert TopLoadConstraint().evaluate(
        context(instance_of("a", stackable=False), placements=(beside,))
    ).allowed


def test_a_candidate_that_would_crush_the_stack_below_is_refused():
    base = placed(instance_of("base", weight="1 kg", max_top_load="1.5 kg"))
    heavy = instance_of("heavy", weight="2 kg")
    result = TopLoadConstraint().evaluate(context(heavy, z=160_000, placements=(base,)))
    assert not result.allowed and result.code == "top_load_exceeded"


def test_a_nested_candidate_carries_its_load_through_the_nested_column():
    base, middle, top = Item.create(
        "crate",
        Dimensions.mm(100, 100, 100),
        "1 kg",
        quantity=3,
        max_top_load="1.5 kg",
        nesting_height=Length.mm(40),
    ).instances()
    step = Length.mm(60).ticks
    result = TopLoadConstraint().evaluate(context(
        top,
        z=2 * step,
        placements=(placed(base), placed(middle, z=step)),
    ))

    assert not result.allowed and result.code == "top_load_exceeded"


def test_a_candidate_within_the_bearing_limit_is_allowed():
    base = placed(instance_of("base", weight="1 kg", max_top_load="2 kg"))
    assert TopLoadConstraint().evaluate(
        context(instance_of("light", weight="1 kg"), z=160_000, placements=(base,))
    ).allowed


def test_the_bearing_check_is_skipped_when_nothing_can_refuse_a_load():
    """A whole-stack walk per candidate dominated the search on large orders, so it is
    skipped when no item in play declares a limit or refuses to be stacked on."""
    base = placed(instance_of("base", stackable=False))
    assert TopLoadConstraint().evaluate(
        context(instance_of("c"), z=160_000, placements=(base,), stack_sensitive=False)
    ).allowed


# ------------------------------------------------------------- axle load

def axled_box(front_mm: int, rear_mm: int, front_limit=None, rear_limit=None, length_mm=1000) -> Container:
    return Container.create(
        "axled", Dimensions.mm(length_mm, 100, 100),
        axles=(Axle(Length.mm(front_mm), front_limit), Axle(Length.mm(rear_mm), rear_limit)),
    )


def test_no_axles_means_no_check_to_run():
    plain = Container.create("plain", Dimensions.mm(100, 100, 100))
    assert AxleLoadConstraint().evaluate(context(instance_of("c", weight="1000 kg"), container=plain)).allowed


def test_a_candidate_centred_between_the_axles_splits_evenly():
    # A candidate spanning the whole container has its own centre at 500mm, exactly
    # midway between axles at 100mm and 900mm -- each axle bears half of 800kg.
    box = axled_box(100, 900, Weight.of(500, "kg"), Weight.of(500, "kg"))
    candidate = instance_of("c", 1000, 100, 100, weight="800 kg")
    assert AxleLoadConstraint().evaluate(context(candidate, container=box)).allowed


def test_a_candidate_that_would_overload_the_front_axle_is_refused():
    box = axled_box(100, 900, Weight.of(399, "kg"), Weight.of(500, "kg"))
    candidate = instance_of("c", 1000, 100, 100, weight="800 kg")
    result = AxleLoadConstraint().evaluate(context(candidate, container=box))
    assert not result.allowed and result.code == "axle_overloaded"


def test_an_existing_placement_and_a_new_candidate_are_both_weighed():
    # Two 400kg items, each spanning the whole container (so each alone splits
    # 200/200 across axles at 100mm/900mm) -- together they still split evenly,
    # and the check must see both, not only the new candidate.
    box = axled_box(100, 900, Weight.of(400, "kg"), Weight.of(400, "kg"))
    already_there = placed(instance_of("base", 1000, 100, 100, weight="400 kg"))
    candidate = instance_of("c", 1000, 100, 100, weight="400 kg")
    assert AxleLoadConstraint().evaluate(context(candidate, placements=(already_there,), container=box)).allowed


# ------------------------------------------------- route unloading order

def test_a_later_stop_resting_on_an_earlier_one_is_refused():
    base = placed(instance_of("first-stop", stop_index=0))
    later = instance_of("last-stop", stop_index=1)
    result = RouteOrderConstraint().evaluate(context(later, z=160_000, placements=(base,)))
    assert not result.allowed and result.code == "unloading_order_violation"


def test_an_earlier_stop_resting_on_a_later_one_is_allowed():
    """The whole point of the rule: stop 0 belongs on top so it comes off first."""
    base = placed(instance_of("last-stop", stop_index=1))
    earlier = instance_of("first-stop", stop_index=0)
    assert RouteOrderConstraint().evaluate(context(earlier, z=160_000, placements=(base,))).allowed


def test_items_due_at_the_same_stop_may_stack_freely():
    base = placed(instance_of("a", stop_index=2))
    same = instance_of("b", stop_index=2)
    assert RouteOrderConstraint().evaluate(context(same, z=160_000, placements=(base,))).allowed


def test_the_rule_reaches_through_an_intermediary():
    """Transitive, not merely direct: burying stop 0 under two levels still buries it."""
    base = placed(instance_of("first-stop", stop_index=0))
    middle = placed(instance_of("same-stop", stop_index=0), z=160_000)
    later = instance_of("last-stop", stop_index=1)
    result = RouteOrderConstraint().evaluate(context(later, z=320_000, placements=(base, middle)))
    assert not result.allowed and result.code == "unloading_order_violation"


def test_an_item_with_no_stop_blocks_a_routed_one_beneath_it():
    """A placement with no stop_index is never scheduled for removal, so it stays put
    for every stop -- exactly what the shared validator treats it as."""
    base = placed(instance_of("first-stop", stop_index=0))
    rider = instance_of("fixture")
    result = RouteOrderConstraint().evaluate(context(rider, z=160_000, placements=(base,)))
    assert not result.allowed and result.code == "unloading_order_violation"


def test_nothing_blocks_an_item_that_rides_the_whole_route():
    base = placed(instance_of("fixture"))
    routed = instance_of("last-stop", stop_index=1)
    assert RouteOrderConstraint().evaluate(context(routed, z=160_000, placements=(base,))).allowed


def test_a_request_with_no_route_never_pays_for_the_walk():
    base = placed(instance_of("a"))
    other = instance_of("b")
    assert RouteOrderConstraint().evaluate(
        context(other, z=160_000, placements=(base,), route_sensitive=False)).allowed


def test_side_by_side_items_do_not_constrain_each_other():
    base = placed(instance_of("first-stop", stop_index=0))
    beside = instance_of("last-stop", stop_index=1)
    assert RouteOrderConstraint().evaluate(context(beside, x=160_000, placements=(base,))).allowed


def test_an_unrouted_column_is_reported_as_no_violation():
    column = (unit(0, 0, 0, 10, 10, 10, label="base"), unit(0, 0, 10, 10, 10, 10, label="top"))
    assert route_order_violated(column, [RIDES_THE_WHOLE_ROUTE, RIDES_THE_WHOLE_ROUTE]) is None


def test_the_blocked_item_and_its_blocker_are_both_named():
    column = (unit(0, 0, 0, 10, 10, 10, label="base"), unit(0, 0, 10, 10, 10, 10, label="top"))
    code, detail = route_order_violated(column, [0.0, 1.0])
    assert code == "unloading_order_violation"
    assert "base" in detail and "top" in detail


# ------------------------------------------------- stop accessibility
#
# The worked examples in docs/STOP-ACCESSIBILITY.md were derived by hand and confirmed
# against the shipped whole-scene replay, and they are the acceptance criterion for this
# constraint. They live in `conformance/scene/stop-accessibility-fixtures.json` and are
# read from there below; the helpers here serve the degenerate and hostile inputs further
# down, which are specific to this engine and have no cross-language counterpart.

DOOR_AT_MINUS_X = ("-x",)
ALL_DIRECTIONS_TUPLE = ("+x", "-x", "+y", "-y", "+z", "-z")


def _mm(millimetres: int) -> int:
    """The examples are written in millimetres and positions are taken in ticks, so the
    conversion is stated once here rather than at every call site."""
    return Length.mm(millimetres).ticks


def _wide(id: str, x: int, length: int, stop=None):
    """A box spanning the container's full width and height, so a corridor it stands in is
    completely filled -- the examples turn on which *stop* blocks which, not on squeezing
    past."""
    instance = instance_of(id, length=length, width=100, height=100, stop_index=stop)
    return instance, _mm(x)


def _scene(first, second, directions=DOOR_AT_MINUS_X):
    """Place `first`, then offer `second` as the candidate."""
    (placed_instance, placed_x), (candidate, candidate_x) = first, second
    constraint = constraints.StopAccessibilityConstraint(directions)
    return constraint.evaluate(context(
        candidate, x=candidate_x, placements=(placed(placed_instance, x=placed_x),),
        dimensions=candidate.item.dimensions,
    ))


#: The worked examples, held once for all four engines instead of transcribed into each.
#: Four copies of nine scenes is four chances for a verdict to drift in one engine and stay
#: green in the other three. A published copy of the package does not carry the corpus, so
#: the test that reads it skips rather than failing for everyone who installed the package.
STOP_SCENES = Path(__file__).parents[2] / "conformance/scene/stop-accessibility-fixtures.json"
requires_stop_scenes = pytest.mark.skipif(
    not STOP_SCENES.is_file(),
    reason="the shared cross-language scene corpus is not part of this package",
)


def _fixture_dimensions(raw) -> Dimensions:
    """The corpus is in ticks, so it is read straight rather than through `Dimensions.mm`:
    all four engines assert the same integers instead of each scaling by its own factor."""
    return Dimensions(*(Length(raw[axis]) for axis in ("length", "width", "height")))


def _fixture_placement(raw) -> Placement:
    one, = Item.create(raw["id"], _fixture_dimensions(raw["dimensions"]),
                       stop_index=raw["stop_index"]).instances()
    return placed(one, *(raw["origin"][axis] for axis in ("x", "y", "z")))


@requires_stop_scenes
def test_shared_four_language_stop_accessibility_scenes():
    """Every scene in the shared corpus, with the verdict every engine must reach.

    `accessible` is asserted by all four engines. `code` is asserted here and in PHP, whose
    constraint returns a reason rather than a boolean, and `route_order_allowed` here, in
    PHP and in Rust -- the corpus records that asymmetry so it is not rediscovered.
    """
    payload = json.loads(STOP_SCENES.read_text())
    assert payload["scenes"], "an empty corpus would pass this loop without asserting anything"
    for scene in payload["scenes"]:
        raw = scene["candidate"]
        one, = Item.create(raw["id"], _fixture_dimensions(raw["dimensions"]),
                           stop_index=raw["stop_index"]).instances()
        evaluated = context(
            one,
            *(raw["origin"][axis] for axis in ("x", "y", "z")),
            placements=tuple(_fixture_placement(each) for each in scene["placements"]),
            container=Container.create("fixture", _fixture_dimensions(scene["container"])),
        )
        result = constraints.StopAccessibilityConstraint(tuple(scene["directions"])).evaluate(evaluated)

        assert result.allowed == scene["accessible"], scene["id"]
        if "code" in scene:
            assert result.code == scene["code"], scene["id"]
        if "route_order_allowed" in scene:
            assert (RouteOrderConstraint().evaluate(evaluated).allowed
                    == scene["route_order_allowed"]), scene["id"]


def test_a_request_with_no_route_is_untouched():
    """`route_sensitive` is False whenever no item in play declares a stop, which is the
    same opt-in gate `RouteOrderConstraint` uses."""
    instance = instance_of("plain", length=60, width=100, height=100)
    scene = context(instance, x=0, placements=(placed(instance_of(
        "other", length=40, width=100, height=100), x=_mm(60)),), route_sensitive=False)
    assert constraints.StopAccessibilityConstraint(DOOR_AT_MINUS_X).evaluate(scene).allowed


def test_directions_are_canonicalised_so_two_callers_search_identically():
    """Order and duplicates in the caller's list must not reach the search: which door is
    tried first decides which of several legal answers comes back."""
    assert (constraints.StopAccessibilityConstraint(("+z", "-x", "-x"))._directions
            == constraints.StopAccessibilityConstraint(("-x", "+z"))._directions)


def test_swept_volume_refuses_an_unknown_direction_at_the_primitive():
    """The constraint validates its doors at construction, but the primitive is public and
    reachable on its own -- `packing_sequence` calls it -- so it owes the same refusal."""
    from packvium.geometry import InvalidDirectionError, swept_volume

    box = AxisAlignedBox(Point(0, 0, 0), Dimensions(Length(10), Length(10), Length(10)))
    with pytest.raises(InvalidDirectionError):
        swept_volume(box, Dimensions(Length(100), Length(100), Length(100)), "sideways")


def test_the_corridor_base_is_reused_for_a_second_candidate_on_the_same_state():
    """The cache is why a candidate costs `O(m * |D|)` rather than `O(m^2 * |D|)`: search
    asks a run of candidates against one immutable state, so one entry covers the run.

    It is keyed on the placements, the container *and* the doors. The container half is a
    guard rather than something a legal scene can demonstrate -- a placement always lies
    inside its own container, so widening a wall only lengthens a sweep into empty space.
    What it protects against is one placement tuple being asked about two different
    containers, which a multi-container solve can do; the key makes the second question
    rebuild instead of inheriting the first answer.

    The doors half is not a guard at all since made them a property of the
    container: two containers of the same size with different doors give *different*
    answers for the same boxes, and nothing else in the key separates them.
    """
    constraint = constraints.StopAccessibilityConstraint(DOOR_AT_MINUS_X)
    early, early_x = _wide("early", 60, 40, stop=0)
    late, late_x = _wide("late", 0, 60, stop=1)
    placements = (placed(early, x=early_x),)
    scene = context(late, x=late_x, placements=placements)
    doors = tuple(DOOR_AT_MINUS_X)

    first = constraint.evaluate(scene)
    built = constraint._base_for(placements, BOX.inner_dimensions, doors)
    # Asking again on the same state must take the cached path and answer identically.
    assert constraint._base_for(placements, BOX.inner_dimensions, doors) == built
    assert constraint.evaluate(scene).allowed == first.allowed

    longer = Container.create("longer", Dimensions.mm(300, 100, 100))
    constraint._base_for(placements, longer.inner_dimensions, doors)
    assert constraint._container == longer.inner_dimensions, (
        "a different container must rebuild the base rather than inherit it")

    # The same boxes and the same walls, through the other door: a different answer, and
    # the entry above must not be handed back for it.
    through_plus_x = constraint._base_for(placements, BOX.inner_dimensions, ("+x",))
    assert constraint._directions == ("+x",)
    assert through_plus_x != built, (
        "the same placements behind two different doors are two different questions")


def test_a_container_states_its_own_doors_and_a_silent_one_inherits_the_default():
    """ . The field is per container because two doors on one trailer and none on
    another is the case that makes the rule worth having; the constructor argument stays as
    the default so the library callers who predate the field keep working."""
    early, early_x = _wide("early", 60, 40, stop=0)
    late, late_x = _wide("late", 0, 60, stop=1)
    placements = (placed(early, x=early_x),)

    sealed = Container.create("sealed", BOX.inner_dimensions)
    through_minus_x = Container.create("through-minus-x", BOX.inner_dimensions,
                                       access_directions=("-x",))
    through_plus_x = Container.create("through-plus-x", BOX.inner_dimensions,
                                      access_directions=("+x",))

    def verdict(constraint, container):
        return constraint.evaluate(
            context(late, x=late_x, placements=placements, container=container)).allowed

    # The stop-1 item fills the stop-0 item's only corridor to `-x`, and does not touch its
    # corridor to `+x`. One container refuses it, the other does not, and nothing about the
    # boxes changed.
    stated = constraints.StopAccessibilityConstraint()
    assert not verdict(stated, through_minus_x)
    assert verdict(stated, through_plus_x)
    assert verdict(stated, sealed), "no doors anywhere leaves the rule inert"

    # A container that states none inherits what the caller configured.
    configured = constraints.StopAccessibilityConstraint(DOOR_AT_MINUS_X)
    assert not verdict(configured, sealed)
    # And a container that states its own overrides it, rather than adding to it.
    assert verdict(configured, through_plus_x)


def test_permanent_cargo_that_blocks_nobody_is_allowed():
    """The `rides the whole route` short-circuit after the placed-item loop. An item with no
    stop needs no door of its own, so once it has taken nobody else's it is simply legal --
    the counterpart to example D, where it took one."""
    fixture = instance_of("fixture", length=40, width=100, height=100, stop_index=None)
    early = instance_of("early", length=60, width=100, height=100, stop_index=0)
    # The stop-0 item stands at the door; the permanent one sits behind it and blocks
    # nothing, because a corridor to `-x` never crosses it.
    scene = context(fixture, x=_mm(60), placements=(placed(early, x=0),))
    assert constraints.StopAccessibilityConstraint(DOOR_AT_MINUS_X).evaluate(scene).allowed


def test_an_unknown_direction_is_refused_rather_than_guessed():
    with pytest.raises(constraints.InvalidDirectionError):
        constraints.StopAccessibilityConstraint(("north",))


# ------------------------------------- stop accessibility: degenerate and hostile input


def test_a_corridor_runs_to_a_wall_so_the_container_is_part_of_the_question():
    """The cached exit sets must not outlive the container they were computed for.

    `+x` ends at the container's far wall, so the same two boxes have different exits in a
    short container than in a long one: in the short one `a` is flush against the wall and
    free, in the long one a later-stop box already stands in its corridor. A cache keyed on
    the placements alone answered the second question with the first one's answer and
    accepted a placement that walls `a` in.
    """
    long_box = Container.create("long", Dimensions.mm(400, 100, 100))
    short_box = Container.create("short", Dimensions.mm(100, 100, 100))
    near = instance_of("a", length=40, width=100, height=100, stop_index=0)
    far = instance_of("b", length=50, width=100, height=100, stop_index=1)
    placements = (placed(near, x=0), placed(far, x=_mm(150)))
    candidate = instance_of("c", length=30, width=100, height=100, stop_index=1)
    constraint = constraints.StopAccessibilityConstraint(("+x",))

    def verdict(container):
        return constraint.evaluate(ConstraintContext(
            container, placements, candidate, Point(_mm(50), 0, 0), Rotation.LWH,
            candidate.item.dimensions, candidate.item.dimensions)).allowed

    assert verdict(long_box), "a had already lost its corridor before the candidate"
    assert not verdict(short_box), "the candidate fills a's only corridor here"


def test_the_largest_admissible_stops_stay_distinct():
    """Exactness at the top of the range the wire contract admits.

    An earlier version of this test used 2**53 and 2**53 + 1 to show the constraint kept
    them apart where a float would merge them. Those values are now refused at
    construction, because JavaScript cannot parse them without collapsing them and the
    four engines would order such a load differently. The hazard is gone at its source, so
    what is left to prove is that nothing widens a stop *inside* the admissible range --
    the two largest neighbours it contains are still two.
    """
    from packvium.models import MAX_EXACT_STOP_INDEX

    early = instance_of("early", length=40, width=100, height=100,
                        stop_index=MAX_EXACT_STOP_INDEX - 1)
    late = instance_of("late", length=60, width=100, height=100,
                       stop_index=MAX_EXACT_STOP_INDEX)

    result = constraints.StopAccessibilityConstraint(DOOR_AT_MINUS_X).evaluate(context(
        late, x=0, placements=(placed(early, x=_mm(60)),)))

    assert not result.allowed
    assert str(MAX_EXACT_STOP_INDEX - 1) in result.detail


def test_two_permanent_items_do_not_block_each_other():
    """Neither is ever unloaded, so neither needs a corridor and neither is the other's
    problem. `inf > inf` being false is what gives that for free -- an ordering sentinel
    that compared greater-or-equal to itself would refuse every pair of fixtures."""
    assert _scene(_wide("fixture-a", 60, 40, stop=None),
                  _wide("fixture-b", 0, 60, stop=None)).allowed


def test_a_permanent_item_needs_no_exit_of_its_own():
    """Nothing is due later than "never", so its blocker set is empty by construction.

    The fixture sits at the far end with a stop-1 item between it and the door, which for
    any ordinary item would be a refusal. It is not one here: the fixture is not coming
    out at stop 1, or at any stop.
    """
    assert _scene(_wide("fixture", 60, 40, stop=None), _wide("late", 0, 60, stop=1)).allowed


def test_permanent_cargo_still_walls_in_an_item_that_does_have_to_come_out():
    """The converse, and the asymmetry is the point.

    An item due at stop 5 behind permanent cargo is refused, because "never" outranks
    every stop. Pairing this with the test above pins the sentinel's direction: it is the
    latest possible stop, not a value excused from the ordering.
    """
    result = _scene(_wide("fixture", 0, 60, stop=None), _wide("late", 60, 40, stop=5))
    assert not result.allowed
    assert "no exit" in result.detail


def test_the_first_box_into_an_empty_container_is_never_refused():
    """There is nothing to be blocked by and nothing to block, whatever its stop."""
    instance = instance_of("only", length=60, width=100, height=100, stop_index=3)
    assert constraints.StopAccessibilityConstraint(DOOR_AT_MINUS_X).evaluate(
        context(instance, x=0)).allowed


def test_a_box_flush_against_a_corridor_wall_does_not_stand_in_it():
    """Half-open on every axis, matching `AxisAlignedBox.intersects`.

    Two boxes side by side across the width: the `-x` corridor of one spans only its own
    `y` band, so its neighbour merely touching that band's edge is not in the way. Treating
    a shared face as an obstruction would refuse most ordinary side-by-side loads.
    """
    left = instance_of("left", length=40, width=50, height=100, stop_index=0)
    right = instance_of("right", length=40, width=50, height=100, stop_index=1)
    scene = ConstraintContext(
        BOX, (placed(left, x=_mm(60), y=0),), right, Point(_mm(60), _mm(50), 0),
        Rotation.LWH, right.item.dimensions, right.item.dimensions)
    assert constraints.StopAccessibilityConstraint(DOOR_AT_MINUS_X).evaluate(scene).allowed


def test_a_box_filling_the_container_has_an_empty_corridor_in_every_direction():
    """Its faces are the walls, so no sweep has room for anything -- including the sweep
    of the item that fills it. A rule that measured the corridor from the container's
    centre, or that treated an empty region as blocked, would refuse a single-item load."""
    whole = instance_of("whole", length=100, width=100, height=100, stop_index=0)
    for directions in (("-x",), ("+x",), ALL_DIRECTIONS_TUPLE):
        assert constraints.StopAccessibilityConstraint(directions).evaluate(
            context(whole, x=0)).allowed


def test_stop_zero_is_a_stop_and_not_an_absent_value():
    """`0` is falsy, and a presence test written as `if stop:` would quietly turn the
    first stop on the route into permanent cargo -- which reverses the rule for it."""
    result = _scene(_wide("early", 60, 40, stop=0), _wide("late", 0, 60, stop=1))
    assert not result.allowed
    assert "stop 0" in result.detail


def test_all_six_doors_accept_what_one_door_refuses():
    """The vacuity the design warns about, pinned so the default cannot drift into it.

    With every wall open a box is almost always free through some face, which is precisely
    why `access_directions` defaults to empty rather than to all six.
    """
    assert _scene(_wide("early", 60, 40, stop=0), _wide("late", 0, 60, stop=1),
                  directions=ALL_DIRECTIONS_TUPLE).allowed


# ------------------------------- the support-polygon redundancy boundary


@pytest.mark.parametrize("seed", range(80))
def test_a_single_supporter_covering_over_half_the_base_always_contains_the_centroid(seed):
    """Why `SupportConstraint`'s polygon test cannot fire above a 0.5 area ratio.

    On one rectangular supporter the contact region is a rectangle inside the base. If it
    covers more than half of each axis it must straddle the midpoint of that axis, so the
    centroid is inside it and the polygon test is decided before it is asked. Measured
    behaviour matches: at ratio 0.6 the conjunction refuses exactly what the area rule
    refuses, and at 0.3 and 0.45 it refuses more (benchmarks/results/support-predicates.json).

    This is the boundary, not a bug -- but it means the hull is built per candidate for a
    verdict that a cheaper comparison already fixed, which is the finding records.
    """
    rng = random.Random(6000 + seed)
    length, width = rng.randint(20, 60), rng.randint(20, 60)
    lower = instance_of("lower", length=100, width=100, height=10)
    upper = instance_of("upper", length=length, width=width, height=10)

    # A contact rectangle covering strictly more than half of the candidate on both axes.
    overlap_l = rng.randint(length // 2 + 1, length)
    overlap_w = rng.randint(width // 2 + 1, width)
    offset_x = rng.randint(0, length - overlap_l)
    offset_y = rng.randint(0, width - overlap_w)

    candidate = AxisAlignedBox(Point(_mm(offset_x), _mm(offset_y), _mm(10)),
                               Dimensions.mm(length, width, 10))
    supporter = AxisAlignedBox(Point(_mm(offset_x), _mm(offset_y), 0),
                               Dimensions.mm(overlap_l, overlap_w, 10))
    del lower, upper  # built only to mirror the shapes the constraint sees

    hull = constraints.convex_hull(constraints.contact_hull_points(candidate, [supporter]))
    assert constraints.point_in_hull(constraints.doubled_centroid(candidate), hull), (
        f"seed {seed}: {overlap_l}x{overlap_w} of {length}x{width} left the centroid outside"
    )


# -------------------------------------------- the representable stop range


def test_a_stop_index_past_double_precision_is_refused_rather_than_mis_ordered():
    """The bound exists because one engine cannot hold the number, not because of a limit.

    Route order is decided by comparing stop indices. JavaScript keeps numbers as doubles,
    and `JSON.parse` collapses 2**53 + 1 to 2**53 before any constraint sees it -- so two
    consecutive stops above the safe range become one number there while Python, PHP and
    Rust keep them apart, and the four engines order the same load differently. The
    JavaScript engine already refused unsafe integers; this is the other three agreeing.

    Refusing is the only honest option: the value cannot cross the wire identically, and
    accepting it would mean returning a load plan that another engine would contradict.
    """
    from packvium.models import MAX_EXACT_STOP_INDEX

    assert MAX_EXACT_STOP_INDEX == 2 ** 53 - 1
    Item.create("ok", Dimensions.mm(10, 10, 10), stop_index=MAX_EXACT_STOP_INDEX)

    for refused in (MAX_EXACT_STOP_INDEX + 1, 2 ** 53 + 1, -1):
        with pytest.raises(ValueError, match="non-negative safe integer"):
            Item.create("bad", Dimensions.mm(10, 10, 10), stop_index=refused)


def test_the_two_stops_that_collapse_into_one_are_exactly_the_pair_the_bound_excludes():
    """Names the failure the bound prevents, so the reason cannot be edited away.

    `float(2**53) == float(2**53 + 1)`, and the route rule compares stops. With both
    admitted, an item due at 2**53 + 1 resting on one due at 2**53 would be read as the
    same stop and allowed -- a later item burying an earlier one, which is the exact
    violation `RouteOrderConstraint` exists to catch.
    """
    assert float(2 ** 53) == float(2 ** 53 + 1)
    assert 2 ** 53 != 2 ** 53 + 1
    with pytest.raises(ValueError, match="non-negative safe integer"):
        Item.create("collapses", Dimensions.mm(10, 10, 10), stop_index=2 ** 53)
