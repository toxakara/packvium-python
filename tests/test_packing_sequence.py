"""Loading and unloading dependency graphs and their safe-order
simulations.

Audit reopen: the original module built one graph (children-dependency, full-scene
replay) and treated it as satisfying both "packing" and "unloading" -- it never
modelled an item depending on its *supporters* nor replayed from an empty container.
This suite tests both graphs, separately and against each other:

- Support dependencies are provably acyclic in both directions: an unloading edge only
  ever points to something with a strictly greater height (children), a loading edge
  only ever points to something with a strictly lesser height (supporters) -- either
  way, following edges is monotonic in height, so a finite set cannot loop back.
  `test_support_edges_are_provably_acyclic` checks both as a property, not a single
  example.
- Accessibility (the six-direction sweep) never deadlocks *this library's* own output
  when every direction is allowed: for unloading, the topmost item(s) always have a
  clear `+z` exit; for loading, the bottom-most item(s) always have a clear `+z`
  entry from above. Restricting the allowed directions is the only way to
  force a cycle in either graph.
- Unloading and loading are genuinely distinct, not mirror images computed once: a
  restricted `directions` set can make one solvable and the other not (see
  `test_a_single_allowed_direction_orders_by_distance_from_the_exit` and its loading
  counterpart), and an unknown direction string is rejected outright rather than
  silently treated as `-z`.
"""

from __future__ import annotations

import random
import json
from pathlib import Path

import pytest

from packvium import (AxisAlignedBox, Container, Dimensions, Item, Length, Placement, Point,
                         Rotation)
from packvium.packing_sequence import (
    InvalidDirectionError,
    LoadingDependencyGraph,
    RouteSequenceError,
    SequenceError,
    SequenceReplayError,
    SequenceStep,
    SequenceWarning,
    UnloadingDependencyGraph,
    placement_reachability,
    replay_loading_order,
    replay_removal_order,
    safe_loading_order,
    safe_loading_order_for_placements,
    safe_loading_order_with_evidence,
    safe_removal_order,
    safe_removal_order_with_evidence,
    safe_route_removal_order,
    verify_loading_prefix_business_rules,
)

#: A scene corpus shared with the other implementations of this library, kept one level
#: above this package. A published copy of the package does not carry it, so the two
#: tests that read it skip instead of failing for everyone who installed the package.
SHARED_SCENES = Path(__file__).parents[2] / "conformance/scene/sequence-fixtures.json"
requires_shared_scenes = pytest.mark.skipif(
    not SHARED_SCENES.is_file(),
    reason="the shared cross-language scene corpus is not part of this package",
)

MM = 16_000


def box(x, y, z, l, w, h) -> AxisAlignedBox:
    return AxisAlignedBox(Point(x * MM, y * MM, z * MM), Dimensions.mm(l, w, h))


# ------------------------------------------------------------------- unloading

def test_a_stacked_pair_requires_the_top_removed_first():
    container = Dimensions.mm(20, 20, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    graph = UnloadingDependencyGraph.build([bottom, top])
    assert graph.depends_on == (frozenset({1}), frozenset())
    assert safe_removal_order([bottom, top], container) == [1, 0]


def test_two_unstacked_items_have_no_unloading_dependency():
    container = Dimensions.mm(20, 10, 10)
    a, b = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    graph = UnloadingDependencyGraph.build([a, b])
    assert graph.depends_on == (frozenset(), frozenset())


def test_a_three_level_stack_must_come_off_top_to_bottom():
    container = Dimensions.mm(10, 10, 30)
    stack = [box(0, 0, level * 10, 10, 10, 10) for level in range(3)]
    assert safe_removal_order(stack, container) == [2, 1, 0]


def test_a_single_allowed_direction_orders_by_distance_from_the_exit():
    """Two side-by-side items, only one exit direction allowed (the LIFO shape):
    the one nearer the exit must come off first, and the order flips with the exit."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    assert safe_removal_order([near, far], container, directions=("-x",)) == [0, 1]
    assert safe_removal_order([near, far], container, directions=("+x",)) == [1, 0]


def test_an_empty_direction_set_is_a_clean_unloading_deadlock():
    container = Dimensions.mm(20, 20, 20)
    with pytest.raises(SequenceError) as excinfo:
        safe_removal_order([box(0, 0, 0, 10, 10, 10)], container, directions=())
    assert excinfo.value.stuck == frozenset({0})


def test_the_removal_order_is_deterministic_across_ties():
    container = Dimensions.mm(30, 10, 10)
    boxes = [box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10), box(20, 0, 0, 10, 10, 10)]
    first = safe_removal_order(boxes, container)
    for _ in range(5):
        assert safe_removal_order(boxes, container) == first


# ---------------------------------------------------------------------- route

def test_a_route_respecting_arrangement_unloads_stop_by_stop():
    """`near` (due at stop 0, the first stop) sits flush against the only door (`-x`);
    `far` (due at stop 1) sits behind it. LIFO holds: the earlier stop is nearer the
    exit, so it never needs `far` moved out of the way first."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    order = safe_route_removal_order([near, far], [0, 1], container, directions=("-x",))
    assert order == [0, 1]


def test_a_later_stop_item_blocking_an_earlier_stop_item_is_a_route_violation():
    """The arrangement flipped: `near` (flush against the only door) is due at stop 1,
    `far` (behind it) is due at stop 0 and must leave first -- but `far` cannot reach
    the door without `near`, which is not due to leave yet, moving first. This is
    exactly the property that separates cartonization from vehicle loading."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    with pytest.raises(RouteSequenceError) as excinfo:
        safe_route_removal_order([near, far], [1, 0], container, directions=("-x",))
    assert excinfo.value.stop == 0
    assert excinfo.value.stuck == frozenset({1})  # `far` (index 1) is the one stuck


def test_an_unrouted_placement_never_leaves_but_can_still_block():
    """A placement with no `stop_index` (a fixture riding the whole route) is never
    scheduled for removal, but still physically occupies its space -- if it happens to
    stand in a routed item's only exit, that is a genuine, permanent route violation."""
    container = Dimensions.mm(20, 10, 10)
    fixture, due = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    with pytest.raises(RouteSequenceError) as excinfo:
        safe_route_removal_order([fixture, due], [None, 0], container, directions=("-x",))
    assert excinfo.value.stop == 0
    assert excinfo.value.stuck == frozenset({1})


def test_a_second_escape_direction_saves_an_otherwise_blocked_stop():
    """The same blocked arrangement as the violation case above, but with `+z` also
    allowed: the usual engine scope (horizontal *or* vertical) is exactly what
    `safe_removal_order`'s `directions` sweep already models, so a clear vertical lift
    is enough even when the horizontal exit is not."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    order = safe_route_removal_order([near, far], [1, 0], container, directions=("-x", "+z"))
    assert order == [1, 0]  # far (stop 0, index 1) lifts straight out first


def test_two_placements_due_at_the_same_stop_may_block_each_other():
    """Within one stop the unloading order is free, so being in the way is not a violation.

    This is the case `docs/STOP-ACCESSIBILITY.md`'s per-candidate rule is deliberately
    optimistic about: its blocker set is `s(q) > s(p)`, strictly greater, and tightening it
    to `>=` would refuse this arrangement -- two pallets for the same delivery, one behind
    the other, which is an ordinary load rather than a defect. The rule is written against
    this test, so a future implementation that gets the comparison wrong fails here rather
    than in a customer's van.
    """
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    assert safe_route_removal_order([near, far], [0, 0], container, directions=("-x",)) == [0, 1]


def test_the_same_packing_can_be_legal_through_one_wall_and_illegal_through_another():
    """Route legality is a property of the packing *and* the door, and only one of the two
    is expressible in a request today.

    The arrangement below is what the solvers actually produce for a two-stop van load, and
    it unloads correctly through `-x` and not at all through `+x`. Nothing in the request
    schema names which wall the door is on -- `stop_index` is there and no access field is --
    so a solver enforcing route order has no way to know which of these two answers it is
    being asked for. `docs/STOP-ACCESSIBILITY.md` records that as the first thing its design
    needs and the reason the constraint cannot simply be switched on.
    """
    container = Dimensions.mm(100, 20, 20)
    early, late = box(0, 0, 0, 50, 20, 20), box(50, 0, 0, 50, 20, 20)
    assert safe_route_removal_order([early, late], [0, 1], container, directions=("-x",)) == [0, 1]
    with pytest.raises(RouteSequenceError) as excinfo:
        safe_route_removal_order([early, late], [0, 1], container, directions=("+x",))
    assert excinfo.value.stop == 0
    assert excinfo.value.stuck == frozenset({0})


def test_no_routed_placements_schedules_nothing():
    """Every existing single-stop request leaves `stop_index` unset on every item --
    this must be a complete no-op, not merely a small one."""
    container = Dimensions.mm(20, 10, 10)
    boxes = [box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)]
    assert safe_route_removal_order(boxes, [None, None], container) == []


def test_items_at_the_same_stop_may_be_removed_in_either_order():
    """Two items due at the same stop, side by side with only one clear at a time along
    the single allowed exit, is not a route violation -- it is an ordinary same-stop
    tie, resolved the same deterministic way `safe_removal_order` resolves any tie."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    order = safe_route_removal_order([near, far], [0, 0], container, directions=("-x",))
    assert order == [0, 1]


def test_route_order_and_stacking_can_agree():
    """`top` (the contact graph: it must come off before `bottom` regardless of
    any route) is also due at the earlier stop here, so the structural requirement and
    the route requirement point the same way and the order satisfies both at once."""
    container = Dimensions.mm(10, 10, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    order = safe_route_removal_order([bottom, top], [1, 0], container)
    assert order == [1, 0]  # top (stop 0, index 1) first, then bottom (stop 1, index 0)


def test_route_order_and_stacking_can_conflict():
    """The reverse of the previous case: `bottom` is due at the earlier stop, but
    `top` -- structurally required off first, whatever the route says -- is not due
    until later. No order can satisfy both, so this must be reported as a route
    violation rather than silently reordered around the structural constraint."""
    container = Dimensions.mm(10, 10, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    with pytest.raises(RouteSequenceError) as excinfo:
        safe_route_removal_order([bottom, top], [0, 1], container)
    assert excinfo.value.stop == 0
    assert excinfo.value.stuck == frozenset({0})


# ------------------------------------------------------------ reachability

def test_an_unblocked_item_is_reachable_with_no_evidence():
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    reachability = placement_reachability([near, far], container, directions=("-x", "+x"))
    assert all(entry.reachable for entry in reachability)
    assert all(not entry.blocked_by_support and not entry.blocked_by_neighbors and not entry.blocked_by_route
               for entry in reachability)


def test_a_single_exit_direction_makes_the_far_item_unreachable_though_geometrically_valid():
    """A geometrically valid packing (no collisions, both boxes within bounds) where
    `far` is nonetheless unreachable right now: its only allowed exit sweep runs
    straight through `near`. This is the acceptance case in miniature -- valid
    placement, unreachable item."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    reachability = placement_reachability([near, far], container, directions=("-x",))
    assert reachability[0].reachable
    assert not reachability[1].reachable
    assert reachability[1].blocked_by_neighbors == frozenset({0})
    assert reachability[1].blocked_by_support == frozenset()
    assert reachability[1].blocked_by_route == frozenset()


def test_an_item_boxed_in_on_every_side_is_unreachable_even_though_nothing_has_moved():
    """The centre item has a neighbour flush against all six of its faces -- every
    exit sweep is blocked and its one supporter-side neighbour also still rests on
    it, so it cannot be reached at all, and every blocking neighbour is named."""
    container = Dimensions.mm(30, 30, 30)
    center = box(10, 10, 10, 10, 10, 10)
    neighbors = [
        box(0, 10, 10, 10, 10, 10), box(20, 10, 10, 10, 10, 10),
        box(10, 0, 10, 10, 10, 10), box(10, 20, 10, 10, 10, 10),
        box(10, 10, 0, 10, 10, 10), box(10, 10, 20, 10, 10, 10),
    ]
    boxes = [center, *neighbors]
    reachability = placement_reachability(boxes, container)
    assert not reachability[0].reachable
    assert reachability[0].blocked_by_neighbors == frozenset({1, 2, 3, 4, 5, 6})
    assert reachability[0].blocked_by_support == frozenset({6})  # the +z neighbour rests on it
    # Every side neighbour is itself reachable except the -z one (index 5), which the
    # centre item still rests on -- it needs the centre gone first, same as any other
    # stacked pair.
    assert [entry.reachable for entry in reachability[1:]] == [True, True, True, True, False, True]
    assert reachability[5].blocked_by_support == frozenset({0})


def test_a_child_still_present_blocks_reachability_even_with_a_clear_sweep():
    """A stacked pair: the bottom item has a clear `+x` sweep (nothing beside it), but
    its child still rests on top, so it is not actually reachable yet --
    `blocked_by_support` catches what a sweep-only check would miss."""
    container = Dimensions.mm(10, 10, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    reachability = placement_reachability([bottom, top], container)
    assert not reachability[0].reachable
    assert reachability[0].blocked_by_support == frozenset({1})
    assert reachability[0].blocked_by_neighbors == frozenset()
    assert reachability[1].reachable


def test_route_order_makes_an_otherwise_clear_item_unreachable():
    """`near` has a clear geometric exit, but it is due at stop 1 while `far` (due at
    the earlier stop 0) is still present -- the route forbids removing it
    out of order, so it is reported unreachable with `far` named as the blocker, even
    though nothing about its own geometry is in the way."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    reachability = placement_reachability([near, far], container, stops=[1, 0], directions=("-x", "+x"))
    assert not reachability[0].reachable
    assert reachability[0].blocked_by_route == frozenset({1})
    assert reachability[0].blocked_by_neighbors == frozenset()


def test_no_stops_leaves_route_blocking_entirely_unaffected():
    """Every existing single-stop caller never populates `stops` -- this must be a
    complete no-op for `blocked_by_route`, not merely a small one."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    reachability = placement_reachability([near, far], container, directions=("-x",))
    assert all(entry.blocked_by_route == frozenset() for entry in reachability)


def test_items_at_the_same_stop_never_block_each_other_by_route():
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    reachability = placement_reachability([near, far], container, stops=[0, 0], directions=("-x", "+x"))
    assert all(entry.blocked_by_route == frozenset() for entry in reachability)


def test_an_empty_direction_set_leaves_every_placement_unreachable_with_no_named_neighbor():
    """No allowed direction at all is a total prohibition, not a specific collision --
    `blocked_by_neighbors` names nothing because there is no direction to check a
    sweep against, but the item is still correctly reported as unreachable."""
    container = Dimensions.mm(10, 10, 10)
    reachability = placement_reachability([box(0, 0, 0, 10, 10, 10)], container, directions=())
    assert not reachability[0].reachable
    assert reachability[0].blocked_by_neighbors == frozenset()


def test_reachability_is_reported_for_every_placement_in_input_order():
    container = Dimensions.mm(30, 10, 10)
    boxes = [box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10), box(20, 0, 0, 10, 10, 10)]
    reachability = placement_reachability(boxes, container)
    assert [entry.index for entry in reachability] == [0, 1, 2]


# --------------------------------------------------------------------- loading

def test_a_stacked_pair_requires_the_bottom_loaded_first():
    container = Dimensions.mm(20, 20, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    graph = LoadingDependencyGraph.build([bottom, top])
    assert graph.depends_on == (frozenset(), frozenset({0}))
    assert safe_loading_order([bottom, top], container) == [0, 1]


def test_two_unstacked_items_have_no_loading_dependency():
    container = Dimensions.mm(20, 10, 10)
    a, b = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    graph = LoadingDependencyGraph.build([a, b])
    assert graph.depends_on == (frozenset(), frozenset())


def test_a_three_level_stack_must_load_bottom_to_top():
    container = Dimensions.mm(10, 10, 30)
    stack = [box(0, 0, level * 10, 10, 10, 10) for level in range(3)]
    assert safe_loading_order(stack, container) == [0, 1, 2]


def test_loading_and_unloading_a_stack_are_exact_opposites():
    """Not a coincidence this module relies on: unloading a stack top-to-bottom and
    loading it bottom-to-top are the same claim about the same geometry, seen from
    opposite ends of the same replay."""
    container = Dimensions.mm(10, 10, 40)
    stack = [box(0, 0, level * 10, 10, 10, 10) for level in range(4)]
    assert list(reversed(safe_removal_order(stack, container))) == safe_loading_order(stack, container)


def test_a_single_allowed_direction_orders_loading_by_distance_from_the_door():
    """The loading counterpart of the unloading LIFO-shape test -- and the reverse of
    it, not a repeat: with only `-x` allowed, `far` must be loaded first, because once
    `near` (flush against the `-x` wall) is placed, nothing can ever enter through
    `-x` again to reach `far`'s position. A naive "place whichever is available now"
    greedy loader gets this backwards -- `near` looks immediately loadable too, and
    taking it first permanently traps `far` (measured directly while building this
    module: that greedy shape raised `SequenceError` here). `safe_loading_order`
    avoids the trap by construction; see its own docstring."""
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    assert safe_loading_order([near, far], container, directions=("-x",)) == [1, 0]
    assert safe_loading_order([near, far], container, directions=("+x",)) == [0, 1]


def test_an_empty_direction_set_is_a_clean_loading_deadlock():
    container = Dimensions.mm(20, 20, 20)
    with pytest.raises(SequenceError) as excinfo:
        safe_loading_order([box(0, 0, 0, 10, 10, 10)], container, directions=())
    assert excinfo.value.stuck == frozenset({0})


def test_the_loading_order_is_deterministic_across_ties():
    container = Dimensions.mm(30, 10, 10)
    boxes = [box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10), box(20, 0, 0, 10, 10, 10)]
    first = safe_loading_order(boxes, container)
    for _ in range(5):
        assert safe_loading_order(boxes, container) == first


# ------------------------------------------------------------------ both graphs

@pytest.mark.parametrize("seed", range(20))
def test_support_edges_are_provably_acyclic(seed):
    """Randomised stacks of axis-aligned boxes at strictly increasing height per
    column -- support dependencies can never cycle in either direction, checked as a
    property across many generated configurations, not asserted from a single
    hand-built example."""
    rng = random.Random(seed)
    boxes = []
    for column in range(rng.randint(1, 4)):
        x = column * 10
        z = 0
        for _ in range(rng.randint(1, 4)):
            boxes.append(box(x, 0, z, 10, 10, 10))
            z += 10
    assert UnloadingDependencyGraph.build(boxes).is_acyclic()
    assert LoadingDependencyGraph.build(boxes).is_acyclic()


def test_a_synthetic_unloading_cycle_is_detected():
    """Not derived from geometry -- a direct graph-level cycle, to prove the detector
    is correct independently of whether real geometry can ever produce one (it
    provably cannot for support edges alone; see the module docstring)."""
    graph = UnloadingDependencyGraph((frozenset({1}), frozenset({0})))
    assert not graph.is_acyclic()


def test_a_synthetic_loading_cycle_is_detected():
    graph = LoadingDependencyGraph((frozenset({1}), frozenset({0})))
    assert not graph.is_acyclic()


def test_an_unknown_direction_is_rejected_not_treated_as_minus_z():
    """The concrete bug the audit reopened this over: an unrecognised direction string
    used to fall through to `-z` silently. Checked against both graphs and both the
    generator and replay entry points."""
    container = Dimensions.mm(20, 20, 20)
    single = [box(0, 0, 0, 10, 10, 10)]
    for call in (
        lambda: safe_removal_order(single, container, directions=("sideways",)),
        lambda: safe_loading_order(single, container, directions=("sideways",)),
        lambda: replay_removal_order(single, container, [0], directions=("sideways",)),
        lambda: replay_loading_order(single, container, [0], directions=("sideways",)),
    ):
        with pytest.raises(InvalidDirectionError) as excinfo:
            call()
        assert excinfo.value.direction == "sideways"


# ---------------------------------------------------------- replay: unloading

# Replay_removal_order/replay_loading_order are second, independent code
# paths over the same geometric primitives -- neither calls its matching generator, so
# a bug in the generator's own search/tie-break logic cannot hide from either.

def test_replaying_a_generated_removal_order_raises_nothing():
    container = Dimensions.mm(10, 10, 30)
    stack = [box(0, 0, level * 10, 10, 10, 10) for level in range(3)]
    order = safe_removal_order(stack, container)
    replay_removal_order(stack, container, order)  # must not raise


def test_replaying_a_real_packed_containers_removal_order_raises_nothing():
    """The same real, 100-item solver-produced packing used to confirm the dependency graph's
    unreachability proof empirically -- here confirming the independent replay agrees
    with the generator on every one of its steps, not only on toy examples."""
    from packvium import Container, Item, Packer, PackingConfig

    rng = random.Random(0)
    items = [
        Item(
            id=f"sku-{i}",
            dimensions=Dimensions.mm(rng.randint(5, 40), rng.randint(5, 40), rng.randint(5, 40)),
        )
        for i in range(100)
    ]
    containers = [Container(id="c", inner_dimensions=Dimensions.mm(200, 200, 200))]
    result = Packer(PackingConfig()).pack(items, containers)
    placed = result.containers[0]
    boxes = [placement.box for placement in placed.placements]
    order = safe_removal_order(boxes, placed.container.inner_dimensions)
    assert len(order) == len(boxes)
    replay_removal_order(boxes, placed.container.inner_dimensions, order)  # must not raise
    loading_order = safe_loading_order(boxes, placed.container.inner_dimensions)
    assert len(loading_order) == len(boxes)
    replay_loading_order(boxes, placed.container.inner_dimensions, loading_order)  # must not raise


def test_replaying_a_reversed_stack_removal_order_is_caught_as_blocked_by_support():
    container = Dimensions.mm(20, 20, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    with pytest.raises(SequenceReplayError) as excinfo:
        replay_removal_order([bottom, top], container, [0, 1])  # bottom first is wrong
    assert excinfo.value.index == 0
    assert excinfo.value.step == 0


def test_replaying_the_wrong_removal_exit_direction_is_caught_as_blocked_by_geometry():
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    with pytest.raises(SequenceReplayError) as excinfo:
        replay_removal_order([near, far], container, [1, 0], directions=("-x",))  # far first blocks on near
    assert excinfo.value.index == 1
    assert excinfo.value.step == 0


def test_replaying_a_malformed_removal_order_is_rejected_before_any_geometry_check():
    container = Dimensions.mm(20, 20, 20)
    boxes = [box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)]
    with pytest.raises(SequenceReplayError):
        replay_removal_order(boxes, container, [0, 0])  # duplicate, missing index 1
    with pytest.raises(SequenceReplayError):
        replay_removal_order(boxes, container, [0])  # too short


# ------------------------------------------------------------- replay: loading

def test_replaying_a_generated_loading_order_raises_nothing():
    container = Dimensions.mm(10, 10, 30)
    stack = [box(0, 0, level * 10, 10, 10, 10) for level in range(3)]
    order = safe_loading_order(stack, container)
    replay_loading_order(stack, container, order)  # must not raise


def test_replaying_a_top_first_loading_order_is_caught_as_blocked_by_support():
    container = Dimensions.mm(20, 20, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    with pytest.raises(SequenceReplayError) as excinfo:
        replay_loading_order([bottom, top], container, [1, 0])  # top first: no supporter yet
    assert excinfo.value.index == 1
    assert excinfo.value.step == 0


def test_replaying_the_wrong_loading_entry_direction_is_caught_as_blocked_by_geometry():
    container = Dimensions.mm(20, 10, 10)
    near, far = box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)
    with pytest.raises(SequenceReplayError) as excinfo:
        replay_loading_order([near, far], container, [0, 1], directions=("-x",))  # near first permanently blocks far
    assert excinfo.value.index == 1
    assert excinfo.value.step == 1


def test_replaying_a_malformed_loading_order_is_rejected_before_any_geometry_check():
    container = Dimensions.mm(20, 20, 20)
    boxes = [box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)]
    with pytest.raises(SequenceReplayError):
        replay_loading_order(boxes, container, [0, 0])
    with pytest.raises(SequenceReplayError):
        replay_loading_order(boxes, container, [0])


def test_replay_rejects_outside_and_colliding_geometry():
    container = Dimensions.mm(20, 20, 20)
    with pytest.raises(SequenceReplayError, match="outside"):
        replay_loading_order([box(15, 0, 0, 10, 10, 10)], container, [0])
    with pytest.raises(SequenceReplayError, match="collides"):
        replay_loading_order(
            [box(0, 0, 0, 10, 10, 10), box(5, 0, 0, 10, 10, 10)],
            container,
            [0, 1],
        )


# ------------------------------------------ business-rule prefix replay

def stacked_placement(item, x, y, z) -> Placement:
    """One instance of `item` placed at `(x, y, z)` mm with no rotation, geometry and
    envelope identical -- the minimal `Placement` `verify_loading_prefix_business_rules`
    needs; support/weight propagation reads it through `constraints.load_units`."""
    instance = item.instances()[0]
    dimensions = instance.item.dimensions
    at = Point(x * MM, y * MM, z * MM)
    return Placement(instance, at, Rotation.LWH, dimensions, at, dimensions)


def test_a_loading_prefix_that_overloads_a_fragile_supporter_is_caught_at_its_step():
    fragile = Item.create("fragile", Dimensions.mm(10, 10, 5), weight="1kg", max_top_load="500g")
    heavy = Item.create("heavy", Dimensions.mm(10, 10, 5), weight="5kg")
    bottom = stacked_placement(fragile, 0, 0, 0)
    top = stacked_placement(heavy, 0, 0, 5)
    container = Container.create("c", Dimensions.mm(10, 10, 20))
    with pytest.raises(SequenceReplayError) as excinfo:
        verify_loading_prefix_business_rules([bottom, top], [0, 1], container)
    assert excinfo.value.step == 1
    assert "top_load_exceeded" in excinfo.value.reason


def test_a_loading_prefix_that_exceeds_a_stacked_item_limit_is_caught_at_its_step():
    base = Item.create("base", Dimensions.mm(10, 10, 5), max_stacked_items=1)
    filler = Item.create("filler", Dimensions.mm(10, 10, 5), quantity=2)
    first, second = filler.instances()
    placements = [
        stacked_placement(base, 0, 0, 0),
        Placement(first, Point(0, 0, 5 * MM), Rotation.LWH, first.item.dimensions, Point(0, 0, 5 * MM), first.item.dimensions),
        Placement(second, Point(0, 0, 10 * MM), Rotation.LWH, second.item.dimensions, Point(0, 0, 10 * MM), second.item.dimensions),
    ]
    container = Container.create("c", Dimensions.mm(10, 10, 20))
    with pytest.raises(SequenceReplayError) as excinfo:
        verify_loading_prefix_business_rules(placements, [0, 1, 2], container)
    assert excinfo.value.step == 2
    assert "stacked_item_limit_exceeded" in excinfo.value.reason


def test_a_loading_prefix_that_crushes_a_containers_floor_density_limit_is_caught():
    heavy = Item.create("heavy", Dimensions.mm(10, 10, 5), weight="10kg")
    container = Container.create("c", Dimensions.mm(10, 10, 10), max_stack_density="1kg")
    with pytest.raises(SequenceReplayError) as excinfo:
        verify_loading_prefix_business_rules([stacked_placement(heavy, 0, 0, 0)], [0], container)
    assert excinfo.value.step == 0
    assert "stack_density_exceeded" in excinfo.value.reason


def test_a_loading_prefix_that_stacks_onto_a_non_stackable_item_is_caught():
    base = Item.create("base", Dimensions.mm(10, 10, 5), stackable=False)
    rider = Item.create("rider", Dimensions.mm(10, 10, 5))
    placements = [stacked_placement(base, 0, 0, 0), stacked_placement(rider, 0, 0, 5)]
    container = Container.create("c", Dimensions.mm(10, 10, 20))
    with pytest.raises(SequenceReplayError) as excinfo:
        verify_loading_prefix_business_rules(placements, [0, 1], container)
    assert excinfo.value.step == 1
    assert "non_stackable_item_has_load" in excinfo.value.reason


def test_a_loading_prefix_that_violates_a_ground_contact_rule_is_caught():
    # "single" requires resting on exactly one supporter; two half-width bases side
    # by side under one full-width rider violate it.
    left = Item.create("left", Dimensions.mm(5, 5, 5))
    right = Item.create("right", Dimensions.mm(5, 5, 5))
    rider = Item.create("rider", Dimensions.mm(10, 5, 5), ground_contact_rule="single")
    placements = [
        stacked_placement(left, 0, 0, 0),
        stacked_placement(right, 5, 0, 0),
        stacked_placement(rider, 0, 0, 5),
    ]
    container = Container.create("c", Dimensions.mm(10, 10, 20))
    with pytest.raises(SequenceReplayError) as excinfo:
        verify_loading_prefix_business_rules(placements, [0, 1, 2], container)
    assert excinfo.value.step == 2
    assert "ground_contact_violation" in excinfo.value.reason


def test_a_loading_prefix_treats_each_nested_predecessor_as_one_supporter():
    crate = Item.create(
        "crate", Dimensions.mm(10, 10, 10), quantity=3,
        nesting_height=Length.mm(5), ground_contact_rule="single",
    )
    instances = crate.instances()
    placements = [
        Placement(instance, Point(0, 0, z * MM), Rotation.LWH, instance.dimensions,
                  Point(0, 0, z * MM), instance.dimensions)
        for instance, z in zip(instances, (0, 5, 10))
    ]
    container = Container.create("c", Dimensions.mm(10, 10, 30))

    verify_loading_prefix_business_rules(placements, [0, 1, 2], container)


def test_a_loading_prefix_that_respects_every_business_rule_raises_nothing():
    fragile = Item.create("fragile", Dimensions.mm(10, 10, 5), weight="1kg", max_top_load="5kg",
                          max_stacked_items=2, stackable=True)
    light = Item.create("light", Dimensions.mm(10, 10, 5), weight="1kg")
    # No max_stack_density set: this scenario's tiny 100 mm^2 footprint under even a
    # light load is already a very high kg/m^2 figure, and this test's purpose is the
    # other three rules -- stack_density_exceeded has its own dedicated test above.
    container = Container.create("c", Dimensions.mm(10, 10, 20))
    placements = [stacked_placement(fragile, 0, 0, 0), stacked_placement(light, 0, 0, 5)]
    verify_loading_prefix_business_rules(placements, [0, 1], container)  # must not raise


def test_composed_safe_loading_api_cannot_return_an_overloaded_order():
    fragile = Item.create("fragile", Dimensions.mm(10, 10, 5), weight="1kg", max_top_load="500g")
    heavy = Item.create("heavy", Dimensions.mm(10, 10, 5), weight="5kg")
    placements = [stacked_placement(fragile, 0, 0, 0), stacked_placement(heavy, 0, 0, 5)]
    container = Container.create("c", Dimensions.mm(10, 10, 20))
    with pytest.raises(SequenceReplayError):
        safe_loading_order_for_placements(placements, container)


def test_a_malformed_business_rule_order_is_rejected_before_any_rule_check():
    item = Item.create("a", Dimensions.mm(10, 10, 5))
    placements = [stacked_placement(item, 0, 0, 0)]
    container = Container.create("c", Dimensions.mm(10, 10, 10))
    with pytest.raises(SequenceReplayError):
        verify_loading_prefix_business_rules(placements, [0, 0], container)


# ------------------------------------------------------------------------ evidence

def test_removal_evidence_names_the_support_dependency_and_direction_used():
    container = Dimensions.mm(20, 20, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    steps = safe_removal_order_with_evidence([bottom, top], container)
    assert [step.index for step in steps] == [1, 0]
    top_step, bottom_step = steps
    assert top_step.depends_on == frozenset()  # nothing rests on top
    assert top_step.direction == "+x"  # first tried direction with nothing else present
    assert bottom_step.depends_on == frozenset({1})  # top structurally rests on bottom


def test_loading_evidence_names_the_support_dependency_and_direction_used():
    container = Dimensions.mm(20, 20, 20)
    bottom, top = box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)
    steps = safe_loading_order_with_evidence([bottom, top], container)
    assert [step.index for step in steps] == [0, 1]
    bottom_step, top_step = steps
    assert bottom_step.depends_on == frozenset()
    assert top_step.depends_on == frozenset({0})  # top depends on bottom being loaded first


def test_evidence_ordering_matches_the_plain_index_orders():
    container = Dimensions.mm(10, 10, 30)
    stack = [box(0, 0, level * 10, 10, 10, 10) for level in range(3)]
    assert [step.index for step in safe_removal_order_with_evidence(stack, container)] == safe_removal_order(stack, container)
    assert [step.index for step in safe_loading_order_with_evidence(stack, container)] == safe_loading_order(stack, container)


@requires_shared_scenes
def test_shared_four_language_sequence_fixtures():
    payload = json.loads(SHARED_SCENES.read_text())
    for scene in payload["scenes"]:
        raw_container = scene["container"]
        container = Dimensions(
            *(Length(raw_container[axis]) for axis in ("length", "width", "height"))
        )
        boxes = [
            AxisAlignedBox(
                Point(*(raw["origin"][axis] for axis in ("x", "y", "z"))),
                Dimensions(*(Length(raw["dimensions"][axis]) for axis in ("length", "width", "height"))),
            )
            for raw in scene["boxes"]
        ]
        directions = tuple(scene["directions"])
        assert [sorted(values) for values in LoadingDependencyGraph.build(boxes).depends_on] == scene["loading_graph"]
        assert [sorted(values) for values in UnloadingDependencyGraph.build(boxes).depends_on] == scene["unloading_graph"]
        if "reachability" in scene:
            stops = scene.get("stops")
            assert [
                {
                    "index": entry.index,
                    "reachable": entry.reachable,
                    "blocked_by_support": sorted(entry.blocked_by_support),
                    "blocked_by_neighbors": sorted(entry.blocked_by_neighbors),
                    "blocked_by_route": sorted(entry.blocked_by_route),
                }
                for entry in placement_reachability(boxes, container, stops, directions)
            ] == scene["reachability"]
        if "expected_error" in scene:
            with pytest.raises(SequenceError) as error:
                safe_loading_order(boxes, container, directions)
            assert error.value.to_dict() == scene["expected_error"]
            continue
        if "loading_steps" not in scene:
            continue
        assert [
            step.to_dict()
            for step in safe_loading_order_with_evidence(boxes, container, directions)
        ] == scene["loading_steps"]
        assert [
            step.to_dict()
            for step in safe_removal_order_with_evidence(boxes, container, directions)
        ] == scene["unloading_steps"]


# ---------------------------------------------------------- canonical DTOs

def test_sequence_step_to_dict_matches_the_cross_language_shape():
    step = SequenceStep(index=1, direction="+x", depends_on=frozenset({2, 0}))
    assert step.to_dict() == {"index": 1, "direction": "+x", "depends_on": [0, 2]}


@requires_shared_scenes
def test_sequence_warning_matches_the_shared_cross_language_shape():
    payload = json.loads(SHARED_SCENES.read_text())
    expected = payload["dto_contract"]["sequence_warning"]
    warning = SequenceWarning("sequence_advisory", 1, "sequence.advisory", {"unit": "mm", "clearance": "2"})
    assert warning.to_dict() == expected


def test_invalid_direction_error_to_dict_matches_the_cross_language_shape():
    error = InvalidDirectionError("sideways")
    assert error.to_dict() == {"code": "invalid_direction", "direction": "sideways"}


def test_sequence_error_to_dict_matches_the_cross_language_shape():
    error = SequenceError(frozenset({2, 0}))
    assert error.to_dict() == {"code": "sequence_stuck", "stuck": [0, 2]}


def test_sequence_replay_error_to_dict_matches_the_cross_language_shape():
    error = SequenceReplayError(3, 1, "no allowed direction is clear of the remaining placements")
    assert error.to_dict() == {
        "code": "sequence_replay",
        "index": 3,
        "step": 1,
        "reason": "no allowed direction is clear of the remaining placements",
    }
