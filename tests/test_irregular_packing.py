"""What the exact hull test buys, measured end to end through `Packer`.

The unit tests in `test_irregular_items.py` prove the predicate; these prove the solver
actually consults it. The scene is two complementary wedges -- a box sliced along its
diagonal -- whose bounding boxes are identical and whose solids merely touch. A box-only
engine fits one of them in a box-sized container. An engine that asks the hull fits both.

Every scenario is re-checked against the independent validator, so a wrong verdict cannot
pass here on a placement count while the layout underneath it is impossible.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from packvium import (AxisAlignedBox, Dimensions, Item, Length, Obstacle, PackingConfig,
                      Point, Rotation)
from packvium.geometry import ShapeType
from packvium.models import Placement, placements_collide
from packvium.nesting import occupied_volume
from packvium.serialization import pack_from_dict
from packvium.units import Weight
from support import assert_sound, container, item, pack

MM = 16_000
SIDE = 10 * MM


def lower_wedge(id: str, **kwargs) -> Item:
    """The half of a 10mm cube below the diagonal `x/L + y/W <= 1`."""
    return Item.create(
        id, Dimensions.mm(10, 10, 10), shape_type=ShapeType.CONVEX_HULL,
        hull_vertices=((0, 0, 0), (SIDE, 0, 0), (0, SIDE, 0),
                       (0, 0, SIDE), (SIDE, 0, SIDE), (0, SIDE, SIDE)),
        **kwargs,
    )


def upper_wedge(id: str, **kwargs) -> Item:
    """Its complement, above the same diagonal."""
    return Item.create(
        id, Dimensions.mm(10, 10, 10), shape_type=ShapeType.CONVEX_HULL,
        hull_vertices=((SIDE, SIDE, 0), (SIDE, 0, 0), (0, SIDE, 0),
                       (SIDE, SIDE, SIDE), (SIDE, 0, SIDE), (0, SIDE, SIDE)),
        **kwargs,
    )


def _placement(one: Item, x: int = 0, y: int = 0, z: int = 0) -> Placement:
    instance, = one.instances()
    point = Point(x, y, z)
    return Placement(instance, point, Rotation.LWH, one.dimensions, point, one.dimensions)


#: One container, so "did it fit" is a question about space rather than about how many
#: containers the packer was willing to open.
ONE_CONTAINER = dict(max_containers=1)


def test_two_complementary_wedges_share_one_box_sized_space():
    """The whole point of the epic, stated as a packing outcome rather than a predicate."""
    items = [lower_wedge("a"), upper_wedge("b")]
    box = container("cube", 10, 10, 10)
    result = pack(items, [box], PackingConfig.balanced(**ONE_CONTAINER))
    placed = sorted(p.instance.id for c in result.containers for p in c.placements)
    assert placed == ["a#1", "b#1"]
    assert_sound(result, items, [box])


def test_the_same_two_items_as_cuboids_do_not_fit():
    """The control. Without it the test above proves only that two items fit somewhere, not
    that the hull is what made room for them."""
    items = [item("a", 10, 10, 10), item("b", 10, 10, 10)]
    box = container("cube", 10, 10, 10)
    result = pack(items, [box], PackingConfig.balanced(**ONE_CONTAINER))
    placed = [p.instance.id for c in result.containers for p in c.placements]
    assert len(placed) == 1
    assert len(result.unpacked) == 1
    assert_sound(result, items, [box])


def test_touching_wedges_are_contact_and_not_collision():
    """Directly on the shared predicate the solver, the validator and the obstacle check all
    go through, so the three cannot drift apart on what "collides" means."""
    assert not placements_collide(_placement(lower_wedge("a")), _placement(upper_wedge("b")))
    assert placements_collide(_placement(lower_wedge("a")), _placement(lower_wedge("b")))


def test_a_route_bound_hull_falls_back_to_its_box():
    """`packing_sequence` reasons with box sweeps, so a hull on a route is deliberately packed
    as its box: one conservative answer in both places rather than two that disagree.

    Asserted on the predicate rather than on a pack outcome, because it is a rule about what
    the engine is allowed to know, not about how many items happen to fit."""
    routed = _placement(lower_wedge("a", stop_index=0))
    assert placements_collide(routed, _placement(upper_wedge("b", stop_index=1)))
    assert routed.hull_shape is None
    assert occupied_volume(routed) == SIDE ** 3 // 2


def test_a_clearance_makes_the_hull_fall_back_to_its_envelope():
    """A margin around a hull is not a hull; refining under clearance would hand back space
    the caller asked to keep empty, so the envelope decides and the hull is not consulted."""
    one, = lower_wedge("a").instances()
    physical = one.item.dimensions
    inflated = physical.expand(Length.mm(1))
    origin = Point(0, 0, 0)
    assert Placement(one, origin, Rotation.LWH, physical, origin, inflated).hull_shape is None
    assert Placement(one, origin, Rotation.LWH, physical, origin, physical).hull_shape is not None
    assert occupied_volume(Placement(one, origin, Rotation.LWH, physical, origin, inflated)) == SIDE ** 3 // 2


@pytest.mark.parametrize("rotation", list(Rotation))
def test_every_rotation_keeps_the_wedge_a_wedge(rotation):
    """A bare coordinate permutation would mirror the shape for three of the six rotations.
    Volume is the cheapest witness that none of them does: a reflected wedge has the same
    bounding box and the opposite handedness."""
    from packvium.hull import _cross, _dot, _subtract, rotate

    vertices = rotate(lower_wedge("a").hull_vertices, rotation)
    a, b, c, d = vertices[0], vertices[1], vertices[2], vertices[3]
    assert _dot(_subtract(d, a), _cross(_subtract(b, a), _subtract(c, a))) != 0


# ------------------------------------------------------------------ compressible items

def cushion(id: str, **kwargs) -> Item:
    """A 100mm cube that gives up a quarter of its height and fails above 100 kPa.

    Its footprint is exactly 0.01 m², which puts the crush boundary between 101 kg and
    102 kg of load -- close enough to state, far enough from a round number that a
    floating-point shortcut would land on the wrong side of it.
    """
    return Item.create(
        id, Dimensions.mm(100, 100, 100), shape_type=ShapeType.COMPRESSIBLE,
        compression_ratio_ppm=250_000, max_compression_pressure_kpa=100, **kwargs,
    )


def _stack(base: Item, topper: Item) -> list[Placement]:
    base_instance, = base.instances()
    top_instance, = topper.instances()
    floor, above = Point(0, 0, 0), Point(0, 0, base.dimensions.height.ticks)
    return [
        Placement(base_instance, floor, Rotation.LWH, base.dimensions, floor, base.dimensions),
        Placement(top_instance, above, Rotation.LWH, topper.dimensions, above, topper.dimensions),
    ]


@pytest.mark.parametrize("kilograms,crushes", [(100, False), (101, False), (102, True)])
def test_the_crush_boundary_decides_whether_a_load_may_rest_on_a_cushion(kilograms, crushes):
    """Asserted through the whole load-propagation path, not on the arithmetic alone, so a
    footprint or a top-load taken from the wrong box would show up here."""
    from packvium.constraints import crushed, load_units

    failure = crushed(load_units(_stack(cushion("soft"), item("brick", 100, 100, 100,
                                                              weight=f"{kilograms}kg"))))
    assert (failure is not None) == crushes
    if crushes:
        assert failure == ("crush_violation", "soft#1")


def test_a_crushing_stack_is_refused_rather_than_packed():
    """A crush is a hard boundary: the heavy item is left unpacked rather than arriving on
    top of a flattened cushion.

    The cushion is pinned to the floor and the crate is exactly one footprint wide, so "on
    top of the cushion" is the only place the brick could go. Without that the solver simply
    puts the brick underneath -- a legal answer, and one that would have made this test pass
    while proving nothing about crushing.
    """
    items = [cushion("soft", must_be_on_floor=True), item("brick", 100, 100, 100, weight="102kg")]
    box = container("crate", 100, 100, 200)
    result = pack(items, [box], PackingConfig.balanced(**ONE_CONTAINER))
    placed = [p.instance.id for c in result.containers for p in c.placements]
    assert placed == ["soft#1"]
    assert [u.instance.id for u in result.unpacked] == ["brick#1"]
    assert_sound(result, items, [box])


def test_a_load_inside_the_limit_still_packs():
    """The control: without it the test above would also pass if a compressible item simply
    refused every neighbour."""
    items = [cushion("soft", must_be_on_floor=True), item("brick", 100, 100, 100, weight="101kg")]
    box = container("crate", 100, 100, 200)
    result = pack(items, [box], PackingConfig.balanced(**ONE_CONTAINER))
    placed = sorted(p.instance.id for c in result.containers for p in c.placements)
    assert placed == ["brick#1", "soft#1"]
    assert_sound(result, items, [box])


def test_the_final_validation_catches_a_crush_the_solver_never_produced():
    """The whole-container bearing pass runs on every result, stack-sensitive or not, so a
    hand-built or externally-supplied plan cannot smuggle a crushed item past it."""
    from packvium.models import PackedContainer
    from support import issues_for

    items = [cushion("soft"), item("brick", 100, 100, 100, weight="500kg")]
    box = container("crate", 100, 100, 200)
    packed = [PackedContainer(box, 0, tuple(_stack(items[0], items[1])))]
    assert "crush_violation" in issues_for(items, [box], packed)


# ------------------------------------------------------------------ hulls against plain boxes

def test_a_wedge_clears_an_obstacle_its_bounding_box_overlaps():
    """The hull-versus-box half of the predicate, end to end.

    The obstacle fills the quarter of the crate the wedge slopes away from. Their boxes
    overlap across the whole footprint and their solids meet only along one edge, so a
    box-only engine has nowhere to put this item and an exact one puts it on the floor.
    """
    obstacle = Obstacle("post", AxisAlignedBox(Point(5 * MM, 5 * MM, 0), Dimensions.mm(5, 5, 10)))
    crate = container("crate", 10, 10, 10, obstacles=(obstacle,))
    items = [lower_wedge("a")]
    result = pack(items, [crate], PackingConfig.balanced(**ONE_CONTAINER))
    assert [p.instance.id for c in result.containers for p in c.placements] == ["a#1"]
    assert_sound(result, items, [crate])


def test_the_same_crate_has_no_room_for_the_wedge_as_a_cuboid():
    """The control for the obstacle case."""
    obstacle = Obstacle("post", AxisAlignedBox(Point(5 * MM, 5 * MM, 0), Dimensions.mm(5, 5, 10)))
    crate = container("crate", 10, 10, 10, obstacles=(obstacle,))
    result = pack([item("a", 10, 10, 10)], [crate], PackingConfig.balanced(**ONE_CONTAINER))
    assert result.containers == () or not result.containers[0].placements
    assert len(result.unpacked) == 1


def test_a_wedge_and_a_box_that_only_touch_are_not_colliding():
    """Directly on the shared predicate, where the second solid is an ordinary cuboid rather
    than another hull -- the branch the obstacle path above reaches through the solver."""
    wedge = _placement(lower_wedge("a"))
    brick, = item("b", 5, 5, 10).instances()
    corner = Point(5 * MM, 5 * MM, 0)
    box = Placement(brick, corner, Rotation.LWH, brick.item.dimensions, corner,
                    brick.item.dimensions)
    assert not placements_collide(wedge, box)
    overlapping = Point(0, 0, 0)
    assert placements_collide(wedge, Placement(brick, overlapping, Rotation.LWH,
                                               brick.item.dimensions, overlapping,
                                               brick.item.dimensions))


def test_a_uniform_run_of_hulls_leaves_the_lattice_to_the_general_solver():
    """`GridSolver` tiles bounding boxes and would call the result exact. Two wedges of one
    type are exactly the input that reaches it, so the delegation is asserted on the outcome
    it protects: a lattice would have claimed two cells and overlapped the solids."""
    items = [lower_wedge("a", quantity=2)]
    box = container("pair", 20, 10, 10)
    result = pack(items, [box], PackingConfig.balanced(**ONE_CONTAINER))
    placed = sorted(p.instance.id for c in result.containers for p in c.placements)
    assert placed == ["a#1", "a#2"]
    assert_sound(result, items, [box])


# ------------------------------------------------------------------ over the wire

def _wire_request(items: list[dict], height: int = 100) -> dict:
    return {
        "units": {"length": "mm"},
        "configuration": {"solver_profile": "balanced", "max_containers": 1},
        "items": items,
        "containers": [{"id": "crate", "inner_dimensions": {
            "length": "100", "width": "100", "height": str(height)}}],
    }


def _wire_point(x: int, y: int, z: int) -> dict:
    return {"x": str(x), "y": str(y), "z": str(z)}


def test_a_hull_request_survives_the_wire_and_packs_as_a_hull():
    """`hull_vertices` is parsed through `Length`, so this also pins the wire convention:
    non-negative offsets from the corner of the item's own bounding box."""
    lower = [_wire_point(0, 0, 0), _wire_point(100, 0, 0), _wire_point(0, 100, 0),
             _wire_point(0, 0, 100), _wire_point(100, 0, 100), _wire_point(0, 100, 100)]
    upper = [_wire_point(100, 100, 0), _wire_point(100, 0, 0), _wire_point(0, 100, 0),
             _wire_point(100, 100, 100), _wire_point(100, 0, 100), _wire_point(0, 100, 100)]
    result = pack_from_dict(_wire_request([
        {"id": "a", "dimensions": {"length": "100", "width": "100", "height": "100"},
         "shape_type": "convex_hull", "hull_vertices": lower},
        {"id": "b", "dimensions": {"length": "100", "width": "100", "height": "100"},
         "shape_type": "convex_hull", "hull_vertices": upper},
    ]))
    packed = [p["item_id"] for c in result["containers"] for p in c["placements"]]
    assert sorted(packed) == ["a#1", "b#1"]
    # Two bounding boxes would fill the crate twice over; two hulls fill it exactly once.
    assert int(result["containers"][0]["used_volume_ticks3"]) == (100 * MM) ** 3


def test_a_hull_coordinate_below_zero_is_refused_at_the_wire():
    """`Length` owns non-negativity, and the refusal names it rather than producing a hull
    mirrored into the wrong octant."""
    with pytest.raises(ValueError, match="cannot be negative"):
        pack_from_dict(_wire_request([
            {"id": "a", "dimensions": {"length": "100", "width": "100", "height": "100"},
             "shape_type": "convex_hull",
             "hull_vertices": [_wire_point(0, 0, 0), _wire_point(-1, 0, 0),
                               _wire_point(0, 100, 0), _wire_point(0, 0, 100)]},
        ]))


def test_a_compressible_request_survives_the_wire_and_compresses():
    """`compression_ratio` crosses as a JSON number and is turned into ppm exactly once, at
    the boundary, so nothing downstream of the parser ever sees a float."""
    result = pack_from_dict(_wire_request([
        {"id": "cushion", "dimensions": {"length": "100", "width": "100", "height": "100"},
         "weight": {"value": "2", "unit": "kg"}, "must_be_on_floor": True,
         "shape_type": "compressible", "compression_ratio": 0.25,
         "max_compression_pressure_kpa": 100},
        {"id": "brick", "dimensions": {"length": "100", "width": "100", "height": "100"},
         "weight": {"value": "101", "unit": "kg"}},
    ], height=200))
    packed = [p["item_id"] for c in result["containers"] for p in c["placements"]]
    assert sorted(packed) == ["brick#1", "cushion#1"]
    assert int(result["containers"][0]["used_volume_ticks3"]) < 2 * (100 * MM) ** 3


def test_a_crushed_placement_reports_its_uncompressed_volume():
    """Reached only by a plan the solver would never build, and deliberately not an
    exception: `crushed` already refuses this arrangement and the validator reports it, so a
    volume property raising here would turn a reported issue into a crash."""
    from packvium.nesting import occupied_volume

    heavy = _stack(cushion("soft"), item("brick", 100, 100, 100, weight="500kg"))
    crushed_placement = heavy[0]
    assert crushed_placement.top_load.ticks == 0
    loaded = replace(crushed_placement, top_load=Weight.parse("500kg"))
    assert occupied_volume(loaded) == loaded.dimensions.volume
