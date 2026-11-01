"""Nesting overlap accounting, checked against hand-computed values.

A naive sum of each placement's own volume overstates how much space a nested
column actually fills, since neighbouring nested layers share part of the same
physical space. Every value here was worked out by hand first, not derived from
the code under test.
"""

from __future__ import annotations

from packvium import Dimensions, Item, Length, Placement, Point, Rotation, Weight
from packvium.nesting import column_layer_z, column_layers, is_valid_nesting, used_volume, used_volume_delta


def placement(item_id: str, x: int, y: int, z: int, size: int, nesting_height=None, item=None) -> Placement:
    dims = Dimensions(Length(size), Length(size), Length(size))
    one = item or Item.create(item_id, dims, Weight(0), nesting_height=nesting_height)
    instance, = one.instances()
    position = Point(x, y, z)
    return Placement(instance, position, Rotation.LWH, dims, position, dims)


def test_a_lone_placement_has_no_overlap_to_subtract():
    p = placement("a", 0, 0, 0, 100)
    assert used_volume([p]) == 100 ** 3


def test_two_nested_layers_do_not_double_count_the_shared_slice():
    item = Item.create("crate", Dimensions(Length(100), Length(100), Length(100)), Weight(0), nesting_height=Length(40))
    lower = placement("crate", 0, 0, 0, 100, item=item)
    upper = placement("crate", 0, 0, 60, 100, item=item)
    assert is_valid_nesting(lower, upper)
    # Union height is 100 + 60 = 160, not 200 -- the naive sum of two 100^3 cubes.
    assert used_volume([lower, upper]) == 100 * 100 * 160
    assert used_volume([lower, upper]) < lower.dimensions.volume + upper.dimensions.volume


def test_a_three_high_nested_column_only_subtracts_each_adjacent_pair_once():
    item = Item.create("crate", Dimensions(Length(100), Length(100), Length(100)), Weight(0), nesting_height=Length(40))
    layers = [placement("crate", 0, 0, z, 100, item=item) for z in (0, 60, 120)]
    # Union height is 220 (100 + 60 + 60), matching the GridSolver placement test.
    assert used_volume(layers) == 100 * 100 * 220
    incremental = 0
    placed = []
    for layer in layers:
        incremental += used_volume_delta(placed, layer)
        placed.append(layer)
    assert incremental == used_volume(layers)


def test_an_overlap_deeper_than_declared_is_not_treated_as_a_valid_nest():
    item = Item.create("crate", Dimensions(Length(100), Length(100), Length(100)), Weight(0), nesting_height=Length(40))
    lower = placement("crate", 0, 0, 0, 100, item=item)
    upper = placement("crate", 0, 0, 50, 100, item=item)  # 50mm overlap, not 40
    assert not is_valid_nesting(lower, upper)
    # No correction applies -- the naive (here, physically wrong, but that is a
    # collision the validator catches elsewhere, not this function's job) sum stands.
    assert used_volume([lower, upper]) == lower.dimensions.volume + upper.dimensions.volume


def test_items_without_nesting_height_are_summed_plainly():
    a = placement("a", 0, 0, 0, 100)
    b = placement("a", 100, 0, 0, 100)
    assert used_volume([a, b]) == 2 * 100 ** 3


def test_two_nestable_items_at_the_same_height_are_not_a_nested_pair():
    """Identical footprint and height is a collision, not one item sunk into another."""
    item = Item.create("crate", Dimensions(Length(100), Length(100), Length(100)), Weight(0),
                       quantity=2, nesting_height=Length(40))
    first, second = item.instances()
    position = Point(0, 0, 0)
    dims = item.dimensions
    one = Placement(first, position, Rotation.LWH, dims, position, dims)
    other = Placement(second, position, Rotation.LWH, dims, position, dims)
    assert not is_valid_nesting(one, other)


def test_under_a_clearance_only_the_items_own_overlap_is_subtracted():
    # 10x1x10 cups nesting 9 under a 1 mm clearance. The 12x3x12 envelopes sink by 9,
    # so the cups overlap by 7 over their own 10 mm^2 -- 70, not the envelope's 9 x 36 = 324,
    # which is more than a cup's own volume of 100.
    item = Item.create("cup", Dimensions(Length(10), Length(1), Length(10)), Weight(0), nesting_height=Length(9))
    physical = item.dimensions
    envelope = Dimensions(Length(12), Length(3), Length(12))
    lower_instance, = item.instances()
    upper_instance, = Item.create("cup", physical, Weight(0), nesting_height=Length(9)).instances()
    lower = Placement(lower_instance, Point(1, 1, 1), Rotation.LWH, physical, Point(0, 0, 0), envelope)
    upper = Placement(upper_instance, Point(1, 1, 4), Rotation.LWH, physical, Point(0, 0, 3), envelope)
    assert is_valid_nesting(lower, upper)
    assert used_volume([lower, upper]) == 2 * 100 - 70
    assert used_volume_delta([lower], upper) == 100 - 70


def test_a_clearance_twice_the_nesting_height_leaves_nothing_to_subtract():
    item = Item.create("tray", Dimensions(Length(10), Length(10), Length(10)), Weight(0), nesting_height=Length(2))
    physical = item.dimensions
    envelope = Dimensions(Length(12), Length(12), Length(12))
    lower_instance, = item.instances()
    upper_instance, = Item.create("tray", physical, Weight(0), nesting_height=Length(2)).instances()
    lower = Placement(lower_instance, Point(1, 1, 1), Rotation.LWH, physical, Point(0, 0, 0), envelope)
    upper = Placement(upper_instance, Point(1, 1, 11), Rotation.LWH, physical, Point(0, 0, 10), envelope)
    assert is_valid_nesting(lower, upper)
    assert used_volume([lower, upper]) == 2 * 1000


def test_a_shallow_nest_chains_every_layer():
    # 2 x 40 <= 100: a third crate sunk into the second only touches the first.
    assert [column_layer_z(layer, 100, 40) for layer in range(4)] == [0, 60, 120, 180]
    assert column_layers(280, 100, 40) == 4
    assert column_layers(279, 100, 40) == 3


def test_a_nest_deeper_than_half_the_height_goes_in_pairs():
    # 120 mm cups nesting 100. A third cup sunk into the second would overlap the first
    # by 80, so each pair is 140 tall and the next pair rests on it.
    assert [column_layer_z(layer, 120, 100) for layer in range(5)] == [0, 20, 140, 160, 280]
    assert column_layers(400, 120, 100) == 5
    assert column_layers(399, 120, 100) == 4
    assert column_layers(140, 120, 100) == 2
    assert column_layers(119, 120, 100) == 0


def test_a_nest_of_exactly_half_the_height_still_chains():
    assert [column_layer_z(layer, 100, 50) for layer in range(3)] == [0, 50, 100]
    assert column_layers(200, 100, 50) == 3
