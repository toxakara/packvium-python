"""Nesting overlap accounting, checked against hand-computed values.

A naive sum of each placement's own volume overstates how much space a nested
column actually fills, since neighbouring nested layers share part of the same
physical space. Every value here was worked out by hand first, not derived from
the code under test.
"""

from __future__ import annotations

from packvium import Dimensions, Item, Length, Placement, Point, Rotation, Weight
from packvium.nesting import is_valid_nesting, used_volume, used_volume_delta


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
