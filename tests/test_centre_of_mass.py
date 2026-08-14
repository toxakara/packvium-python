"""Centre of mass and its offset, checked against hand-computed values.

Needed for axle load and side-to-side balance, both of which care about the worst
horizontal axis, not a single blended distance -- which is also why this is a
Chebyshev offset (the worse of the two axis ratios), not a Euclidean one: a square
root would break the exact-integer arithmetic this library is built on.
"""

from __future__ import annotations

from packvium import Container, Dimensions, Item, Length, PackedContainer, Placement, Point, Rotation, Weight
from packvium.centre_of_mass import centre_of_mass_offset_ppm

INNER = Dimensions(Length(1000), Length(1000), Length(100))


def placement(weight: int, x: int, length: int, y: int = 0, width: int = 1000) -> Placement:
    dims = Dimensions(Length(length), Length(width), Length(100))
    one, = Item.create(f"i{x}-{y}", dims, weight).instances()
    position = Point(x, y, 0)
    return Placement(one, position, Rotation.LWH, dims, position, dims)


def test_a_single_item_flush_against_one_wall_gives_an_exact_hand_checked_offset():
    # Container centre is x=500; the item's own centre is x=100 (0 + 200/2), a
    # distance of 400 out of a 500 half-length -- 80% exactly, and 0% on y since the
    # item spans the full width and is therefore already centred on that axis.
    placements = [placement(weight=1, x=0, length=200)]
    assert centre_of_mass_offset_ppm(INNER, placements) == 800_000


def test_two_symmetric_items_cancel_to_a_centred_offset():
    placements = [placement(weight=1, x=0, length=100), placement(weight=1, x=900, length=100)]
    assert centre_of_mass_offset_ppm(INNER, placements) == 0


def test_a_heavier_item_pulls_the_centre_of_mass_toward_it():
    """Unequal weights, not just unequal positions, move the centre of mass."""
    light = placement(weight=1, x=0, length=100)
    heavy = placement(weight=3, x=900, length=100)
    # centre_x = (1*50 + 3*950) / 4 = 725; container centre 500; offset 225/500 = 45%.
    assert centre_of_mass_offset_ppm(INNER, [light, heavy]) == 450_000


def test_the_worse_axis_wins_not_a_blended_distance():
    # x spans the full length, so it is centred (0%); y spans [0, 200], centre 100,
    # a distance of 400 out of a 500 half-width -- 80%. The Chebyshev offset reports
    # that 80%, not some smaller blend with x's 0%.
    dims = Dimensions(Length(1000), Length(200), Length(100))
    one, = Item.create("a", dims, 1).instances()
    position = Point(0, 0, 0)
    off_center_y = Placement(one, position, Rotation.LWH, dims, position, dims)
    assert centre_of_mass_offset_ppm(INNER, [off_center_y]) == 800_000


def test_no_placements_report_a_zero_offset():
    assert centre_of_mass_offset_ppm(INNER, []) == 0


def test_massless_placements_report_a_zero_offset_rather_than_dividing_by_zero():
    weightless = placement(weight=0, x=0, length=100)
    assert centre_of_mass_offset_ppm(INNER, [weightless]) == 0


def test_the_packed_container_property_matches_the_free_function():
    box = Container.create("c", INNER)
    placements = (placement(weight=1, x=0, length=200),)
    packed = PackedContainer(box, 1, placements)
    assert packed.centre_of_mass_offset_ppm == centre_of_mass_offset_ppm(INNER, placements)
