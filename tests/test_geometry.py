"""Integer geometry: rotations, containment and the half-open box convention.

Every collision decision in the library is made on these primitives with plain
integers. The half-open convention — a box owns `[origin, origin + size)` — is what
lets two boxes share a face without colliding, so it is asserted directly rather
than assumed.
"""

from __future__ import annotations

import pytest

from packvium import AxisAlignedBox, Dimensions, Length, Point, Rotation, Weight, dimensional_weight


def box(x, y, z, length, width, height) -> AxisAlignedBox:
    return AxisAlignedBox(Point(x, y, z), Dimensions(Length(length), Length(width), Length(height)))


# --------------------------------------------------------------------- dimensions

def test_dimensions_must_be_positive():
    for bad in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        with pytest.raises(ValueError):
            Dimensions(Length(bad[0]), Length(bad[1]), Length(bad[2]))


def test_derived_measures():
    dims = Dimensions.mm(2, 3, 5)
    assert dims.volume == 32_000 * 48_000 * 80_000
    assert dims.base_area == 32_000 * 48_000
    assert dims.max_edge == 80_000


def test_from_dict_accepts_mixed_notations():
    dims = Dimensions.from_dict({"length": {"value": "1", "unit": "in"}, "width": "10", "height": 20})
    assert dims.length == Length.inches(1)
    assert dims.width == Length.mm(10)
    assert dims.height == Length.mm(20)


# ---------------------------------------------------------------------- rotations

@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (Rotation.LWH, (2, 3, 5)),
        (Rotation.LHW, (2, 5, 3)),
        (Rotation.WLH, (3, 2, 5)),
        (Rotation.WHL, (3, 5, 2)),
        (Rotation.HLW, (5, 2, 3)),
        (Rotation.HWL, (5, 3, 2)),
    ],
)
def test_every_rotation_is_the_permutation_its_name_spells(rotation, expected):
    rotated = Dimensions.mm(2, 3, 5).rotated(rotation)
    assert (rotated.length, rotated.width, rotated.height) == tuple(Length.mm(v) for v in expected)


def test_rotation_preserves_volume():
    dims = Dimensions.mm(2, 3, 5)
    assert all(dims.rotated(r).volume == dims.volume for r in Rotation.all())


@pytest.mark.parametrize(
    ("dimensions", "count"),
    [
        (Dimensions.mm(2, 3, 5), 6),  # three distinct edges
        (Dimensions.mm(4, 4, 7), 3),  # square base
        (Dimensions.mm(4, 4, 4), 1),  # cube
    ],
)
def test_unique_rotations_collapses_duplicate_shapes(dimensions, count):
    """Trying the same physical shape six times multiplies the search for nothing."""
    assert len(dimensions.unique_rotations(Rotation.all())) == count


def test_unique_rotations_keeps_the_first_name_for_a_shape():
    (rotation, _), = Dimensions.mm(4, 4, 4).unique_rotations(Rotation.all())
    assert rotation is Rotation.LWH


def test_upright_rotations_keep_the_height_axis():
    dims = Dimensions.mm(2, 3, 5)
    assert all(dims.rotated(r).height == dims.height for r in Rotation.upright())


# ---------------------------------------------------------------------- containment

def test_fits_inside_is_per_axis_and_inclusive():
    assert Dimensions.mm(10, 10, 10).fits_inside(Dimensions.mm(10, 10, 10))
    assert not Dimensions.mm(11, 10, 10).fits_inside(Dimensions.mm(10, 10, 10))


def test_expand_adds_the_clearance_to_both_sides():
    grown = Dimensions.mm(10, 20, 30).expand(Length.mm(2))
    assert grown == Dimensions.mm(14, 24, 34)


def test_expanding_by_zero_is_the_identity():
    dims = Dimensions.mm(10, 20, 30)
    assert dims.expand(Length(0)) == dims


# -------------------------------------------------------------------------- points

def test_points_cannot_be_negative():
    for bad in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
        with pytest.raises(ValueError):
            Point(*bad)


def test_points_order_by_x_then_y_then_z():
    assert sorted([Point(1, 0, 0), Point(0, 1, 0), Point(0, 0, 1)]) == [
        Point(0, 0, 1), Point(0, 1, 0), Point(1, 0, 0)
    ]


# --------------------------------------------------------------------------- boxes

def test_far_corner_is_origin_plus_size():
    unit = box(10, 20, 30, 1, 2, 3)
    assert (unit.x2, unit.y2, unit.z2) == (11, 22, 33)


def test_touching_faces_do_not_intersect():
    """The half-open convention. Without it, a perfectly tiled container would report
    a collision between every pair of neighbours."""
    left = box(0, 0, 0, 10, 10, 10)
    for touching in (box(10, 0, 0, 10, 10, 10), box(0, 10, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)):
        assert not left.intersects(touching)
        assert not touching.intersects(left)


def test_a_single_overlapping_tick_is_an_intersection():
    assert box(0, 0, 0, 10, 10, 10).intersects(box(9, 9, 9, 10, 10, 10))


def test_intersection_is_symmetric_and_reflexive():
    one, other = box(0, 0, 0, 10, 10, 10), box(5, 5, 5, 10, 10, 10)
    assert one.intersects(other) == other.intersects(one)
    assert one.intersects(one)


def test_containment_allows_a_flush_fit_but_not_an_overhang():
    outer = box(0, 0, 0, 10, 10, 10)
    assert outer.contains(box(0, 0, 0, 10, 10, 10))
    assert outer.contains(box(1, 1, 1, 8, 8, 8))
    assert not outer.contains(box(1, 1, 1, 10, 10, 10))


def test_contains_point_excludes_the_far_faces():
    unit = box(0, 0, 0, 10, 10, 10)
    assert unit.contains_point(Point(0, 0, 0))
    assert unit.contains_point(Point(9, 9, 9))
    assert not unit.contains_point(Point(10, 0, 0))


def test_overlap_area_ignores_height():
    """Support is decided by the footprint two boxes share, so this projection is
    deliberately two-dimensional; the caller compares `z` separately."""
    floor = box(0, 0, 0, 10, 10, 1)
    above = box(5, 0, 900, 10, 10, 1)
    assert floor.overlap_area_xy(above) == 5 * 10


def test_disjoint_and_edge_touching_footprints_have_no_area():
    floor = box(0, 0, 0, 10, 10, 1)
    assert floor.overlap_area_xy(box(20, 20, 0, 5, 5, 1)) == 0
    assert floor.overlap_area_xy(box(10, 0, 0, 5, 5, 1)) == 0


def test_overlap_area_is_symmetric():
    one, other = box(0, 0, 0, 10, 10, 1), box(3, 4, 0, 10, 10, 1)
    assert one.overlap_area_xy(other) == other.overlap_area_xy(one) == 7 * 6


# ---------------------------------------------------------------- dimensional weight

def test_dimensional_weight_matches_the_textbook_inches_and_pounds_example():
    # A classic carrier example: a 10x10x10in box at a 139 divisor weighs 1000/139 lb.
    result = dimensional_weight(Dimensions.inches(10, 10, 10), 139, "in", "lb")
    expected_ticks = (1000 * Weight.TICKS_PER_LB) // 139
    assert result.ticks == expected_ticks


def test_dimensional_weight_is_exact_for_a_round_centimetres_and_kilograms_case():
    # 40 x 30 x 20 cm at a 5000 divisor is exactly 4.8 kg -- no rounding involved.
    result = dimensional_weight(Dimensions.of(40, 30, 20, "cm"), 5000, "cm", "kg")
    assert result.ticks == Weight.of(4.8, "kg").ticks


def test_dimensional_weight_scales_inversely_with_the_divisor():
    dims = Dimensions.inches(20, 20, 20)
    lower_divisor = dimensional_weight(dims, 139, "in", "lb")
    higher_divisor = dimensional_weight(dims, 166, "in", "lb")
    assert lower_divisor.ticks > higher_divisor.ticks


def test_dimensional_weight_rejects_a_non_positive_divisor():
    with pytest.raises(ValueError):
        dimensional_weight(Dimensions.inches(1, 1, 1), 0, "in", "lb")
