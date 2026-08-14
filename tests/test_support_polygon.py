"""Convex hull and tipping geometry, checked against hand-computed values.

Overlap area is not the same as stability: an item can be supported over enough area
and still overhang its own centre of gravity. Every value here was worked out by hand
first, not derived from the code under test.
"""

from __future__ import annotations

from packvium import AxisAlignedBox, Dimensions, Length, Point
from packvium.support_polygon import contact_hull_points, convex_hull, doubled_centroid, point_in_hull


def box(x, y, l, w, h=10) -> AxisAlignedBox:
    return AxisAlignedBox(Point(x, y, 0), Dimensions(Length(l), Length(w), Length(h)))


# ------------------------------------------------------------------- convex hull

def test_the_hull_of_a_rectangle_is_its_four_corners_counter_clockwise():
    points = [(0, 0), (80, 0), (80, 200), (0, 200)]
    assert convex_hull(points) == [(0, 0), (80, 0), (80, 200), (0, 200)]


def test_an_interior_point_is_dropped_from_the_hull():
    points = [(0, 0), (80, 0), (80, 200), (0, 200), (40, 100)]
    assert convex_hull(points) == [(0, 0), (80, 0), (80, 200), (0, 200)]


def test_duplicate_points_collapse_to_one():
    points = [(0, 0), (0, 0), (80, 0), (80, 0)]
    assert convex_hull(points) == [(0, 0), (80, 0)]


def test_a_single_point_is_its_own_degenerate_hull():
    assert convex_hull([(5, 5)]) == [(5, 5)]


def test_collinear_points_form_a_degenerate_segment_hull():
    assert convex_hull([(0, 0), (10, 0), (20, 0)]) == [(0, 0), (20, 0)]


# --------------------------------------------------------------- point in hull

def test_the_centre_of_a_square_hull_is_inside():
    hull = convex_hull([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert point_in_hull((50, 50), hull)


def test_a_point_outside_the_hull_is_rejected():
    hull = convex_hull([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert not point_in_hull((150, 50), hull)


def test_a_point_exactly_on_the_boundary_counts_as_supported():
    hull = convex_hull([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert point_in_hull((100, 50), hull)
    assert point_in_hull((0, 0), hull)


def test_a_point_matching_a_degenerate_single_point_hull_is_inside():
    assert point_in_hull((5, 5), [(5, 5)])
    assert not point_in_hull((5, 6), [(5, 5)])


def test_a_point_on_a_degenerate_segment_hull_is_inside():
    hull = [(0, 0), (20, 0)]
    assert point_in_hull((10, 0), hull)
    assert not point_in_hull((10, 1), hull)
    assert not point_in_hull((30, 0), hull)


def test_an_empty_hull_supports_nothing():
    assert not point_in_hull((0, 0), [])


# ------------------------------------------------------------- physical scenario

def test_full_support_keeps_the_centroid_inside_the_hull():
    candidate = box(0, 0, 100, 100)
    supporter = box(0, 0, 100, 100)
    hull = convex_hull(contact_hull_points(candidate, [supporter]))
    assert point_in_hull(doubled_centroid(candidate), hull)


def test_an_overhang_that_meets_an_area_ratio_can_still_tip():
    """The whole point of this rule: 40% overlap can satisfy a modest support-ratio
    threshold while leaving the candidate's own centroid entirely unsupported."""
    candidate = box(0, 0, 100, 100)
    supporter = box(0, 0, 40, 100)  # covers only the candidate's left 40%
    hull = convex_hull(contact_hull_points(candidate, [supporter]))
    assert hull == [(0, 0), (80, 0), (80, 200), (0, 200)]
    assert doubled_centroid(candidate) == (100, 100)
    assert not point_in_hull(doubled_centroid(candidate), hull)


def test_two_narrow_supporters_can_still_bracket_the_centroid():
    """Two thin rails, one on each side, support nothing directly under the middle by
    area alone, but their combined hull still spans across the centroid."""
    candidate = box(0, 0, 100, 100)
    left_rail = box(0, 0, 10, 100)
    right_rail = box(90, 0, 10, 100)
    hull = convex_hull(contact_hull_points(candidate, [left_rail, right_rail]))
    assert point_in_hull(doubled_centroid(candidate), hull)
