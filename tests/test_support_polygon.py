"""Convex hull and tipping geometry, checked against hand-computed values.

Overlap area is not the same as stability: an item can be supported over enough area
and still overhang its own centre of gravity. Every value here was worked out by hand
first, not derived from the code under test.
"""

from __future__ import annotations

from packvium import AxisAlignedBox, Dimensions, Length, Point
from packvium.support_polygon import (contact_hull_points, convex_hull, doubled_centroid,
                                      eight_times_area, point_in_hull)


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


# ---------------------------------- exact support-polygon area

def test_the_area_of_a_hull_is_eight_times_the_true_one():
    """Hand-computed, and the factor is the whole reason this needs a test.

    `contact_hull_points` doubles every coordinate, which scales area by four, and the
    shoelace sum is twice an area. A 100x100 footprint therefore reports 80,000 rather
    than 10,000, and any caller comparing against a base area must scale it the same way.
    """
    candidate = box(0, 0, 100, 100)
    whole = convex_hull(contact_hull_points(candidate, [candidate]))
    assert eight_times_area(whole) == 8 * 100 * 100

    half = convex_hull(contact_hull_points(candidate, [box(0, 0, 100, 50)]))
    assert eight_times_area(half) == 8 * 100 * 50


def test_a_degenerate_hull_has_no_area():
    """A single contact point and a razor-thin strip are real placements, not errors."""
    candidate = box(0, 0, 100, 100)
    assert eight_times_area(convex_hull([])) == 0
    assert eight_times_area(convex_hull([(0, 0)])) == 0
    assert eight_times_area(convex_hull([(0, 0), (200, 0)])) == 0
    # A supporter meeting the candidate along an edge alone contributes no area.
    assert eight_times_area(convex_hull(contact_hull_points(candidate, [box(100, 0, 10, 100)]))) == 0


def test_the_hull_area_is_not_the_contact_area_and_the_rails_prove_it():
    """Two thin rails: a fifth of the base is touched, and the hull covers all of it.

    This is the distinction the partial-base polygon predicate turns on. A rule reading
    summed contact area sees 20%; a rule reading the support polygon sees 100%. Neither is
    a worse measurement of the other -- they are answers to different questions, and the
    published predicate asks the second.
    """
    candidate = box(0, 0, 100, 100)
    rails = [box(0, 0, 10, 100), box(90, 0, 10, 100)]
    contact = sum(10 * 100 for _ in rails)
    hull = convex_hull(contact_hull_points(candidate, rails))
    assert contact == 2_000
    assert eight_times_area(hull) == 8 * 100 * 100


def test_one_centred_strip_is_where_the_two_polygon_rules_part():
    """The authors' Figure 4.4: centroid inside the hull, hull under half the base."""
    candidate = box(0, 0, 100, 100)
    hull = convex_hull(contact_hull_points(candidate, [box(0, 40, 100, 20)]))
    assert point_in_hull(doubled_centroid(candidate), hull)
    assert eight_times_area(hull) == 8 * 100 * 20
    assert eight_times_area(hull) <= 8 * (100 * 100) // 2
