"""Two-axle weight distribution, checked against hand-computed values.

Two-point beam statics: taking moments about each axle gives an exact fraction for
what the other axle carries. Every value here was worked out by hand first, not
derived from the code under test.
"""

from __future__ import annotations

from packvium import AxisAlignedBox, Axle, Dimensions, Length, Point, Weight
from packvium.axle_load import axle_load_exceeded, axle_reactions
from packvium.constraints import LoadUnit


def unit(weight: int, x: int, length: int, label: str = "u") -> LoadUnit:
    box = AxisAlignedBox(Point(x, 0, 0), Dimensions(Length(length), Length(1000), Length(100)))
    return LoadUnit(box, weight, None, None, label)


def test_a_load_centred_between_the_axles_splits_evenly():
    # A single item spanning the whole container has its own centre at x=500,
    # exactly midway between axles at 100 and 900 -- each axle bears half of 800.
    axles = (Axle(Length(100), Weight(500)), Axle(Length(900), Weight(500)))
    assert axle_load_exceeded(axles, [unit(800, 0, 1000)]) is None


def test_a_load_resting_exactly_on_the_front_axle_puts_nothing_on_the_rear():
    # The item's own centre (x=100) coincides with the front axle -- by lever
    # physics the rear axle bears none of it, so a zero-limit rear axle still passes.
    axles = (Axle(Length(100), Weight(800)), Axle(Length(900), Weight(0)))
    assert axle_load_exceeded(axles, [unit(800, 0, 200)]) is None


def test_a_load_resting_exactly_on_the_front_axle_would_overload_a_lighter_front_limit():
    axles = (Axle(Length(100), Weight(799)), Axle(Length(900), Weight(800)))
    assert axle_load_exceeded(axles, [unit(800, 0, 200)]) == ("axle_overloaded", "front")


def test_the_rear_axle_boundary_is_exact_to_the_tick():
    # Item centred at x=500 (weight 800) over axles at 100/900 puts exactly 400 on
    # each axle -- a rear limit of 400 passes, one tick less does not.
    passing = (Axle(Length(100), Weight(1000)), Axle(Length(900), Weight(400)))
    failing = (Axle(Length(100), Weight(1000)), Axle(Length(900), Weight(399)))
    load = [unit(800, 0, 1000)]
    assert axle_load_exceeded(passing, load) is None
    assert axle_load_exceeded(failing, load) == ("axle_overloaded", "rear")


def test_two_items_combine_by_superposition():
    # 400 at x=100 (all front) plus 400 at x=900 (all rear) -- front gets 400,
    # rear gets 400, neither axle sees the other item's contribution.
    axles = (Axle(Length(100), Weight(400)), Axle(Length(900), Weight(400)))
    at_front = unit(400, 20, 160, "front")  # centre exactly 100
    at_rear = unit(400, 820, 160, "rear")  # centre exactly 900
    assert axle_load_exceeded(axles, [at_front, at_rear]) is None


def test_no_limit_on_an_axle_means_that_axle_is_never_checked():
    axles = (Axle(Length(100), None), Axle(Length(900), Weight(0)))
    # All the weight sits at the front axle's position, so the (unlimited) front
    # axle would bear everything and the (zero-limit) rear axle bears nothing.
    assert axle_load_exceeded(axles, [unit(10_000, 20, 160)]) is None


def test_no_units_means_no_load_and_no_violation():
    axles = (Axle(Length(100), Weight(0)), Axle(Length(900), Weight(0)))
    assert axle_load_exceeded(axles, []) is None


def test_centred_tare_is_part_of_the_gross_reaction_and_is_reported_exactly():
    axles = (Axle(Length(100), Weight(100)), Axle(Length(900), Weight(100)))
    assert axle_reactions(axles, [], tare_weight_ticks=200, tare_doubled_center_x=1000) == (
        1600, 160_000, 160_000,
    )
    assert axle_load_exceeded(
        axles, [], tare_weight_ticks=200, tare_doubled_center_x=1000
    ) is None


def test_a_load_outside_the_axle_span_has_a_negative_opposite_reaction():
    axles = (Axle(Length(100), None), Axle(Length(900), None))
    denominator, front, rear = axle_reactions(axles, [unit(800, 900, 200)])
    assert denominator == 1600
    assert front < 0
    assert rear > 0
