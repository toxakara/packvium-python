"""Exact fixed-point arithmetic — the foundation every other guarantee rests on.

One length tick is 1/16000 mm and one weight tick is 1/8 microgram. If a value ever
reaches a solver having passed through binary floating point, feasibility decisions
stop being reproducible, so the parsers reject floats outright and round exactly.
See docs/UNITS-AND-NUMERICS.md.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from packvium import Length, Rounding, Weight
from packvium.units import _fraction, _rounded


# --------------------------------------------------------------------- constants

def test_length_tick_definitions():
    assert Length.mm(1).ticks == 16_000
    assert Length.of(1, "cm").ticks == 160_000
    assert Length.of(1, "m").ticks == 16_000_000
    assert Length.inches(1).ticks == 406_400
    assert Length.of(1, "ft").ticks == 406_400 * 12


def test_inch_is_exactly_25_point_4_millimetres():
    """The whole point of the tick size: the imperial/metric bridge is not approximate."""
    assert Length.inches(1) == Length.mm("25.4")


def test_weight_tick_definitions():
    assert Weight.of(1, "mg").ticks == 8_000
    assert Weight.of(1, "g").ticks == 8_000_000
    assert Weight.of(1, "kg").ticks == 8_000_000_000
    assert Weight.of(1, "lb").ticks == 3_628_738_960


def test_imperial_weight_conversions_are_exact():
    assert Weight.of(16, "oz") == Weight.of(1, "lb")
    assert Weight.of(1, "lb") == Weight.of("453.59237", "g")
    assert Weight.of(1, "oz") == Weight.of("28.349523125", "g")


# ------------------------------------------------------------------------ parsing

@pytest.mark.parametrize(
    ("text", "ticks"),
    [
        ("1", 406_400),
        ("1/2", 203_200),
        ("1/128", 3_175),
        ("12 3/8", 5_029_200),
        ("0.5", 203_200),
        ("+1", 406_400),
    ],
)
def test_inch_notations(text, ticks):
    assert Length.inches(text).ticks == ticks


def test_mixed_fraction_takes_its_sign_from_the_text():
    """"-0 1/2" is minus one half; the sign cannot be read off a whole part of zero."""
    assert _fraction("-0 1/2") == Fraction(-1, 2)
    assert _fraction("-1 1/2") == Fraction(-3, 2)
    assert _fraction("1 1/2") == Fraction(3, 2)


def test_accepted_scalar_types():
    assert Length.of(2, "mm") == Length.of("2", "mm")
    assert Length.of(Decimal("2.5"), "mm") == Length.of("2.5", "mm")
    assert Length.of(Fraction(5, 2), "mm") == Length.of("2.5", "mm")


def test_unit_suffix_in_the_string():
    assert Length.parse("4 in") == Length.inches(4)
    assert Length.parse("100mm") == Length.mm(100)
    assert Length.parse("2 FT") == Length.of(2, "ft")
    assert Weight.parse("1 kg") == Weight.of(1, "kg")
    assert Weight.parse("1lbs") == Weight.of(1, "lb")


def test_dictionary_form():
    assert Length.parse({"value": "4", "unit": "in"}) == Length.inches(4)
    assert Length.parse({"value": "4"}, default_unit="in") == Length.inches(4)
    assert Weight.parse({"value": "2", "unit": "kg"}) == Weight.of(2, "kg")


def test_default_unit_applies_to_a_bare_number():
    assert Length.parse("100") == Length.mm(100)
    assert Length.parse("100", default_unit="in") == Length.inches(100)


def test_already_parsed_values_pass_through():
    length = Length.mm(10)
    assert Length.parse(length) is length


# ------------------------------------------------------------------------ rounding

@pytest.mark.parametrize(
    ("rounding", "ticks"),
    [(Rounding.FLOOR, 1), (Rounding.CEIL, 2), (Rounding.NEAREST, 1)],
)
def test_rounding_modes(rounding, ticks):
    """1.4 ticks: floor keeps 1, ceil takes 2, nearest agrees with floor here."""
    assert Length.of("1.4", "tick", rounding).ticks == ticks


def test_nearest_breaks_ties_to_even():
    """Half-to-even, so a long run of .5 values does not drift upwards."""
    assert Length.of("0.5", "tick").ticks == 0
    assert Length.of("1.5", "tick").ticks == 2
    assert Length.of("2.5", "tick").ticks == 2
    assert Length.of("3.5", "tick").ticks == 4


def test_rounding_of_negative_values():
    """Not reachable through Length or Weight, which forbid negatives, but the helper
    is shared and floor must mean floor rather than truncation towards zero."""
    assert _rounded(Fraction(-3, 2), Rounding.FLOOR) == -2
    assert _rounded(Fraction(-3, 2), Rounding.CEIL) == -1
    assert _rounded(Fraction(-3, 2), Rounding.NEAREST) == -2


def test_exact_values_are_untouched_by_every_mode():
    for rounding in Rounding:
        assert Length.mm(7, rounding).ticks == 112_000


# ------------------------------------------------------------------------- refusals

def test_float_input_is_rejected():
    """A float has already lost the exactness the library promises, so it never enters."""
    with pytest.raises(TypeError):
        Length.parse(1.2)
    with pytest.raises(TypeError):
        Weight.parse(1.2)


def test_unsupported_units_are_rejected():
    with pytest.raises(ValueError):
        Length.of(1, "furlong")
    with pytest.raises(ValueError):
        Weight.of(1, "stone")
    with pytest.raises(ValueError):
        Length.mm(1).as_fraction("furlong")


def test_negative_measures_are_rejected():
    with pytest.raises(ValueError):
        Length(-1)
    with pytest.raises(ValueError):
        Weight(-1)


def test_absurdly_precise_decimals_are_rejected():
    """A shared cap with PHP, which cannot hold 10**19 in an integer. Diverging on the
    accepted range between languages is exactly what this library exists to avoid."""
    assert Length.of("0." + "1" * 18, "mm").ticks >= 0
    with pytest.raises(ValueError):
        Length.of("0." + "1" * 19, "mm")


@pytest.mark.parametrize("text", ["", "abc", "1 2 3", "--1", "1/2/3"])
def test_unparseable_text_is_rejected(text):
    with pytest.raises(Exception):
        Length.mm(text)


# ---------------------------------------------------------------------- projection

def test_ordering_is_by_tick_count():
    assert Length.mm(1) < Length.mm(2)
    assert max(Weight.of(1, "kg"), Weight.of(999, "g")) == Weight.of(1, "kg")


def test_as_fraction_is_exact():
    assert Length.inches(1).as_fraction("mm") == Fraction(127, 5)
    assert Length.mm(100).as_fraction("cm") == Fraction(10)


@pytest.mark.parametrize("text", ["100", "25.4", "0.0625"])
def test_decimal_rendering_round_trips(text):
    assert Length.mm(Length.mm(text).decimal("mm")) == Length.mm(text)


def test_to_dict_carries_the_exact_ticks_alongside_the_display_value():
    assert Length.inches(1).to_dict("mm") == {"ticks": 406_400, "value": "25.4", "unit": "mm"}
    assert Weight.of(1, "kg").to_dict("g") == {"ticks": 8_000_000_000, "value": "1000", "unit": "g"}
