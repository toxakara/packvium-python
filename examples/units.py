"""Units: why there are no floats anywhere, and what you can type.

Run it:

    PYTHONPATH=src python3 examples/units.py

Packing is arithmetic about physical space, and floating point is the wrong tool for it:
`0.1 + 0.2` is famously not `0.3`, and a box that "almost" fits either fits or does not.
So every length and weight in this library is an exact integer count of ticks --
1/16000 mm for length, 1/8 microgram for weight -- and nothing in the request, the search
or the result is ever a float.

That choice is invisible until it saves you. This example shows where it does.
"""

from fractions import Fraction

from packvium import Container, Dimensions, Item, Length, Packer, PackingConfig, Weight, dimensional_weight

# ---------------------------------------------------------------------------------
# What you can type. Integers, decimals, fractions and mixed fractions, in mm, cm, m,
# in and ft -- because a spec sheet says "12 3/8 in" and retyping that as 12.375 is a
# transcription step where mistakes live.
# ---------------------------------------------------------------------------------
for text in ("30", "30.5 mm", "3/16 in", "12 3/8 in", "2 ft", "1.5 m"):
    length = Length.parse(text)
    print(f"{text:>12}  ->  {length.ticks:>12} ticks  =  {length.decimal('mm')} mm")

print()
for text in ("450 g", "12 oz", "1.5 kg", "2 3/4 lb"):
    weight = Weight.parse(text)
    print(f"{text:>12}  ->  {weight.ticks:>14} ticks  =  {weight.decimal('g')} g")

# ---------------------------------------------------------------------------------
# Common fractional inches are exact, not rounded. 1/16000 mm was chosen so that every
# binary fraction of an inch down to 1/128 lands on a whole number of ticks -- which is
# what makes an imperial spec survive a conversion to millimetres and back.
# ---------------------------------------------------------------------------------
print()
print("1 inch is", Length.TICKS_PER_INCH, "ticks, so 1/128 in is", Length.TICKS_PER_INCH // 128, "ticks exactly")
sixteenth = Length.parse("1/16 in")
print("1/16 in as a fraction of a mm:", sixteenth.as_fraction("mm"), "==", Fraction(sixteenth.ticks, Length.TICKS_PER_MM))

# 128 sixteenth-of-a-128th steps really do add back up to one inch:
print("128 x (1/128 in) == 1 in ?",
      Length(Length.parse("1/128 in").ticks * 128) == Length.parse("1 in"))

# And the honest other half: a *third* of an inch is not a binary fraction, so it does
# not land on a whole tick. The library rounds it, deterministically and visibly, rather
# than carrying an error that only shows up as a box that "almost" fits.
third = Length.parse("1/3 in")
print("  3 x (1/3 in) == 1 in ?", Length(third.ticks * 3) == Length.parse("1 in"),
      f"-- 1/3 in is {Length.TICKS_PER_INCH}/3 ticks, which is not an integer")

# ---------------------------------------------------------------------------------
# Where floats would actually bite. The classic case, and the same three tenths handled
# as lengths instead.
# ---------------------------------------------------------------------------------
print()
print("0.1 + 0.2 == 0.3 in floats:", 0.1 + 0.2 == 0.3)
print("the same three tenths as lengths:",
      Length.parse("0.1 mm").ticks + Length.parse("0.2 mm").ticks == Length.parse("0.3 mm").ticks)

# ---------------------------------------------------------------------------------
# The same exactness decides whether something fits. A 100mm cube into a 100mm cube is a
# fit, not a coin toss -- and one tick over is a refusal, with no tolerance to tune.
# ---------------------------------------------------------------------------------
print()
opening = Dimensions.mm("100", "100", "100")
print("exactly 100mm fits:      ", Dimensions.mm("100", "100", "100").fits_inside(opening))
one_tick_over = Dimensions(Length(opening.length.ticks + 1), opening.width, opening.height)
print("one tick over does not:  ", one_tick_over.fits_inside(opening))

# ---------------------------------------------------------------------------------
# Dimensional weight, the number carriers actually bill on. Volume divided by a divisor,
# rounded *up* -- never down, and never through a float. This is the same helper the
# `shipping_cost` and `lowest_landed_cost` objectives use, so a quote you compute here
# and a container the solver picks cannot disagree about what is being priced.
# ---------------------------------------------------------------------------------
print()
box = Dimensions.mm("400", "400", "400")
billed = dimensional_weight(box, divisor=5000, length_unit="cm", weight_unit="kg")
print("400mm cube = 64,000 cm^3 / 5,000 =", billed.decimal("kg"), "kg dimensional weight")

# ---------------------------------------------------------------------------------
# And it survives a round trip through JSON, because the wire format carries the decimal
# string rather than a float. What you typed is what the other language reads.
# ---------------------------------------------------------------------------------
print()
print("on the wire:", Length.parse("12 3/8 in").to_dict(), Weight.parse("2 3/4 lb").to_dict())

# An example must not change answer merely because the host was busy. This solve needs
# a fraction of the budget below; the generous wall-clock value is only a safety fuse, so
# a loaded machine cannot cut the multi-start portfolio short and let a different start win.
result = Packer(PackingConfig.balanced(time_limit_ms=60_000)).pack(
    [Item.create("shelf", Dimensions.inches("12 3/8", "9 1/2", "3/4"), "2 3/4 lb", quantity=3)],
    [Container.create("carton", Dimensions.inches("13", "10", "4"), max_payload="20 lb")],
)
print("packed", sum(len(c.placements) for c in result.containers), "shelves,", len(result.unpacked), "left over")
