"""Nested packing: cartons into a pallet, in one call.

Run it:

    PYTHONPATH=src python3 examples/nested.py

Real fulfilment is rarely one level. Units go into cartons, cartons go onto a pallet, and
sometimes pallets go into a trailer. `NestedPacker` runs those levels in order and feeds
each level's *packed containers* into the next level as items -- a carton that came out
of level one arrives at level two as a box with its own outer dimensions and its total
packed weight.

The levels stay independent on purpose. Level two does not reach back and repack level
one to get a better pallet, because that would make the carton contents depend on the
pallet, and a carton you already taped shut cannot be repacked. To explore that
trade-off, `NestedPacker(level_beam_width=3)` carries the three best carton sets up a
level and keeps the one whose pallet scores best.
"""

from packvium import (
    Container,
    Dimensions,
    Item,
    NestedPacker,
    PackingConfig,
    PackingLevel,
)

# What the customer ordered.
items = [
    Item.create("mug", Dimensions.mm("120", "120", "100"), "400 g", quantity=24),
    Item.create("plate", Dimensions.mm("260", "260", "20"), "600 g", quantity=16),
]

levels = [
    # Level 1: choose cartons. `outer_dimensions` matters here -- the next level packs
    # the *outside* of this carton, including its wall thickness.
    PackingLevel(
        "carton",
        (
            Container.create(
                "box-s",
                Dimensions.mm("300", "300", "300"),
                outer_dimensions=Dimensions.mm("310", "310", "310"),
                tare_weight="300 g",
                max_payload="15 kg",
                cost_minor=120,
            ),
            Container.create(
                "box-l",
                Dimensions.mm("400", "400", "400"),
                outer_dimensions=Dimensions.mm("412", "412", "412"),
                tare_weight="500 g",
                max_payload="25 kg",
                cost_minor=180,
            ),
        ),
        # An example must not change answer merely because it is executed under a
        # profiler or coverage tool. One deterministic start is sufficient to teach
        # nested packing, and the generous wall-clock value remains only a safety fuse.
        config=PackingConfig.fast(time_limit_ms=60_000),
    ),
    # Level 2: put those cartons on a pallet. The deck is the inner dimension and the
    # usable stack height is the rest.
    PackingLevel(
        "pallet",
        (
            Container.create(
                "euro",
                Dimensions.mm("1200", "800", "1400"),
                tare_weight="25 kg",
                max_payload="700 kg",
                cost_minor=1500,
            ),
        ),
    ),
]

result = NestedPacker().pack(items, levels)

# `result.levels` is one PackingResult per level, in the order you supplied them, so zip
# it back against the level names you chose.
for level, packed_level in zip(levels, result.levels):
    print(f"== {level.name} ==")
    print(f"   status: {packed_level.status.value}")
    print(f"   containers used: {len(packed_level.containers)}")
    for packed in packed_level.containers:
        print(f"     {packed.container.id:8s} holding {len(packed.placements)} item(s)")
    if packed_level.unpacked:
        print(f"   left over: {len(packed_level.unpacked)}")
    print()

cartons = len(result.levels[0].containers)
pallets = len(result.levels[-1].containers)
print(f"{cartons} carton(s) travelling on {pallets} pallet(s)")

# The run above is greedy: level one keeps its own best carton set and level two packs
# whatever it was handed. `level_beam_width=K` carries the K best carton sets up instead,
# and the pallet's score -- the outermost level's -- picks the winner. A width of 1 is the
# greedy chain, answer for answer.
print()
for width in (1, 3):
    chain = NestedPacker(level_beam_width=width).pack(items, levels)
    carton_level, pallet_level = chain.levels
    cartons_used = ", ".join(packed.container.id for packed in carton_level.containers)
    print(f"level_beam_width={width}: cartons {cartons_used} "
          f"(cost {carton_level.score[2]}), pallet unused volume {pallet_level.score[3]} ppm")

# Under a width of 3 the carton level is dearer on its own -- two large boxes instead of a
# large and a small one -- and it wins anyway, because the pallet built from those cartons
# wastes less space. That is the trade the greedy chain cannot see.
