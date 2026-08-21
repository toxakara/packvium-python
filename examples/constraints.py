"""Constraints: how to say "this may not go there" and get told why.

Run it:

    PYTHONPATH=src python3 examples/constraints.py

The solver's job is not only to fit boxes. Most real packing rules are refusals -- this
side up, nothing on top of that, keep the chemicals away from the food -- and the useful
part of the answer is often the item that did *not* fit and the reason it did not.

Every constraint here is a field on `Item` or `Container`. None of them needs a custom
class, and none of them changes how you call `pack`.
"""

from packvium import (
    Container,
    Dimensions,
    Item,
    Length,
    Packer,
    PackingConfig,
    Rotation,
    explain_unpacked_item,
)

items = [
    # `keep_upright` forbids every rotation that would tip the item over. An open tub of
    # paint is the usual reason.
    Item.create(
        "paint",
        Dimensions.mm("200", "200", "250"),
        "5 kg",
        quantity=2,
        keep_upright=True,
    ),
    # `must_be_on_floor` keeps the item on the container floor, and `max_top_load` caps
    # what may rest directly on it. Note "directly": this is not a whole-stack limit.
    Item.create(
        "glass-panel",
        Dimensions.mm("400", "300", "40"),
        "8 kg",
        quantity=1,
        must_be_on_floor=True,
        max_top_load="2 kg",
    ),
    # `stackable=False` means nothing may be placed on this item at all.
    Item.create(
        "cake",
        Dimensions.mm("250", "250", "150"),
        "1 kg",
        quantity=1,
        stackable=False,
    ),
    # Tags are how two items refuse each other. `incompatible_tags` is checked both ways,
    # so tagging one side is enough.
    Item.create(
        "bleach",
        Dimensions.mm("120", "120", "300"),
        "2 kg",
        quantity=2,
        tags=("hazmat",),
        incompatible_tags=("food",),
    ),
    Item.create(
        "flour",
        Dimensions.mm("200", "150", "100"),
        "1500 g",
        quantity=3,
        tags=("food",),
    ),
    # Longer than the crate's longest inner edge in every orientation, so no solver can
    # place it. It is here to show what a refusal looks like.
    Item.create("ladder", Dimensions.mm("1800", "300", "100"), "6 kg", quantity=1),
]

containers = [
    # `max_payload` is the weight the container may carry, excluding its own tare.
    Container.create(
        "crate",
        Dimensions.mm("600", "500", "500"),
        tare_weight="3 kg",
        max_payload="25 kg",
        cost_minor=400,
    ),
]

result = Packer(PackingConfig.balanced()).pack(items, containers)


def millimetres(ticks: int) -> str:
    """Positions are exact integers in 1/16000 mm; render them for a human."""
    return f"{ticks / Length.TICKS_PER_MM:g}"


print(f"status: {result.status.value}")
print(f"containers opened: {len(result.containers)}")

for index, packed in enumerate(result.containers, start=1):
    print(f"\ncrate #{index} ({packed.container.id}): {len(packed.placements)} placement(s)")
    for placement in packed.placements:
        position = placement.position
        print(
            f"  {placement.instance.item.id:12s} at "
            f"({millimetres(position.x)}, {millimetres(position.y)}, {millimetres(position.z)}) mm"
        )

# Two crates for a load that would fit in one by volume: `bleach` is tagged `hazmat` and
# refuses `food`, so it cannot share a container with `flour`. Nothing asked the solver
# to open a second crate -- the constraint did.

# The refusals are the interesting half. `explain_unpacked_item` turns the structured
# reason into a sentence, so you can show a human why their order will not ship as one
# box without teaching them the reason codes.
if result.unpacked:
    print("\nnot packed:")
    for unpacked in result.unpacked:
        print(f"  {unpacked.instance.item.id:12s} {explain_unpacked_item(unpacked)}")
else:
    print("\neverything fitted -- widen the crate or add items to see a refusal explained")




# ------------------------------------------------------------- one rule at a time
#
# The four rules below are each shown twice: the same items, the same container, once
# without the rule and once with it. A constraint you cannot watch change the answer is
# one the reader has to take on faith, and the pair makes the rule -- rather than the
# geometry -- provably the reason.
#
# Note what "the rule bit" looks like. Only sometimes is it a refusal; more often the
# solver satisfies the rule by opening another container, which costs money and is the
# answer you actually wanted to see coming. So both numbers are printed.

def compare(rule: str, without: list[Item], with_rule: list[Item], containers: list[Container]) -> None:
    print(f"\n{rule}")
    for label, variant in (("without the rule", without), ("with the rule   ", with_rule)):
        outcome = Packer(PackingConfig.balanced()).pack(variant, containers)
        placements = sum(len(container.placements) for container in outcome.containers)
        print(
            f"  {label}: {len(outcome.containers)} container(s), "
            f"{placements} placed, {len(outcome.unpacked)} refused"
        )
        for unpacked in outcome.unpacked:
            print(f"      {explain_unpacked_item(unpacked)}")


shelf = [Container.create("shelf", Dimensions.mm("800", "400", "500"), max_payload="40 kg")]

# `allowed_rotations` narrows the six orientations to the ones you permit, and
# `Rotation.upright()` is the pair that keeps the item's own height vertical -- what you
# want for anything with a printed face or an open top. The pole is 700 mm tall and the
# shelf is 500 mm deep, so it fits only by being laid down, which is what this forbids.
pole = Dimensions.mm("90", "90", "700")
compare(
    "allowed_rotations -- a pole that only fits lying down, forbidden from lying down",
    [Item.create("pole", pole, "1 kg")],
    [Item.create("pole", pole, "1 kg", allowed_rotations=Rotation.upright())],
    shelf,
)

# `max_stacked_items` caps how many units may sit above one item -- a pallet-pattern
# rule ("three high, no more"), not a weight limit. The column below is one tin wide, so
# height is the only way to fit more, and the second container is the price of the cap.
column = [Container.create("column", Dimensions.mm("160", "160", "600"), max_payload="40 kg")]
tin = Dimensions.mm("150", "150", "120")
compare(
    "max_stacked_items -- five tins fit in one column; three-high needs two columns",
    [Item.create("tin", tin, "800 g", quantity=5)],
    [Item.create("tin", tin, "800 g", quantity=5, max_stacked_items=3)],
    column,
)

# `minimum_support_ratio` is how much of an item's base must rest on something solid.
# The plinth stands on the floor and covers a quarter of the ledge, and the ledge is too
# shallow for the slab to stand on edge -- so the only place the slab fits is perched on
# the plinth, on a quarter of its base. At 0.9 that is refused and a second ledge opens.
ledge = [Container.create("ledge", Dimensions.mm("400", "400", "350"), max_payload="40 kg")]
plinth = Item.create("plinth", Dimensions.mm("200", "200", "300"), "5 kg", must_be_on_floor=True)
slab = Dimensions.mm("400", "400", "60")
compare(
    "minimum_support_ratio -- a slab perched on a quarter of its base",
    [plinth, Item.create("slab", slab, "9 kg")],
    [plinth, Item.create("slab", slab, "9 kg", minimum_support_ratio=0.9)],
    ledge,
)

# `group` is atomic: every member ships in one container or none of them does. The third
# part is deliberately too long for the shelf, so it takes the other two down with it
# rather than shipping two thirds of an assembly nobody can use.
parts = [
    Dimensions.mm("200", "200", "100"),
    Dimensions.mm("200", "200", "100"),
    Dimensions.mm("900", "100", "100"),
]
compare(
    "group -- one member cannot be placed, so none of them is",
    [Item.create(f"kit-{n}", d, "2 kg") for n, d in enumerate(parts, start=1)],
    [Item.create(f"kit-{n}", d, "2 kg", group="assembly") for n, d in enumerate(parts, start=1)],
    shelf,
)

# Every reason code above is a fact about the request, not a solver failure -- which is
# why `explain_unpacked_item` can turn it into a sentence a customer is allowed to read.
