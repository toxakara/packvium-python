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
