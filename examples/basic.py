"""The smallest useful call: some items, some boxes, one answer.

Run it:

    PYTHONPATH=src python3 examples/basic.py

Three things are worth noticing in eight lines of code.

`Dimensions.mm` and `Dimensions.inches` are both exact -- "4 in" is not converted to a
rounded number of millimetres, it is stored as an exact tick count, so an imperial spec
sheet and a metric container agree without a tolerance to tune (see units.py).

`keep_upright` is a rule, not a hint. The mug will never be laid on its side, and if
that makes it not fit you are told which item failed and why, rather than getting a
plausible-looking arrangement that spills coffee.

`cost_minor` is what the box costs *you*, in minor currency units. The default objective
ignores it -- it opens as few containers as possible and packs them tightly. Ranking by
packaging cost, by carrier-billed weight or by actual money is one setting away; that is
what objectives.py is for.
"""

from packvium import Container, Dimensions, Item, Packer, PackingConfig

result = Packer(PackingConfig.balanced()).pack(
    [
        Item.create("book", Dimensions.mm("210", "140", "30"), "450 g", quantity=4),
        Item.create("mug", Dimensions.inches("4", "4", "5"), "12 oz", quantity=2, keep_upright=True),
    ],
    [
        Container.create("box-m", Dimensions.mm("400", "300", "250"), max_payload="20 kg", cost_minor=180),
        Container.create("box-l", Dimensions.mm("500", "400", "350"), max_payload="30 kg", cost_minor=250),
    ],
)

print("status    ", result.status.value)
print("containers", [c.container.id for c in result.containers])
print("packed    ", sum(len(c.placements) for c in result.containers), "of 6")
print("unpacked  ", [(u.instance.item.id, u.reason) for u in result.unpacked])
print("score     ", result.score, " <- lexicographic, exact integers, lower is better")
print()
print("the full result as a plain dict is what serialization.py explores:")
print(sorted(result.to_dict()))
