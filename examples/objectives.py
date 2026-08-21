"""Objectives: six ways to be "best", and the scenes where they disagree.

Run it:

    PYTHONPATH=src python3 examples/objectives.py

Every solve returns the arrangement that scores best -- but "best" is a choice, and it is
the one setting most likely to make the library look wrong when it is merely answering a
different question than you meant to ask. This example builds scenes where two objectives
genuinely pick different containers, so the difference is visible rather than asserted.

The score is always a lexicographic vector of exact integers, never a float, and its first
key is always `unpacked_count`: no objective will ever leave an item behind to save money.
Ratios are parts per million. See docs/OBJECTIVE.md for the full key ordering.
"""

from packvium import Container, Dimensions, Item, Packer, PackingConfig
from packvium.models import RateTable, UnratedWeightError

WIDGETS = [Item.create("widget", Dimensions.mm("100", "100", "100"), "500 g", quantity=8)]


def solve(config: PackingConfig, containers) -> tuple[str, tuple[int, ...]]:
    result = Packer(config).pack(WIDGETS, containers)
    return (result.containers[0].container.id if result.containers else "none", result.score)


# ---------------------------------------------------------------------------------
# `default` -- fewest containers, then tightest fit. The objective you want when the
# containers are interchangeable and you are simply trying not to open another box.
# ---------------------------------------------------------------------------------
snug = Container.create("snug", Dimensions.mm("300", "300", "300"), max_payload="20 kg", cost_minor=500)
roomy = Container.create("roomy", Dimensions.mm("400", "400", "400"), max_payload="20 kg", cost_minor=150)

print("default          ", solve(PackingConfig.balanced(), [snug, roomy]))

# ---------------------------------------------------------------------------------
# `lowest_cost` -- the cheapest *packaging*. `cost_minor` is what the box itself costs
# you, so this is the objective for a warehouse buying cartons, not for a shipper paying
# a carrier. Here it prefers the roomy box precisely because the snug one costs more.
# ---------------------------------------------------------------------------------
print("lowest_cost      ", solve(PackingConfig(objective="lowest_cost"), [snug, roomy]))

# ---------------------------------------------------------------------------------
# `shipping_cost` -- carrier-billable *weight*. Billed weight is the greater of actual
# gross weight and dimensional weight, so a big light box can bill more than a small
# heavy one. That is the whole reason this objective is not the same as `lowest_cost`:
# the roomy box is cheaper to buy and dearer to ship.
#
# It needs a divisor. Without one the library refuses rather than guessing, because a
# wrong divisor silently misprices every shipment.
# ---------------------------------------------------------------------------------
by_weight = PackingConfig(
    objective="shipping_cost",
    dimensional_weight_divisor=5000,
    dimensional_weight_length_unit="cm",
    dimensional_weight_weight_unit="kg",
)
print("shipping_cost    ", solve(by_weight, [snug, roomy]))

# ---------------------------------------------------------------------------------
# `lowest_landed_cost` -- carrier-billable *money*. This is where a rate card enters the
# request as data. It exists because weight and money do not always agree: a bracket
# step, or a minimum charge, can make the cheaper shipment the heavier one.
#
# Below, the roomy box bills heavier (12,800 g dimensional against the snug box's 5,400)
# and yet costs less, because the snug box's carrier charges a steep first bracket. Rank
# by weight and you pick the wrong box; rank by money and you pick the right one.
# ---------------------------------------------------------------------------------
dear_per_gram = Container.create(
    "snug", Dimensions.mm("300", "300", "300"), max_payload="20 kg",
    rate_table=RateTable(weight_brackets_g=(6_000, 20_000), prices_minor=(2_400, 3_100)),
)
cheap_per_gram = Container.create(
    "roomy", Dimensions.mm("400", "400", "400"), max_payload="20 kg",
    rate_table=RateTable(weight_brackets_g=(6_000, 20_000), prices_minor=(900, 1_500)),
)
by_money = PackingConfig(
    objective="lowest_landed_cost",
    dimensional_weight_divisor=5000,
    dimensional_weight_length_unit="cm",
    dimensional_weight_weight_unit="kg",
)
print("landed_cost      ", solve(by_money, [dear_per_gram, cheap_per_gram]))

# A rate card that stops short of the shipment is a refusal, never a silent clamp to the
# top bracket. You would otherwise be quoted a price the carrier never published.
too_narrow = Container.create(
    "roomy", Dimensions.mm("400", "400", "400"), max_payload="20 kg",
    rate_table=RateTable(weight_brackets_g=(2_000,), prices_minor=(900,)),
)
try:
    solve(by_money, [too_narrow])
except UnratedWeightError as refusal:
    print("landed_cost*     ", "refused:", refusal)

# ---------------------------------------------------------------------------------
# `open_dimension_height` -- pack into the shortest stack. For a container with no lid,
# or a pallet whose height you are trying to keep under a doorway.
# ---------------------------------------------------------------------------------
print("open_dimension   ", solve(PackingConfig(objective="open_dimension_height"), [snug, roomy]))

# ---------------------------------------------------------------------------------
# `maximum_value` -- when not everything fits, leave the *cheap* things behind. Ranked by
# value forgone, after unpacked count. Note the honest limitation: this orders by value,
# it does not solve the knapsack problem to optimality. See docs/LIMITATIONS-AND-ROADMAP.md.
# ---------------------------------------------------------------------------------
# `quantity=1` is what makes this a choice at all: with an unlimited supply of boxes the
# packer simply opens a second one and nothing is left behind.
tiny = [Container.create("tiny", Dimensions.mm("200", "100", "100"), max_payload="20 kg", quantity=1)]
mixed = [
    Item.create("gold", Dimensions.mm("100", "100", "100"), "500 g", quantity=2, value=90_000),
    Item.create("gravel", Dimensions.mm("100", "100", "100"), "500 g", quantity=2, value=10),
]
result = Packer(PackingConfig(objective="maximum_value")).pack(mixed, tiny)
kept = sorted(p.instance.item.id for c in result.containers for p in c.placements)
left = sorted(u.instance.item.id for u in result.unpacked)
print("maximum_value    ", "packed:", kept, "left behind:", left)
