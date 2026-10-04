"""Even out the weight across pallets after packing.

Run it:

    PYTHONPATH=src python3 examples/rebalancing.py

The packer opens as few containers as it can and fills the first one tightly. For two
pallets on a forklift, that is the wrong shape of answer: one pallet near its weight
limit and one nearly empty, when two pallets of similar weight are safer to lift, wrap
and stack.

`rebalance_weight` is a separate, opt-in step you run on a finished packing. It moves one
item at a time from the heaviest container to a lighter one, and keeps a move only when
the item has a real place in its new container and the whole packing still passes the
independent validator. It never adds a container, never drops an item, and stops when no
single move narrows the gap (or after `max_moves`, 64 by default). It is greedy, so the
result is better, not necessarily the most even split possible.
"""

from __future__ import annotations

from packvium import (Container, Dimensions, EffortBudget, IndependentSolutionValidator, Item, Packer,
                      PackingConfig, PackingRequest, rebalance_weight)

ITEMS = (
    Item.create("paint-20l", Dimensions.mm("300", "300", "380"), "24 kg", quantity=16, keep_upright=True),
    Item.create("brush-case", Dimensions.mm("400", "300", "200"), "3 kg", quantity=10),
)
PALLET = Container.create("pallet", Dimensions.mm("1200", "800", "1000"), tare_weight="25 kg",
                          max_payload="300 kg")

# Counted work, not a clock, decides the plan (see reproducibility.py); the time limits
# here and in `rebalance_weight` below are fuses far above what either step needs.
CONFIG = PackingConfig.balanced(time_limit_ms=60_000, effort_budget=EffortBudget(max_restarts=8))


def weights(containers) -> str:
    return ", ".join(f"{packed.container.id}#{packed.sequence} {packed.payload_weight.decimal('kg', 1)} kg"
                     f" ({len(packed.placements)} items)" for packed in containers)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
section("1. As packed: as few pallets as possible, the first one full")

result = Packer(CONFIG).pack(ITEMS, [PALLET])
print(f"  {weights(result.containers)}")

# --------------------------------------------------------------------------------------
section("2. Rebalanced")

# The request is what the validator checks the new packing against: every item that was
# asked for must still be accounted for, exactly once.
request = PackingRequest(ITEMS, (PALLET,))
rebalanced = rebalance_weight(request, result.containers, result.unpacked, CONFIG, time_limit_ms=60_000)
for move in rebalanced.moves:
    print(f"  moved {move.item_id} from {move.from_container_id} to {move.to_container_id}")
print(f"  {weights(rebalanced.containers)}")

report = IndependentSolutionValidator().validate(
    request, rebalanced.containers, CONFIG.minimum_support_ratio, CONFIG.clearance, result.unpacked)
print(f"  still a valid packing: {report.valid}")
print("""
  Heavy items move first, because each one narrows the gap most. The paint stays
  upright in its new place and nothing is left resting on a gap where a moved can
  used to be -- a move that would do that is rejected, not made.""")

# --------------------------------------------------------------------------------------
section("3. When there is nothing to improve")

single = Packer(CONFIG).pack(ITEMS[1:], [PALLET])
unchanged = rebalance_weight(PackingRequest(ITEMS[1:], (PALLET,)), single.containers, single.unpacked,
                             CONFIG, time_limit_ms=60_000)
print(f"  {weights(unchanged.containers)}  improved: {unchanged.improved}")
print("""
  One pallet has nothing to balance against, and the result says so rather than
  inventing a second pallet. The same holds when every candidate move would break a
  rule: `moves` is empty and the packing comes back as it was.""")
