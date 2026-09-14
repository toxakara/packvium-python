"""Turn a packing result into instructions someone can follow on a dock.

Run it:

    PYTHONPATH=src python3 examples/execution.py

`pack()` answers where every box goes. That answer is not yet a work order: it does not
say what to lift first, it does not separate what the solver *decided* from what a screen
should *say*, and it has no way for the operator who is standing there to tell you the
printer must go in the corner.

The execution plan is that second document. It is derived from an already validated
result -- it calls no solver and no validator, and there is a test in each language that
asserts so. Anything it could decide on its own would be a decision made twice.
"""

from __future__ import annotations

import json

from packvium import pack_from_dict
from packvium.execution import build_execution_plan, canonical_plan_json
from packvium.locks import (
    LockSetError,
    PlacementLock,
    locks_from_plan,
    resolve_with_locks,
)

# --------------------------------------------------------------------------------------
# One crate, a printer that must stay upright, four toner cartridges, and a pallet jack
# that was never going to fit. The last one is deliberate: an execution plan has to say
# what is *not* going on the truck as clearly as what is.
# --------------------------------------------------------------------------------------
REQUEST = {
    "units": {"length": "mm"},
    "configuration": {
        "objective": "default",
        "profile": "balanced",
        "seed": 42,
        # A safety fuse, not a target -- nothing in this scene comes close to it.
        "time_limit_ms": 60_000,
    },
    "items": [
        {"id": "printer", "quantity": 1, "weight": "9 kg", "keep_upright": True,
         "dimensions": {"length": "420", "width": "340", "height": "260"}},
        {"id": "toner", "quantity": 4, "weight": "900 g",
         "dimensions": {"length": "180", "width": "120", "height": "100"}},
        {"id": "pallet-jack", "quantity": 1, "weight": "80 kg",
         "dimensions": {"length": "1200", "width": "550", "height": "1200"}},
    ],
    "containers": [
        {"id": "crate", "quantity": 1, "max_payload": "30 kg",
         "inner_dimensions": {"length": "600", "width": "400", "height": "400"}},
    ],
}

result = pack_from_dict(REQUEST)
plan = build_execution_plan(REQUEST, result)
container = plan["containers"][0]


print("=" * 78)
print("1. What the solver decided, kept apart from what a screen says")
print("=" * 78)
print()

print(f"  format:          {plan['format']}")
print(f"  status:          {plan['facts']['status']}")
print(f"  containers used: {plan['facts']['container_count']}")
print(f"  score:           {plan['facts']['score']}")
print(f"  utilization:     {container['facts']['volume_utilization']}")
print()
print("  Everything above is under `facts`. It is the solver's own answer, copied and")
print("  not re-derived, so a downstream system that reads only `facts` loses nothing it")
print("  is entitled to rely on. The score stays a vector: collapsing five axes into one")
print("  number is a judgement about your priorities that this document does not make.")
print()


print("=" * 78)
print("2. The step order is injected, or it is honestly absent")
print("=" * 78)
print()

print(f"  order: {container['order']}")
for step in container["steps"]:
    reference = step["placement"]
    ticks = reference["position_ticks"]
    print(f"    {reference['item_type']:<9} {reference['orientation']}  "
          f"at ({ticks['x']}, {ticks['y']}, {ticks['z']})")
print()
print("  Every placement is listed and not one is numbered. The engines compute a safe")
print("  loading order from geometry this adapter never sees, so without one it says")
print("  `unavailable` rather than guessing.")
print()
print("  There is no third behaviour on purpose. Falling back to the order placements")
print("  happen to appear in would present an artifact of how the solver walked its")
print("  candidate points as an order that is safe to lift boxes in. It is not.")
print()

ordered = build_execution_plan(
    REQUEST, result,
    loading_orders={0: list(range(len(container["steps"]))[::-1])},
)
print(f"  order: {ordered['containers'][0]['order']}")
for step in ordered["containers"][0]["steps"]:
    print(f"    {step['sequence']}. {step['placement']['item_type']}")
print()
print("  Hand it an order and each step is numbered. The order above is reversed on")
print("  purpose, to show that the sequence is the one you supplied and not one the")
print("  adapter re-derived behind your back.")
print()


print("=" * 78)
print("3. Every sentence names the fields it was built from")
print("=" * 78)
print()

for entry in plan["unplaced"]:
    facts = entry["facts"]
    print(f"  facts:        item_type={facts['item_type']!r}")
    print(f"                reason={facts['reason']!r} proof_level={facts['proof_level']!r}")
    print(f"  presentation: {entry['presentation']['summary']}")
    print(f"  cites:        {', '.join(entry['presentation']['cites'])}")
print()
print("  `proven` is a claim about a search, not a summary of one: no orientation of the")
print("  pallet jack fits any offered crate, so nothing was tried and nothing needed to")
print("  be. A reason with no citation would be a sentence nobody can check, which is")
print("  why `cites` is part of the format rather than a convention.")
print()


print("=" * 78)
print("4. One plan, four engines, the same bytes")
print("=" * 78)
print()

canonical = canonical_plan_json(plan)
print(f"  canonical form: {len(canonical)} bytes, first 68 of them")
print(f"    {canonical[:68]}...")
print()
print("  Hand the same *result* to all four adapters and they emit the same bytes. That")
print("  is stricter than the packing contract, and it can be: a plan is derived from a")
print("  result, so there is nothing left to differ about.")
print()
print("  It does not follow that four engines packing the same *request* agree. Python")
print("  and PHP are held to identical placements and do produce this exact form; Rust")
print("  and JavaScript are held to a valid answer at or above the objective floor, and")
print("  `examples/execution.mjs` prints 1280 bytes here rather than 1877 for that")
print("  reason. Whose packing is better is a question the plan never answers.")
print()


print("=" * 78)
print("5. The operator pins a box, and gets a second plan — never an edited one")
print("=" * 78)
print()

locks = locks_from_plan(plan, container_index=0, item_types=["printer"])
lock = locks[0]
print(f"  locked: {lock.item_type} {lock.orientation} at {lock.position_ticks}")
print("  The lock is read out of the plan's own placement reference. That is the only")
print("  reason the two can be trusted to mean the same box: there is no second address")
print("  format to drift.")
print()

resolved = resolve_with_locks(REQUEST, locks)
print(f"  preserved: {resolved.preserved}")
print(f"  the approved plan is untouched: "
      f"{canonical_plan_json(build_execution_plan(REQUEST, result)) == canonical}")
print()
print("  A lock becomes an ordinary placement constraint, so the re-solve is the same")
print("  portfolio under the same independent validator. Nothing here can produce a")
print("  placement the engine would otherwise refuse — the strongest thing a lock can do")
print("  is reserve its own slot and make the search work around it.")
print()


print("=" * 78)
print("6. Two ways a lock can fail, and they are not the same kind of thing")
print("=" * 78)
print()

# Far outside the crate: a single lock that is individually impossible. The solve still
# runs and still answers; it simply cannot honour this one.
unreachable = PlacementLock(container_index=0, item_type="printer", orientation="LWH",
                            position_ticks=(99_000_000, 0, 0))
outcome = resolve_with_locks(REQUEST, (unreachable,))
print(f"  a lock the solve cannot honour -> preserved={outcome.preserved}, "
      f"{len(outcome.missing)} lock reported back")
print(f"    {outcome.missing[0].item_type} at {outcome.missing[0].position_ticks}")
print("  Not an exception. The operator asked for something the geometry does not allow,")
print("  and what they need back is a valid plan plus the news that their pin was not")
print("  honoured — not a stack trace and no plan at all.")
print()

# An item the request never contained: the lock set is wrong about the world, and no
# solve could make it right.
try:
    resolve_with_locks(REQUEST, (lock, PlacementLock(container_index=0, item_type="scanner",
                                                     orientation="LWH",
                                                     position_ticks=(0, 0, 0))))
except LockSetError as refusal:
    print(f"  a lock set that contradicts itself -> {refusal}")
print()
print("  Refused before any solve runs, because no result could satisfy it. The same goes")
print("  for two locks whose boxes overlap. The distinction is worth keeping: the first")
print("  case is an answer the operator may not like, the second is a question that has")
print("  no answer, and reporting them the same way would hide which one happened.")
