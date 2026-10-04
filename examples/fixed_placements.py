"""Pack a trailer that is already partly loaded.

Run it:

    PYTHONPATH=src python3 examples/fixed_placements.py

A trailer arrives at the depot with four pallets of returns already on board. They are
staying on for the next leg, and nobody is going to unload them to make the plan tidier.
The depot has fourteen store pallets to add.

`fixed_placements` describes what is already there: which item, in which container, at
which position and orientation -- exactly the way a result reports a placement, so a
placement from an earlier plan can be fixed by quoting it. Fixed items are real cargo:
they carry weight, count against payload and can support what is stacked on them. The
engine packs around them, never moves them, and marks them `fixed: true` in the result.

A fixed item is also an instance of its item type: `quantity` counts the whole order,
and the fixed entries take its first instances.
"""

from __future__ import annotations

import copy

from packvium import FixedPlacementError, Weight, pack_from_dict


def on_board(x: str, y: str, orientation: str = "LWH") -> dict:
    return {"item_type": "returns-pallet", "container_type": "trailer", "orientation": orientation,
            "position": {"x": x, "y": y, "z": "0"}}


REQUEST = {
    "units": {"length": "mm"},
    # Counted work decides the answer, not the clock (see reproducibility.py); the time
    # limit is a fuse far above what this solve needs.
    "configuration": {"time_limit_ms": 60_000, "effort_budget": {"max_restarts": 8}},
    "items": [
        {"id": "returns-pallet", "quantity": 4, "weight": "350 kg", "keep_upright": True, "stackable": False,
         "dimensions": {"length": "1200", "width": "800", "height": "1500"}},
        {"id": "store-pallet", "quantity": 14, "weight": "500 kg", "keep_upright": True,
         "dimensions": {"length": "1200", "width": "800", "height": "1100"}},
    ],
    "containers": [{"id": "trailer", "max_payload": "9000 kg",
                    "inner_dimensions": {"length": "7200", "width": "2400", "height": "2400"}}],
    # Three across the nose, and one turned behind them.
    "fixed_placements": [on_board("0", "0"), on_board("0", "800"), on_board("0", "1600"),
                         on_board("1200", "0", orientation="WLH")],
}


def solve(change=None) -> dict:
    request = copy.deepcopy(REQUEST)
    if change:
        change(request)
    return pack_from_dict(request)


def show(result: dict) -> None:
    trailer = result["containers"][0]
    fixed = [p for p in trailer["placements"] if p.get("fixed")]
    added = [p for p in trailer["placements"] if not p.get("fixed")]
    payload = Weight(trailer["payload_weight"]["ticks"]).decimal("kg", 1)
    print(f"  status {result['status']}, payload {payload} kg,"
          f" left out {len(result.get('unpacked_items') or ())}")
    print(f"  already on board, unmoved: {', '.join(p['item_id'] for p in fixed)}")
    print(f"  added: {len(added)} store pallets")
    for placement in added:
        if placement["position"]["z"]["value"] != "0":
            print(f"    {placement['item_id']} is stacked, at z = {placement['position']['z']['value']} mm")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
section("1. Packed around the pallets already on board")

show(solve())
print("""
  The returns pallets keep their ids, positions and orientations. They took the
  first four instances of `returns-pallet`, so none are left for the engine to
  place; ask for five and it would place one more.""")

# --------------------------------------------------------------------------------------
section("2. One pallet was left in the middle of the floor")

show(solve(lambda r: r["fixed_placements"][3].update(position={"x": "1200", "y": "400", "z": "0"})))
print("""
  The fourth returns pallet stands 400 mm off the side wall, and no pallet fits in
  the strip it leaves. The plan still takes all fourteen store pallets, but it has
  to put one on top of another to do it. Whether that is acceptable is
  your call; the plan shows you the price of not moving that pallet.""")

# --------------------------------------------------------------------------------------
section("3. What is on board cannot be where the record says")

for label, change in (
    ("two pallets recorded in the same place",
     lambda r: r["fixed_placements"][3].update(position={"x": "600", "y": "0", "z": "0"})),
    ("a 1000 kg trailer limit, already exceeded by the returns",
     lambda r: r["containers"][0].update(max_payload="1000 kg")),
):
    try:
        solve(change)
    except FixedPlacementError as refusal:
        print(f"  {label}:")
        print(f"    {refusal.reason}, {refusal.field}: {refusal}")
print("""
  Nothing is solved when the starting point is impossible: a plan built on a
  collision or an overweight trailer would be wrong before the first pallet moved.
  `cannot_hold` names the fixed set as a whole; the detail says what is wrong.""")
