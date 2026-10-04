"""Fleet and vehicle limits: what may go where, and how much of it.

Run it:

    PYTHONPATH=src python3 examples/limits.py

A same-day courier has one refrigerated van and a few cargo bikes. Every rule below is a
request field, so it holds in every Packvium engine and travels with the request:

- `obstacles` on a container: space inside it that cargo may not use -- here the fridge
  unit under the roof, and the two wheel arches as one obstacle made of two boxes
  (`additional_boxes`), since an arch on each side is one thing to the person loading.
- `max_items` on a container: a cap on how many items it takes, whatever their size --
  a bike's rack holds four parcels however small they are.
- `tags` and `tag_limits`: at most one item tagged `dangerous_goods` in the van.
- `eligible_container_tags` on an item: frozen goods only go in a container tagged
  `refrigerated`.
- `max_containers` in the configuration: how many vehicles the whole plan may use.
"""

from __future__ import annotations

import copy
from collections import Counter

from packvium import pack_from_dict

REQUEST = {
    "units": {"length": "mm"},
    # Counted work decides the answer, not the clock (see reproducibility.py); the time
    # limit is a fuse far above what this solve needs.
    "configuration": {"time_limit_ms": 60_000, "effort_budget": {"max_restarts": 8}},
    "items": [
        {"id": "frozen-tote", "quantity": 3, "weight": "15 kg",
         "dimensions": {"length": "600", "width": "400", "height": "300"},
         "eligible_container_tags": ["refrigerated"]},
        {"id": "lithium-battery", "quantity": 3, "weight": "12 kg",
         "dimensions": {"length": "400", "width": "300", "height": "250"},
         "tags": ["dangerous_goods"]},
        {"id": "parcel", "quantity": 24, "weight": "8 kg",
         "dimensions": {"length": "500", "width": "400", "height": "400"}},
    ],
    "containers": [
        {"id": "cargo-bike", "quantity": 3, "cost_minor": 900, "max_items": 4, "max_payload": "100 kg",
         "inner_dimensions": {"length": "1000", "width": "800", "height": "900"}},
        {"id": "reefer-van", "quantity": 1, "cost_minor": 4000,
         "inner_dimensions": {"length": "1800", "width": "1400", "height": "1200"},
         "tags": ["refrigerated"],
         "tag_limits": {"dangerous_goods": 1},
         "obstacles": [
             {"id": "fridge-unit", "origin": {"x": "0", "y": "0", "z": "800"},
              "dimensions": {"length": "400", "width": "1400", "height": "400"}},
             {"id": "wheel-arches", "origin": {"x": "700", "y": "0", "z": "0"},
              "dimensions": {"length": "800", "width": "200", "height": "350"},
              "additional_boxes": [{"origin": {"x": "700", "y": "1200", "z": "0"},
                                    "dimensions": {"length": "800", "width": "200", "height": "350"}}]},
         ]},
    ],
}


def solve(change=None) -> dict:
    request = copy.deepcopy(REQUEST)
    if change:
        change(request)
    return pack_from_dict(request)


def show(result: dict) -> None:
    print(f"  status: {result['status']}")
    for container in result["containers"]:
        contents = Counter(placement["item_type"] for placement in container["placements"])
        listed = ", ".join(f"{count} {name}" for name, count in sorted(contents.items()))
        print(f"    {container['id']:<14} {listed}")
    # One line per kind of refusal: a reader needs "five parcels, no room", not five lines.
    refusals = Counter((unpacked["item_type"], unpacked["reason"], unpacked["proof"]["level"])
                       for unpacked in result.get("unpacked_items") or ())
    for (name, reason, level), number in sorted(refusals.items()):
        print(f"    left out: {number} {name}, {reason} ({level})")


def count(result: dict, container_id: str, item_type: str) -> int:
    return sum(placement["item_type"] == item_type
               for container in result["containers"] if container["id"] == container_id
               for placement in container["placements"])


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
section("1. The plan")

plan = solve()
show(plan)
print("""
  All three frozen totes are in the van, because nothing else is refrigerated. Only
  one battery rides in the van; the other two go by bike. The bike takes four items,
  its `max_items`, although there is room and payload for more.""")

# --------------------------------------------------------------------------------------
section("2. What the obstacles cost")

open_van = solve(lambda r: r["containers"][1].pop("obstacles"))
print(f"  parcels in the van with the fridge unit and arches: {count(plan, 'reefer-van#1', 'parcel')}")
print(f"  parcels in the van if it were an empty box:        {count(open_van, 'reefer-van#1', 'parcel')}")
print("""
  An obstacle is exact space, not a hint. Model what is really there, or the plan
  will put a parcel where the wheel arch is.""")

# --------------------------------------------------------------------------------------
section("3. `max_containers`: the plan may use one vehicle")

show(solve(lambda r: r["configuration"].update(max_containers=1)))
print("""
  The cap is honoured, and what does not fit is listed rather than silently sent in
  a vehicle you did not allow. The reason is `observed`, not `proven`: the search
  found no room for these, which is not a claim that no arrangement has room.""")

# --------------------------------------------------------------------------------------
section("4. `eligible_container_tags`: the van is off the road today")

show(solve(lambda r: r["containers"].pop(1)))
print("""
  The frozen totes cannot go anywhere, and their reason is `proven`: no container in
  the request carries the tag they need, which is a fact about the request rather
  than about how far the search got. The rest is `observed`: three bikes of four
  items each were all the request offered, and the search ran out of them.""")
