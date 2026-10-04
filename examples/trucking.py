"""Load a delivery van for a three-stop route, within its axle ratings.

Run it:

    PYTHONPATH=src python3 examples/trucking.py

A van is not a big box. Its payload rides on two axles, and each axle has its own rating
that the total payload limit does not protect: a van can be under its payload and still
overload the front axle. Its cargo also comes out through one door, in the order the
stops come up, so the goods for the first stop must not be walled in by the goods for
the last one.

Three request fields carry that:

- `axles` on the container: `[front, rear]`, each a position along the length and an
  optional `max_load`. Checked for every placement against the gross load (payload plus
  tare), so no plan puts either axle over its rating.
- `access_directions` on the container: the walls cargo can leave through. `+x` is the
  wall at the far end of the length axis -- here, the rear doors.
- `stop_index` on each item: which stop it is delivered at, `0` first.

After the solve, the sequence functions answer the questions a driver asks: in what order
do things come off, in what order must they go on, and what can I reach when I open the
doors at the first stop?
"""

from __future__ import annotations

from fractions import Fraction

from packvium import (Axle, Container, Dimensions, EffortBudget, Item, Length, Packer, PackingConfig,
                      RouteSequenceError, Weight, placement_reachability, replay_loading_order,
                      safe_route_removal_order)

REAR_DOORS = ("+x",)

# Length runs from the cab bulkhead (x = 0) to the rear doors. The axle positions are
# measured along that same axis, which is what lets the engine take moments about them.
FRONT_AXLE = Axle(Length.mm("600"), Weight.parse("1650 kg"))
REAR_AXLE = Axle(Length.mm("3500"), Weight.parse("2200 kg"))


def van(front_axle: Axle = FRONT_AXLE) -> Container:
    return Container.create(
        "van", Dimensions.mm("4200", "1800", "1900"),
        # Tare matters here: axle ratings are gross limits, and the empty van already
        # puts most of the front rating to use before anything is loaded.
        tare_weight="2400 kg", max_payload="1300 kg",
        axles=(front_axle, REAR_AXLE), access_directions=REAR_DOORS,
    )


def manifest(with_route: bool = True) -> list[Item]:
    def stop(index: int) -> dict:
        return {"stop_index": index} if with_route else {}

    return [
        Item.create("bakery-rack", Dimensions.mm("800", "600", "1700"), "90 kg", quantity=2,
                    keep_upright=True, **stop(0)),
        Item.create("drinks-pallet", Dimensions.mm("1200", "800", "1100"), "420 kg",
                    keep_upright=True, must_be_on_floor=True, **stop(1)),
        Item.create("parcel", Dimensions.mm("600", "400", "400"), "18 kg", quantity=6, **stop(2)),
        Item.create("dry-goods-pallet", Dimensions.mm("1200", "800", "1400"), "260 kg",
                    keep_upright=True, **stop(2)),
    ]


# The effort budget decides where the search stops, so every run gives this same plan;
# the time limit is only a fuse far above what this needs (see reproducibility.py).
# `max_containers=1` because there is one van: anything that does not fit is reported,
# not sent in a second vehicle.
CONFIG = PackingConfig.balanced(time_limit_ms=60_000, effort_budget=EffortBudget(max_restarts=8),
                                max_containers=1)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def millimetres(ticks: int) -> str:
    return Length(ticks).decimal("mm", 1)


def axle_loads_kg(result) -> tuple[str, str]:
    """The exact axle reactions from the result document, shown in kilograms.

    They arrive as a numerator over a shared denominator so no engine has to round
    them; rounding happens here, for display only.
    """
    reactions = result.to_dict()["containers"][0]["axle_reactions"]
    denominator = int(reactions["denominator"])

    def kilograms(numerator: str) -> str:
        ticks = round(Fraction(int(numerator), denominator))
        return Weight(ticks).decimal("kg", 1)

    return kilograms(reactions["front_numerator"]), kilograms(reactions["rear_numerator"])


# --------------------------------------------------------------------------------------
section("1. The plan: nothing for a later stop stands in an earlier stop's way out")

result = Packer(CONFIG).pack(manifest(), [van()])
packed = result.containers[0]
placements = packed.placements
print(f"  status: {result.status.value}, left behind: {len(result.unpacked)}")
for placement in sorted(placements, key=lambda p: (p.position.x, p.position.y, p.position.z)):
    print(f"    stop {placement.instance.item.stop_index}  {placement.instance.id:<20}"
          f" x={millimetres(placement.position.x):>6} mm  y={millimetres(placement.position.y):>6} mm")
print("""
  x is the distance from the bulkhead, so the rear doors are at x = 4200 mm. The
  bakery racks sit at the front, but the aisle beside them runs clear to the doors:
  the rule is that nothing due later blocks every way out, not that stop 0 is last in.""")

# --------------------------------------------------------------------------------------
section("2. Axle loads against their ratings")

front, rear = axle_loads_kg(result)
print(f"  payload: {packed.payload_weight.decimal('kg', 1)} kg of 1300 kg allowed")
print(f"  front axle: {front} kg of {FRONT_AXLE.max_load.decimal('kg', 1)} kg")
print(f"  rear axle:  {rear} kg of {REAR_AXLE.max_load.decimal('kg', 1)} kg")

tight = Packer(CONFIG).pack(manifest(), [van(Axle(Length.mm("600"), Weight.parse("1450 kg")))])
print()
print("  the same load with a 1450 kg front axle rating:")
for unpacked in tight.unpacked:
    print(f"    left behind: {unpacked.instance.id}, reason {unpacked.reason} ({unpacked.proof.level})")
print("""
  The axle rule refuses a placement rather than reporting an overload afterwards.
  Note the reason: an item the axles cannot carry is reported as the generic
  `no_feasible_placement`, not as an axle problem, so when a load comes up short on
  a vehicle with axle ratings, check the reactions before shopping for a bigger van.""")

# --------------------------------------------------------------------------------------
section("3. Unloading order along the route, and the loading order it implies")

boxes = [placement.envelope_box for placement in placements]
stops = [placement.instance.item.stop_index for placement in placements]
inside = packed.container.inner_dimensions

unloading = safe_route_removal_order(boxes, stops, inside, REAR_DOORS)
for step, index in enumerate(unloading, start=1):
    print(f"    off {step:>2}: stop {stops[index]}  {placements[index].instance.id}")

# Loading is unloading played backwards: whatever comes off last goes on first.
# `replay_loading_order` checks that independently -- every item supported by what is
# already in, and each one able to travel in through the doors -- and raises if not.
loading = list(reversed(unloading))
replay_loading_order(boxes, inside, loading, REAR_DOORS)
print()
print("  load in this order:", ", ".join(placements[index].instance.id for index in loading))

# --------------------------------------------------------------------------------------
section("4. What the driver can reach on opening the doors at stop 0")

for reach in placement_reachability(boxes, inside, stops, REAR_DOORS):
    if stops[reach.index] != 0:
        continue
    blockers = sorted(placements[i].instance.id for i in reach.blocked_by_neighbors | reach.blocked_by_support)
    print(f"    {placements[reach.index].instance.id}: reachable={reach.reachable}"
          + (f", behind {', '.join(blockers)}" if blockers else ""))
print("""
  Reachability is a snapshot, not a plan: one rack is behind the other, which is
  fine because both are for this stop and the first one comes off first.""")

# --------------------------------------------------------------------------------------
section("5. The same goods without `stop_index`")

unrouted = Packer(CONFIG).pack(manifest(with_route=False), [van()])
unrouted_placements = unrouted.containers[0].placements
due = {"bakery-rack": 0, "drinks-pallet": 1, "parcel": 2, "dry-goods-pallet": 2}
unrouted_stops = [due[p.instance.item.id] for p in unrouted_placements]
try:
    safe_route_removal_order([p.envelope_box for p in unrouted_placements], unrouted_stops,
                             inside, REAR_DOORS)
except RouteSequenceError as stuck:
    names = sorted(unrouted_placements[i].instance.id for i in stuck.stuck)
    print(f"  status {unrouted.status.value}, every item packed -- and at stop {stuck.stop}"
          f" the driver cannot get these out: {', '.join(names)}")
print("""
  Without the route the plan is still a valid packing, and still within the axle
  ratings; it just cannot be delivered in order. The route is only enforced when
  you state it.""")
