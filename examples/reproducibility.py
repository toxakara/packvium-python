"""Get the same answer on every run: counted work instead of a clock.

Run it:

    PYTHONPATH=src python3 examples/reproducibility.py

The search tries several starts and keeps the best. Something has to decide when it
stops, and there are two choices:

- `time_limit_ms`, a wall clock. Its default is one second. When it runs out, the search
  keeps what it has, and how far it got depends on the machine: its speed, what else is
  running, the garbage collector. The same request can give a different answer on a
  loaded server than on your laptop.
- `effort_budget`, counted work: candidates evaluated, placements attempted, search
  nodes, restarts. The count is the same on every machine, so the answer is too.

Set an effort budget whenever the answer is stored, compared, audited or replayed, and
keep the time limit as a generous fuse against a genuine hang. The result says which one
stopped the search, in `termination`.

To make a slow machine repeatable here, this example uses the injected clock the
`Packer` accepts for testing: a clock that advances a fixed amount each time it is read
stands in for a faster or slower host. A real deployment never passes one.
"""

from __future__ import annotations

import itertools
from typing import Callable

from packvium import Container, Dimensions, EffortBudget, Item, Packer, PackingConfig
from packvium.artifacts import replay_level

ITEMS = [
    Item.create("kettle", Dimensions.mm("310", "220", "140"), "2 kg", quantity=5),
    Item.create("toaster", Dimensions.mm("250", "250", "250"), "3 kg", quantity=4),
    Item.create("knife-block", Dimensions.mm("400", "150", "100"), "1 kg", quantity=6),
    Item.create("scale", Dimensions.mm("180", "120", "90"), "0.5 kg", quantity=8),
]
BOXES = [
    Container.create("small", Dimensions.mm("450", "350", "300"), cost_minor=100),
    Container.create("medium", Dimensions.mm("600", "400", "400"), cost_minor=160),
    Container.create("large", Dimensions.mm("800", "600", "500"), cost_minor=240),
]

#: A fuse, not a limit: nothing below comes close to it on a real machine.
FUSE_MS = 60_000


def host(nanoseconds_per_read: int) -> Callable[[], int]:
    """A monotonic clock that moves `nanoseconds_per_read` every time it is read: a larger
    step is a slower or busier machine doing the same work."""
    ticks = itertools.count(0, nanoseconds_per_read)
    return lambda: next(ticks)


def describe(result) -> str:
    document = result.to_dict()
    boxes = ",".join(packed.container.id for packed in result.containers)
    return (f"{boxes:<6} left out {len(result.unpacked):>2}  termination {result.termination.code:<12}"
            f" replay {replay_level(document)}")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
section("1. A clock-limited search answers differently on different machines")

# The same 100 ms limit on four simulated hosts.
for label, step in (("fast host", 10_000), ("busy host", 50_000),
                    ("slower host", 200_000), ("overloaded", 1_000_000)):
    config = PackingConfig.balanced(time_limit_ms=100)
    print(f"  {label:<12} {describe(Packer(config, clock=host(step)).pack(ITEMS, BOXES))}")
print("""
  Same request, four answers. Each is a valid packing of what the search reached
  before the clock ran out, and the items it never reached are listed as left out,
  but none can be reproduced: run it again on a different day and you get whichever
  one the machine allows. `termination` says
  `time_limit`, and `replay_level` refuses to promise a replay.""")

# --------------------------------------------------------------------------------------
section("2. An effort budget answers the same everywhere")

enough = EffortBudget(max_candidates_evaluated=5_000)
for label, step in (("fast host", 10_000), ("slower host", 200_000), ("overloaded", 1_000_000)):
    config = PackingConfig.balanced(time_limit_ms=FUSE_MS, effort_budget=enough)
    print(f"  {label:<12} {describe(Packer(config, clock=host(step)).pack(ITEMS, BOXES))}")
print("""
  The budget is large enough that the search finishes, so `termination` is
  `complete`, and it would stop at the same count on any machine if it were not.""")

# --------------------------------------------------------------------------------------
section("3. A budget that is too small is still reproducible, and says so")

too_small = EffortBudget(max_candidates_evaluated=500)
for label, step in (("fast host", 10_000), ("overloaded", 1_000_000)):
    config = PackingConfig.balanced(time_limit_ms=FUSE_MS, effort_budget=too_small)
    print(f"  {label:<12} {describe(Packer(config, clock=host(step)).pack(ITEMS, BOXES))}")
print("""
  The search stopped early and left items out -- `termination` is `effort_limit`,
  not `complete` -- but it stopped at the same point on both hosts, so the answer
  can be replayed exactly. Raise the budget until the answer is good enough, then
  keep it fixed.""")

# --------------------------------------------------------------------------------------
section("4. Reading `termination`")

config = PackingConfig.balanced(time_limit_ms=FUSE_MS, effort_budget=EffortBudget(max_restarts=4))
termination = Packer(config).pack(ITEMS, BOXES).to_dict()["termination"]
print(f"  code: {termination['code']}")
for start in termination["starts"]:
    marker = "  <- selected" if start["selected"] else ""
    print(f"    {start['id']:<28} completed={start['completed']!s:<5}{marker}".rstrip())
print("""
  `max_restarts` caps how many portfolio starts run -- here four -- which bounds the
  work of a large request as well as fixing it. `code` describes the selected start:
  a losing start that was cut short leaves it `complete`, and shows up in its own
  record and in `all_required_starts_completed` instead. Over JSON, the same budget
  is `configuration.effort_budget`.""")
