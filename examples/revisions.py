"""Replan a half-loaded job without losing what was already done, or the record of why.

Run it:

    PYTHONPATH=src python3 examples/revisions.py

A plan is approved, and the dock starts loading. Then something changes: a slab is missing
from the shelf, and a cube that is already in the tote must stay exactly where it is. The next
plan has to keep the cube in place and pack around it, and anyone auditing the job later has
to see what changed, in what order, and which approved plan each change was recorded against.

A plan revision is that record. It is append-only and hash-chained: each revision names its
parent and the artifact it replaced by SHA-256, and carries the request the events produce.
The replan is an ordinary solve of that request, and every Packvium engine computes the same
revision bytes from the same inputs.
"""

from __future__ import annotations

import copy

from packvium import pack_from_dict
from packvium.artifacts import build_operational_artifact
from packvium.revisions import (PlanRevisionError, RevisionIssue, derive_revision,
                                root_revision, verify_revision_chain)

REQUEST = {
    "units": {"length": "mm"},
    "configuration": {
        # Counted work decides where the search stops, not a clock: a plan the clock stopped
        # could not be replayed, and its revision would say `replay: not_guaranteed`. The time
        # limit is only a fuse, far above what this needs even on a slow or emulated host.
        "effort_budget": {"max_search_nodes": 20000},
        "time_limit_ms": 60000,
    },
    "items": [
        {"id": "cube", "quantity": 4, "weight": "1 kg", "dimensions": {"length": "100", "width": "100", "height": "100"}},
        {"id": "slab", "quantity": 2, "weight": "2 kg", "dimensions": {"length": "200", "width": "100", "height": "50"}},
    ],
    "containers": [{"id": "tote", "quantity": 2, "inner_dimensions": {"length": "200", "width": "200", "height": "150"}}],
}


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def approve(request: dict) -> dict:
    """Solve a request and wrap the answer as the artifact the dock works from."""
    return build_operational_artifact(request, pack_from_dict(copy.deepcopy(request)))


def codes(issues: list[RevisionIssue]) -> str:
    return ", ".join(issue.code for issue in issues) or "no issues"


# --------------------------------------------------------------------------------------
section("1. The approved plan, and the revision that records it")

root = root_revision(REQUEST)
approved = approve(root["request"])
first_step = approved["plan"]["containers"][0]["steps"][0]["placement"]
print(f"  revision {root['revision']}, parent {root['parent']}")
print(f"  first step: a {first_step['item_type']} at x={first_step['position_ticks']['x']} in container 0")
print("""
  The root records nothing against the request. It exists so the first change has a
  parent to name.""")

# --------------------------------------------------------------------------------------
section("2. What happened on the dock, recorded against that plan")

loaded = pack_from_dict(copy.deepcopy(root["request"]))["containers"][0]["placements"][0]
events = [
    {"sequence": 1, "type": "placement_locked", "placement": {
        "item_type": "cube", "container_type": "tote", "container_instance": 1,
        "position": {axis: loaded["position"][axis]["value"] for axis in ("x", "y", "z")},
        "orientation": loaded["orientation"]}},
    {"sequence": 2, "type": "item_missing", "item_type": "slab", "quantity": 1},
]
revision = derive_revision(root, approved, events)
print(f"  revision {revision['revision']}, parent {revision['parent'][:23]}...")
print(f"  approved artifact {revision['approved']['artifact'][:23]}..., replay {revision['approved']['replay']['level']}")
print(f"  slabs still to pack: {revision['request']['items'][1]['quantity']}")
print(f"  fixed placements:    {len(revision['request']['fixed_placements'])}")
print("""
  The events are applied to the request, not to the result: one slab fewer, and the
  cube becomes a fixed placement. The revision names its parent and the artifact it
  replaces by SHA-256 over their RFC 8785 bytes, so every engine computes the same
  digest.""")

# --------------------------------------------------------------------------------------
section("3. The replan keeps the cube where it is")

replanned = pack_from_dict(copy.deepcopy(revision["request"]))
fixed = [f"{placement['item_id']} in {container['id']}"
         for container in replanned["containers"]
         for placement in container["placements"] if placement.get("fixed")]
placed = sum(len(container["placements"]) for container in replanned["containers"])
print(f"  fixed in the answer: {', '.join(fixed)}")
print(f"  items placed:        {placed}")
print("""
  An ordinary solve of the derived request, with nothing remembered from the first
  one. The fixed cube is marked `fixed: true`, and the validator refuses any answer
  that moves it.""")

# --------------------------------------------------------------------------------------
section("4. An audit that notices tampering, and a refusal instead of a guess")

chain = [root, revision]
print(f"  intact chain: {codes(verify_revision_chain(chain, [None, approved]))}")
edited = copy.deepcopy(chain)
edited[1]["request"]["items"][1]["quantity"] = 2
print(f"  edited chain: {codes(verify_revision_chain(edited))}")
try:
    derive_revision(revision, approve(revision["request"]),
                    [{"sequence": 3, "type": "item_missing", "item_type": "pallet", "quantity": 1}])
except PlanRevisionError as error:
    print(f"  a pallet that was never requested: refused with {error.code}")
print("""
  An edited request no longer equals what its parent's request and its own events
  produce, and any later revision would stop naming it as a parent. An event that
  contradicts the request is refused by name rather than applied as a best guess.""")
