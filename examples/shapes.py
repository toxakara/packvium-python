"""Shapes: when an item is not its box.

Run it:

    PYTHONPATH=src python3 examples/shapes.py

Every other example treats an item as the box it declares. That is the default and it is
right for almost everything, because a carton *is* a cuboid. Two kinds of goods are not:
a moulded or tapered part that leaves a usable void beside it, and a soft one that gives
way under whatever is stacked on it.

`shape_type` narrows the box in one direction each -- `convex_hull` in space,
`compressible` in height under load -- and neither is ever inferred. An engine that quietly
packed a hull as its bounding box would return a plan that validates and does not
physically fit, so the value must be asked for.

The last section is a container rather than an item: a bag whose sides fold out
(`gusseted_bag`) holds a thicker load than a flat one of the same width.

All of it is written here through `pack_from_dict`, the request contract the four engines
share. That is deliberate: the shape fields are part of the JSON contract, so the same
request runs unchanged against the Python, PHP, Rust and JavaScript engines.
"""

# `list | None` in a signature is PEP 604, which needs Python 3.10 at runtime. This
# package supports 3.9, so the annotation is deferred rather than evaluated.
from __future__ import annotations

from packvium import Rotation, pack_from_dict


def summarise(label: str, request: dict) -> None:
    """Run one request and print only what the shape changed: containers and refusals.

    `pack_from_dict` answers in the same JSON shape the other three engines return, so
    everything read here is the cross-language contract rather than a Python attribute.
    """
    result = pack_from_dict({**request, **SAFETY_FUSE})
    containers = result["containers"]
    placed = sum(len(container["placements"]) for container in containers)
    print(
        f"  {label:22s} {result['status']:10s} "
        f"{len(containers)} container(s), {placed} placed, "
        f"{len(result['unpacked_items'])} refused"
    )


def crate(length: str, width: str, height: str) -> list:
    return [{"id": "crate",
             "inner_dimensions": {"length": length, "width": width, "height": height}}]


MM = {"units": {"length": "mm"}}
# An example must not change answer merely because the host was busy. These solves need a
# fraction of the budget; the generous wall-clock value is only a safety fuse, so a loaded
# machine cannot cut the multi-start portfolio short and let a different start win.
SAFETY_FUSE = {"configuration": {"time_limit_ms": 60_000}}


# ------------------------------------------------------------------ convex_hull
#
# Two triangular prisms, each cut from the same 100 mm cube along the diagonal. Their
# bounding boxes are identical and fill the crate on their own, so as cuboids the second
# one has nowhere to go. As hulls they are complementary halves and share the crate
# exactly -- the collision test is an exact integer separating-axis test on the vertices,
# not a box overlap.
#
# The hull is given in the item's own coordinates, in the request's length unit, and must
# fit inside the declared dimensions. It is not a replacement for them: the box still
# bounds the item, the hull only says how much of that box is solid.

LOWER_WEDGE = [{"x": "0", "y": "0", "z": "0"}, {"x": "100", "y": "0", "z": "0"},
               {"x": "0", "y": "100", "z": "0"}, {"x": "0", "y": "0", "z": "100"},
               {"x": "100", "y": "0", "z": "100"}, {"x": "0", "y": "100", "z": "100"}]
UPPER_WEDGE = [{"x": "100", "y": "100", "z": "0"}, {"x": "100", "y": "0", "z": "0"},
               {"x": "0", "y": "100", "z": "0"}, {"x": "100", "y": "100", "z": "100"},
               {"x": "100", "y": "0", "z": "100"}, {"x": "0", "y": "100", "z": "100"}]


def wedge(item_id: str, vertices: list | None) -> dict:
    item = {"id": item_id, "quantity": 1,
            "dimensions": {"length": "100", "width": "100", "height": "100"},
            "weight": {"value": "1", "unit": "kg"}}
    if vertices is not None:
        item["shape_type"] = "convex_hull"
        item["hull_vertices"] = vertices
    return item


print("convex_hull -- two complementary wedges cut from one cube")
summarise("as cuboids", {**MM,
                         "items": [wedge("wedge-lower", None), wedge("wedge-upper", None)],
                         "containers": crate("100", "100", "100")})
summarise("as hulls", {**MM,
                       "items": [wedge("wedge-lower", LOWER_WEDGE),
                                 wedge("wedge-upper", UPPER_WEDGE)],
                       "containers": crate("100", "100", "100")})

# One crate instead of two, for the same goods and the same crate. Nothing about the
# request changed except the claim that the items are wedges rather than blocks.


# ----------------------------------------------------------------- compressible
#
# `compression_ratio` is the fraction of its own height an item may lose when something
# rests on it -- 0.25 means it can give up a quarter. The mass above it is what decides
# how much it actually gives, so the occupied height of a compressible item is not a
# property of the item alone; it depends on what the solver put on top.
#
# `max_compression_pressure_kpa` is the other half of the same field. Past that pressure
# the item is not compressed further, it is crushed, and the load is refused instead.
#
# Note `must_be_on_floor` on the cushion. Without it the solver is free to put the brick
# underneath, nothing bears on the cushion, and the feature never engages -- which is the
# honest reason the rule is here and not an incidental detail of the example.

def cushion(crush_kpa: int) -> dict:
    return {"id": "cushion", "quantity": 1,
            "dimensions": {"length": "100", "width": "100", "height": "100"},
            "weight": {"value": "2", "unit": "kg"},
            "must_be_on_floor": True,
            "shape_type": "compressible",
            "compression_ratio": 0.25,
            "max_compression_pressure_kpa": crush_kpa}


def brick(kilograms: int) -> dict:
    return {"id": "brick", "quantity": 1,
            "dimensions": {"length": "100", "width": "100", "height": "100"},
            "weight": {"value": str(kilograms), "unit": "kg"}}


def load(label: str, kilograms: int) -> None:
    """One crate, one cushion, one brick -- only the brick's mass changes."""
    result = pack_from_dict({**MM, **SAFETY_FUSE, "items": [cushion(100), brick(kilograms)],
                             "containers": crate("100", "100", "200")})
    unused = result["score"][3]
    print(f"  {label:22s} {len(result['containers'])} container(s), "
          f"unused volume {unused} ppm")


# The crate is 100x100x200 and the two items are 100 mm cubes, so rigidly they fill it
# exactly and nothing is unused. Under 101 kg the cushion gives up part of its quarter,
# the pair still ships as one stack, and the volume it stopped occupying shows up as
# unused. One more kilogram crosses 100 kPa over the cushion's 0.01 m^2 face: the stack
# is refused, the brick opens a second crate, and half of each crate is empty.
print("\ncompressible -- a cushion that yields to the load above it")
load("brick 101 kg", 101)
load("brick 102 kg", 102)

# Both shapes are refused rather than approximated wherever an engine cannot honour them
# exactly -- a hull on a route, a hull under a configured clearance, a compressible item
# with `nesting_height`. A wrong answer that validates is worse than a refusal that does
# not, which is the whole reason these are opt-in.


# ------------------------------------------------------- 24 rotations for a hull
#
# The two wedges above are mirror images of each other, so the six rotations a box knows
# -- each one a permutation of the axes -- can turn the lower wedge into the upper one.
# Two *copies of the same wedge* are a different problem: the second copy must be turned
# half a revolution about the vertical axis to meet the first along the diagonal, and that
# is a signed rotation, `-L-W+H`, which no permutation of the axes reaches.
#
# A `convex_hull` item may list all 24 proper rotations of a cube. The six legacy codes keep
# their names and stay the default list, so a request that names no signed code is answered
# exactly as before; a signed code on anything but a hull is refused.

def two_copies(rotations: list[str] | None) -> dict:
    item = wedge("wedge", LOWER_WEDGE)
    item["quantity"] = 2
    if rotations is not None:
        item["allowed_rotations"] = rotations
    return {**MM, "items": [item], "containers": crate("100", "100", "100")}


def orientations(request: dict) -> str:
    result = pack_from_dict({**request, **SAFETY_FUSE})
    return ", ".join(placement["orientation"]
                     for container in result["containers"]
                     for placement in container["placements"])


EVERY_ROTATION = [rotation.value for rotation in Rotation.every()]

print("\nconvex_hull -- two copies of one wedge")
summarise("six rotations", two_copies(None))
summarise("all 24 rotations", two_copies(EVERY_ROTATION))
print(f"  {'orientations used':22s} {orientations(two_copies(EVERY_ROTATION))}")


# ---------------------------------------------------------------- gusseted bags
#
# A flat bag (`deformable_boundary.kind: flat_bag`) is a pouch with no sides: whatever
# goes in pushes its two faces apart, and the bag's width pays for that thickness. Its
# rule is `b + c <= W` -- the load's width plus its thickness within the bag's flat width.
# A gusseted bag folds extra material into its sides, `side_gusset_depth` of it, so the same
# rule reads `b + c <= W + G`. Everything else about the bag is the flat bag's.
#
# The books are 100 mm wide and 20 mm thick, and the pouch is 150 mm wide. Flat, it holds
# 50 mm of thickness, so two books; a 25 mm gusset lets it hold 75 mm, so three.

def pouch(boundary: dict) -> dict:
    return {**MM,
            "items": [{"id": "book", "quantity": 4, "weight": "300 g", "allowed_rotations": ["LWH"],
                       "dimensions": {"length": "200", "width": "100", "height": "20"}}],
            "containers": [{"id": "pouch", "quantity": 1, "deformable_boundary": boundary,
                            "inner_dimensions": {"length": "300", "width": "150", "height": "100"}}]}


print("\ndeformable_boundary -- one pouch, flat and gusseted")
summarise("flat_bag", pouch({"kind": "flat_bag"}))
summarise("gusseted_bag, 25 mm", pouch({"kind": "gusseted_bag", "side_gusset_depth": "25"}))
