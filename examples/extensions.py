"""Extension points: rules the schema does not have a field for.

Run it:

    PYTHONPATH=src python3 examples/extensions.py

Most packing rules are already fields on `Item` and `Container` -- see constraints.py.
This example is about the rules that are not, and it is deliberately paired with an
honest warning about what you are giving up by using one.

**An extension point is an in-process, one-language interface.** A custom constraint has
no representation on the wire, so an engine driven over JSON cannot see it, and the
cross-language conformance harness cannot check that four implementations agree about it.
Use these to specialise one application in one language. A rule that must hold for every
caller of every binding belongs in the request as data instead -- as a `policy` rule, a
tag, or a field. docs/EXTENDING.md carries the full reasoning.
"""

from packvium import Container, Dimensions, Item, Length, Packer, PackingConfig
from packvium.constraints import ConstraintContext, ConstraintResult
from packvium.extensions import DefaultSolutionScorer, ExtensionRegistry

#: An example must not change answer merely because the host was busy. These solves need
#: a fraction of the budget; the generous wall-clock value is only a safety fuse, so a
#: loaded machine cannot cut the multi-start portfolio short and let a different start win.
SAFETY_FUSE_MS = 60_000

# ---------------------------------------------------------------------------------
# A custom placement constraint. `max_top_load` caps what may rest on an item, and
# `must_be_on_floor` pins one to the bottom -- but neither says "nothing fragile above
# waist height, because that is where it gets knocked off a trolley". That is a real
# warehouse rule with no field, so it is a constraint.
#
# A constraint answers about *one candidate position*. It never searches; it is asked
# many thousands of times per solve, so keep it O(1) in the number of placements
# wherever you can. This one is: it looks at the candidate's z coordinate and nothing else.
# ---------------------------------------------------------------------------------
class FragileHeightLimit:
    """Refuse to place a `fragile`-tagged item with its base above `limit`."""

    def __init__(self, limit: Length, tag: str = "fragile") -> None:
        self.limit = limit
        self.tag = tag

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if self.tag not in context.item.item.tags:
            return ConstraintResult.allow()
        # `Point` carries raw tick counts, not `Length` objects -- it is built once per
        # candidate and this is the hot path.
        if context.point.z <= self.limit.ticks:
            return ConstraintResult.allow()
        # The code is yours. It travels into the unpacked reason, so make it something a
        # human reading a failed order will understand.
        return ConstraintResult.reject(
            "fragile_too_high",
            f"base at {Length(context.point.z).decimal('mm')}mm is above the "
            f"{self.limit.decimal('mm')}mm limit for {self.tag!r} items",
        )


# The footprint is deliberately only as wide as one crate, so the column has to grow
# upwards and the rule has something to refuse. A rule that never fires teaches nothing.
items = [
    Item.create("crate", Dimensions.mm("400", "400", "300"), "12 kg", quantity=3),
    Item.create("vase", Dimensions.mm("400", "400", "200"), "2 kg", quantity=2, tags={"fragile"}),
]
containers = [Container.create("column", Dimensions.mm("400", "400", "1300"), max_payload="200 kg", quantity=1)]

unrestricted = Packer(PackingConfig.balanced(time_limit_ms=SAFETY_FUSE_MS)).pack(items, containers)
highest_vase = max(
    p.position.z for c in unrestricted.containers for p in c.placements if p.instance.item.id == "vase"
)
print("without the rule, the highest vase sits at", Length(highest_vase).decimal("mm"), "mm")

restricted = Packer(
    PackingConfig.balanced(time_limit_ms=SAFETY_FUSE_MS),
    ExtensionRegistry(placement_constraints=(FragileHeightLimit(Length.parse("400 mm")),)),
).pack(items, containers)

placed_vases = [p for c in restricted.containers for p in c.placements if p.instance.item.id == "vase"]
print("with a 400mm limit, vases sit at",
      sorted(Length(p.position.z).decimal("mm") for p in placed_vases),
      "and", len(restricted.unpacked), "were left behind")
for unpacked in restricted.unpacked:
    print("   left behind:", unpacked.instance.item.id, "->", unpacked.reason, unpacked.details)

# ---------------------------------------------------------------------------------
# A custom solution scorer. The six built-in objectives rank by space, packaging cost,
# billed weight, landed money, stack height or value. None of them cares whether the
# weight is spread *evenly across the containers* -- which is what a two-person lift or
# a van's axle balance actually depends on, and is nowhere in the request.
#
# Return a tuple of exact integers, lower is better, compared lexicographically. Keep
# `unpacked_count` first unless you genuinely mean "leave items behind to score better",
# and fall back to the canonical vector for everything your rule does not care about --
# otherwise two equally-balanced solutions are ranked by luck.
# ---------------------------------------------------------------------------------
class EvenlyLoadedContainers:
    """Prefer solutions whose containers weigh about the same."""

    def score(self, solution) -> tuple[int, ...]:
        loads = [sum(p.instance.item.weight.ticks for p in c.placements) for c in solution.containers]
        spread = max(loads) - min(loads) if loads else 0
        return (len(solution.unpacked), spread) + DefaultSolutionScorer().score(solution)[1:]


# Two heavy items and two light ones, two boxes, two slots each. Every arrangement uses
# the same volume, so the built-in objectives are indifferent -- and land on both anvils
# in one box. That is a 20kg box and a 2kg box.
lopsided_items = [
    Item.create("anvil", Dimensions.mm("200", "200", "200"), "10 kg", quantity=2),
    Item.create("pillow", Dimensions.mm("200", "200", "200"), "1 kg", quantity=2),
]
two_boxes = [Container.create("box", Dimensions.mm("400", "200", "200"), max_payload="50 kg", quantity=2)]

print()
for label, scorer in (("default", None), ("evenly loaded", EvenlyLoadedContainers())):
    result = Packer(PackingConfig.balanced(time_limit_ms=SAFETY_FUSE_MS), solution_scorer=scorer).pack(lopsided_items, two_boxes)
    contents = [sorted(p.instance.item.id for p in c.placements) for c in result.containers]
    weights = [c.payload_weight.decimal("kg") + " kg" for c in result.containers]
    print(f"{label:>15}: {contents}  ->  {weights}")

print()
print("Neither rule above exists on the wire. Hand the same request to the Rust or")
print("JavaScript engine and you get the unrestricted answer -- which is exactly why a")
print("rule everyone must obey belongs in the request as data, not in a class.")
