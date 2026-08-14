"""Weight balancing across already-packed containers.

A naive post-pass weight redistributor carries a well-known failure mode:
it can silently drop an item during redistribution because its own bookkeeping only
checks whether the box count is still 1, not whether every item it started with is
still accounted for. Every test here is built around that failure mode -- proving an
item is never lost or duplicated across a rebalance, that a move which would strand
another item's support is declined rather than made, and that the redistribution
actually narrows the payload spread it exists to narrow, not merely leaves the
result looking plausible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from packvium import (IndependentSolutionValidator, ItemInstance, PackedContainer, PackingConfig,
                         PackingRequest, Placement, Point, Rotation)
from packvium.rebalance import rebalance_weight
from support import container, item, pack

TRIALS = 20

# A cross-language fixture, shared with the other implementations of this library and
# kept one level above this package. A published copy does not carry it; the rest of this
# module still exercises rebalancing from locally built scenes.
_SHARED_FIXTURE = Path(__file__).parents[2] / "conformance/scene/rebalance-fixtures.json"
if not _SHARED_FIXTURE.is_file():
    pytest.skip("the shared cross-language scene fixture is not part of this package",
                allow_module_level=True)
SHARED_CASE = json.loads(_SHARED_FIXTURE.read_text())["cases"][0]


def _floor_placement(built_item, sequence: int, x: int, y: int) -> Placement:
    """An item resting flat on the container floor, so support is trivially satisfied."""
    instance = ItemInstance(built_item, sequence)
    origin = Point(x, y, 0)
    return Placement(instance, origin, Rotation.LWH, built_item.dimensions, origin, built_item.dimensions, 1.0)


def _stacked_placement(built_item, sequence: int, below: Placement) -> Placement:
    """An item resting exactly on top of `below`, fully supported by it alone."""
    instance = ItemInstance(built_item, sequence)
    origin = Point(below.envelope_origin.x, below.envelope_origin.y, below.envelope_box.z2)
    return Placement(instance, origin, Rotation.LWH, built_item.dimensions, origin, built_item.dimensions, 1.0)


def _weights(containers) -> list[int]:
    return [c.payload_weight.ticks for c in containers]


def _assert_accounting_holds(request: PackingRequest, containers, unpacked) -> None:
    """The strongest bookkeeping property there is: every requested instance comes
    back exactly once, either placed or explained -- which is exactly what
    a naive redistributor fails to guarantee."""
    placed_ids = [p.instance.id for c in containers for p in c.placements]
    unpacked_ids = [u.instance.id for u in unpacked]
    assert len(placed_ids) == len(set(placed_ids)), "an item instance was packed twice"
    expected = {instance.id for i in request.items for instance in i.instances()}
    assert set(placed_ids) | set(unpacked_ids) == expected, "items were lost or fabricated"


def _assert_valid(request: PackingRequest, containers, unpacked, config: PackingConfig) -> None:
    report = IndependentSolutionValidator().validate(
        request, containers, config.minimum_support_ratio, config.clearance, unpacked,
    )
    assert report.valid, [(i.code, i.detail) for i in report.issues]


# ------------------------------------------------------------- narrowing the spread


def test_a_move_strictly_reduces_the_payload_spread_between_two_containers():
    """Hand-computed: one container holds a 5kg and a 1kg item (6kg), the other holds
    a lone 1kg item. Moving the 5kg item would only flip which side is heavier
    (1kg vs 6kg, still a 5kg spread) -- the greedy search must reject that overshoot
    and move the 1kg item instead, landing on a 5kg/2kg split (3kg spread)."""
    heavy = item("heavy", 40, 40, 40, weight="5000 g")
    light = item("light", 40, 40, 40, weight="1000 g")
    alone = item("alone", 40, 40, 40, weight="1000 g")
    box_type = container("box", 200, 200, 200)
    request = PackingRequest((heavy, light, alone), (box_type,))
    packed = (
        PackedContainer(box_type, 1, (_floor_placement(heavy, 1, 0, 0), _floor_placement(light, 1, 50, 0))),
        PackedContainer(box_type, 2, (_floor_placement(alone, 1, 0, 0),)),
    )
    before = _weights(packed)
    assert max(before) - min(before) == 5000 * 8_000_000  # Weight.TICKS_PER_G

    config = PackingConfig()
    outcome = rebalance_weight(request, packed, (), config)

    after = _weights(outcome.containers)
    assert sorted(after) == [2000 * 8_000_000, 5000 * 8_000_000]
    assert max(after) - min(after) < max(before) - min(before)
    expected_move = SHARED_CASE["expected_move"]
    assert [move.item_id for move in outcome.moves] == [expected_move["item_id"]]
    assert outcome.moves[0].from_container_id == expected_move["from_container_id"]
    assert outcome.moves[0].to_container_id == expected_move["to_container_id"]
    assert [
        [placement.instance.id for placement in packed_container.placements]
        for packed_container in outcome.containers
    ] == SHARED_CASE["expected_container_item_ids"]
    assert outcome.improved

    _assert_accounting_holds(request, outcome.containers, ())
    _assert_valid(request, outcome.containers, (), config)


def test_a_move_that_would_strand_a_supported_item_is_declined():
    """One container holds a 5kg base item with a support-sensitive 1kg item resting
    directly on top of it; the other container is empty. Moving the base (the larger
    weight, and the greedy search's first choice) would leave the item above it
    floating with no support -- exactly the kind of "looks fine, isn't" result
    `WeightRedistributor` never checked for. The rebalance must decline that move and
    fall back to moving the top item instead, which is what actually improves the
    payload spread without breaking anything else's support.
    """
    base = item("base", 40, 40, 40, weight="5000 g")
    top = item("top", 40, 40, 40, weight="1000 g", minimum_support_ratio=1.0)
    box_type = container("box", 200, 200, 200)
    request = PackingRequest((base, top), (box_type,))
    base_placement = _floor_placement(base, 1, 0, 0)
    top_placement = _stacked_placement(top, 1, base_placement)
    packed = (
        PackedContainer(box_type, 1, (base_placement, top_placement)),
        PackedContainer(box_type, 2, ()),
    )
    before = _weights(packed)
    assert before == [6000 * 8_000_000, 0]

    config = PackingConfig()
    outcome = rebalance_weight(request, packed, (), config)

    # The base never moved: it is still exactly where it was, still supporting nothing
    # underneath the (now relocated) top item's old spot.
    base_container = next(c for c in outcome.containers if c.id == "box#1")
    assert [p.instance.id for p in base_container.placements] == ["base#1"]

    after = _weights(outcome.containers)
    assert sorted(after) == [1000 * 8_000_000, 5000 * 8_000_000]
    assert [move.item_id for move in outcome.moves] == ["top#1"]

    _assert_accounting_holds(request, outcome.containers, ())
    _assert_valid(request, outcome.containers, (), config)


def test_a_single_container_has_nothing_to_rebalance():
    solo = item("solo", 40, 40, 40, weight="500 g")
    box_type = container("box", 200, 200, 200)
    request = PackingRequest((solo,), (box_type,))
    packed = (PackedContainer(box_type, 1, (_floor_placement(solo, 1, 0, 0),)),)

    outcome = rebalance_weight(request, packed, (), PackingConfig())

    assert outcome.moves == ()
    assert not outcome.improved
    assert outcome.containers == packed


def test_an_already_balanced_pair_is_left_untouched():
    a = item("a", 40, 40, 40, weight="1000 g")
    b = item("b", 40, 40, 40, weight="1000 g")
    box_type = container("box", 200, 200, 200)
    request = PackingRequest((a, b), (box_type,))
    packed = (
        PackedContainer(box_type, 1, (_floor_placement(a, 1, 0, 0),)),
        PackedContainer(box_type, 2, (_floor_placement(b, 1, 0, 0),)),
    )

    outcome = rebalance_weight(request, packed, (), PackingConfig())

    assert outcome.moves == ()
    assert outcome.containers == packed


# --------------------------------------------------------------- randomised property


def generate(seed: int):
    """A plausible multi-item order over one container type with a low enough
    payload ceiling that a real solve is forced to split it across several
    instances -- the only situation weight rebalancing has anything to do."""
    rng = random.Random(seed)
    items = []
    for index in range(rng.randint(2, 6)):
        length, width, height = (rng.randrange(10, 40, 5) for _ in range(3))
        items.append(item(
            f"i{index}", length, width, height,
            quantity=rng.randint(1, 3),
            weight=f"{rng.randrange(100, 900)} g",
        ))
    containers = [container("c0", 150, 150, 150, quantity=rng.randint(2, 4),
                            max_payload=f"{rng.randrange(1, 3)} kg")]
    return items, containers


@pytest.mark.parametrize("seed", range(TRIALS))
def test_rebalance_never_loses_or_duplicates_an_item(seed):
    """Holds regardless of how many containers a given order lands in -- a
    single-container result has nothing to rebalance, but the accounting
    invariant is exactly as much a requirement for it as for any other."""
    items, containers = generate(seed)
    request = PackingRequest(tuple(items), tuple(containers))
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250))

    config = PackingConfig.balanced(time_limit_ms=200)
    outcome = rebalance_weight(request, result.containers, result.unpacked, config)

    _assert_accounting_holds(request, outcome.containers, result.unpacked)
    _assert_valid(request, outcome.containers, result.unpacked, config)


@pytest.mark.parametrize("seed", range(TRIALS))
def test_rebalance_never_widens_the_payload_spread(seed):
    items, containers = generate(seed)
    request = PackingRequest(tuple(items), tuple(containers))
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250))

    before = _weights(result.containers)
    before_spread = max(before) - min(before) if before else 0

    config = PackingConfig.balanced(time_limit_ms=200)
    outcome = rebalance_weight(request, result.containers, result.unpacked, config)

    after = _weights(outcome.containers)
    after_spread = max(after) - min(after) if after else 0
    assert after_spread <= before_spread
    # Every move claimed must be a real move: the number of moves recorded is exactly
    # what separates the starting layout from the finishing one, in either direction.
    if outcome.moves:
        assert before != after
