"""Multi-level packing: items into cartons, cartons onto a pallet.

Each level re-packs the previous level's full containers as items, so the handover —
outer dimensions, gross weight, upright orientation — is what these tests pin down.
"""

from __future__ import annotations

from packvium import (Dimensions, NestedPacker, PackingConfig, PackingLevel, Weight)
from support import container, item


def test_two_levels_are_packed_in_order():
    cartons = [container("carton", 200, 200, 100)]
    pallets = [container("pallet", 200, 200, 200)]
    result = NestedPacker().pack(
        [item("cube", 100, 100, 100, quantity=8)],
        [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)],
    )

    assert len(result.levels) == 2
    assert all(level.complete for level in result.levels)
    assert result.levels[0].to_dict()["summary"]["container_count"] == 2
    assert result.levels[1].to_dict()["summary"]["container_count"] == 1


def test_the_second_level_packs_the_first_level_containers_by_their_ids():
    cartons = [container("carton", 200, 200, 100)]
    pallets = [container("pallet", 200, 200, 200)]
    result = NestedPacker().pack(
        [item("cube", 100, 100, 100, quantity=8)],
        [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)],
    )
    on_pallet = {p.instance.item.id for p in result.levels[1].containers[0].placements}
    assert on_pallet == {"carton#1", "carton#2"}


def test_gross_weight_carries_up_to_the_next_level():
    """The pallet has to bear the cartons and their contents, not just the contents."""
    cartons = [container("carton", 200, 200, 100, tare_weight="500 g")]
    pallets = [container("pallet", 200, 200, 200, max_payload="10 kg")]
    result = NestedPacker().pack(
        [item("cube", 100, 100, 100, quantity=8, weight="1 kg")],
        [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)],
    )
    carton_as_item = result.levels[1].containers[0].placements[0].instance.item
    assert carton_as_item.weight == Weight.of("4.5", "kg")


def test_outer_dimensions_are_what_the_next_level_sees():
    cartons = [container("carton", 100, 100, 100, outer_dimensions=Dimensions.mm(110, 110, 110))]
    pallets = [container("pallet", 220, 110, 110)]
    result = NestedPacker().pack(
        [item("cube", 100, 100, 100, quantity=2)],
        [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)],
    )
    assert result.levels[1].containers[0].placements[0].instance.item.dimensions == Dimensions.mm(110, 110, 110)


def test_packing_stops_at_the_first_level_that_could_not_finish():
    """Feeding an incomplete level upward would silently drop whatever it left behind."""
    cartons = [container("carton", 100, 100, 100, quantity=1)]
    pallets = [container("pallet", 1_000, 1_000, 1_000)]
    result = NestedPacker().pack(
        [item("cube", 90, 90, 90, quantity=4)],
        [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)],
    )

    assert len(result.levels) == 1
    assert not result.levels[0].complete


def test_a_level_may_carry_its_own_configuration():
    cartons = [container("carton", 200, 200, 100)]
    pallets = [container("pallet", 200, 200, 200)]
    result = NestedPacker().pack(
        [item("cube", 100, 100, 100, quantity=8)],
        [PackingLevel("carton", cartons, PackingConfig.fast(seed=5)),
         PackingLevel("pallet", pallets, PackingConfig.quality(time_limit_ms=1_000, seed=9))],
    )
    assert [level.algorithm.seed for level in result.levels] == [5, 9]
    assert [level.algorithm.profile for level in result.levels] == ["fast", "quality"]


def test_a_single_level_behaves_like_an_ordinary_pack():
    cartons = [container("carton", 200, 200, 100)]
    result = NestedPacker().pack([item("cube", 100, 100, 100, quantity=4)], [PackingLevel("carton", cartons)])

    assert len(result.levels) == 1
    assert result.levels[0].complete
