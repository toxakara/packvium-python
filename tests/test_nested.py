"""Multi-level packing: items into cartons, cartons onto a pallet.

Each level re-packs the previous level's full containers as items, so the handover —
outer dimensions, gross weight, upright orientation — is what these tests pin down.
"""

from __future__ import annotations

import pytest

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


def test_a_level_beam_width_of_one_is_the_greedy_chain():
    cartons = [container("long", 200, 100, 100), container("single", 100, 100, 100)]
    pallets = [container("tower", 100, 100, 200)]
    items = [item("cube", 100, 100, 100, quantity=2)]
    levels = [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)]

    def shape(result):
        return [(level.complete, [c.id for c in level.containers], level.score) for level in result.levels]

    assert shape(NestedPacker(1).pack(items, levels)) == shape(NestedPacker().pack(items, levels))


def test_a_wider_beam_finds_the_carton_set_that_fits_the_pallet():
    """The single long carton is the better carton level and cannot stand on the tower."""
    cartons = [container("long", 200, 100, 100), container("single", 100, 100, 100)]
    pallets = [container("tower", 100, 100, 200)]
    levels = [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)]
    items = [item("cube", 100, 100, 100, quantity=2)]

    greedy = NestedPacker(1).pack(items, levels)
    beam = NestedPacker(2).pack(items, levels)

    assert [level.complete for level in greedy.levels] == [True, False]
    assert [level.complete for level in beam.levels] == [True, True]
    assert [c.container.id for c in beam.levels[0].containers] == ["single", "single"]


def test_handoffs_that_are_the_same_items_occupy_one_beam_slot():
    """Packing against every carton and against `long` alone both hand up one `long#1`;
    counting that twice would fill a width of two and drop the chain that fits."""
    cartons = [container("long", 200, 100, 100), container("single", 100, 100, 100)]
    pallets = [container("tower", 100, 100, 200)]
    result = NestedPacker(2).pack([item("cube", 100, 100, 100, quantity=2)],
                                  [PackingLevel("carton", cartons), PackingLevel("pallet", pallets)])

    assert all(level.complete for level in result.levels)


def test_no_levels_is_an_empty_answer_at_any_width():
    assert NestedPacker(3).pack([item("cube", 100, 100, 100)], []).levels == ()


@pytest.mark.parametrize("width", [0, -1, True, 1.0, "2"])
def test_the_level_beam_width_is_a_positive_integer(width):
    with pytest.raises(ValueError, match="level_beam_width must be a positive integer"):
        NestedPacker(width)
