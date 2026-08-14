"""Domain invariants enforced at construction.

An impossible item or container must be refused where it is built, not carried into
a solver to fail somewhere unrecognisable. These tests pin down what the constructors
guarantee to everything downstream.
"""

from __future__ import annotations

import pytest

from packvium import (AxisAlignedBox, Axle, Container, Dimensions, Item, Length, Obstacle,
                         PackedContainer, PackingRequest, Placement, Point, Rotation, Weight)


# ---------------------------------------------------------------------------- item

def test_an_item_needs_an_id():
    with pytest.raises(ValueError):
        Item.create("", Dimensions.mm(1, 1, 1))


def test_quantity_must_be_positive():
    with pytest.raises(ValueError):
        Item.create("a", Dimensions.mm(1, 1, 1), quantity=0)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_support_ratio_is_a_proportion(ratio):
    with pytest.raises(ValueError):
        Item.create("a", Dimensions.mm(1, 1, 1), minimum_support_ratio=ratio)


def test_keep_upright_narrows_the_allowed_rotations():
    assert Item.create("a", Dimensions.mm(1, 2, 3), keep_upright=True).allowed_rotations == Rotation.upright()


def test_keep_upright_conflicting_with_the_rotation_list_is_refused():
    """Silently allowing zero orientations would make the item unplaceable for a reason
    no caller could see in the result."""
    with pytest.raises(ValueError):
        Item.create("a", Dimensions.mm(1, 2, 3), keep_upright=True, allowed_rotations=(Rotation.LHW,))


def test_weights_are_parsed_on_every_construction_path():
    """Both `create` and the plain constructor: an unparsed weight used to travel all
    the way into the solver before failing on a missing attribute."""
    assert Item.create("a", Dimensions.mm(1, 1, 1), "1 kg").weight == Weight.of(1, "kg")
    direct = Item("a", Dimensions.mm(1, 1, 1), weight="2 kg", max_top_load="500 g")
    assert direct.weight == Weight.of(2, "kg")
    assert direct.max_top_load == Weight.of(500, "g")


def test_an_absent_bearing_limit_stays_absent():
    assert Item.create("a", Dimensions.mm(1, 1, 1)).max_top_load is None


def test_tag_collections_are_frozen():
    item = Item.create("a", Dimensions.mm(1, 1, 1), tags=["fragile"], incompatible_tags=["heavy"])
    assert item.tags == frozenset({"fragile"})
    assert item.incompatible_tags == frozenset({"heavy"})


def test_stop_index_is_absent_by_default():
    """The single-stop, no-route case (every request without a route) must stay
    completely unaffected: nothing populates this field, so it stays unset."""
    assert Item.create("a", Dimensions.mm(1, 1, 1)).stop_index is None


def test_stop_index_must_be_non_negative():
    with pytest.raises(ValueError):
        Item.create("a", Dimensions.mm(1, 1, 1), stop_index=-1)


def test_a_zero_stop_index_is_accepted():
    assert Item.create("a", Dimensions.mm(1, 1, 1), stop_index=0).stop_index == 0


def test_instances_are_numbered_from_one():
    instances = Item.create("widget", Dimensions.mm(1, 1, 1), quantity=3).instances()
    assert [i.id for i in instances] == ["widget#1", "widget#2", "widget#3"]
    assert all(i.item.id == "widget" for i in instances)


def test_instances_share_the_item_measurements():
    instance, = Item.create("a", Dimensions.mm(1, 2, 3), "1 kg").instances()
    assert instance.dimensions == Dimensions.mm(1, 2, 3)
    assert instance.weight == Weight.of(1, "kg")


# ----------------------------------------------------------------------- container

def test_a_container_needs_an_id():
    with pytest.raises(ValueError):
        Container.create("", Dimensions.mm(1, 1, 1))


def test_container_quantity_and_cost_are_sanity_checked():
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(1, 1, 1), quantity=0)
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(1, 1, 1), cost_minor=-1)


def test_outer_dimensions_cannot_be_smaller_than_inner():
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(10, 10, 10), outer_dimensions=Dimensions.mm(9, 10, 10))


def test_the_front_axle_must_be_nearer_the_origin_than_the_rear_axle():
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(1000, 100, 100), axles=(Axle(Length.mm(500)), Axle(Length.mm(500))))
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(1000, 100, 100), axles=(Axle(Length.mm(600)), Axle(Length.mm(400))))


def test_axle_positions_must_lie_within_the_container_length():
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(1000, 100, 100), axles=(Axle(Length.mm(100)), Axle(Length.mm(1001))))


def test_an_obstacle_must_lie_inside_the_container():
    outside = Obstacle("post", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(20, 5, 5)))
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(10, 10, 10), obstacles=(outside,))


def test_an_obstacle_flush_with_the_far_wall_is_accepted():
    """Containment is inclusive at the far face, so a shelf filling one end is legal."""
    flush = Obstacle("shelf", AxisAlignedBox(Point(Length.mm(5).ticks, 0, 0), Dimensions.mm(5, 10, 10)))
    assert Container.create("c", Dimensions.mm(10, 10, 10), obstacles=(flush,)).obstacles == (flush,)


def test_a_single_box_obstacle_is_the_one_box_case_of_the_union():
    """Nothing about existing single-box construction changes."""
    post = Obstacle("post", AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(5, 5, 5)))
    assert post.boxes == (post.box,)


def test_a_multi_box_obstacle_approximates_a_non_rectangular_shape():
    """A wheel arch or tapered roof: a union of exact boxes, not a single
    rectangle, expressed via additional_boxes."""
    step1 = AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(10, 10, 5))
    step2 = AxisAlignedBox(Point(0, 0, 5), Dimensions.mm(6, 10, 5))
    arch = Obstacle("wheel_arch", step1, additional_boxes=(step2,))
    assert arch.boxes == (step1, step2)


def test_every_box_in_a_union_must_lie_inside_the_container():
    step1 = AxisAlignedBox(Point(0, 0, 0), Dimensions.mm(5, 5, 5))
    outside = AxisAlignedBox(Point(Length.mm(50).ticks, 0, 0), Dimensions.mm(5, 5, 5))
    arch = Obstacle("arch", step1, additional_boxes=(outside,))
    with pytest.raises(ValueError):
        Container.create("c", Dimensions.mm(10, 10, 10), obstacles=(arch,))


# ------------------------------------------------------------------- packed result

def placement(instance, x=0, y=0, z=0) -> Placement:
    dims = instance.item.dimensions
    return Placement(instance, Point(x, y, z), Rotation.LWH, dims, Point(x, y, z), dims)


def test_packed_container_aggregates_weight_and_volume():
    item = Item.create("a", Dimensions.mm(10, 10, 10), "500 g", quantity=2)
    box = Container.create("c", Dimensions.mm(20, 10, 10), tare_weight="200 g")
    first, second = item.instances()
    packed = PackedContainer(box, 1, (placement(first), placement(second, x=160_000)))

    assert packed.id == "c#1"
    assert packed.payload_weight == Weight.of(1, "kg")
    assert packed.gross_weight == Weight.of("1.2", "kg")
    assert packed.used_volume == 2 * item.dimensions.volume
    assert packed.utilization == pytest.approx(1.0)


def test_a_packed_container_can_be_re_packed_as_an_item():
    """Nested packing feeds a full carton into the next level, so it must present the
    outer dimensions and the gross weight, not the inner ones."""
    box = Container.create("c", Dimensions.mm(10, 10, 10), tare_weight="100 g",
                           outer_dimensions=Dimensions.mm(12, 12, 12))
    instance, = Item.create("a", Dimensions.mm(10, 10, 10), "900 g").instances()
    as_item = PackedContainer(box, 1, (placement(instance),)).as_item()

    assert as_item.dimensions == Dimensions.mm(12, 12, 12)
    assert as_item.weight == Weight.of(1, "kg")
    assert as_item.keep_upright is True
    assert as_item.metadata["source_packed_container"] == "c#1"


# ------------------------------------------------------------------------- request

def test_a_request_needs_at_least_one_item_and_one_container():
    item = Item.create("a", Dimensions.mm(1, 1, 1))
    box = Container.create("c", Dimensions.mm(1, 1, 1))
    with pytest.raises(ValueError):
        PackingRequest((), (box,))
    with pytest.raises(ValueError):
        PackingRequest((item,), ())


def test_duplicate_ids_are_refused():
    """Two items sharing an id would make every per-instance report ambiguous."""
    item = Item.create("a", Dimensions.mm(1, 1, 1))
    box = Container.create("c", Dimensions.mm(1, 1, 1))
    with pytest.raises(ValueError):
        PackingRequest((item, item), (box,))
    with pytest.raises(ValueError):
        PackingRequest((item,), (box, box))


def test_request_expands_quantities_into_instances():
    request = PackingRequest(
        (Item.create("a", Dimensions.mm(1, 1, 1), quantity=2), Item.create("b", Dimensions.mm(1, 1, 1))),
        (Container.create("c", Dimensions.mm(1, 1, 1)),),
    )
    assert [i.id for i in request.instances] == ["a#1", "a#2", "b#1"]


# ------------------------------------------------------------- rate table

@pytest.mark.parametrize("brackets, prices, why", [
    ((), (), "no bracket at all"),
    ((100, 200), (10,), "a length mismatch pairs a weight with another band's price"),
    ((0,), (10,), "a non-positive bound"),
    ((200, 100), (10, 20), "a descending ladder hides an unreachable band"),
    ((100, 100), (10, 20), "equal adjacent bounds leave the second permanently unreachable"),
    ((100,), (-1,), "a negative price"),
])
def test_a_malformed_tariff_is_refused_where_it_is_built(brackets, prices, why):
    """A malformed tariff misprices silently, which is why each of these is refused at
    construction rather than at the point a shipment is quoted."""
    from packvium.models import RateTable

    with pytest.raises(ValueError):
        RateTable(brackets, prices)


def test_negative_surcharge_components_are_refused():
    from packvium.models import RateTable

    with pytest.raises(ValueError):
        RateTable((100,), (10,), minimum_charge_minor=-1)
    with pytest.raises(ValueError):
        RateTable((100,), (10,), fuel_surcharge_permille=-1)


def test_a_price_band_may_dip_because_a_promotional_rate_card_is_real():
    """Deliberately *not* rejected: the table is read by bracket, so a cheaper upper band
    prices correctly. Asserting this keeps a future "prices must ascend" guard from being
    added as an obvious-looking improvement."""
    from packvium.models import RateTable

    table = RateTable((100, 200), (900, 500))
    assert table.charge_minor(50) == 900
    assert table.charge_minor(150) == 500


def test_the_minimum_charge_is_a_floor_and_the_fuel_surcharge_rounds_up():
    """Both are exact integer arithmetic: the surcharge is a share of the base, and
    rounding it down would let a fractional unit of revenue vanish."""
    from packvium.models import RateTable

    floored = RateTable((100,), (10,), minimum_charge_minor=250)
    assert floored.charge_minor(50) == 250  # the published 10 never applies
    surcharged = RateTable((100,), (1_001,), fuel_surcharge_permille=1)
    assert surcharged.charge_minor(50) == 1_001 + 2  # 1.001 rounds up to 2, not down to 1


def test_a_weight_above_the_last_bracket_names_the_bracket_it_passed():
    from packvium.models import RateTable, UnratedWeightError

    with pytest.raises(UnratedWeightError, match="100 g"):
        RateTable((100,), (10,)).charge_minor(101)
