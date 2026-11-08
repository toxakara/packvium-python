"""A pallet whose load may pass its deck edge by `pallet_overhang_limit` per side."""

from __future__ import annotations

import copy

import pytest

from packvium import OverhangLimit, Packer, PackingConfig
from packvium._request_error import ObjectRefusal
from packvium.geometry import AxisAlignedBox, Dimensions, Point
from packvium.models import Axle, Container, Item, ItemInstance
from packvium.nested import NestedPacker, PackingLevel
from packvium.pallet_overhang import (Deck, achieved_dimensions, billed_dimensions, deck_alone,
                                      deck_contact_area, floor_support_ratio, overhang_used,
                                      solving_frame)
from packvium.request_errors import InvalidRequestError
from packvium.result_validation import validate
from packvium.serialization import UnsupportedFeatureError, pack_from_dict
from packvium.solvers import Deadline, GridSolver, HomogeneousBlockSolver, SearchStats
from packvium.units import Length, Weight


def ticks(millimetres: int) -> int:
    return Length.of(millimetres, "mm").ticks


def box(x: int, y: int, z: int, length: int, width: int, height: int) -> AxisAlignedBox:
    return AxisAlignedBox(Point(ticks(x), ticks(y), ticks(z)), Dimensions.mm(length, width, height))


DECK = Deck(ticks(50), ticks(50), ticks(1200), ticks(800))


def pallet(length: int = 50, width: int = 50, **fields) -> Container:
    return Container("pallet", Dimensions.mm(1200, 800, 1000),
                     pallet_overhang_limit=OverhangLimit(Length.of(length, "mm"), Length.of(width, "mm")),
                     **fields)


def request(items: list[dict], limit: dict | None = None, ratio: float = 0, **container) -> dict:
    declared = {"id": "pallet", "inner_dimensions": {"length": "1200", "width": "800", "height": "1000"},
                **container}
    if limit is not None:
        declared["pallet_overhang_limit"] = limit
    return {"units": {"length": "mm"}, "items": items, "containers": [declared],
            "configuration": {"minimum_support_ratio": ratio, "time_limit_ms": 60000}}


def carton(length: int, width: int, height: int, quantity: int = 1, **fields) -> dict:
    return {"id": "carton", "quantity": quantity, "allowed_rotations": ["LWH"],
            "dimensions": {"length": str(length), "width": str(width), "height": str(height)}, **fields}


FIFTY = {"length": "50", "width": "50"}


def positions(result: dict) -> list[tuple[int, int, int]]:
    return sorted((p["position"]["x"]["ticks"] // 16000, p["position"]["y"]["ticks"] // 16000,
                   p["position"]["z"]["ticks"] // 16000) for c in result["containers"] for p in c["placements"])


def achieved(result: dict) -> tuple[int, ...]:
    sides = result["containers"][0]["achieved_dimensions"]
    return tuple(sides[side]["ticks"] // 16000 for side in ("length", "width", "height"))


# ------------------------------------------------------------------- the frame and deck

def test_the_frame_widens_both_horizontal_axes_by_twice_the_limit_and_keeps_the_deck():
    frame = solving_frame(pallet(50, 30))
    assert frame.inner_dimensions == Dimensions.mm(1300, 860, 1000)
    assert frame.deck == Deck(ticks(50), ticks(30), ticks(1200), ticks(800))
    assert frame.pallet_overhang_limit is None
    assert frame.declared_inner == Dimensions.mm(1200, 800, 1000)
    assert solving_frame(frame) is frame
    assert deck_alone(frame).inner_dimensions == Dimensions.mm(1200, 800, 1000)
    assert deck_alone(frame).deck is None


def test_a_container_without_a_limit_is_its_own_frame():
    plain = Container("box", Dimensions.mm(10, 10, 10))
    assert solving_frame(plain) is plain


def test_only_the_deck_under_a_footprint_counts_as_floor_support():
    assert deck_contact_area(DECK, box(0, 0, 0, 100, 100, 10)) == ticks(50) * ticks(50)
    assert deck_contact_area(DECK, box(0, 0, 0, 50, 100, 10)) == 0
    assert floor_support_ratio(DECK, box(0, 0, 0, 100, 100, 10)) == 0.25
    assert floor_support_ratio(None, box(0, 0, 0, 100, 100, 10)) == 1.0


def test_the_overhang_used_is_the_sum_past_both_edges_of_each_axis():
    boxes = [box(0, 50, 0, 600, 400, 10), box(650, 450, 0, 600, 420, 10)]
    assert overhang_used(DECK, boxes) == (ticks(50), ticks(20))
    assert overhang_used(DECK, []) == (0, 0)
    assert achieved_dimensions(DECK, boxes) == Dimensions.mm(1250, 820, 10)


def test_billing_takes_the_larger_of_the_declared_footprint_and_the_grown_deck_per_axis():
    boxes = [box(0, 50, 0, 1250, 800, 10)]
    frame = solving_frame(pallet())
    assert billed_dimensions(frame, DECK, boxes) == Dimensions.mm(1250, 800, 1000)
    outer = solving_frame(pallet(outer_dimensions=Dimensions.mm(1210, 840, 1150)))
    assert billed_dimensions(outer, DECK, boxes) == Dimensions.mm(1250, 840, 1150)


# ------------------------------------------------------------------- packing

def test_a_load_the_deck_holds_never_overhangs():
    result = pack_from_dict(request([carton(400, 400, 300, quantity=6)], FIFTY))
    assert result["summary"]["unpacked_item_count"] == 0
    assert all(x >= 50 and y >= 50 for x, y, _ in positions(result))
    assert achieved(result) == (1200, 800, 300)


def test_overhang_is_used_when_it_places_more():
    result = pack_from_dict(request([carton(1250, 850, 100)], FIFTY, ratio=0.75))
    assert result["summary"]["packed_item_count"] == 1
    placement = result["containers"][0]["placements"][0]
    assert placement["support_ratio"] == f"{(1200 * 800) / (1250 * 850):.6f}"
    assert achieved(result) == (1250, 850, 100)
    assert validate(request([carton(1250, 850, 100)], FIFTY, ratio=0.75), result) == []


def test_the_deck_alone_bears_weight_so_a_high_ratio_refuses_the_overhang():
    """Alone, the crate opens no pallet, so there is no layout to diagnose support against;
    beside a carton that opens one, the same refusal is named (feature-pallet-overhang-
    support-binds.json)."""
    alone = pack_from_dict(request([carton(1250, 850, 100)], FIFTY, ratio=0.95))
    assert [item["reason"] for item in alone["unpacked_items"]] == ["no_feasible_placement"]
    beside = pack_from_dict(request([carton(1250, 850, 100), carton(400, 300, 200) | {"id": "small"}],
                                    FIFTY, ratio=0.95))
    assert [item["reason"] for item in beside["unpacked_items"]] == ["insufficient_support"]


def test_a_zero_limit_forbids_overhang_and_still_reports_the_achieved_extent():
    zero = {"length": "0", "width": "0"}
    result = pack_from_dict(request([carton(1250, 850, 100), carton(400, 300, 200, quantity=1) | {"id": "small"}], zero))
    assert [item["reason"] for item in result["unpacked_items"]] == ["no_compatible_container_dimensions"]
    assert achieved(result) == (1200, 800, 200)


def test_a_limit_beyond_the_need_is_not_spent():
    tight = pack_from_dict(request([carton(1250, 850, 100)], FIFTY))
    loose = pack_from_dict(request([carton(1250, 850, 100)], {"length": "500", "width": "500"}))
    assert achieved(tight) == achieved(loose) == (1250, 850, 100)


def test_the_result_reports_the_declared_container_and_the_frame_utilization():
    result = pack_from_dict(request([carton(1250, 850, 100)], FIFTY))
    container = result["containers"][0]
    assert {side: v["ticks"] // 16000 for side, v in container["inner_dimensions"].items()} == {
        "length": 1200, "width": 800, "height": 1000}
    assert container["outer_dimensions"] == container["inner_dimensions"]
    assert container["volume_utilization"] == f"{(1250 * 850 * 100) / (1300 * 900 * 1000):.6f}"


def test_the_lattice_and_block_fast_paths_defer_for_a_deck():
    frame = solving_frame(pallet())
    assert not HomogeneousBlockSolver()._supports(frame, (), PackingConfig())
    item = Item("cube", Dimensions.mm(100, 100, 100), Weight(1))
    instances = [ItemInstance(item, index) for index in range(4)]
    solution = GridSolver().pack_one(frame, 1, instances, PackingConfig(), SearchStats(), Deadline(60000))
    assert solution.dominant_lattice is False
    assert solution.state.lattice_summary is None
    assert len(solution.state.placements) == 4


def test_a_fixed_placement_past_the_deck_reports_its_deck_support():
    payload = request([carton(1250, 850, 100)], FIFTY)
    payload["fixed_placements"] = [{"item_type": "carton", "container_type": "pallet", "container_instance": 1,
                                    "orientation": "LWH",
                                    "position": {"x": "0", "y": "0", "z": "0"}}]
    result = pack_from_dict(payload)
    assert result["containers"][0]["placements"][0]["support_ratio"] == f"{(1200 * 800) / (1250 * 850):.6f}"
    assert validate(payload, result) == []


def test_a_nested_level_hands_up_the_overhanging_pallet_at_its_reached_size():
    crate = Item("crate", Dimensions.mm(1250, 850, 400), Weight.parse("10 kg"))
    truck = Container("truck", Dimensions.mm(1300, 900, 1100))
    result = NestedPacker().pack([crate], [PackingLevel("pallet", (pallet(),)), PackingLevel("truck", (truck,))])
    handed = result.levels[1].containers[0].placements[0]
    assert handed.dimensions == Dimensions.mm(1250, 850, 1000)


# ------------------------------------------------------------------- refusals

@pytest.mark.parametrize(("field", "value", "name"), [
    ("access_directions", ["+x"], "access_directions"),
    ("axles", [{"position": "0"}, {"position": "1200"}], "axles"),
    ("obstacles", [{"id": "post", "dimensions": {"length": "10", "width": "10", "height": "10"}}], "obstacles"),
    ("void_fill_reserve_ratio", 0.25, "void_fill_reserve_ratio"),
    ("made_to_measure", {"minimum": {"length": "100", "width": "100", "height": "100"},
                         "increment": {"length": "100", "width": "100", "height": "100"}}, "made_to_measure"),
])
def test_a_combination_without_a_defined_meaning_is_refused_by_name(field, value, name):
    with pytest.raises(UnsupportedFeatureError, match=rf"container\.pallet_overhang_limit\+{name}"):
        pack_from_dict(request([carton(100, 100, 100)], FIFTY, **{field: value}))


def test_an_empty_list_and_a_zero_reserve_beside_a_pallet_change_nothing():
    result = pack_from_dict(request([carton(100, 100, 100)], FIFTY, access_directions=[], obstacles=[],
                                    void_fill_reserve_ratio=0))
    assert result["status"] == "feasible"


def test_a_routed_item_is_refused_beside_any_pallet_that_allows_overhang():
    with pytest.raises(UnsupportedFeatureError, match=r"container\.pallet_overhang_limit\+item\.stop_index"):
        pack_from_dict(request([carton(100, 100, 100, stop_index=1)], FIFTY))
    assert pack_from_dict(request([carton(100, 100, 100, stop_index=1)]))["status"] == "feasible"


@pytest.mark.parametrize(("limit", "reason", "field"), [
    ("50", "wrong_type", "/containers/0/pallet_overhang_limit"),
    ({"width": "50"}, "missing_field", "/containers/0/pallet_overhang_limit/length"),
    ({"length": "50", "width": "50", "height": "1"}, "not_allowed", "/containers/0/pallet_overhang_limit/height"),
    ({"length": "-1", "width": "0"}, "negative_measure", "/containers/0/pallet_overhang_limit/length"),
    ({"length": {"value": str(2**52), "unit": "ticks"}, "width": "0"}, "above_maximum",
     "/containers/0/pallet_overhang_limit/length"),
])
def test_a_malformed_limit_is_a_request_error_at_its_own_pointer(limit, reason, field):
    with pytest.raises(InvalidRequestError) as refusal:
        pack_from_dict(request([carton(100, 100, 100)], limit))
    assert (refusal.value.reason, refusal.value.field) == (reason, field)


def test_a_flat_bag_refuses_the_limit_last_among_its_exclusions():
    with pytest.raises(InvalidRequestError) as refusal:
        pack_from_dict(request([carton(100, 100, 100)], FIFTY, deformable_boundary={"kind": "flat_bag"}))
    assert refusal.value.field == "/containers/0/pallet_overhang_limit"
    with pytest.raises(ObjectRefusal, match="pallet_overhang_limit cannot be combined with deformable_boundary"):
        Container("bag", Dimensions.mm(100, 100, 10), deformable_boundary="flat_bag",
                  pallet_overhang_limit=OverhangLimit(Length(0), Length(0)))


@pytest.mark.parametrize(("fields", "name"), [
    ({"axles": (Axle(Length(0)), Axle(Length.of(1200, "mm")))}, "axles"),
    ({"access_directions": ("+x",)}, "access_directions"),
    ({"void_fill_reserve_ratio": 0.5}, "void_fill_reserve_ratio"),
])
def test_the_object_path_refuses_the_same_combinations(fields, name):
    with pytest.raises(ObjectRefusal, match=f"{name} cannot be combined with pallet_overhang_limit"):
        pallet(**fields)


def test_the_object_path_holds_the_frame_to_the_admission_ceiling():
    huge = Container("pallet", Dimensions.mm(1200, 800, 1000),
                     pallet_overhang_limit=OverhangLimit(Length(2**52), Length(0)))
    with pytest.raises(InvalidRequestError) as refusal:
        Packer().pack([Item("cube", Dimensions.mm(10, 10, 10), Weight(1))], [huge])
    assert refusal.value.field == "/containers/0/pallet_overhang_limit/length"


# ------------------------------------------------------------------- the validator

def test_the_validator_recomputes_the_achieved_extent():
    payload = request([carton(1250, 850, 100)], FIFTY)
    result = pack_from_dict(payload)
    forged = copy.deepcopy(result)
    forged["containers"][0]["achieved_dimensions"]["length"]["ticks"] -= 1
    assert "achieved_dimensions_mismatch" in [issue.code for issue in validate(payload, forged)]
    del forged["containers"][0]["achieved_dimensions"]
    assert "achieved_dimensions_mismatch" in [issue.code for issue in validate(payload, forged)]


def test_the_validator_refuses_a_floor_placement_the_deck_does_not_carry():
    payload = request([carton(400, 400, 100)], {"length": "300", "width": "300"}, ratio=0.5)
    result = pack_from_dict(payload)
    assert validate(payload, result) == []
    moved = copy.deepcopy(result)
    where = moved["containers"][0]["placements"][0]["position"]
    where["x"]["ticks"], where["y"]["ticks"] = 0, 0
    codes = [issue.code for issue in validate(payload, moved)]
    assert "insufficient_support" in codes


def test_the_validator_refuses_a_centroid_off_the_deck_below_half_support():
    payload = request([carton(400, 400, 100)], {"length": "300", "width": "300"}, ratio=0.1)
    result = pack_from_dict(payload)
    moved = copy.deepcopy(result)
    where = moved["containers"][0]["placements"][0]["position"]
    where["x"]["ticks"], where["y"]["ticks"] = ticks(50), ticks(300)
    codes = [issue.code for issue in validate(payload, moved)]
    assert "centre_of_gravity_unsupported" in codes


def test_the_object_path_refuses_a_routed_item_beside_a_pallet():
    routed = Item("cube", Dimensions.mm(10, 10, 10), Weight(1), stop_index=1)
    with pytest.raises(InvalidRequestError) as refusal:
        Packer().pack([routed], [pallet()])
    assert (refusal.value.reason, refusal.value.field) == ("not_allowed", "/items/0/stop_index")


# ------------------------------------------------------------------- edges of the trial

from packvium.geometry import Rotation  # noqa: E402
from packvium.pallet_overhang import centring_offset  # noqa: E402
from packvium.models import Placement  # noqa: E402
from packvium.solvers import (ContainerState, ExtremePointSolver, SingleContainerSolution,  # noqa: E402
                              SolverOrchestrator, TimeLimitReached, _seated)


def test_no_boxes_need_no_seat():
    assert centring_offset(DECK, []) == (0, 0)


def test_a_load_spanning_the_frame_is_already_seated():
    result = pack_from_dict(request([carton(1300, 900, 100)], FIFTY))
    assert positions(result) == [(0, 0, 0)]


def _plank(index: int, x: int) -> Placement:
    item = Item("plank", Dimensions.mm(100, 100, 100), Weight(1))
    origin = Point(ticks(x), 0 + ticks(350), 0)
    return Placement(ItemInstance(item, index), origin, Rotation.LWH, item.dimensions, origin, item.dimensions)


def test_a_seat_that_would_starve_a_floor_placement_is_not_taken():
    frame = solving_frame(pallet())
    state = ContainerState(frame, 1)
    for placement in (_plank(0, 0), _plank(1, 1150)):
        state.add(placement)
    trial = SingleContainerSolution(state, ())
    assert _seated(frame, trial, PackingConfig(minimum_support_ratio=0.8)) is trial
    seated = _seated(frame, trial, PackingConfig(minimum_support_ratio=0.5))
    assert [p.position.x for p in seated.state.placements] == [ticks(25), ticks(1175)]


def test_stacked_cartons_ride_the_seat():
    payload = request([carton(1250, 850, 400) | {"id": "crate"}, carton(400, 300, 200, quantity=4)], FIFTY,
                      ratio=0.75)
    result = pack_from_dict(payload)
    assert result["summary"]["unpacked_item_count"] == 0
    assert validate(payload, result) == []


class _DeckThenTimeout:
    def __init__(self):
        self.calls = 0

    def pack_one(self, container, sequence, items, config, stats, deadline):
        self.calls += 1
        if self.calls == 2:
            raise TimeLimitReached()
        return ExtremePointSolver().pack_one(container, sequence, items, config, stats, deadline)


def test_a_frame_solve_that_runs_out_of_time_keeps_the_deck_rung():
    frame = solving_frame(pallet())
    crate = Item("crate", Dimensions.mm(1250, 850, 100), Weight(1))
    small = Item("small", Dimensions.mm(100, 100, 100), Weight(1))
    items = [ItemInstance(crate, 0), ItemInstance(small, 0)]
    one, trials = SolverOrchestrator()._pallet_trial(_DeckThenTimeout(), frame, 1, items, PackingConfig(),
                                                     SearchStats(), Deadline(60000))
    assert (trials, one.time_limit_reached, one.exhaustive) == (1, True, False)
    assert [p.instance.item.id for p in one.state.placements] == ["small"]
    assert one.state.placements[0].position.x == ticks(50)


def test_the_validator_bills_a_pallet_at_its_reached_size():
    payload = request([carton(1250, 850, 400)], FIFTY,
                      outer_dimensions={"length": "1210", "width": "810", "height": "1150"},
                      rate_table={"weight_brackets_g": [100000, 1000000], "prices_minor": [1000, 2500]})
    payload["configuration"].update(objective="shipping_cost", dimensional_weight_divisor=5000,
                                    dimensional_weight_length_unit="cm", dimensional_weight_weight_unit="kg")
    result = pack_from_dict(payload)
    assert result["score"][1] == 1955000000000
    assert validate(payload, result) == []
    forged = copy.deepcopy(result)
    forged["score"][1] = 1000 * 10**9
    assert "score_mismatch" in [issue.code for issue in validate(payload, forged)]
