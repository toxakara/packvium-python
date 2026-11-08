"""1.10: gusseted bags, admitted by the flat-bag rule with the width girth b + c <= W + G."""

from __future__ import annotations

import pytest

from packvium import PackingConfig
from packvium.config import SolverProfile
from packvium.extensions import LandedCostSolutionScorer
from packvium.fixed_placements import FixedPlacementError
from packvium.flat_bag import fits_alone, girth_admits
from packvium.geometry import Dimensions, Point, Rotation
from packvium.models import (Container, Item, ItemInstance, PackedContainer, PackingRequest,
                             Placement, RateTable)
from packvium.request_errors import InvalidRequestError
from packvium.serialization import pack_from_dict
from packvium.solvers import (ContainerState, Deadline, GridSolver, HomogeneousBlockSolver,
                              SearchStats, default_constraints, find_candidates)
from packvium.units import Length, Weight
from packvium.validation import IndependentSolutionValidator


def ticks(millimetres: int) -> int:
    return Length.of(millimetres, "mm").ticks


def mm(length: int, width: int, height: int) -> Dimensions:
    return Dimensions.mm(length, width, height)


def pouch(gusset: int = 10, length: int = 300, width: int = 150, height: int = 100, **fields) -> Container:
    return Container("pouch", mm(length, width, height), deformable_boundary="gusseted_bag",
                     side_gusset_depth=Length(ticks(gusset)), **fields)


def book(quantity: int = 1) -> Item:
    return Item("book", mm(200, 100, 20), Weight(300), quantity=quantity, allowed_rotations=(Rotation.LWH,))


def placed(item: Item, index: int, z: int) -> Placement:
    origin = Point(0, 0, ticks(z))
    return Placement(ItemInstance(item, index), origin, Rotation.LWH, item.dimensions, origin, item.dimensions)


def request(quantity: int, boundary: dict, **container) -> dict:
    return {"items": [{"id": "book", "quantity": quantity, "weight": "300 g", "allowed_rotations": ["LWH"],
                       "dimensions": {"length": "200", "width": "100", "height": "20"}}],
            "containers": [{"id": "pouch", "quantity": 1, "deformable_boundary": boundary,
                            "inner_dimensions": {"length": "300", "width": "150", "height": "100"}, **container}],
            "configuration": {"solver_profile": "fast", "time_limit_ms": 60000,
                              "effort_budget": {"max_search_nodes": 100000}}}


def gusseted(depth: object) -> dict:
    return {"kind": "gusseted_bag", "side_gusset_depth": depth}


class TestTheRule:
    def test_the_gusset_widens_only_the_width_girth(self):
        inner = mm(300, 150, 100)
        assert not girth_admits(inner, ticks(200), ticks(100), ticks(60))
        assert girth_admits(inner, ticks(200), ticks(100), ticks(60), ticks(10))
        assert not girth_admits(inner, ticks(200), ticks(100), ticks(60) + 1, ticks(10))
        assert not girth_admits(inner, ticks(241), 0, ticks(60), ticks(150))

    def test_a_single_box_fits_a_gusseted_bag_its_flat_twin_refuses(self):
        inner = mm(200, 150, 100)
        assert not fits_alone(inner, mm(100, 100, 60))
        assert fits_alone(inner, mm(100, 100, 60), ticks(10))
        assert not fits_alone(inner, mm(100, 160, 10), ticks(150))


class TestTheModel:
    def test_the_depth_is_kept_and_read_in_ticks(self):
        assert pouch(25).side_gusset_ticks == ticks(25) and pouch().is_flat_bag
        assert Container("box", mm(1, 1, 1)).side_gusset_ticks == 0

    @pytest.mark.parametrize("fields, reason, detail", [
        ({"deformable_boundary": "gusseted_bag"}, "missing_field", "a gusseted_bag requires side_gusset_depth"),
        ({"deformable_boundary": "flat_bag", "side_gusset_depth": Length(1)}, "not_allowed",
         "side_gusset_depth requires deformable_boundary gusseted_bag"),
        ({"side_gusset_depth": Length(1)}, "not_allowed",
         "side_gusset_depth requires deformable_boundary gusseted_bag"),
        ({"deformable_boundary": "gusseted_bag", "side_gusset_depth": Length(0)}, "invalid_value",
         "side_gusset_depth must be positive"),
        ({"deformable_boundary": "gusseted_bag", "side_gusset_depth": Length(ticks(150) + 1)}, "invalid_value",
         "side_gusset_depth exceeds inner_dimensions width"),
    ])
    def test_a_malformed_gusset_is_refused(self, fields, reason, detail):
        with pytest.raises(InvalidRequestError) as caught:
            Container("pouch", mm(300, 150, 100), **fields)
        assert (caught.value.reason, caught.value.detail) == (reason, detail)

    def test_a_gusset_as_deep_as_the_bag_is_wide_is_admitted(self):
        assert pouch(150).side_gusset_ticks == ticks(150)

    def test_a_gusseted_bag_refuses_the_wall_anchored_fields_after_its_gusset(self):
        with pytest.raises(InvalidRequestError, match="outer_dimensions cannot be combined"):
            pouch(outer_dimensions=mm(310, 160, 110))


class TestTheRequest:
    def test_the_json_path_reads_the_depth_in_the_request_unit(self):
        result = pack_from_dict({**request(3, gusseted({"value": "1", "unit": "cm"})), "units": {"length": "mm"}})
        assert result["unpacked_items"] == []

    def test_a_flat_bag_refuses_a_gusset_as_an_unknown_field(self):
        with pytest.raises(InvalidRequestError) as caught:
            pack_from_dict(request(1, {"kind": "flat_bag", "side_gusset_depth": "10"}))
        assert (caught.value.field, caught.value.reason) == (
            "/containers/0/deformable_boundary/side_gusset_depth", "not_allowed")


class TestThePacking:
    def test_the_gusset_admits_the_layer_the_flat_bag_refuses(self):
        flat = pack_from_dict(request(4, {"kind": "flat_bag"}))
        wide = pack_from_dict(request(4, gusseted("25")))
        assert [len(c["placements"]) for c in flat["containers"]] == [2]
        assert [len(c["placements"]) for c in wide["containers"]] == [3]
        assert wide["containers"][0]["achieved_dimensions"]["height"]["value"] == "60"

    def test_the_lattice_and_block_solvers_defer_to_the_candidate_check(self):
        items = [ItemInstance(book(), index) for index in range(1, 5)]
        config = PackingConfig(profile=SolverProfile.FAST)
        solution = GridSolver().pack_one(pouch(), 1, items, config, SearchStats(), Deadline(60_000))
        assert solution.state.lattice_summary is None
        assert len(solution.state.placements) == 3
        assert not HomogeneousBlockSolver()._supports(pouch(), items, config)

    def test_a_candidate_one_tick_past_the_widened_girth_is_never_proposed(self):
        state = ContainerState(pouch(), 1)
        for level in range(3):
            state.add(placed(book(), level + 1, 20 * level))
        config = PackingConfig()
        assert find_candidates(state, ItemInstance(book(), 4), config, default_constraints(config),
                               SearchStats(), Deadline(60_000)) == []

    def test_an_item_only_the_gusset_admits_is_not_proven_unfit(self):
        brick = {"id": "brick", "allowed_rotations": ["LWH"],
                 "dimensions": {"length": "100", "width": "100", "height": "60"}}
        base = request(1, {"kind": "flat_bag"})
        base["items"] = [brick]
        base["containers"][0]["inner_dimensions"]["length"] = "200"
        assert pack_from_dict(base)["unpacked_items"][0]["reason"] == "no_compatible_container_dimensions"
        base["containers"][0]["deformable_boundary"] = gusseted("10")
        assert pack_from_dict(base)["unpacked_items"] == []


class TestBillingValidationAndFixedPlacements:
    def test_a_gusseted_bag_is_billed_on_its_achieved_package(self):
        table = RateTable((500, 2000), (400, 1100))
        packed = PackedContainer(pouch(rate_table=table), 1, (placed(book(), 1, 0),))
        assert packed.billed_dimensions == mm(200, 100, 20)
        config = PackingConfig(objective="lowest_landed_cost", dimensional_weight_divisor=5000,
                               dimensional_weight_length_unit="cm", dimensional_weight_weight_unit="kg")
        assert LandedCostSolutionScorer.from_config(config).score_containers([packed], [])[1] == 400

    def test_the_validator_reads_the_gusset(self):
        three = tuple(placed(book(3), i, 20 * (i - 1)) for i in (1, 2, 3))
        validator = IndependentSolutionValidator()
        assert validator.validate(PackingRequest((book(3),), (pouch(),)),
                                  (PackedContainer(pouch(), 1, three),)).issues == ()
        narrow = pouch(9)
        issues = validator.validate(PackingRequest((book(3),), (narrow,)),
                                    (PackedContainer(narrow, 1, three),)).issues
        assert [(issue.code, issue.detail) for issue in issues] == [("bag_girth_exceeded", "pouch#1")]

    def test_a_fixed_set_one_tick_past_the_widened_girth_cannot_hold(self):
        fixed = [{"item_type": "book", "container_type": "pouch", "orientation": "LWH",
                  "position": {"x": "0", "y": "0", "z": str(20 * level)}} for level in range(3)]
        past = {**request(3, gusseted({"value": str(ticks(10) - 1), "unit": "ticks"})), "fixed_placements": fixed}
        with pytest.raises(FixedPlacementError) as caught:
            pack_from_dict(past)
        assert str(caught.value) == "invalid_fixed_placement: bag_girth_exceeded: pouch#1"
        assert pack_from_dict({**request(3, gusseted("10")), "fixed_placements": fixed})["unpacked_items"] == []
