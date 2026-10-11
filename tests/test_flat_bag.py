"""Flat bags, admitted by the girth rule a + c <= L and b + c <= W."""

from __future__ import annotations

import random

import pytest

from packvium import PackingConfig
from packvium.config import SolverProfile
from packvium.constraints import StopAccessibilityConstraint
from packvium.extensions import LandedCostSolutionScorer, ShippingCostSolutionScorer
from packvium.fixed_placements import FixedPlacementError
from packvium.flat_bag import (as_dimensions, extent_of, fits_alone, girth_admits,
                               grown_extent)
from packvium.geometry import Dimensions, Point, Rotation
from packvium.models import (Container, Item, ItemInstance, PackedContainer, PackingRequest,
                             Placement, RateTable)
from packvium.request_errors import InvalidRequestError
from packvium.serialization import pack_from_dict
from packvium.solvers import (ContainerState, Deadline, GridSolver, HomogeneousBlockSolver,
                              SearchStats, SolverOrchestrator, _billed_dimensional_weight,
                              find_candidates, default_constraints)
from packvium.units import Length, Weight
from packvium.validation import IndependentSolutionValidator

BAG = {"deformable_boundary": {"kind": "flat_bag"}}


def ticks(millimetres: int) -> int:
    return Length.of(millimetres, "mm").ticks


def mm(length: int, width: int, height: int) -> Dimensions:
    return Dimensions.mm(length, width, height)


def bag(length: int = 240, width: int = 140, height: int = 100, **fields) -> Container:
    return Container("mailer", mm(length, width, height), deformable_boundary="flat_bag", **fields)


def book(quantity: int = 1, rotations: tuple[Rotation, ...] = (Rotation.LWH,)) -> Item:
    return Item("book", mm(200, 100, 20), Weight(300), quantity=quantity, allowed_rotations=rotations)


def placed(item: Item, index: int, x: int, y: int, z: int) -> Placement:
    dimensions = item.dimensions
    origin = Point(ticks(x), ticks(y), ticks(z))
    return Placement(ItemInstance(item, index), origin, Rotation.LWH, dimensions, origin, dimensions)


def request(items: list[dict], containers: list[dict], **configuration) -> dict:
    return {"items": items,
            "containers": containers,
            "configuration": {"solver_profile": "fast", "time_limit_ms": 60000,
                              "effort_budget": {"max_search_nodes": 100000}, **configuration}}


def books(quantity: int) -> dict:
    return {"id": "book", "quantity": quantity, "weight": "300 g", "allowed_rotations": ["LWH"],
            "dimensions": {"length": "200", "width": "100", "height": "20"}}


def mailer(length: str = "240", width: str = "140", height: str = "100", **fields) -> dict:
    return {"id": "mailer", "inner_dimensions": {"length": length, "width": width, "height": height},
            **BAG, **fields}


def extent_mm(container: dict) -> tuple[int, ...]:
    return tuple(int(container["achieved_dimensions"][side]["value"])
                 for side in ("length", "width", "height"))


class TestTheRule:
    def test_the_worked_examples_of_the_oracle(self):
        inner = mm(240, 140, 100)
        assert girth_admits(inner, ticks(200), ticks(100), ticks(40))
        assert not girth_admits(inner, ticks(200), ticks(100), ticks(40) + 1)
        assert not girth_admits(inner, ticks(200) + ticks(40) + 1, 0, 0)
        # A rod as long as the bag is flat passes the frame and fails the length girth.
        assert not girth_admits(mm(1000, 480, 300), ticks(1000), ticks(10), ticks(10))

    def test_the_running_extent_equals_the_extent_from_scratch(self):
        generator = random.Random(2602)
        for _ in range(200):
            boxes = [tuple(generator.randrange(1, 10_000) for _ in range(3))
                     for _ in range(generator.randrange(1, 12))]
            running = (0, 0, 0)
            verdicts = []
            for box in boxes:
                running = grown_extent(running, *box)
                verdicts.append(girth_admits(mm(5000, 5000, 5000), *running))
            assert running == extent_of(boxes)
            # Extents only grow, so a refused prefix is never admitted again.
            assert verdicts == sorted(verdicts, reverse=True)

    def test_a_single_box_fits_only_when_both_girths_hold(self):
        inner = mm(300, 300, 100)
        assert not fits_alone(inner, mm(250, 150, 80))
        assert fits_alone(inner, mm(220, 150, 80))
        assert not fits_alone(inner, mm(310, 10, 10))
        assert as_dimensions((1, 2, 3)) == Dimensions(Length(1), Length(2), Length(3))


class TestTheModel:
    @pytest.mark.parametrize("fields, refused", [
        ({"axles": None, "obstacles": ()}, None),
        ({"outer_dimensions": mm(250, 150, 110)}, "outer_dimensions"),
        ({"max_stack_density": Weight(1)}, "max_stack_density"),
        ({"access_directions": ("+x",)}, "access_directions"),
        ({"void_fill_reserve_ratio": 0.1}, "void_fill_reserve_ratio"),
    ])
    def test_a_bag_refuses_every_wall_anchored_field(self, fields, refused):
        if refused is None:
            assert bag(**fields).is_flat_bag
            return
        with pytest.raises(ValueError, match=f"{refused} cannot be combined with deformable_boundary"):
            bag(**fields)

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="unknown deformable_boundary kind"):
            Container("pouch", mm(1, 1, 1), deformable_boundary="gusseted")

    def test_a_bag_inherits_no_doors(self):
        constraint = StopAccessibilityConstraint(("+x",))
        assert constraint.directions_for(bag()) == ()
        assert constraint.directions_for(Container("box", mm(1, 1, 1))) == ("+x",)


class TestTheRequest:
    @pytest.mark.parametrize("fields, field, reason, detail", [
        ({"deformable_boundary": "flat_bag"}, "/deformable_boundary", "wrong_type", "must be an object"),
        ({"deformable_boundary": {}}, "/deformable_boundary/kind", "missing_field", "is required"),
        ({"deformable_boundary": {"kind": "pouch"}}, "/deformable_boundary/kind", "not_allowed",
         'must be one of ["flat_bag"]'),
        ({**BAG, "deformable_boundary": {"kind": "flat_bag", "seam": 1}}, "/deformable_boundary/seam",
         "not_allowed", "is not a known field"),
        ({**BAG, "made_to_measure": {}}, "/made_to_measure", "not_allowed",
         "cannot be combined with deformable_boundary"),
        ({**BAG, "void_fill_reserve_ratio": 0.5}, "/void_fill_reserve_ratio", "not_allowed",
         "cannot be combined with deformable_boundary"),
    ])
    def test_a_malformed_bag_is_refused_with_its_pointer(self, fields, field, reason, detail):
        container = {"id": "mailer", "inner_dimensions": {"length": "1", "width": "1", "height": "1"},
                     **fields}
        with pytest.raises(InvalidRequestError) as caught:
            pack_from_dict(request([books(1)], [container]))
        assert (caught.value.field, caught.value.reason, caught.value.detail) == (
            "/containers/0" + field, reason, detail)


class TestThePacking:
    def test_girth_not_the_frame_decides_how_high_a_bag_stacks(self):
        result = pack_from_dict(request([books(3)], [mailer(quantity=1)]))
        assert [extent_mm(c) for c in result["containers"]] == [(200, 100, 40)]
        assert [u["reason"] for u in result["unpacked_items"]] == ["container_inventory_exhausted"]

    def test_girth_opens_a_second_bag(self):
        result = pack_from_dict(request([books(3)], [mailer()]))
        assert sorted(extent_mm(c) for c in result["containers"]) == [(200, 100, 20), (200, 100, 40)]
        rigid = pack_from_dict(request([books(3)], [{**mailer(), "deformable_boundary": None}]))
        assert len(rigid["containers"]) == 1 and "achieved_dimensions" not in rigid["containers"][0]

    def test_an_item_no_rotation_admits_is_proven_unplaceable(self):
        carton = {"id": "carton", "dimensions": {"length": "250", "width": "150", "height": "80"}}
        result = pack_from_dict(request([carton], [mailer("300", "300", "100")]))
        assert result["unpacked_items"][0]["reason"] == "no_compatible_container_dimensions"
        assert result["unpacked_items"][0]["proof"]["level"] == "proven"
        upright = {**carton, "allowed_rotations": ["LWH"]}
        assert pack_from_dict(request([upright], [mailer("330", "300", "100")]))["containers"]
        assert pack_from_dict(request([{**carton, "allowed_rotations": ["LHW"]}],
                                      [mailer("330", "300", "100")])
                              )["unpacked_items"][0]["reason"] == "rotation_restricted"

    def test_the_lattice_and_block_solvers_defer_to_the_candidate_check(self):
        items = [ItemInstance(book(), index) for index in range(1, 4)]
        config = PackingConfig(profile=SolverProfile.FAST)
        solution = GridSolver().pack_one(bag(), 1, items, config, SearchStats(), Deadline(60_000))
        assert not solution.dominant_lattice
        assert solution.state.lattice_summary is None
        assert len(solution.state.placements) == 2
        assert not HomogeneousBlockSolver()._supports(bag(), items, config)

    def test_a_candidate_one_tick_past_either_girth_is_never_proposed(self):
        state = ContainerState(bag(), 1)
        state.add(placed(book(), 1, 0, 0, 0))
        config = PackingConfig()
        found = find_candidates(state, ItemInstance(book(), 2), config, default_constraints(config),
                                SearchStats(), Deadline(60_000))
        assert {(c.point.x, c.point.z) for c in found} == {(0, ticks(20))}
        state.add(placed(book(), 2, 0, 0, 20))
        assert find_candidates(state, ItemInstance(book(), 3), config, default_constraints(config),
                               SearchStats(), Deadline(60_000)) == []


class TestBilling:
    def landed(self) -> PackingConfig:
        return PackingConfig(objective="lowest_landed_cost", dimensional_weight_divisor=5000,
                             dimensional_weight_length_unit="cm", dimensional_weight_weight_unit="kg")

    def test_a_bag_is_billed_on_its_achieved_package(self):
        table = RateTable((500, 2000), (400, 1100))
        packed = PackedContainer(bag(300, 250, 100, rate_table=table), 1, (placed(book(), 1, 0, 0, 0),))
        assert packed.billed_dimensions == mm(200, 100, 20)
        assert LandedCostSolutionScorer.from_config(self.landed()).score_containers([packed], [])[1] == 400
        assert ShippingCostSolutionScorer.from_config(self.landed()).score_containers([packed], [])[1] == Weight.parse("80 g").ticks
        rigid = PackedContainer(Container("box", mm(300, 250, 100)), 1, (placed(book(), 1, 0, 0, 0),))
        assert rigid.billed_dimensions == mm(300, 250, 100) and rigid.achieved_dimensions is None

    def test_an_empty_bag_bills_no_dimensional_weight(self):
        state = ContainerState(bag(), 1)
        assert _billed_dimensional_weight(state, self.landed()) == 0
        state.add(placed(book(), 1, 0, 0, 0))
        # 20 x 10 x 2 cm = 400 cm3 over 5000 is 0.08 kg.
        assert _billed_dimensional_weight(state, self.landed()) == Weight.parse("80 g").ticks

    def test_landed_cost_prefers_the_bag_its_achieved_size_makes_cheap(self):
        rate = {"weight_brackets_g": [500, 1000, 2000], "prices_minor": [400, 700, 1100]}
        carton = {"id": "carton", "inner_dimensions": {"length": "300", "width": "200", "height": "150"},
                  "rate_table": rate}
        result = pack_from_dict(request(
            [books(1)], [carton, mailer("300", "250", "100", rate_table=rate)],
            objective="lowest_landed_cost", dimensional_weight_divisor=5000,
            dimensional_weight_length_unit="cm", dimensional_weight_weight_unit="kg"))
        assert [c["container_type"] for c in result["containers"]] == ["mailer"]
        assert result["score"][1] == 400

    def test_exact_small_ranks_a_bag_by_its_achieved_package(self):
        rate = {"weight_brackets_g": [500, 2000], "prices_minor": [400, 1100]}
        result = pack_from_dict(request(
            [books(1)], [mailer("300", "250", "100", rate_table=rate)], solver_profile="exact_small",
            objective="lowest_landed_cost", dimensional_weight_divisor=5000,
            dimensional_weight_length_unit="cm", dimensional_weight_weight_unit="kg"))
        assert result["score"][1] == 400

    def test_a_nested_bag_is_an_item_of_its_achieved_size(self):
        packed = PackedContainer(bag(), 1, (placed(book(), 1, 0, 0, 0), placed(book(), 2, 0, 0, 20)))
        assert packed.as_item().dimensions == mm(200, 100, 40)


class TestValidationAndFixedPlacements:
    def test_the_validator_reports_a_bag_past_its_girth(self):
        packed = PackedContainer(bag(), 1, tuple(placed(book(3), i, 0, 0, 20 * (i - 1)) for i in (1, 2, 3)))
        issues = IndependentSolutionValidator().validate(
            PackingRequest((book(3),), (bag(),)), (packed,)).issues
        assert [(issue.code, issue.detail) for issue in issues] == [("bag_girth_exceeded", "mailer#1")]

    def test_a_fixed_set_past_its_girth_cannot_hold(self):
        fixed = [{"item_type": "book", "container_type": "mailer", "orientation": "LWH",
                  "position": {"x": "0", "y": "0", "z": str(20 * level)}} for level in range(3)]
        with pytest.raises(FixedPlacementError) as caught:
            pack_from_dict({**request([books(3)], [mailer()]), "fixed_placements": fixed})
        assert str(caught.value) == "invalid_fixed_placement: bag_girth_exceeded: mailer#1"
        assert caught.value.reason == "cannot_hold"

    def test_a_fixed_set_inside_its_girth_is_seeded_and_extended(self):
        fixed = [{"item_type": "book", "container_type": "mailer", "orientation": "LWH",
                  "position": {"x": "0", "y": "0", "z": "0"}}]
        result = pack_from_dict({**request([books(2)], [mailer(quantity=1)]), "fixed_placements": fixed})
        assert [extent_mm(c) for c in result["containers"]] == [(200, 100, 40)]
