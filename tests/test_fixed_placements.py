"""Fixed placements: items already in a known place before the solve (docs/PLAN-REVISIONS.md)."""

from __future__ import annotations

import copy

import pytest

from packvium import Packer, PackingConfig
from packvium.fixed_placements import FixedPlacementError
from packvium.geometry import Dimensions, Point, Rotation
from packvium.models import Container, FixedPlacement, Item, PackingRequest
from packvium.units import Length
from packvium.rebalance import rebalance_weight
from packvium.serialization import pack_from_dict
from packvium.validation import IndependentSolutionValidator


def cube(quantity: int, weight: str = "1000", **extra) -> dict:
    return {"id": "cube", "quantity": quantity, "weight": weight,
            "dimensions": {"length": "100", "width": "100", "height": "100"}, **extra}


def box(quantity: int = 3, **extra) -> dict:
    return {"id": "box", "quantity": quantity,
            "inner_dimensions": {"length": "200", "width": "100", "height": "200"}, **extra}


def fixed(x: str = "0", y: str = "0", z: str = "0", instance: int = 1, **extra) -> dict:
    return {"item_type": "cube", "container_type": "box", "container_instance": instance,
            "position": {"x": x, "y": y, "z": z}, "orientation": "LWH", **extra}


def request(items=None, containers=None, placements=None, configuration=None) -> dict:
    data = {"items": items or [cube(5)], "containers": containers or [box()],
            "fixed_placements": placements if placements is not None else [fixed("100")]}
    if configuration is not None:
        data["configuration"] = configuration
    return data


def fixed_rows(result: dict) -> list[tuple]:
    return [
        (container["id"], placement["item_id"], placement["position"]["x"]["value"],
         placement["position"]["y"]["value"], placement["position"]["z"]["value"])
        for container in result["containers"] for placement in container["placements"]
        if placement.get("fixed")
    ]


def domain_request(quantity: int = 5, weight: str = "1000g", boxes: int = 3):
    items = [Item.create("cube", Dimensions.mm(100, 100, 100), weight=weight, quantity=quantity)]
    containers = [Container.create("box", Dimensions.mm(200, 100, 200), quantity=boxes)]
    return items, containers


def at(x_mm: int, y_mm: int, z_mm: int, instance: int = 1) -> FixedPlacement:
    origin = Point(Length.mm(x_mm).ticks, Length.mm(y_mm).ticks, Length.mm(z_mm).ticks)
    return FixedPlacement("cube", "box", origin, Rotation.LWH, instance)


def without_durations(value):
    """A result without its wall-clock fields, the only part a clock may change."""
    if isinstance(value, dict):
        return {key: without_durations(item) for key, item in value.items() if key != "duration_ms"}
    if isinstance(value, list):
        return [without_durations(item) for item in value]
    return value


def refusal(data: dict) -> str:
    with pytest.raises(FixedPlacementError) as caught:
        pack_from_dict(data)
    return str(caught.value)


@pytest.mark.parametrize("profile", ["fast", "balanced", "quality", "exact_small"])
def test_a_fixed_item_is_reported_where_the_request_put_it(profile):
    result = pack_from_dict(request(configuration={"solver_profile": profile}))
    assert result["status"] in {"feasible", "optimal"}
    assert fixed_rows(result) == [("box#1", "cube#1", "100", "0", "0")]
    assert sum(len(c["placements"]) for c in result["containers"]) == 5


def test_only_fixed_placements_carry_the_flag():
    result = pack_from_dict(request())
    flags = [p.get("fixed") for c in result["containers"] for p in c["placements"]]
    assert flags.count(True) == 1
    assert flags.count(None) == 4


def test_a_request_without_fixed_placements_answers_as_before():
    data = request()
    del data["fixed_placements"]
    empty = copy.deepcopy(data)
    empty["fixed_placements"] = []
    assert without_durations(pack_from_dict(data)) == without_durations(pack_from_dict(empty))


def test_fixed_items_take_the_first_instances_of_their_type_in_listed_order():
    result = pack_from_dict(request(placements=[fixed("100"), fixed("0")]))
    assert fixed_rows(result) == [("box#1", "cube#1", "100", "0", "0"),
                                  ("box#1", "cube#2", "0", "0", "0")]


def test_fixed_containers_open_first_and_free_ones_are_numbered_after_them():
    result = pack_from_dict(request(items=[cube(9)], placements=[fixed(), fixed(instance=2)]))
    ids = [container["id"] for container in result["containers"]]
    assert ids[:2] == ["box#1", "box#2"]
    assert ids[2:] == ["box#3"]


def test_a_fixed_container_is_kept_when_no_free_item_fits_in_it():
    small = {"id": "tray", "quantity": 1,
             "inner_dimensions": {"length": "100", "width": "100", "height": "100"}}
    data = request(items=[cube(3)], containers=[small, box()],
                   placements=[{**fixed(), "container_type": "tray"}])
    result = pack_from_dict(data)
    assert result["containers"][0]["id"] == "tray#1"
    assert [p["item_id"] for p in result["containers"][0]["placements"]] == ["cube#1"]


def test_a_fixed_container_survives_a_search_that_runs_out_of_effort():
    data = request(items=[cube(40)], containers=[box(quantity=20)],
                   placements=[fixed(), fixed(instance=2)],
                   configuration={"effort_budget": {"max_search_nodes": 1}})
    result = pack_from_dict(data)
    assert {row[0] for row in fixed_rows(result)} == {"box#1", "box#2"}
    assert not result["warnings"]


def test_a_free_item_resting_on_a_fixed_one_loads_it():
    data = request(items=[cube(2)], containers=[box(quantity=1)],
                   placements=[fixed()],
                   configuration={"solvers": ["extreme_points"]})
    data["containers"][0]["inner_dimensions"]["length"] = "100"
    result = pack_from_dict(data)
    (container,) = result["containers"]
    by_id = {p["item_id"]: p for p in container["placements"]}
    assert by_id["cube#2"]["position"]["z"]["value"] == "100"
    assert by_id["cube#1"]["top_load"]["value"] == "1000"


def test_fixed_weight_counts_against_payload():
    data = request(items=[cube(3)], containers=[box(quantity=2, max_payload="1000")],
                   placements=[fixed()])
    result = pack_from_dict(data)
    assert [len(c["placements"]) for c in result["containers"]] == [1, 1]
    assert [item["item_id"] for item in result["unpacked_items"]] in (["cube#2"], ["cube#3"])


@pytest.mark.parametrize("solver", ["grid", "homogeneous_blocks", "maximal_spaces", "layer"])
def test_every_solver_packs_around_fixed_items(solver):
    result = pack_from_dict(request(configuration={"solvers": [solver]}))
    assert fixed_rows(result) == [("box#1", "cube#1", "100", "0", "0")]
    assert not result["warnings"]
    assert sum(len(c["placements"]) for c in result["containers"]) == 5


def test_the_lattice_still_serves_containers_without_fixed_items():
    data = request(items=[cube(12)], containers=[box(quantity=4)],
                   configuration={"solver_profile": "fast",
                                  "require_placement_coordinates": False})
    result = pack_from_dict(data)
    assert result["containers"][0]["placements"][0].get("fixed") is True
    assert any("lattice_summary" in container for container in result["containers"][1:])
    assert not result["warnings"]


def test_the_concurrent_portfolio_keeps_fixed_items():
    items, containers = domain_request(quantity=6)
    fixed_items = [at(100, 0, 0)]
    serial = Packer(PackingConfig.balanced()).pack(items, containers, fixed_items)
    concurrent = Packer(PackingConfig.balanced(parallel_starts=2)).pack(items, containers, fixed_items)
    assert not concurrent.warnings
    assert fixed_rows(concurrent.to_dict()) == fixed_rows(serial.to_dict())


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["fixed_placements"].append(fixed("100")), "collision"),
    (lambda d: d["fixed_placements"][0].update(item_type="crate"), "unknown item type"),
    (lambda d: d["fixed_placements"][0].update(container_type="crate"), "unknown container type"),
    (lambda d: d["fixed_placements"][0].update(container_instance=2), "[2] are not numbered 1..1"),
    (lambda d: d["fixed_placements"][0]["position"].update(x="150"), "outside_container"),
    (lambda d: d["items"][0].update(quantity=1) or d["fixed_placements"].append(fixed()),
     "2 cube fixed, 1 requested"),
    (lambda d: d["items"][0].update(keep_upright=True)
     or d["fixed_placements"][0].update(orientation="HWL"), "orientation HWL"),
    (lambda d: d["containers"][0].update(obstacles=[{
        "id": "pillar", "origin": {"x": "150"},
        "dimensions": {"length": "10", "width": "10", "height": "10"}}]), "obstacle_collision"),
    (lambda d: d["containers"][0].update(max_payload="500"), "payload_exceeded"),
    (lambda d: d.update(configuration={"clearance": "1"}), "outside_container"),
    (lambda d: d.update(configuration={"minimum_support_ratio": 1})
     or d["fixed_placements"][0]["position"].update(z="50"), "insufficient_support"),
    (lambda d: d["containers"][0].update(quantity=1)
     or d["fixed_placements"].append(fixed(instance=2)), "2 box named, 1 available"),
    (lambda d: d.update(configuration={"max_containers": 1})
     or d["fixed_placements"].append(fixed(instance=2)), "max_containers is 1"),
])
def test_a_fixed_set_that_cannot_hold_is_refused_before_search(mutate, fragment):
    data = request()
    mutate(data)
    message = refusal(data)
    assert message.startswith("invalid_fixed_placement: ")
    assert fragment in message


def test_the_validator_catches_a_moved_fixed_item():
    items, containers = domain_request()
    packed = Packer(PackingConfig()).pack(items, containers, [at(100, 0, 0)])
    moved = PackingRequest(tuple(items), tuple(containers), (at(0, 0, 0),))
    report = IndependentSolutionValidator().validate(moved, packed.containers, 0.0, None,
                                                     packed.unpacked)
    assert {issue.code for issue in report.issues} == {"fixed_placement_moved",
                                                        "unexpected_fixed_placement"}


def test_the_validator_catches_a_missing_fixed_container():
    items, containers = domain_request()
    packed = Packer(PackingConfig()).pack(items, containers)
    wanted = PackingRequest(tuple(items), tuple(containers), (at(0, 0, 0, instance=3),))
    report = IndependentSolutionValidator().validate(wanted, packed.containers)
    assert "fixed_container_missing" in {issue.code for issue in report.issues}


def test_rebalancing_never_moves_a_fixed_item():
    items, containers = domain_request(quantity=5, weight="5000g", boxes=2)
    fixed_items = [at(0, 0, 0), at(100, 0, 0), at(0, 0, 100), at(0, 0, 0, instance=2)]
    config = PackingConfig()
    packed = Packer(config).pack(items, containers, fixed_items)
    model = PackingRequest(tuple(items), tuple(containers), tuple(fixed_items))
    rebalanced = rebalance_weight(model, packed.containers, packed.unpacked, config)
    assert len(packed.containers) == 2
    assert {move.item_id for move in rebalanced.moves}.isdisjoint({"cube#1", "cube#2", "cube#3", "cube#4"})
    report = IndependentSolutionValidator().validate(model, rebalanced.containers, 0.0, None,
                                                     packed.unpacked)
    assert report.valid, report.issues


@pytest.mark.parametrize("placements, detail", [
    ("x", "fixed_placements is a list"),
    ([5], "fixed_placements[0] is an object"),
    ([fixed(note="strapped")], 'fixed_placements[0] cannot carry ["note"]'),
    ([{"item_type": "cube", "container_type": "box"}], 'fixed_placements[0] needs ["orientation"]'),
    ([fixed(item_type="")], "fixed_placements[0].item_type is a non-empty string"),
    ([fixed(container_type=5)], "fixed_placements[0].container_type is a non-empty string"),
    ([fixed(orientation="XYZ")], "fixed_placements[0].orientation is one of the six codes"),
    ([fixed(instance="1")], "fixed_placements[0].container_instance counts from 1"),
    ([fixed(instance=True)], "fixed_placements[0].container_instance counts from 1"),
    ([fixed(instance=1.5)], "fixed_placements[0].container_instance counts from 1"),
    ([fixed(instance=2**53)], "fixed_placements[0].container_instance counts from 1"),
    ([fixed(position=["100", "0", "0"])], "fixed_placements[0].position is a point object"),
    ([fixed(position=None)], "fixed_placements[0].position is a point object"),
    ([fixed(position={"w": "5"})], 'fixed_placements[0].position cannot carry ["w"]'),
    ([fixed(position={"y": True})], "fixed_placements[0].position.y is a measure"),
])
def test_a_fixed_placement_that_is_not_the_schemas_shape_is_refused_not_coerced(placements, detail):
    assert refusal(request(placements=placements)) == f"invalid_fixed_placement: {detail}"


@pytest.mark.parametrize("placements", [None, [fixed("100", instance=1.0)]])
def test_null_fixed_placements_and_an_integral_float_instance_are_accepted(placements):
    data = request(placements=placements)
    if placements is None:
        data["fixed_placements"] = None
    assert pack_from_dict(data)["status"] != "error"


def test_an_unknown_item_type_is_quoted_as_json():
    assert refusal(request(placements=[fixed(item_type="cr\"ate")])) == \
        'invalid_fixed_placement: unknown item type "cr\\"ate"'
