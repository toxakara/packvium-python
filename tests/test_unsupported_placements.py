"""The default minimum_support_ratio of 0 lets a placement rest on nothing, and says so."""

from __future__ import annotations

from packvium import Packer, PackingConfig
from packvium.geometry import Dimensions, Point, Rotation
from packvium.models import Container, FixedPlacement, Item
from packvium.result import unsupported_placements_warning
from packvium.serialization import pack_from_dict
from packvium.units import Length

WARNING = (
    "unsupported_placements: {count} placement(s) rest on nothing above the container floor; "
    "set configuration.minimum_support_ratio above 0 to require support"
)


def cube(item_id: str, quantity: int, height: str = "100") -> dict:
    return {"id": item_id, "quantity": quantity, "weight": "500",
            "dimensions": {"length": "100", "width": "100", "height": height}}


def fixed(item_type: str, x: str, y: str, z: str) -> dict:
    return {"item_type": item_type, "container_type": "crate",
            "position": {"x": x, "y": y, "z": z}, "orientation": "LWH"}


def scene(floating: bool = True, configuration: dict | None = None) -> dict:
    """Two stacked bases, plus a slab in mid-air and a slab on an obstacle when floating."""
    placements = [fixed("base", "0", "0", "0"), fixed("base", "0", "0", "100")]
    items = [cube("base", 2), cube("loose", 4)]
    if floating:
        placements += [fixed("slab", "100", "100", "300"), fixed("slab", "200", "0", "100")]
        items.append(cube("slab", 2, height="50"))
    return {
        "items": items,
        "containers": [{
            "id": "crate", "quantity": 1,
            "inner_dimensions": {"length": "400", "width": "200", "height": "400"},
            "obstacles": [{"id": "shelf", "origin": {"x": "200", "y": "0", "z": "0"},
                           "dimensions": {"length": "100", "width": "100", "height": "100"}}],
        }],
        "fixed_placements": placements,
        "configuration": {"solver_profile": "quality", "alternatives": 3,
                          "effort_budget": {"max_search_nodes": 20000}, "time_limit_ms": 60000,
                          **(configuration or {})},
    }


def unsupported_warnings(result: dict) -> list[str]:
    return [line for line in result["warnings"] if line.startswith("unsupported_placements:")]


def test_a_placement_in_mid_air_and_one_on_an_obstacle_are_both_counted():
    result = pack_from_dict(scene())

    assert result["warnings"][-1] == WARNING.format(count=2)
    assert len(unsupported_warnings(result)) == 1


def test_every_alternative_carries_its_own_warning():
    result = pack_from_dict(scene())

    assert result["alternatives"]
    for alternative in result["alternatives"]:
        assert unsupported_warnings(alternative) == [WARNING.format(count=2)]


def test_a_fully_supported_packing_has_no_warning():
    result = pack_from_dict(scene(floating=False))

    assert unsupported_warnings(result) == []


def test_requiring_support_silences_the_warning():
    result = pack_from_dict(scene(floating=False, configuration={"minimum_support_ratio": 0.5}))

    assert result["warnings"] == []


def test_a_nonzero_ratio_never_counts_even_an_unsupported_placement():
    lifted = Point(0, 0, Length.mm(100).ticks)
    items = [Item.create("cube", Dimensions.mm(100, 100, 100), weight="1000g", quantity=1)]
    containers = [Container.create("box", Dimensions.mm(100, 100, 300), quantity=1)]
    packed = Packer(PackingConfig()).pack(
        items, containers, [FixedPlacement("cube", "box", lifted, Rotation.LWH, 1)]
    ).containers

    assert unsupported_placements_warning(packed, 0.0) == WARNING.format(count=1)
    assert unsupported_placements_warning(packed, 0.25) is None


def test_no_containers_means_no_warning():
    assert unsupported_placements_warning((), 0.0) is None
