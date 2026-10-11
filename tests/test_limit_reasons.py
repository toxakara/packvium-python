"""One reason code per container, axle and stop limit, and: a
restart budget that changes the plan.

An axle that refuses an item alone in every container otherwise admitting it is
`axle_overloaded`; every other limit only refuses an item a fresh container would take, so
once stock or `max_containers` runs out the item is `container_inventory_exhausted`. The
same scenes are pinned across all four engines in `conformance/unpacked-reasons/`.
"""

from __future__ import annotations

import pytest

from packvium.serialization import pack_from_dict

CONFIGURATION = {"solver_profile": "fast", "time_limit_ms": 300000}


def cube(item_id: str, quantity: int, side: str = "10", **fields) -> dict:
    return {"id": item_id, "quantity": quantity,
            "dimensions": {"length": side, "width": side, "height": side}, **fields}


def box(length: str, width: str, height: str, **fields) -> dict:
    return {"id": "box", "inner_dimensions": {"length": length, "width": width, "height": height},
            **fields}


def grams(value: str) -> dict:
    return {"value": value, "unit": "g"}


def reasons(items: list[dict], containers: list[dict], **configuration) -> list[tuple[str, str]]:
    result = pack_from_dict({"units": {"length": "mm"},
                             "configuration": {**CONFIGURATION, **configuration},
                             "items": items, "containers": containers})
    return sorted((u["reason"], u["proof"]["level"]) for u in result["unpacked_items"])


INVENTORY = [("container_inventory_exhausted", "inferred")]


@pytest.mark.parametrize("items, containers, configuration", [
    ([cube("cube", 2)], [box("10", "10", "10", quantity=1)], {}),
    ([cube("cube", 2)], [box("10", "10", "10")], {"max_containers": 1}),
    ([cube("cube", 2)], [box("20", "20", "20", quantity=1, max_items=1)], {}),
    ([cube("cube", 2, weight=grams("600"))],
     [box("20", "20", "20", quantity=1, max_payload=grams("1000"))], {}),
    ([cube("hazmat", 2, tags=["hazmat"])],
     [box("20", "20", "20", quantity=1, tag_limits={"hazmat": 1})], {}),
    ([cube("early", 1, side="100", stop_index=0),
      cube("late", 1, side="100", stop_index=1, stackable=False)],
     [box("100", "100", "200", quantity=1, access_directions=["+z"])], {}),
    ([cube("payload", 2, weight=grams("500"), allowed_rotations=["LWH"])],
     [box("100", "10", "10", quantity=1, axles=[{"position": "0", "max_load": grams("600")},
                                                {"position": "100", "max_load": grams("350")}])],
     {}),
], ids=["quantity", "max_containers", "max_items", "max_payload", "tag_limits", "stop_access",
        "axle_share"])
def test_a_limit_a_fresh_container_would_lift_is_the_inventory(items, containers, configuration):
    assert reasons(items, containers, **configuration) == INVENTORY


def test_the_same_limits_with_stock_left_open_another_container():
    assert reasons([cube("cube", 2)], [box("20", "20", "20", max_items=1)]) == []


@pytest.mark.parametrize("length, front, rear", [
    ("10", "400", "400"),   # 1000 g over 800 g of axle capacity
    ("100", "400", "700"),  # enough capacity, but its one centre overloads the front axle
])
def test_an_axle_that_refuses_the_item_alone_is_named(length, front, rear):
    item = {"id": "load", "quantity": 1, "weight": grams("1000"), "allowed_rotations": ["LWH"],
            "dimensions": {"length": length, "width": "10", "height": "10"}}
    vehicle = box("100", "10", "10", axles=[{"position": "0", "max_load": grams(front)},
                                            {"position": "100", "max_load": grams(rear)}])
    assert reasons([item], [vehicle]) == [("axle_overloaded", "inferred")]


def test_a_container_without_axles_that_takes_the_item_keeps_the_axle_out_of_it():
    item = cube("load", 2, weight=grams("1000"), allowed_rotations=["LWH"])
    vehicle = box("100", "10", "10", axles=[{"position": "0", "max_load": grams("400")},
                                            {"position": "100", "max_load": grams("400")}])
    crate = {**box("10", "10", "10", quantity=1), "id": "crate"}
    assert reasons([item], [vehicle, crate]) == INVENTORY


@pytest.mark.parametrize("restarts, unpacked", [(1, 4), (None, 3)])
def test_a_single_restart_packs_worse_than_the_full_portfolio(restarts, unpacked):
    """The shared conformance scene feature-effort-restarts-first-start-loses."""
    budget = {"max_candidates_evaluated": 1000000, "max_placement_attempts": 1000000,
              "max_search_nodes": 1000000}
    if restarts is not None:
        budget["max_restarts"] = restarts
    items = [{"id": "panel", "quantity": 4, "dimensions": {"length": "20", "width": "25", "height": "50"}},
             {"id": "crate", "quantity": 4, "dimensions": {"length": "60", "width": "40", "height": "40"}}]
    assert len(reasons(items, [box("50", "100", "60", quantity=1)], solver_profile="balanced",
                       effort_budget=budget)) == unpacked

