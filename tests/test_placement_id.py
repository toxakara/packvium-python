"""`placement_id` (1.10): `<container id>@<x>,<y>,<z>`, a pure function of a placement.

Emitted on every result placement, recomputed by the validator, restated optionally by a fixed
placement, copied into the execution plan and used there to resolve operator locks.
"""

from __future__ import annotations

import copy
import json

import pytest

from packvium import Packer, PackingConfig, pack_from_dict, validate
from packvium.artifacts import build_operational_artifact
from packvium.execution import build_execution_plan, placement_reference
from packvium.fixed_placements import FixedPlacementError
from packvium.geometry import Dimensions, Point, Rotation
from packvium.locks import PlacementLock, locks_from_plan, resolve_with_locks
from packvium.models import Container, FixedPlacement, Item
from packvium.revisions import PlanRevisionError, derive_revision, root_revision
from packvium.units import Length

MM = 16_000


def dims(length: int, width: int, height: int) -> dict:
    return {"length": str(length), "width": str(width), "height": str(height)}


def request(container_id: str = "box", quantity: int = 4, **extra) -> dict:
    return {
        "configuration": {"solver_profile": "balanced", "time_limit_ms": 300000, "alternatives": 3},
        "items": [{"id": "cube", "quantity": quantity, "dimensions": dims(100, 100, 100)}],
        "containers": [{"id": container_id, "quantity": 3, "inner_dimensions": dims(200, 100, 100)}],
        **extra,
    }


def ident(container_id: str, x: int, y: int, z: int, item_type: str = "cube", orientation: str = "LWH") -> str:
    return json.dumps([container_id, x, y, z, item_type, orientation], ensure_ascii=False, separators=(",", ":"))


def expected_id(container: dict, placement: dict) -> str:
    return ident(container["id"], *(placement["position"][axis]["ticks"] for axis in "xyz"),
                 placement["item_type"], placement["orientation"])


def every_placement(result: dict):
    for document in (result, *result["alternatives"]):
        for container in document["containers"]:
            for placement in container["placements"]:
                yield container, placement


@pytest.mark.parametrize("container_id", ["box", "a@b#c", "böx-箱", 'q"uo\\te\n\u2028'])
def test_every_placement_of_the_answer_and_its_alternatives_carries_its_id(container_id):
    result = pack_from_dict(request(container_id))
    placements = list(every_placement(result))
    assert len(placements) >= 4
    for container, placement in placements:
        assert placement["placement_id"] == expected_id(container, placement)
    assert validate(request(container_id), result) == []


def test_a_tampered_id_is_a_mismatch_naming_both():
    result = pack_from_dict(request())
    placement = result["containers"][0]["placements"][0]
    placement["placement_id"] = ident("box#1", 1, 2, 3)
    issues = validate(request(), result)
    assert [(issue.code, issue.detail) for issue in issues] == [
        ("placement_id_mismatch",
         f"{placement['item_id']} in box#1: reported {ident('box#1', 1, 2, 3)}, "
         f"expected {expected_id(result['containers'][0], placement)}"),
    ]


def test_a_missing_id_is_a_mismatch():
    result = pack_from_dict(request())
    placement = result["containers"][0]["placements"][0]
    del placement["placement_id"]
    issues = validate(request(), result)
    assert [issue.code for issue in issues] == ["placement_id_mismatch"]
    assert 'no placement_id, expected ["box#1",' in issues[0].detail


def test_a_repeated_id_is_a_duplicate():
    result = pack_from_dict(request())
    first, second = result["containers"][0]["placements"][:2]
    second["placement_id"] = first["placement_id"]
    assert [issue.code for issue in validate(request(), result)] == [
        "placement_id_mismatch", "duplicate_placement_id"]


def test_a_lone_surrogate_in_an_id_is_written_as_its_escape():
    result = pack_from_dict(request("b\udc00x"))
    placement = result["containers"][0]["placements"][0]
    assert placement["placement_id"].startswith('["b\\udc00x#1",')
    assert validate(request("b\udc00x"), result) == []


def test_a_lattice_summary_has_no_placements_and_so_no_ids():
    compact = request(quantity=2)
    compact["configuration"] = {"solver_profile": "fast", "require_placement_coordinates": False}
    result = pack_from_dict(compact)
    assert all(container["placements"] == [] for container in result["containers"])
    assert validate(compact, result) == []


# ------------------------------------------------------------------ fixed placements


def fixed_request(**placement) -> dict:
    entry = {"item_type": "cube", "container_type": "box", "position": {"x": "100"},
             "orientation": "LWH", **placement}
    return request(fixed_placements=[entry])


def test_a_fixed_placement_may_quote_the_id_its_result_placement_carries():
    result = pack_from_dict(fixed_request(placement_id=ident("box#1", 100 * MM, 0, 0)))
    fixed = [p for p in result["containers"][0]["placements"] if p.get("fixed")]
    assert [p["placement_id"] for p in fixed] == [ident("box#1", 100 * MM, 0, 0)]


def test_a_null_fixed_id_is_absent():
    assert pack_from_dict(fixed_request(placement_id=None))["status"] == "feasible"


def test_a_fixed_id_naming_another_place_is_refused():
    with pytest.raises(FixedPlacementError) as refusal:
        pack_from_dict(fixed_request(placement_id=ident("box#1", 0, 0, 0)))
    error = refusal.value
    assert (error.code, error.reason, error.field) == (
        "invalid_fixed_placement", "invalid_value", "/fixed_placements/0/placement_id")
    assert str(error) == ("invalid_fixed_placement: fixed_placements[0].placement_id must be "
                          + json.dumps(ident("box#1", 100 * MM, 0, 0)))


def test_a_fixed_id_counts_the_instance_and_defaults_it_to_one():
    with pytest.raises(FixedPlacementError) as refusal:
        pack_from_dict(fixed_request(container_instance=2, position={}, placement_id=ident("box#1", 0, 0, 0)))
    assert refusal.value.detail.endswith(json.dumps(ident("box#2", 0, 0, 0)))


def test_a_fixed_id_past_every_exact_integer_is_left_to_admission():
    """A coordinate past 2^53 - 1 has no canonical spelling and no container holds it, so the
    entry is refused as outside its container, not for its id."""
    with pytest.raises(FixedPlacementError) as refusal:
        pack_from_dict(fixed_request(position={"x": {"value": str(2**53), "unit": "ticks"}},
                                     placement_id=ident("box#1", 0, 0, 0)))
    assert refusal.value.reason == "cannot_hold" and "outside_container" in refusal.value.detail


def test_a_fixed_id_that_is_not_a_string_is_malformed():
    with pytest.raises(FixedPlacementError) as refusal:
        pack_from_dict(fixed_request(placement_id=7))
    assert (refusal.value.reason, refusal.value.field, refusal.value.detail) == (
        "malformed", "/fixed_placements/0/placement_id", "fixed_placements[0].placement_id is a string")


def object_pack(placement_id):
    item = Item("cube", Dimensions(Length.parse("100"), Length.parse("100"), Length.parse("100")), quantity=2)
    container = Container("box", Dimensions(Length.parse("200"), Length.parse("100"), Length.parse("100")))
    fixed = FixedPlacement("cube", "box", Point(100 * MM, 0, 0), Rotation.LWH, placement_id=placement_id)
    return Packer(PackingConfig()).pack([item], [container], [fixed])


def test_the_object_api_holds_a_fixed_id_to_the_same_rule():
    result = object_pack(ident("box#1", 100 * MM, 0, 0))
    assert result.complete
    with pytest.raises(FixedPlacementError) as refusal:
        object_pack(ident("box#1", 100 * MM, 0, 0, "cube", "WLH"))
    assert (refusal.value.reason, refusal.value.field) == ("invalid_value", "/fixed_placements/0/placement_id")
    with pytest.raises(FixedPlacementError, match="is a string"):
        object_pack(5)


# ------------------------------------------------------------------ execution plan and locks


def test_the_plan_copies_each_id_and_a_projection_without_ids_keeps_its_bytes():
    result = pack_from_dict(request())
    plan = build_execution_plan(request(), result)
    references = [step["placement"] for container in plan["containers"] for step in container["steps"]]
    assert [r["placement_id"] for r in references] == [
        p["placement_id"] for c in result["containers"] for p in c["placements"]]
    projection = copy.deepcopy(result)
    for container in projection["containers"]:
        for placement in container["placements"]:
            del placement["placement_id"]
    bare = build_execution_plan(request(), projection)
    assert all("placement_id" not in step["placement"]
               for container in bare["containers"] for step in container["steps"])
    assert placement_reference(0, result["containers"][0]["placements"][0]).keys() == {
        "container_index", "item_type", "orientation", "position_ticks"}


def test_the_artifact_v1_carries_no_id():
    result = pack_from_dict(request())
    artifact = build_operational_artifact(request(), result)
    assert "placement_id" not in repr(artifact)
    projection = copy.deepcopy(result)
    for container in projection["containers"]:
        for placement in container["placements"]:
            del placement["placement_id"]
    assert artifact == build_operational_artifact(request(), projection)


def test_a_plan_with_a_non_string_id_is_refused():
    from packvium.execution import ExecutionPlanError
    result = pack_from_dict(request())
    result["containers"][0]["placements"][0]["placement_id"] = 3
    with pytest.raises(ExecutionPlanError, match="placement_id"):
        build_execution_plan(request(), result)


def test_a_lock_read_from_a_plan_is_resolved_by_its_id_not_the_emitted_index():
    approved = pack_from_dict(request())
    plan = build_execution_plan(request(), approved)
    lock, *_ = locks_from_plan(plan, container_index=0, item_types=["cube"])
    assert lock.placement_id == approved["containers"][0]["placements"][0]["placement_id"]
    # The index is the emitted container order, which the contract does not fix; a lock that
    # names its placement by id is found wherever the re-solve listed that container.
    moved = PlacementLock(7, lock.item_type, lock.orientation, lock.position_ticks, lock.placement_id)
    assert resolve_with_locks(request(), [moved]).preserved


@pytest.mark.parametrize("identifier", [
    '["box#1",1,0,0,"cube","LWH"]', '["box#1",0,0,0,"slab","LWH"]', '["box#1",0,0,0,"cube","WLH"]',
    '["",0,0,0,"cube","LWH"]', '["box#1", 0,0,0,"cube","LWH"]', '{"box#1":0}', "box#1@0,0,0", 5])
def test_a_lock_id_must_be_this_locks_own(identifier):
    with pytest.raises(ValueError, match="not the placement id of this lock"):
        PlacementLock(0, "cube", "LWH", (0, 0, 0), identifier)
    assert PlacementLock(0, "cube", "LWH", (0, 0, 0), ident("box#1", 0, 0, 0)).placement_id


# ------------------------------------------------------------------ revisions


def test_a_lock_event_may_carry_the_id_and_is_one_box_with_the_same_entry_without_it():
    base = request()
    root = root_revision(base)
    artifact = build_operational_artifact(base, pack_from_dict(copy.deepcopy(base)))
    entry = {"item_type": "cube", "container_type": "box", "position": {"x": "0", "y": "0", "z": "0"},
             "orientation": "LWH"}
    quoted = {**entry, "placement_id": ident("box#1", 0, 0, 0)}
    events = [{"sequence": 1, "type": "placement_locked", "placement": quoted},
              {"sequence": 2, "type": "placement_verified", "placement": entry}]
    revision = derive_revision(root, artifact, events)
    assert revision["request"]["fixed_placements"] == [quoted]
    with pytest.raises(PlanRevisionError, match="placement.placement_id is a string"):
        derive_revision(root, artifact, [{"sequence": 1, "type": "placement_locked",
                                          "placement": {**entry, "placement_id": 1}}])
