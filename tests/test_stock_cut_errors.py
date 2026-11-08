import copy

import pytest

from packvium.artifacts import build_operational_artifact
from packvium.revisions import PlanRevisionError, derive_revision, root_revision, stock_cut_boxes
from packvium.serialization import pack_from_dict


def approved():
    sides = {"length": "10", "width": "10", "height": "10"}
    request = {"items": [{"id": "cube", "dimensions": sides}], "containers": [
        {"id": "machine", "inner_dimensions": sides, "made_to_measure": {"minimum": sides, "increment": sides}}]}
    return request, build_operational_artifact(request, pack_from_dict(request))


@pytest.mark.parametrize("field,value,detail", [
    ("made_to_measure", True, "made_to_measure is not an object"),
    ("cost_minor", "price", "cost_minor is not an integer"),
    ("tare_weight", "heavy", "tare_weight is not a measure"),
])
def test_a_replacement_and_lock_refuse_unreadable_stock_fields(field, value, detail):
    request, artifact = approved()
    replacement = copy.deepcopy(request["containers"][0])
    replacement[field] = value
    events = [{"sequence": 1, "type": "container_substituted", "container_type": "machine", "replacement": replacement},
              {"sequence": 2, "type": "placement_locked", "placement": {"item_type": "cube", "container_type": "machine", "orientation": "LWH"}}]
    with pytest.raises(PlanRevisionError) as refusal:
        derive_revision(root_revision(request), artifact, events)
    assert (refusal.value.code, str(refusal.value)) == ("invalid_revision", "a made-to-measure container's " + detail)


def test_stocking_still_defers_fixed_placement_physics():
    request, artifact = approved()
    event = {"sequence": 1, "type": "placement_locked", "placement": {"item_type": "cube", "container_type": "machine",
              "orientation": "LWH", "position": {"x": "999999"}}}
    revision = derive_revision(root_revision(request), artifact, [event])
    assert revision["request"]["fixed_placements"][0]["position"] == {"x": "999999"}


@pytest.mark.parametrize("units,detail", [([], "units is not an object"), ({"length": []}, "units.length is not a string")])
def test_stocking_refuses_an_unreadable_measure_context(units, detail):
    request, artifact = approved()
    request["units"] = units
    request["fixed_placements"] = [{"container_type": "machine"}]
    with pytest.raises(PlanRevisionError) as refusal:
        stock_cut_boxes(request, artifact, 1)
    assert str(refusal.value) == "a made-to-measure container's " + detail


@pytest.mark.parametrize("cost", [None, 0, 1, 1.0, -1])
def test_stock_costs_keep_exact_integers_and_null_defaults(cost):
    request, artifact = approved()
    request["containers"][0]["cost_minor"] = cost
    request["fixed_placements"] = [{"container_type": "machine"}]
    assert stock_cut_boxes(request, artifact, 1)["containers"][1]["cost_minor"] == (0 if cost is None else cost)
