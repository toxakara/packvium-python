"""Stocking many approved boxes must not rescan the artifact for every lock."""
from packvium.revisions import stock_cut_boxes


class CountedOrder(dict):
    reads = 0

    def get(self, key, *args):
        if key == "container_type":
            type(self).reads += 1
        return super().get(key, *args)


def test_stocking_indexes_artifact_rows_once_and_keeps_first_lock_order():
    count = 100
    request = {"containers": [{"id": "machine", "quantity": count, "made_to_measure": {}}],
               "fixed_placements": [{"container_type": "machine", "container_instance": instance}
                                    for instance in range(count, 0, -1)]}
    artifact = {"work_order": {"containers": [CountedOrder(container_type="machine") for _ in range(count)]},
                "geometry": {"containers": [{"inner_dimensions": {"length": str(instance), "width": "2", "height": "3"}}
                                             for instance in range(1, count + 1)]}}
    CountedOrder.reads = 0
    stocked = stock_cut_boxes(request, artifact, 7)
    assert CountedOrder.reads <= count * 2
    assert [box["id"] for box in stocked["containers"]] == [f"machine@7#{i}" for i in range(count, 0, -1)]
    assert stocked["containers"][0]["inner_dimensions"]["length"]["value"] == "100"
    assert stocked["containers"][-1]["inner_dimensions"]["length"]["value"] == "1"


def test_stocking_joins_by_type_and_does_not_parse_unselected_geometry():
    request = {"containers": [{"id": "machine", "made_to_measure": {}, "cost_minor": 1},
                               {"id": "machine", "made_to_measure": {}, "cost_minor": 9}],
               "fixed_placements": [{"container_type": "machine", "container_instance": 2}]}
    artifact = {"work_order": {"containers": [{"container_type": "machine"}, {"container_type": "other"},
                                               {"container_type": "machine"}, {"container_type": "machine"}]},
                "geometry": {"containers": [None, None, {"inner_dimensions": {"length": "1", "width": "2", "height": "3"}}]}}
    stocked = stock_cut_boxes(request, artifact, 1)
    assert stocked["containers"][1]["cost_minor"] == 9
    assert stocked["containers"][1]["inner_dimensions"]["length"]["value"] == "1"
