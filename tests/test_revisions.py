"""Plan revisions: an append-only, hash-chained record of exceptions (docs/PLAN-REVISIONS.md)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from packvium.artifacts import SUITE_VERSION, build_operational_artifact
from packvium.revisions import (PlanRevisionError, apply_events, canonical_revision_json,
                                derive_revision, document_digest, root_revision,
                                verify_revision_chain)
from packvium.serialization import pack_from_dict

SCHEMAS = Path(__file__).resolve().parents[2] / "conformance" / "schema"


def cube_request() -> dict:
    return {
        "items": [
            {"id": "cube", "quantity": 4, "weight": "1000",
             "dimensions": {"length": "100", "width": "100", "height": "100"}},
            {"id": "slab", "quantity": 1,
             "dimensions": {"length": "200", "width": "100", "height": "50"}},
        ],
        "containers": [
            {"id": "box", "quantity": 3,
             "inner_dimensions": {"length": "200", "width": "100", "height": "200"}},
            {"id": "crate", "inner_dimensions": {"length": "300", "width": "200", "height": "200"}},
        ],
        "configuration": {"solver_profile": "balanced", "minimum_support_ratio": 1.0},
    }


def artifact_for(request: dict) -> dict:
    return build_operational_artifact(request, pack_from_dict(request))


def locked(x: str = "0", z: str = "0", kind: str = "placement_locked", sequence: int = 1) -> dict:
    return {"sequence": sequence, "type": kind, "placement": {
        "item_type": "cube", "container_type": "box",
        "position": {"x": x, "z": z}, "orientation": "LWH"}}


def chain() -> tuple[list[dict], list[dict | None]]:
    root = root_revision(cube_request())
    first_artifact = artifact_for(root["request"])
    first = derive_revision(root, first_artifact, [
        locked(sequence=1),
        {"sequence": 2, "type": "item_missing", "item_type": "cube", "quantity": 1},
    ])
    second_artifact = artifact_for(first["request"])
    second = derive_revision(first, second_artifact, [locked(kind="placement_verified", sequence=3)])
    return [root, first, second], [None, first_artifact, second_artifact]


def schema_validator():
    # The schemas and jsonschema live in the workspace; the published package's CI installs
    # only pytest and ships no conformance/ tree.
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    documents = [json.loads((SCHEMAS / name).read_text())
                 for name in ("plan-revision.schema.json", "packing-request.schema.json")]
    registry = referencing.Registry().with_resources(
        [(document["$id"], referencing.Resource.from_contents(document)) for document in documents]
    )
    return jsonschema.Draft202012Validator(documents[0], registry=registry)


@pytest.mark.skipif(not SCHEMAS.is_dir(), reason="the schemas live in the workspace only")
def test_every_revision_in_a_chain_matches_the_schema():
    revisions, _ = chain()
    validator = schema_validator()
    for revision in revisions:
        assert list(validator.iter_errors(revision)) == []


def test_the_root_records_nothing_against_the_request():
    root = root_revision(cube_request())
    assert root == {
        "format": "packvium-plan-revision/v1", "suite_version": SUITE_VERSION, "revision": 0,
        "parent": None, "approved": None, "events": [], "request": cube_request(),
    }


def test_a_revision_links_its_parent_and_approved_artifact_by_digest():
    revisions, artifacts = chain()
    root, first, _ = revisions
    assert first["revision"] == 1
    assert first["parent"] == document_digest(root)
    assert first["approved"]["artifact"] == document_digest(artifacts[1])
    assert first["approved"]["replay"] == artifacts[1]["provenance"]["replay"]


def test_a_digest_is_sha256_over_the_canonical_bytes():
    root = root_revision(cube_request())
    expected = hashlib.sha256(canonical_revision_json(root).encode("utf-8")).hexdigest()
    assert document_digest(root) == f"sha256:{expected}"


def test_transport_spelling_never_changes_a_digest():
    root = root_revision(cube_request())
    reparsed = json.loads(json.dumps(root, indent=4))
    reparsed["request"]["configuration"]["minimum_support_ratio"] = 1
    assert document_digest(reparsed) == document_digest(root)


def test_the_same_inputs_derive_the_same_bytes():
    first, _ = chain()
    second, _ = chain()
    assert [canonical_revision_json(r) for r in first] == [canonical_revision_json(r) for r in second]


def test_a_missing_item_lowers_its_quantity_and_a_last_one_removes_the_type():
    request = cube_request()
    lowered = apply_events(request, [{"sequence": 1, "type": "item_missing", "item_type": "cube", "quantity": 3}])
    assert lowered["items"][0]["quantity"] == 1
    removed = apply_events(request, [{"sequence": 1, "type": "item_missing", "item_type": "slab", "quantity": 1}])
    assert [item["id"] for item in removed["items"]] == ["cube"]
    assert request == cube_request()


def test_a_substituted_container_is_replaced_where_it_stood():
    replacement = {"id": "box-b", "inner_dimensions": {"length": "210", "width": "110", "height": "210"}}
    derived = apply_events(cube_request(), [{"sequence": 1, "type": "container_substituted",
                                            "container_type": "box", "replacement": replacement}])
    assert [c["id"] for c in derived["containers"]] == ["box-b", "crate"]


def test_a_lock_then_a_verification_of_the_same_box_fixes_it_once():
    derived = apply_events(cube_request(), [locked(), locked(kind="placement_verified", sequence=2)])
    assert derived["fixed_placements"] == [locked()["placement"]]


def test_a_fixed_set_with_no_canonical_form_still_deduplicates_by_first_match():
    unholdable = locked(x="5")["placement"]
    unholdable["position"]["x"] = 2 ** 60
    request = {**cube_request(), "fixed_placements": [locked()["placement"], unholdable]}
    derived = apply_events(request, [locked()])
    assert derived["fixed_placements"] == [locked()["placement"], unholdable]


def test_a_lock_with_no_canonical_form_is_recorded_for_admission_to_judge():
    event = locked()
    event["placement"]["position"]["x"] = 2 ** 60
    derived = apply_events(cube_request(), [event])
    assert derived["fixed_placements"] == [event["placement"]]


def test_many_distinct_locks_do_not_repeat_canonical_comparison_for_each_prior_lock(monkeypatch):
    import packvium.revisions as revisions

    events = [locked(x=str(index), sequence=index + 1) for index in range(80)]
    calls = 0
    original = revisions.canonical_revision_json

    def counted(document):
        nonlocal calls
        calls += 1
        return original(document)

    monkeypatch.setattr(revisions, "canonical_revision_json", counted)
    derived = apply_events(cube_request(), events)
    assert len(derived["fixed_placements"]) == len(events)
    assert calls <= 2 * len(events)


def test_a_derived_request_solves_with_its_fixed_items_in_place():
    revisions, _ = chain()
    result = pack_from_dict(revisions[-1]["request"])
    fixed = [(c["id"], p["item_id"]) for c in result["containers"] for p in c["placements"] if p.get("fixed")]
    assert fixed == [("box#1", "cube#1")]
    assert sum(len(c["placements"]) for c in result["containers"]) == 4


@pytest.mark.parametrize("event, fragment", [
    ({"sequence": 1, "type": "item_missing", "item_type": "pallet", "quantity": 1}, 'no item "pallet"'),
    ({"sequence": 1, "type": "container_substituted", "container_type": "box",
      "replacement": {"id": "crate", "inner_dimensions": {"length": 1, "width": 1, "height": 1}}},
     "already crate"),
])
def test_an_event_that_contradicts_the_request_is_refused(event, fragment):
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(cube_request(), [event])
    assert caught.value.code == "event_conflict"
    assert fragment in str(caught.value)


def test_fixed_items_cannot_go_missing_or_lose_their_container():
    fixed = apply_events(cube_request(), [locked()])
    for event in (
        {"sequence": 2, "type": "item_missing", "item_type": "cube", "quantity": 4},
        {"sequence": 2, "type": "container_substituted", "container_type": "box",
         "replacement": {"id": "box-b", "inner_dimensions": {"length": 1, "width": 1, "height": 1}}},
    ):
        with pytest.raises(PlanRevisionError) as caught:
            apply_events(fixed, [event])
        assert caught.value.code == "event_conflict"


def test_the_last_item_type_cannot_go_missing():
    request = cube_request()
    request["items"] = request["items"][:1]
    with pytest.raises(PlanRevisionError, match="no item would remain"):
        apply_events(request, [{"sequence": 1, "type": "item_missing", "item_type": "cube", "quantity": 4}])


@pytest.mark.parametrize("events", [
    [],
    [locked(sequence=2)],
    [{**locked(), "type": "placement_moved"}],
    [{**locked(), "note": "strapped"}],
    [{"sequence": 1, "type": "item_missing", "item_type": "cube"}],
    [{"sequence": 1, "type": "item_missing", "item_type": "cube", "quantity": 0}],
    [{"sequence": True, "type": "item_missing", "item_type": "cube", "quantity": 1}],
    [{"sequence": 1, "type": "placement_locked", "placement": {**locked()["placement"], "orientation": "XYZ"}}],
    [{"sequence": 1, "type": "placement_locked", "placement": {**locked()["placement"], "container_instance": 0}}],
])
def test_a_malformed_event_is_refused(events):
    root = root_revision(cube_request())
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(root, artifact_for(root["request"]), events)
    assert caught.value.code == "invalid_event"


def test_an_artifact_built_from_another_request_is_refused():
    root = root_revision(cube_request())
    other = cube_request()
    other["items"][0]["quantity"] = 3
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(root, artifact_for(other), [locked()])
    assert caught.value.code == "invalid_artifact"


def test_a_parent_that_is_not_a_revision_is_refused():
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(cube_request(), artifact_for(cube_request()), [locked()])
    assert caught.value.code == "invalid_revision"


def test_a_plan_that_stopped_on_time_is_marked_in_the_chain():
    root = root_revision(cube_request())
    artifact = artifact_for(root["request"])
    artifact["provenance"]["solver"]["time_limit_reached"] = True
    artifact["provenance"]["replay"] = {"level": "not_guaranteed",
                                        "because": "provenance.solver.time_limit_reached"}
    revision = derive_revision(root, artifact, [locked()])
    assert revision["approved"]["replay"]["level"] == "not_guaranteed"


def test_an_intact_chain_verifies_clean():
    revisions, artifacts = chain()
    assert verify_revision_chain(revisions, artifacts) == []


def codes(revisions, artifacts=None) -> set[str]:
    return {issue.code for issue in verify_revision_chain(revisions, artifacts)}


def test_reordered_events_are_caught():
    revisions, _ = chain()
    events = revisions[1]["events"]
    events[0], events[1] = events[1], events[0]
    assert "sequence_gap" in codes(revisions)


def test_a_missing_event_is_caught():
    revisions, _ = chain()
    del revisions[1]["events"][1]
    assert {"sequence_gap", "request_mismatch"} <= codes(revisions)


def test_an_altered_event_is_caught():
    revisions, _ = chain()
    revisions[1]["events"][1]["quantity"] = 2
    assert "request_mismatch" in codes(revisions)


def test_an_altered_request_breaks_the_link_to_the_next_revision():
    revisions, _ = chain()
    revisions[1]["request"]["items"][0]["quantity"] = 9
    assert {"request_mismatch", "parent_mismatch"} <= codes(revisions)


def test_a_dropped_revision_is_caught():
    revisions, _ = chain()
    del revisions[1]
    assert {"revision_number", "parent_mismatch", "sequence_gap"} <= codes(revisions)


def test_a_substituted_artifact_is_caught():
    revisions, artifacts = chain()
    swapped = copy.deepcopy(artifacts)
    swapped[2] = artifacts[1]
    assert "artifact_mismatch" in codes(revisions, swapped)


def test_a_chain_that_is_not_made_of_revisions_stops_the_audit():
    revisions, _ = chain()
    revisions[1] = {"format": "something-else"}
    issues = verify_revision_chain(revisions)
    assert [issue.code for issue in issues] == ["invalid_revision"]


@pytest.mark.parametrize("parent", [
    {"format": "packvium-plan-revision/v1", "revision": -1, "request": {}, "events": []},
    {"format": "packvium-plan-revision/v1", "revision": 0, "request": [], "events": []},
    {"format": "packvium-plan-revision/v1", "revision": 0, "request": {}, "events": {}},
    {"format": "packvium-plan-revision/v1", "revision": 2, "request": cube_request(), "events": []},
])
def test_a_malformed_parent_is_refused(parent):
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(parent, artifact_for(cube_request()), [locked()])
    assert caught.value.code == "invalid_revision"


def test_a_root_needs_a_request_object_with_a_canonical_spelling():
    with pytest.raises(PlanRevisionError) as caught:
        root_revision([])
    assert caught.value.code == "invalid_revision"
    with pytest.raises(PlanRevisionError) as caught:
        root_revision({"items": [], "containers": [], "configuration": {"seed": 2**60}})
    assert caught.value.code == "number_out_of_range"


@pytest.mark.parametrize("artifact", [
    {"format": "packvium-execution-plan/v1"},
    {"format": "packvium-operational-artifact/v1", "provenance": {"request": {}}},
])
def test_a_malformed_artifact_is_refused(artifact):
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(root_revision(cube_request()), artifact, [locked()])
    assert caught.value.code == "invalid_artifact"


@pytest.mark.parametrize("event", [
    "item_missing",
    {"sequence": 1, "type": "container_substituted", "container_type": "", "replacement": {"id": "b"}},
    {"sequence": 1, "type": "container_substituted", "container_type": "box", "replacement": "b"},
    {"sequence": 1, "type": "placement_locked", "placement": "box#1"},
    {"sequence": 1, "type": "placement_locked", "placement": {**locked()["placement"], "item_id": "cube#1"}},
    {"sequence": 1, "type": "placement_locked", "placement": {**locked()["placement"], "position": [0, 0, 0]}},
])
def test_a_malformed_event_payload_is_refused(event):
    root = root_revision(cube_request())
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(root, artifact_for(root["request"]), [event])
    assert caught.value.code == "invalid_event"


def test_an_item_without_a_quantity_counts_as_one():
    request = cube_request()
    del request["items"][1]["quantity"]
    derived = apply_events(request, [{"sequence": 1, "type": "item_missing", "item_type": "slab", "quantity": 1}])
    assert [item["id"] for item in derived["items"]] == ["cube"]


def test_a_substitution_needs_the_container_it_replaces():
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(cube_request(), [{"sequence": 1, "type": "container_substituted",
                                       "container_type": "pallet", "replacement": {"id": "p2"}}])
    assert caught.value.code == "event_conflict"


def test_a_request_whose_entries_are_not_objects_is_refused():
    request = cube_request()
    request["items"] = ["cube"]
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(request, [{"sequence": 1, "type": "item_missing", "item_type": "cube", "quantity": 1}])
    assert caught.value.code == "invalid_revision"


def test_a_root_with_events_and_a_revision_without_them_are_caught():
    revisions, _ = chain()
    revisions[0]["events"] = [locked()]
    revisions[2]["events"] = []
    issues = verify_revision_chain(revisions)
    assert [(issue.code, issue.revision) for issue in issues if issue.code == "sequence_gap"][:2] == [
        ("sequence_gap", 0), ("sequence_gap", 1)]
    assert ("sequence_gap", 2) in [(issue.code, issue.revision) for issue in issues]


def test_events_that_no_longer_apply_are_a_request_mismatch():
    revisions, _ = chain()
    revisions[1]["events"][1]["item_type"] = "pallet"
    assert "request_mismatch" in codes(revisions)


def test_an_artifact_for_another_request_or_with_another_replay_is_caught():
    revisions, artifacts = chain()
    tampered = copy.deepcopy(artifacts)
    tampered[1]["provenance"]["replay"] = {"level": "not_guaranteed", "because": "provenance.solver"}
    revisions[1]["approved"]["artifact"] = document_digest(tampered[1])
    assert "artifact_mismatch" in codes(revisions[:2], tampered[:2])
    tampered[1]["provenance"]["request"] = {"items": []}
    revisions[1]["approved"]["artifact"] = document_digest(tampered[1])
    assert "artifact_mismatch" in codes(revisions[:2], tampered[:2])


@pytest.mark.parametrize("sequence", ["1", True, None])
def test_a_parent_whose_events_lost_their_sequence_is_refused(sequence):
    revisions, artifacts = chain()
    parent = copy.deepcopy(revisions[1])
    parent["events"][0]["sequence"] = sequence
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(parent, artifacts[2], [locked(sequence=3)])
    assert (caught.value.code, str(caught.value)) == (
        "invalid_revision", "a revision's events each carry an integer sequence")


def test_an_integral_float_is_the_integer_it_spells():
    revisions, artifacts = chain()
    event = {**locked(), "sequence": 1.0}
    derived = derive_revision(revisions[0], artifacts[1], [event])
    assert derived["events"][0]["sequence"] == 1.0
    assert canonical_revision_json(derived) == canonical_revision_json(
        derive_revision(revisions[0], artifacts[1], [locked()]))


@pytest.mark.parametrize("value", [{"k": 1}, [5], "x"])
def test_a_request_whose_fixed_placements_are_malformed_cannot_be_edited(value):
    request = {**cube_request(), "fixed_placements": value}
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(request, [locked()])
    assert (caught.value.code, str(caught.value)) == (
        "invalid_revision", "request.fixed_placements is a list of objects")


def test_null_fixed_placements_are_none_and_a_lock_creates_the_list():
    request = {**cube_request(), "fixed_placements": None}
    assert len(apply_events(request, [locked()])["fixed_placements"]) == 1


@pytest.mark.parametrize("quantity", ["3", 2.5, None])
def test_an_item_quantity_that_is_not_an_integer_cannot_be_lowered(quantity):
    request = cube_request()
    request["items"][0]["quantity"] = quantity
    event = {"sequence": 1, "type": "item_missing", "item_type": request["items"][0]["id"], "quantity": 1}
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(request, [event])
    assert (caught.value.code, str(caught.value)) == (
        "invalid_revision", "request.items[0].quantity is an integer")


def test_the_audit_refuses_what_is_not_a_chain_or_an_artifact_list():
    revisions, artifacts = chain()
    with pytest.raises(PlanRevisionError) as caught:
        verify_revision_chain({"0": revisions[0]})
    assert caught.value.code == "invalid_revision"
    with pytest.raises(PlanRevisionError) as caught:
        verify_revision_chain(revisions, {"1": artifacts[1]})
    assert caught.value.code == "invalid_artifact"
    assert codes(revisions, [None, "x", *artifacts[2:]]) == {"artifact_mismatch"}
    assert codes(revisions, [None, {"n": 2**60}, *artifacts[2:]]) == {"artifact_mismatch"}


def test_a_boolean_sequence_in_a_chain_is_caught():
    revisions, _ = chain()
    revisions[1]["events"][0]["sequence"] = True
    assert codes(revisions) == {"invalid_revision"}


def test_a_value_is_quoted_by_its_json_spelling():
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(cube_request(), [{**locked(), "type": ["placement_locked"]}])
    assert str(caught.value) == 'unknown event type ["placement_locked"]'
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(cube_request(), [{**locked(), "placement": {**locked()["placement"],
                                                                 "position": {"x": True}}}])
    assert str(caught.value) == "placement.position.x is a measure"


def test_apply_refuses_what_is_not_a_request_or_an_event_list():
    for request, events, code, message in [
        ([1], [], "invalid_revision", "a request is a JSON object"),
        (cube_request(), {"0": locked()}, "invalid_event", "events are a JSON array"),
        (cube_request(), [{k: v for k, v in locked().items() if k != "sequence"}], "invalid_event",
         "event sequence null is not an integer"),
    ]:
        with pytest.raises(PlanRevisionError) as caught:
            apply_events(request, events)
        assert (caught.value.code, str(caught.value)) == (code, message)


def test_a_value_with_no_json_spelling_is_named_by_what_it_is():
    revisions, artifacts = chain()
    with pytest.raises(PlanRevisionError) as caught:
        derive_revision(revisions[0], artifacts[1], [{**locked(), "sequence": 2**53}])
    assert str(caught.value) == "event sequence an out-of-range number does not continue the chain at 1"
    event = {"sequence": 1, "type": "item_missing", "item_type": "\ud800", "quantity": 1}
    with pytest.raises(PlanRevisionError) as caught:
        apply_events(cube_request(), [event])
    assert str(caught.value) == "the request has no item an unspellable value"
