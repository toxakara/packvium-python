"""The staged-rollout guard.

`pack_from_dict` reads the keys it knows and skips the rest, so a public field added to
the schema before this engine implements it would otherwise produce a confident answer
computed as though the caller had never sent it -- indistinguishable, from the outside,
from an engine that honoured the field. Python was the one engine with no such guard at
all; PHP, Rust and the JavaScript fallback have carried theirs since the first staged
rollout.

The lists are injected rather than read from the module constant, because a test against
a list that happens to be empty proves only that nothing is rejected, which is equally
true of a guard that does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packvium.serialization import (
    UNSUPPORTED_FIELDS,
    UNSUPPORTED_SHAPE_TYPES,
    UnsupportedFeatureError,
    pack_from_dict,
    reject_unsupported,
)

ROOT = Path(__file__).resolve().parents[2]

REQUEST = {
    "policy": {"rules": []},
    "configuration": {"tariff": {}},
    "items": [{"id": "a", "rate": 1}, {"id": "b", "rate": 2}],
    "containers": [{"id": "c", "rate_table": {}}],
}
LISTS = {"request": ("policy",), "configuration": ("tariff",), "item": ("rate",), "container": ("rate_table",)}


def test_a_listed_field_is_rejected_wherever_it_appears() -> None:
    with pytest.raises(UnsupportedFeatureError) as caught:
        reject_unsupported(REQUEST, LISTS)

    message = str(caught.value)
    assert message.startswith("unsupported_feature:")
    for expected in ("policy", "configuration.tariff", "item.rate", "container.rate_table"):
        assert expected in message


def test_one_field_on_several_entries_is_named_once() -> None:
    # The name identifies the field, not each position it was found in: fifty containers
    # carrying one unimplemented field are one complaint, not fifty.
    with pytest.raises(UnsupportedFeatureError) as caught:
        reject_unsupported(REQUEST, {"item": ("rate",)})

    assert str(caught.value).count("item.rate") == 1


def test_a_request_that_touches_nothing_listed_is_accepted() -> None:
    reject_unsupported(REQUEST, {"request": ("other",), "item": ("unrelated",)})


def test_the_unsupported_lists_match_what_the_field_matrix_records() -> None:
    """Every refusal this engine makes is recorded in the matrix, and the reverse.

    The assertion used to be that all four lists are empty, which was the same thing while
    they were -- and stopped being the same thing the moment one was populated. What the
    coupling is actually for is that the corpus *asserts* each rejection instead of merely
    tolerating it, so read the matrix and compare both directions.
    """
    matrix = json.loads(
        (ROOT / "conformance/public-field-matrix.json").read_text()
    )
    rejected_by_matrix = {
        path
        for path, row in matrix["fields"].items()
        if matrix["support_sets"][row["support"]]["python"]
        == "rejected:unsupported_feature"
    }
    declared = {f"{scope}s.*.{name}" if scope in ("item", "container") else name
                for scope, names in UNSUPPORTED_FIELDS.items() for name in names}
    # A value-keyed refusal is one matrix row for the field itself. `hull_vertices` is an
    # array of points, so the schema's leaves -- and therefore its rows -- are the three
    # coordinates, not the array.
    declared |= {"items.*.shape_type"} if UNSUPPORTED_SHAPE_TYPES else set()
    if "items.*.hull_vertices" in declared:
        declared.discard("items.*.hull_vertices")
        declared |= {f"items.*.hull_vertices.*.{axis}" for axis in "xyz"}

    assert declared == rejected_by_matrix, (
        "the engine and the matrix disagree about what Python refuses; "
        f"engine only: {sorted(declared - rejected_by_matrix)}, "
        f"matrix only: {sorted(rejected_by_matrix - declared)}"
    )


def test_the_default_shape_type_is_served_rather_than_refused() -> None:
    """`rigid_cuboid` is implemented, so spelling the default out must not be a rejection.

    This is why `shape_type` is not in the presence-keyed table: that table means "this
    engine does not implement the field at all", and a value-keyed refusal is a different
    claim. A caller who writes the default explicitly is asking for what they already get.
    """
    request = {
        "units": {"length": "mm"},
        "items": [{
            "id": "a",
            "shape_type": "rigid_cuboid",
            "dimensions": {"length": "100", "width": "100", "height": "100"},
        }],
        "containers": [{
            "id": "c",
            "inner_dimensions": {"length": "200", "width": "200", "height": "200"},
        }],
    }
    assert pack_from_dict(request)["status"] == "feasible"


def test_an_unimplemented_shape_type_names_the_value_it_refused() -> None:
    with pytest.raises(UnsupportedFeatureError) as caught:
        reject_unsupported(
            {"items": [{"id": "a", "shape_type": "convex_hull"}]},
            {"request": (), "configuration": (), "item": (), "container": ()},
            ("convex_hull",),
        )
    assert "item.shape_type=convex_hull" in str(caught.value)


def test_the_guard_is_wired_into_the_real_entry_point() -> None:
    # A guard nothing calls rejects nothing, so this goes through `pack_from_dict` and
    # the real constant. With every list empty the assertion is that an ordinary request
    # still packs -- the guard's own rejection is covered above, against injected lists.
    request = {
        "units": {"length": "mm"},
        "items": [{"id": "a", "dimensions": {"length": "100", "width": "100", "height": "100"}}],
        "containers": [{"id": "c", "inner_dimensions": {"length": "200", "width": "200", "height": "200"}}],
    }
    assert pack_from_dict(request)["status"] == "feasible"
