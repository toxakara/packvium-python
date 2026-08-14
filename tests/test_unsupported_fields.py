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

import pytest

from packvium.serialization import (
    UNSUPPORTED_FIELDS,
    UnsupportedFeatureError,
    pack_from_dict,
    reject_unsupported,
)

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
    # Each name here must carry a rejected:unsupported_feature level for Python in
    # conformance/public-field-matrix.json, which is what makes the corpus assert the
    # rejection rather than merely tolerate it.
    assert UNSUPPORTED_FIELDS["request"] == ()
    assert UNSUPPORTED_FIELDS["configuration"] == ()
    assert UNSUPPORTED_FIELDS["item"] == ()
    assert UNSUPPORTED_FIELDS["container"] == ()


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
