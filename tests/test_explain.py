"""Rendering an `UnpackedItem` into a human-readable sentence must be driven
entirely by its structured `reason`/`proof`, never by string-matching anything, and
must be deterministic -- the same item always explains the same way.
"""

from __future__ import annotations

import pytest

from packvium import (Dimensions, Item, ReasonProof, RejectionObservation, UnknownReasonError,
                         UnpackedItem)
from packvium.explain import (LEVEL_PREFIXES, REASON_MESSAGES, explain_reason,
                                 explain_unpacked_item, explain_unpacked_items,
                                 explanation_for_unpacked_item)


def instance(id: str = "a"):
    item = Item.create(id, Dimensions.mm(1, 1, 1))
    return item.instances()[0]


# ------------------------------------------------------------------- reason coverage

def test_every_reason_this_library_can_produce_has_a_registered_message():
    """The reason-code vocabulary, pinned here so a new reason code introduced without a
    matching message fails a test instead of shipping a silently mute explanation."""
    produced = {
        "no_compatible_container_dimensions", "rotation_restricted", "payload_exceeded",
        "no_eligible_container", "time_limit", "effort_limit", "group_cannot_fit_together",
        "insufficient_support", "no_feasible_placement",
    }
    assert produced <= REASON_MESSAGES.keys()


def test_an_unregistered_reason_code_raises_rather_than_rendering_silently():
    with pytest.raises(UnknownReasonError):
        explain_reason("not_a_real_reason_code")


# --------------------------------------------------------------------- rendering

def test_explanation_names_the_item_and_the_reason_sentence():
    item = UnpackedItem(instance("crate"), "payload_exceeded")
    text = explain_unpacked_item(item)
    assert text.startswith("crate#1: ")
    assert REASON_MESSAGES["payload_exceeded"] in text


def test_explanation_carries_the_proof_level_as_a_prefix():
    proven = UnpackedItem(instance(), "no_compatible_container_dimensions")
    assert proven.proof.level == "proven"
    assert explain_unpacked_item(proven).startswith("a#1: Proven: ")

    limited = UnpackedItem(instance(), "time_limit")
    assert limited.proof.level == "unknown_due_to_limit"
    assert LEVEL_PREFIXES["unknown_due_to_limit"] in explain_unpacked_item(limited)


def test_a_proof_with_no_registered_level_renders_with_no_prefix_rather_than_raising():
    # Constructed directly, bypassing ReasonProof.for_reason's classification, to
    # prove the renderer degrades gracefully instead of assuming every level it might
    # ever see is one of the four it currently knows about.
    item = UnpackedItem(instance(), "payload_exceeded",
                        proof=ReasonProof("a_future_level", (RejectionObservation("payload_exceeded"),)))
    text = explain_unpacked_item(item)
    assert text == f"a#1: {REASON_MESSAGES['payload_exceeded']}"


def test_details_already_attached_are_appended_verbatim_not_reformatted():
    item = UnpackedItem(instance(), "effort_limit", details=("max_search_nodes", "exhausted at node 4021"))
    text = explain_unpacked_item(item)
    assert text.endswith(" (max_search_nodes; exhausted at node 4021)")


def test_no_details_produces_no_trailing_parenthetical():
    item = UnpackedItem(instance(), "no_feasible_placement")
    assert not explain_unpacked_item(item).endswith(")")


# ------------------------------------------------------------------- batch + determinism

def test_explain_unpacked_items_preserves_input_order():
    items = (UnpackedItem(instance("first"), "payload_exceeded"),
             UnpackedItem(instance("second"), "time_limit"))
    explained = explain_unpacked_items(items)
    assert explained[0].startswith("first#1:")
    assert explained[1].startswith("second#1:")


def test_rendering_the_same_item_twice_is_byte_identical():
    item = UnpackedItem(instance(), "insufficient_support", details=("x",))
    assert explain_unpacked_item(item) == explain_unpacked_item(item)


def test_localization_descriptor_is_stable_and_contains_arguments():
    item = UnpackedItem(instance("crate"), "payload_exceeded", details=("limit=1kg",))
    explanation = explanation_for_unpacked_item(item)
    assert explanation.message_key == "packvium.unpacked.payload_exceeded"
    assert explanation.argument("item_id") == "crate#1"
    assert explanation.argument("evidence_level") == "proven"
    assert explanation.argument("details") == "limit=1kg"
    assert explanation.default_message == REASON_MESSAGES["payload_exceeded"]
