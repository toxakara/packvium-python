"""The refusal paths in the request decoder and the domain validators.

Coverage put `serialization.py` at 90.24% and `models.py` at 96.83%, and every uncovered
line in both was a `raise`. That is the worst shape a coverage gap can take here: a
validator nothing exercises does not fail loudly when it breaks, it starts *accepting*
what it was written to refuse, and the engine then answers confidently about a request
that never made sense.

`catalog_versions_used` is the sharpest case. It is provenance -- which immutable catalog
version produced the request -- so a malformed entry that slips through does not corrupt a
packing, it corrupts the record of what the packing was computed from, and that is
discovered much later than a wrong box.
"""

from __future__ import annotations

import pytest

from packvium.models import MAX_EXACT_STOP_INDEX, Item
from packvium.geometry import Dimensions
from packvium.serialization import (UnsupportedFeatureError, pack_from_dict,
                                    reject_unsupported)
from packvium.units import Length

MM = 16_000


def cube(**kwargs) -> Item:
    side = Dimensions(Length(10 * MM), Length(10 * MM), Length(10 * MM))
    return Item(id="cube", dimensions=side, **kwargs)


# ------------------------------------------------------------------- item validation

def test_a_stack_limit_below_one_is_refused():
    """Zero would mean "may not be stacked", which `stackable=False` already says, and a
    negative one means nothing at all. Two spellings of one rule is how they drift."""
    assert cube(max_stacked_items=1).max_stacked_items == 1
    with pytest.raises(ValueError, match="max_stacked_items must be at least 1"):
        cube(max_stacked_items=0)


@pytest.mark.parametrize("stop", [-1, MAX_EXACT_STOP_INDEX + 1])
def test_a_stop_index_outside_the_exact_range_is_refused(stop):
    """The ceiling is 2**53 - 1 and not an arbitrary limit: past it a float cannot tell two
    consecutive stops apart, and the route rule uses `inf` as an ordering sentinel beside
    real stops. Merging two stops into one would silently excuse a blocker."""
    with pytest.raises(ValueError):
        cube(stop_index=stop)


def test_the_two_ends_of_the_exact_stop_range_are_accepted():
    """The boundary itself is legal; only past it is not."""
    assert cube(stop_index=0).stop_index == 0
    assert cube(stop_index=MAX_EXACT_STOP_INDEX).stop_index == MAX_EXACT_STOP_INDEX


@pytest.mark.parametrize("height", [10 * MM, 11 * MM])
def test_a_nesting_height_outside_its_own_item_is_refused(height):
    """Nesting height is how far an item sinks into the one below. Equal to its own height
    means it vanishes; more means it occupies negative space."""
    with pytest.raises(ValueError, match="nesting_height"):
        cube(nesting_height=Length(height))


def test_a_negative_nesting_height_is_refused_by_the_unit_not_the_item():
    """Worth pinning rather than folding into the case above: the refusal comes from
    `Length`, one layer below, so the item validator never sees it. If `Length` ever grew
    permissive, `Item` would silently inherit that -- and this test is what would fail."""
    with pytest.raises(ValueError, match="length cannot be negative"):
        Length(-1)


def test_an_unknown_ground_contact_rule_is_refused():
    with pytest.raises(ValueError, match="ground_contact_rule must be one of"):
        cube(ground_contact_rule="hovering")


# ------------------------------------------------------- catalog provenance refusals

def _with_catalog(catalog):
    return {"units": {"length": "mm"},
            "items": [{"id": "cube", "quantity": 1,
                       "dimensions": {"length": "10", "width": "10", "height": "10"}}],
            "containers": [{"id": "crate",
                            "inner_dimensions": {"length": "100", "width": "100",
                                                 "height": "100"}}],
            "catalog_versions_used": catalog}


GOOD = {"catalog_id": "boxes", "version": 3, "effective_at": 0, "resolved_at": 0}


def test_a_well_formed_catalog_reference_is_carried_through():
    """The accepting case first: a refusal test that never sees an acceptance proves only
    that the field is rejected, which is equally true of a decoder that refuses everything."""
    result = pack_from_dict(_with_catalog([GOOD]))
    assert result["catalog_versions_used"] == [GOOD]


def test_catalog_versions_used_must_be_an_array():
    with pytest.raises(ValueError, match="must be an array"):
        pack_from_dict(_with_catalog({"catalog_id": "boxes"}))


@pytest.mark.parametrize("reference", [
    {"catalog_id": "boxes"},
    {**GOOD, "extra": 1},
    "boxes",
])
def test_a_reference_with_the_wrong_key_set_is_refused(reference):
    """Exactly the four keys, not "at least". An extra key is a claim the format does not
    define, and accepting it would make two engines disagree about what was recorded."""
    with pytest.raises(ValueError, match="must contain exactly"):
        pack_from_dict(_with_catalog([reference]))


@pytest.mark.parametrize("catalog_id", ["", 7, None])
def test_an_empty_or_non_string_catalog_id_is_refused(catalog_id):
    with pytest.raises(ValueError, match="catalog_id must be non-empty"):
        pack_from_dict(_with_catalog([{**GOOD, "catalog_id": catalog_id}]))


def test_a_duplicate_catalog_id_is_refused_as_ambiguous():
    """Two versions of one catalog in one request do not say which was used. Silently
    keeping the last would make provenance a function of ordering."""
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        pack_from_dict(_with_catalog([GOOD, {**GOOD, "version": 4}]))


@pytest.mark.parametrize("field,value", [
    ("version", 0), ("version", -1), ("effective_at", -1), ("resolved_at", -1),
    ("version", True), ("version", 1.0), ("effective_at", "0"),
])
def test_out_of_range_or_mistyped_catalog_numbers_are_refused(field, value):
    """`True` is the one worth spelling out: it is an `int` in Python and `True >= 1`, so
    a bare range check would accept `version: true` and record a provenance nobody wrote."""
    with pytest.raises(ValueError, match=field):
        pack_from_dict(_with_catalog([{**GOOD, field: value}]))


# --------------------------------------------------------------- the rest of the decoder

def test_a_rate_table_is_decoded_rather_than_ignored():
    """`_rate_table`'s constructing arm was uncovered: the unit suite only ever reached the
    `None` branch, so the shape of a decoded card rested on conformance alone."""
    request = {"units": {"length": "mm"},
               "items": [{"id": "cube", "quantity": 1, "weight": {"value": "1", "unit": "kg"},
                          "dimensions": {"length": "10", "width": "10", "height": "10"}}],
               "containers": [{"id": "crate",
                               "inner_dimensions": {"length": "100", "width": "100",
                                                    "height": "100"},
                               "rate_table": {"weight_brackets_g": [1000, 5000],
                                              "prices_minor": [500, 900]}}],
               "configuration": {"objective": "lowest_landed_cost",
                                 "dimensional_weight_divisor": 139}}
    assert pack_from_dict(request)["status"]


def test_a_malformed_entry_does_not_derail_the_unsupported_field_scan():
    """The scan walks `items` and `containers` looking for reserved names, and a non-object
    entry is the schema's problem rather than the guard's.

    Asserted on the guard directly, with an injected list, for the reason the guard takes
    its lists as parameters at all: against the shipped lists this would prove only that
    nothing was rejected, which is equally true of a guard that does nothing. The
    injected list names a field the malformed entry cannot carry, so reaching the end
    without raising is the whole assertion -- a guard that indexed into the string would
    raise from inside itself and mask the schema error the caller should see.
    """
    reject_unsupported(
        {"items": ["not-an-object", 7, None], "containers": ["neither"]},
        {"item": ("hull_vertices",), "container": ("pallet_overhang_limit",)},
    )


def test_the_scan_still_finds_a_reserved_field_beside_a_malformed_entry():
    """Skipping the junk must not skip the sibling: a request mixing one bad entry with a
    real reserved field is still refused, and named."""
    with pytest.raises(UnsupportedFeatureError, match="container.pallet_overhang_limit"):
        reject_unsupported(
            {"items": ["not-an-object"],
             "containers": ["junk", {"id": "c", "pallet_overhang_limit": {}}]},
            {"item": (), "container": ("pallet_overhang_limit",)},
        )
