"""Catalog date validation at the exported commerce boundary."""

import pytest

from packvium.commerce import CommerceInputError, catalog_version_info


@pytest.mark.parametrize("catalog_id", ["c", "missing"])
def test_negative_resolution_time_is_an_input_error_before_catalog_lookup(catalog_id):
    document = {"catalogs": [{"catalog_id": "c", "versions": [
        {"effective_at": 0, "published_at": 0, "snapshot": {}},
    ]}]}
    with pytest.raises(CommerceInputError, match="resolved_at"):
        catalog_version_info(document, {"catalog_id": catalog_id, "version": 1, "resolved_at": -1})


@pytest.mark.parametrize("rollback", [
    {"rollback_to": 1, "published_at": 0, "effective_at": -1},
    {"rollback_to": 1, "published_at": -1, "effective_at": 0},
    {"rollback_to": 1, "published_at": -1},
])
def test_negative_rollback_dates_are_input_errors(rollback):
    document = {"catalogs": [{"catalog_id": "c", "versions": [
        {"effective_at": 0, "published_at": 0, "snapshot": {}}, rollback,
    ]}]}
    with pytest.raises(CommerceInputError):
        catalog_version_info(document, {"catalog_id": "c", "version": 2, "resolved_at": 0})


def test_zero_dates_are_valid_for_a_rollback():
    document = {"catalogs": [{"catalog_id": "c", "versions": [
        {"effective_at": 0, "published_at": 0, "snapshot": {}},
        {"rollback_to": 1, "published_at": 0},
    ]}]}
    result = catalog_version_info(document, {"catalog_id": "c", "as_of": 0, "resolved_at": 0})
    assert result["status"] == "ok"
    assert result["catalog"]["version"] == 2
    assert result["catalog"]["effective_at"] == 0
    assert result["catalog"]["published_at"] == 0
    assert result["catalog"]["resolved_at"] == 0
