"""Packvium's exported commercial and control-plane API.

Three deterministic functions over one canonical JSON document:

    >>> from packvium.commerce import quote
    >>> document = {"tariffs": [{"carrier_id": "acme", "service_id": "ground", "versions": [
    ...     {"effective_at": 0, "dimensional_weight_divisor": 5000,
    ...      "cost_per_dimensional_kg_minor": {"zone-a": 450}}]}]}
    >>> quote(document, {"carrier_id": "acme", "service_id": "ground", "tariff_version": 1,
    ...                  "zone": "zone-a", "actual_weight_g": 2000,
    ...                  "volume_mm3": 1000000})["quote"]["total_minor"]
    900

The full contract -- document format, result shapes, rejection codes, complexity and
limitations -- is docs/COMMERCE-API.md.

The models underneath (`rating`, `policy`, `catalog`) are the same objects the
workspace application modules import through `commerce/rating/model.py`,
`domain/policy/model.py` and `domain/catalog/model.py`; those paths are re-export shims
onto this package, so there is exactly one implementation and an exported quote cannot
drift from the price a packing request was optimised against.
"""

from __future__ import annotations

from .api import (
    API_VERSION,
    REJECTION_CODES,
    canonical_json,
    catalog_version_info,
    evaluate_policy,
    quote,
)
from .document import CommerceDocument, load_document
from .errors import CommerceError, CommerceInputError

__all__ = [
    "API_VERSION",
    "REJECTION_CODES",
    "CommerceDocument",
    "CommerceError",
    "CommerceInputError",
    "canonical_json",
    "catalog_version_info",
    "evaluate_policy",
    "load_document",
    "quote",
]
