"""Quote a shipment, apply a policy rule, and inspect a catalog version.

Run it:

    python examples/commerce.py

Everything the three functions need arrives in one *commerce document*: the tariffs you
publish, the eligibility rules you publish, and the catalog versions you publish. Each
history is a list, and a version's number is simply its position in that list starting
at 1 -- so `tariff_version: 2` always means "the second entry under this carrier and
service", with no separate numbering to keep in sync.

Nothing here reads the clock. Every "which version applies" question is answered from a
pin you supply, so the same call replays to the same answer next year.
"""

from __future__ import annotations

import json

from packvium.commerce import canonical_json, catalog_version_info, evaluate_policy, quote

# --------------------------------------------------------------------------------------
# One document, three histories. You would normally load this from your own storage.
# --------------------------------------------------------------------------------------
DOCUMENT = {
    "tariffs": [
        {
            "carrier_id": "acme",
            "service_id": "ground",
            # Two published versions. The second takes effect at instant 1000.
            "versions": [
                {
                    "effective_at": 0,
                    # Volume in mm^3 divided by this gives dimensional weight in grams.
                    "dimensional_weight_divisor": 5000,
                    # Minor currency units (cents) per billed kilogram, per zone.
                    "cost_per_dimensional_kg_minor": {"zone-a": 450, "zone-b": 610},
                    "minimum_charge_minor": 900,
                    # Permille: 120 means 12.0%.
                    "fuel_surcharge_permille": 120,
                    "accessorials": [
                        {"accessorial_id": "liftgate", "flat_charge_minor": 250},
                        {"accessorial_id": "residential", "permille_of_base": 75},
                    ],
                },
                {
                    "effective_at": 1000,
                    "dimensional_weight_divisor": 4000,
                    "cost_per_dimensional_kg_minor": {"zone-a": 480},
                    "minimum_charge_minor": 950,
                    "fuel_surcharge_permille": 140,
                    "accessorials": [{"accessorial_id": "liftgate", "flat_charge_minor": 275}],
                },
            ],
        },
    ],
    "policy_rules": [
        {
            "rule_id": "no-hazmat-air",
            "versions": [
                {
                    "scope": "hazmat",
                    "action": "reject",
                    "priority": 10,
                    "effective_at": 0,
                    "reason": "class 1.4 is not accepted on air services",
                    "predicates": [
                        {"scope": "hazmat", "field": "un_class",
                         "operator": "equals", "value": "1.4"},
                    ],
                },
            ],
        },
    ],
    "catalogs": [
        {
            "catalog_id": "dc-12",
            "versions": [
                {
                    "effective_at": 0, "published_at": 0, "note": "initial",
                    "snapshot": {
                        "items": [{"id": "sku-1", "dimensions_mm": [100, 200, 300],
                                   "weight_g": 1200}],
                        "cartons": [{"id": "box-m", "inner_dimensions_mm": [320, 240, 180],
                                     "max_payload_g": 15000, "cost_minor": 85}],
                    },
                },
                # A rollback is a new, higher-numbered version, never an edit of history.
                {"rollback_to": 1, "published_at": 900, "effective_at": 900,
                 "note": "revert the weight correction"},
            ],
        },
    ],
}


def show(title: str, result: dict) -> None:
    print(f"\n== {title}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------------------
# 1. Quote: what does this shipment cost?
# --------------------------------------------------------------------------------------
pinned = quote(DOCUMENT, {
    "carrier_id": "acme",
    "service_id": "ground",
    "tariff_version": 1,          # replay against exactly this version...
    "zone": "zone-a",
    "actual_weight_g": 1200,
    "volume_mm3": 6_000_000,
    "requested_accessorials": ["liftgate"],
})
show("a quote pinned to tariff version 1", pinned)
print(f"   -> the caller pays {pinned['quote']['total_minor']} minor units")

effective = quote(DOCUMENT, {
    "carrier_id": "acme",
    "service_id": "ground",
    "as_of": 1500,                # ...or against whatever was in force at this instant
    "zone": "zone-a",
    "actual_weight_g": 1200,
    "volume_mm3": 6_000_000,
    "requested_accessorials": ["liftgate"],
})
print(f"\n   as of instant 1500 the tariff is version "
      f"{effective['quote']['tariff_version']}, and the price is "
      f"{effective['quote']['total_minor']}")

# A request the model cannot answer is not an exception. It is a result with a status,
# a code from a closed set, and the structured fields that say what was missing.
unpriceable = quote(DOCUMENT, {
    "carrier_id": "acme", "service_id": "ground", "tariff_version": 1,
    "zone": "zone-nowhere", "actual_weight_g": 1200, "volume_mm3": 6_000_000,
})
show("a zone this tariff does not price", unpriceable)

# --------------------------------------------------------------------------------------
# 2. Policy: may this shipment go at all?
# --------------------------------------------------------------------------------------
decision = evaluate_policy(DOCUMENT, {
    "scope": "hazmat",
    "context": {"un_class": "1.4"},
    "as_of": 0,
})
show("a policy decision, with the rule that made it", decision)

allowed = evaluate_policy(DOCUMENT, {
    "scope": "hazmat", "context": {"un_class": "9"}, "as_of": 0,
})
print(f"\n   nothing matched, so the shipment is allowed with no citation: "
      f"{allowed['decision']['citation']}")

# --------------------------------------------------------------------------------------
# 3. Catalog: which master data was this decision made against?
# --------------------------------------------------------------------------------------
catalog = catalog_version_info(DOCUMENT, {
    "catalog_id": "dc-12",
    "version": 2,
    "resolved_at": 1700,
})
show("catalog version metadata", catalog)
print(f"\n   version {catalog['catalog']['version']} is a rollback of version "
      f"{catalog['catalog']['rolled_back_from']}")

# --------------------------------------------------------------------------------------
# Storing or comparing a result: use the canonical form, never str().
# --------------------------------------------------------------------------------------
print("\n== the canonical form is what you store, log and compare")
print(canonical_json(pinned))
