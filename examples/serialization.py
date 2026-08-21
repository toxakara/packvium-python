"""Serialization: the same request as JSON, and what comes back.

Run it:

    PYTHONPATH=src python3 examples/serialization.py

Everything the library can do is reachable over one JSON document, and that is not a
convenience wrapper -- it is the contract four independent implementations are held to.
The Python, PHP, Rust and JavaScript engines read this exact shape and are checked
against each other on a shared fixture corpus, so a request you build here is a request
you can hand to any of them.

Two consequences worth knowing:

- lengths and weights travel as *decimal strings*, never as floats, so "12 3/8 in"
  survives the trip intact (see units.py for why that matters);
- a field this engine has deliberately not implemented yet is refused by name, never
  quietly ignored -- but a key the parser simply does not recognise *is* ignored. The
  difference matters, and the last section shows both.
"""

import json

from packvium import pack_from_dict
from packvium.serialization import UNSUPPORTED_FIELDS, UnsupportedFeatureError, reject_unsupported

# ---------------------------------------------------------------------------------
# A request is a plain dict. This one is the whole vocabulary in miniature: units,
# solver configuration, items with rules, and containers with a carrier rate card.
# ---------------------------------------------------------------------------------
request = {
    "units": {"length": "mm"},
    "configuration": {
        "objective": "lowest_landed_cost",
        "dimensional_weight_divisor": 5000,
        "dimensional_weight_length_unit": "cm",
        "dimensional_weight_weight_unit": "kg",
        "profile": "balanced",
        "seed": 42,
        "top_k": 2,
    },
    "items": [
        {
            "id": "book",
            "quantity": 6,
            "dimensions": {"length": "210", "width": "140", "height": "30"},
            "weight": "450 g",
        },
        {
            "id": "mug",
            "quantity": 2,
            "dimensions": {"length": "100", "width": "100", "height": "120"},
            "weight": "380 g",
            "keep_upright": True,
            "max_top_load": "1 kg",
        },
    ],
    "containers": [
        {
            "id": "box-m",
            "inner_dimensions": {"length": "400", "width": "300", "height": "250"},
            "max_payload": "20 kg",
            "cost_minor": 180,
            "rate_table": {
                "weight_brackets_g": [5_000, 10_000, 30_000],
                "prices_minor": [890, 1_240, 2_050],
                "minimum_charge_minor": 650,
                "fuel_surcharge_permille": 78,
            },
        },
    ],
}

result = pack_from_dict(request)

# ---------------------------------------------------------------------------------
# The result is a plain dict too, and it is deliberately verbose: every placement has
# exact coordinates, every unplaced item has a structured reason, and the algorithm
# report says which solver won and what it spent getting there.
# ---------------------------------------------------------------------------------
print("status     ", result["status"])
print("score      ", result["score"], "  <- lexicographic, exact integers, cheapest first")
print("solver     ", result["algorithm"]["solver"], "in", result["algorithm"]["duration_ms"], "ms")
print("containers ", [c["container_type"] for c in result["containers"]])
print("placed     ", sum(len(c["placements"]) for c in result["containers"]))
print("unplaced   ", [(u["item_id"], u["reason"]) for u in (result.get("unpacked_items") or ())])

first = result["containers"][0]["placements"][0]
print()
print("one placement, in full:")
print(json.dumps(first, indent=2)[:400], "...")

# ---------------------------------------------------------------------------------
# `top_k` asks for runners-up. They are real alternative arrangements, already scored
# and already validated -- useful when you want to show a human a choice rather than a
# verdict.
# ---------------------------------------------------------------------------------
# Two alternatives can share a score and still be different arrangements -- equal cost,
# different geometry. Compare their placements, not their scores, when showing a choice.
print()
print("alternatives:", len(result.get("alternatives") or ()))
for alternative in result.get("alternatives") or ():
    positions = [(p["item_id"], p["position"]["x"]["value"]) for c in alternative["containers"] for p in c["placements"]]
    print("   score", alternative["score"], "first two placements", positions[:2])

# ---------------------------------------------------------------------------------
# What is refused, and what is not. Worth knowing exactly, because the two look alike
# from the outside.
#
# A key the parser does not recognise is *ignored*. Misspell `keep_upright` and you get
# a silently unrotated mug, not an error -- the strictness lives in the request JSON
# Schema, which sets `additionalProperties: false` and ships with the project rather
# than with this package. Validate against it if you want typo protection; the library
# alone will not give you any. See docs/SERIALIZATION.md.
# ---------------------------------------------------------------------------------
print()
typo = json.loads(json.dumps(request))
typo["items"][1]["keep_uprght"] = True
print("misspelled field:", pack_from_dict(typo)["status"], "-- accepted; the misspelling is invisible to the parser,")
print("                  so `keep_upright` was never applied to the mug")

# An unknown *value* where the engine has to choose a behaviour is a different matter.
# There is no sensible default for "rank by something I have never heard of".
bad_objective = json.loads(json.dumps(request))
bad_objective["configuration"]["objective"] = "cheapest"
try:
    pack_from_dict(bad_objective)
except ValueError as refusal:
    print("unknown objective:", str(refusal)[:100])

# And a field this engine has named as not-yet-implemented is refused explicitly, so a
# request written for a newer engine fails loudly instead of being half-honoured. The
# list is empty right now, which is what "this engine is caught up" looks like.
print("fields this engine refuses by name:",
      {scope: fields for scope, fields in UNSUPPORTED_FIELDS.items() if fields} or "none")
from_the_future = json.loads(json.dumps(request))
from_the_future["items"][0]["shape_type"] = "convex_hull"
try:
    reject_unsupported(from_the_future, {"item": ("shape_type",), "request": (), "configuration": (), "container": ()})
except UnsupportedFeatureError as refusal:
    print("  what it looks like when one is:", str(refusal)[:110])

# ---------------------------------------------------------------------------------
# The same document drives the command line, which reads a request on stdin and writes
# a result on stdout -- which is how the cross-language conformance harness talks to
# every engine, and how you would call this from a language with no binding yet:
#
#     echo '<request json>' | python3 -m packvium
#
# ---------------------------------------------------------------------------------
print()
print("the CLI takes exactly the document above:  echo '...' | python3 -m packvium")
