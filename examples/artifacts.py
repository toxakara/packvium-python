"""Hand a packing result to a warehouse system that has no engine.

Run it:

    PYTHONPATH=src python3 examples/artifacts.py

An execution plan says what to lift first. A warehouse or transport system needs more than
that before it can act on its own: how big each box is, what the load weighs, a sheet to
print for the dock, and a record of which request and solver produced all of it. Without
those, it has to call the engine again.

The operational artifact is that one document. It wraps the plan unchanged and adds geometry,
display values and provenance. It exports to JSON, CSV and a printable HTML work order, and
none of that calls a solver, a renderer or a clock. Four engines build the same bytes from it.
"""

from __future__ import annotations

from packvium import AxisAlignedBox, Dimensions, Length, Point, pack_from_dict, safe_loading_order
from packvium.artifact_exports import export_csv, export_json, export_work_order_html
from packvium.artifacts import OperationalArtifactError, build_operational_artifact

REQUEST = {
    "units": {"length": "mm"},
    "configuration": {
        # A fuse far above what this solve needs, so the answer never depends on machine load.
        "time_limit_ms": 60_000,
    },
    "items": [
        {"id": "printer", "dimensions": {"length": 420, "width": 300, "height": 250}, "weight": "12 kg",
         "metadata": {"sales_order": "SO-1042"}},
        {"id": "toner", "quantity": 3, "dimensions": {"length": 300, "width": 100, "height": 100},
         "weight": "1.5 kg", "minimum_support_ratio": 0.5},
    ],
    "containers": [{"id": "crate", "inner_dimensions": {"length": 800, "width": 400, "height": 400}}],
}

result = pack_from_dict(REQUEST)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
section("1. One document that carries everything a consumer needs")

def box(placement: dict) -> AxisAlignedBox:
    position, size = placement["position"], placement["dimensions"]
    return AxisAlignedBox(Point(*(position[axis]["ticks"] for axis in ("x", "y", "z"))),
                          Dimensions(*(Length(size[axis]["ticks"]) for axis in ("length", "width", "height"))))


crate = result["containers"][0]
inside = Dimensions(*(Length(crate["inner_dimensions"][axis]["ticks"]) for axis in ("length", "width", "height")))
# The order boxes can go in without lifting one over another comes from the engine's own
# sequence API. The artifact is handed that order; it never invents one.
order = safe_loading_order([box(placement) for placement in crate["placements"]], inside)
artifact = build_operational_artifact(REQUEST, result, loading_orders={0: order})
provenance = artifact["provenance"]
print(f"  format:          {artifact['format']}")
print(f"  plan steps:      {len(artifact['plan']['containers'][0]['steps'])}")
print(f"  crate inside:    {artifact['geometry']['containers'][0]['inner_dimensions']['length']} ticks long")
print(f"  replay:          {provenance['replay']['level']}")
print(f"  sales order:     {provenance['request']['items'][0]['metadata']['sales_order']}")
print("""
  The request is inside the artifact, not a hash of it: a digest names a request,
  and only the request itself lets someone replay the artifact without a lookup.
  `replay` is `exact` because the search finished inside its fuse; a search the
  clock stopped would say `not_guaranteed` and name the field that decided it.""")

# --------------------------------------------------------------------------------------
section("2. A CSV a warehouse system can import")

for row in export_csv(artifact).split("\r\n")[:3]:
    print(f"  {row}")
print("""
  One row per step, in the plan's order, then one row per item that was not
  packed. The tick columns are the identifiers a system matches on; the rendered
  columns are for people. Nothing is re-rendered: every value is the result's own.""")

# --------------------------------------------------------------------------------------
section("3. A work order to print")

html = export_work_order_html(artifact)
print(f"  {len(html.encode())} bytes of HTML, one section per container, one checkbox per step")
print(f"  contains a script or an external resource: {'<script' in html or 'http' in html}")
print("""
  Self-contained on purpose: a work order that loads a stylesheet from the network
  is a work order that prints blank on a dock with no signal.""")

# --------------------------------------------------------------------------------------
section("4. The same bytes in every engine, and a refusal instead of a guess")

canonical = export_json(artifact)
print(f"  canonical JSON: {len(canonical.encode())} bytes, starting {canonical[:40]}...")
try:
    export_csv(dict(artifact, format="packvium-operational-artifact/v2"))
except OperationalArtifactError as error:
    print(f"  a v2 document: refused with {error.code}")
print("""
  The canonical form is RFC 8785, so PHP, Rust and JavaScript build these exact
  bytes from the same request and result. A reader that meets a format it does not
  know refuses it by name rather than drawing half of it.""")
