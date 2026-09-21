"""JSON, CSV and print-ready work orders from an operational artifact.

Each export is a pure function of the artifact: no solver, carrier, renderer or clock, so an
export is replayable from the artifact it came from and four engines emit the same bytes.

- JSON is the artifact's RFC 8785 canonical form.
- CSV has one row per packing step and one per unplaced item. It is RFC 4180: a header row,
  CRLF after every row, and a field quoted only when it holds a comma, quote, CR or LF.
- The work order is one self-contained HTML document: inline print styles, no script, no
  external resource, ASCII source with entities for the few typographic characters.

Values are copied, never re-rendered or reinterpreted. That includes a CSV field beginning
with `=`: prefixing it would change an identifier a warehouse system matches on, so opening
the file in a spreadsheet is the reader's decision (`docs/OPERATIONAL-ARTIFACTS.md`).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from .artifacts import FORMAT, OperationalArtifactError, canonical_artifact_json

__all__ = ["CSV_COLUMNS", "export_csv", "export_json", "export_work_order_html"]

CSV_COLUMNS = (
    "record", "container_index", "container_type", "sequence", "item_type", "orientation",
    "x_ticks", "y_ticks", "z_ticks", "x", "y", "z", "length", "width", "height", "length_unit",
    "reason", "proof_level",
)

_CSV_SPECIAL = (",", '"', "\r", "\n")

_STYLE = (
    "body{font-family:system-ui,sans-serif;margin:24px;color:#111}"
    "h1{font-size:20px}h2{font-size:16px;margin-top:24px}"
    "table{border-collapse:collapse;width:100%;margin:8px 0 16px}"
    "th,td{border:1px solid #999;padding:4px 6px;text-align:left;font-size:12px;vertical-align:top}"
    "th{background:#eee}"
    ".facts td:first-child{width:28%;font-weight:600}"
    "@media print{body{margin:0}section.container{break-after:page}tr{break-inside:avoid}}"
)

_DASH = "&mdash;"


def export_json(artifact: Mapping[str, Any]) -> str:
    return canonical_artifact_json(_require(artifact))


def export_csv(artifact: Mapping[str, Any]) -> str:
    document = _require(artifact)
    work_order = document["work_order"]
    rows: List[Sequence[Any]] = [CSV_COLUMNS]
    for container in work_order["containers"]:
        for line in container["lines"]:
            reference = line["placement"]
            ticks = reference["position_ticks"]
            position, dimensions = line["position"], line["dimensions"]
            rows.append((
                "step", reference["container_index"], container["container_type"], line.get("sequence"),
                reference["item_type"], reference["orientation"], ticks["x"], ticks["y"], ticks["z"],
                position["x"], position["y"], position["z"],
                dimensions["length"], dimensions["width"], dimensions["height"], work_order["length_unit"],
                None, None,
            ))
    for unplaced in document["plan"]["unplaced"]:
        facts = unplaced["facts"]
        rows.append(("unplaced", None, None, None, facts["item_type"], None, None, None, None,
                     None, None, None, None, None, None, None, facts["reason"], facts["proof_level"]))
    return "".join(",".join(_csv_field(value) for value in row) + "\r\n" for row in rows)


def export_work_order_html(artifact: Mapping[str, Any]) -> str:
    document = _require(artifact)
    plan, work_order, provenance = document["plan"], document["work_order"], document["provenance"]
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Packing work order</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        "<h1>Packing work order</h1>",
        '<table class="facts">',
        *_fact_rows(document, plan, provenance),
        "</table>",
    ]
    for plan_container, container in zip(plan["containers"], work_order["containers"]):
        lines.extend(_container_section(plan_container, container, work_order))
    lines.extend(_unplaced_section(plan["unplaced"]))
    lines.extend(["</body>", "</html>"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------ HTML


def _fact_rows(document: Mapping[str, Any], plan: Mapping[str, Any], provenance: Mapping[str, Any]) -> List[str]:
    facts, replay, solver = plan["facts"], provenance["replay"], provenance["solver"]
    feasibility = facts["feasibility"]
    replay_text = _escape(replay["level"]) + (f" ({_escape(replay['because'])})" if replay["because"] else "")
    solver_text = (
        f"{_escape(solver['profile'])} / {_escape(solver['solver'])} / seed {_escape(solver['seed'])}"
        if solver else _DASH
    )
    catalogs = ", ".join(f"{_escape(catalog.get('catalog_id'))} v{_escape(catalog.get('version'))}"
                         for catalog in provenance["catalog_versions_used"])
    rows = (
        ("Status", _escape(facts["status"])),
        ("Objective", _or_dash(plan["objective"])),
        ("Containers", _escape(facts["container_count"])),
        ("Score", "[" + ", ".join(_escape(term) for term in facts["score"]) + "]"),
        ("Feasibility", _or_dash(feasibility.get("code") if isinstance(feasibility, Mapping) else None)),
        ("Replay", replay_text),
        ("Solver", solver_text),
        ("Catalogs", catalogs or _DASH),
        ("Packvium", _escape(document["suite_version"])),
    )
    return [f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows]


def _container_section(plan_container: Mapping[str, Any], container: Mapping[str, Any],
                       work_order: Mapping[str, Any]) -> List[str]:
    facts = plan_container["facts"]
    weight_unit, length_unit = _escape(work_order["weight_unit"]), _escape(work_order["length_unit"])
    lines = [
        '<section class="container">',
        f"<h2>Container {_ordinal(container['container_index'])}: {_or_dash(container['container_type'])}</h2>",
        f"<p>Payload {_escape(container['payload_weight'])} {weight_unit} &middot; "
        f"Gross {_escape(container['gross_weight'])} {weight_unit} &middot; "
        f"Utilization {_or_dash(facts['volume_utilization'])}</p>",
    ]
    if plan_container["order"] == "loading":
        lines.append("<p>Order: loading. Follow the steps in sequence.</p>")
    else:
        lines.append("<p>Order: unavailable. No safe loading order was supplied, so the steps are not numbered.</p>")
    lines.extend([
        "<table>",
        f"<thead><tr><th>Step</th><th>Item</th><th>Orientation</th><th>Position x, y, z ({length_unit})</th>"
        f"<th>Size l &times; w &times; h ({length_unit})</th><th>Done</th></tr></thead>",
        "<tbody>",
    ])
    for line in container["lines"]:
        reference, position, size = line["placement"], line["position"], line["dimensions"]
        lines.append(
            f"<tr><td>{_escape(line.get('sequence', ''))}</td><td>{_escape(reference['item_type'])}</td>"
            f"<td>{_escape(reference['orientation'])}</td>"
            f"<td>{_escape(position['x'])}, {_escape(position['y'])}, {_escape(position['z'])}</td>"
            f"<td>{_escape(size['length'])} &times; {_escape(size['width'])} &times; {_escape(size['height'])}</td>"
            "<td>&#9744;</td></tr>"
        )
    lines.extend(["</tbody>", "</table>", "</section>"])
    return lines


def _unplaced_section(unplaced: Iterable[Mapping[str, Any]]) -> List[str]:
    entries = list(unplaced)
    lines = ["<section>", "<h2>Not packed</h2>"]
    if not entries:
        lines.extend(["<p>Every item was packed.</p>", "</section>"])
        return lines
    lines.extend(["<table>", "<thead><tr><th>Item</th><th>Reason</th><th>Proof</th></tr></thead>", "<tbody>"])
    for entry in entries:
        facts = entry["facts"]
        lines.append(f"<tr><td>{_or_dash(facts['item_type'])}</td><td>{_or_dash(facts['reason'])}</td>"
                     f"<td>{_or_dash(facts['proof_level'])}</td></tr>")
    lines.extend(["</tbody>", "</table>", "</section>"])
    return lines


def _text(value: Any) -> str:
    """The one rendering every engine shares: a string as itself, an integer in decimal, null as
    nothing. Anything else -- a boolean, a float, an object -- has a different default spelling
    in each language, so it is refused rather than printed four ways."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise OperationalArtifactError("invalid_value", f"a {type(value).__name__} has no single rendering in a work order")


def _ordinal(index: Any) -> str:
    """A container's 1-based number on the sheet. Computed, so it is checked like `_text`:
    `0.5 + 1` would print `1.5` in one engine and be refused in another."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise OperationalArtifactError("invalid_value", f"container index {index!r} is not an integer")
    return str(index + 1)


def _escape(value: Any) -> str:
    text = _text(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def _or_dash(value: Optional[Any]) -> str:
    return _DASH if value is None else _escape(value)


# ------------------------------------------------------------------------------------- CSV


def _csv_field(value: Any) -> str:
    text = _text(value)
    if any(special in text for special in _CSV_SPECIAL):
        return '"' + text.replace('"', '""') + '"'
    return text


def _require(artifact: Any) -> Mapping[str, Any]:
    """An export reads only a document it knows how to read, and says so when it cannot.

    It reads the artifact *through its canonical form*, so an export is a function of the bytes
    four engines agree on and not of how one language happens to hold them in memory: a `1.0`
    in a Python dict is the `1` that PHP, Rust and JavaScript read back from those bytes.
    """
    found = artifact.get("format") if isinstance(artifact, Mapping) else None
    if found != FORMAT:
        raise OperationalArtifactError("unknown_format", f"cannot export format {found!r}; this exporter reads {FORMAT}")
    return json.loads(canonical_artifact_json(artifact))
