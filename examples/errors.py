"""Turn a refused request into a useful answer, the way a web handler would.

Run it:

    PYTHONPATH=src python3 examples/errors.py

A packing service takes JSON from a caller it does not control. Most failures are the
caller's: a quantity of zero, a negative width, a unit nobody has heard of. Those raise
`InvalidRequestError` before anything is solved, with three fields a program can use:

- `reason`: one of a closed set (`missing_field`, `wrong_type`, `below_minimum`, ...),
  the thing to branch on;
- `field`: a JSON Pointer to the bad value in the request the caller sent, such as
  `/items/0/quantity`;
- `str(error)`: `invalid_request: <field>: <detail>`, the same text in every Packvium
  engine, fit for a log line.

`FixedPlacementError` is a subclass for items already in place that cannot be where the
request says. And a request that is valid but does not fit completely is not an error at
all: it is a result with `unpacked_items`.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from packvium import FixedPlacementError, InvalidRequestError, pack_from_dict

ORDER = {
    "units": {"length": "mm"},
    "configuration": {"time_limit_ms": 60_000, "effort_budget": {"max_restarts": 4}},
    "items": [
        {"id": "mug", "quantity": 2, "weight": "380 g",
         "dimensions": {"length": "100", "width": "100", "height": "120"}},
        {"id": "teapot", "quantity": 1, "weight": "1.2 kg",
         "dimensions": {"length": "220", "width": "160", "height": "180"}},
    ],
    "containers": [{"id": "box", "inner_dimensions": {"length": "300", "width": "250", "height": "200"}}],
}


def resolve(document: Any, pointer: str) -> list[Any]:
    """Every value along a JSON Pointer, root first, so a message can name the item by id."""
    values = [document]
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        current = values[-1]
        if isinstance(current, list) and key.isdigit() and int(key) < len(current):
            values.append(current[int(key)])
        elif isinstance(current, dict) and key in current:
            values.append(current[key])
        else:
            break
    return values


def where(request: dict, pointer: str) -> str:
    """`/items/1/dimensions/width` becomes `item "teapot", dimensions/width`.

    The pointer is exact but written for programs. People know their items by id, so the
    nearest object that has one names the place, and the rest of the pointer says which
    value in it.
    """
    tokens = pointer.split("/")[1:]
    values = resolve(request, pointer)
    for depth in range(len(values) - 1, 0, -1):
        value = values[depth]
        if isinstance(value, dict) and "id" in value:
            kind = tokens[depth - 2].rstrip("s")
            rest = "/".join(tokens[depth:])
            return f'{kind} "{value["id"]}"' + (f", {rest}" if rest else "")
    return pointer or "the request"


def handle(body: str) -> tuple[int, dict]:
    """What a POST /pack endpoint would do: a status code and a JSON body."""
    request = json.loads(body)
    try:
        result = pack_from_dict(request)
    except FixedPlacementError as error:
        # A subclass of InvalidRequestError, so it must be caught first. Its detail names
        # the physical problem (a collision, too much weight), which is what a dock
        # operator needs to hear.
        return 422, {"code": error.code, "reason": error.reason, "field": error.field,
                     "message": f"The items already loaded cannot stay as recorded: {error.detail}."}
    except InvalidRequestError as error:
        return 422, {"code": error.code, "reason": error.reason, "field": error.field,
                     "message": f"Please check {where(request, error.field)}: {error.detail}."}
    # Not an error: the caller gets the plan, and a clear list of what is not in it.
    left_out = [entry["item_id"] for entry in result.get("unpacked_items") or ()]
    return 200, {"status": result["status"], "boxes": len(result["containers"]), "left_out": left_out}


def show(label: str, request: dict) -> None:
    status, body = handle(json.dumps(request))
    print(f"  {label}")
    print(f"    HTTP {status}  {json.dumps(body)}")


def broken(change) -> dict:
    request = copy.deepcopy(ORDER)
    change(request)
    return request


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
section("1. Refused before solving: one reason, one pointer, one message")

show("a valid order", ORDER)
show("quantity 0",
     broken(lambda r: r["items"][0].update(quantity=0)))
show("a negative width",
     broken(lambda r: r["items"][1]["dimensions"].update(width="-160")))
show("a missing height",
     broken(lambda r: r["items"][1]["dimensions"].pop("height")))
show("a float where a measure belongs",
     broken(lambda r: r["items"][1]["dimensions"].update(width=160.5)))
show("the same container id twice",
     broken(lambda r: r["containers"].append(copy.deepcopy(r["containers"][0]))))
show("an unknown solver profile",
     broken(lambda r: r["configuration"].update(solver_profile="turbo")))
print("""
  A float is refused as a measure on purpose: binary floating point cannot hold most
  decimal lengths exactly, and every length here is exact. Send "160.5" as a string.
  When several values are wrong, the first is reported, in a fixed order every
  engine shares: units, configuration, items, containers, fixed placements.""")

# --------------------------------------------------------------------------------------
section("2. Items already in place that cannot be there")

# Turned (`WLH`: its width along the box's length) so the mugs still fit beside it.
on_board = {"item_type": "teapot", "container_type": "box", "orientation": "WLH",
            "position": {"x": "0", "y": "0", "z": "0"}}
show("a teapot already in the box, packed around",
     broken(lambda r: r.update(fixed_placements=[on_board])))
show("a teapot recorded at x = 200 mm, hanging out of a 300 mm box",
     broken(lambda r: r.update(fixed_placements=[dict(on_board, position={"x": "200"})])))
show("a mug and the teapot recorded in the same place",
     broken(lambda r: r.update(fixed_placements=[
         on_board, {"item_type": "mug", "container_type": "box", "orientation": "LWH"}])))
show("an orientation code that does not exist",
     broken(lambda r: r.update(fixed_placements=[dict(on_board, orientation="XYZ")])))
print("""
  `cannot_hold` means the fixed set is well formed but is not a valid packing on its
  own, and `field` is the whole list. `malformed` points at the one bad value.""")

# --------------------------------------------------------------------------------------
section("3. Not an error: a valid order that does not fit")

show("a teapot too tall for any box",
     broken(lambda r: r["items"][1]["dimensions"].update(height="450")))
print("""
  The caller asked a fair question and got an answer: the rest of the order is
  packed, and `unpacked_items` says what is not and why. Retrying it will not help,
  and neither will treating it as a 4xx.""")

# --------------------------------------------------------------------------------------
section("4. Two refusals that do not carry a pointer")

for label, change in (
    ("an unknown objective", lambda r: r["configuration"].update(objective="cheapest")),
    ("an unknown door", lambda r: r["containers"][0].update(access_directions=["rear"])),
):
    try:
        pack_from_dict(broken(change))
    except InvalidRequestError:
        print(f"  {label}: InvalidRequestError")
    except ValueError as error:
        print(f"  {label}: {type(error).__name__}: {str(error)[:70]}")
print("""
  Both are `ValueError`s, so a handler that ends with `except ValueError` still turns
  them into a 422 -- but they have no `reason` or `field` to highlight, only a message.""")
