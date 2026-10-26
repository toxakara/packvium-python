"""A request built from objects is refused as the same request sent as JSON is.

Each case is one request two ways: built from objects and passed to `Packer.pack`, and spelled
as JSON and passed to `pack_from_dict`. Both must raise `InvalidRequestError` with the same
reason. A value refused once the objects are in a request carries the JSON field too; one an
object's own constructor refuses has no position yet, so its field is empty."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from packvium import (Container, Dimensions, EffortBudget, InvalidRequestError, Item, Packer,
                      PackingConfig, pack_from_dict)
from packvium.flat_bag import FLAT_BAG
from packvium.geometry import Point, Rotation
from packvium.made_to_measure import MadeToMeasure, as_triple
from packvium.models import FixedPlacement, RateTable
from packvium.units import Length

TOO_LARGE = 2**60
Json = dict[str, Any]


def json_request(edit: Callable[[Json], None] = lambda request: None) -> Json:
    request: Json = {
        "items": [{"id": "cube", "dimensions": {"length": "100", "width": "100", "height": "100"}}],
        "containers": [{"id": "box", "inner_dimensions": {"length": "200", "width": "100", "height": "200"}}],
        "configuration": {"solver_profile": "fast"},
    }
    edit(request)
    return request


def cube(**fields: Any) -> Item:
    return Item.create("cube", Dimensions.mm(100, 100, 100), **fields)


def box(**fields: Any) -> Container:
    return Container.create("box", Dimensions.mm(200, 100, 200), **fields)


def packed(items: list[Item] | None = None, containers: list[Container] | None = None,
           config: PackingConfig | None = None,
           fixed_placements: tuple[FixedPlacement, ...] = ()) -> Callable[[], Any]:
    return lambda: Packer(config or PackingConfig.fast()).pack(items or [cube()], containers or [box()],
                                                               fixed_placements)


def fixed(**fields: Any) -> tuple[FixedPlacement, ...]:
    return (FixedPlacement(**{"item_id": "cube", "container_id": "box", "position": Point(0, 0, 0),
                              "rotation": Rotation.LWH, **fields}),)


def json_fixed(**fields: Any) -> Callable[[Json], None]:
    placement = {"item_type": "cube", "container_type": "box", "container_instance": 1,
                 "position": {"x": "0", "y": "0", "z": "0"}, "orientation": "LWH", **fields}
    return lambda r: r.update(fixed_placements=[placement])


def json_refusal(edit: Callable[[Json], None]) -> InvalidRequestError:
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(json_request(edit))
    return caught.value


def object_refusal(build: Callable[[], Any]) -> InvalidRequestError:
    with pytest.raises(InvalidRequestError) as caught:
        build()
    return caught.value


def machine(cost: int = 0) -> MadeToMeasure:
    return MadeToMeasure(minimum=as_triple(Dimensions.mm(100, 100, 100)),
                         increment=as_triple(Dimensions.mm(50, 50, 50)), cost_per_square_metre_minor=cost)


JSON_MACHINE = {"minimum": {"length": "100", "width": "100", "height": "100"},
                "increment": {"length": "50", "width": "50", "height": "50"}}

#: Refused once the objects form a request: same reason and same field as JSON.
IN_A_REQUEST = [
    ("quantity past 2^53 - 1", packed([cube(quantity=TOO_LARGE)]),
     lambda r: r["items"][0].update(quantity=TOO_LARGE)),
    ("repeated item id", packed([cube(), cube()]),
     lambda r: r["items"].append(dict(r["items"][0]))),
    ("repeated container id", packed(containers=[box(), box()]),
     lambda r: r["containers"].append(dict(r["containers"][0]))),
    ("empty stacking group", packed([cube(stacking_group="")]),
     lambda r: r["items"][0].update(stacking_group="")),
    ("max_stacked_items past 2^53 - 1", packed([cube(max_stacked_items=TOO_LARGE)]),
     lambda r: r["items"][0].update(max_stacked_items=TOO_LARGE)),
    ("container cost past 2^53 - 1", packed(containers=[box(cost_minor=TOO_LARGE)]),
     lambda r: r["containers"][0].update(cost_minor=TOO_LARGE)),
    ("tag limit past 2^53 - 1", packed(containers=[box(tag_limits={"fragile": TOO_LARGE})]),
     lambda r: r["containers"][0].update(tag_limits={"fragile": TOO_LARGE})),
    ("rate table price past 2^53 - 1",
     packed(containers=[box(rate_table=RateTable((1000,), (TOO_LARGE,)))]),
     lambda r: r["containers"][0].update(rate_table={"weight_brackets_g": [1000], "prices_minor": [TOO_LARGE]})),
    ("time limit past 2^53 - 1", packed(config=PackingConfig.fast(time_limit_ms=TOO_LARGE)),
     lambda r: r["configuration"].update(time_limit_ms=TOO_LARGE)),
    ("effort limit past 2^53 - 1",
     packed(config=PackingConfig.fast(effort_budget=EffortBudget(max_restarts=TOO_LARGE))),
     lambda r: r["configuration"].update(effort_budget={"max_restarts": TOO_LARGE})),
    ("made to measure beside outer dimensions",
     packed(containers=[box(made_to_measure=machine(), outer_dimensions=Dimensions.mm(210, 110, 210))]),
     lambda r: r["containers"][0].update(made_to_measure=JSON_MACHINE,
                                         outer_dimensions={"length": "210", "width": "110", "height": "210"})),
    ("made to measure priced past 2^53 - 1", packed(containers=[box(made_to_measure=machine(cost=10**18))]),
     lambda r: r["containers"][0].update(made_to_measure={**JSON_MACHINE, "cost_per_square_metre_minor": 10**18})),
    # The one bag exclusion `Container` itself does not refuse.
    ("made to measure beside a flat bag",
     packed(containers=[box(deformable_boundary=FLAT_BAG, made_to_measure=machine())]),
     lambda r: r["containers"][0].update(deformable_boundary={"kind": FLAT_BAG}, made_to_measure=JSON_MACHINE)),
    ("item id not a string", packed([Item.create(5, Dimensions.mm(100, 100, 100))]),
     lambda r: r["items"][0].update(id=5)),
    ("container id not a string", packed(containers=[Container.create(5, Dimensions.mm(200, 100, 200))]),
     lambda r: r["containers"][0].update(id=5)),
    ("fixed placement item id not a string", packed(fixed_placements=fixed(item_id=5)), json_fixed(item_type=5)),
    ("fixed placement rotation not a Rotation", packed(fixed_placements=fixed(rotation="XYZ")),
     json_fixed(orientation="XYZ")),
    ("fixed placement instance zero", packed(fixed_placements=fixed(container_instance=0)),
     json_fixed(container_instance=0)),
    ("fixed placement position not a point", packed(fixed_placements=fixed(position=(0, 0, 0))),
     json_fixed(position=[0, 0, 0])),
    ("fixed placement coordinate not ticks", packed(fixed_placements=fixed(position=Point(1.5, 0, 0))),
     json_fixed(position={"x": 1.5, "y": "0", "z": "0"})),
]

#: Refused by an object's own constructor: same reason as JSON, and no field yet.
ON_ITS_OWN = [
    ("zero quantity", lambda: cube(quantity=0), lambda r: r["items"][0].update(quantity=0)),
    ("negative side", lambda: Dimensions.mm("-1", 100, 100),
     lambda r: r["items"][0]["dimensions"].update(length="-1")),
    ("unknown unit", lambda: Length.of(1, "furlong"), lambda r: r.update(units={"length": "furlong"})),
    ("boolean side", lambda: Dimensions.mm(True, 100, 100),
     lambda r: r["items"][0]["dimensions"].update(length=True)),
    ("support ratio above one", lambda: cube(minimum_support_ratio=2),
     lambda r: r["items"][0].update(minimum_support_ratio=2)),
    ("zero container quantity", lambda: box(quantity=0), lambda r: r["containers"][0].update(quantity=0)),
    ("too few candidate points", lambda: PackingConfig.fast(max_candidate_points=8),
     lambda r: r["configuration"].update(max_candidate_points=8)),
    ("zero effort limit", lambda: EffortBudget(max_restarts=0),
     lambda r: r["configuration"].update(effort_budget={"max_restarts": 0})),
    # A truthy value is not `True`.
    *[(f"{name} not a boolean", lambda n=name: cube(**{n: "no"}), lambda r, n=name: r["items"][0].update({n: "no"}))
      for name in ("keep_upright", "stackable", "must_be_on_floor")],
    ("require_placement_coordinates not a boolean", lambda: PackingConfig.fast(require_placement_coordinates=0),
     lambda r: r["configuration"].update(require_placement_coordinates=0)),
]


@pytest.mark.parametrize("name, build, edit", IN_A_REQUEST, ids=[case[0] for case in IN_A_REQUEST])
def test_a_request_of_objects_is_refused_with_the_json_reason_and_field(name: str, build, edit) -> None:
    from_objects, from_json = object_refusal(build), json_refusal(edit)
    assert (from_objects.code, from_objects.reason, from_objects.field) == (
        from_json.code, from_json.reason, from_json.field)
    assert str(from_objects) == str(from_json)


@pytest.mark.parametrize("name, build, edit", ON_ITS_OWN, ids=[case[0] for case in ON_ITS_OWN])
def test_an_object_refuses_with_the_json_reason_and_no_field(name: str, build, edit) -> None:
    from_objects, from_json = object_refusal(build), json_refusal(edit)
    assert (from_objects.code, from_objects.reason, from_objects.field) == (
        from_json.code, from_json.reason, "")


def test_a_quantity_past_the_exact_range_is_refused_before_any_solve() -> None:
    """It used to be expanded into that many instances, which never finished."""
    error = object_refusal(packed([cube(quantity=TOO_LARGE)]))
    assert error.field == "/items/0/quantity"


def test_accepted_stacking_groups_are_checked_in_code_point_order() -> None:
    """A set has no order, so hash order must not pick which value an error names."""
    error = object_refusal(packed([cube(accepts_stacking_groups=frozenset({"pallet", "", "box"}))]))
    assert (error.reason, error.field) == ("invalid_value", "/items/0/accepts_stacking_groups/0")


def test_a_mistyped_measure_is_still_a_type_error() -> None:
    with pytest.raises(TypeError):
        Length.mm(1.5)


def test_a_mistyped_flag_is_also_a_type_error() -> None:
    with pytest.raises(TypeError, match="stackable must be a boolean"):
        cube(stackable="no")


def test_a_constructor_refusal_keeps_its_own_message() -> None:
    """A wrapper that quotes the text (a commerce document) reads what it always read."""
    assert str(object_refusal(lambda: cube(quantity=0))) == "item quantity must be positive"


def test_the_json_path_still_reports_what_its_rules_do_not_name_as_invalid_value() -> None:
    """A model rule the JSON table lacks is answered as every engine answers it."""
    error = json_refusal(lambda r: r["containers"][0].update(
        rate_table={"weight_brackets_g": [5000, 1000], "prices_minor": [500, 900]}))
    assert (error.reason, error.field) == ("invalid_value", "")
    assert str(error) == "invalid_request: rate_table weight brackets must be strictly ascending"
