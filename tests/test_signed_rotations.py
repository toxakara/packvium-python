"""1.10: the 24 proper rotations a convex hull may take, and the six a box keeps.

The legacy six keep their matrices, so every answer that lists none of the 18 signed codes is
unchanged; the signed codes are refused on anything but a hull, and `keep_upright` keeps the
codes whose z entry is `+H`.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from packvium import Dimensions, Item, PackingConfig, Rotation
from packvium.geometry import ShapeType
from packvium.hull import HullShape, rotate, shape_for
from packvium.locks import PlacementLock
from packvium.orientations import AXIS_SIGNS, CODES, LEGACY_CODES, SOURCE_AXES, UPRIGHT_CODES
from packvium.request_errors import InvalidRequestError, check_request
from support import assert_sound, container, pack

MM = 16_000
SIDE = 10 * MM
WEDGE = ((0, 0, 0), (SIDE, 0, 0), (0, SIDE, 0), (0, 0, SIDE), (SIDE, 0, SIDE), (0, SIDE, SIDE))

#: The table every engine and validator shares, spelled out (docs/IRREGULAR-ITEMS.md).
TABLE = (
    ("LWH", (0, 1, 2), (1, 1, 1)), ("LHW", (0, 2, 1), (-1, 1, 1)),
    ("WLH", (1, 0, 2), (-1, 1, 1)), ("WHL", (1, 2, 0), (1, 1, 1)),
    ("HLW", (2, 0, 1), (1, 1, 1)), ("HWL", (2, 1, 0), (-1, 1, 1)),
    ("+L-W-H", (0, 1, 2), (1, -1, -1)), ("-L+W-H", (0, 1, 2), (-1, 1, -1)),
    ("-L-W+H", (0, 1, 2), (-1, -1, 1)), ("+L-H+W", (0, 2, 1), (1, -1, 1)),
    ("+L+H-W", (0, 2, 1), (1, 1, -1)), ("-L-H-W", (0, 2, 1), (-1, -1, -1)),
    ("+W-L+H", (1, 0, 2), (1, -1, 1)), ("+W+L-H", (1, 0, 2), (1, 1, -1)),
    ("-W-L-H", (1, 0, 2), (-1, -1, -1)), ("+W-H-L", (1, 2, 0), (1, -1, -1)),
    ("-W+H-L", (1, 2, 0), (-1, 1, -1)), ("-W-H+L", (1, 2, 0), (-1, -1, 1)),
    ("+H-L-W", (2, 0, 1), (1, -1, -1)), ("-H+L-W", (2, 0, 1), (-1, 1, -1)),
    ("-H-L+W", (2, 0, 1), (-1, -1, 1)), ("+H-W+L", (2, 1, 0), (1, -1, 1)),
    ("+H+W-L", (2, 1, 0), (1, 1, -1)), ("-H-W-L", (2, 1, 0), (-1, -1, -1)),
)


def _determinant(axes, signs) -> int:
    matrix = [[signs[row] if column == axes[row] else 0 for column in range(3)] for row in range(3)]
    a, b, c = matrix
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def wedge(id: str = "wedge", **kwargs) -> Item:
    return Item.create(id, Dimensions.mm(10, 10, 10), shape_type=ShapeType.CONVEX_HULL,
                       hull_vertices=WEDGE, **kwargs)


def test_the_table_is_the_canonical_order_and_every_entry_is_a_proper_rotation():
    assert tuple((code, SOURCE_AXES[code], AXIS_SIGNS[code]) for code in CODES) == TABLE
    assert tuple(rotation.value for rotation in Rotation) == CODES
    assert all(_determinant(axes, signs) == 1 for _, axes, signs in TABLE)
    # Every proper rotation of the cube's axes appears exactly once.
    assert len({(axes, signs) for _, axes, signs in TABLE}) == 24
    assert {axes for _, axes, _ in TABLE} == set(permutations(range(3)))


def test_the_legacy_vocabulary_and_its_defaults_are_unchanged():
    assert tuple(r.value for r in Rotation.all()) == LEGACY_CODES
    assert Rotation.upright() == (Rotation.LWH, Rotation.WLH)
    assert Rotation.every() == tuple(Rotation)
    assert UPRIGHT_CODES == ("LWH", "WLH", "-L-W+H", "+W-L+H")
    assert [r.value for r in Rotation.every() if r.is_upright] == list(UPRIGHT_CODES)
    assert not any(r.is_signed for r in Rotation.all())
    assert all(r.is_signed for r in Rotation.every()[6:])


@pytest.mark.parametrize("rotation", list(Rotation))
def test_a_box_ignores_the_signs(rotation):
    dims = Dimensions.mm(1, 2, 3)
    legacy = Rotation(LEGACY_CODES[[axes for _, axes, _ in TABLE[:6]].index(SOURCE_AXES[rotation.value])])
    assert dims.rotated(rotation) == dims.rotated(legacy)


@pytest.mark.parametrize("code, axes, signs", TABLE)
def test_a_hull_turns_by_its_signed_matrix_then_back_to_the_origin(code, axes, signs):
    turned = [tuple(signs[i] * vertex[axes[i]] for i in range(3)) for vertex in WEDGE]
    low = [min(vertex[i] for vertex in turned) for i in range(3)]
    assert rotate(WEDGE, code) == tuple(tuple(v[i] - low[i] for i in range(3)) for v in turned)
    assert HullShape.of(rotate(WEDGE, code)).volume == HullShape.of(WEDGE).volume


def test_the_24_rotations_of_a_chiral_hull_are_24_shapes_cached_apart():
    shapes = {code: shape_for(WEDGE, code) for code in CODES}
    # The wedge has a mirror plane, so pairs coincide; what matters is that the cache keys on
    # the full code and that the half-turn about z is its complement, not itself.
    assert shapes["-L-W+H"] != shapes["LWH"]
    assert all(shape_for(WEDGE, code) is shapes[code] for code in CODES)


def test_a_signed_code_is_a_hull_only_rule_on_the_object_path():
    with pytest.raises(InvalidRequestError, match="requires shape_type convex_hull"):
        Item.create("box", Dimensions.mm(1, 2, 3), allowed_rotations=(Rotation.NL_NW_PH,))
    assert wedge(allowed_rotations=(Rotation.NL_NW_PH,)).allowed_rotations == (Rotation.NL_NW_PH,)


def test_keep_upright_keeps_the_codes_whose_z_entry_is_plus_h():
    kept = wedge(allowed_rotations=Rotation.every(), keep_upright=True).allowed_rotations
    assert [r.value for r in kept] == list(UPRIGHT_CODES)


def test_a_box_with_keep_upright_still_gets_the_legacy_pair():
    assert Item.create("a", Dimensions.mm(1, 2, 3), keep_upright=True).allowed_rotations == Rotation.upright()


def _request(item: dict) -> dict:
    return {"items": [{"id": "a", "dimensions": {"length": "10", "width": "10", "height": "10"},
                       **item}],
            "containers": [{"id": "c", "inner_dimensions": {"length": "10", "width": "10",
                                                            "height": "10"}}]}


@pytest.mark.parametrize("item, reason, field", [
    ({"allowed_rotations": ["LWH", "-L-W+H"]}, "invalid_value", "/items/0/allowed_rotations/1"),
    ({"allowed_rotations": ["+L+W+H"]}, "not_allowed", "/items/0/allowed_rotations/0"),
    ({"shape_type": "convex_hull", "hull_vertices": [], "allowed_rotations": ["-L+W+H"]},
     "not_allowed", "/items/0/allowed_rotations/0"),
])
def test_the_json_path_names_the_code_it_refuses(item, reason, field):
    with pytest.raises(InvalidRequestError) as caught:
        check_request(_request(item))
    assert (caught.value.reason, caught.value.field) == (reason, field)


def test_the_legacy_vocabulary_is_what_a_box_is_told():
    with pytest.raises(InvalidRequestError, match=r'must be one of \["LWH","LHW","WLH","WHL","HLW","HWL"\]$'):
        check_request(_request({"allowed_rotations": ["XYZ"]}))


def test_two_chiral_wedges_share_a_crate_only_under_a_signed_code():
    config = PackingConfig(max_containers=1)
    crates = [container("crate", 10, 10, 10)]
    legacy = pack([wedge(quantity=2)], crates, config)
    assert len(legacy.unpacked) == 1
    items = [wedge(quantity=2, allowed_rotations=Rotation.every())]
    every = pack(items, crates, config)
    assert_sound(every, items, crates)
    assert not every.unpacked
    assert [p.rotation.value for p in every.containers[0].placements] == ["LWH", "-L-W+H"]


def test_a_lock_may_name_a_signed_orientation():
    assert PlacementLock(0, "wedge", "-L-W+H", (0, 0, 0)).orientation == "-L-W+H"
    with pytest.raises(ValueError):
        PlacementLock(0, "wedge", "-L+W+H", (0, 0, 0))


def _chiral(**fixed) -> dict:
    corners = [{"x": str(x), "y": str(y), "z": str(z)}
               for x, y, z in ((0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (0, 10, 10))]
    return {"items": [{"id": "wedge", "quantity": 2, "dimensions": {"length": "10", "width": "10", "height": "10"},
                       "shape_type": "convex_hull", "hull_vertices": corners,
                       "allowed_rotations": list(CODES)}],
            "containers": [{"id": "crate", "quantity": 1,
                            "inner_dimensions": {"length": "10", "width": "10", "height": "10"}}],
            "fixed_placements": [{"item_type": "wedge", "container_type": "crate", **fixed}]}


def test_a_hull_may_be_fixed_in_a_signed_orientation():
    from packvium.serialization import pack_from_dict

    result = pack_from_dict(_chiral(orientation="-L-W+H"))
    assert [(p["orientation"], p.get("fixed", False)) for p in result["containers"][0]["placements"]] == [
        ("-L-W+H", True), ("LWH", False)]


@pytest.mark.parametrize("orientation, reason, detail", [
    ("-L+W+H", "malformed", "must be one of [\"LWH\""),
    ("+L-W-H", "cannot_hold", "may not be placed in orientation +L-W-H"),
])
def test_a_fixed_hull_orientation_is_read_from_its_vocabulary_and_its_list(orientation, reason, detail):
    from packvium.fixed_placements import FixedPlacementError
    from packvium.serialization import pack_from_dict

    request = _chiral(orientation=orientation)
    request["items"][0]["allowed_rotations"] = ["LWH", "-L-W+H"]
    with pytest.raises(FixedPlacementError) as caught:
        pack_from_dict(request)
    assert caught.value.reason == reason and detail in str(caught.value)


def test_a_signed_orientation_cannot_fix_a_box_on_either_path():
    from packvium import FixedPlacement, Packer, Point
    from packvium.fixed_placements import FixedPlacementError
    from packvium.serialization import pack_from_dict

    request = _chiral(orientation="-L-W+H")
    for key in ("shape_type", "hull_vertices", "allowed_rotations"):
        del request["items"][0][key]
    with pytest.raises(FixedPlacementError) as from_json:
        pack_from_dict(request)
    assert (from_json.value.reason, from_json.value.field) == ("invalid_value", "/fixed_placements/0/orientation")
    box = Item.create("wedge", Dimensions.mm(10, 10, 10), quantity=2)
    with pytest.raises(FixedPlacementError) as from_objects:
        Packer(PackingConfig()).pack([box], [container("crate", 10, 10, 10)],
                                     [FixedPlacement("wedge", "crate", Point(0, 0, 0), Rotation.NL_NW_PH)])
    assert str(from_objects.value) == str(from_json.value)


def test_a_revision_locks_a_hull_in_a_signed_orientation_and_refuses_one_on_a_box():
    from packvium.revisions import PlanRevisionError, apply_events

    request = _chiral(orientation="LWH")
    del request["fixed_placements"]
    lock = {"sequence": 1, "type": "placement_locked",
            "placement": {"item_type": "wedge", "container_type": "crate", "orientation": "-L-W+H"}}
    assert apply_events(request, [lock])["fixed_placements"] == [lock["placement"]]
    request["items"][0]["shape_type"] = "rigid_cuboid"
    with pytest.raises(PlanRevisionError, match="placement.orientation: a signed rotation code requires"):
        apply_events(request, [lock])


def test_a_request_with_no_item_list_names_no_hull():
    from packvium.fixed_placements import hull_item_types

    assert hull_item_types(None) == frozenset()
    assert hull_item_types([{"id": "w", "shape_type": "convex_hull"}, "x", {"id": 5, "shape_type": "convex_hull"}]) == {"w"}
