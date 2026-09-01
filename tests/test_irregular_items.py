"""The worked boundaries docs/IRREGULAR-ITEMS.md pins, asserted on the engine.

These are the numbers the document commits to in prose. Pinning them here means a later
optimisation of the axis set or the compression arithmetic has to keep answering the
published examples, and it means PHP, Rust and JavaScript ( through ) have a
concrete target rather than a paragraph to interpret.

The cross-implementation property tests live in `conformance/tests/test_irregular_items.py`,
where the independent oracle is.
"""

from __future__ import annotations

import pytest

from packvium.compression import (PPM, CrushViolation, Pressure, applied_pressure,
                                  effective_height_ticks, effective_volume_ticks3,
                                  ratio_to_ppm)
from packvium.hull import MAX_COORDINATE, DegenerateHullError, HullShape, collide, validate
from packvium.geometry import Dimensions, ShapeType
from packvium.models import Item
from packvium.units import Length, Weight


def cube(side: int) -> HullShape:
    return HullShape.of(
        (x * side, y * side, z * side)
        for x in (0, 1) for y in (0, 1) for z in (0, 1)
    )


TETRAHEDRON = ((0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10))


# ------------------------------------------------------------------ hull admission

def test_a_hull_needs_four_vertices():
    with pytest.raises(DegenerateHullError, match="at least 4 vertices"):
        validate(((0, 0, 0), (1, 0, 0), (0, 1, 0)))


def test_a_hull_may_not_repeat_a_vertex():
    """Four vertices of which two coincide describe a triangle, not a solid, and the
    coplanarity test below would pass it by accident on some inputs."""
    with pytest.raises(DegenerateHullError, match="unique"):
        validate(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)))


def test_a_flat_hull_is_refused_rather_than_packed():
    """The failure this guards is not a crash. A zero-volume hull is separated on its own
    normal from everything, so it would pass through every other item and still be
    reported as a valid placement."""
    with pytest.raises(DegenerateHullError, match="coplanar"):
        validate(((0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0)))


def test_a_hull_authored_around_a_negative_origin_is_admitted():
    """Local frames are the author's choice; only placement moves a hull into container
    coordinates, where the non-negative rule applies."""
    assert HullShape.of(((-5, -5, -5), (5, -5, -5), (-5, 5, -5), (-5, -5, 5))).vertices


def test_a_hull_coordinate_must_stay_inside_the_shared_exact_arithmetic_bound():
    """Python used to accept this while PHP, Rust and JavaScript refused it.

    The cap is a cross-engine input contract, not a limitation of Python's unbounded integer,
    so accepting a wider domain here would make results depend on which language binding answered.
    Both signs are checked because direct library callers may author a centred local frame.
    """
    for outside in (MAX_COORDINATE + 1, -MAX_COORDINATE - 1):
        with pytest.raises(DegenerateHullError, match="within 100000000 ticks"):
            validate(((0, 0, 0), (outside, 0, 0), (0, 1, 0), (0, 0, 1)))


# ------------------------------------------------------------------ the worked SAT boundary

@pytest.mark.parametrize("offset,expected", [(9, True), (10, False), (11, False)])
def test_the_documented_ten_tick_cube_boundary(offset, expected):
    """Offset 9 overlaps, 10 touches, 11 is clear -- touching is contact, not collision,
    which is what lets a hull rest on a surface instead of colliding with it."""
    unit = cube(10)
    assert collide(unit, (0, 0, 0), unit, (offset, 0, 0)) is expected


def test_a_tetrahedron_misses_a_cube_that_its_bounding_box_overlaps():
    """The case the whole SAT path exists for. The tetrahedron's envelope covers the cube's
    corner, so the broad phase says maybe; the diagonal face says no."""
    tetra = HullShape.of(TETRAHEDRON)
    unit = cube(4)
    assert tetra.projection((1, 1, 1))[1] == 10
    assert collide(tetra, (0, 0, 0), unit, (7, 7, 7)) is False


def test_collision_does_not_depend_on_argument_order():
    tetra = HullShape.of(TETRAHEDRON)
    unit = cube(6)
    assert (collide(tetra, (0, 0, 0), unit, (3, 3, 0))
            == collide(unit, (3, 3, 0), tetra, (0, 0, 0)))


def test_face_axes_exclude_planes_that_cut_through_the_hull():
    """The optimisation that separates this module from the oracle. A cube has three
    distinct face normals once opposite faces collapse onto one canonical direction; every
    other vertex triple spans a plane that slices the solid."""
    assert cube(10).face_axes == ((0, 0, 1), (0, 1, 0), (1, 0, 0))


# ------------------------------------------------------------------ pressure and compression

def test_the_documented_compression_examples():
    height, ratio, limit = 100, 250_000, 100
    assert effective_height_ticks(height, ratio, limit, Pressure.zero()) == 100
    assert effective_height_ticks(height, ratio, limit, Pressure(50, 1)) == 88
    assert effective_height_ticks(height, ratio, limit, Pressure(100, 1)) == 75


def test_the_crush_boundary_is_inclusive_and_the_next_step_is_a_violation():
    """`100.000001 kPa` in the document, expressed exactly: one part in a million above the
    limit. A float would have to round this somewhere, which is why the contract is a
    rational."""
    assert effective_height_ticks(100, 250_000, 100, Pressure(100, 1)) == 75
    with pytest.raises(CrushViolation, match="exceeds the declared limit"):
        effective_height_ticks(100, 250_000, 100, Pressure(100_000_001, 1_000_000))


def test_a_zero_limit_admits_only_zero_pressure():
    assert effective_height_ticks(40, PPM, 0, Pressure.zero()) == 40
    with pytest.raises(CrushViolation):
        effective_height_ticks(40, PPM, 0, Pressure(1, 1_000_000))


def test_a_fully_compressible_item_still_occupies_one_tick():
    """Zero height would let an item escape collision and support invariants rather than
    merely occupy very little, so the floor is part of the contract, not a rounding guard."""
    assert effective_height_ticks(100, PPM, 10, Pressure(10, 1)) == 1


def test_compression_never_claims_less_space_than_the_continuous_model():
    """87.5 rounds to 88, not 87: a discrete packer rounds occupied space up."""
    assert effective_height_ticks(100, 250_000, 100, Pressure(50, 1)) == 88


def test_only_the_height_compresses():
    volume = effective_volume_ticks3(20, 30, 100, 250_000, 100, Pressure(100, 1))
    assert volume == 20 * 30 * 75


def test_pressure_is_exact_under_standard_gravity():
    """One kilogram over one square metre is 9.80665 Pa, so 980665/100000000 kPa, which
    reduces to 196133/20000000. Held reduced, so two engines that agree on the value cannot
    disagree on the representation."""
    metre_ticks = Length.TICKS_PER_MM * 1_000
    pressure = applied_pressure(Weight(Weight.TICKS_PER_KG), metre_ticks * metre_ticks)
    assert (pressure.numerator, pressure.denominator) == (196_133, 20_000_000)


def test_pressure_compares_without_leaving_the_integers():
    assert Pressure(100_000_001, 1_000_000).exceeds_kpa(100)
    assert not Pressure(100_000_000, 1_000_000).exceeds_kpa(100)


def test_a_negative_load_is_an_error_rather_than_a_lifted_item():
    with pytest.raises(ValueError, match="cannot be negative"):
        Pressure(-1, 1)


def test_the_public_ratio_rule_is_applied_once_at_the_boundary():
    assert ratio_to_ppm(0.25) == 250_000
    assert ratio_to_ppm(0.0) == 0
    assert ratio_to_ppm(1.0) == PPM
    with pytest.raises(ValueError, match="between zero and one"):
        ratio_to_ppm(1.5)


# ------------------------------------------------------------------ item admission

def wedge_item(**kwargs) -> Item:
    """A hull that fills half its declared box: the case an AABB would over-claim."""
    return Item.create(
        "wedge", Dimensions.mm(10, 10, 10), shape_type=ShapeType.CONVEX_HULL,
        hull_vertices=((0, 0, 0), (160_000, 0, 0), (0, 160_000, 0), (0, 0, 160_000)),
        **kwargs,
    )


def cushion_item(**kwargs) -> Item:
    return Item.create(
        "cushion", Dimensions.mm(10, 10, 10), shape_type=ShapeType.COMPRESSIBLE,
        compression_ratio_ppm=250_000, max_compression_pressure_kpa=100, **kwargs,
    )


def test_an_item_is_a_rigid_cuboid_unless_it_says_otherwise():
    """The default is what keeps every existing caller byte-identical."""
    item = Item.create("plain", Dimensions.mm(10, 10, 10))
    assert item.shape_type is ShapeType.RIGID_CUBOID
    assert item.hull_vertices is None
    assert item.compression_ratio_ppm is None


def test_the_wire_spelling_of_a_shape_is_accepted_and_normalised():
    assert wedge_item().shape_type is ShapeType.CONVEX_HULL
    assert Item.create("plain", Dimensions.mm(1, 1, 1), shape_type="rigid_cuboid").shape_type \
        is ShapeType.RIGID_CUBOID


@pytest.mark.parametrize("field,value", [
    ("hull_vertices", ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ("compression_ratio_ppm", 250_000),
    ("max_compression_pressure_kpa", 100),
])
def test_a_rigid_cuboid_refuses_data_belonging_to_another_shape(field, value):
    """Refused, not ignored: a dropped `compression_ratio` reads back as an item packed to
    its declared limits when nothing ever applied them."""
    with pytest.raises(ValueError, match="not part of a rigid_cuboid item"):
        Item.create("plain", Dimensions.mm(10, 10, 10), **{field: value})


def test_a_hull_item_refuses_compression_data():
    with pytest.raises(ValueError, match="not part of a convex_hull item"):
        wedge_item(compression_ratio_ppm=250_000)


def test_a_compressible_item_refuses_hull_vertices():
    with pytest.raises(ValueError, match="not part of a compressible item"):
        cushion_item(hull_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_a_hull_item_without_vertices_is_refused():
    with pytest.raises(ValueError, match="requires hull_vertices"):
        Item.create("wedge", Dimensions.mm(10, 10, 10), shape_type=ShapeType.CONVEX_HULL)


def test_a_compressible_item_needs_both_of_its_numbers():
    with pytest.raises(ValueError, match="requires both"):
        Item.create("cushion", Dimensions.mm(10, 10, 10),
                    shape_type=ShapeType.COMPRESSIBLE, compression_ratio_ppm=250_000)


def test_a_hull_that_pokes_out_of_its_declared_box_is_refused():
    """`dimensions` stays the broad phase and the candidate envelope, so a hull larger than
    it would be collision-tested against space the solver never reserved."""
    with pytest.raises(ValueError, match="does not fit inside dimensions"):
        Item.create(
            "spike", Dimensions.mm(10, 10, 10), shape_type=ShapeType.CONVEX_HULL,
            hull_vertices=((0, 0, 0), (160_001, 0, 0), (0, 160_000, 0), (0, 0, 160_000)),
        )


def test_a_hull_exactly_filling_its_box_is_admitted():
    assert Item.create(
        "block", Dimensions.mm(10, 10, 10), shape_type=ShapeType.CONVEX_HULL,
        hull_vertices=tuple(
            (x * 160_000, y * 160_000, z * 160_000)
            for x in (0, 1) for y in (0, 1) for z in (0, 1)
        ),
    ).hull_vertices


@pytest.mark.parametrize("build", [wedge_item, cushion_item])
def test_nesting_stays_unsupported_against_either_new_shape(build):
    """Both rewrite occupied height. Choosing an order quietly would give four engines four
    contracts, so the interaction is refused until a task defines it."""
    with pytest.raises(ValueError, match="not supported yet"):
        build(nesting_height=Length.mm(2))


# ------------------------------------------------------------------ refusals on bad input

@pytest.mark.parametrize("call,message", [
    (lambda: Pressure(1, 0), "denominator must be positive"),
    (lambda: Pressure.reduced(1, -2), "denominator must be positive"),
    (lambda: applied_pressure(Weight(1), 0), "footprint area must be positive"),
    (lambda: effective_height_ticks(0, 0, 1, Pressure.zero()), "height must be positive"),
    (lambda: effective_height_ticks(1, PPM + 1, 1, Pressure.zero()), "one million ppm"),
    (lambda: effective_height_ticks(1, 0, -1, Pressure.zero()), "cannot be negative"),
    (lambda: effective_volume_ticks3(0, 1, 1, 0, 1, Pressure.zero()), "footprint dimensions"),
])
def test_the_arithmetic_refuses_input_it_cannot_answer_for(call, message):
    """Each of these would otherwise divide by zero, loop past a bound, or return a number
    with no meaning. They are cheap to state and each one is a silent wrong answer avoided."""
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize("field,value,message", [
    ("compression_ratio_ppm", -1, "between zero and one"),
    ("compression_ratio_ppm", PPM + 1, "between zero and one"),
    ("max_compression_pressure_kpa", -1, "cannot be negative"),
])
def test_a_compressible_item_refuses_a_number_outside_its_range(field, value, message):
    numbers = {"compression_ratio_ppm": 250_000, "max_compression_pressure_kpa": 100}
    numbers[field] = value
    with pytest.raises(ValueError, match=message):
        Item.create("cushion", Dimensions.mm(10, 10, 10),
                    shape_type=ShapeType.COMPRESSIBLE, **numbers)


def test_the_shape_memo_returns_what_a_fresh_build_would():
    """The memo may change how often a shape is built and never what it is.

    Asserted rather than assumed: a cache is the classic place for a determinism regression to
    hide, because a wrong entry is invisible on the first call and only shows on the second.
    Every rotation is asked for by name, because a memo keyed on the vertices alone would pass
    any test that packs one orientation and hand a hull its neighbour's shape on the second.
    """
    from packvium.hull import HullShape, rotate, shape_for
    from packvium.models import Rotation

    vertices = ((0, 0, 0), (12, 0, 0), (0, 9, 0), (0, 0, 7), (12, 9, 0), (4, 3, 7))
    for rotation in Rotation:
        fresh = HullShape.of(rotate(vertices, rotation.value))
        first = shape_for(vertices, rotation.value)
        assert first == fresh
        assert shape_for(vertices, rotation.value) is first


def test_the_memo_is_bounded_rather_than_growing_for_the_life_of_the_process():
    """Memory is part of the contract too: the memo drops rather than accumulating.

    A cache that never evicts turns a long-lived process packing many distinct catalogues into
    a slow leak -- which would be trading one resource for another rather than saving anything.
    """
    from packvium.hull import SHAPE_CACHE_ENTRIES, shape_for

    shape_for.cache_clear()
    for offset in range(SHAPE_CACHE_ENTRIES + 50):
        shape_for(((0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10 + offset)), "LWH")
    assert shape_for.cache_info().currsize <= SHAPE_CACHE_ENTRIES


def test_a_face_carrying_a_non_corner_vertex_is_wound_past_it():
    """A vertex sitting part-way along a face's edge must be walked past, not doubled back
    through.

    This is the case the gift-wrap's collinear tie-break exists for: among candidates that
    leave every other vertex on one side, it takes the farthest, which skips an edge-interior
    point instead of turning the face into a degenerate spur. Getting it wrong does not raise
    -- it produces a surface that fails to close and a volume that is quietly too small, which
    is how the defect fixed in  survived a review cycle.

    A pyramid on a square base, with a redundant vertex at the midpoint of one base edge. The
    base is five coplanar points and only four of them are corners; the volume is a third of
    the enclosing box either way, so the number is decided by whether the walk skipped it.
    """
    from packvium.hull import HullShape, _cross, _dot, _ordered_face, _subtract

    shape = HullShape.of(((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (5, 0, 0), (5, 5, 10)))
    assert shape.volume == 10 * 10 * 10 // 3

    # The base face wound as four corners, with the midpoint dropped.
    downward = (0, 0, -1)
    extreme = max(_dot(vertex, downward) for vertex in shape.vertices)
    base = [vertex for vertex in shape.vertices if _dot(vertex, downward) == extreme]
    assert len(base) == 5, "the midpoint is still a vertex of the hull"
    assert len(_ordered_face(base, downward)) == 4, "but it is not a corner of the face"

    # And the surface closes, which is the property the volume actually rests on.
    residual = [0, 0, 0]
    for axis in shape.face_axes:
        for outward in (axis, (-axis[0], -axis[1], -axis[2])):
            reach = max(_dot(vertex, outward) for vertex in shape.vertices)
            face = [vertex for vertex in shape.vertices if _dot(vertex, outward) == reach]
            if len(face) < 3:
                continue
            ordered = _ordered_face(face, outward)
            apex = ordered[0]
            for second, third in zip(ordered[1:], ordered[2:]):
                normal = _cross(_subtract(second, apex), _subtract(third, apex))
                residual = [residual[i] + normal[i] for i in range(3)]
    assert residual == [0, 0, 0]
