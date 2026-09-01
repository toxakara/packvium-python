"""Exact separating-axis collision for `convex_hull` items.

The rule is fixed by [docs/IRREGULAR-ITEMS.md](../../../docs/IRREGULAR-ITEMS.md).
`scripts/irregular_items_model.py` is the independent oracle; this module is written from
the document and never imports it, so the property tests compare two implementations rather
than one implementation with itself.

The oracle takes every vertex triple as a candidate face normal because it is optimising for
being obviously right. That is the wrong trade in a solver: the axis set is the inner loop.
This module keeps the document's licence to "precompute canonical faces and edges and use
the smaller standard axis set" and does two things with it.

*Only supporting planes survive.* A triple whose plane cuts through the hull is not a face,
and its normal cannot separate anything the real face normals do not. On a cube this turns
13 triple normals into 3. On its own that did not make the *predicate* cheaper: while edge
directions were every vertex pair, the edge-against-edge product regenerated the discarded
directions anyway, leaving 25 axes for a cube against itself. It took narrowing the edges too
-- see below -- to collapse that to 3.

*The axes belong to the shape, not to the placement.* This is where the predicate actually
gets cheap. A hull's face normals and edge directions depend on its vertices in the rotated
local frame and not at all on where the candidate sits, so `HullShape` is built once per
item-and-rotation and reused across every candidate position. Translation enters only as
`origin . axis` added to a projection interval, so no translated vertex tuple is ever
materialised.

*Edge directions are the hull's real edges.* The supporting planes are already wound into
faces to compute the volume, and the consecutive corners of a wound face are exactly the
polyhedron's edges -- so the narrower set the separating-axis theorem actually asks for costs
nothing beyond the walk that was happening anyway. Taking every vertex pair instead was never
wrong, only a superset, and an expensive one: `3v - 6` edges against `v(v - 1) / 2` pairs, in
a predicate whose axis set is the *product* of the two hulls' sets, so the gap squares. On a
20-vertex hull that is 1351 candidate axes rather than 15616.

Complexity. Building a shape is `O(v^4)`: `O(v^3)` triples, each checked against `O(v)`
vertices. It is paid once per item and rotation and no longer once per collision test --
`shape_for` memoises it, which is what turned a two-item hull request from 78 shape builds
into 4. The predicate is `O((f_a + f_b + e_a * e_b) * (a + b))` for `f` face axes and `e` edge
directions, inside the document's `O(a^2 * b^2)` bound and reached only after the axis-aligned
broad phase has already said the envelopes overlap.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import gcd
from typing import Iterable, Sequence, Tuple

from ._compat import dataclass

#: A point in an item's local tick frame. Deliberately not `geometry.Point`, which forbids
#: negative coordinates: a hull is authored around whatever origin its author chose, and is
#: only moved into container coordinates at placement time.
Vertex = Tuple[int, int, int]
Axis = Tuple[int, int, int]

#: Smallest number of vertices that can enclose a volume.
MINIMUM_HULL_VERTICES = 4

#: Largest absolute vertex coordinate a hull may carry, in ticks (6.25 m).
#:
#: Shared verbatim with PHP, Rust and JavaScript. Besides keeping admission identical across
#: engines, the bound keeps their exact cross products and projections inside the arithmetic
#: ranges documented in ``docs/IRREGULAR-ITEMS.md``.
MAX_COORDINATE = 100_000_000

#: Sorted, canonical, and both the face normals and the edge directions of any axis-aligned
#: box -- which is what makes `HullShape.box` constant time.
UNIT_AXES: Tuple[Axis, ...] = ((0, 0, 1), (0, 1, 0), (1, 0, 0))


class DegenerateHullError(ValueError):
    """Hull vertices do not enclose a three-dimensional volume.

    Rejected rather than repaired. A flat or duplicated vertex set has no interior, so every
    separating-axis answer about it would be vacuously "no collision" -- an item that passes
    through everything, which reads as a successful pack.
    """


def _subtract(left: Vertex, right: Vertex) -> Vertex:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross(left: Vertex, right: Vertex) -> Axis:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(point: Vertex, axis: Axis) -> int:
    return point[0] * axis[0] + point[1] * axis[1] + point[2] * axis[2]


def _reduce(axis: Axis) -> Axis:
    """Divide out the gcd and fix the sign, so parallel axes collapse to one entry.

    A canonical direction is what makes the axis set a set. Sign is irrelevant to separation
    -- negating an axis mirrors both projection intervals -- so the sign of the first
    non-zero component is fixed positive and opposite normals stop being counted twice.

    The caller guarantees a non-zero vector; `_primitive` is the variant that decides.
    """
    divisor = gcd(gcd(abs(axis[0]), abs(axis[1])), abs(axis[2]))
    reduced = (axis[0] // divisor, axis[1] // divisor, axis[2] // divisor)
    leading = next(value for value in reduced if value)
    return reduced if leading > 0 else (-reduced[0], -reduced[1], -reduced[2])


def _primitive(axis: Axis) -> "Axis | None":
    """`_reduce`, plus the one case that legitimately has no direction.

    A cross product of two parallel directions is the zero vector and names no axis. That
    happens constantly among face candidates and edge pairs, so it is an ordinary outcome
    here rather than an error.
    """
    return None if axis == (0, 0, 0) else _reduce(axis)


def _is_supporting(vertices: Sequence[Vertex], origin: Vertex, axis: Axis) -> bool:
    """Does the plane through `origin` with normal `axis` leave every vertex on one side?"""
    seen_above = seen_below = False
    offset = _dot(origin, axis)
    for vertex in vertices:
        side = _dot(vertex, axis) - offset
        if side > 0:
            seen_above = True
        elif side < 0:
            seen_below = True
        if seen_above and seen_below:
            return False
    return True


def _ordered_face(face: Sequence[Vertex], outward: Axis) -> Tuple[Vertex, ...]:
    """Corners of one planar convex face, in cyclic order seen from outside.

    The vertices at a supporting plane are *not* all corners of the polygon they lie on: one
    can sit inside the face, or part-way along one of its edges. Fanning over the raw extreme
    set therefore triangulates the wrong region, and the resulting surface does not close --
    which is exactly how this was found, by a closed-surface residual on a random hull rather
    than by any test of the volume itself.

    So the face is gift-wrapped, which keeps only the corners. Starting from the
    lexicographically smallest vertex -- extreme in any linear order, therefore a corner --
    each step takes the vertex that leaves every other on one side. Collinear candidates
    resolve to the farthest, which is what skips a vertex lying on an edge instead of
    doubling back through it.

    Only the *sign* of a turn is ever needed, never its size, and the winding is consistent
    with `outward` for every face, so the signed volumes below add rather than cancel.
    """
    start = min(face)
    ordered = [start]
    current = start
    for _ in range(len(face)):
        following: "Vertex | None" = None
        for candidate in face:
            if candidate == current:
                continue
            if following is None:
                following = candidate
                continue
            turn = _dot(
                _cross(_subtract(following, current), _subtract(candidate, current)), outward
            )
            if turn < 0 or (turn == 0 and _square_length(_subtract(candidate, current))
                            > _square_length(_subtract(following, current))):
                following = candidate
        if following is None or following == start:
            break
        ordered.append(following)
        current = following
    return tuple(ordered)


def _square_length(vector: Vertex) -> int:
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]


def _wound_faces(
    vertices: Sequence[Vertex], face_axes: Sequence[Axis]
) -> Tuple[Tuple[Vertex, ...], ...]:
    """Every face of the hull, each as its own corners in outward cyclic order.

    One walk, because the faces answer two questions at once. The volume needs them wound
    consistently; the edge directions are the consecutive corner pairs of the same walk. They
    were computed separately before -- the volume from here, the edges from every vertex pair
    -- which made the edge set quadratically larger than the hull actually has.

    Each canonical axis stands for up to two opposite faces, so both the maximal and the
    minimal supporting plane along it are collected. A plane carrying fewer than three
    vertices is an edge or a corner of the hull, not a face; it contributes no area and no
    edge that its two adjoining faces do not already carry.
    """
    faces = []
    for axis in face_axes:
        for outward in (axis, (-axis[0], -axis[1], -axis[2])):
            extreme = max(_dot(vertex, outward) for vertex in vertices)
            face = [vertex for vertex in vertices if _dot(vertex, outward) == extreme]
            if len(face) < 3:
                continue
            faces.append(_ordered_face(face, outward))
    return tuple(faces)


def _volume_of(faces: Sequence[Sequence[Vertex]]) -> int:
    """Exact volume in cubic ticks, by the divergence theorem over the hull's own faces.

    `6V = sum over outward-oriented surface triangles of a . (b x c)`, which is an integer for
    integer vertices and therefore exact -- no tolerance decides whether a wedge is half a
    cube.
    """
    six_volumes = 0
    for ordered in faces:
        apex = ordered[0]
        for second, third in zip(ordered[1:], ordered[2:]):
            six_volumes += _dot(apex, _cross(second, third))
    return abs(six_volumes) // 6


def _edge_directions(faces: Sequence[Sequence[Vertex]]) -> Tuple[Axis, ...]:
    """Directions of the hull's real edges, deduplicated and canonical.

    Every edge of a convex polyhedron is shared by exactly two faces, so walking each wound
    face and taking its consecutive corner pairs -- closing the cycle -- reaches all of them.
    The separating-axis theorem asks for exactly these: cross products of true edge
    directions, not of every vertex pair.

    The distinction is the whole cost of the predicate. A hull has at most `3v - 6` edges but
    `v(v - 1) / 2` vertex pairs, and the axis set is the *product* of the two hulls' sets, so
    the gap squares. Measured on a 20-vertex hull: 190 pair directions against 54 real edge
    directions, and 15616 candidate axes against 1351 -- the same verdict for a twelfth of the
    work. Vertex pairs were never wrong, only a superset: a pair that is not an edge names a
    direction no face can separate along, so it can add an axis but never remove one.
    """
    edges = set()
    for ordered in faces:
        for index, start in enumerate(ordered):
            end = ordered[(index + 1) % len(ordered)]
            edges.add(_reduce(_subtract(end, start)))
    return tuple(sorted(edges))


@dataclass(frozen=True, slots=True)
class HullShape:
    """A convex hull's separating axes in its own local frame, computed once.

    Immutable like every other value object here, and safe to share between candidates: it
    describes the shape, and a placement contributes only an offset.
    """

    vertices: Tuple[Vertex, ...]
    face_axes: Tuple[Axis, ...]
    edge_directions: Tuple[Axis, ...]
    #: Exact occupied volume in cubic ticks. Computed once with the axes, because a hull's
    #: volume is a property of the shape and utilisation would otherwise be reported from the
    #: bounding box -- two interlocking wedges in one crate reading as 200% full.
    volume: int

    @classmethod
    def of(cls, vertices: Iterable[Vertex]) -> "HullShape":
        points = validate(vertices)
        normals = set()
        for first, second, third in combinations(points, 3):
            axis = _primitive(_cross(_subtract(second, first), _subtract(third, first)))
            if axis is not None and _is_supporting(points, first, axis):
                normals.add(axis)
        face_axes = tuple(sorted(normals))
        faces = _wound_faces(points, face_axes)
        return cls(points, face_axes, _edge_directions(faces), _volume_of(faces))

    @classmethod
    def box(cls, length: int, width: int, height: int) -> "HullShape":
        """A cuboid, built without searching for its own faces.

        A box's face normals and edge directions are both exactly the three unit axes, so the
        `O(v^4)` supporting-plane search would spend 56 triple tests rediscovering what is
        already known. Naming them directly is what lets a hull be tested against an ordinary
        item without a cache anywhere: building this is `O(1)`.

        No positivity guard: every extent reaching here comes from a `Dimensions`, which
        already refuses a non-positive side at construction. A second copy of that invariant
        would be unreachable code claiming to protect something.
        """
        vertices = tuple(
            (x * length, y * width, z * height)
            for x in (0, 1) for y in (0, 1) for z in (0, 1)
        )
        return cls(vertices, UNIT_AXES, UNIT_AXES, length * width * height)

    def projection(self, axis: Axis) -> Tuple[int, int]:
        """Closed projection interval on `axis`, in local coordinates."""
        values = [_dot(vertex, axis) for vertex in self.vertices]
        return min(values), max(values)


#: How many rotated hulls stay resident. A request is bounded by its distinct hull items times
#: the six orientations, so this holds far more than any single request; the bound is what
#: stops a long-lived process packing many catalogues from accumulating shapes forever.
#:
#: Measured full, the ceiling it buys: 3.8 MB at six vertices per hull, 5.2 MB at eight, 13.6
#: MB at twenty. Eviction order depends on call order and nothing observable does -- the
#: function is pure, so a hit and a miss return equal shapes.
SHAPE_CACHE_ENTRIES = 1024


@lru_cache(maxsize=SHAPE_CACHE_ENTRIES)
def shape_for(vertices: Tuple[Vertex, ...], rotation: str) -> HullShape:
    """The rotated hull of one item in one orientation, built at most once.

    A hull depends on the item and the orientation and on nothing about where a candidate
    sits, but `Placement.hull_shape` was recomputing it inside the collision predicate --
    which is `O(v^4)` work in an `O(n^2)` loop. Measured on the two-wedge fixture: 78 builds
    for two items, where four are needed.

    Memoisation is safe here in the way it is not in general: `HullShape` is frozen, the
    inputs are the whole of what determines the output, and the shape is read through
    projections that never mutate it. Determinism is untouched -- this changes how often the
    answer is computed, never what it is.
    """
    return HullShape.of(rotate(vertices, rotation))


def validate(vertices: Iterable[Vertex]) -> Tuple[Vertex, ...]:
    """Canonicalise an authored vertex list or refuse a hull with no interior."""
    points = tuple(vertices)
    if len(points) < MINIMUM_HULL_VERTICES:
        raise DegenerateHullError(
            f"a convex hull needs at least {MINIMUM_HULL_VERTICES} vertices, got {len(points)}"
        )
    if len(set(points)) != len(points):
        raise DegenerateHullError("convex hull vertices must be unique")
    if any(abs(component) > MAX_COORDINATE for point in points for component in point):
        raise DegenerateHullError(
            f"convex hull coordinates must stay within {MAX_COORDINATE} ticks"
        )
    for first, second, third, fourth in combinations(points, 4):
        volume = _dot(
            _subtract(fourth, first),
            _cross(_subtract(second, first), _subtract(third, first)),
        )
        if volume != 0:
            return points
    raise DegenerateHullError("convex hull vertices are coplanar and enclose no volume")


def separating_axes(left: HullShape, right: HullShape) -> Tuple[Axis, ...]:
    """Both hulls' face normals plus every edge-against-edge direction, deduplicated."""
    axes = set(left.face_axes)
    axes.update(right.face_axes)
    for left_edge in left.edge_directions:
        for right_edge in right.edge_directions:
            axis = _primitive(_cross(left_edge, right_edge))
            if axis is not None:
                axes.add(axis)
    return tuple(sorted(axes))


def collide(left: HullShape, left_origin: Vertex, right: HullShape, right_origin: Vertex) -> bool:
    """Do two placed hulls overlap with positive volume?

    Touching is contact, not collision: the comparison is `<=`, which keeps hulls consistent
    with the half-open convention `AxisAlignedBox.intersects` already uses for cuboids, so a
    hull resting exactly on a box is supported rather than colliding with it.
    """
    for axis in separating_axes(left, right):
        left_low, left_high = left.projection(axis)
        right_low, right_high = right.projection(axis)
        left_shift = _dot(left_origin, axis)
        right_shift = _dot(right_origin, axis)
        if (left_high + left_shift <= right_low + right_shift
                or right_high + right_shift <= left_low + left_shift):
            return False
    return True


#: Which local axis each container axis takes its extent from, per `Rotation`. Read off
#: `Dimensions.rotated`, which is the definition every other part of the engine already
#: follows: `LHW` means x spans the item's length, y its height, z its width.
_SOURCE_AXES = {
    "LWH": (0, 1, 2), "LHW": (0, 2, 1),
    "WLH": (1, 0, 2), "WHL": (1, 2, 0),
    "HLW": (2, 0, 1), "HWL": (2, 1, 0),
}


def _permutation_is_odd(axes: Tuple[int, int, int]) -> bool:
    return sum(
        1 for first in range(3) for second in range(first + 1, 3) if axes[first] > axes[second]
    ) % 2 == 1


def rotate(vertices: Sequence[Vertex], rotation) -> Tuple[Vertex, ...]:
    """Reorient a hull the way `Dimensions.rotated` reorients its box.

    Three of the six rotations are odd permutations of the coordinate axes. On a cuboid that
    is invisible -- a mirrored box is the same box. On a hull it is not: a bare permutation
    would hand back the item's mirror image, a shape the caller does not own. So the sign of
    one axis is flipped whenever the permutation is odd, which makes every one of the six a
    proper rotation with determinant +1 and never a reflection.

    The scope limit that follows is worth stating rather than discovering: `allowed_rotations`
    distinguishes six orientations because six is all a box has, and a hull has 24. These six
    are pinned, deterministic and physically real; the other 18 are simply not expressible in
    today's vocabulary, and widening it is a contract change rather than a fix here.

    Vertices come back translated so the rotated hull's own bounding box starts at the origin,
    which is the frame every placement position is expressed in.
    """
    axes = _SOURCE_AXES[rotation.value if hasattr(rotation, "value") else rotation]
    sign = -1 if _permutation_is_odd(axes) else 1
    turned = [
        (sign * vertex[axes[0]], vertex[axes[1]], vertex[axes[2]]) for vertex in vertices
    ]
    lower, _ = bounding_extent(turned)
    return tuple(
        (x - lower[0], y - lower[1], z - lower[2]) for x, y, z in turned
    )


def bounding_extent(vertices: Sequence[Vertex]) -> Tuple[Vertex, Vertex]:
    """Inclusive lower and upper corners of a hull's axis-aligned envelope.

    The broad phase stays mandatory, so every hull still needs the box that encloses it.
    """
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    zs = [vertex[2] for vertex in vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
