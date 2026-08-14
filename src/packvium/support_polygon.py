from __future__ import annotations

from typing import Sequence

from .geometry import AxisAlignedBox

Point2D = tuple[int, int]


def _cross(o: Point2D, a: Point2D, b: Point2D) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Sequence[Point2D]) -> list[Point2D]:
    """Andrew's monotone chain. Exact integer cross products, no trig or division.

    Returns the hull counter-clockwise, duplicate points collapsed. Degenerate input
    (0, 1, or all-collinear points) returns that same degenerate shape rather than
    raising -- a single contact point or a razor-thin support strip is a real, if
    precarious, physical case, not an error.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    lower: list[Point2D] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point2D] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull if hull else pts


def _on_segment(p: Point2D, a: Point2D, b: Point2D) -> bool:
    if _cross(a, b, p) != 0:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def point_in_hull(point: Point2D, hull: Sequence[Point2D]) -> bool:
    """Whether `point` lies inside or on the boundary of a hull from `convex_hull`."""
    if len(hull) == 0:
        return False
    if len(hull) == 1:
        return point == hull[0]
    if len(hull) == 2:
        return _on_segment(point, hull[0], hull[1])
    return all(_cross(hull[i], hull[(i + 1) % len(hull)], point) >= 0 for i in range(len(hull)))


def contact_hull_points(candidate: AxisAlignedBox, supporters: Sequence[AxisAlignedBox]) -> list[Point2D]:
    """Doubled-tick corners of every candidate/supporter overlap rectangle.

    Doubled so a centroid computed as `x1 + x2` (see `doubled_centroid`) is always an
    exact integer even when a box's length or width is odd -- the usual trick for
    keeping a midpoint computation free of fractions.
    """
    cx1, cy1, cx2, cy2 = candidate.origin.x, candidate.origin.y, candidate.x2, candidate.y2
    points: list[Point2D] = []
    for box in supporters:
        ox1, oy1 = max(cx1, box.origin.x), max(cy1, box.origin.y)
        ox2, oy2 = min(cx2, box.x2), min(cy2, box.y2)
        if ox2 > ox1 and oy2 > oy1:
            for x in (ox1, ox2):
                for y in (oy1, oy2):
                    points.append((2 * x, 2 * y))
    return points


def doubled_centroid(box: AxisAlignedBox) -> Point2D:
    """Twice the box's own footprint centroid, exact for odd as well as even extents."""
    return (box.origin.x + box.x2, box.origin.y + box.y2)
