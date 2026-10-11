"""The flat-bag girth rule (docs/DEFORMABLE-CONTAINERS.md "1.6 readiness and contract").

A `flat_bag` container is two sheets of usable flat size `L x W` filled to at most thickness
`H`, its `inner_dimensions`. With `a`, `b` and `c` the largest envelope `x2`, `y2` and `z2`
over its placements, measured from the container origin, a packing is admitted only when the
ordinary boundary holds and `a + c <= L` and `b + c <= W`: a loop of film round the package's
cross-section is at least as long as that section's perimeter. Exact integer additions and
comparisons, so every check here is `O(1)` and needs no wider type than a coordinate.

The extents only grow as boxes are added, so a verdict reached for a candidate never flips
when more items follow, and any subset of an admitted set is admitted.
"""

from __future__ import annotations

from typing import Iterable

from .geometry import Dimensions
from .units import Length

FLAT_BAG = "flat_bag"
DEFORMABLE_BOUNDARY_KINDS = (FLAT_BAG,)

Extent = tuple[int, int, int]


def girth_admits(inner: Dimensions, a: int, b: int, c: int) -> bool:
    """Whether a package of extents `(a, b, c)` passes both girth inequalities.

    The rigid boundary (`a <= L`, `b <= W`, `c <= H`) is the caller's own check and stays
    where it already is; only the two sums are new.
    """
    return a + c <= inner.length.ticks and b + c <= inner.width.ticks


def grown_extent(extent: Extent, x2: int, y2: int, z2: int) -> Extent:
    """The running extents once a box reaching `(x2, y2, z2)` is added."""
    return (max(extent[0], x2), max(extent[1], y2), max(extent[2], z2))


def extent_of(boxes: Iterable[tuple[int, int, int]]) -> Extent:
    """`(max x2, max y2, max z2)` over upper corners, `(0, 0, 0)` for none. O(p)."""
    extent = (0, 0, 0)
    for x2, y2, z2 in boxes:
        extent = grown_extent(extent, x2, y2, z2)
    return extent


def fits_alone(inner: Dimensions, dimensions: Dimensions) -> bool:
    """Whether one box of these dimensions, alone at the origin, fits a flat bag."""
    a, b, c = dimensions.length.ticks, dimensions.width.ticks, dimensions.height.ticks
    return dimensions.fits_inside(inner) and girth_admits(inner, a, b, c)


def as_dimensions(extent: Extent) -> Dimensions:
    """The achieved extent as dimensions; a packed bag always holds at least one box."""
    return Dimensions(Length(extent[0]), Length(extent[1]), Length(extent[2]))
