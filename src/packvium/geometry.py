from __future__ import annotations

from ._compat import dataclass
from enum import Enum
from functools import lru_cache
from itertools import product
from typing import Iterable

from .orientations import LEGACY_CODES, SOURCE_AXES, UPRIGHT_CODES
from .units import Length, Rounding, Weight
from ._request_error import refused


class Rotation(str, Enum):
    """The 24 proper rotations in canonical table order (`orientations`, 1.10).

    The six legacy codes come first and are the whole vocabulary of a box; the 18 signed codes
    after them are reachable only by a `convex_hull`. Member order is the tie-break order.
    """

    LWH = "LWH"
    LHW = "LHW"
    WLH = "WLH"
    WHL = "WHL"
    HLW = "HLW"
    HWL = "HWL"
    PL_NW_NH = "+L-W-H"
    NL_PW_NH = "-L+W-H"
    NL_NW_PH = "-L-W+H"
    PL_NH_PW = "+L-H+W"
    PL_PH_NW = "+L+H-W"
    NL_NH_NW = "-L-H-W"
    PW_NL_PH = "+W-L+H"
    PW_PL_NH = "+W+L-H"
    NW_NL_NH = "-W-L-H"
    PW_NH_NL = "+W-H-L"
    NW_PH_NL = "-W+H-L"
    NW_NH_PL = "-W-H+L"
    PH_NL_NW = "+H-L-W"
    NH_PL_NW = "-H+L-W"
    NH_NL_PW = "-H-L+W"
    PH_NW_PL = "+H-W+L"
    PH_PW_NL = "+H+W-L"
    NH_NW_NL = "-H-W-L"

    @classmethod
    def all(cls) -> tuple["Rotation", ...]:
        """The six legacy codes: an item's default list, and every box a cuboid can make."""
        return _LEGACY

    @classmethod
    def upright(cls) -> tuple["Rotation", ...]:
        """The legacy codes `keep_upright` leaves of the default list."""
        return _LEGACY_UPRIGHT

    @classmethod
    def every(cls) -> tuple["Rotation", ...]:
        """All 24 proper rotations, in table order; the 18 past `all()` need a hull."""
        return tuple(cls)

    @property
    def is_signed(self) -> bool:
        return self.value not in LEGACY_CODES

    @property
    def is_upright(self) -> bool:
        """Whether the item's own height stays vertical and pointing up (z takes `+H`)."""
        return self.value in UPRIGHT_CODES


_LEGACY: tuple[Rotation, ...] = tuple(Rotation(code) for code in LEGACY_CODES)
_LEGACY_UPRIGHT: tuple[Rotation, ...] = (Rotation.LWH, Rotation.WLH)

# Which declared side lands on each axis under a rotation. A module-level table rather
# than a dict literal inside `rotated`: that method runs once per rotation per candidate
# search, and rebuilding the tuples there was measurable. A box ignores the signs.
_ROTATION_AXES: dict[Rotation, tuple[int, int, int]] = {
    rotation: SOURCE_AXES[rotation.value] for rotation in Rotation
}


class ShapeType(str, Enum):
    """How much of an item's declared box the item actually occupies.

    `RIGID_CUBOID` is the default and the whole of the contract before this epic: the item
    is its box. The other two narrow that in one dimension each -- `CONVEX_HULL` in space,
    `COMPRESSIBLE` in height under load -- and neither may be inferred. An engine that
    packed a hull as its bounding box would return a plan that validates and does not
    physically fit, which is why the value is refused rather than approximated until the
    engine implements it.
    """

    RIGID_CUBOID = "rigid_cuboid"
    CONVEX_HULL = "convex_hull"
    COMPRESSIBLE = "compressible"


@dataclass(frozen=True, slots=True)
class Dimensions:
    length: Length
    width: Length
    height: Length

    def __post_init__(self) -> None:
        if min(self.length.ticks, self.width.ticks, self.height.ticks) <= 0:
            raise ValueError("all dimensions must be positive")

    @classmethod
    def of(cls, length, width, height, unit: str = "mm", rounding: Rounding = Rounding.NEAREST) -> "Dimensions":
        return cls._read(Length.of(length, unit, rounding), Length.of(width, unit, rounding),
                         Length.of(height, unit, rounding))

    @classmethod
    def mm(cls, length, width, height, rounding: Rounding = Rounding.NEAREST) -> "Dimensions":
        return cls.of(length, width, height, "mm", rounding)

    @classmethod
    def inches(cls, length, width, height, rounding: Rounding = Rounding.NEAREST) -> "Dimensions":
        return cls.of(length, width, height, "in", rounding)

    @classmethod
    def from_dict(cls, data: dict, default_unit: str = "mm", rounding: Rounding = Rounding.NEAREST) -> "Dimensions":
        return cls._read(
            Length.parse(data["length"], default_unit, rounding),
            Length.parse(data["width"], default_unit, rounding),
            Length.parse(data["height"], default_unit, rounding),
        )

    @classmethod
    def _read(cls, length: Length, width: Length, height: Length) -> "Dimensions":
        """Sides read from the caller's input, where a zero side is a request refusal;
        `__post_init__` still guards the arithmetic with a plain error."""
        if min(length.ticks, width.ticks, height.ticks) <= 0:
            raise refused("invalid_value", "all dimensions must be positive")
        return cls(length, width, height)

    @property
    def volume(self) -> int:
        return self.length.ticks * self.width.ticks * self.height.ticks

    @property
    def base_area(self) -> int:
        return self.length.ticks * self.width.ticks

    @property
    def max_edge(self) -> int:
        return max(self.length.ticks, self.width.ticks, self.height.ticks)

    def rotated(self, rotation: Rotation) -> "Dimensions":
        sides = (self.length, self.width, self.height)
        first, second, third = _ROTATION_AXES[rotation]
        return Dimensions(sides[first], sides[second], sides[third])

    def unique_rotations(self, allowed: Iterable[Rotation]) -> tuple[tuple[Rotation, "Dimensions"], ...]:
        return _unique_rotations(self, tuple(allowed))

    def fits_inside(self, other: "Dimensions") -> bool:
        return self.length.ticks <= other.length.ticks and self.width.ticks <= other.width.ticks and self.height.ticks <= other.height.ticks

    def expand(self, clearance: Length) -> "Dimensions":
        c = clearance.ticks * 2
        return Dimensions(Length(self.length.ticks + c), Length(self.width.ticks + c), Length(self.height.ticks + c))

    def to_dict(self, unit: str = "mm") -> dict:
        return {"length": self.length.to_dict(unit), "width": self.width.to_dict(unit), "height": self.height.to_dict(unit)}


@lru_cache(maxsize=4096)
def _unique_rotations(dimensions: Dimensions, allowed: tuple[Rotation, ...]) -> tuple[tuple[Rotation, Dimensions], ...]:
    """Memoised on the (immutable) box and rotation list: every candidate search of an
    item asks this again, and the answer is the same tuple of the same frozen values."""
    seen: set[tuple[int, int, int]] = set()
    result = []
    for rotation in allowed:
        dims = dimensions.rotated(rotation)
        key = (dims.length.ticks, dims.width.ticks, dims.height.ticks)
        if key not in seen:
            seen.add(key)
            result.append((rotation, dims))
    return tuple(result)


def dimensional_weight(dimensions: Dimensions, divisor: int, length_unit: str = "in", weight_unit: str = "lb") -> Weight:
    """The carrier-style volumetric weight of a box: (L * W * H in `length_unit`) / divisor.

    `divisor` is a carrier's published constant (commonly 139 or 166 for inches/lb,
    5000 or 6000 for cm/kg) -- it already encodes the length/weight convention, so the
    caller names both explicitly rather than the library guessing one. Exact integer
    throughout: floating volumetric-weight math would round differently in every
    language, and this number is meant to be reproduced by hand against a carrier's
    own calculator.
    """
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    length_ticks_per_unit = Length._MULTIPLIERS[length_unit.lower().strip()]
    weight_ticks_per_unit = Weight._MULTIPLIERS[weight_unit.lower().strip()]
    numerator = (dimensions.length.ticks * dimensions.width.ticks * dimensions.height.ticks) * weight_ticks_per_unit
    denominator = (length_ticks_per_unit ** 3) * divisor
    return Weight(numerator // denominator)


@dataclass(frozen=True, order=True, slots=True)
class Point:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if min(self.x, self.y, self.z) < 0:
            raise ValueError("coordinates cannot be negative")

    def to_dict(self, unit: str = "mm") -> dict:
        return {"x": Length(self.x).to_dict(unit), "y": Length(self.y).to_dict(unit), "z": Length(self.z).to_dict(unit)}


@dataclass(frozen=True, slots=True)
class AxisAlignedBox:
    origin: Point
    dimensions: Dimensions

    @property
    def x2(self) -> int: return self.origin.x + self.dimensions.length.ticks
    @property
    def y2(self) -> int: return self.origin.y + self.dimensions.width.ticks
    @property
    def z2(self) -> int: return self.origin.z + self.dimensions.height.ticks

    def intersects(self, other: "AxisAlignedBox") -> bool:
        return not (self.x2 <= other.origin.x or other.x2 <= self.origin.x or self.y2 <= other.origin.y or other.y2 <= self.origin.y or self.z2 <= other.origin.z or other.z2 <= self.origin.z)

    def contains(self, other: "AxisAlignedBox") -> bool:
        return self.origin.x <= other.origin.x and self.origin.y <= other.origin.y and self.origin.z <= other.origin.z and other.x2 <= self.x2 and other.y2 <= self.y2 and other.z2 <= self.z2

    def contains_point(self, point: Point) -> bool:
        return (self.origin.x <= point.x < self.x2 and self.origin.y <= point.y < self.y2
                and self.origin.z <= point.z < self.z2)

    def overlap_area_xy(self, other: "AxisAlignedBox") -> int:
        dx = max(0, min(self.x2, other.x2) - max(self.origin.x, other.origin.x))
        dy = max(0, min(self.y2, other.y2) - max(self.origin.y, other.origin.y))
        return dx * dy


#: The six axis-aligned faces a box can leave a container through, in a fixed order.
#: Fixed because callers iterate it to pick the *first* clear direction, and a set would
#: make which one they pick depend on hash order.
ALL_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")


class InvalidDirectionError(ValueError):
    """A direction outside the six-value vocabulary was supplied. Rejected rather than
    silently treated as one of the six -- `-z` in particular, since that was the sequence
    module's own previous (wrong) default for anything unrecognised."""

    code = "invalid_direction"

    def __init__(self, direction: str):
        self.direction = direction
        super().__init__(f"unknown movement direction {direction!r}; expected one of {ALL_DIRECTIONS}")

    def to_dict(self) -> dict:
        return {"code": self.code, "direction": self.direction}


def swept_volume(box: AxisAlignedBox, container: Dimensions,
                 direction: str) -> tuple[int, int, int, int, int, int]:
    """The region between `box`'s own face and the matching container wall along
    `direction`.

    Identical whether a box leaves through that wall (unloading) or arrives through it
    (loading), which is why one primitive serves both. It lives here rather than beside
    either caller because both the constraint layer and the sequence layer need it, and
    the sequence layer already depends on the constraint layer -- putting it there would
    invert the dependency direction the architecture fixes as one-way.
    """
    x1, y1, z1, x2, y2, z2 = box.origin.x, box.origin.y, box.origin.z, box.x2, box.y2, box.z2
    if direction == "+x":
        x1 = x2
        x2 = container.length.ticks
    elif direction == "-x":
        x2 = x1
        x1 = 0
    elif direction == "+y":
        y1 = y2
        y2 = container.width.ticks
    elif direction == "-y":
        y2 = y1
        y1 = 0
    elif direction == "+z":
        z1 = z2
        z2 = container.height.ticks
    elif direction == "-z":
        z2 = z1
        z1 = 0
    else:
        raise InvalidDirectionError(direction)
    return x1, y1, z1, x2, y2, z2


def sweep_intersects(sweep: tuple[int, int, int, int, int, int], box: AxisAlignedBox) -> bool:
    """Does `box` stand anywhere inside a swept region?

    Half-open on every axis, so boxes that merely touch a corridor's face do not block
    it -- the same convention `AxisAlignedBox.intersects` uses, and the reason a box
    flush against another's exit face is not treated as being in its way.
    """
    sx1, sy1, sz1, sx2, sy2, sz2 = sweep
    return (sx1 < box.x2 and box.origin.x < sx2
            and sy1 < box.y2 and box.origin.y < sy2
            and sz1 < box.z2 and box.origin.z < sz2)
