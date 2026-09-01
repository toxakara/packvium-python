from __future__ import annotations

from ._compat import dataclass
from enum import Enum
from itertools import product
from typing import Iterable

from .units import Length, Rounding, Weight


class Rotation(str, Enum):
    LWH = "LWH"
    LHW = "LHW"
    WLH = "WLH"
    WHL = "WHL"
    HLW = "HLW"
    HWL = "HWL"

    @classmethod
    def all(cls) -> tuple["Rotation", ...]:
        return tuple(cls)

    @classmethod
    def upright(cls) -> tuple["Rotation", ...]:
        return (cls.LWH, cls.WLH)


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
        return cls(Length.of(length, unit, rounding), Length.of(width, unit, rounding), Length.of(height, unit, rounding))

    @classmethod
    def mm(cls, length, width, height, rounding: Rounding = Rounding.NEAREST) -> "Dimensions":
        return cls.of(length, width, height, "mm", rounding)

    @classmethod
    def inches(cls, length, width, height, rounding: Rounding = Rounding.NEAREST) -> "Dimensions":
        return cls.of(length, width, height, "in", rounding)

    @classmethod
    def from_dict(cls, data: dict, default_unit: str = "mm", rounding: Rounding = Rounding.NEAREST) -> "Dimensions":
        return cls(
            Length.parse(data["length"], default_unit, rounding),
            Length.parse(data["width"], default_unit, rounding),
            Length.parse(data["height"], default_unit, rounding),
        )

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
        l, w, h = self.length, self.width, self.height
        values = {
            Rotation.LWH: (l, w, h), Rotation.LHW: (l, h, w),
            Rotation.WLH: (w, l, h), Rotation.WHL: (w, h, l),
            Rotation.HLW: (h, l, w), Rotation.HWL: (h, w, l),
        }[rotation]
        return Dimensions(*values)

    def unique_rotations(self, allowed: Iterable[Rotation]) -> tuple[tuple[Rotation, "Dimensions"], ...]:
        seen: set[tuple[int, int, int]] = set()
        result = []
        for rotation in allowed:
            dims = self.rotated(rotation)
            key = (dims.length.ticks, dims.width.ticks, dims.height.ticks)
            if key not in seen:
                seen.add(key)
                result.append((rotation, dims))
        return tuple(result)

    def fits_inside(self, other: "Dimensions") -> bool:
        return self.length.ticks <= other.length.ticks and self.width.ticks <= other.width.ticks and self.height.ticks <= other.height.ticks

    def expand(self, clearance: Length) -> "Dimensions":
        c = clearance.ticks * 2
        return Dimensions(Length(self.length.ticks + c), Length(self.width.ticks + c), Length(self.height.ticks + c))

    def to_dict(self, unit: str = "mm") -> dict:
        return {"length": self.length.to_dict(unit), "width": self.width.to_dict(unit), "height": self.height.to_dict(unit)}


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
