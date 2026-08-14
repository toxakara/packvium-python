from __future__ import annotations

from ._compat import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import re
from typing import Any


class Rounding(str, Enum):
    FLOOR = "floor"
    CEIL = "ceil"
    NEAREST = "nearest"


def _fraction(value: int | str | Decimal | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, Decimal):
        return Fraction(value)
    text = str(value).strip().replace("\u00a0", " ")
    mixed = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", text)
    if mixed:
        whole, numerator, denominator = map(int, mixed.groups())
        # The sign comes from the written text, not from the value of the whole part, so
        # "-0 1/2" is minus one half rather than plus one half.
        magnitude = abs(whole) * denominator + numerator
        return Fraction(-magnitude if text.startswith("-") else magnitude, denominator)
    simple = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", text)
    if simple:
        return Fraction(int(simple.group(1)), int(simple.group(2)))
    decimal = re.fullmatch(r"[+-]?\d+(?:\.(\d+))?", text)
    # A shared cap, not a Python limitation: PHP cannot hold 10**19 in an integer, and a
    # silently different accepted range between the two is exactly the kind of divergence
    # this library exists to avoid. Eighteen digits is far past what a tick can express.
    if decimal and decimal.group(1) and len(decimal.group(1)) > 18:
        raise ValueError("decimal has more fractional digits than a tick can distinguish")
    return Fraction(Decimal(text))


def _rounded(value: Fraction, rounding: Rounding) -> int:
    numerator, denominator = value.numerator, value.denominator
    quotient, remainder = divmod(abs(numerator), denominator)
    sign = -1 if numerator < 0 else 1
    if remainder == 0:
        return sign * quotient
    if rounding is Rounding.FLOOR:
        return -(quotient + 1) if sign < 0 else quotient
    if rounding is Rounding.CEIL:
        return -quotient if sign < 0 else quotient + 1
    doubled = remainder * 2
    if doubled < denominator:
        return sign * quotient
    if doubled > denominator:
        return sign * (quotient + 1)
    return sign * (quotient if quotient % 2 == 0 else quotient + 1)


@dataclass(frozen=True, order=True, slots=True)
class Length:
    """Exact fixed-point length. One tick equals 1/16000 mm."""

    ticks: int
    TICKS_PER_MM = 16_000
    TICKS_PER_INCH = 406_400
    _MULTIPLIERS = {
        "tick": 1,
        "ticks": 1,
        "mm": TICKS_PER_MM,
        "millimeter": TICKS_PER_MM,
        "millimeters": TICKS_PER_MM,
        "cm": TICKS_PER_MM * 10,
        "m": TICKS_PER_MM * 1000,
        "in": TICKS_PER_INCH,
        "inch": TICKS_PER_INCH,
        "inches": TICKS_PER_INCH,
        "ft": TICKS_PER_INCH * 12,
    }

    def __post_init__(self) -> None:
        if self.ticks < 0:
            raise ValueError("length cannot be negative")

    @classmethod
    def of(cls, value: int | str | Decimal | Fraction, unit: str = "mm", rounding: Rounding = Rounding.NEAREST) -> "Length":
        key = unit.lower().strip()
        if key not in cls._MULTIPLIERS:
            raise ValueError(f"unsupported length unit: {unit}")
        return cls(_rounded(_fraction(value) * cls._MULTIPLIERS[key], rounding))

    @classmethod
    def mm(cls, value: Any, rounding: Rounding = Rounding.NEAREST) -> "Length":
        return cls.of(value, "mm", rounding)

    @classmethod
    def inches(cls, value: Any, rounding: Rounding = Rounding.NEAREST) -> "Length":
        return cls.of(value, "in", rounding)

    @classmethod
    def parse(cls, value: Any, default_unit: str = "mm", rounding: Rounding = Rounding.NEAREST) -> "Length":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.of(value["value"], value.get("unit", default_unit), rounding)
        if isinstance(value, float):
            raise TypeError("float length input is intentionally rejected; use a decimal string")
        if isinstance(value, str):
            match = re.fullmatch(r"(.+?)\s*(mm|cm|m|in|inch|inches|ft|ticks?)", value.strip(), re.I)
            if match:
                return cls.of(match.group(1).strip(), match.group(2), rounding)
        return cls.of(value, default_unit, rounding)

    def as_fraction(self, unit: str = "mm") -> Fraction:
        key = unit.lower()
        if key not in self._MULTIPLIERS:
            raise ValueError(f"unsupported length unit: {unit}")
        return Fraction(self.ticks, self._MULTIPLIERS[key])

    def decimal(self, unit: str = "mm", places: int = 8) -> str:
        value = Decimal(self.as_fraction(unit).numerator) / Decimal(self.as_fraction(unit).denominator)
        return format(value.quantize(Decimal(1).scaleb(-places)).normalize(), "f")

    def to_dict(self, unit: str = "mm") -> dict[str, int | str]:
        return {"ticks": self.ticks, "value": self.decimal(unit), "unit": unit}


@dataclass(frozen=True, order=True, slots=True)
class Weight:
    """Exact weight. One tick equals 1/8 microgram."""

    ticks: int
    TICKS_PER_MG = 8_000
    TICKS_PER_G = 8_000_000
    TICKS_PER_KG = 8_000_000_000
    TICKS_PER_OZ = 226_796_185
    TICKS_PER_LB = 3_628_738_960
    _MULTIPLIERS = {
        "tick": 1,
        "ticks": 1,
        "mg": TICKS_PER_MG,
        "g": TICKS_PER_G,
        "kg": TICKS_PER_KG,
        "oz": TICKS_PER_OZ,
        "lb": TICKS_PER_LB,
        "lbs": TICKS_PER_LB,
    }

    def __post_init__(self) -> None:
        if self.ticks < 0:
            raise ValueError("weight cannot be negative")

    @classmethod
    def of(cls, value: Any, unit: str = "g", rounding: Rounding = Rounding.NEAREST) -> "Weight":
        key = unit.lower().strip()
        if key not in cls._MULTIPLIERS:
            raise ValueError(f"unsupported weight unit: {unit}")
        return cls(_rounded(_fraction(value) * cls._MULTIPLIERS[key], rounding))

    @classmethod
    def parse(cls, value: Any, default_unit: str = "g", rounding: Rounding = Rounding.NEAREST) -> "Weight":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.of(value["value"], value.get("unit", default_unit), rounding)
        if isinstance(value, float):
            raise TypeError("float weight input is intentionally rejected; use a decimal string")
        if isinstance(value, str):
            match = re.fullmatch(r"(.+?)\s*(mg|kg|g|oz|lbs?|ticks?)", value.strip(), re.I)
            if match:
                return cls.of(match.group(1).strip(), match.group(2), rounding)
        return cls.of(value, default_unit, rounding)

    def decimal(self, unit: str = "g", places: int = 8) -> str:
        multiplier = self._MULTIPLIERS[unit.lower()]
        value = Decimal(self.ticks) / Decimal(multiplier)
        return format(value.quantize(Decimal(1).scaleb(-places)).normalize(), "f")

    def to_dict(self, unit: str = "g") -> dict[str, int | str]:
        return {"ticks": self.ticks, "value": self.decimal(unit), "unit": unit}
