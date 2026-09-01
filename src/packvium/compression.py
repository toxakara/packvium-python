"""Occupied height of a `compressible` item under the load resting on it.

The model is fixed by [docs/IRREGULAR-ITEMS.md](../../../docs/IRREGULAR-ITEMS.md) and is
reproduced here rather than imported: `scripts/irregular_items_model.py` is the independent
oracle those numbers are cross-checked against, and an engine that imported it would be
checking the oracle against itself.

Every value on this path is an exact integer or a reduced rational. Pressure is carried as a
numerator/denominator pair rather than a `Fraction` for two reasons: the hard limit is a
comparison, which cross multiplication answers without dividing at all, and PHP, Rust and
JavaScript have no rational type to port a `Fraction` to ( through ).
"""

from __future__ import annotations

from math import gcd

from ._compat import dataclass
from .units import Length, Weight

#: Parts per million, the scale `compression_ratio` is carried at once parsed. Shared with
#: `constraints.SUPPORT_SCALE` by value rather than by import: the two ratios are unrelated
#: quantities that happen to use the same precision, and coupling them would make a change
#: to one silently redefine the other.
PPM = 1_000_000

#: Conventional standard gravity, exactly 9.80665 m/s^2. A decimal literal would put the
#: first float on a path the contract requires to be exact.
STANDARD_GRAVITY_NUMERATOR = 980_665
STANDARD_GRAVITY_DENOMINATOR = 100_000

#: Pascals per kilopascal.
PASCALS_PER_KILOPASCAL = 1_000

_TICKS_PER_METRE = Length.TICKS_PER_MM * 1_000


class CrushViolation(ValueError):
    """Applied pressure is strictly above the item's declared limit.

    A hard boundary, not a warning: a placement that crushes its item is not a worse
    placement, it is an invalid one, and no score derived after it may be returned.
    """


@dataclass(frozen=True, slots=True)
class Pressure:
    """An exact pressure in kPa, held as a reduced non-negative rational."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("pressure denominator must be positive")
        if self.numerator < 0:
            raise ValueError("pressure cannot be negative")

    @classmethod
    def zero(cls) -> "Pressure":
        return cls(0, 1)

    @classmethod
    def reduced(cls, numerator: int, denominator: int) -> "Pressure":
        if denominator <= 0:
            raise ValueError("pressure denominator must be positive")
        divisor = gcd(numerator, denominator)
        return cls(numerator // divisor, denominator // divisor)

    def exceeds_kpa(self, limit_kpa: int) -> bool:
        """Cross multiplication, so the comparison never leaves the integers."""
        return self.numerator > limit_kpa * self.denominator

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator} kPa"


def applied_pressure(top_load: Weight, footprint_area_ticks2: int) -> Pressure:
    """Pressure from the cumulative mass resting above an item, over its footprint.

    The item's own mass is excluded -- it is not a load on itself -- and the footprint is
    the uncompressed one, which compression never changes.
    """
    # No negative-load guard here: `Weight` already refuses one at construction, and a
    # second copy of that invariant would be unreachable code claiming to protect something.
    if footprint_area_ticks2 <= 0:
        raise ValueError("footprint area must be positive")
    numerator = top_load.ticks * STANDARD_GRAVITY_NUMERATOR * _TICKS_PER_METRE * _TICKS_PER_METRE
    denominator = (
        Weight.TICKS_PER_KG
        * STANDARD_GRAVITY_DENOMINATOR
        * PASCALS_PER_KILOPASCAL
        * footprint_area_ticks2
    )
    return Pressure.reduced(numerator, denominator)


def effective_height_ticks(
    height_ticks: int,
    compression_ratio_ppm: int,
    max_pressure_kpa: int,
    pressure: Pressure,
) -> int:
    """Occupied height under load, rounded up, never below one tick.

    Rounding up is what keeps a discrete packer honest: it may never claim less space than
    the continuous model allows. The one-tick floor stops a fully compressible item from
    reaching zero height, where it would slip past collision and support invariants
    entirely rather than merely occupying very little.
    """
    if height_ticks <= 0:
        raise ValueError("height must be positive")
    if not 0 <= compression_ratio_ppm <= PPM:
        raise ValueError("compression ratio must be between zero and one million ppm")
    if max_pressure_kpa < 0:
        raise ValueError("maximum pressure cannot be negative")
    if pressure.exceeds_kpa(max_pressure_kpa):
        raise CrushViolation(
            f"applied pressure {pressure} exceeds the declared limit of {max_pressure_kpa} kPa"
        )
    # With no headroom declared, the only admissible pressure is zero -- already proven by
    # the guard above -- so the item is simply uncompressed. Returning here also keeps the
    # divisor below non-zero.
    if max_pressure_kpa == 0:
        return height_ticks
    divisor = max_pressure_kpa * PPM * pressure.denominator
    # Non-negative: the crush guard above bounds `numerator` by `max_pressure_kpa *
    # denominator`, and `compression_ratio_ppm` by `PPM`, so the product cannot exceed
    # `divisor`. A fully compressible item at its exact limit retains zero and floors at one.
    retained = divisor - compression_ratio_ppm * pressure.numerator
    return max(1, (height_ticks * retained + divisor - 1) // divisor)


def effective_volume_ticks3(
    length_ticks: int,
    width_ticks: int,
    height_ticks: int,
    compression_ratio_ppm: int,
    max_pressure_kpa: int,
    pressure: Pressure,
) -> int:
    """Occupied volume under load. Only the height compresses; the footprint is fixed."""
    if length_ticks <= 0 or width_ticks <= 0:
        raise ValueError("footprint dimensions must be positive")
    return length_ticks * width_ticks * effective_height_ticks(
        height_ticks, compression_ratio_ppm, max_pressure_kpa, pressure
    )


def ratio_to_ppm(ratio: float) -> int:
    """The existing public ratio rule, `floor(ratio * 1000000 + 0.5)`, applied once.

    Applied once and at the boundary, so the float a caller supplied never reaches the
    geometry: everything downstream of this function is an integer.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("compression_ratio must be between zero and one")
    return int(ratio * PPM + 0.5)
