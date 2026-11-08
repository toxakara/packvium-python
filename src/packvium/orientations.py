"""The 24 proper rotations an item may be packed in, in canonical table order (1.10).

A box has six distinguishable orientations and a convex hull has 24, so the vocabulary is the
six legacy codes followed by 18 signed ones. A legacy code names the declared side each
container axis takes (`LHW`: x spans the length, y the height, z the width) and keeps the
matrix it has always had: an even permutation carries no sign, an odd one negates x, so all
six are proper rotations. A signed code spells the sign out per axis -- `-L-W+H` is the
half-turn about the vertical -- and only a `convex_hull` may ask for one, because on a box it
is a legacy box under another name.

The table order is the tie-break order everywhere: the six legacy codes, then for each
permutation in that order its three remaining determinant +1 sign patterns, even permutations
in the order `+++, +--, -+-, --+` and odd ones in `-++, +-+, ++-, ---`, the legacy pattern
skipped. Codes are never compared as strings. This module is a leaf: the hull geometry and
the box geometry both read it, so the two can never disagree on what a code means.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

Axes = Tuple[int, int, int]
Signs = Tuple[int, int, int]

#: The declared sides in the order a source axis index names them.
SIDES = "LWH"

LEGACY_CODES = ("LWH", "LHW", "WLH", "WHL", "HLW", "HWL")

_EVEN_PATTERNS: Tuple[Signs, ...] = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
_ODD_PATTERNS: Tuple[Signs, ...] = ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, -1))


def _axes_of(legacy: str) -> Axes:
    first, second, third = (SIDES.index(side) for side in legacy)
    return first, second, third


def _is_odd(axes: Axes) -> bool:
    return sum(1 for first in range(3) for second in range(first + 1, 3)
               if axes[first] > axes[second]) % 2 == 1


def _legacy_signs(axes: Axes) -> Signs:
    return (-1, 1, 1) if _is_odd(axes) else (1, 1, 1)


def _signed_code(axes: Axes, signs: Signs) -> str:
    return "".join(("+" if sign > 0 else "-") + SIDES[axis] for axis, sign in zip(axes, signs))


def _build() -> Tuple[Tuple[str, ...], Dict[str, Axes], Dict[str, Signs]]:
    codes = list(LEGACY_CODES)
    axes_of = {code: _axes_of(code) for code in LEGACY_CODES}
    signs_of = {code: _legacy_signs(axes_of[code]) for code in LEGACY_CODES}
    for legacy in LEGACY_CODES:
        axes = axes_of[legacy]
        for signs in _ODD_PATTERNS if _is_odd(axes) else _EVEN_PATTERNS:
            if signs == signs_of[legacy]:
                continue
            code = _signed_code(axes, signs)
            codes.append(code)
            axes_of[code] = axes
            signs_of[code] = signs
    return tuple(codes), axes_of, signs_of


#: Every code in canonical table order; `SOURCE_AXES[code][i]` is the declared side container
#: axis `i` takes its extent from and `AXIS_SIGNS[code][i]` the sign it carries.
CODES, SOURCE_AXES, AXIS_SIGNS = _build()

SIGNED_CODES = CODES[len(LEGACY_CODES):]

#: The codes that keep the item's own height vertical and pointing up: z takes `+H`.
UPRIGHT_CODES = tuple(code for code in CODES if SOURCE_AXES[code][2] == 2 and AXIS_SIGNS[code][2] == 1)

#: What a signed code on anything but a hull is refused with, at its own pointer.
SIGNED_CODE_NEEDS_A_HULL = "a signed rotation code requires shape_type convex_hull"


def vocabulary(hull: bool) -> Tuple[str, ...]:
    """The codes an item may name: all 24 for a hull, the six legacy codes for anything else."""
    return CODES if hull else LEGACY_CODES


def refusal_reason(value: object, hull: bool) -> Optional[str]:
    """Why `value` cannot orient an item of this shape, or None when it can.

    `invalid_value` for a signed code beside a non-hull -- a rule broken, not an unknown word --
    and `not_allowed` for anything outside the item's vocabulary. Every field that names an
    orientation of a requested item (`allowed_rotations`, `fixed_placements[].orientation`, a
    revision's `placement.orientation`) reads it through here. O(1).
    """
    if not hull and value in SIGNED_CODES:
        return "invalid_value"
    return None if value in vocabulary(hull) else "not_allowed"
