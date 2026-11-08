"""Pallet overhang (docs/PALLET-AND-ROBOT-RULES.md "container.pallet_overhang_limit").

A container with `pallet_overhang_limit` `(l, w)` is a deck `L x W`, its `inner_dimensions`,
that a load may overhang by at most `l` past each end and `w` past each side. Every solver
packs the solving frame `(L + 2l) x (W + 2w) x H`, whose origin is the corner of the overhang
zone, so the deck lies on `[l, l + L] x [w, w + W]` and every coordinate stays non-negative.

Only the deck carries weight. A placement on the floor rests on its overlap with the deck,
which `minimum_support_ratio` and the centre-of-gravity rule judge exactly as they judge a
supporting box; at a ratio of 0 nothing is enforced, as everywhere else. The packed pallet
is billed and handed up at the size it reached: its declared footprint (outer if stated,
else inner) grown by the overhang actually used past each edge. Exact integers throughout.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from ._compat import dataclass
from .geometry import AxisAlignedBox, Dimensions, Point
from .units import Length

if TYPE_CHECKING:
    from .models import Container


@dataclass(frozen=True, slots=True)
class OverhangLimit:
    """How far a load may pass the deck edge, per end along `length` and per side along `width`.
    Zero forbids overhang on that axis; it is not the same as the field being absent."""

    length: Length
    width: Length


@dataclass(frozen=True, slots=True)
class Deck:
    """The weight-bearing footprint inside a solving frame, in frame ticks."""

    x: int
    y: int
    length: int
    width: int

    @property
    def x2(self) -> int: return self.x + self.length
    @property
    def y2(self) -> int: return self.y + self.width

    def as_box(self) -> AxisAlignedBox:
        """The deck as a one-tick slab under the floor: only its footprint is ever read."""
        return AxisAlignedBox(Point(self.x, self.y, 0), Dimensions(Length(self.length), Length(self.width), Length(1)))


def solving_frame(container: "Container") -> "Container":
    """The container every solver packs: itself without a limit, else the widened frame
    carrying its deck. The frame states no limit, so a second call returns it unchanged."""
    limit = container.pallet_overhang_limit
    if limit is None:
        return container
    inner = container.inner_dimensions
    overhang_x, overhang_y = limit.length.ticks, limit.width.ticks
    frame = Dimensions(Length(inner.length.ticks + 2 * overhang_x),
                       Length(inner.width.ticks + 2 * overhang_y), inner.height)
    deck = Deck(overhang_x, overhang_y, inner.length.ticks, inner.width.ticks)
    return replace(container, inner_dimensions=frame, pallet_overhang_limit=None, deck=deck)


def deck_alone(frame: "Container") -> "Container":
    """A solving frame's container as declared, without the overhang zone: the rung a pallet
    trial packs first (`SolverOrchestrator._pallet_trial`)."""
    deck = frame.deck
    assert deck is not None
    inner = Dimensions(Length(deck.length), Length(deck.width), frame.inner_dimensions.height)
    return replace(frame, inner_dimensions=inner, deck=None)


def centring_offset(deck: Deck, boxes: Iterable[AxisAlignedBox]) -> tuple[int, int]:
    """The shift that seats a frame trial's load with the least overhang. Per axis: flush with
    the deck's near edge when the load is no longer than the deck, else centred on the deck
    with an odd tick past the far edge. Either way the load stays inside the frame, since its
    span is at most the deck plus both allowances. O(p); `(0, 0)` for none."""
    boxes = tuple(boxes)
    if not boxes:
        return 0, 0
    return (_seat(deck.x, deck.length, min(b.origin.x for b in boxes), max(b.x2 for b in boxes)),
            _seat(deck.y, deck.width, min(b.origin.y for b in boxes), max(b.y2 for b in boxes)))


def _seat(start: int, length: int, low: int, high: int) -> int:
    span = high - low
    target = start if span <= length else start - (span - length) // 2
    return target - low


def deck_contact_area(deck: Deck, box: AxisAlignedBox) -> int:
    """The area of a floor placement's footprint that rests on the deck. O(1)."""
    dx = min(box.x2, deck.x2) - max(box.origin.x, deck.x)
    dy = min(box.y2, deck.y2) - max(box.origin.y, deck.y)
    return dx * dy if dx > 0 and dy > 0 else 0


def floor_support_ratio(deck: "Deck | None", box: AxisAlignedBox) -> float:
    """The support ratio a placement on the floor reports: whole without a deck."""
    if deck is None:
        return 1.0
    return deck_contact_area(deck, box) / box.dimensions.base_area


def overhang_used(deck: Deck, boxes: Iterable[AxisAlignedBox]) -> tuple[int, int]:
    """How far the boxes pass the deck in total along each axis: both ends, then both sides.
    O(p); `(0, 0)` for none."""
    low_x, low_y, high_x, high_y = deck.x, deck.y, deck.x2, deck.y2
    for box in boxes:
        low_x, low_y = min(low_x, box.origin.x), min(low_y, box.origin.y)
        high_x, high_y = max(high_x, box.x2), max(high_y, box.y2)
    return (deck.x - low_x) + (high_x - deck.x2), (deck.y - low_y) + (high_y - deck.y2)


def achieved_dimensions(deck: Deck, boxes: Iterable[AxisAlignedBox]) -> Dimensions:
    """The deck grown by the overhang used, at the load's height: the footprint and height
    the loaded pallet reaches. A packed container holds at least one box. O(p)."""
    boxes = tuple(boxes)
    over_x, over_y = overhang_used(deck, boxes)
    return Dimensions(Length(deck.length + over_x), Length(deck.width + over_y),
                      Length(max(box.z2 for box in boxes)))


def billed_dimensions(container: "Container", deck: Deck, boxes: Iterable[AxisAlignedBox]) -> Dimensions:
    """Per horizontal axis the larger of the declared footprint, outer if stated, and the deck
    grown by the overhang used; the declared height. An outer margin is not positioned
    against the deck, so it is never added to the overhang. O(p)."""
    base = container.outer_dimensions or Dimensions(Length(deck.length), Length(deck.width),
                                                    container.inner_dimensions.height)
    over_x, over_y = overhang_used(deck, boxes)
    return Dimensions(Length(max(base.length.ticks, deck.length + over_x)),
                      Length(max(base.width.ticks, deck.width + over_y)), base.height)
