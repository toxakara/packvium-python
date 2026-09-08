"""`container.access_directions` at its boundaries.

The field was reserved at the 1.1.0 freeze and implemented in all four engines in one
change. What the wave shipped with it was a *validation and canonicalisation* path per
engine, and coverage showed that path untested in every one of them: the happy request
was exercised by conformance, the refusals by nothing.

That is the failure mode worth guarding against here rather than the packing rule, which
`test_constraints.py` and the shared corpus already hold. A field whose canonicalisation
silently stops working does not raise -- it makes two callers who named the same doors in
a different order search differently, which is a determinism break that no fixture
notices, because every fixture names its doors one way.
"""

from __future__ import annotations

import pytest

from packvium.geometry import ALL_DIRECTIONS, Dimensions, InvalidDirectionError
from packvium.models import Container
from packvium.serialization import pack_from_dict
from packvium.units import Length

MM = 16_000


def crate(**kwargs) -> Container:
    side = Dimensions(Length(100 * MM), Length(100 * MM), Length(100 * MM))
    return Container(id="crate", inner_dimensions=side, **kwargs)


# ------------------------------------------------------------------ canonicalisation

def test_doors_are_deduplicated_into_the_canonical_order():
    """Two callers naming the same doors differently must search identically.

    This is the assertion the corpus cannot make: every fixture states its doors once, in
    one order, so a canonicalisation that quietly stopped working would leave all 399
    green while making the engine order-sensitive.
    """
    assert crate(access_directions=("+z", "-x", "+z", "-x")).access_directions == ("-x", "+z")
    assert crate(access_directions=("-x", "+z")).access_directions == ("-x", "+z")


def test_every_legal_direction_survives_canonicalisation():
    """All six, in the declared order, so a typo in ALL_DIRECTIONS cannot silently drop
    one -- a dropped door is a refusal the caller never asked for."""
    assert crate(access_directions=tuple(reversed(ALL_DIRECTIONS))).access_directions \
        == tuple(ALL_DIRECTIONS)


def test_a_container_states_no_doors_by_default():
    """The pre- default, and it is *inert* rather than permissive: six walls and
    none are both nearly-vacuous, but they are different nearly-vacuous, and defaulting to
    six would switch a real constraint on for every caller who never set the field."""
    assert crate().access_directions == ()


# --------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("direction", ["north", "x", "+X", "+w", "", "±x", "-x "])
def test_an_unknown_direction_is_refused_rather_than_dropped(direction):
    """Refused, not filtered out. Silently discarding an unrecognised door would leave a
    container with fewer exits than the caller believes it has, and the packing would be
    legal for a vehicle that does not exist."""
    with pytest.raises(InvalidDirectionError) as refusal:
        crate(access_directions=(direction,))
    assert repr(direction) in str(refusal.value)


def test_one_bad_direction_refuses_the_whole_list():
    """A partially-honoured list is the worst outcome available: it validates and means
    something the caller did not write."""
    with pytest.raises(InvalidDirectionError):
        crate(access_directions=("-x", "sideways", "+z"))


# ------------------------------------------------------- the same rules through a request

def _request(doors):
    container = {"id": "van",
                 "inner_dimensions": {"length": "200", "width": "100", "height": "100"}}
    if doors is not None:
        container["access_directions"] = doors
    return {"units": {"length": "mm"},
            "items": [{"id": "cube", "quantity": 1,
                       "dimensions": {"length": "100", "width": "100", "height": "100"}}],
            "containers": [container]}


def test_a_request_reaches_the_same_validation_as_the_constructor():
    """The decoder is a fourth way to name the doors, and it must not be a way around the
    canonicalisation. `docs/STOP-ACCESSIBILITY.md` records that a rule no request can
    switch on is untested along the path it will be switched on through; this is that
    path."""
    with pytest.raises(InvalidDirectionError):
        pack_from_dict(_request(["upwards"]))


def _without_wall_clock(result: dict) -> dict:
    """Everything the contract promises to reproduce.

    `algorithm.duration_ms` is wall clock and is the one field a determinism assertion
    must not read -- comparing whole documents passes or fails on how busy the machine
    is, which is a test that reports the host rather than the engine.
    """
    trimmed = {key: value for key, value in result.items() if key != "algorithm"}
    trimmed["algorithm"] = {key: value for key, value in result["algorithm"].items()
                            if key != "duration_ms"}
    return trimmed


def test_a_request_naming_doors_in_either_order_gives_one_answer():
    """Determinism across two spellings of one intent, end to end rather than on the
    domain object alone."""
    first = pack_from_dict(_request(["-x", "+z"]))
    second = pack_from_dict(_request(["+z", "-x"]))
    assert _without_wall_clock(first) == _without_wall_clock(second)


def test_an_empty_door_list_is_accepted_and_inert():
    """`[]` is a caller saying "no doors stated", not a malformed request. It has to
    behave exactly like the absent field, or the two spellings of the default diverge."""
    stated = pack_from_dict(_request([]))
    absent = pack_from_dict(_request(None))
    assert _without_wall_clock(stated) == _without_wall_clock(absent)
