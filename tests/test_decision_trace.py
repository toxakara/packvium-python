"""Decision trace events (spec/explainability/decision-trace.md).

Two properties matter, and only one of them is "the events look right":

1. The golden trace (`spec/explainability/golden/rejection-then-placement.json`) is
   byte-reproducible from a real run of the scenario it describes -- not hand-edited,
   not approximate.
2. The one invariant the whole feature exists to keep: running the identical request
   *without* a trace callback produces the identical packing result. A trace is a pure
   observer; this is the concrete check that it never becomes anything else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packvium import AxisAlignedBox, Container, Dimensions, Item, Obstacle, Packer, PackingConfig, Point, Rotation, use_trace

# The trace-event schema and its golden document live with the specification, one level
# above this package. A published copy of the package is a tree on its own and does not
# carry them, so this module skips rather than failing collection for everyone who
# installs it. `jsonschema` is likewise only needed to validate against that schema.
SPEC_DIR = Path(__file__).resolve().parents[2] / "spec/explainability"
if not SPEC_DIR.is_dir():
    pytest.skip("the shared trace-event specification is not part of this package",
                allow_module_level=True)

Draft202012Validator = pytest.importorskip("jsonschema").Draft202012Validator
GOLDEN = json.loads((SPEC_DIR / "golden/rejection-then-placement.json").read_text())
SCHEMA = json.loads((SPEC_DIR / "schema/trace-event.schema.json").read_text())


def _scenario():
    """Two base slabs with a floor gap between them (an obstacle, so nothing can ever
    rest in it) and a third item wide enough to span the gap. Its first tried candidate
    only reaches the far slab by half its width -- rejected for insufficient support --
    before a fully-supported candidate is found. Every rotation is pinned to LWH so the
    search cannot sidestep the geometry by rotating into a thinner envelope that fits
    the gap itself, which is exactly what happened while first constructing this
    scenario and is why it is called out here rather than left as a silent trap for
    the next person to adjust these numbers."""
    base1 = Item.create("base1", Dimensions.mm(40, 100, 10), 0, quantity=1, priority=2, allowed_rotations=[Rotation.LWH])
    base2 = Item.create("base2", Dimensions.mm(40, 100, 10), 0, quantity=1, priority=2, allowed_rotations=[Rotation.LWH])
    narrow = Item.create("narrow", Dimensions.mm(40, 100, 10), 0, quantity=1, minimum_support_ratio=0.6, priority=1, allowed_rotations=[Rotation.LWH])
    gap_filler = Obstacle("gap", AxisAlignedBox(Point(640_000, 0, 0), Dimensions.mm(20, 100, 10)))
    box = Container.create("box", Dimensions.mm(100, 100, 40), obstacles=(gap_filler,))
    # A counted effort budget, not a bare time_limit_ms: the spec's own invariant
    # section explains why a wall-clock-only budget cannot promise "tracing changes
    # nothing" (the same determinism rules apply here).
    config = PackingConfig.balanced(time_limit_ms=10_000, solvers=("extreme_points",), multi_start_orders=1)
    return [base1, base2, narrow], [box], config


def test_the_golden_trace_is_schema_valid():
    validator = Draft202012Validator(SCHEMA)
    for event in GOLDEN:
        errors = sorted(validator.iter_errors(event), key=lambda e: list(e.path))
        assert not errors, (event, errors[0].message)


def test_the_golden_trace_reproduces_byte_for_byte():
    items, containers, config = _scenario()
    events = []
    with use_trace(events.append):
        Packer(config).pack(items, containers)
    assert events == GOLDEN


def test_the_golden_trace_contains_a_rejection_and_a_later_acceptance_for_the_same_item():
    rejections = [e for e in GOLDEN if e["type"] == "placement_rejection" and e["item_id"] == "narrow#1"]
    acceptances = [e for e in GOLDEN if e["type"] == "score" and e["item_id"] == "narrow#1"]
    assert rejections and rejections[0]["code"] == "insufficient_support"
    assert acceptances
    assert GOLDEN.index(rejections[0]) < GOLDEN.index(acceptances[-1])


def _chosen_answer(report: dict) -> dict:
    """The answer a caller receives, without the effort diagnostics.

    `duration_ms` is real wall-clock time, so this scenario -- like every other
    reproducibility check in this suite (see `test_packing.py`'s own `chosen_answer`)
    -- must not compare it directly: two back-to-back real solves can legitimately
    round to a different millisecond even when nothing about the decision differs."""
    for field in ("duration_ms", "candidates_evaluated", "placements_attempted", "metrics"):
        report["algorithm"].pop(field, None)
    return report


def test_enabling_a_trace_never_changes_the_packing_result():
    items, containers, config = _scenario()
    traced_events = []
    with use_trace(traced_events.append):
        traced = Packer(config).pack(items, containers)
    untraced = Packer(config).pack(items, containers)

    assert traced_events  # the scenario is pointless if nothing was ever emitted
    assert _chosen_answer(traced.to_dict()) == _chosen_answer(untraced.to_dict())


def test_no_trace_argument_and_trace_none_are_the_same_as_never_having_this_feature():
    items, containers, config = _scenario()
    default = Packer(config).pack(items, containers)
    explicit_none = Packer(config, trace=None).pack(items, containers)
    assert _chosen_answer(default.to_dict()) == _chosen_answer(explicit_none.to_dict())


def test_a_packer_level_trace_takes_precedence_over_an_ambient_context():
    """`Packer(config, trace=...)` must not be silently shadowed by -- or silently
    shadow -- an unrelated ambient `use_trace` scope a caller did not intend for this
    particular pack() call."""
    items, containers, config = _scenario()
    packer_events = []
    ambient_events = []
    with use_trace(ambient_events.append):
        Packer(config, trace=packer_events.append).pack(items, containers)
    assert packer_events == GOLDEN
    assert ambient_events == []
