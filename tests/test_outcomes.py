"""The append-only outcome ledger, through the package's own import path.

The ledger is the argument `evaluate_on_history` is built around, so a consumer who cannot
construct one cannot use historical replay at all. `tests/` ships inside the wheel; these
assert the ledger a consumer gets rather than the workspace shim in front of it.

The distinction worth reading twice is `view_as_of` against `current_view`. Both fold the
same immutable events; only one of them is safe to reconstruct a past belief with.
"""

from __future__ import annotations

import pytest

from packvium.outcomes import (
    DuplicateEventMismatchError,
    OutcomeEvent,
    OutcomeEventNotFoundError,
    OutcomeEventType,
    OutcomeLedger,
)

SHIPPED = OutcomeEventType.ACTUAL_CARTON
DAMAGE = OutcomeEventType.DAMAGE


def shipped(event_id: str, decision_id: str, at: int, carton: str = "box-a", **extra):
    return OutcomeEvent(event_id=event_id, decision_id=decision_id, event_type=SHIPPED,
                        payload={"carton_id": carton, "catalog_version": 1},
                        recorded_at=at, **extra)


class TestRecordingIsAppendOnlyAndIdempotent:
    def test_a_recorded_event_comes_back_for_its_decision(self):
        ledger = OutcomeLedger()
        ledger.record(shipped("e1", "d1", 100))
        assert [e.event_id for e in ledger.events_for_decision("d1")] == ["e1"]

    def test_an_unknown_decision_has_an_empty_history_rather_than_raising(self):
        assert OutcomeLedger().events_for_decision("never-shipped") == ()

    def test_recording_the_identical_event_twice_is_a_no_op(self):
        """Duplicate delivery is a fact of every queue. Repeating the same event must not
        double it, and must not be an error either."""
        ledger = OutcomeLedger()
        first = ledger.record(shipped("e1", "d1", 100))
        again = ledger.record(shipped("e1", "d1", 100))
        assert again is first
        assert len(ledger.events_for_decision("d1")) == 1

    def test_reusing_an_event_id_for_different_content_is_refused(self):
        # A duplicate delivery repeats an event; it does not carry a different one wearing
        # the same id. Accepting that would make the ledger's history unreadable.
        ledger = OutcomeLedger()
        ledger.record(shipped("e1", "d1", 100, carton="box-a"))
        with pytest.raises(DuplicateEventMismatchError, match="different content"):
            ledger.record(shipped("e1", "d1", 100, carton="box-b"))

    def test_events_come_back_in_recorded_order(self):
        ledger = OutcomeLedger()
        for index, at in enumerate((300, 100, 200), start=1):
            ledger.record(shipped(f"e{index}", "d1", at))
        assert [e.event_id for e in ledger.events_for_decision("d1")] == ["e1", "e2", "e3"]

    def test_decisions_are_kept_apart(self):
        ledger = OutcomeLedger()
        ledger.record(shipped("e1", "d1", 100))
        ledger.record(shipped("e2", "d2", 100))
        assert [e.event_id for e in ledger.events_for_decision("d1")] == ["e1"]
        assert [e.event_id for e in ledger.events_for_decision("d2")] == ["e2"]


class TestACorrectionSupersedesRatherThanOverwrites:
    def _corrected(self):
        ledger = OutcomeLedger()
        ledger.record(shipped("e1", "d1", 100, carton="box-a"))
        ledger.record(shipped("e2", "d1", 200, carton="box-b", supersedes="e1"))
        return ledger

    def test_the_correction_replaces_the_original_in_the_current_view(self):
        view = self._corrected().current_view("d1")
        assert [e.event_id for e in view] == ["e2"]

    def test_the_original_is_still_in_the_unfolded_history(self):
        """"Append-only" is the whole claim. A correction that erased what it corrected
        would make the ledger a mutable store wearing an immutable name."""
        assert [e.event_id for e in self._corrected().events_for_decision("d1")] == ["e1", "e2"]

    def test_superseding_an_event_the_ledger_has_never_seen_is_refused(self):
        ledger = OutcomeLedger()
        with pytest.raises(OutcomeEventNotFoundError, match="unknown event"):
            ledger.record(shipped("e2", "d1", 200, supersedes="ghost"))

    def test_an_event_cannot_supersede_itself(self):
        with pytest.raises(ValueError, match="cannot supersede itself"):
            shipped("e1", "d1", 100, supersedes="e1")


class TestTimeTravelIsNarrowerThanTheCurrentView:
    """`view_as_of` exists because `current_view` folds every correction ever recorded.
    Using the latter to reconstruct a past belief pulls later knowledge backward through
    time, and a holdout score built that way reports a backtest nobody could have run."""

    def _ledger(self):
        ledger = OutcomeLedger()
        ledger.record(shipped("e1", "d1", 100, carton="box-a"))
        ledger.record(shipped("e2", "d1", 300, carton="box-b", supersedes="e1"))
        return ledger

    def test_before_anything_was_recorded_the_view_is_empty(self):
        assert self._ledger().view_as_of("d1", 50) == ()

    def test_between_the_event_and_its_correction_the_original_still_stands(self):
        view = self._ledger().view_as_of("d1", 200)
        assert [e.event_id for e in view] == ["e1"]
        assert view[0].payload["carton_id"] == "box-a"

    def test_after_the_correction_the_correction_applies(self):
        view = self._ledger().view_as_of("d1", 400)
        assert [e.event_id for e in view] == ["e2"]

    def test_the_boundary_is_strictly_before(self):
        # `at` is the instant the question is asked; an event recorded at that same
        # instant is not yet part of what was believed when it was asked.
        assert self._ledger().view_as_of("d1", 100) == ()

    def test_the_current_view_disagrees_with_the_past_one_on_purpose(self):
        ledger = self._ledger()
        assert [e.event_id for e in ledger.view_as_of("d1", 200)] == ["e1"]
        assert [e.event_id for e in ledger.current_view("d1")] == ["e2"]

    def test_a_negative_instant_is_refused(self):
        with pytest.raises(ValueError, match="as-of time cannot be negative"):
            self._ledger().view_as_of("d1", -1)


class TestSeveralEventKindsCoexistOnOneDecision:
    def test_a_shipment_and_its_damage_both_stand(self):
        ledger = OutcomeLedger()
        ledger.record(shipped("e1", "d1", 100))
        ledger.record(OutcomeEvent(event_id="e2", decision_id="d1", event_type=DAMAGE,
                                   payload={"reason_code": "crushed_corner",
                                            "severity": "minor"},
                                   recorded_at=200))
        kinds = sorted(e.event_type.value for e in ledger.current_view("d1"))
        assert kinds == ["actual_carton", "damage"]


class TestAnEventMustNameItsDecision:
    def test_an_event_without_a_decision_id_is_refused(self):
        # The decision id is the only index the ledger has; an event without one could
        # never be read back.
        with pytest.raises(ValueError, match="decision_id is required"):
            OutcomeEvent(event_id="e1", decision_id="", event_type=SHIPPED,
                         payload={"carton_id": "box-a", "catalog_version": 1},
                         recorded_at=100)
