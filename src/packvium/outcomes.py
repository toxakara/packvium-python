"""Domain model for first-party shipment-outcome and packing-exception events.

Closing the operating loop -- which alternative was actually chosen, what carton was
really used, measured dimensions/weight, a repack, damage, a return, an operator override
-- needs its own append-only ledger, not a mutable "decision" row a later event can quietly
edit:

  * every event names the exact decision it is about (`decision_id` -- expected to be a
    job's own idempotency key in the caller's system; this module never imports one, to
    stay decoupled) and is itself keyed by its own `event_id`, so a duplicate
    delivery of the same event is idempotent (`OutcomeLedger.record`) the same way
    `JobRegistry.submit` dedups a duplicate job submission;
  * a correction never rewrites an earlier event -- it is a new event that names the
    event id it supersedes (`OutcomeEvent.supersedes`), so "late corrections preserve
    history": the raw ledger (`events_for_decision`) always has everything that was ever
    recorded, while `current_view` folds corrections in to answer "what do we believe
    now" without discarding what was believed before;
  * every event's payload is validated against a closed, named field list per event type
    (`ALLOWED_FIELDS`) -- a free-text field this module does not recognize (the PII risk
    this task's acceptance calls out: a customer name, address, phone number typed into
    an unstructured note) is a structured rejection at construction, not something that
    is silently stored;
  * "validation remains a hard gate" is not a rule this module enforces at runtime -- it
    is a structural guarantee: this module has no method that reads, re-runs or
    overrides a packing decision's own feasibility validation. It only ever appends
    structured facts *about* a decision that some other, already-validated system
    produced. There is no code path here that could bypass validation, because none of
    this module's code ever touches a placement or a validator.

Scope: the same kind of in-memory, domain-layer reference implementation
`packvium.commerce.catalog` and `packvium.commerce.policy` already are. A production
deployment would back `OutcomeLedger` with durable, truly-append-only storage; this module
defines and proves the contract, not the store.
"""

from __future__ import annotations

from ._compat import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


# --------------------------------------------------------------------------------- errors

class OutcomeError(Exception):
    """Base class for every outcome-domain error raised by this module."""


class UnsupportedOutcomeFieldError(OutcomeError):
    """A payload field is not in the closed, named vocabulary for its event type --
    refused at construction, the same "fail admission instead of being ignored"
    discipline `packvium.commerce.policy` uses for predicates. This is the module's PII
    guard: a field this module does not explicitly know about (e.g. a free-text note
    that might contain a name or address) can never be silently stored."""


class DuplicateEventMismatchError(OutcomeError):
    """The same `event_id` was recorded again with different content -- an idempotent
    duplicate delivery must repeat the identical event, never a different one under the
    same id."""


class OutcomeEventNotFoundError(OutcomeError):
    """No event is recorded under the given event id."""


# ---------------------------------------------------------------------------- event types

class OutcomeEventType(str, Enum):
    ALTERNATIVE_CHOSEN = "alternative_chosen"
    ACTUAL_CARTON = "actual_carton"
    MEASURED_DIMENSIONS = "measured_dimensions"
    MEASURED_WEIGHT = "measured_weight"
    REPACK = "repack"
    DAMAGE = "damage"
    RETURN = "return"
    OPERATOR_OVERRIDE = "operator_override"


#: The closed, named payload vocabulary per event type. Deliberately narrow and
#: structured (ids, numeric measurements, enumerated reason codes) -- nothing here is a
#: free-text field a caller could use to smuggle in personally identifiable data. A field
#: outside this list is refused at construction (`UnsupportedOutcomeFieldError`), never
#: silently dropped or silently stored.
ALLOWED_FIELDS: dict[OutcomeEventType, frozenset[str]] = {
    OutcomeEventType.ALTERNATIVE_CHOSEN: frozenset({"alternative_index", "objective_score"}),
    OutcomeEventType.ACTUAL_CARTON: frozenset({"carton_id", "catalog_version"}),
    OutcomeEventType.MEASURED_DIMENSIONS: frozenset({
        "length_mm", "width_mm", "height_mm",
    }),
    OutcomeEventType.MEASURED_WEIGHT: frozenset({"weight_g"}),
    OutcomeEventType.REPACK: frozenset({"reason_code", "new_carton_id"}),
    OutcomeEventType.DAMAGE: frozenset({"reason_code", "severity"}),
    OutcomeEventType.RETURN: frozenset({"reason_code"}),
    OutcomeEventType.OPERATOR_OVERRIDE: frozenset({"reason_code", "operator_id"}),
}


def _validate_payload(event_type: OutcomeEventType, payload: Mapping[str, Any]) -> None:
    allowed = ALLOWED_FIELDS.get(event_type)
    if allowed is None:
        raise UnsupportedOutcomeFieldError(f"unsupported event type {event_type!r}")
    unsupported = set(payload) - allowed
    if unsupported:
        raise UnsupportedOutcomeFieldError(
            f"event type {event_type.value!r} does not support field(s) {sorted(unsupported)}; "
            f"allowed: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------------- event

@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    """One immutable fact about one decision. Never mutated once recorded; a correction
    is a *new* `OutcomeEvent` whose `supersedes` names this one's `event_id`."""

    event_id: str
    decision_id: str
    event_type: OutcomeEventType
    payload: Mapping[str, Any]
    recorded_at: int
    supersedes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.decision_id:
            raise ValueError("decision_id is required")
        if self.recorded_at < 0:
            raise ValueError("recorded_at cannot be negative")
        if self.supersedes == self.event_id:
            raise ValueError("an event cannot supersede itself")
        _validate_payload(self.event_type, self.payload)


# --------------------------------------------------------------------------------- ledger

class OutcomeLedger:
    """Append-only store of `OutcomeEvent`s, indexed by `decision_id`. See the module
    docstring for what this contract does and does not cover."""

    def __init__(self) -> None:
        self._by_id: dict[str, OutcomeEvent] = {}
        self._by_decision: dict[str, list[str]] = {}

    def record(self, event: OutcomeEvent) -> OutcomeEvent:
        """Append `event`, or return the existing one unchanged if this `event_id` was
        already recorded with identical content (idempotent duplicate delivery). Raises
        `DuplicateEventMismatchError` if the same `event_id` is recorded again with
        different content -- a duplicate delivery must repeat the same event, not a
        different one wearing its id."""
        existing = self._by_id.get(event.event_id)
        if existing is not None:
            if existing != event:
                raise DuplicateEventMismatchError(
                    f"event id {event.event_id!r} was already recorded with different content"
                )
            return existing
        if event.supersedes is not None and event.supersedes not in self._by_id:
            raise OutcomeEventNotFoundError(
                f"event {event.event_id!r} supersedes unknown event {event.supersedes!r}"
            )
        self._by_id[event.event_id] = event
        self._by_decision.setdefault(event.decision_id, []).append(event.event_id)
        return event

    def events_for_decision(self, decision_id: str) -> tuple[OutcomeEvent, ...]:
        """The complete, unfolded history for one decision, in recorded order -- every
        event ever recorded, including every one a later correction superseded."""
        return tuple(self._by_id[event_id] for event_id in self._by_decision.get(decision_id, ()))

    def view_as_of(self, decision_id: str, at: int) -> tuple[OutcomeEvent, ...]:
        """What was believed about one decision at time `at` -- events recorded strictly
        before it, with only the corrections that had also been recorded by then.

        `current_view` cannot answer this. It folds every correction ever recorded, so
        using it to reconstruct a past belief pulls later knowledge backward through time:
        a holdout evaluation built on it would train on evidence that did not exist when
        the recommendation was formed, and report a backtest that nobody could have run.

        History is still never discarded. This is a narrower fold over the same immutable
        events, not a second store, and `events_for_decision` continues to return
        everything.
        """
        if at < 0:
            raise ValueError("as-of time cannot be negative")
        known = tuple(
            event for event in self.events_for_decision(decision_id)
            if event.recorded_at < at
        )
        # A correction only applies if it too was known by `at`; one recorded later leaves
        # the event it supersedes standing, because that is what was believed at the time.
        superseded = {
            event.supersedes for event in known if event.supersedes is not None
        }
        return tuple(event for event in known if event.event_id not in superseded)

    def current_view(self, decision_id: str) -> tuple[OutcomeEvent, ...]:
        """The current, corrected understanding of one decision: every recorded event
        for it *except* those a later correction has superseded. History itself is
        never discarded -- `events_for_decision` still returns every event -- this only
        filters which ones currently apply."""
        superseded = {
            event.supersedes
            for event in self.events_for_decision(decision_id)
            if event.supersedes is not None
        }
        return tuple(
            event for event in self.events_for_decision(decision_id)
            if event.event_id not in superseded
        )
