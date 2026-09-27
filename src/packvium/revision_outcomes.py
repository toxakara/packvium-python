"""Approved plan revisions as labeled evidence for historical evaluation.

A revision records what differed from an approved plan. This module appends those records to
an `OutcomeLedger`, where `packvium.holdout` already evaluates recommendations on untouched
history. It adds no second evaluator and no second store.

Three properties carry the task's acceptance, and each is structural rather than a check:

- **Traceable.** Every ledger event's id is the decision, the SHA-256 of the revision that
  recorded it and the event's sequence, so evidence behind a recommendation names one hashed,
  versioned document. The decision is part of the id because a ledger's ids are global.
- **Append-only.** Recording only calls `OutcomeLedger.record`, which never replaces an event.
  Neither the approved plan nor the revision is read back or changed, so an outcome cannot
  overwrite what was planned. Re-recording a revision is idempotent.
- **Learning cannot reach a result.** No solver, validator or request reads the ledger.

The one input a revision does not carry is when it happened: revisions take no clock, so the
integrator passes `recorded_at` from its own system, as for any ledger event.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

from .outcomes import OutcomeEvent, OutcomeEventType, OutcomeLedger
from .revisions import FORMAT, PlanRevisionError, document_digest

__all__ = ["record_revision", "revision_outcome_events"]


def revision_outcome_events(
    revision: Mapping[str, Any],
    decision_id: str,
    recorded_at: int,
) -> Tuple[OutcomeEvent, ...]:
    """The ledger events one revision's exceptions become. The root records none."""
    if not isinstance(revision, Mapping) or revision.get("format") != FORMAT:
        raise PlanRevisionError("invalid_revision", f"not a {FORMAT} document")
    digest = document_digest(revision)
    return tuple(
        OutcomeEvent(
            event_id=f"{decision_id}@{digest}#{event['sequence']}",
            decision_id=decision_id,
            event_type=_TYPES[event["type"]],
            payload=_PAYLOADS[event["type"]](event),
            recorded_at=recorded_at,
        )
        for event in revision["events"]
    )


def record_revision(
    ledger: OutcomeLedger,
    revision: Mapping[str, Any],
    decision_id: str,
    recorded_at: int,
) -> Tuple[OutcomeEvent, ...]:
    """Append one revision's exceptions to `ledger`, and return what the ledger holds for them."""
    return tuple(
        ledger.record(event)
        for event in revision_outcome_events(revision, decision_id, recorded_at)
    )


_TYPES = {
    "item_missing": OutcomeEventType.ITEM_MISSING,
    "container_substituted": OutcomeEventType.ACTUAL_CARTON,
    # A lock is an operator deciding where an item goes instead of the plan, which is what
    # an override is -- and why holdout reports it against the baseline, never the treatment.
    "placement_locked": OutcomeEventType.OPERATOR_OVERRIDE,
    "placement_verified": OutcomeEventType.PLACEMENT_VERIFIED,
}

_PAYLOADS: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
    "item_missing": lambda event: {"item_type": event["item_type"], "quantity": event["quantity"]},
    "container_substituted": lambda event: {"carton_id": event["replacement"]["id"]},
    "placement_locked": lambda event: {"reason_code": "placement_locked"},
    "placement_verified": lambda event: {
        "item_type": event["placement"]["item_type"],
        "container_type": event["placement"]["container_type"],
    },
}
