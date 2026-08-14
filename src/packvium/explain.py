"""Deterministic, localization-ready human-readable explanations for why an
item was not packed.

Rendering is driven entirely by `UnpackedItem.reason` and `.proof` -- the structured
evidence already attached to every unpacked item -- never by matching or
parsing free text a caller would otherwise have to scrape out of a log. `REASON_MESSAGES`
is the one place English prose lives; a caller wanting another locale swaps that map
(or calls `explain_reason` against their own), not the code that walks the evidence.
"""

from __future__ import annotations

from ._compat import dataclass

from .models import UnpackedItem

#: One deterministic English sentence per reason code -- the reason-code vocabulary
#: (`Packer._unpacked_reason`/`_support_is_the_blocker`). Every reason this library can
#: actually produce must have an entry here: `explain_reason` raises rather than
#: silently falling back to the bare code, so a new reason code introduced without a
#: matching message is caught immediately, not shipped mute.
REASON_MESSAGES: dict[str, str] = {
    "no_compatible_container_dimensions":
        "does not fit inside any offered container in any rotation",
    "rotation_restricted":
        "would fit in some container with more rotations allowed, but not with the "
        "rotations this item permits",
    "payload_exceeded":
        "exceeds the maximum payload of every offered container",
    "policy_rule":
        "is forbidden from every offered container by a policy rule the request "
        "declared -- the rule and version are in the details",
    "no_eligible_container":
        "shares no eligible container tag with any offered container",
    "time_limit":
        "was not reached before the configured time limit expired",
    "effort_limit":
        "was not reached before the configured effort budget was exhausted",
    "group_cannot_fit_together":
        "belongs to a group that could not all be placed together",
    "insufficient_support":
        "would fit geometrically, but only by resting on support the minimum "
        "support ratio forbids",
    "no_feasible_placement":
        "found no feasible placement in the containers offered, for a reason the "
        "search could not further isolate",
    "search_exhausted":
        "was not placed before the configured search strategies were exhausted",
    "exact_search_incomplete":
        "was not placed because the exact search ended before proving a final answer",
    "container_inventory_exhausted":
        "requires another compatible container, but the declared inventory is exhausted",
}

#: Prefixes an explanation with what kind of evidence backs it, from `ReasonProof.level`
#:: a caller deciding whether to trust "this will never fit" (proven) versus
#: "try again with more time" (unknown_due_to_limit) needs this distinction as much as
#: the sentence itself.
LEVEL_PREFIXES: dict[str, str] = {
    "proven": "Proven",
    "unknown_due_to_limit": "Unknown (limit reached)",
    "observed": "Observed",
    "inferred": "Inferred",
}


class UnknownReasonError(KeyError):
    """A reason code with no registered message -- caught here, at the boundary,
    rather than surfacing a raw `KeyError` from a dict lookup deep inside a renderer."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"no explanation registered for reason code {reason!r}")


@dataclass(frozen=True)
class Explanation:
    """Locale-neutral explanation payload plus the bundled English fallback."""

    message_key: str
    arguments: tuple[tuple[str, str], ...]
    default_message: str

    def argument(self, name: str) -> str | None:
        return dict(self.arguments).get(name)


def explain_reason(reason: str) -> str:
    """The bare English sentence for one reason code, with no item identity or
    evidence-level prefix -- the building block `explain_unpacked_item` composes."""
    try:
        return REASON_MESSAGES[reason]
    except KeyError:
        raise UnknownReasonError(reason) from None


def explain_unpacked_item(item: UnpackedItem) -> str:
    """One deterministic sentence naming the item, its evidence level, and why it was
    not packed -- built entirely from `item.reason` and `item.proof.level`, never by
    inspecting or parsing any log output. Any `item.details` already attached are
    appended verbatim, in order, not reformatted or re-derived."""
    sentence = explanation_for_unpacked_item(item).default_message
    level = item.proof.level if item.proof is not None else None
    prefix = LEVEL_PREFIXES.get(level, "") if level is not None else ""
    detail = f" ({'; '.join(item.details)})" if item.details else ""
    lead = f"{prefix}: " if prefix else ""
    return f"{item.instance.id}: {lead}{sentence}{detail}"


def explanation_for_unpacked_item(item: UnpackedItem) -> Explanation:
    """Stable localization key and ordered string arguments for one rejection."""
    level = item.proof.level if item.proof is not None else ""
    return Explanation(
        message_key=f"packvium.unpacked.{item.reason}",
        arguments=(
            ("item_id", item.instance.id),
            ("evidence_level", level),
            ("details", "; ".join(item.details)),
        ),
        default_message=explain_reason(item.reason),
    )


def explain_unpacked_items(items: tuple[UnpackedItem, ...]) -> tuple[str, ...]:
    """`explain_unpacked_item` over every item, in the order given -- deterministic
    because the input order is, and this makes no ordering decision of its own."""
    return tuple(explain_unpacked_item(item) for item in items)
