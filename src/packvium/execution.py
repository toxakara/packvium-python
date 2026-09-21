"""An execution plan derived from an already validated packing result.

`docs/EXECUTION-PLAN.md` is the contract. A packing result answers *what goes where*; an
operator needs *what to do first, and why this carton*. This module turns the first into
the second and is built so that it cannot do anything else.

It imports no solver and no validator, holds no registry and reads no clock. Everything it
emits is a function of the request and result it was handed, so the same pair yields the
same plan forever -- which is the only reason a plan can be printed, signed and audited.

Two rules do most of the work.

**Authoritative facts and presentation text are separated in the output, not just in the
prose.** Anything the solver or validator decided -- a position, a score vector, a
`feasibility.code`, an `unpacked_items[].proof` and its `level` -- appears under `facts`.
Anything a human reads appears under `presentation`, and every entry there names the
authoritative fields it was derived from in `cites`. A reason with no citation is not a
reason, and a downstream system that only ever reads `facts` loses nothing it is entitled
to rely on.

**A placement is referenced by what the cross-language contract promises, not by its id.**
A live result does carry `item_id` (`cube#1`), and `conformance/canonical.py` drops it,
along with the container's `id`, from the projection two implementations are diffed
against -- an id there is "an instance count rather than a semantic property". Four engines
need not number instances alike, and this adapter is held to byte-identical Python and PHP
output, so citing `item_id` would fail in the worst way for an operator: two correct
systems disagreeing about which box a label names. The reference is derived from fields the
projection keeps, and uses `position.*.ticks` -- the exact integer -- rather than `value`,
the rendering the same `exactScalar` also carries.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ._canonical_json import canonical_json

__all__ = [
    "FORMAT",
    "ExecutionPlanError",
    "build_execution_plan",
    "canonical_plan_json",
    "placement_reference",
]

#: The plan's own format tag. It is not the packing schema's version and does not move
#: with it: this document is derived from a result, and a result can gain fields without
#: changing what a plan says.
FORMAT = "packvium-execution-plan/v1"

#: What `score` indices mean is a property of the request's objective, and the adapter
#: does not know it. Naming an index it cannot explain would be inventing meaning, so an
#: unnamed index is reported as an index.
UNNAMED_AXIS = "unnamed objective axis"


class ExecutionPlanError(Exception):
    """The adapter was handed something it cannot describe."""


def placement_reference(container_index: int, placement: Mapping[str, Any]) -> dict[str, Any]:
    """A reference two languages agree on, for one placement in one container.

    `container_index` is the container's position in the result, not its `id`: the id is an
    instance counter that the cross-language projection drops. Position is read from
    `ticks`, never from `value`.
    """
    try:
        position = placement["position"]
        return {
            "container_index": container_index,
            "item_type": placement["item_type"],
            "orientation": placement["orientation"],
            "position_ticks": {
                axis: int(position[axis]["ticks"]) for axis in ("x", "y", "z")
            },
        }
    except (KeyError, TypeError) as error:
        raise ExecutionPlanError(
            f"placement is missing a field the reference is built from: {error}"
        ) from error


def _steps(container_index: int, container: Mapping[str, Any],
           loading_order: Optional[Sequence[int]]) -> dict[str, Any]:
    """The operator sequence for one container, or an honest absence of one.

    The engines compute a loading order under support and accessibility rules, from domain
    objects this adapter never sees. It is therefore *injected*: a caller who has the order
    passes it, and one who does not gets no order at all. The alternative -- falling back to
    the order placements happen to appear in -- would present an artifact of how the solver
    walked its candidates as if it were a safe order to lift boxes in.
    """
    placements = list(container.get("placements") or ())
    if loading_order is None:
        return {
            "order": "unavailable",
            "steps": [
                {"placement": placement_reference(container_index, placement)}
                for placement in placements
            ],
        }
    if sorted(loading_order) != list(range(len(placements))):
        raise ExecutionPlanError(
            f"loading order for container {container_index} is not a permutation of its "
            f"{len(placements)} placements"
        )
    return {
        "order": "loading",
        "steps": [
            {"sequence": step, "placement": placement_reference(container_index, placements[index])}
            for step, index in enumerate(loading_order, start=1)
        ],
    }


def _first_difference(winner: Sequence[int], loser: Sequence[int]) -> Optional[dict[str, Any]]:
    """The first index at which two score vectors differ, and by how much.

    Never a blended number. The score is compared lexicographically by the portfolio that
    produced it, so the first differing index *is* the decision; summing or weighting the
    vector would replace a decision that was made with one that was not.
    """
    for index, (a, b) in enumerate(zip(winner, loser)):
        if a != b:
            return {"index": index, "winner": a, "alternative": b, "difference": b - a}
    if len(winner) != len(loser):
        raise ExecutionPlanError(
            "score vectors of different length cannot be compared lexicographically"
        )
    return None


def _alternative(index: int, winner_score: Sequence[int],
                 alternative: Mapping[str, Any]) -> dict[str, Any]:
    difference = _first_difference(winner_score, list(alternative.get("score") or ()))
    facts = {
        "alternative_index": index,
        "score": list(alternative.get("score") or ()),
        "status": alternative.get("status"),
        "first_difference": difference,
    }
    if difference is None:
        text = ("This option scored identically to the chosen one on every objective axis; "
                "the score does not record why one was taken.")
    else:
        text = (f"This option differs first at objective axis {difference['index']} "
                f"({UNNAMED_AXIS}): chosen {difference['winner']}, this {difference['alternative']}.")
    return {
        "facts": facts,
        # Deliberately not "it lost because it is taller". The solver recorded a score, not
        # a cause; a sentence naming a cause would be a claim nothing in the result supports.
        "presentation": {"summary": text, "cites": ["score", "alternatives[].score"]},
    }


def build_execution_plan(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    loading_orders: Optional[Mapping[int, Sequence[int]]] = None,
) -> dict[str, Any]:
    """Derive the execution plan for one validated result.

    `loading_orders` maps a container's index to the engine-computed order its placements
    should be loaded in. It is optional and injected because the order is a statement about
    physics that this adapter must not make for itself; see `_steps`.
    """
    if result.get("status") is None:
        raise ExecutionPlanError("a result without a status is not a validated result")

    orders = dict(loading_orders or {})
    containers = list(result.get("containers") or ())
    winner_score = list(result.get("score") or ())

    plan_containers = []
    for index, container in enumerate(containers):
        sequence = _steps(index, container, orders.get(index))
        plan_containers.append({
            "container_index": index,
            "facts": {
                "container_type": container.get("container_type"),
                "placement_count": len(container.get("placements") or ()),
                "volume_utilization": container.get("volume_utilization"),
            },
            **sequence,
        })

    unplaced = [
        {
            "facts": {
                "item_type": item.get("item_type"),
                "reason": item.get("reason"),
                # The proof's `level` is carried through unchanged. Softening `observed`
                # into "could not fit" would turn an honest limit into a false certainty.
                "proof_level": (item.get("proof") or {}).get("level"),
                "details": list(item.get("details") or ()),
            },
            "presentation": {
                "summary": f"Not packed: {item.get('reason')} "
                           f"({(item.get('proof') or {}).get('level')}).",
                "cites": ["unpacked_items[].reason", "unpacked_items[].proof.level"],
            },
        }
        for item in (result.get("unpacked_items") or ())
    ]

    alternatives = [
        _alternative(index, winner_score, alternative)
        for index, alternative in enumerate(result.get("alternatives") or ())
    ]

    return {
        "format": FORMAT,
        "objective": result.get("objective"),
        "facts": {
            "status": result.get("status"),
            "score": winner_score,
            "feasibility": result.get("feasibility"),
            "optimality": result.get("optimality"),
            "container_count": len(containers),
        },
        "containers": plan_containers,
        # Often empty, and not for one reason. Four produce an empty list: the `fast`
        # profile runs a single solver, stops the start loop once the grid lattice
        # packs everything, only one start completed, or `alternatives: 1` -- the cap counts
        # the winner. Measured over the corpus, 165 of 399 requests do carry one, so this is
        # not the rare case an earlier draft of this comment claimed. An empty
        # list is well-formed and is never an error.
        "alternatives": alternatives,
        "unplaced": unplaced,
    }


def canonical_plan_json(plan: Mapping[str, Any]) -> str:
    """The one byte-comparable spelling of a plan.

    Cross-language equality is asserted on this string rather than on a parsed object, so
    key order and whitespace cannot make two identical plans look different -- the same
    discipline `packvium.commerce.canonical_json` applies to a quote.

    The spelling is RFC 8785, shared with the operational artifact. Until 1.3.0 this
    was `json.dumps(sort_keys=True)`, which is the same bytes for every plan an engine emits
    but not for a string holding U+2028, a key outside the Basic Multilingual Plane, or a
    float, where the four adapters disagreed.
    """
    return canonical_json(plan)
