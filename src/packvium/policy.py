"""Versioned eligibility rules, compiled into the engine's own constraint pipeline.

See `docs/POLICY-RULES.md` for the contract and the reasoning behind its shape. The
short version: rules travel in the request as data because an engine is driven over JSON
as a subprocess, so a rule registered inside one process has no wire representation and
nothing can check that four engines agree about it.

Deliberately not a predicate language. `Item.eligible_container_tags`,
`Item.incompatible_tags` and `Container.tag_limits` already express the predicates in
every engine, so each rule form here compiles to a constraint the pipeline already runs.
What a rule adds is only what tags cannot carry: identity, effective dating, priority,
and the shipment-scoped facts a request had nowhere to put.
"""

from __future__ import annotations

from ._compat import dataclass
from typing import Sequence

from .constraints import ConstraintContext, ConstraintResult

#: The shipment-scoped facts a rule may select on. Properties of the shipment rather
#: than of any item or container, which is why the request had nowhere to put them.
SHIPMENT_FACTS = ("facility", "customer", "carrier", "service")


class PolicyError(ValueError):
    """A rule set this engine cannot honour exactly as written.

    Structured rather than skipped: a rule silently dropped for being malformed would
    let a request pack in a way its own policy forbids, which is the failure the whole
    contract exists to prevent.
    """


@dataclass(frozen=True, slots=True)
class ShipmentContext:
    """Declared facts, or the absence of one. A fact nobody declared is not a wildcard:
    a rule naming it simply never participates, so an unstated facility cannot silently
    match a rule written for a specific one."""
    facility: str | None = None
    customer: str | None = None
    carrier: str | None = None
    service: str | None = None

    @classmethod
    def from_dict(cls, raw: object, where: str) -> "ShipmentContext":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise PolicyError(f"{where} must be an object")
        unknown = sorted(set(raw) - set(SHIPMENT_FACTS))
        if unknown:
            raise PolicyError(f"{where} names unknown shipment facts: {', '.join(unknown)}")
        for name, value in raw.items():
            if not isinstance(value, str) or not value:
                raise PolicyError(f"{where}.{name} must be a non-empty string")
        return cls(**{name: raw.get(name) for name in SHIPMENT_FACTS})

    def satisfied_by(self, shipment: "ShipmentContext") -> bool:
        """Whether every fact this selector names equals the shipment's own."""
        return all(
            getattr(self, name) is None or getattr(self, name) == getattr(shipment, name)
            for name in SHIPMENT_FACTS
        )


@dataclass(frozen=True, slots=True)
class SeparateTags:
    """No container may hold both tags. Compiles to the compatibility constraint."""
    tag: str
    from_tag: str


@dataclass(frozen=True, slots=True)
class RequireContainerTag:
    """An item carrying `item_tag` may only enter a container carrying `container_tag`.
    Compiles to the container-eligibility constraint."""
    item_tag: str
    container_tag: str


@dataclass(frozen=True, slots=True)
class LimitTagPerContainer:
    """At most `max_count` items carrying the tag may share one container. Compiles to
    the tag-count constraint."""
    tag: str
    max_count: int


#: Wire name -> (form class, required keys in constructor order). Exactly one may appear
#: on a rule: a rule naming two forms would have no single meaning for a citation.
FORMS = {
    "separate_tags": (SeparateTags, ("tag", "from_tag")),
    "require_container_tag": (RequireContainerTag, ("item_tag", "container_tag")),
    "limit_tag_per_container": (LimitTagPerContainer, ("tag", "max")),
}


@dataclass(frozen=True, slots=True)
class PolicyRule:
    id: str
    version: int
    effective_at: int
    priority: int
    applies_to: ShipmentContext
    form: SeparateTags | RequireContainerTag | LimitTagPerContainer

    @property
    def citation(self) -> str:
        return f"{self.id}@{self.version}"

    @classmethod
    def from_dict(cls, raw: object, index: int) -> "PolicyRule":
        where = f"policy.rules[{index}]"
        if not isinstance(raw, dict):
            raise PolicyError(f"{where} must be an object")
        named = [name for name in FORMS if name in raw]
        if len(named) != 1:
            raise PolicyError(
                f"{where} must name exactly one rule form ({', '.join(sorted(FORMS))}), "
                f"not {len(named)}"
            )
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise PolicyError(f"{where}.id must be a non-empty string")
        version = _integer(raw.get("version"), f"{where}.version", minimum=1)
        return cls(
            id=identifier,
            version=version,
            effective_at=_integer(raw.get("effective_at"), f"{where}.effective_at", minimum=0),
            priority=_integer(raw.get("priority"), f"{where}.priority", minimum=0),
            applies_to=ShipmentContext.from_dict(raw.get("applies_to"), f"{where}.applies_to"),
            form=_form(named[0], raw[named[0]], f"{where}.{named[0]}"),
        )


def _integer(value: object, where: str, *, minimum: int) -> int:
    # `bool` is an `int` in Python and would pass a naive check, quietly turning
    # `"version": true` into version 1.
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PolicyError(f"{where} must be an integer >= {minimum}")
    return value


def _form(name: str, raw: object, where: str):
    form_class, keys = FORMS[name]
    if not isinstance(raw, dict):
        raise PolicyError(f"{where} must be an object")
    unknown = sorted(set(raw) - set(keys))
    if unknown:
        raise PolicyError(f"{where} has unknown keys: {', '.join(unknown)}")
    missing = [key for key in keys if key not in raw]
    if missing:
        raise PolicyError(f"{where} is missing {', '.join(missing)}")
    values = []
    for key in keys:
        value = raw[key]
        if key == "max":
            values.append(_integer(value, f"{where}.max", minimum=0))
        elif not isinstance(value, str) or not value:
            raise PolicyError(f"{where}.{key} must be a non-empty string")
        else:
            values.append(value)
    return form_class(*values)


@dataclass(frozen=True, slots=True)
class PolicyRuleSet:
    """The rules that participate in one request, already resolved and ordered.

    Resolution is fixed by the contract and must be identical in every engine, or the
    same request packs differently depending on which one answered it.
    """
    rules: tuple[PolicyRule, ...]

    @classmethod
    def from_dict(cls, raw: object) -> "PolicyRuleSet":
        if raw is None:
            return cls(())
        if not isinstance(raw, dict):
            raise PolicyError("policy must be an object")
        unknown = sorted(set(raw) - {"as_of", "shipment", "rules"})
        if unknown:
            raise PolicyError(f"policy has unknown keys: {', '.join(unknown)}")
        declared = raw.get("rules") or []
        if not isinstance(declared, list):
            raise PolicyError("policy.rules must be an array")
        if not declared:
            return cls(())
        if "as_of" not in raw:
            # No default: a guessed instant silently activates or hides a restriction,
            # and reading a clock here would make one request pack differently on
            # different days.
            raise PolicyError("policy.as_of is required whenever policy.rules is non-empty")
        as_of = _integer(raw.get("as_of"), "policy.as_of", minimum=0)
        shipment = ShipmentContext.from_dict(raw.get("shipment"), "policy.shipment")
        parsed = [PolicyRule.from_dict(rule, index) for index, rule in enumerate(declared)]
        return cls(cls._resolve(parsed, as_of, shipment))

    @staticmethod
    def _resolve(
        rules: Sequence[PolicyRule], as_of: int, shipment: ShipmentContext
    ) -> tuple[PolicyRule, ...]:
        participating = [
            rule for rule in rules
            if rule.effective_at <= as_of and rule.applies_to.satisfied_by(shipment)
        ]
        # Append-only per id: among participating versions of one id the highest
        # `effective_at` wins, ties broken by the highest `version`. The same resolution
        # the catalog registry already uses for `as_of` lookups, deliberately, so a
        # reader learns one rule and not two.
        latest: dict[str, PolicyRule] = {}
        for rule in participating:
            current = latest.get(rule.id)
            if current is None or (rule.effective_at, rule.version) > (current.effective_at, current.version):
                latest[rule.id] = rule
        # Citation order, not evaluation order: the first rule that rejects a candidate
        # is the one cited, so sorting here is what makes the citation deterministic.
        # Ties go to the lexicographically smallest id -- never to dict iteration order,
        # which would make the citation depend on the order the caller happened to write.
        return tuple(sorted(latest.values(), key=lambda rule: (-rule.priority, rule.id)))

    def constraints(self) -> tuple["PolicyConstraint", ...]:
        return (PolicyConstraint(self.rules),) if self.rules else ()


@dataclass(frozen=True, slots=True)
class PolicyConstraint:
    """Rejects a candidate placement that any participating rule forbids.

    `O(m + r)` per candidate for `m` placements already in the container and `r`
    resolved rules: one pass collecting the tags present, then one pass over the rules.
    That is the same bound class as the built-in compatibility and tag-count
    constraints it compiles onto, so the published complexity bounds are unchanged.

    Rules are evaluated in citation order, so the first rejection is already the one the
    contract says to cite -- highest priority, ties to the smallest id.
    """
    rules: tuple[PolicyRule, ...]

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        item_tags = context.item.item.tags
        container_tags = context.container.tags
        present: dict[str, int] | None = None
        for rule in self.rules:
            form = rule.form
            if isinstance(form, RequireContainerTag):
                if form.item_tag in item_tags and form.container_tag not in container_tags:
                    return self._reject(rule, f"requires a container tagged {form.container_tag!r}")
                continue
            # Both remaining forms need to know what is already in the container, so the
            # walk is done once and only when a rule of that kind exists at all.
            if present is None:
                present = _tag_counts(context.placements)
            if isinstance(form, SeparateTags):
                if form.tag in item_tags and present.get(form.from_tag):
                    return self._reject(rule, f"{form.tag!r} may not share a container with {form.from_tag!r}")
                if form.from_tag in item_tags and present.get(form.tag):
                    return self._reject(rule, f"{form.from_tag!r} may not share a container with {form.tag!r}")
            elif form.tag in item_tags and present.get(form.tag, 0) >= form.max_count:
                return self._reject(rule, f"at most {form.max_count} item(s) tagged {form.tag!r} per container")
        return ConstraintResult.allow()

    def proves_unplaceable(self, item, containers) -> tuple[str, str] | None:
        """The rule that rules this item out of every offered container, if one does.

        Only `require_container_tag` can be answered here, and that is not a gap. It is
        a statement about the request alone -- this item carries the tag, no offered
        container carries the one it requires -- so it holds however the search goes.
        Segregation and per-container caps depend on what else was packed, so an item
        they leave behind was left behind by the search, and reporting that as proven
        would claim more than the engine knows.

        `O(r * c)` for `r` rules and `c` containers, once per unpacked item rather than
        per candidate.
        """
        for rule in self.rules:
            form = rule.form
            if not isinstance(form, RequireContainerTag):
                continue
            if form.item_tag not in item.item.tags:
                continue
            if not any(form.container_tag in container.tags for container in containers):
                return (
                    "policy_rule",
                    f"{rule.citation}: requires a container tagged "
                    f"{form.container_tag!r}, which none of the containers offered carries",
                )
        return None

    @staticmethod
    def _reject(rule: PolicyRule, why: str) -> ConstraintResult:
        return ConstraintResult.reject("policy_rule", f"{rule.citation}: {why}")


def _tag_counts(placements) -> dict[str, int]:
    counts: dict[str, int] = {}
    for placement in placements:
        for tag in placement.instance.item.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts
