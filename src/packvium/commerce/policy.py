"""Domain model for a versioned eligibility/policy rule engine.

Facility, customer, carrier, material, hazmat, temperature and service eligibility are
modelled as first-party, versioned, effective-dated predicates rather than an
ad-hoc if/else wall bolted onto the solver:

  * every rule is identified and versioned the same way `packvium.commerce.catalog`'s
    catalog is -- append-only per rule id, effective-dated, never mutated in place
    (`PolicyRule`, `PolicyRegistry.publish`) -- so "every rejection cites rule id and
    version" has a concrete version number to cite, not just a rule name;
  * a rule's predicate is a closed, validated vocabulary (`PolicyScope`, `PolicyOperator`)
    rather than an arbitrary callable, so a predicate this engine does not understand is a
    registration-time `UnsupportedPredicateError`, never a rule that gets silently
    admitted and then silently ignored during evaluation ("unsupported predicates fail
    admission instead of being ignored");
  * conflicting rules resolve deterministically: an explicit REJECT always outranks an
    ALLOW for the same context (deny-takes-precedence), and among several matching
    REJECTs the highest `priority` wins, ties broken by the lexicographically smallest
    rule id -- never by dict/set iteration order or by which rule happened to register
    first (`PolicyRegistry.evaluate`);
  * `evaluate()` is a pure function of its inputs with no side effect to "commit" or
    undo, so it doubles as its own dry-run interface ("dry-run evaluation is
    available") -- there is no separate code path that behaves differently once a
    decision is acted on.

Scope: this remains a solver-independent domain model. The dependency points inward
from `integration/product/policy_constraint.py`, which resolves an immutable versioned
rule snapshot and implements Packvium's native `PlacementConstraint` protocol. Policies
therefore reject candidates inside the solve rather than wrapping or rewriting output,
without making this lower layer import a solver.

Exported surface: this module is the one definition of the policy model in
the Python tree and ships inside the installed `packvium` distribution.
`domain/policy/model.py` re-exports it so every workspace import keeps resolving to
these exact objects. See docs/COMMERCE-API.md for the wrapper contract built on top.
"""

from __future__ import annotations

from .._compat import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


# --------------------------------------------------------------------------------- errors

class PolicyError(Exception):
    """Base class for every policy-domain error raised by this module."""


class UnsupportedPredicateError(PolicyError):
    """A rule's predicate names a scope or operator this engine version does not
    recognize. Raised at registration time -- an unsupported predicate is refused
    admission outright, never silently admitted and then ignored during evaluation."""


class PolicyRuleNotFoundError(PolicyError):
    """No rule is registered under the given rule id. Carries the id itself so an
    adapter reporting the rejection machine-readably (docs/COMMERCE-API.md's
    `policy_rule_not_found`) reads it off the exception rather than out of prose."""

    def __init__(self, message: str, *, rule_id: str) -> None:
        super().__init__(message)
        self.rule_id = rule_id


class PolicyVersionNotFoundError(PolicyError):
    """An explicitly referenced rule version number does not exist in that rule's
    history. Carries the id and the requested number, for the same reason."""

    def __init__(self, message: str, *, rule_id: str, version: int) -> None:
        super().__init__(message)
        self.rule_id = rule_id
        self.version = version


# --------------------------------------------------------------------------------- scope

class PolicyScope(str, Enum):
    """The named eligibility domains this task's description enumerates."""

    FACILITY = "facility"
    CUSTOMER = "customer"
    CARRIER = "carrier"
    MATERIAL = "material"
    HAZMAT = "hazmat"
    TEMPERATURE = "temperature"
    SERVICE = "service"


class PolicyOperator(str, Enum):
    """The closed set of predicate comparisons this engine understands. Anything outside
    this enum cannot even be constructed as a `PolicyPredicate` (see `__post_init__`),
    and `PolicyRegistry.publish` re-checks it defensively (see that method's docstring)
    so a rule built by hand rather than through this enum still fails admission."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    ABSENT = "absent"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"


_SUPPORTED_OPERATORS = frozenset(operator.value for operator in PolicyOperator)
_SUPPORTED_SCOPES = frozenset(scope.value for scope in PolicyScope)

#: Operators that take no `value` (a bare presence/absence check on `field`).
_UNARY_OPERATORS = frozenset({PolicyOperator.EXISTS, PolicyOperator.ABSENT})


@dataclass(frozen=True, slots=True)
class PolicyPredicate:
    """One condition over the caller-supplied context: `context[field] <op> value`
    (or, for `EXISTS`/`ABSENT`, just whether `field` is present)."""

    scope: PolicyScope
    field: str
    operator: PolicyOperator
    value: Any = None

    def __post_init__(self) -> None:
        # A caller may pass either the enum member or its raw string value (e.g. a
        # predicate deserialized from an untrusted wire payload); coerce to the enum
        # member so the rest of this module only ever compares members, and turn an
        # unrecognized value into this module's own `UnsupportedPredicateError` rather
        # than a generic `ValueError` from the stdlib enum machinery.
        try:
            scope = self.scope if isinstance(self.scope, PolicyScope) else PolicyScope(self.scope)
        except ValueError:
            raise UnsupportedPredicateError(f"unsupported policy scope {self.scope!r}") from None
        try:
            operator = self.operator if isinstance(self.operator, PolicyOperator) else PolicyOperator(self.operator)
        except ValueError:
            raise UnsupportedPredicateError(f"unsupported policy operator {self.operator!r}") from None
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "operator", operator)
        if not self.field:
            raise ValueError("field is required")
        if operator not in _UNARY_OPERATORS and self.value is None:
            raise ValueError(f"operator {operator.value!r} requires a value")

    def matches(self, context: Mapping[str, Any]) -> bool:
        if self.operator is PolicyOperator.EXISTS:
            return self.field in context
        if self.operator is PolicyOperator.ABSENT:
            return self.field not in context
        if self.field not in context:
            return False
        actual = context[self.field]
        if self.operator is PolicyOperator.EQUALS:
            return actual == self.value
        if self.operator is PolicyOperator.NOT_EQUALS:
            return actual != self.value
        if self.operator is PolicyOperator.IN:
            return actual in self.value
        if self.operator is PolicyOperator.NOT_IN:
            return actual not in self.value
        raise UnsupportedPredicateError(f"unsupported policy operator {self.operator!r}")  # pragma: no cover


# ----------------------------------------------------------------------------------- rule

@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One immutable, numbered version of one rule id's history (mirrors
    `packvium.commerce.catalog`'s `CatalogVersion`: append-only, never mutated in place).
    All of a rule id's predicates must match (logical AND) for the rule to apply."""

    rule_id: str
    version: int
    scope: PolicyScope
    action: PolicyAction
    predicates: tuple[PolicyPredicate, ...]
    priority: int
    effective_at: int
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id is required")
        if self.version <= 0:
            raise ValueError("version must be positive")
        if not self.predicates:
            raise ValueError("a rule must have at least one predicate")
        if self.effective_at < 0:
            raise ValueError("effective_at cannot be negative")
        mismatched = [p for p in self.predicates if p.scope is not self.scope]
        if mismatched:
            raise ValueError(f"every predicate of rule {self.rule_id!r} must share the rule's own scope")

    def matches(self, context: Mapping[str, Any]) -> bool:
        return all(predicate.matches(context) for predicate in self.predicates)


@dataclass(frozen=True, slots=True)
class PolicyCitation:
    """The evidence a decision cites: exactly which rule id/version produced it, and
    why. `None` when no rule matched at all (the open-by-default ALLOW case, see
    `PolicyRegistry.evaluate`'s docstring)."""

    rule_id: str
    version: int
    action: PolicyAction
    priority: int
    reason: str


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The outcome of one `evaluate()` call: whether the context is admitted for the
    given scope, and the single rule (if any) whose citation explains why."""

    scope: PolicyScope
    allowed: bool
    citation: Optional[PolicyCitation] = None

    def __post_init__(self) -> None:
        if not self.allowed and self.citation is None:
            raise ValueError("a REJECT decision must carry a citation")


# ------------------------------------------------------------------------------ registry

def _resolve_effective(versions: Sequence[PolicyRule], *, as_of: int) -> PolicyRule | None:
    """The same effective-dating resolution `CatalogRegistry._resolve_version` uses for
    `as_of` lookups: the highest `effective_at` not after `as_of`, ties broken by the
    higher (later-published) version number. `None` if no version of this rule id has
    yet taken effect by `as_of`."""
    candidates = [v for v in versions if v.effective_at <= as_of]
    if not candidates:
        return None
    return max(candidates, key=lambda v: (v.effective_at, v.version))


def decide(rules: Sequence[PolicyRule], scope: PolicyScope, context: Mapping[str, Any]) -> PolicyDecision:
    """Evaluate an already-resolved immutable rule set.

    Keeping resolution separate lets a solver adapter pin exact rule versions once per
    request and reuse them for every placement candidate. Evaluation is ``O(r * p)`` in
    matching rules and predicates, with no registry lookup and no mutable state.
    """
    matching = [rule for rule in rules if rule.scope is scope and rule.matches(context)]
    rejects = [rule for rule in matching if rule.action is PolicyAction.REJECT]
    allows = [rule for rule in matching if rule.action is PolicyAction.ALLOW]
    pool = rejects or allows
    if not pool:
        return PolicyDecision(scope=scope, allowed=True, citation=None)

    winner = min(pool, key=lambda rule: (-rule.priority, rule.rule_id))
    citation = PolicyCitation(
        rule_id=winner.rule_id, version=winner.version, action=winner.action,
        priority=winner.priority, reason=winner.reason,
    )
    return PolicyDecision(scope=scope, allowed=winner.action is PolicyAction.ALLOW, citation=citation)


class PolicyRegistry:
    """Per-tenant (or global) append-only history of policy rules, one history per
    `rule_id`. See the module docstring for what this contract does and does not cover."""

    def __init__(self) -> None:
        self._versions: dict[str, list[PolicyRule]] = {}

    def publish(
        self, rule_id: str, *, scope: PolicyScope, action: PolicyAction,
        predicates: Sequence[PolicyPredicate], priority: int, effective_at: int, reason: str = "",
    ) -> PolicyRule:
        """Append a new, numbered version for `rule_id`. Re-validates scope/operator
        support defensively (on top of `PolicyPredicate.__post_init__`) so a caller
        cannot bypass the "unsupported predicates fail admission" guarantee by
        constructing a predicate through any path other than the public enums."""
        for predicate in predicates:
            scope_value = predicate.scope.value if isinstance(predicate.scope, PolicyScope) else predicate.scope
            operator_value = (
                predicate.operator.value if isinstance(predicate.operator, PolicyOperator) else predicate.operator
            )
            if scope_value not in _SUPPORTED_SCOPES or operator_value not in _SUPPORTED_OPERATORS:
                raise UnsupportedPredicateError(
                    f"rule {rule_id!r} uses an unsupported scope/operator combination"
                )
        history = self._versions.setdefault(rule_id, [])
        rule = PolicyRule(
            rule_id=rule_id,
            version=len(history) + 1,
            scope=scope,
            action=action,
            predicates=tuple(predicates),
            priority=priority,
            effective_at=effective_at,
            reason=reason,
        )
        history.append(rule)
        return rule

    def versions(self, rule_id: str) -> tuple[PolicyRule, ...]:
        if rule_id not in self._versions:
            raise PolicyRuleNotFoundError(f"no rule registered under id {rule_id!r}", rule_id=rule_id)
        return tuple(self._versions[rule_id])

    def version(self, rule_id: str, number: int) -> PolicyRule:
        for rule in self.versions(rule_id):
            if rule.version == number:
                return rule
        raise PolicyVersionNotFoundError(
            f"rule {rule_id!r} has no version {number}", rule_id=rule_id, version=number,
        )

    def resolve_versions(self, versions: Sequence[tuple[str, int]]) -> tuple[PolicyRule, ...]:
        """Resolve and deterministically order an explicit policy snapshot."""
        pins = tuple(versions)
        if len({rule_id for rule_id, _ in pins}) != len(pins):
            raise ValueError("a policy snapshot cannot pin the same rule id twice")
        return tuple(self.version(rule_id, number) for rule_id, number in sorted(pins))

    def evaluate(self, scope: PolicyScope, context: Mapping[str, Any], *, as_of: int) -> PolicyDecision:
        """Resolve one deterministic decision for `scope`/`context` as of `as_of`.

        For every rule id, only the version effective as of `as_of` is considered (an
        as-yet-ineffective or not-yet-published version never participates). Among the
        rules of matching `scope` whose predicates all match `context`:

          * if any matching rule's action is REJECT, the REJECT with the highest
            `priority` wins (ties broken by the lexicographically smallest `rule_id`) --
            an explicit REJECT always outranks an ALLOW for the same context, so one
            permissive rule can never quietly override a more specific denial;
          * otherwise, if any matching rule's action is ALLOW, the highest-priority one
            (same tie-break) is cited;
          * if nothing matches at all, the context is allowed with no citation
            (open-by-default) -- a scope with zero registered rules is not the same as
            a scope where everything is rejected.

        This method has no side effect and nothing to undo, so it is itself the
        dry-run interface: dry-run evaluation is available without a second code path.
        """
        matching: list[PolicyRule] = []
        for rule_id, history in self._versions.items():
            effective = _resolve_effective(history, as_of=as_of)
            if effective is None or effective.scope is not scope:
                continue
            if effective.matches(context):
                matching.append(effective)

        return decide(matching, scope, context)
