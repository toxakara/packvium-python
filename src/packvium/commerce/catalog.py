"""Domain model for versioned item, carton and pallet master data.

A packing decision is only as trustworthy as the master data it was made against. This
module gives SKUs, cartons, pallets, exclusion rules and facility-specific overrides a
first-party, immutable, effective-dated catalog so that:

  * every request resolves to concrete, numbered catalog versions rather than a
    moving "current" — a result can record exactly what master data it used
    (`CatalogReference`, `ResolvedCatalog`);
  * publishing a new version can never reach back and change a reference or snapshot
    a caller already resolved, because every object this module hands out is a frozen
    dataclass and `CatalogRegistry` only ever *appends* to its history
    (`CatalogSnapshot`, `CatalogVersion`, `CatalogRegistry.publish`);
  * a version can be rolled back, and a rollback is itself a new, higher-numbered
    version — history is never rewritten in place (`CatalogRegistry.rollback`);
  * a version can be published now but only take effect later, so future-dated
    corrections can be queued ahead of time (`CatalogVersion.effective_at`,
    `CatalogRegistry.resolve(as_of=...)`);
  * a reference that is missing, ambiguous, or predates any effective version is a
    distinct, structured rejection rather than a silent wrong answer
    (`CatalogVersionNotFoundError`, `AmbiguousCatalogReferenceError`,
    `NoEffectiveCatalogVersionError`, `CatalogEntryNotFoundError`);
  * an old decision's recorded `CatalogReference` still resolves to the exact data it
    was made against after the catalog has since been corrected, rolled back, or
    republished (`CatalogRegistry.resolve_reference`) — historical replay stays
    reproducible, and bad master data can be told apart from a solver defect.

Scope note: this module intentionally does not import `packvium.units.Length` /
`Weight`. Versioning, effective-dating and reference resolution are the concern here,
not geometry or unit parsing, so master-data fields use plain non-negative integer
millimetres/grams rather than pulling in a per-language runtime package. See
docs/CATALOG-VERSIONING.md for the wire-format proposal this model backs.

Follows the same conventions as packvium.models and optimizer/spec/model.py: frozen,
slotted dataclasses, `__post_init__` validation, no floats, no duplicated logic across
kinds (see the module docstring above and docs/OBJECTIVE.md's exact-arithmetic rationale
for why this codebase avoids floats generally).

Exported surface: this module is the one definition of the catalog model in
the Python tree and ships inside the installed `packvium` distribution.
`domain/catalog/model.py` re-exports it so every workspace import keeps resolving to
these exact objects. See docs/COMMERCE-API.md for the wrapper contract built on top.
"""

from __future__ import annotations

from .._compat import dataclass
from dataclasses import fields
from enum import Enum
from types import MappingProxyType
from typing import Sequence


# --------------------------------------------------------------------------------- errors

class CatalogError(Exception):
    """Base class for every catalog-domain error raised by this module."""


class CatalogEntryNotFoundError(CatalogError):
    """A referenced item, carton or pallet id does not exist within a resolved snapshot."""


class CatalogVersionNotFoundError(CatalogError):
    """An explicitly referenced catalog version number does not exist in the history."""


class NoEffectiveCatalogVersionError(CatalogError):
    """An as-of time was resolved before any catalog version had become effective."""


class AmbiguousCatalogReferenceError(CatalogError):
    """A reference gave neither an explicit version nor an as-of time while more than one
    version exists in the catalog's history, so which version is "the" answer is
    undefined. Precisely: `resolve(version=None, as_of=None)` is ambiguous if and only if
    the catalog has more than one published version; with zero or exactly one version the
    answer is unambiguous and is returned instead of raising.
    """


# ------------------------------------------------------------------------------- entries

class CatalogEntryKind(str, Enum):
    ITEM = "item"
    CARTON = "carton"
    PALLET = "pallet"


def _require_id(label: str, id: str) -> None:
    if not id:
        raise ValueError(f"{label} id is required")


def _require_positive_dimensions(label: str, dimensions_mm: Sequence[int]) -> None:
    if any(d <= 0 for d in dimensions_mm):
        raise ValueError(f"{label} dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ItemMaster:
    """A first-party SKU master record: the dimensions and weight a catalog version pins
    for one stock keeping unit, independent of any one packing request."""

    id: str
    dimensions_mm: tuple[int, int, int]
    weight_g: int
    description: str = ""

    def __post_init__(self) -> None:
        _require_id("item", self.id)
        if len(self.dimensions_mm) != 3:
            raise ValueError("item dimensions must have exactly three axes")
        _require_positive_dimensions("item", self.dimensions_mm)
        if self.weight_g <= 0:
            raise ValueError("item weight must be positive")


@dataclass(frozen=True, slots=True)
class CartonMaster:
    """A first-party carton (box) master record a catalog version pins."""

    id: str
    inner_dimensions_mm: tuple[int, int, int]
    max_payload_g: int
    cost_minor: int = 0

    def __post_init__(self) -> None:
        _require_id("carton", self.id)
        if len(self.inner_dimensions_mm) != 3:
            raise ValueError("carton dimensions must have exactly three axes")
        _require_positive_dimensions("carton", self.inner_dimensions_mm)
        if self.max_payload_g <= 0:
            raise ValueError("carton max_payload_g must be positive")
        if self.cost_minor < 0:
            raise ValueError("cost_minor cannot be negative")


@dataclass(frozen=True, slots=True)
class PalletMaster:
    """A first-party pallet master record a catalog version pins."""

    id: str
    deck_dimensions_mm: tuple[int, int]
    max_payload_g: int
    max_stack_height_mm: int | None = None

    def __post_init__(self) -> None:
        _require_id("pallet", self.id)
        if len(self.deck_dimensions_mm) != 2:
            raise ValueError("pallet deck dimensions must have exactly two axes")
        _require_positive_dimensions("pallet", self.deck_dimensions_mm)
        if self.max_payload_g <= 0:
            raise ValueError("pallet max_payload_g must be positive")
        if self.max_stack_height_mm is not None and self.max_stack_height_mm <= 0:
            raise ValueError("max_stack_height_mm must be positive")


class ExclusionScope(str, Enum):
    ITEM_CARTON = "item_carton"
    ITEM_PALLET = "item_pallet"


@dataclass(frozen=True, slots=True)
class ExclusionRule:
    """A first-party rule forbidding one master-data entry from being packed with
    another — e.g. a hazmat SKU forbidden from a given carton type."""

    id: str
    scope: ExclusionScope
    subject_id: str
    excluded_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        _require_id("exclusion", self.id)
        if not self.subject_id or not self.excluded_id:
            raise ValueError("an exclusion rule must reference both a subject and an excluded id")


@dataclass(frozen=True, slots=True)
class FacilityOverride:
    """A facility-specific override of a base master-data entry — e.g. facility 'DC-12'
    stocks carton 'box-m' at different inner dimensions than the network default.

    `entry_kind` is derived from `override`'s type rather than stored separately, so the
    two can never disagree.
    """

    id: str
    facility_id: str
    entry_id: str
    override: ItemMaster | CartonMaster | PalletMaster

    def __post_init__(self) -> None:
        _require_id("facility override", self.id)
        if not self.facility_id:
            raise ValueError("facility_id is required")
        if self.override.id != self.entry_id:
            raise ValueError("a facility override's entry_id must match override.id")

    @property
    def entry_kind(self) -> CatalogEntryKind:
        if isinstance(self.override, ItemMaster):
            return CatalogEntryKind.ITEM
        if isinstance(self.override, CartonMaster):
            return CatalogEntryKind.CARTON
        return CatalogEntryKind.PALLET


# ------------------------------------------------------------------------------ snapshot

def _require_unique_ids(label: str, entries: Sequence) -> None:
    ids = [entry.id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {label} ids in catalog snapshot")


def _find(entries: Sequence, id: str, kind: CatalogEntryKind):
    for entry in entries:
        if entry.id == id:
            return entry
    raise CatalogEntryNotFoundError(f"no {kind.value} with id {id!r} in this catalog snapshot")


class _SnapshotLookupCache:
    __slots__ = ("_entry_indexes",)


@dataclass(frozen=True, slots=True)
class CatalogSnapshot(_SnapshotLookupCache):
    """The complete, immutable content of one catalog version: every item, carton and
    pallet master record, exclusion rule and facility override active under that version.

    Built entirely from frozen dataclasses and stored as tuples, so a `CatalogSnapshot`
    handed out by `CatalogRegistry.resolve()` can never be mutated by a later `publish()`
    — the invariant that makes concurrent publication safe (the publication contract:
    "concurrent catalog publication cannot change an in-flight request").
    """

    items: tuple[ItemMaster, ...] = ()
    cartons: tuple[CartonMaster, ...] = ()
    pallets: tuple[PalletMaster, ...] = ()
    exclusions: tuple[ExclusionRule, ...] = ()
    overrides: tuple[FacilityOverride, ...] = ()

    def __post_init__(self) -> None:
        _require_unique_ids("item", self.items)
        _require_unique_ids("carton", self.cartons)
        _require_unique_ids("pallet", self.pallets)
        _require_unique_ids("exclusion", self.exclusions)
        _require_unique_ids("facility override", self.overrides)

    def item(self, id: str) -> ItemMaster:
        return self._lookup(self.items, id, CatalogEntryKind.ITEM, ItemMaster)

    def carton(self, id: str) -> CartonMaster:
        return self._lookup(self.cartons, id, CatalogEntryKind.CARTON, CartonMaster)

    def pallet(self, id: str) -> PalletMaster:
        return self._lookup(self.pallets, id, CatalogEntryKind.PALLET, PalletMaster)


    def _lookup(self, entries, id, kind, entry_type):
        if type(entries) is not tuple:
            return _find(entries, id, kind)
        indexes = getattr(self, "_entry_indexes", {})
        index = indexes.get(kind)
        if index is None:
            index = (MappingProxyType({entry.id: entry for entry in entries})
                     if all(type(entry) is entry_type and type(entry.id) is str for entry in entries)
                     else False)
            object.__setattr__(self, "_entry_indexes", {**indexes, kind: index})
        if index is False or type(id) is not str:
            return _find(entries, id, kind)
        try:
            return index[id]
        except KeyError:
            raise CatalogEntryNotFoundError(f"no {kind.value} with id {id!r} in this catalog snapshot") from None

    def __getstate__(self):
        # Derived indexes are not part of the value, serialization or a copied snapshot.
        return [getattr(self, field.name) for field in fields(self)]

    def __setstate__(self, state):
        # Also accept the dictionary state produced by older non-slotted runtimes.
        for index, field in enumerate(fields(self)):
            object.__setattr__(self, field.name, state[field.name] if isinstance(state, dict) else state[index])


# ------------------------------------------------------------------------------- version

@dataclass(frozen=True, slots=True)
class CatalogVersion:
    """One immutable, numbered entry in a catalog's append-only publication history.

    `effective_at` is when this version starts governing lookups (future-effective
    versions are supported — a version may be published now but not take effect until
    later). `published_at` is when the publication itself was recorded, purely for audit;
    it never affects resolution. A rollback is not a mutation of history — it is a new,
    higher-numbered version whose snapshot equals a prior one's, recorded via
    `rolled_back_from`.
    """

    number: int
    snapshot: CatalogSnapshot
    effective_at: int
    published_at: int
    rolled_back_from: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.number <= 0:
            raise ValueError("version number must be positive")
        if self.effective_at < 0:
            raise ValueError("effective_at cannot be negative")
        if self.published_at < 0:
            raise ValueError("published_at cannot be negative")
        if self.rolled_back_from is not None and self.rolled_back_from <= 0:
            raise ValueError("rolled_back_from must reference a positive version number")


# ----------------------------------------------------------------------------- reference

@dataclass(frozen=True, slots=True)
class CatalogReference:
    """A pinned pointer to one concrete, immutable catalog version.

    Embeddable directly in a packing result so every catalog version used is recorded
    literally in the result, not merely re-derivable
    after the fact. See `domain/catalog/schema/catalog-versions-used.schema.json` for the proposed wire
    shape and `as_dict()` for the exact serialization it validates against.
    """

    catalog_id: str
    version: int
    effective_at: int
    resolved_at: int

    def __post_init__(self) -> None:
        if not self.catalog_id:
            raise ValueError("catalog_id is required")
        if self.version <= 0:
            raise ValueError("version must be positive")
        if self.effective_at < 0:
            raise ValueError("effective_at cannot be negative")
        if self.resolved_at < 0:
            raise ValueError("resolved_at cannot be negative")

    def as_dict(self) -> dict[str, int | str]:
        """The wire representation matching `domain/catalog/schema/catalog-versions-used.schema.json`."""
        return {
            "catalog_id": self.catalog_id,
            "version": self.version,
            "effective_at": self.effective_at,
            "resolved_at": self.resolved_at,
        }


@dataclass(frozen=True, slots=True)
class ResolvedCatalog:
    """The concrete, frozen result of resolving a `CatalogReference`: one version's
    snapshot plus the reference that pins it.

    Handed to a solver as "the" master data for one request. Because both `reference` and
    `snapshot` are frozen and nothing in `CatalogRegistry.publish()` reaches back into
    already-issued objects, a `ResolvedCatalog` is safe to hold across an "in-flight"
    request while other publications happen concurrently.
    """

    reference: CatalogReference
    snapshot: CatalogSnapshot

    def item(self, id: str) -> ItemMaster:
        return self.snapshot.item(id)

    def carton(self, id: str) -> CartonMaster:
        return self.snapshot.carton(id)

    def pallet(self, id: str) -> PalletMaster:
        return self.snapshot.pallet(id)


# ------------------------------------------------------------------------------ registry

class CatalogRegistry:
    """Per-catalog (e.g. per-warehouse or per-tenant) append-only publication history.

    This is the one place catalog state can change; every object it hands out
    (`CatalogVersion`, `ResolvedCatalog`, `CatalogSnapshot`, `CatalogReference`) is frozen,
    so publishing a new version can only ever append to the history — it can never reach
    back and mutate something a caller already resolved. Concurrent publication cannot
    change an in-flight request.
    """

    def __init__(self, catalog_id: str) -> None:
        if not catalog_id:
            raise ValueError("catalog_id is required")
        self._catalog_id = catalog_id
        self._versions: list[CatalogVersion] = []

    @property
    def catalog_id(self) -> str:
        return self._catalog_id

    @property
    def versions(self) -> tuple[CatalogVersion, ...]:
        """The full append-only history, oldest first. Never mutated in place."""
        return tuple(self._versions)

    def publish(
        self, snapshot: CatalogSnapshot, *, effective_at: int, published_at: int, note: str = ""
    ) -> CatalogVersion:
        """Append a new version. `effective_at` may be in the future relative to
        `published_at` (future-effective publication) or in the past (backdated
        correction) — both are valid; only ordering of `effective_at` values across
        versions affects what `resolve()` later returns."""
        version = CatalogVersion(
            number=len(self._versions) + 1,
            snapshot=snapshot,
            effective_at=effective_at,
            published_at=published_at,
            note=note,
        )
        self._versions.append(version)
        return version

    def rollback(
        self, to_version: int, *, published_at: int, effective_at: int | None = None, note: str = ""
    ) -> CatalogVersion:
        """Publish a new version whose snapshot equals a prior version's, recorded as a
        rollback via `rolled_back_from`. History is append-only: `to_version` and every
        version after it remain in the history untouched."""
        target = self._version(to_version)
        version = CatalogVersion(
            number=len(self._versions) + 1,
            snapshot=target.snapshot,
            effective_at=published_at if effective_at is None else effective_at,
            published_at=published_at,
            rolled_back_from=to_version,
            note=note or f"rollback to version {to_version}",
        )
        self._versions.append(version)
        return version

    def resolve(
        self, *, resolved_at: int, version: int | None = None, as_of: int | None = None
    ) -> ResolvedCatalog:
        """Resolve a concrete `ResolvedCatalog`, pinned by an explicit `version` number or
        by the version effective `as_of` a given time. Raises `CatalogVersionNotFoundError`
        for an unknown explicit version, `NoEffectiveCatalogVersionError` if `as_of`
        predates every version, and `AmbiguousCatalogReferenceError` if neither is given
        while more than one version exists.
        """
        target = self._resolve_version(version=version, as_of=as_of)
        reference = CatalogReference(
            catalog_id=self._catalog_id,
            version=target.number,
            effective_at=target.effective_at,
            resolved_at=resolved_at,
        )
        return ResolvedCatalog(reference=reference, snapshot=target.snapshot)

    def resolve_reference(self, reference: CatalogReference, *, resolved_at: int) -> ResolvedCatalog:
        """Re-resolve a previously recorded `CatalogReference` (historical replay
        stays reproducible). Always resolves by the reference's pinned version number,
        never by re-deriving "current" or "as of", so a corrected, rolled-back or
        republished catalog can never change what an old decision replays to.
        """
        if reference.catalog_id != self._catalog_id:
            raise CatalogError(
                f"reference is for catalog {reference.catalog_id!r}, not {self._catalog_id!r}"
            )
        return self.resolve(resolved_at=resolved_at, version=reference.version)

    def _resolve_version(self, *, version: int | None, as_of: int | None) -> CatalogVersion:
        if version is not None:
            return self._version(version)
        if as_of is None:
            if len(self._versions) > 1:
                raise AmbiguousCatalogReferenceError(
                    f"catalog {self._catalog_id!r} has {len(self._versions)} versions; "
                    "resolve() requires an explicit version or as_of to avoid an ambiguous 'current'"
                )
            if not self._versions:
                raise CatalogVersionNotFoundError(f"catalog {self._catalog_id!r} has no published versions")
            return self._versions[0]
        target = None
        for candidate in self._versions:
            if candidate.effective_at <= as_of and (target is None or candidate.effective_at >= target.effective_at):
                target = candidate
        if target is None:
            raise NoEffectiveCatalogVersionError(
                f"catalog {self._catalog_id!r} has no version effective as of {as_of}"
            )
        # Ties in effective_at are broken by the higher (later-published) version number,
        # so a same-instant correction or rollback deterministically wins rather than
        # being ambiguous.
        return target

    def _version(self, number: int) -> CatalogVersion:
        if type(number) is int:
            if 1 <= number <= len(self._versions):
                return self._versions[number - 1]
        else:
            # Preserve equality-based resolution for existing non-int callers.
            for v in self._versions:
                if v.number == number:
                    return v
        raise CatalogVersionNotFoundError(f"catalog {self._catalog_id!r} has no version {number}")
