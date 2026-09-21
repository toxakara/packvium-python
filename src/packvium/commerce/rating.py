"""Domain model for a versioned carrier rating / landed-cost engine.

Carrier services, dimensional-weight divisors, zone rates, minimum charges, fuel and
accessorial surcharges are modelled as versioned first-party rules, the same append-only
discipline `packvium.commerce.catalog`'s catalog and `packvium.commerce.policy`'s policy rules
already use:

  * every `Tariff` is a numbered, effective-dated version of one `(carrier_id,
    service_id)` pair, never mutated in place -- a `RateBreakdown`'s `tariff_version`
    is a citation someone can independently look up later, not a label that could have
    silently drifted ("each alternative carries an auditable rate breakdown and tariff
    version");
  * dimensional weight, the minimum-charge floor, the fuel surcharge and every
    accessorial surcharge are computed with exact integer arithmetic only -- ticks for
    length, minor currency units (cents) for cost, permille (parts-per-1000) for
    percentage-shaped rates -- and any division that is not exact rounds up
    (`_ceil_div`), never down and never through a float ("dimensional weight and
    surcharges are exact");
  * a zone this tariff has no rate for, or a requested accessorial this tariff does not
    define, is a structured `UnavailableServiceError` naming exactly which zone or
    accessorial id was missing -- never a silently-zero or silently-skipped charge
    ("unavailable services are structured rejections");
  * `rate_with_version` pins an explicit tariff version rather than resolving "current",
    so a stored `RateBreakdown` can be reproduced byte-for-byte later purely from its own
    `tariff_id`/`tariff_version`, independent of whatever the registry's history has
    grown to since ("offline deterministic replay is possible").

Scope: this module computes a rate breakdown from first-party tariff data a caller
publishes into `CarrierRegistry` -- it does not fetch, scrape or embed any real carrier's
published rates (the illustrative tariffs in this module's own tests are synthetic).
The dependency remains one-way: `commerce/rating/objective.py` (workspace) adapts this independent
domain model to Packvium's solution scorer and container selector, so the exact
`RateBreakdown.total_minor` participates in selection without importing a solver here.

Exported surface: this module is the one definition of the rating model in
the Python tree and ships inside the installed `packvium` distribution.
`commerce/rating/model.py` re-exports it so every workspace import keeps resolving to
these exact objects. See docs/COMMERCE-API.md for the wrapper contract built on top.
"""

from __future__ import annotations

from .._compat import dataclass
from typing import Mapping, Optional


# --------------------------------------------------------------------------------- errors

class RatingError(Exception):
    """Base class for every rating-domain error raised by this module."""


class UnavailableServiceError(RatingError):
    """The requested zone or accessorial is not defined by the resolved tariff -- a
    structured rejection naming exactly what was missing, never a silently-zero charge.

    Exactly one of `zone` / `accessorial_ids` is set, so a caller that has to report
    the rejection in a machine-readable form (docs/COMMERCE-API.md's `unavailable_zone`
    and `unavailable_accessorial` codes) reads it off the exception instead of parsing
    the message back out of prose."""

    def __init__(
        self, message: str, *, zone: Optional[str] = None, accessorial_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.zone = zone
        self.accessorial_ids = tuple(accessorial_ids)


class TariffNotFoundError(RatingError):
    """No tariff is registered under the given carrier/service id, or no version of it
    is effective at the requested time / exists at the requested version number."""


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division -- every inexact division in this module (dimensional
    weight, permille-based surcharges) rounds up, never down, and never through a float."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


# --------------------------------------------------------------------------------- tariff

@dataclass(frozen=True, slots=True)
class AccessorialCharge:
    """One named accessorial (e.g. residential delivery, liftgate, signature-required),
    either a flat charge or a permille-of-base charge -- never both."""

    accessorial_id: str
    flat_charge_minor: Optional[int] = None
    permille_of_base: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.accessorial_id:
            raise ValueError("accessorial_id is required")
        has_flat = self.flat_charge_minor is not None
        has_permille = self.permille_of_base is not None
        if has_flat == has_permille:
            raise ValueError("an accessorial must set exactly one of flat_charge_minor or permille_of_base")
        if has_flat and self.flat_charge_minor < 0:
            raise ValueError("flat_charge_minor cannot be negative")
        if has_permille and self.permille_of_base < 0:
            raise ValueError("permille_of_base cannot be negative")

    def charge_minor(self, base_charge_minor: int) -> int:
        if self.flat_charge_minor is not None:
            return self.flat_charge_minor
        return _ceil_div(base_charge_minor * self.permille_of_base, 1000)


@dataclass(frozen=True, slots=True)
class Tariff:
    """One immutable, numbered version of one `(carrier_id, service_id)` pair's rate
    card. Mirrors `packvium.commerce.catalog`'s `CatalogVersion`: append-only, effective-
    dated, never mutated in place."""

    carrier_id: str
    service_id: str
    version: int
    effective_at: int
    dimensional_weight_divisor: int
    cost_per_dimensional_kg_minor: Mapping[str, int]  # zone -> minor cost per kg (1000 g)
    minimum_charge_minor: int
    fuel_surcharge_permille: int
    accessorials: Mapping[str, AccessorialCharge]

    def __post_init__(self) -> None:
        if not self.carrier_id:
            raise ValueError("carrier_id is required")
        if not self.service_id:
            raise ValueError("service_id is required")
        if self.version <= 0:
            raise ValueError("version must be positive")
        if self.effective_at < 0:
            raise ValueError("effective_at cannot be negative")
        if self.dimensional_weight_divisor <= 0:
            raise ValueError("dimensional_weight_divisor must be positive")
        if self.minimum_charge_minor < 0:
            raise ValueError("minimum_charge_minor cannot be negative")
        if self.fuel_surcharge_permille < 0:
            raise ValueError("fuel_surcharge_permille cannot be negative")
        if any(cost < 0 for cost in self.cost_per_dimensional_kg_minor.values()):
            raise ValueError("cost_per_dimensional_kg_minor entries cannot be negative")
        mismatched = [key for key, value in self.accessorials.items() if key != value.accessorial_id]
        if mismatched:
            raise ValueError(f"accessorials dict key must match its own accessorial_id: {mismatched}")


# ---------------------------------------------------------------------------- breakdown

@dataclass(frozen=True, slots=True)
class RateBreakdown:
    """The fully itemized, auditable result of one `rate()`/`rate_with_version()` call.
    Every component that contributed to `total_minor` is named individually, and the
    exact tariff version that produced it is recorded alongside."""

    carrier_id: str
    service_id: str
    tariff_version: int
    zone: str
    actual_weight_g: int
    dimensional_weight_g: int
    billed_weight_g: int
    base_charge_minor: int
    minimum_charge_applied: bool
    fuel_surcharge_minor: int
    accessorial_charges_minor: tuple[tuple[str, int], ...]
    total_minor: int


# ------------------------------------------------------------------------------ request

@dataclass(frozen=True, slots=True)
class RatingRequest:
    """What is being rated: a shipment's real weight and volume, the zone it is moving
    to, and whichever accessorial services this specific shipment needs."""

    zone: str
    actual_weight_g: int
    volume_mm3: int
    requested_accessorials: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.zone:
            raise ValueError("zone is required")
        if self.actual_weight_g < 0:
            raise ValueError("actual_weight_g cannot be negative")
        if self.volume_mm3 < 0:
            raise ValueError("volume_mm3 cannot be negative")
        if any(not accessorial for accessorial in self.requested_accessorials):
            raise ValueError("requested accessorial ids must be non-empty")
        if len(set(self.requested_accessorials)) != len(self.requested_accessorials):
            raise ValueError("requested accessorial ids must be unique")


def rate_tariff(tariff: Tariff, request: RatingRequest) -> RateBreakdown:
    """Rate a request against one already-resolved immutable tariff version."""
    if request.zone not in tariff.cost_per_dimensional_kg_minor:
        raise UnavailableServiceError(
            f"tariff {tariff.carrier_id}/{tariff.service_id} v{tariff.version} has no rate for "
            f"zone {request.zone!r}",
            zone=request.zone,
        )
    unknown_accessorials = set(request.requested_accessorials) - set(tariff.accessorials)
    if unknown_accessorials:
        raise UnavailableServiceError(
            f"tariff {tariff.carrier_id}/{tariff.service_id} v{tariff.version} does not offer "
            f"accessorial(s) {sorted(unknown_accessorials)}",
            accessorial_ids=tuple(sorted(unknown_accessorials)),
        )

    # 1 ticks-cubed volume unit maps to 1 mm^3 in this repo's own units convention
    # (packvium.units); dimensional weight in grams is volume (mm^3) / divisor,
    # rounded up -- never down, never through a float.
    dimensional_weight_g = _ceil_div(request.volume_mm3, tariff.dimensional_weight_divisor)
    billed_weight_g = max(request.actual_weight_g, dimensional_weight_g)

    rate_per_kg = tariff.cost_per_dimensional_kg_minor[request.zone]
    raw_base_charge_minor = _ceil_div(billed_weight_g * rate_per_kg, 1000)
    minimum_applied = raw_base_charge_minor < tariff.minimum_charge_minor
    base_charge_minor = tariff.minimum_charge_minor if minimum_applied else raw_base_charge_minor

    fuel_surcharge_minor = _ceil_div(base_charge_minor * tariff.fuel_surcharge_permille, 1000)

    accessorial_charges = tuple(
        (accessorial_id, tariff.accessorials[accessorial_id].charge_minor(base_charge_minor))
        for accessorial_id in request.requested_accessorials
    )
    total_minor = base_charge_minor + fuel_surcharge_minor + sum(
        amount for _, amount in accessorial_charges
    )

    return RateBreakdown(
        carrier_id=tariff.carrier_id,
        service_id=tariff.service_id,
        tariff_version=tariff.version,
        zone=request.zone,
        actual_weight_g=request.actual_weight_g,
        dimensional_weight_g=dimensional_weight_g,
        billed_weight_g=billed_weight_g,
        base_charge_minor=base_charge_minor,
        minimum_charge_applied=minimum_applied,
        fuel_surcharge_minor=fuel_surcharge_minor,
        accessorial_charges_minor=accessorial_charges,
        total_minor=total_minor,
    )


# ------------------------------------------------------------------------------ registry

def _resolve_effective(versions: list[Tariff], *, as_of: int) -> Optional[Tariff]:
    """Same effective-dating resolution `packvium.commerce.catalog`'s `CatalogRegistry`
    and `packvium.commerce.policy`'s `PolicyRegistry` use: the highest `effective_at` not
    after `as_of`, ties broken by the higher (later-published) version."""
    winner = None
    for candidate in versions:
        if candidate.effective_at <= as_of and (winner is None or
                (candidate.effective_at, candidate.version) > (winner.effective_at, winner.version)):
            winner = candidate
    return winner


class CarrierRegistry:
    """Per-`(carrier_id, service_id)` append-only tariff history. See the module
    docstring for what this contract does and does not cover."""

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str], list[Tariff]] = {}

    def publish(
        self, carrier_id: str, service_id: str, *, effective_at: int,
        dimensional_weight_divisor: int, cost_per_dimensional_kg_minor: Mapping[str, int],
        minimum_charge_minor: int = 0, fuel_surcharge_permille: int = 0,
        accessorials: Mapping[str, AccessorialCharge] = (),
    ) -> Tariff:
        """Append a new, numbered tariff version for `(carrier_id, service_id)`."""
        key = (carrier_id, service_id)
        history = self._versions.setdefault(key, [])
        tariff = Tariff(
            carrier_id=carrier_id,
            service_id=service_id,
            version=len(history) + 1,
            effective_at=effective_at,
            dimensional_weight_divisor=dimensional_weight_divisor,
            cost_per_dimensional_kg_minor=dict(cost_per_dimensional_kg_minor),
            minimum_charge_minor=minimum_charge_minor,
            fuel_surcharge_permille=fuel_surcharge_permille,
            accessorials=dict(accessorials),
        )
        history.append(tariff)
        return tariff

    def versions(self, carrier_id: str, service_id: str) -> tuple[Tariff, ...]:
        return tuple(self._history(carrier_id, service_id))

    def _history(self, carrier_id: str, service_id: str) -> list[Tariff]:
        key = (carrier_id, service_id)
        if key not in self._versions:
            raise TariffNotFoundError(f"no tariff registered for {carrier_id}/{service_id}")
        return self._versions[key]

    def tariff(self, carrier_id: str, service_id: str, version: int) -> Tariff:
        """Resolve one immutable tariff version without performing a rating.

        Application adapters use this once and then evaluate every candidate against
        the same pinned object, avoiding an ``O(h)`` history scan per candidate where
        ``h`` is the number of published tariff versions.
        """
        history = self._history(carrier_id, service_id)
        if type(version) is int:
            if 1 <= version <= len(history):
                return history[version - 1]
        else:
            for tariff in history:
                if tariff.version == version:
                    return tariff
        raise TariffNotFoundError(f"{carrier_id}/{service_id} has no version {version}")

    def effective_tariff(self, carrier_id: str, service_id: str, *, as_of: int) -> Tariff:
        """Resolve the tariff version effective at `as_of` without performing a rating.

        The counterpart of `tariff()` for the effective-dated path: an adapter that has
        to report *which* resolution step failed, or that evaluates many requests
        against one instant, resolves once here instead of re-scanning the history.
        """
        tariff = _resolve_effective(self._history(carrier_id, service_id), as_of=as_of)
        if tariff is None:
            raise TariffNotFoundError(
                f"{carrier_id}/{service_id} has no tariff version effective as of {as_of}"
            )
        return tariff

    def rate(self, carrier_id: str, service_id: str, request: RatingRequest, *, as_of: int) -> RateBreakdown:
        """Resolve the tariff version effective at `as_of` for `(carrier_id,
        service_id)` and compute its rate breakdown for `request`."""
        return rate_tariff(self.effective_tariff(carrier_id, service_id, as_of=as_of), request)

    def rate_with_version(
        self, carrier_id: str, service_id: str, version: int, request: RatingRequest,
    ) -> RateBreakdown:
        """Resolve one explicit, pinned tariff version rather than an `as_of` lookup --
        the deterministic-offline-replay path: given the same `(carrier_id, service_id,
        version)` and the same `request`, this always reproduces the identical
        `RateBreakdown`, regardless of what the registry's history has grown to since."""
        return rate_tariff(self.tariff(carrier_id, service_id, version), request)
