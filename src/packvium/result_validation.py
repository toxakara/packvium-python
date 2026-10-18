"""`packvium.validate`: language-neutral validation of a packing result.

Takes a request and a result **as JSON** -- the documented wire contract -- and decides
whether the result is physically and structurally sound. It shares no code with any
implementation: units, geometry, support and load are all recomputed here from first
principles. That is the point. A validator built on the library's own constraint classes
cannot catch a bug inside those classes, and one built on the library's unit parser
cannot catch a bug in unit parsing. So this module imports nothing else from `packvium` but
the Python-version dataclass shim, and `tests/test_result_validation.py` holds it to that.

It is therefore usable against Python, PHP, Rust and the JavaScript fallback alike, and
against any future implementation that speaks the same JSON; the repository's conformance
suite runs every engine's output through this same module.

    from packvium import validate
    issues = validate(request, result)

An empty list means the result is valid. See docs/VALIDATION-CONTRACT.md.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from math import gcd
from typing import Any, Sequence

# The package's dataclass shim is a Python-version adapter, not engine code: the one import
# from `packvium` this module allows itself.
from ._compat import dataclass

# ---------------------------------------------------------------------------- units

#: Length ticks per unit. One tick is 1/16000 mm, so an international inch
#: (25.4 mm) is exactly 406 400 ticks and common fractional inches stay integral.
LENGTH_TICKS = {
    "mm": 16_000, "cm": 160_000, "m": 16_000_000,
    "in": 406_400, "inch": 406_400, "inches": 406_400,
    "ft": 4_876_800, "tick": 1, "ticks": 1,
}

#: Weight ticks per unit. One tick is 1/8 microgram.
WEIGHT_TICKS = {
    "mg": 8_000, "g": 8_000_000, "kg": 8_000_000_000,
    "oz": 226_796_185, "lb": 3_628_738_960, "lbs": 3_628_738_960,
    "tick": 1, "ticks": 1,
}

#: Axis permutation each orientation code applies to (length, width, height).
ROTATIONS = {
    "LWH": (0, 1, 2), "LHW": (0, 2, 1), "WLH": (1, 0, 2),
    "WHL": (1, 2, 0), "HLW": (2, 0, 1), "HWL": (2, 1, 0),
}

#: Orientations that keep the item's own height vertical.
UPRIGHT_ROTATIONS = ("LWH", "WLH")

#: One square metre in length ticks squared -- the reference area a
#: `Container.max_stack_density` limit is expressed per.
SQUARE_METRE_TICKS = (LENGTH_TICKS["mm"] * 1000) ** 2

#: The six axis-aligned exit directions a placement could be slid out through.
_EXIT_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")


class MalformedResult(Exception):
    """The result does not have the shape the contract requires."""


def parse_rational(text: Any) -> Fraction:
    """Parse an integer, decimal, fraction (``3/16``) or mixed fraction (``12 3/8``)."""
    if isinstance(text, int):
        return Fraction(text)
    value = str(text).strip()
    if " " in value:
        whole, fraction = value.split(None, 1)
        sign = -1 if whole.startswith("-") else 1
        return Fraction(whole) + sign * Fraction(fraction)
    return Fraction(value)


def to_ticks(measure: Any, default_unit: str, table: dict[str, int]) -> int:
    """Convert a schema ``measure`` to exact ticks, rounding halves to even."""
    unit = default_unit
    raw = measure
    if isinstance(measure, dict):
        raw = measure["value"]
        unit = measure.get("unit", default_unit)
    elif isinstance(measure, str):
        stripped = measure.strip()
        for suffix in sorted(table, key=len, reverse=True):
            if stripped.lower().endswith(suffix) and stripped[: -len(suffix)].strip():
                raw = stripped[: -len(suffix)].strip()
                unit = suffix
                break
    scaled = parse_rational(raw) * table[unit.lower()]
    floor = scaled.numerator // scaled.denominator
    remainder = scaled - floor
    if remainder > Fraction(1, 2):
        return floor + 1
    if remainder < Fraction(1, 2):
        return floor
    return floor if floor % 2 == 0 else floor + 1


def dimensions_to_ticks(value: dict, unit: str) -> tuple[int, int, int]:
    return (
        to_ticks(value["length"], unit, LENGTH_TICKS),
        to_ticks(value["width"], unit, LENGTH_TICKS),
        to_ticks(value["height"], unit, LENGTH_TICKS),
    )


# ------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in ticks."""

    x: int
    y: int
    z: int
    length: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.length

    @property
    def y2(self) -> int:
        return self.y + self.width

    @property
    def z2(self) -> int:
        return self.z + self.height

    @property
    def base_area(self) -> int:
        return self.length * self.width

    def intersects(self, other: "Box") -> bool:
        return (
            self.x < other.x2 and other.x < self.x2
            and self.y < other.y2 and other.y < self.y2
            and self.z < other.z2 and other.z < self.z2
        )

    def contains(self, other: "Box") -> bool:
        return (
            self.x <= other.x and self.y <= other.y and self.z <= other.z
            and other.x2 <= self.x2 and other.y2 <= self.y2 and other.z2 <= self.z2
        )

    def overlap_area_xy(self, other: "Box") -> int:
        dx = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        dy = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        return dx * dy


@dataclass(frozen=True)
class Issue:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


# -------------------------------------------------------------------------- request


@dataclass(frozen=True)
class ItemSpec:
    id: str
    dimensions: tuple[int, int, int]
    weight: int
    quantity: int
    allowed_rotations: tuple[str, ...]
    stackable: bool
    must_be_on_floor: bool
    max_top_load: int | None
    minimum_support_ratio: Fraction
    tags: frozenset[str]
    eligible_container_tags: frozenset[str]
    max_stacked_items: int | None
    ground_contact_rule: str | None
    nesting_height: int | None
    incompatible_tags: frozenset[str]
    stop_index: int | None
    value: int | None
    shape_type: str
    hull_vertices: tuple[tuple[int, int, int], ...] | None
    compression_ratio_ppm: int | None
    max_compression_pressure_kpa: int | None
    stacking_group: str | None = None
    accepts_stacking_groups: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MachineSpec:
    """A made-to-measure container's machine, in ticks, minor units and weight ticks."""

    minimum: tuple[int, int, int]
    increment: tuple[int, int, int]
    price_per_square_metre_minor: int
    tare_per_square_metre: int
    allowance: tuple[int, int, int]


@dataclass(frozen=True)
class ContainerSpec:
    id: str
    inner: tuple[int, int, int]
    outer: tuple[int, int, int] | None
    cost_minor: int
    max_payload: int | None
    max_items: int | None
    quantity: int | None
    obstacles: tuple[Box, ...]
    tags: frozenset[str]
    tag_limits: dict[str, int]
    void_fill_reserve_ratio: Fraction
    tare_weight: int
    axles: tuple[tuple[int, int | None], tuple[int, int | None]] | None
    max_stack_density: int | None
    #: 's carrier rate card, kept raw: this validator re-derives landed cost from
    #: the request rather than trusting any engine's arithmetic, which is the whole point
    #: of recomputing an objective independently.
    rate_table: dict | None
    #: The walls this container may be unloaded through. Empty means the
    #: horizontal half of route order is not checked -- not that the container is sealed.
    #: Six walls and none are both nearly-vacuous, but they are *different*
    #: nearly-vacuous, so the absent field keeps the pre-reading.
    access_directions: tuple[str, ...]
    #: A flat bag, admitted only when a + c <= L and b + c <= W over the largest
    #: envelope x2, y2 and z2 of its placements.
    flat_bag: bool = False
    #: Whether a packed instance must report `achieved_dimensions`: a flat bag or a
    #: made-to-measure box. Either way it is the tight extent of the placements.
    reports_achieved_extent: bool = False
    #: The on-demand box machine this container is cut by, or None for stock.
    made_to_measure: "MachineSpec | None" = None


def _parse_items(request: dict, unit: str) -> dict[str, ItemSpec]:
    items = {}
    for raw in request["items"]:
        rotations = raw.get("allowed_rotations")
        if rotations is None:
            rotations = UPRIGHT_ROTATIONS if raw.get("keep_upright") else tuple(ROTATIONS)
        items[raw["id"]] = ItemSpec(
            id=raw["id"],
            dimensions=dimensions_to_ticks(raw["dimensions"], unit),
            weight=to_ticks(raw.get("weight", 0), "g", WEIGHT_TICKS),
            quantity=raw.get("quantity", 1),
            allowed_rotations=tuple(rotations),
            stackable=raw.get("stackable", True),
            must_be_on_floor=raw.get("must_be_on_floor", False),
            max_top_load=(
                None if raw.get("max_top_load") is None
                else to_ticks(raw["max_top_load"], "g", WEIGHT_TICKS)
            ),
            minimum_support_ratio=Fraction(str(raw.get("minimum_support_ratio", 0))),
            tags=frozenset(raw.get("tags", [])),
            eligible_container_tags=frozenset(raw.get("eligible_container_tags", [])),
            max_stacked_items=raw.get("max_stacked_items"),
            ground_contact_rule=raw.get("ground_contact_rule"),
            nesting_height=(
                None if raw.get("nesting_height") is None
                else to_ticks(raw["nesting_height"], unit, LENGTH_TICKS)
            ),
            incompatible_tags=frozenset(raw.get("incompatible_tags", [])),
            stop_index=raw.get("stop_index"),
            value=raw.get("value"),
            shape_type=raw.get("shape_type", "rigid_cuboid"),
            hull_vertices=(
                None if raw.get("hull_vertices") is None
                else tuple(
                    tuple(to_ticks(vertex[axis], unit, LENGTH_TICKS) for axis in "xyz")
                    for vertex in raw["hull_vertices"]
                )
            ),
            compression_ratio_ppm=(
                None if raw.get("compression_ratio") is None
                # The solvers' ratio rule, applied once here so nothing downstream sees a
                # float: floor(ratio * 1000000 + 0.5) on the double.
                else _ratio_ppm(Fraction(str(raw["compression_ratio"])))
            ),
            max_compression_pressure_kpa=raw.get("max_compression_pressure_kpa"),
            stacking_group=raw.get("stacking_group"),
            accepts_stacking_groups=frozenset(raw.get("accepts_stacking_groups") or ()),
        )
    return items


def _parse_containers(request: dict, unit: str) -> dict[str, ContainerSpec]:
    containers = {}
    for raw in request["containers"]:
        obstacles = []
        for obstacle in raw.get("obstacles", []):
            for component in (obstacle, *obstacle.get("additional_boxes", [])):
                origin = component.get("origin", {})
                dims = dimensions_to_ticks(component["dimensions"], unit)
                obstacles.append(Box(
                    to_ticks(origin.get("x", 0), unit, LENGTH_TICKS),
                    to_ticks(origin.get("y", 0), unit, LENGTH_TICKS),
                    to_ticks(origin.get("z", 0), unit, LENGTH_TICKS),
                    *dims,
                ))
        raw_axles = raw.get("axles")
        axles = None if raw_axles is None else tuple(
            (
                to_ticks(axle["position"], unit, LENGTH_TICKS),
                None if axle.get("max_load") is None
                else to_ticks(axle["max_load"], "g", WEIGHT_TICKS),
            )
            for axle in raw_axles
        )
        containers[raw["id"]] = ContainerSpec(
            id=raw["id"],
            inner=dimensions_to_ticks(raw["inner_dimensions"], unit),
            access_directions=tuple(
                direction for direction in _EXIT_DIRECTIONS
                if direction in set(raw.get("access_directions", ()))
            ),
            outer=(
                None if raw.get("outer_dimensions") is None
                else dimensions_to_ticks(raw["outer_dimensions"], unit)
            ),
            cost_minor=raw.get("cost_minor", 0),
            max_payload=(
                None if raw.get("max_payload") is None
                else to_ticks(raw["max_payload"], "g", WEIGHT_TICKS)
            ),
            max_items=raw.get("max_items"),
            quantity=raw.get("quantity"),
            obstacles=tuple(obstacles),
            tags=frozenset(raw.get("tags", [])),
            tag_limits={str(tag): int(limit) for tag, limit in raw.get("tag_limits", {}).items()},
            void_fill_reserve_ratio=Fraction(str(raw.get("void_fill_reserve_ratio", 0))),
            tare_weight=to_ticks(raw.get("tare_weight", 0), "g", WEIGHT_TICKS),
            axles=axles,
            max_stack_density=(
                None if raw.get("max_stack_density") is None
                else to_ticks(raw["max_stack_density"], "g", WEIGHT_TICKS)
            ),
            rate_table=raw.get("rate_table"),
            flat_bag=raw.get("deformable_boundary") is not None,
            reports_achieved_extent=(raw.get("deformable_boundary") is not None
                                     or raw.get("made_to_measure") is not None),
            made_to_measure=_parse_machine(raw.get("made_to_measure"), unit),
        )
    return containers


def _parse_machine(raw: dict | None, unit: str) -> MachineSpec | None:
    if raw is None:
        return None

    def sides(name: str) -> tuple[int, int, int]:
        value = raw.get(name)
        return (0, 0, 0) if value is None else dimensions_to_ticks(value, unit)

    tare = raw.get("tare_per_square_metre")
    return MachineSpec(
        minimum=sides("minimum"),
        increment=sides("increment"),
        price_per_square_metre_minor=raw.get("cost_per_square_metre_minor") or 0,
        tare_per_square_metre=0 if tare is None else to_ticks(tare, "g", WEIGHT_TICKS),
        allowance=sides("outer_allowance"),
    )


# ------------------------------------------------------------------------ validation


def _envelope(placement: dict, clearance: int) -> Box:
    """Reconstruct the envelope box: the physical box grown by the clearance margin."""
    position = placement["position"]
    dimensions = placement["dimensions"]
    try:
        return Box(
            position["x"]["ticks"] - clearance,
            position["y"]["ticks"] - clearance,
            position["z"]["ticks"] - clearance,
            dimensions["length"]["ticks"] + 2 * clearance,
            dimensions["width"]["ticks"] + 2 * clearance,
            dimensions["height"]["ticks"] + 2 * clearance,
        )
    except (KeyError, TypeError) as error:
        raise MalformedResult(f"placement is missing tick coordinates: {error}") from error


def _materialize_lattice_summaries(
    request: dict,
    result: dict,
    items: dict[str, ItemSpec],
    containers: dict[str, ContainerSpec],
    clearance: int,
    issues: list[Issue],
) -> dict:
    """Independently expand compact lattice results before ordinary validation.

    The wire summary is treated as an untrusted, reconstructible representation. Its
    lattice parameters must agree with the request before they are used to synthesize
    placements; every synthesized placement then passes through the same geometry,
    accounting, load and objective checks as an explicitly materialized result.
    """
    if not any("lattice_summary" in packed for packed in result["containers"]):
        return result

    expanded = deepcopy(result)
    coordinates_optional = (
        request.get("configuration", {}).get("require_placement_coordinates") is False
    )
    for packed in expanded["containers"]:
        summary = packed.get("lattice_summary")
        if summary is None:
            continue

        container_id = packed.get("id", "?")
        if not coordinates_optional:
            issues.append(Issue(
                "unexpected_lattice_summary",
                f"{container_id}: request requires explicit placement coordinates",
            ))
        if packed.get("placements"):
            issues.append(Issue(
                "lattice_placements_conflict",
                f"{container_id}: compact summary and explicit placements are mutually exclusive",
            ))

        item_type = summary.get("item_type")
        item = items.get(item_type)
        container = containers.get(packed.get("container_type"))
        if item is None or container is None:
            # The ordinary accounting/container checks report the unknown reference.
            continue

        try:
            orientation = summary["orientation"]
            physical = tuple(
                summary["physical_dimensions"][axis]["ticks"]
                for axis in ("length", "width", "height")
            )
            envelope = tuple(
                summary["envelope_dimensions"][axis]["ticks"]
                for axis in ("length", "width", "height")
            )
            nx = summary["nx"]
            ny = summary["ny"]
            layers_used = summary["layers_used"]
            layer_step = summary["layer_step"]["ticks"]
            count = summary["count"]
        except (KeyError, TypeError) as error:
            raise MalformedResult(
                f"{container_id}: lattice_summary is missing a canonical field: {error}"
            ) from error

        problems: list[str] = []
        if orientation not in ROTATIONS:
            problems.append(f"unknown orientation {orientation!r}")
        else:
            expected_physical = tuple(item.dimensions[axis] for axis in ROTATIONS[orientation])
            if physical != expected_physical:
                problems.append(
                    f"physical dimensions {physical} != rotated item {expected_physical}"
                )
            if orientation not in item.allowed_rotations:
                problems.append(f"orientation {orientation} is forbidden for {item_type}")

        expected_envelope = tuple(dimension + 2 * clearance for dimension in physical)
        if envelope != expected_envelope:
            problems.append(
                f"envelope dimensions {envelope} != physical+clearance {expected_envelope}"
            )
        if item.nesting_height is not None:
            problems.append("compact lattice form is not valid for nesting items")
        if layer_step != envelope[2]:
            problems.append(f"layer_step {layer_step} != envelope height {envelope[2]}")

        positive_integers = {
            "nx": nx,
            "ny": ny,
            "layers_used": layers_used,
            "layer_step": layer_step,
            "count": count,
        }
        for field, value in positive_integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                problems.append(f"{field} must be a positive integer, got {value!r}")

        if not problems:
            expected_nx = container.inner[0] // envelope[0]
            expected_ny = container.inner[1] // envelope[1]
            expected_layers = (count + nx * ny - 1) // (nx * ny)
            if nx != expected_nx:
                problems.append(f"nx {nx} != independently computed {expected_nx}")
            if ny != expected_ny:
                problems.append(f"ny {ny} != independently computed {expected_ny}")
            if layers_used != expected_layers:
                problems.append(
                    f"layers_used {layers_used} != independently computed {expected_layers}"
                )
            if count > item.quantity:
                problems.append(f"count {count} exceeds requested quantity {item.quantity}")
            vertical_capacity = container.inner[2] // layer_step
            if layers_used > vertical_capacity:
                problems.append(
                    f"layers_used {layers_used} exceeds vertical capacity {vertical_capacity}"
                )

        if problems:
            issues.extend(
                Issue("lattice_summary_mismatch", f"{container_id}: {problem}")
                for problem in problems
            )
            packed["placements"] = []
            continue

        placements = []
        per_layer = nx * ny
        for index in range(count):
            x_index = index % nx
            y_index = (index // nx) % ny
            z_index = index // per_layer
            placements.append({
                "item_id": f"{item_type}#lattice:{container_id}:{index + 1}",
                "item_type": item_type,
                "orientation": orientation,
                "position": {
                    "x": {"ticks": x_index * envelope[0] + clearance},
                    "y": {"ticks": y_index * envelope[1] + clearance},
                    "z": {"ticks": z_index * layer_step + clearance},
                },
                "dimensions": deepcopy(summary["physical_dimensions"]),
            })
        packed["placements"] = placements

    return expanded


def _check_structure(result: dict, issues: list[Issue]) -> None:
    summary = result["summary"]
    packed = sum(len(c["placements"]) for c in result["containers"])
    unpacked = len(result["unpacked_items"])
    if summary["container_count"] != len(result["containers"]):
        issues.append(Issue("summary_mismatch",
                            f"container_count {summary['container_count']} != {len(result['containers'])}"))
    if summary["packed_item_count"] != packed:
        issues.append(Issue("summary_mismatch",
                            f"packed_item_count {summary['packed_item_count']} != {packed}"))
    if summary["unpacked_item_count"] != unpacked:
        issues.append(Issue("summary_mismatch",
                            f"unpacked_item_count {summary['unpacked_item_count']} != {unpacked}"))
    if result["complete"] != (unpacked == 0):
        issues.append(Issue("complete_mismatch",
                            f"complete={result['complete']} with {unpacked} unpacked"))


def _resolve_container_tag_rules(request: dict) -> tuple[dict, ...]:
    """Participating `require_container_tag` rules, resolved from the request alone.

    Deliberately a second implementation of the resolution in docs/POLICY-RULES.md
    rather than an import from any engine: a validator that resolved the rule set with
    the engine's own code would agree with it by construction, including when both are
    wrong -- which is the case this exists to catch.

    Only this rule form is resolved, because only it can support a *proven* unpacked
    reason. Segregation and per-container caps depend on what else was packed, so an
    item they leave behind was left behind by the search.
    """
    policy = request.get("policy")
    if not isinstance(policy, dict):
        return ()
    declared = policy.get("rules") or []
    if not isinstance(declared, list) or not declared:
        return ()
    as_of = policy.get("as_of")
    if not isinstance(as_of, int):
        return ()
    shipment = policy.get("shipment") or {}
    latest: dict[str, dict] = {}
    for rule in declared:
        if not isinstance(rule, dict) or "require_container_tag" not in rule:
            continue
        applies_to = rule.get("applies_to") or {}
        if any(shipment.get(fact) != value for fact, value in applies_to.items()):
            continue
        if rule.get("effective_at", 0) > as_of:
            continue
        current = latest.get(rule.get("id"))
        if current is None or (
            (rule.get("effective_at", 0), rule.get("version", 0))
            > (current.get("effective_at", 0), current.get("version", 0))
        ):
            latest[rule.get("id")] = rule
    return tuple(latest.values())


def _check_accounting(
    result: dict,
    items: dict[str, ItemSpec],
    containers: dict[str, ContainerSpec],
    clearance: int,
    issues: list[Issue],
    container_tag_rules: tuple[dict, ...] = (),
) -> None:
    seen: set[str] = set()
    per_type: dict[str, int] = {}
    for container in result["containers"]:
        for placement in container["placements"]:
            item_id = placement["item_id"]
            if item_id in seen:
                issues.append(Issue("duplicate_item", item_id))
            seen.add(item_id)
            per_type[placement["item_type"]] = per_type.get(placement["item_type"], 0) + 1
    for unpacked in result["unpacked_items"]:
        item_id = unpacked["item_id"]
        if item_id in seen:
            issues.append(Issue("duplicate_item", f"{item_id} is both packed and unpacked"))
        seen.add(item_id)
        per_type[unpacked["item_type"]] = per_type.get(unpacked["item_type"], 0) + 1
        if not unpacked.get("reason"):
            issues.append(Issue("missing_reason", item_id))
        _check_reason_proof(
            unpacked,
            items.get(unpacked["item_type"]),
            containers,
            clearance,
            result["algorithm"]["time_limit_reached"],
            result["algorithm"]["effort_limit_reached"],
            issues,
            container_tag_rules,
        )

    for item_type, count in per_type.items():
        if item_type not in items:
            issues.append(Issue("unknown_item", item_type))
        elif count != items[item_type].quantity:
            issues.append(Issue("quantity_mismatch",
                                f"{item_type}: {count} accounted for, {items[item_type].quantity} requested"))
    for item_type, spec in items.items():
        if item_type not in per_type:
            issues.append(Issue("quantity_mismatch",
                                f"{item_type}: 0 accounted for, {spec.quantity} requested"))


def _fits_empty(container: ContainerSpec, size: tuple[int, int, int]) -> bool:
    """Whether one envelope of this size fits the empty container: its frame and, for a flat
    bag, both girth sums with the envelope alone at the origin."""
    if any(size[index] > container.inner[index] for index in range(3)):
        return False
    return not container.flat_bag or (size[0] + size[2] <= container.inner[0]
                                      and size[1] + size[2] <= container.inner[1])


def _achieved_extent(packed: dict, clearance: int) -> tuple[int, int, int]:
    """The largest envelope x2, y2 and z2 over a packed container's placements."""
    envelopes = [_envelope(placement, clearance) for placement in packed["placements"]]
    return (max((box.x2 for box in envelopes), default=0),
            max((box.y2 for box in envelopes), default=0),
            max((box.z2 for box in envelopes), default=0))


def _check_achieved_extent(packed: dict, spec: ContainerSpec, clearance: int,
                           issues: list[Issue]) -> None:
    """`achieved_dimensions` is present exactly when the container declares a soft size, and
    equals the tight extent; a flat bag's extent must pass the girth rule."""
    reported = packed.get("achieved_dimensions")
    if not spec.reports_achieved_extent:
        if reported is not None:
            issues.append(Issue("achieved_dimensions_mismatch",
                                f"{packed['id']}: reported for a container of fixed size"))
        return
    a, b, c = _achieved_extent(packed, clearance)
    stated = None if not isinstance(reported, dict) else tuple(
        reported.get(side, {}).get("ticks") for side in ("length", "width", "height"))
    if stated != (a, b, c):
        issues.append(Issue("achieved_dimensions_mismatch",
                            f"{packed['id']}: reported {stated}, computed {(a, b, c)}"))
    if spec.flat_bag and (a + c > spec.inner[0] or b + c > spec.inner[1]):
        issues.append(Issue("bag_girth_exceeded",
                            f"{packed['id']}: a + c = {a + c}, b + c = {b + c}, "
                            f"flat size {spec.inner[0]} x {spec.inner[1]}"))


def _fabricated_side(achieved: int, minimum: int, increment: int) -> int:
    """The smallest size on the machine grid `minimum + k * increment` that is at least
    `achieved` (docs/MADE-TO-MEASURE-CONTAINERS.md, `grid_a`)."""
    return minimum + -(-(max(achieved, minimum) - minimum) // increment) * increment


def _per_square_metre_ceiling(rate: int, sides: tuple[int, int, int]) -> int:
    """`ceil(rate * 2(LW + LH + WH) / 1 m^2)`, the material and board rounding."""
    length, width, height = sides
    area = 2 * (length * width + length * height + width * height)
    return -(-rate * area // SQUARE_METRE_TICKS)


def _reported_ticks(value: Any, keys: tuple[str, ...]) -> tuple[int, ...] | None:
    if not isinstance(value, dict):
        return None
    return tuple((value.get(key) or {}).get("ticks") for key in keys)


def _check_made_to_measure(packed: dict, compact: bool, spec: ContainerSpec,
                           items: dict[str, ItemSpec], clearance: int,
                           issues: list[Issue]) -> ContainerSpec:
    """Recompute the box cut for a made-to-measure container from its placements alone, and
    return the spec every later check uses: the recomputed box, never the reported one.

    O(p) for `p` placements; every product is an exact integer (step 6).
    """
    container_id = packed.get("id", "?")
    block = packed.get("made_to_measure")
    machine = spec.made_to_measure
    if machine is None:
        if block is not None:
            issues.append(Issue("made_to_measure_unexpected", container_id))
        return spec
    if not isinstance(block, dict):
        issues.append(Issue("made_to_measure_missing", container_id))
        block = {}
    if compact:
        issues.append(Issue("lattice_summary_on_made_to_measure", container_id))
    sides = ("length", "width", "height")
    achieved = _achieved_extent(packed, clearance)
    fabricated = tuple(
        _fabricated_side(achieved[axis], machine.minimum[axis], machine.increment[axis])
        for axis in range(3)
    )
    if any(fabricated[axis] > spec.inner[axis] for axis in range(3)):
        issues.append(Issue("fabricated_dimensions_exceed_machine",
                            f"{container_id}: {fabricated} beyond {spec.inner}"))
    reported_inner = _reported_ticks(packed.get("inner_dimensions"), sides)
    if reported_inner != fabricated:
        issues.append(Issue("fabricated_dimensions_mismatch",
                            f"{container_id}: reported {reported_inner}, computed {fabricated}"))
    outer = tuple(fabricated[axis] + machine.allowance[axis] for axis in range(3))
    reported_outer = _reported_ticks(packed.get("outer_dimensions"), sides)
    if reported_outer != outer:
        issues.append(Issue("outer_dimensions_mismatch",
                            f"{container_id}: reported {reported_outer}, computed {outer}"))
    cost = spec.cost_minor + _per_square_metre_ceiling(machine.price_per_square_metre_minor, fabricated)
    if block.get("cost_minor") != cost:
        issues.append(Issue("made_to_measure_cost_mismatch",
                            f"{container_id}: reported {block.get('cost_minor')}, computed {cost}"))
    tare = spec.tare_weight + _per_square_metre_ceiling(machine.tare_per_square_metre, fabricated)
    reported_tare = (block.get("tare_weight") or {}).get("ticks")
    if reported_tare != tare:
        issues.append(Issue("made_to_measure_tare_mismatch",
                            f"{container_id}: reported {reported_tare}, computed {tare}"))
    payload = sum(items[p["item_type"]].weight for p in packed["placements"] if p["item_type"] in items)
    reported_gross = (packed.get("gross_weight") or {}).get("ticks")
    if reported_gross != tare + payload:
        issues.append(Issue("gross_weight_mismatch",
                            f"{container_id}: reported {reported_gross}, computed {tare + payload}"))
    return replace(spec, inner=fabricated, outer=outer, cost_minor=cost, tare_weight=tare)


def _check_reason_proof(
    unpacked: dict,
    spec: ItemSpec | None,
    containers: dict[str, ContainerSpec],
    clearance: int,
    timed_out: bool,
    effort_limited: bool,
    issues: list[Issue],
    container_tag_rules: tuple[dict, ...] = (),
) -> None:
    item_id = unpacked.get("item_id", "?")
    reason = unpacked.get("reason", "")
    proof = unpacked.get("proof")
    if not isinstance(proof, dict):
        issues.append(Issue("missing_reason_proof", item_id))
        return
    level = proof.get("level")
    expected = (
        "proven" if reason in {
            "no_compatible_container_dimensions", "payload_exceeded", "no_eligible_container",
            "rotation_restricted", "policy_rule",
        }
        else "unknown_due_to_limit" if reason in {"time_limit", "effort_limit"}
        else "observed" if reason in {
            "no_feasible_placement", "search_exhausted", "exact_search_incomplete",
            "insufficient_support",
        }
        else "inferred"
    )
    if level != expected:
        issues.append(Issue(
            "reason_proof_mismatch",
            f"{item_id}: {reason!r} has proof {level!r}, expected {expected!r}",
        ))
    if reason == "time_limit" and not timed_out:
        issues.append(Issue(
            "reason_limit_mismatch",
            f"{item_id}: time_limit reason without a reached time limit",
        ))
    if reason == "effort_limit" and not effort_limited:
        issues.append(Issue(
            "reason_limit_mismatch",
            f"{item_id}: effort_limit reason without a reached effort limit",
        ))
    if spec is not None and reason == "no_compatible_container_dimensions":
        fits = any(
            _fits_empty(container, tuple(spec.dimensions[axis] + 2 * clearance
                                         for axis in ROTATIONS[rotation]))
            for container in containers.values()
            for rotation in spec.allowed_rotations
        )
        if fits:
            issues.append(Issue(
                "invalid_proven_reason",
                f"{item_id}: at least one allowed orientation fits a container",
            ))
        # A genuine "no rotation at all works" claim must fail even with every
        # physical orientation tried, not only the item's own restricted set --
        # otherwise this reason and `rotation_restricted` below would overlap.
        fits_any_rotation = any(
            _fits_empty(container, tuple(spec.dimensions[axis] + 2 * clearance
                                         for axis in ROTATIONS[rotation]))
            for container in containers.values()
            for rotation in ROTATIONS
        )
        if fits_any_rotation:
            issues.append(Issue(
                "invalid_proven_reason",
                f"{item_id}: fits at least one container once every rotation is tried, "
                f"not only its own allowed set -- should be rotation_restricted",
            ))
    if spec is not None and reason == "rotation_restricted":
        fits_own_rotations = any(
            _fits_empty(container, tuple(spec.dimensions[axis] + 2 * clearance
                                         for axis in ROTATIONS[rotation]))
            for container in containers.values()
            for rotation in spec.allowed_rotations
        )
        if fits_own_rotations:
            issues.append(Issue(
                "invalid_proven_reason",
                f"{item_id}: already fits a container within its own allowed rotations",
            ))
        fits_any_rotation = any(
            _fits_empty(container, tuple(spec.dimensions[axis] + 2 * clearance
                                         for axis in ROTATIONS[rotation]))
            for container in containers.values()
            for rotation in ROTATIONS
        )
        if not fits_any_rotation:
            issues.append(Issue(
                "invalid_proven_reason",
                f"{item_id}: does not fit any container even with every rotation tried "
                f"-- should be no_compatible_container_dimensions",
            ))
    if spec is not None and reason == "payload_exceeded":
        payload_fits = any(
            container.max_payload is None or spec.weight <= container.max_payload
            for container in containers.values()
        )
        if payload_fits:
            issues.append(Issue(
                "invalid_proven_reason",
                f"{item_id}: its weight is within at least one container payload limit",
            ))
    if spec is not None and reason == "no_eligible_container":
        eligible = any(
            not spec.eligible_container_tags
            or spec.eligible_container_tags.intersection(container.tags)
            for container in containers.values()
        )
        if eligible:
            issues.append(Issue(
                "invalid_proven_reason",
                f"{item_id}: at least one container satisfies its eligible tags",
            ))
    if spec is not None and reason == "policy_rule":
        # Two separate claims, and both are checked. That some rule rules this item out
        # of every container, and that the rule the result *names* is that rule: a
        # citation naming an innocent rule sends a caller to change the wrong policy,
        # which is worse than no citation at all.
        blocking = [
            rule for rule in container_tag_rules
            if rule["require_container_tag"].get("item_tag") in spec.tags
            and not any(
                rule["require_container_tag"].get("container_tag") in container.tags
                for container in containers.values()
            )
        ]
        if not blocking:
            issues.append(Issue(
                "invalid_proven_reason",
                f"{item_id}: no participating policy rule rules it out of every container",
            ))
        else:
            cited = " ".join(unpacked.get("details") or ())
            if not any(f"{rule['id']}@{rule['version']}" in cited for rule in blocking):
                issues.append(Issue(
                    "invalid_proven_reason",
                    f"{item_id}: cites {cited!r}, which names no rule that actually blocks it",
                ))
    observations = proof.get("observations")
    if not isinstance(observations, list) or not observations:
        issues.append(Issue("missing_reason_observations", item_id))
        return
    observed_reason = False
    for observation in observations:
        if not isinstance(observation, dict):
            issues.append(Issue("malformed_reason_observation", item_id))
            continue
        code = observation.get("code")
        count = observation.get("count")
        details = observation.get("details")
        if not isinstance(code, str) or not code or not isinstance(count, int) or count < 1 or not isinstance(details, list):
            issues.append(Issue("malformed_reason_observation", item_id))
            continue
        if code == reason:
            observed_reason = True
    if not observed_reason:
        issues.append(Issue("reason_observation_mismatch", f"{item_id}: {reason}"))


# ---------------------------------------------------------------- irregular geometry


def _rotate_hull(
    vertices: tuple[tuple[int, int, int], ...], orientation: str
) -> tuple[tuple[int, int, int], ...]:
    """Reorient a hull the way `ROTATIONS` reorients its box, without ever mirroring it.

    Three of the six orientations are odd permutations of the coordinate axes. A bare
    permutation would hand back the item's mirror image -- invisible on a cuboid, wrong on a
    hull -- so the sign of one axis flips when the permutation is odd and every orientation
    stays a proper rotation. The result is translated so the rotated hull's bounding box
    starts at the origin, which is the frame a placement position is expressed in.
    """
    axes = ROTATIONS[orientation]
    inversions = sum(
        1 for first in range(3) for second in range(first + 1, 3) if axes[first] > axes[second]
    )
    sign = -1 if inversions % 2 else 1
    turned = [
        (sign * vertex[axes[0]], vertex[axes[1]], vertex[axes[2]]) for vertex in vertices
    ]
    low = tuple(min(vertex[axis] for vertex in turned) for axis in range(3))
    return tuple(
        (x - low[0], y - low[1], z - low[2]) for x, y, z in turned
    )


def _box_vertices(box: Box) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (box.x + dx * box.length, box.y + dy * box.width, box.z + dz * box.height)
        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
    )


def _placed_hull(
    envelope: Box, placement: dict, spec: "ItemSpec | None"
) -> tuple[tuple[int, int, int], ...]:
    """The solid a placement actually occupies, as vertices in container coordinates.

    A `rigid_cuboid`, a `compressible` item and any hull the engine was not allowed to treat
    exactly all come back as their eight envelope corners, so one collision routine answers
    every pair. The three exact-hull conditions are re-derived here from the request and the
    reported placement rather than trusted from the engine.
    """
    if (spec is None or spec.shape_type != "convex_hull" or spec.hull_vertices is None
            or spec.stop_index is not None
            or (envelope.length, envelope.width, envelope.height)
            != tuple(spec.dimensions[axis] for axis in ROTATIONS[placement["orientation"]])):
        return _box_vertices(envelope)
    local = _rotate_hull(spec.hull_vertices, placement["orientation"])
    return tuple((envelope.x + x, envelope.y + y, envelope.z + z) for x, y, z in local)


def _hull_axes(
    left: tuple[tuple[int, int, int], ...], right: tuple[tuple[int, int, int], ...]
) -> set[tuple[int, int, int]]:
    """The complete separating-axis superset of docs/IRREGULAR-ITEMS.md.

    Deliberately the naive set -- every vertex triple of each hull, plus every pair of pair
    directions across the two -- rather than the engine's supporting-plane filter. This is a
    validator: it is allowed to be slow and has to be obviously right, and taking the same
    shortcut as the thing it checks would make the check circular.
    """
    axes: set[tuple[int, int, int]] = set()
    for hull in (left, right):
        for first, second, third in combinations(hull, 3):
            axes.add(_primitive_axis(_cross_product(
                _difference(second, first), _difference(third, first))))
    left_directions = [_difference(b, a) for a, b in combinations(left, 2)]
    right_directions = [_difference(b, a) for a, b in combinations(right, 2)]
    for one in left_directions:
        for other in right_directions:
            axes.add(_primitive_axis(_cross_product(one, other)))
    axes.discard((0, 0, 0))
    return axes


def _difference(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross_product(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return (left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0])


def _primitive_axis(axis: tuple[int, int, int]) -> tuple[int, int, int]:
    divisor = gcd(gcd(abs(axis[0]), abs(axis[1])), abs(axis[2]))
    if divisor == 0:
        return (0, 0, 0)
    reduced = (axis[0] // divisor, axis[1] // divisor, axis[2] // divisor)
    for value in reduced:
        if value:
            return reduced if value > 0 else (-reduced[0], -reduced[1], -reduced[2])
    return (0, 0, 0)


def _solids_overlap(
    left: tuple[tuple[int, int, int], ...], right: tuple[tuple[int, int, int], ...]
) -> bool:
    """Whether two convex solids share interior volume. Touching is contact, not collision."""
    for axis in _hull_axes(left, right):
        left_values = [_dot_product(vertex, axis) for vertex in left]
        right_values = [_dot_product(vertex, axis) for vertex in right]
        if max(left_values) <= min(right_values) or max(right_values) <= min(left_values):
            return False
    return True


def _occupied_volume(
    envelope: Box, placement: dict, spec: "ItemSpec | None", load: int
) -> int:
    """Space a placement actually takes: its hull, its compressed box, or its box.

    Recomputed from the request rather than read from the result, which is the whole point of
    this file. Both non-default shapes get utilisation wrong in the flattering direction if
    the box is used instead -- a hull item reports two interlocking wedges as 200% of a crate,
    and a compressible one never shows the height it gave up under load.
    """
    if spec is not None and spec.max_compression_pressure_kpa is not None:
        return envelope.base_area * _compressed_height(envelope, spec, load)
    if spec is not None and spec.shape_type == "convex_hull" and spec.hull_vertices is not None:
        # Collision intentionally uses the box for routes and clearance. Volume is a property
        # of the physical hull and must not inherit that conservative collision fallback.
        return _hull_volume(_rotate_hull(spec.hull_vertices, placement["orientation"]))
    solid = _placed_hull(envelope, placement, spec)
    if len(solid) == 8 and solid == _box_vertices(envelope):
        return (placement["dimensions"]["length"]["ticks"]
                * placement["dimensions"]["width"]["ticks"]
                * placement["dimensions"]["height"]["ticks"])
    return _hull_volume(solid)


def _compressed_height(envelope: Box, spec: ItemSpec, load: int) -> int:
    """Occupied height under load, rounded up, never below one tick.

    The formula of docs/IRREGULAR-ITEMS.md, in exact rationals. A crushed item keeps its
    uncompressed height here: the arrangement is already invalid and `_check_crush` reports
    it, so this stays a reported issue rather than a second, quieter one.
    """
    assert spec.compression_ratio_ppm is not None
    assert spec.max_compression_pressure_kpa is not None
    pressure = Fraction(
        load * 980_665 * (LENGTH_TICKS["mm"] * 1000) ** 2,
        WEIGHT_TICKS["kg"] * 100_000 * 1_000 * envelope.base_area,
    )
    if pressure > spec.max_compression_pressure_kpa or spec.max_compression_pressure_kpa == 0:
        return envelope.height
    divisor = spec.max_compression_pressure_kpa * 1_000_000 * pressure.denominator
    retained = divisor - spec.compression_ratio_ppm * pressure.numerator
    return max(1, -(-envelope.height * retained // divisor))


def _hull_volume(vertices: tuple[tuple[int, int, int], ...]) -> int:
    """Exact volume of a convex hull, `6V = sum of a . (b x c)` over its outward faces.

    Written the slow, obvious way -- every vertex triple is tested for being a supporting
    plane, and every face is wound with integer cross products -- because a validator that
    took the engine's shortcuts would stop being a second opinion.
    """
    normals = set()
    for first, second, third in combinations(vertices, 3):
        axis = _primitive_axis(_cross_product(
            _difference(second, first), _difference(third, first)))
        if axis != (0, 0, 0):
            normals.add(axis)
    six_volumes = 0
    for axis in sorted(normals):
        for outward in (axis, (-axis[0], -axis[1], -axis[2])):
            extreme = max(_dot_product(vertex, outward) for vertex in vertices)
            face = [v for v in vertices if _dot_product(v, outward) == extreme]
            if len(face) < 3:
                continue
            ordered = _wind_face(face, outward)
            apex = ordered[0]
            for second, third in zip(ordered[1:], ordered[2:]):
                six_volumes += _dot_product(apex, _cross_product(second, third))
    return abs(six_volumes) // 6


def _wind_face(
    face: list[tuple[int, int, int]], outward: tuple[int, int, int]
) -> tuple[tuple[int, int, int], ...]:
    """Corners of one planar convex face, in cyclic order seen from outside.

    The vertices sharing a supporting plane are not all corners of the polygon they lie on:
    one can sit inside the face or part-way along an edge, and fanning over the raw set
    triangulates the wrong region. The surface then fails to close and the volume is wrong --
    which is how the defect this replaces was found. Gift-wrapping keeps only the corners,
    starting from the lexicographically smallest vertex, which is extreme in any linear order
    and therefore a corner. Collinear candidates resolve to the farthest, so a vertex lying on
    an edge is walked past instead of doubled back through.
    """
    start = min(face)
    ordered = [start]
    current = start
    for _ in range(len(face)):
        following = None
        for candidate in face:
            if candidate == current:
                continue
            if following is None:
                following = candidate
                continue
            turn = _dot_product(
                _cross_product(_difference(following, current), _difference(candidate, current)),
                outward,
            )
            reach = _dot_product(_difference(candidate, current), _difference(candidate, current))
            held = _dot_product(_difference(following, current), _difference(following, current))
            if turn < 0 or (turn == 0 and reach > held):
                following = candidate
        if following is None or following == start:
            break
        ordered.append(following)
        current = following
    return tuple(ordered)


def _dot_product(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _check_rotation(placement: dict, spec: ItemSpec, issues: list[Issue]) -> None:
    orientation = placement["orientation"]
    if orientation not in ROTATIONS:
        issues.append(Issue("unknown_orientation", f"{placement['item_id']}: {orientation}"))
        return
    if orientation not in spec.allowed_rotations:
        issues.append(Issue("forbidden_rotation", f"{placement['item_id']}: {orientation}"))
    expected = tuple(spec.dimensions[axis] for axis in ROTATIONS[orientation])
    actual = (
        placement["dimensions"]["length"]["ticks"],
        placement["dimensions"]["width"]["ticks"],
        placement["dimensions"]["height"]["ticks"],
    )
    if actual != expected:
        issues.append(Issue("dimension_mismatch",
                            f"{placement['item_id']}: {orientation} of {spec.dimensions} is {expected}, got {actual}"))


def _check_container(
    packed: dict,
    spec: ContainerSpec,
    items: dict[str, ItemSpec],
    clearance: int,
    global_support: Fraction,
    issues: list[Issue],
) -> None:
    boundary = Box(0, 0, 0, *spec.inner)
    envelopes = {}
    payload = 0

    for placement in packed["placements"]:
        item_id = placement["item_id"]
        item_spec = items.get(placement["item_type"])
        envelope = _envelope(placement, clearance)
        envelopes[item_id] = (envelope, placement, item_spec)

        if not boundary.contains(envelope):
            issues.append(Issue("outside_container", f"{item_id} in {packed['id']}"))
        for obstacle in spec.obstacles:
            if envelope.intersects(obstacle) and _solids_overlap(
                _placed_hull(envelope, placement, item_spec), _box_vertices(obstacle)
            ):
                issues.append(Issue("obstacle_collision", f"{item_id} in {packed['id']}"))
        if item_spec is None:
            continue

        payload += item_spec.weight
        _check_rotation(placement, item_spec, issues)
        if (item_spec.eligible_container_tags
                and not item_spec.eligible_container_tags.intersection(spec.tags)):
            issues.append(Issue(
                "container_ineligible",
                f"{item_id}: {packed['container_type']} has none of "
                f"{sorted(item_spec.eligible_container_tags)}",
            ))
        if item_spec.must_be_on_floor and envelope.z != 0:
            issues.append(Issue("floor_violation", f"{item_id} at z={envelope.z}"))

    for (left_id, left_value), (right_id, right_value) in combinations(envelopes.items(), 2):
        left, left_placement, left_spec = left_value
        right, right_placement, right_spec = right_value
        if (left.intersects(right)
                and _solids_overlap(_placed_hull(left, left_placement, left_spec),
                                    _placed_hull(right, right_placement, right_spec))
                and not _valid_nesting(
                    left, left_placement, left_spec, right, right_placement, right_spec)):
            issues.append(Issue("collision", f"{left_id} with {right_id} in {packed['id']}"))
        if (
            left_spec is not None and right_spec is not None
            and (left_spec.incompatible_tags & right_spec.tags
                 or right_spec.incompatible_tags & left_spec.tags)
        ):
            issues.append(Issue(
                "incompatible_items", f"{left_id} is incompatible with {right_id} in {packed['id']}"
            ))

    supporters, direct_above = _support_graph(envelopes)
    _check_support_and_load(
        packed, envelopes, supporters, direct_above, global_support, issues
    )
    _check_stack_density(packed, envelopes, supporters, spec.max_stack_density, issues)
    _check_crush(packed, envelopes, supporters, issues)
    _check_route_removal_order(packed, envelopes, direct_above, spec.inner,
                               spec.access_directions, issues)

    if spec.max_payload is not None and payload > spec.max_payload:
        issues.append(Issue("payload_exceeded",
                            f"{packed['id']}: {payload} > {spec.max_payload} ticks"))
    reported = packed.get("payload_weight", {}).get("ticks")
    if reported is not None and reported != payload:
        issues.append(Issue("payload_weight_mismatch",
                            f"{packed['id']}: reported {reported}, computed {payload}"))
    if spec.max_items is not None and len(packed["placements"]) > spec.max_items:
        issues.append(Issue("max_items_exceeded",
                            f"{packed['id']}: {len(packed['placements'])} > {spec.max_items}"))
    for tag, limit in sorted(spec.tag_limits.items()):
        count = sum(
            1
            for placement in packed["placements"]
            if (items.get(placement["item_type"]) is not None
                and tag in items[placement["item_type"]].tags)
        )
        if count > limit:
            issues.append(Issue(
                "tag_count_exceeded",
                f"{packed['id']}: {tag} count {count} > limit {limit}",
            ))

    loads = _propagate_load(envelopes, supporters)
    used = sum(
        _occupied_volume(envelopes[p["item_id"]][0], p, items.get(p["item_type"]),
                         loads[p["item_id"]])
        for p in packed["placements"]
    )
    for (_, left_value), (_, right_value) in combinations(envelopes.items(), 2):
        left, left_placement, left_spec = left_value
        right, right_placement, right_spec = right_value
        if _valid_nesting(left, left_placement, left_spec, right, right_placement, right_spec):
            assert left_spec is not None and left_spec.nesting_height is not None
            used -= left_spec.nesting_height * left.length * left.width
    reported_volume = packed.get("used_volume_ticks3")
    if reported_volume is not None and int(reported_volume) != used:
        issues.append(Issue("used_volume_mismatch",
                            f"{packed['id']}: reported {reported_volume}, computed {used}"))
    volume = spec.inner[0] * spec.inner[1] * spec.inner[2]
    scaled_reserve = _ratio_ppm(spec.void_fill_reserve_ratio)
    reserved = (volume // SCORE_SCALE) * scaled_reserve
    reserved += ((volume % SCORE_SCALE) * scaled_reserve) // SCORE_SCALE
    reported_reserve = packed.get("void_fill_reserve_ticks3")
    if reported_reserve is not None and int(reported_reserve) != reserved:
        issues.append(Issue(
            "void_fill_reserve_mismatch",
            f"{packed['id']}: reported {reported_reserve}, computed {reserved}",
        ))
    elif reserved and reported_reserve is None:
        issues.append(Issue(
            "void_fill_reserve_mismatch",
            f"{packed['id']}: non-zero reserve {reserved} is not reported",
        ))
    if used > volume - reserved:
        issues.append(Issue(
            "void_fill_reserve_exceeded",
            f"{packed['id']}: used {used} > usable {volume - reserved}",
        ))

    total_weight = sum(
        items[p["item_type"]].weight
        for p in packed["placements"]
        if p["item_type"] in items
    )
    if total_weight == 0:
        expected_com = 0
    else:
        doubled_x = sum(
            items[p["item_type"]].weight
            * (2 * p["position"]["x"]["ticks"] + p["dimensions"]["length"]["ticks"])
            for p in packed["placements"]
            if p["item_type"] in items
        )
        doubled_y = sum(
            items[p["item_type"]].weight
            * (2 * p["position"]["y"]["ticks"] + p["dimensions"]["width"]["ticks"])
            for p in packed["placements"]
            if p["item_type"] in items
        )
        x_ppm = abs(doubled_x - total_weight * spec.inner[0]) * SCORE_SCALE // (
            total_weight * spec.inner[0]
        )
        y_ppm = abs(doubled_y - total_weight * spec.inner[1]) * SCORE_SCALE // (
            total_weight * spec.inner[1]
        )
        expected_com = max(x_ppm, y_ppm)
    if packed.get("centre_of_mass_offset_ppm") != expected_com:
        issues.append(Issue(
            "centre_of_mass_mismatch",
            f"{packed['id']}: reported {packed.get('centre_of_mass_offset_ppm')}, "
            f"computed {expected_com}",
        ))

    if spec.axles is not None:
        (front_x, front_limit), (rear_x, rear_limit) = spec.axles
        denominator = 2 * (rear_x - front_x)
        gross = spec.tare_weight + total_weight
        doubled_weighted_x = spec.tare_weight * spec.inner[0] + sum(
            items[p["item_type"]].weight
            * (2 * p["position"]["x"]["ticks"] + p["dimensions"]["length"]["ticks"])
            for p in packed["placements"]
            if p["item_type"] in items
        )
        front_numerator = 2 * gross * rear_x - doubled_weighted_x
        rear_numerator = doubled_weighted_x - 2 * gross * front_x
        expected_reactions = {
            "basis": "gross",
            "denominator": str(denominator),
            "front_numerator": str(front_numerator),
            "rear_numerator": str(rear_numerator),
        }
        if packed.get("axle_reactions") != expected_reactions:
            issues.append(Issue(
                "axle_reaction_mismatch",
                f"{packed['id']}: reported {packed.get('axle_reactions')}, "
                f"computed {expected_reactions}",
            ))
        if (
            front_limit is not None
            and front_numerator > front_limit * denominator
        ) or (
            rear_limit is not None
            and rear_numerator > rear_limit * denominator
        ):
            issues.append(Issue("axle_overloaded", packed["id"]))
    elif "axle_reactions" in packed:
        issues.append(Issue(
            "axle_reaction_mismatch",
            f"{packed['id']}: reactions reported without configured axles",
        ))


def _valid_nesting(
    left: Box,
    left_placement: dict,
    left_spec: ItemSpec | None,
    right: Box,
    right_placement: dict,
    right_spec: ItemSpec | None,
) -> bool:
    if (
        left_spec is None
        or right_spec is None
        or left_spec.id != right_spec.id
        or left_placement["item_type"] != right_placement["item_type"]
        or left_spec.nesting_height is None
        or right_spec.nesting_height != left_spec.nesting_height
    ):
        return False
    if (left.x, left.y, left.x2, left.y2) != (right.x, right.y, right.x2, right.y2):
        return False
    lower, upper = (left, right) if left.z <= right.z else (right, left)
    return lower.z != upper.z and lower.z2 - upper.z == left_spec.nesting_height


def _support_graph(
    envelopes: dict[str, tuple[Box, dict, ItemSpec | None]],
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, set[str]]]:
    """Build exact direct-support edges in deterministic placement order.

    Ordinary face contacts retain their existing order. Nested instances are grouped
    by item type and exact XY footprint, sorted once by height, and connected only to
    their adjacent predecessor. A predecessor contributes the upper item's full base
    area and replaces shadowed face contacts from that same column. Construction is
    O(n log n + q + e) time and O(n + e) space, where ``q`` is the number of same-plane
    candidates and ``e`` the represented support edges.
    """
    item_ids = list(envelopes)
    ordinal = {item_id: index for index, item_id in enumerate(item_ids)}
    by_top: dict[int, list[str]] = {}
    for item_id, (envelope, _, _) in envelopes.items():
        by_top.setdefault(envelope.z2, []).append(item_id)

    supporters: dict[str, list[tuple[str, int]]] = {item_id: [] for item_id in item_ids}
    for upper_id, (upper, _, _) in envelopes.items():
        for lower_id in by_top.get(upper.z, ()):
            if lower_id == upper_id:
                continue
            area = envelopes[lower_id][0].overlap_area_xy(upper)
            if area > 0:
                supporters[upper_id].append((lower_id, area))

    group_keys: dict[str, tuple[str, int, int, int, int, int]] = {}
    groups: dict[tuple[str, int, int, int, int, int], list[str]] = {}
    for item_id, (envelope, placement, spec) in envelopes.items():
        if spec is None or spec.nesting_height is None:
            continue
        key = (
            placement["item_type"], envelope.x, envelope.y,
            envelope.x2, envelope.y2, spec.nesting_height,
        )
        group_keys[item_id] = key
        groups.setdefault(key, []).append(item_id)

    predecessors: dict[str, str] = {}
    for group in groups.values():
        group.sort(key=lambda item_id: (envelopes[item_id][0].z, ordinal[item_id]))
        for lower_id, upper_id in zip(group, group[1:]):
            lower = envelopes[lower_id][0]
            upper = envelopes[upper_id][0]
            lower_spec = envelopes[lower_id][2]
            assert lower_spec is not None and lower_spec.nesting_height is not None
            nesting = lower_spec.nesting_height
            if lower.z2 - upper.z == nesting:
                predecessors[upper_id] = lower_id

    for upper_id, lower_id in predecessors.items():
        key = group_keys[upper_id]
        edges = [
            edge for edge in supporters[upper_id]
            if group_keys.get(edge[0]) != key
        ]
        edge = (lower_id, envelopes[upper_id][0].base_area)
        position = next(
            (index for index, (candidate_id, _) in enumerate(edges)
             if ordinal[candidate_id] > ordinal[lower_id]),
            len(edges),
        )
        edges.insert(position, edge)
        supporters[upper_id] = edges

    direct_above: dict[str, set[str]] = {item_id: set() for item_id in item_ids}
    for upper_id, edges in supporters.items():
        for lower_id, _ in edges:
            direct_above[lower_id].add(upper_id)
    return supporters, direct_above


def _check_support_and_load(
    packed: dict,
    envelopes: dict[str, tuple[Box, dict, ItemSpec | None]],
    supporters: dict[str, list[tuple[str, int]]],
    direct_above: dict[str, set[str]],
    global_support: Fraction,
    issues: list[Issue],
) -> None:
    for item_id, (envelope, placement, spec) in envelopes.items():
        if spec is None or envelope.z == 0:
            continue

        supporting = [
            (other_id, envelopes[other_id][0], envelopes[other_id][2], area)
            for other_id, area in supporters[item_id]
        ]

        required = max(global_support, spec.minimum_support_ratio)
        rule = spec.ground_contact_rule
        if rule not in (None, "free"):
            if rule == "covered":
                corners = (
                    (envelope.x, envelope.y),
                    (envelope.x2, envelope.y),
                    (envelope.x, envelope.y2),
                    (envelope.x2, envelope.y2),
                )
                covered = all(
                    any(
                        surface.x <= x <= surface.x2 and surface.y <= y <= surface.y2
                        for _, surface, _, _ in supporting
                    )
                    for x, y in corners
                )
                if not covered:
                    issues.append(Issue(
                        "ground_contact_violation",
                        f"{item_id}: covered requires all four base corners",
                    ))
            elif rule == "single" and len(supporting) != 1:
                issues.append(Issue(
                    "ground_contact_violation",
                    f"{item_id}: single has {len(supporting)} supporters",
                ))
            elif rule == "multiple" and len(supporting) < 2:
                issues.append(Issue(
                    "ground_contact_violation",
                    f"{item_id}: multiple has {len(supporting)} supporters",
                ))
        if required > 0:
            area = sum(area for _, _, _, area in supporting)
            if area < _required_support_area(envelope.base_area, _ratio_ppm(required)):
                ratio = Fraction(area, envelope.base_area)
                issues.append(Issue("insufficient_support",
                                    f"{item_id}: {float(ratio):.6f} < {float(required):.6f}"))

        for lower_id, _, lower_spec, _ in supporting:
            if lower_spec is None:
                continue
            if not lower_spec.stackable:
                issues.append(Issue("non_stackable", f"{item_id} rests on {lower_id}"))
            elif (lower_spec.accepts_stacking_groups
                  and spec.stacking_group not in lower_spec.accepts_stacking_groups):
                issues.append(Issue("stacking_group_incompatible", f"{item_id} rests on {lower_id}"))
            if lower_spec.max_top_load is not None and spec.weight > lower_spec.max_top_load:
                issues.append(Issue("top_load_exceeded",
                                    f"{item_id} ({spec.weight} ticks) on {lower_id} "
                                    f"(limit {lower_spec.max_top_load})"))

    for item_id, (_, _, spec) in envelopes.items():
        if spec is None or spec.max_stacked_items is None:
            continue
        descendants: set[str] = set()
        pending = list(direct_above[item_id])
        while pending:
            above_id = pending.pop()
            if above_id in descendants:
                continue
            descendants.add(above_id)
            pending.extend(direct_above[above_id])
        if len(descendants) > spec.max_stacked_items:
            issues.append(Issue(
                "max_stacked_items_exceeded",
                f"{item_id}: {len(descendants)} above > limit {spec.max_stacked_items}",
            ))


def _propagate_load(
    envelopes: dict[str, tuple[Box, dict, ItemSpec | None]],
    supporters: dict[str, list[tuple[str, int]]],
) -> dict[str, int]:
    """Cumulative weight each item bears, from everything resting on it however
    indirectly, split across its supporters by contact area and conserved exactly
    (the integer remainder goes to the last supporter). Processes top-down, by
    descending envelope ceiling, so an item's own inherited load is finalized
    before it is pushed further down onto whatever it itself rests on.
    """
    loads: dict[str, int] = {item_id: 0 for item_id in envelopes}
    order = sorted(
        envelopes,
        key=lambda item_id: (-envelopes[item_id][0].z2, -envelopes[item_id][0].z, item_id),
    )
    for upper_id in order:
        upper, _, upper_spec = envelopes[upper_id]
        if upper_spec is None:
            continue
        supports = supporters[upper_id]
        total_area = sum(area for _, area in supports)
        if total_area == 0:
            continue
        downward = upper_spec.weight + loads[upper_id]
        assigned = 0
        for index, (lower_id, area) in enumerate(supports):
            share = downward - assigned if index == len(supports) - 1 else downward * area // total_area
            assigned += share
            loads[lower_id] += share
    return loads


def _check_crush(
    packed: dict,
    envelopes: dict[str, tuple[Box, dict, "ItemSpec | None"]],
    supporters: dict[str, list[tuple[str, int]]],
    issues: list[Issue],
) -> None:
    """No `compressible` item may carry more pressure than it declared.

    Recomputed from the request and the reported placements, never from anything the engine
    said about compression. The cumulative mass above each item comes from `_propagate_load`,
    which this validator already derives for `max_stack_density` -- the same number the
    pressure model needs as its numerator, so the crush verdict rides one traversal rather
    than inventing a second notion of "resting on".

    Exact rational throughout: the boundary is inclusive, and a floating-point comparison
    would put loads a millionth over the limit on the wrong side of it.
    """
    if all(spec is None or spec.max_compression_pressure_kpa is None
           for _, _, spec in envelopes.values()):
        return
    loads = _propagate_load(envelopes, supporters)
    for item_id, (envelope, _, spec) in envelopes.items():
        if spec is None or spec.max_compression_pressure_kpa is None:
            continue
        # P(kPa) = load_ticks * g * (ticks per metre)^2 / (ticks per kg * g_denominator
        #          * pascals per kPa * footprint_ticks^2)
        pressure = Fraction(
            loads[item_id] * 980_665 * (LENGTH_TICKS["mm"] * 1000) ** 2,
            WEIGHT_TICKS["kg"] * 100_000 * 1_000 * envelope.base_area,
        )
        if pressure > spec.max_compression_pressure_kpa:
            issues.append(Issue(
                "crush_violation",
                f"{item_id} in {packed['id']}: {float(pressure):.6f} kPa exceeds "
                f"{spec.max_compression_pressure_kpa} kPa",
            ))


def _check_stack_density(
    packed: dict,
    envelopes: dict[str, tuple[Box, dict, ItemSpec | None]],
    supporters: dict[str, list[tuple[str, int]]],
    max_density: int | None,
    issues: list[Issue],
) -> None:
    """`Container.max_stack_density` limits weight per square metre of an
    item's own base area, at every level of the stack, not only the floor -- a wide
    item narrowing to a small waist part-way up is exactly the case a flat
    `max_top_load` cannot express.
    """
    if max_density is None:
        return
    loads = _propagate_load(envelopes, supporters)
    for item_id, (envelope, _, spec) in envelopes.items():
        if spec is None:
            continue
        total = spec.weight + loads[item_id]
        if total * SQUARE_METRE_TICKS > max_density * envelope.base_area:
            issues.append(Issue(
                "stack_density_exceeded",
                f"{item_id} in {packed['id']}: {total} ticks over {envelope.base_area} "
                f"ticks^2 exceeds {max_density} ticks/m^2",
            ))


def _swept_region(envelope: Box, container: tuple[int, int, int], direction: str) -> Box:
    """The region between `envelope`'s own face and the matching container wall
    along `direction` -- the space a placement leaving that way would sweep."""
    x1, y1, z1, x2, y2, z2 = envelope.x, envelope.y, envelope.z, envelope.x2, envelope.y2, envelope.z2
    if direction == "+x":
        x1, x2 = x2, container[0]
    elif direction == "-x":
        x1, x2 = 0, x1
    elif direction == "+y":
        y1, y2 = y2, container[1]
    elif direction == "-y":
        y1, y2 = 0, y1
    elif direction == "+z":
        z1, z2 = z2, container[2]
    else:
        z1, z2 = 0, z1
    return Box(x1, y1, z1, x2 - x1, y2 - y1, z2 - z1)


def _has_clear_exit(
    item_id: str,
    envelopes: dict[str, tuple[Box, dict, ItemSpec | None]],
    present: set[str],
    container: tuple[int, int, int],
    directions: Sequence[str] = _EXIT_DIRECTIONS,
) -> bool:
    """Can `item_id` be slid straight out through some container wall without
    passing through any other currently-present placement?"""
    envelope = envelopes[item_id][0]
    for direction in directions:
        region = _swept_region(envelope, container, direction)
        if not any(
            other_id != item_id and region.intersects(envelopes[other_id][0])
            for other_id in present
        ):
            return True
    return False



def corridor_replay_succeeds(
    boxes: dict[str, Box],
    stops: dict[str, float],
    container: tuple[int, int, int],
    directions: Sequence[str],
) -> bool:
    """The whole-scene stop-accessibility replay, recomputed from the rule alone.

    Deliberately a second implementation rather than a call into the engine: a check that
    shares code with the thing it checks can only find typos. It answers the *sufficient*
    question of docs/STOP-ACCESSIBILITY.md -- can every item due at stop `t` actually be
    taken out at stop `t`, in some order -- against an explicit door set, which the request
    schema cannot yet express.

    Vertical blocking is not consulted here. That half is `_check_route_removal_order`'s,
    and keeping them apart is what lets a caller test either one without the other
    masking it.
    """
    present = set(boxes)
    for stop in sorted({value for value in stops.values() if value != float("inf")}):
        due = {item_id for item_id, value in stops.items() if value == stop}
        while due:
            removable = [
                item_id for item_id in sorted(due)
                if any(
                    not any(other != item_id
                            and _swept_region(boxes[item_id], container, direction)
                            .intersects(boxes[other])
                            for other in present)
                    for direction in directions
                )
            ]
            if not removable:
                return False
            chosen = removable[0]
            present.discard(chosen)
            due.discard(chosen)
    return True


def _check_route_removal_order(
    packed: dict,
    envelopes: dict[str, tuple[Box, dict, ItemSpec | None]],
    direct_above: dict[str, set[str]],
    container: tuple[int, int, int],
    doors: tuple[str, ...],
    issues: list[Issue],
) -> None:
    """Items due at an earlier route stop must all be removable --
    unblocked by whatever still rests on them, and slidable straight out some wall --
    before the route moves on to the next stop. A placement with no `stop_index` is
    never scheduled for removal and stays present for every stop, exactly like a
    fixture that rides the whole route, but still counts as a potential blocker.

    gave the request a way to say *which* wall (`container.access_directions`),
    and this is where an engine that ignores it is caught. Without doors the check runs
    against all six, which is what it always did and is what a container that states none
    still gets. With doors it is the whole-scene replay `docs/STOP-ACCESSIBILITY.md` calls
    the definition of correctness -- and because it is computed here from the request
    rather than read off the result, an engine cannot satisfy it by agreeing with itself.
    """
    stops = {
        item_id: spec.stop_index
        for item_id, (_, _, spec) in envelopes.items()
        if spec is not None and spec.stop_index is not None
    }
    if not stops:
        return
    present = set(envelopes)
    for stop in sorted(set(stops.values())):
        due = {item_id for item_id, value in stops.items() if value == stop and item_id in present}
        while due:
            removable = [
                item_id for item_id in due
                if not (direct_above[item_id] & present)
                and _has_clear_exit(item_id, envelopes, present, container,
                                    doors or _EXIT_DIRECTIONS)
            ]
            if not removable:
                issues.append(Issue(
                    "unloading_order_violation",
                    f"{packed['id']}: stop {stop} cannot fully unload {sorted(due)} "
                    "before the route moves on",
                ))
                return
            chosen = min(removable)
            present.discard(chosen)
            due.discard(chosen)


def validate(request: dict, result: dict) -> list[Issue]:
    """Return every way ``result`` fails to be a sound answer to ``request``.

    An empty list means valid. Raises :class:`MalformedResult` when the result does not
    have the documented shape at all — that is a contract break, not a packing defect.
    """
    for key in ("status", "feasibility", "termination", "optimality", "complete",
                "algorithm", "summary", "containers", "unpacked_items", "score"):
        if key not in result:
            raise MalformedResult(f"result is missing {key!r}")

    unit = request.get("units", {}).get("length", "mm")
    configuration = request.get("configuration", {})
    clearance = to_ticks(configuration.get("clearance", 0), unit, LENGTH_TICKS)
    global_support = Fraction(str(configuration.get("minimum_support_ratio", 0)))

    items = _parse_items(request, unit)
    containers = _parse_containers(request, unit)

    issues: list[Issue] = []
    validated_result = _materialize_lattice_summaries(
        request, result, items, containers, clearance, issues
    )
    _check_structure(validated_result, issues)
    _check_algorithm_metrics(validated_result, issues)
    _check_effort_budget(request, validated_result, issues)
    _check_result_facts(validated_result, issues)
    _check_accounting(
        validated_result, items, containers, clearance, issues,
        _resolve_container_tag_rules(request),
    )

    if validated_result["status"] == "infeasible" and any(
        unpacked.get("proof", {}).get("level") != "proven"
        for unpacked in validated_result["unpacked_items"]
    ):
        issues.append(Issue(
            "unproven_infeasible",
            "infeasible status requires a proven rejection for every unpacked item",
        ))

    inventory: dict[str, int] = {}
    fabricated: dict[str, ContainerSpec] = {}
    for packed, reported in zip(validated_result["containers"], result["containers"]):
        container_type = packed["container_type"]
        spec = containers.get(container_type)
        if spec is None:
            issues.append(Issue("unknown_container_type", container_type))
            continue
        inventory[container_type] = inventory.get(container_type, 0) + 1
        spec = _check_made_to_measure(packed, "lattice_summary" in reported, spec, items,
                                      clearance, issues)
        if spec.made_to_measure is not None:
            fabricated[packed["id"]] = spec
        _check_container(packed, spec, items, clearance, global_support, issues)
        _check_achieved_extent(packed, spec, clearance, issues)

    for container_type, count in inventory.items():
        quantity = containers[container_type].quantity
        if quantity is not None and count > quantity:
            issues.append(Issue("container_inventory_exceeded",
                                f"{container_type}: used {count}, available {quantity}"))

    max_containers = configuration.get("max_containers")
    opened = len(validated_result["containers"])
    if max_containers is not None and opened > max_containers:
        issues.append(Issue("max_containers_exceeded",
                            f"{opened} containers opened, limit {max_containers}"))

    _check_fixed_placements(request, validated_result, unit, issues)
    _check_groups(request, validated_result, issues)
    _check_solver_selection(request, validated_result, issues)
    _check_objective(request, validated_result, containers, items, clearance, issues, fabricated)
    return issues


FixedKey = tuple[str, str, int, int, int, str]


def _check_fixed_placements(request: dict, result: dict, unit: str, issues: list[Issue]) -> None:
    """Every fixed placement is reported where the request put it, and nothing else claims to be.

    Physics needs no rule here: a fixed item is an ordinary placement, so every check above
    already applied to it. What only this rule can see is whether it moved
    (docs/PLAN-REVISIONS.md).
    """
    requested: set[FixedKey] = set()
    for entry in request.get("fixed_placements", ()):
        container_id = f"{entry['container_type']}#{entry.get('container_instance', 1)}"
        position = entry.get("position", {})
        requested.add((
            container_id,
            entry["item_type"],
            *(to_ticks(position.get(axis, 0), unit, LENGTH_TICKS) for axis in "xyz"),
            entry["orientation"],
        ))
    reported: set[FixedKey] = set()
    container_types: dict[str, str] = {}
    for container in result["containers"]:
        container_types[container.get("id", "")] = container["container_type"]
        for placement in container["placements"]:
            if placement.get("fixed") is not True:
                continue
            position = placement["position"]
            reported.add((
                container.get("id", ""),
                placement["item_type"],
                *(position[axis]["ticks"] for axis in "xyz"),
                placement["orientation"],
            ))
    for key in sorted(requested):
        container_id = key[0]
        if container_types.get(container_id) != container_id.rsplit("#", 1)[0]:
            issues.append(Issue("fixed_container_missing", container_id))
        elif key not in reported:
            issues.append(Issue("fixed_placement_moved", f"{key[1]} fixed in {container_id} at {key[2:5]} {key[5]}"))
    for key in sorted(reported - requested):
        issues.append(Issue("unexpected_fixed_placement", f"{key[1]} in {key[0]} at {key[2:5]} {key[5]}"))


def _check_algorithm_metrics(result: dict, issues: list[Issue]) -> None:
    algorithm = result["algorithm"]
    if not isinstance(algorithm, dict):
        raise MalformedResult("algorithm must be an object")
    metrics = algorithm.get("metrics")
    fields = (
        "candidate_points_considered",
        "orientations_considered",
        "feasible_candidates",
        "collision_checks",
        "support_checks",
        "space_partitions",
        "search_nodes_expanded",
    )
    if not isinstance(metrics, dict) or any(
        not isinstance(metrics.get(field), int)
        or isinstance(metrics.get(field), bool)
        or metrics[field] < 0
        for field in fields
    ):
        issues.append(Issue(
            "solver_metrics_mismatch",
            "algorithm.metrics must contain every non-negative integer counter",
        ))
        return
    legacy_candidates = algorithm.get("candidates_evaluated")
    legacy_attempts = algorithm.get("placements_attempted")
    if (
        not isinstance(legacy_candidates, int)
        or isinstance(legacy_candidates, bool)
        or not isinstance(legacy_attempts, int)
        or isinstance(legacy_attempts, bool)
        or legacy_candidates != metrics["feasible_candidates"]
        or legacy_attempts != metrics["orientations_considered"]
    ):
        issues.append(Issue(
            "solver_metrics_mismatch",
            "legacy candidate counters must mirror algorithm.metrics",
        ))


def _check_effort_budget(request: dict, result: dict, issues: list[Issue]) -> None:
    """Verify that counted-work limits are respected and the stop flag is evidenced."""
    algorithm = result["algorithm"]
    effort_limited = algorithm.get("effort_limit_reached")
    if not isinstance(effort_limited, bool):
        raise MalformedResult("algorithm.effort_limit_reached must be a boolean")

    budget = request.get("configuration", {}).get("effort_budget")
    if budget is None:
        if effort_limited:
            issues.append(Issue(
                "effort_budget_mismatch",
                "effort_limit_reached is true but the request has no effort_budget",
            ))
        return
    if not isinstance(budget, dict):
        raise MalformedResult("config.effort_budget must be an object")

    metrics = algorithm.get("metrics")
    if not isinstance(metrics, dict):
        return
    metric_limits = (
        ("max_candidates_evaluated", "feasible_candidates"),
        ("max_placement_attempts", "orientations_considered"),
        ("max_search_nodes", "search_nodes_expanded"),
    )
    reached = False
    for budget_name, metric_name in metric_limits:
        limit = budget.get(budget_name)
        value = metrics.get(metric_name)
        if limit is None or not isinstance(value, int) or isinstance(value, bool):
            continue
        if value > limit:
            issues.append(Issue(
                "effort_budget_exceeded",
                f"{metric_name} is {value}, above {budget_name}={limit}",
            ))
        reached = reached or value >= limit

    restart_limit = budget.get("max_restarts")
    starts = result.get("termination", {}).get("starts", [])
    if isinstance(restart_limit, int) and isinstance(starts, list):
        started = sum(
            1 for start in starts
            if isinstance(start, dict) and start.get("started") is True
        )
        if started > restart_limit:
            issues.append(Issue(
                "effort_budget_exceeded",
                f"started {started} solver runs, above max_restarts={restart_limit}",
            ))
        reached = reached or started >= restart_limit

    if effort_limited and not reached:
        issues.append(Issue(
            "effort_budget_mismatch",
            "effort_limit_reached is true but no configured counted-work boundary was reached",
        ))


def _check_result_facts(result: dict, issues: list[Issue]) -> None:
    def code(name: str) -> str:
        fact = result[name]
        if not isinstance(fact, dict):
            raise MalformedResult(f"{name} must be an object")
        value = fact.get("code")
        if not isinstance(value, str) or not value:
            raise MalformedResult(f"{name}.code must be a non-empty string")
        return value

    algorithm = result["algorithm"]
    if not isinstance(algorithm, dict):
        raise MalformedResult("algorithm must be an object")
    timed_out = algorithm.get("time_limit_reached")
    if not isinstance(timed_out, bool):
        raise MalformedResult("algorithm.time_limit_reached must be a boolean")
    effort_limited = algorithm.get("effort_limit_reached")
    if not isinstance(effort_limited, bool):
        raise MalformedResult("algorithm.effort_limit_reached must be a boolean")
    if timed_out and effort_limited:
        issues.append(Issue(
            "result_fact_mismatch",
            "time_limit_reached and effort_limit_reached cannot both be true",
        ))

    termination = result["termination"]
    if not isinstance(termination, dict):
        raise MalformedResult("termination must be an object")
    starts = termination.get("starts")
    aggregate = None
    if not isinstance(starts, list) or not starts:
        issues.append(Issue(
            "termination_aggregate_mismatch",
            "termination.starts must contain at least one start",
        ))
    else:
        malformed = False
        for start in starts:
            if not isinstance(start, dict) or not isinstance(start.get("id"), str):
                malformed = True
                continue
            flags = (
                start.get("started"),
                start.get("completed"),
                start.get("truncated"),
                start.get("selected"),
                start.get("global_deadline_reached"),
            )
            if not all(isinstance(flag, bool) for flag in flags):
                malformed = True
                continue
            started, completed, truncated, _, _ = flags
            if (completed and (not started or truncated)) or (
                truncated and (not started or completed)
            ) or (not started and (completed or truncated)):
                issues.append(Issue(
                    "start_record_mismatch",
                    f"{start['id']}: inconsistent lifecycle flags",
                ))
        if malformed:
            issues.append(Issue(
                "start_record_mismatch",
                "a start record is missing its id or boolean lifecycle fields",
            ))
        selected = [
            start for start in starts
            if isinstance(start, dict) and start.get("selected") is True
        ]
        if len(selected) != 1:
            issues.append(Issue(
                "termination_aggregate_mismatch",
                f"expected one selected start, found {len(selected)}",
            ))
        elif all(isinstance(start, dict) for start in starts):
            aggregate = {
                "any_start_truncated": any(
                    start.get("truncated") is True for start in starts
                ),
                "all_required_starts_completed": all(
                    start.get("completed") is True for start in starts
                ),
                "winning_start_truncated": selected[0].get("truncated") is True,
                "global_deadline_reached": any(
                    start.get("global_deadline_reached") is True for start in starts
                ),
            }
            for name, expected in aggregate.items():
                if termination.get(name) is not expected:
                    issues.append(Issue(
                        "termination_aggregate_mismatch",
                        f"{name} is {termination.get(name)!r}, expected {expected!r}",
                    ))
            solver = algorithm.get("solver")
            selected_id = selected[0].get("id", "")
            if isinstance(solver, str) and not (
                selected_id == solver or selected_id.startswith(f"{solver}#")
            ):
                issues.append(Issue(
                    "winning_start_mismatch",
                    f"selected start {selected_id!r} does not identify solver {solver!r}",
                ))
            aggregate_limited = (
                aggregate["global_deadline_reached"]
                or aggregate["winning_start_truncated"]
            )
            if aggregate["global_deadline_reached"] and not timed_out:
                issues.append(Issue(
                    "legacy_time_limit_mismatch",
                    "global_deadline_reached requires time_limit_reached",
                ))
            if aggregate_limited != (timed_out or effort_limited):
                issues.append(Issue(
                    "termination_limit_mismatch",
                    "selected-start truncation must identify exactly one reached limit",
                ))

    status = result["status"]
    complete = result["complete"]
    expected_feasibility = (
        "infeasible" if status == "infeasible"
        else "feasible" if complete and status != "invalid_result"
        else "unknown"
    )
    expected_termination = (
        "error" if status == "invalid_result"
        else "time_limit" if timed_out
        else "effort_limit" if effort_limited
        else "complete"
    )
    expected_optimality = (
        "proven_optimal" if status == "optimal"
        else "proven_infeasible" if status == "infeasible"
        else "best_found" if not complete and status != "invalid_result"
        else "not_proven"
    )
    for name, expected in (
        ("feasibility", expected_feasibility),
        ("termination", expected_termination),
        ("optimality", expected_optimality),
    ):
        actual = code(name)
        if actual != expected:
            issues.append(Issue(
                "result_fact_mismatch",
                f"{name}.code is {actual!r}, expected {expected!r}",
            ))


#: Ratios in the objective vector are carried as parts per million — see docs/OBJECTIVE.md.
SCORE_SCALE = 1_000_000


def _ratio_ppm(ratio: Fraction) -> int:
    """A public ratio in parts per million exactly as every solver converts it.

    The solvers read a support, void-fill reserve or compression ratio as an IEEE double and take
    ``floor(ratio * 10^6 + 0.5)`` in double arithmetic (docs/UNITS-AND-NUMERICS.md). That
    differs from rounding the exact decimal on about one in a hundred seven-decimal halves
    (``0.0001245`` is 124 ppm, not 125), so judging by the exact rational would refuse
    placements, and reported reserves, every solver produces. Each step is a
    correctly rounded IEEE operation, so the value is the same in every language.
    """
    return int(float(ratio) * SCORE_SCALE + 0.5)


def _required_support_area(base_area: int, ppm: int) -> int:
    """Exact ``floor(base_area * ppm / 10^6)``, the least supported area the ratio allows."""
    return base_area * ppm // SCORE_SCALE


def _dimensional_weight_ticks(inner: tuple[int, int, int], divisor: int, length_unit: str, weight_unit: str) -> int:
    """The independent counterpart of `packvium.geometry.dimensional_weight` /
    `Dimensions::dimensionalWeight`: same exact-integer formula, floor
    division, computed here from the container's ticks alone so a mismatched divisor
    or unit in any engine's own computation cannot hide behind agreement with itself."""
    length_ticks_per_unit = LENGTH_TICKS[length_unit]
    weight_ticks_per_unit = WEIGHT_TICKS[weight_unit]
    numerator = (inner[0] * inner[1] * inner[2]) * weight_ticks_per_unit
    denominator = (length_ticks_per_unit ** 3) * divisor
    return numerator // denominator


def objective(
    result: dict,
    containers: dict[str, ContainerSpec],
    items: dict[str, ItemSpec],
    clearance: int,
    name: str = "default",
    dimensional_weight_divisor: int | None = None,
    dimensional_weight_length_unit: str = "in",
    dimensional_weight_weight_unit: str = "lb",
    fabricated: dict[str, ContainerSpec] | None = None,
) -> list[int]:
    """Recompute the objective vector from the result alone.

    Derived here from first principles so the *formula* can be checked against every
    implementation, independently of which solution that implementation happened to
    choose. Two implementations may legitimately return different packings; neither may
    misreport the score of the packing it did return.
    """
    cost = unused = height = billable = achieved_height = landed = 0
    for packed in result["containers"]:
        # A made-to-measure container is scored as the box recomputed for it.
        spec = (fabricated or {}).get(packed.get("id")) or containers.get(packed["container_type"])
        if spec is None:
            continue
        cost += spec.cost_minor
        volume = spec.inner[0] * spec.inner[1] * spec.inner[2]
        if not volume:
            continue
        # Independently checked by `_check_container`, including the exact overlap
        # deduction for nesting. Re-summing physical item volumes here would
        # deliberately disagree with that contract for every valid nested column.
        used = int(packed["used_volume_ticks3"])
        unused += (volume - used) * SCORE_SCALE // volume
        # The height key is measured on envelope tops, so the clearance margin counts.
        top = max((p["position"]["z"]["ticks"] + p["dimensions"]["height"]["ticks"] + clearance
                   for p in packed["placements"]), default=0)
        height += top * SCORE_SCALE // spec.inner[2]
        # The raw achieved height in ticks, not a ratio against the
        # container's own (possibly loose, "open dimension") declared height.
        achieved_height += top
        if name in ("shipping_cost", "lowest_landed_cost") and dimensional_weight_divisor is not None:
            gross = int(packed.get("gross_weight", {}).get("ticks", 0))
            # A flat bag is billed on the package it conformed to.
            billed_size = (_achieved_extent(packed, clearance) if spec.flat_bag
                           else spec.outer if spec.outer is not None else spec.inner)
            dimensional = _dimensional_weight_ticks(
                billed_size, dimensional_weight_divisor,
                dimensional_weight_length_unit, dimensional_weight_weight_unit,
            )
            billed = max(gross, dimensional)
            billable += billed
            if name == "lowest_landed_cost":
                landed += _landed_charge_minor(spec, billed)
    common = [len(result["unpacked_items"]), len(result["containers"]), cost, unused, height]
    if name == "default":
        return common
    if name == "lowest_cost":
        return [common[0], common[2], common[1], common[3], common[4]]
    if name == "shipping_cost":
        if dimensional_weight_divisor is None:
            raise MalformedResult("shipping_cost objective requires configuration.dimensional_weight_divisor")
        return [common[0], billable, common[1], common[3], common[4]]
    if name == "lowest_landed_cost":
        if dimensional_weight_divisor is None:
            raise MalformedResult(
                "lowest_landed_cost objective requires configuration.dimensional_weight_divisor"
            )
        return [common[0], landed, common[1], common[3], common[4]]
    if name == "open_dimension_height":
        return [common[0], achieved_height, common[1], common[2], common[3]]
    if name == "maximum_value":
        # Ranks by the total declared value of unpacked items ahead of
        # container count, cost and unused volume -- unpacked_count still leads,
        # this only tie-breaks among solutions leaving the same number of items
        # unpacked but not the same value forgone.
        value_forgone = sum(
            (items[unpacked["item_type"]].value or 0)
            for unpacked in result["unpacked_items"]
            if unpacked["item_type"] in items
        )
        return [common[0], value_forgone, common[1], common[2], common[3]]
    if name == "maximum_volume":
        # The share of the requested item volume left unpacked, rounded up so an
        # incomplete answer never scores zero, leads; the piece count follows.
        def nominal(spec: "ItemSpec") -> int:
            length, width, height = spec.dimensions
            return length * width * height

        unpacked_volume = sum(
            nominal(items[unpacked["item_type"]])
            for unpacked in result["unpacked_items"]
            if unpacked["item_type"] in items
        )
        requested = sum(nominal(spec) * spec.quantity for spec in items.values())
        unpacked_ppm = -(-unpacked_volume * SCORE_SCALE // requested) if requested else 0
        return [unpacked_ppm, common[0], common[1], common[2], common[3]]
    raise MalformedResult(f"unknown requested objective {name!r}")


def _landed_charge_minor(spec: "ContainerSpec", billed_weight_ticks: int) -> int:
    """Re-derive one container's carrier charge from the request's own rate card.

    Deliberately a second implementation rather than a shared one: this validator exists to
    disagree with the engines when they are wrong, and importing their arithmetic would make
    the check tautological. Rounding matches the published contract -- billed weight up to
    whole grams because a carrier reads a scale that way, and the fuel surcharge up because a
    fractional unit of revenue is not forgiven.
    """
    if spec.rate_table is None:
        raise MalformedResult(
            f"container {spec.id!r} has no rate_table, so lowest_landed_cost cannot price it"
        )
    brackets = [int(bound) for bound in spec.rate_table["weight_brackets_g"]]
    prices = [int(price) for price in spec.rate_table["prices_minor"]]
    grams = -(-billed_weight_ticks // WEIGHT_TICKS["g"])
    for bound, price in zip(brackets, prices):
        if grams <= bound:
            base = max(price, int(spec.rate_table.get("minimum_charge_minor", 0)))
            permille = int(spec.rate_table.get("fuel_surcharge_permille", 0))
            return base + -(-base * permille // 1000)
    raise MalformedResult(
        f"container {spec.id!r} bills at {grams} g, above its rate table's last bracket "
        f"({brackets[-1]} g); the shipment has no published price"
    )


def _check_objective(request: dict, result: dict, containers: dict[str, ContainerSpec],
                     items: dict[str, ItemSpec], clearance: int, issues: list[Issue],
                     fabricated: dict[str, ContainerSpec] | None = None) -> None:
    configuration = request.get("configuration", {})
    selected = configuration.get("objective", "default")
    reported_name = result.get("objective")
    if ("objective" in configuration or reported_name is not None) and reported_name != selected:
        issues.append(Issue(
            "objective_mismatch",
            f"reported {reported_name!r}, requested {selected!r}",
        ))
    reported = [int(component) for component in result["score"]]
    recomputed = objective(
        result, containers, items, clearance, selected,
        dimensional_weight_divisor=configuration.get("dimensional_weight_divisor"),
        dimensional_weight_length_unit=configuration.get("dimensional_weight_length_unit", "in"),
        dimensional_weight_weight_unit=configuration.get("dimensional_weight_weight_unit", "lb"),
        fabricated=fabricated,
    )
    if reported != recomputed:
        issues.append(Issue("score_mismatch", f"reported {reported}, recomputed {recomputed}"))


def _check_solver_selection(request: dict, result: dict, issues: list[Issue]) -> None:
    selected = request.get("configuration", {}).get("solvers", [])
    if not selected:
        return
    reported = result["algorithm"].get("solver")
    base = reported.split(":", 1)[0] if isinstance(reported, str) else reported
    if base not in selected:
        issues.append(Issue(
            "solver_selection_mismatch",
            f"reported {reported!r}, requested one of {selected!r}",
        ))


def _check_groups(request: dict, result: dict, issues: list[Issue]) -> None:
    """A group travels together or not at all.

    Members must share one container, and a group must not be half-packed: leaving one
    member behind splits it just as surely as putting it in a second container.
    """
    groups = {raw["id"]: raw["group"] for raw in request["items"] if raw.get("group") is not None}
    if not groups:
        return
    located: dict[str, set[str]] = {}
    packed: dict[str, int] = {}
    for container in result["containers"]:
        for placement in container["placements"]:
            group = groups.get(placement["item_type"])
            if group is None:
                continue
            located.setdefault(group, set()).add(container["id"])
            packed[group] = packed.get(group, 0) + 1
    expected: dict[str, int] = {}
    for raw in request["items"]:
        group = raw.get("group")
        if group is not None:
            expected[group] = expected.get(group, 0) + raw.get("quantity", 1)
    for group in sorted(located):
        where = located[group]
        if len(where) > 1:
            issues.append(Issue("group_split", f"{group}: {', '.join(sorted(where))}"))
        if packed.get(group, 0) != expected[group]:
            issues.append(Issue("group_partially_packed",
                                f"{group}: {packed.get(group, 0)} of {expected[group]} placed"))
