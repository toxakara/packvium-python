from __future__ import annotations

from dataclasses import replace

from .config import PackingConfig, SolverProfile
from .effort import EffortBudget
from .compression import ratio_to_ppm
from .geometry import AxisAlignedBox, Dimensions, Point, Rotation, ShapeType
from .models import Axle, Container, Item, Obstacle, RateTable
from .extensions import ExtensionRegistry
from .packer import Packer
from .policy import PolicyRuleSet
from .units import Length, Weight


def _axles(raw: list | None, unit: str) -> tuple[Axle, Axle] | None:
    if raw is None: return None
    front, rear = raw
    return (
        Axle(Length.parse(front["position"], unit), None if front.get("max_load") is None else Weight.parse(front["max_load"])),
        Axle(Length.parse(rear["position"], unit), None if rear.get("max_load") is None else Weight.parse(rear["max_load"])),
    )


def _effort_budget(raw: dict | None) -> EffortBudget | None:
    if raw is None: return None
    return EffortBudget(
        max_candidates_evaluated=raw.get("max_candidates_evaluated"),
        max_placement_attempts=raw.get("max_placement_attempts"),
        max_search_nodes=raw.get("max_search_nodes"),
        max_restarts=raw.get("max_restarts"),
    )


def _hull_vertices(raw, unit: str) -> "tuple[tuple[int, int, int], ...] | None":
    """Parse `hull_vertices` into the integer tick frame before any geometry runs.

    Coordinates go through `Length`, which refuses a negative value, so a hull crossing the
    wire is authored as non-negative offsets from the corner of its own bounding box. A
    library caller may still centre a hull wherever it likes -- `hull.rotate` normalises
    either way -- but the wire keeps one convention so four engines cannot disagree about
    where an item's frame starts.
    """
    if raw is None:
        return None
    return tuple(
        (Length.parse(vertex["x"], unit).ticks,
         Length.parse(vertex["y"], unit).ticks,
         Length.parse(vertex["z"], unit).ticks)
        for vertex in raw
    )


def _item(raw: dict, unit: str) -> Item:
    rotations = tuple(Rotation(v) for v in raw.get("allowed_rotations", [r.value for r in Rotation.all()]))
    return Item(
        id=str(raw["id"]), dimensions=Dimensions.from_dict(raw["dimensions"], unit), weight=Weight.parse(raw.get("weight", 0)),
        quantity=int(raw.get("quantity", 1)), allowed_rotations=rotations, keep_upright=bool(raw.get("keep_upright", False)),
        stackable=bool(raw.get("stackable", True)), must_be_on_floor=bool(raw.get("must_be_on_floor", False)),
        max_top_load=None if raw.get("max_top_load") is None else Weight.parse(raw["max_top_load"]),
        max_stacked_items=raw.get("max_stacked_items"),
        minimum_support_ratio=float(raw.get("minimum_support_ratio", 0)), group=raw.get("group"), tags=frozenset(raw.get("tags", [])),
        incompatible_tags=frozenset(raw.get("incompatible_tags", [])), priority=int(raw.get("priority", 0)),
        eligible_container_tags=frozenset(raw.get("eligible_container_tags", [])),
        ground_contact_rule=raw.get("ground_contact_rule"), metadata=raw.get("metadata", {}),
        nesting_height=None if raw.get("nesting_height") is None else Length.parse(raw["nesting_height"], unit),
        stop_index=raw.get("stop_index"),
        value=raw.get("value"),
        shape_type=ShapeType(raw.get("shape_type", ShapeType.RIGID_CUBOID.value)),
        hull_vertices=_hull_vertices(raw.get("hull_vertices"), unit),
        compression_ratio_ppm=(None if raw.get("compression_ratio") is None
                               else ratio_to_ppm(float(raw["compression_ratio"]))),
        max_compression_pressure_kpa=raw.get("max_compression_pressure_kpa"),
    )


def _box(raw: dict, unit: str) -> AxisAlignedBox:
    origin = raw.get("origin", {})
    return AxisAlignedBox(
        Point(Length.parse(origin.get("x", 0), unit).ticks, Length.parse(origin.get("y", 0), unit).ticks, Length.parse(origin.get("z", 0), unit).ticks),
        Dimensions.from_dict(raw["dimensions"], unit),
    )


def _rate_table(raw: dict | None) -> RateTable | None:
    if raw is None:
        return None
    return RateTable(
        weight_brackets_g=tuple(int(bound) for bound in raw["weight_brackets_g"]),
        prices_minor=tuple(int(price) for price in raw["prices_minor"]),
        minimum_charge_minor=int(raw.get("minimum_charge_minor", 0)),
        fuel_surcharge_permille=int(raw.get("fuel_surcharge_permille", 0)),
    )


def _container(raw: dict, unit: str) -> Container:
    obstacles = tuple(
        Obstacle(str(o["id"]), _box(o, unit), additional_boxes=tuple(_box(b, unit) for b in o.get("additional_boxes", [])))
        for o in raw.get("obstacles", [])
    )
    return Container(
        id=str(raw["id"]), inner_dimensions=Dimensions.from_dict(raw["inner_dimensions"], unit),
        outer_dimensions=None if raw.get("outer_dimensions") is None else Dimensions.from_dict(raw["outer_dimensions"], unit),
        tare_weight=Weight.parse(raw.get("tare_weight", 0)), max_payload=None if raw.get("max_payload") is None else Weight.parse(raw["max_payload"]),
        cost_minor=int(raw.get("cost_minor", 0)), rate_table=_rate_table(raw.get("rate_table")),
        quantity=None if raw.get("quantity") is None else int(raw["quantity"]),
        obstacles=obstacles, tags=frozenset(raw.get("tags", [])), max_items=None if raw.get("max_items") is None else int(raw["max_items"]),
        void_fill_reserve_ratio=float(raw.get("void_fill_reserve_ratio", 0)),
        tag_limits={str(k): int(v) for k, v in raw.get("tag_limits", {}).items()}, metadata=raw.get("metadata", {}),
        max_stack_density=None if raw.get("max_stack_density") is None else Weight.parse(raw["max_stack_density"]),
        axles=_axles(raw.get("axles"), unit),
        access_directions=tuple(str(d) for d in raw.get("access_directions", ())),
    )


class UnsupportedFeatureError(ValueError):
    """A public request field this engine has not implemented yet.

    Structured rather than ignored: `pack_from_dict` reads the keys it knows and skips
    the rest, so without this guard an unimplemented field would produce a confident
    answer computed as though the caller had never sent it -- indistinguishable, from
    the outside, from an engine that honoured it. PHP, Rust and the JavaScript fallback
    have carried the same table since the first staged rollout; this is Python's
    counterpart, so a staged
    rollout works the same way in every implementation.
    """


#: Public request fields this engine does not implement yet, by the scope they appear in.
#:
#: A name added here must also be recorded in `conformance/public-field-matrix.json` with
#: a `rejected:unsupported_feature` support level for Python, which is what makes the
#: conformance corpus assert the rejection instead of merely tolerating it.
UNSUPPORTED_FIELDS: dict[str, tuple[str, ...]] = {
    "request": (),
    "configuration": (),
    # `hull_vertices`, `compression_ratio` and `max_compression_pressure_kpa` left this list
    # in, when Python gained both the solver behaviour and the independent validation
    # the staged rollout requires. PHP, Rust and the JavaScript fallback still carry them.
    "item": (),
    # `pallet_overhang_limit` was reserved in the schema by at the 1.1.0 contract
    # freeze and is refused everywhere until an engine implements it from a request: a field
    # a caller can set and the solver ignores is worse than a refusal.
    # `access_directions` left this list in, which wired the reserved field through
    # to `StopAccessibilityConstraint` in all four engines at once.
    "container": ("pallet_overhang_limit",),
}

#: `item.shape_type` values this engine does not implement.
#:
#: Presence is the wrong test for this one field: `rigid_cuboid` is the default and is
#: implemented, so a caller that spells the default out must be served, not refused. What
#: is unimplemented is a *value*, and the refusal has to name it -- an engine that packed a
#: `convex_hull` item as its bounding box would return a plan that looks valid and does not
#: physically fit.
#: Empty since: this engine implements every value the schema defines. The guard
#: stays because the next reserved value will need it, and because `reject_unsupported` takes
#: its lists as parameters precisely so it remains testable when they are empty.
UNSUPPORTED_SHAPE_TYPES: tuple[str, ...] = ()


def reject_unsupported(
    data: dict,
    unsupported: dict[str, tuple[str, ...]] | None = None,
    shape_types: tuple[str, ...] | None = None,
) -> None:
    """Refuse a request that uses a field this engine has not implemented.

    The lists are a parameter rather than read from the module constant directly so the
    guard itself is testable: with every list empty -- the correct state whenever this
    engine is caught up -- a test can only prove that nothing is rejected, which is
    equally true of a guard that does nothing at all.
    """
    unsupported = UNSUPPORTED_FIELDS if unsupported is None else unsupported
    shape_types = UNSUPPORTED_SHAPE_TYPES if shape_types is None else shape_types
    # Keyed by name, not appended per occurrence: fifty containers carrying one
    # unimplemented field are one complaint, not fifty.
    found: set[str] = set()
    # Top-level scope: a block such as `policy` describes the whole request rather than
    # one item or container, so neither loop below would ever see it.
    found.update(key for key in unsupported.get("request", ()) if key in data)
    configuration = data.get("configuration") or {}
    found.update(f"configuration.{key}" for key in unsupported.get("configuration", ()) if key in configuration)
    for scope, collection in (("item", "items"), ("container", "containers")):
        for entry in data.get(collection) or ():
            if not isinstance(entry, dict):
                continue
            found.update(f"{scope}.{key}" for key in unsupported.get(scope, ()) if key in entry)
    for entry in data.get("items") or ():
        if isinstance(entry, dict) and entry.get("shape_type") in shape_types:
            found.add(f"item.shape_type={entry['shape_type']}")
    if not found:
        return
    raise UnsupportedFeatureError(
        "unsupported_feature: the Python engine does not yet implement "
        + ", ".join(sorted(found))
        + "; the request was rejected instead of silently ignoring public fields"
    )


def pack_from_dict(data: dict, *,
                   extensions: ExtensionRegistry | None = None) -> dict:
    reject_unsupported(data)
    unit = data.get("units", {}).get("length", "mm"); cfg = data.get("configuration", {})
    profile = SolverProfile(cfg.get("solver_profile", "balanced"))
    quality = profile is SolverProfile.QUALITY
    config = PackingConfig(
        profile=profile, time_limit_ms=int(cfg.get("time_limit_ms", 1000)),
        top_k=int(cfg.get("alternatives", 3)), seed=int(cfg.get("seed", 42)), max_containers=cfg.get("max_containers"),
        clearance=Length.parse(cfg.get("clearance", 0), unit), minimum_support_ratio=float(cfg.get("minimum_support_ratio", 0)),
        exact_item_limit=int(cfg.get("exact_item_limit", 7)), multi_start_orders=int(cfg.get("multi_start_orders", 8)),
        max_candidates_per_item=int(cfg.get("max_candidates_per_item", 16 if quality else 1)),
        max_candidate_points=int(cfg.get("max_candidate_points", 4096)),
        solvers=tuple(cfg.get("solvers", [])),
        objective=str(cfg.get("objective", "default")),
        effort_budget=_effort_budget(cfg.get("effort_budget")),
        dimensional_weight_divisor=cfg.get("dimensional_weight_divisor"),
        dimensional_weight_length_unit=str(cfg.get("dimensional_weight_length_unit", "in")),
        dimensional_weight_weight_unit=str(cfg.get("dimensional_weight_weight_unit", "lb")),
        require_placement_coordinates=bool(cfg.get("require_placement_coordinates", True)),
        container_plan_beam_width=int(cfg.get("container_plan_beam_width", 16 if quality else 1)),
        container_plan_node_limit=int(cfg.get("container_plan_node_limit", 100_000 if quality else 1)),
    )
    references = _catalog_versions_used(data.get("catalog_versions_used", []))
    # Rules compile into this engine's own constraint pipeline rather than post-filtering
    # a chosen answer: an illegal candidate is rejected during search, so the packing that
    # wins was never allowed to be illegal in the first place.
    # A caller's extensions are *added to* the compiled policy rules, never substituted for
    # them. Replacing would let an operator lock silently drop a policy rule the
    # request asked for, which is the one thing a lock must not be able to do.
    supplied = extensions or ExtensionRegistry()
    extensions = ExtensionRegistry(
        placement_constraints=(
            *PolicyRuleSet.from_dict(data.get("policy")).constraints(),
            *supplied.placement_constraints,
        ),
        item_order_strategies=supplied.item_order_strategies,
        solvers=supplied.solvers,
        container_selector=supplied.container_selector,
    )
    result = Packer(config, extensions).pack([_item(i, unit) for i in data["items"]], [_container(c, unit) for c in data["containers"]])
    result = replace(result, catalog_versions_used=references)
    output = data.get("output", {})
    return result.to_dict(output.get("length_unit", unit), output.get("weight_unit", "g"))


def _catalog_versions_used(raw: object) -> tuple[dict, ...]:
    if not isinstance(raw, list):
        raise ValueError("catalog_versions_used must be an array")
    references: list[dict] = []
    seen: set[str] = set()
    required = {"catalog_id", "version", "effective_at", "resolved_at"}
    for index, reference in enumerate(raw):
        if not isinstance(reference, dict) or set(reference) != required:
            raise ValueError(f"catalog_versions_used[{index}] must contain exactly {sorted(required)}")
        catalog_id = reference["catalog_id"]
        if not isinstance(catalog_id, str) or not catalog_id:
            raise ValueError(f"catalog_versions_used[{index}].catalog_id must be non-empty")
        if catalog_id in seen:
            raise ValueError(f"catalog_versions_used contains ambiguous duplicate {catalog_id!r}")
        seen.add(catalog_id)
        for field, minimum in (("version", 1), ("effective_at", 0), ("resolved_at", 0)):
            value = reference[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"catalog_versions_used[{index}].{field} must be >= {minimum}")
        references.append(dict(reference))
    return tuple(references)
