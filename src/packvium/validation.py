from __future__ import annotations

from ._compat import dataclass

from .axle_load import axle_load_exceeded
from .constraints import (CompatibilityConstraint, ConstraintContext, LoadSupportGraph,
                          SupportConstraint, TagCountConstraint, load_units,
                          non_stackable_failure, overloaded, stack_density_exceeded,
                          stack_limit_exceeded)
from .geometry import AxisAlignedBox, Point
from .models import PackedContainer, PackingRequest, UnpackedItem
from .nesting import is_valid_nesting as _is_valid_nesting
from .packing_sequence import RouteSequenceError, safe_route_removal_order
from .units import Length


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    issues: tuple[ValidationIssue, ...]


class IndependentSolutionValidator:
    """Re-derives every guarantee from the placements alone.

    It shares no state with the solvers, so a solver that reports a placement it never
    actually verified is caught here rather than reaching the caller.
    """

    def validate(
        self,
        request: PackingRequest,
        containers: tuple[PackedContainer, ...],
        minimum_support_ratio: float = 0.0,
        clearance: Length | None = None,
        unpacked: tuple[UnpackedItem, ...] | None = None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []; seen = set(); expected = {i.id for i in request.instances}; inventory = {}
        clearance_ticks = 0 if clearance is None else clearance.ticks
        for packed in containers:
            inventory[packed.container.id] = inventory.get(packed.container.id, 0) + 1
            boundary = AxisAlignedBox(Point(0, 0, 0), packed.container.inner_dimensions); payload = 0
            placements = packed.placements
            compatibility_sensitive = any(p.instance.item.tags or p.instance.item.incompatible_tags for p in placements)
            support_sensitive = (minimum_support_ratio > 0
                                 or any(p.instance.item.minimum_support_ratio > 0
                                        or p.instance.item.ground_contact_rule not in (None, "free")
                                        for p in placements))
            stack_sensitive = any(not p.instance.item.stackable for p in placements)
            stack_graph = LoadSupportGraph(load_units(placements)) if stack_sensitive else None
            for left, right in self._collision_pairs(placements):
                issues.append(ValidationIssue(
                    "collision",
                    f"{placements[left].instance.id} with {placements[right].instance.id}",
                ))
            for index, placement in enumerate(packed.placements):
                item_id = placement.instance.id
                if item_id in seen: issues.append(ValidationIssue("duplicate_item", item_id))
                seen.add(item_id); payload += placement.instance.weight.ticks
                if not boundary.contains(placement.envelope_box): issues.append(ValidationIssue("outside_container", item_id))
                if placement.rotation not in placement.instance.item.allowed_rotations: issues.append(ValidationIssue("forbidden_rotation", item_id))
                if placement.dimensions != placement.instance.item.dimensions.rotated(placement.rotation): issues.append(ValidationIssue("dimension_mismatch", item_id))
                if not self._envelope_matches(placement, clearance_ticks): issues.append(ValidationIssue("clearance_mismatch", item_id))
                if any(placement.envelope_box.intersects(box) for o in packed.container.obstacles for box in o.boxes): issues.append(ValidationIssue("obstacle_collision", item_id))
                if placement.instance.item.must_be_on_floor and placement.envelope_origin.z != 0:
                    issues.append(ValidationIssue("must_be_on_floor", f"{item_id}: "))
                eligible_tags = placement.instance.item.eligible_container_tags
                if eligible_tags and not (eligible_tags & packed.container.tags):
                    issues.append(ValidationIssue("container_ineligible", item_id))
                # The common lattice case has no pair-dependent rules. Avoid building
                # an O(n) "all other placements" tuple for every item in that case.
                if compatibility_sensitive or support_sensitive:
                    others = placements[:index] + placements[index + 1:]
                    context = ConstraintContext(packed.container, others, placement.instance, placement.envelope_origin,
                                                placement.rotation, placement.dimensions, placement.envelope_dimensions)
                else:
                    context = None
                constraints = ()
                if compatibility_sensitive:
                    constraints += (CompatibilityConstraint(), TagCountConstraint())
                if support_sensitive:
                    constraints += (SupportConstraint(minimum_support_ratio),)
                for constraint in constraints:
                    result = constraint.evaluate(context)
                    if not result.allowed: issues.append(ValidationIssue(result.code, f"{item_id}: {result.detail}"))
                if stack_sensitive:
                    assert stack_graph is not None
                    result = non_stackable_failure(
                        placements, placement.instance, stack_graph, index
                    )
                    if result is not None:
                        issues.append(ValidationIssue(result.code, f"{item_id}: {result.detail}"))
            # A second, whole-container bearing pass: the per-placement check above
            # reports the first offender it meets, this one is anchored on the container.
            units = load_units(packed.placements)
            density_limit = None if packed.container.max_stack_density is None else packed.container.max_stack_density.ticks
            failure = overloaded(units) or stack_limit_exceeded(units) or stack_density_exceeded(units, density_limit)
            if failure is not None: issues.append(ValidationIssue(failure[0], f"{packed.id}: {failure[1]}"))
            if packed.container.axles is not None:
                axle_failure = axle_load_exceeded(
                    packed.container.axles,
                    units,
                    packed.container.tare_weight.ticks,
                    packed.container.inner_dimensions.length.ticks,
                )
                if axle_failure is not None: issues.append(ValidationIssue(axle_failure[0], f"{packed.id}: {axle_failure[1]}"))
            if any(p.instance.item.stop_index is not None for p in placements):
                route_issue = self._unloading_order_violation(packed)
                if route_issue is not None: issues.append(route_issue)
            if packed.container.max_payload is not None and payload > packed.container.max_payload.ticks: issues.append(ValidationIssue("payload_exceeded", packed.id))
            if packed.container.max_items is not None and len(packed.placements) > packed.container.max_items: issues.append(ValidationIssue("max_items_exceeded", packed.id))
        for container in request.containers:
            if container.quantity is not None and inventory.get(container.id, 0) > container.quantity: issues.append(ValidationIssue("container_inventory_exceeded", container.id))
        if unpacked is not None:
            for item in unpacked:
                item_id = item.instance.id
                if item_id in seen:
                    issues.append(ValidationIssue("duplicate_item", f"{item_id} is both packed and unpacked"))
                seen.add(item_id)
                if not item.reason:
                    issues.append(ValidationIssue("missing_reason", item_id))
            for item_id in sorted(expected - seen):
                issues.append(ValidationIssue("missing_item", item_id))
        for item_id in sorted(seen - expected): issues.append(ValidationIssue("unknown_item", item_id))
        self._check_groups(containers, issues)
        if unpacked is not None:
            self._check_group_accounting(request, containers, issues)
        return ValidationReport(not issues, tuple(issues))

    @staticmethod
    def _collision_pairs(placements) -> list[tuple[int, int]]:
        """Return intersecting pairs with a sweep along x, bucketed by z-level first.

        A valid nested pair is excluded: it is a deliberate, exactly
        bounded overlap between two identical items, one sunk into the other, not a
        collision.

        Pruning the sweep's active set by x-overlap alone -- as a single sweep over
        every placement did previously -- leaves a large multi-layer lattice (many
        z-levels that all share similar x ranges) with an active set that stays close
        to O(n), turning the sweep into O(n * active_size) instead of the intended
        near-linear behaviour. Bucketing by z-level first, mirroring
        `contact.py`'s `_LevelIndex`, bounds each bucket's sweep by that level's own
        density instead of the whole container's.
        """
        if not placements:
            return []
        cell = max((p.envelope_box.z2 - p.envelope_box.origin.z for p in placements), default=1) or 1
        buckets: dict[int, list[int]] = {}
        for index, placement in enumerate(placements):
            box = placement.envelope_box
            for z_cell in {box.origin.z // cell, (box.z2 - 1) // cell}:
                buckets.setdefault(z_cell, []).append(index)
        pairs: set[tuple[int, int]] = set()
        for indices in buckets.values():
            ordered = sorted(
                ((placements[i].envelope_box.origin.x, placements[i].envelope_box.x2, i) for i in indices),
                key=lambda entry: (entry[0], entry[1], entry[2]),
            )
            active: list[tuple[int, int]] = []
            for x1, x2, index in ordered:
                active = [(right, other) for right, other in active if right > x1]
                box = placements[index].envelope_box
                for _, other in active:
                    if box.intersects(placements[other].envelope_box) and not _is_valid_nesting(placements[index], placements[other]):
                        pairs.add((min(index, other), max(index, other)))
                active.append((x2, index))
        return sorted(pairs)

    @staticmethod
    def _unloading_order_violation(packed: PackedContainer) -> "ValidationIssue | None":
        """Whether this container's route can actually be unloaded stop by
        stop, using the same geometry `safe_route_removal_order` already proves safe
        removal orders against -- no separate notion of "blocked" for this check to
        disagree with the one `packing_sequence` uses.

        Physical, not envelope, boxes: reachability is about what a forklift or hand
        would actually collide with, and clearance is a solver placement margin, not a
        real obstruction.
        """
        placements = packed.placements
        boxes = tuple(p.box for p in placements)
        stops = tuple(p.instance.item.stop_index for p in placements)
        try:
            safe_route_removal_order(boxes, stops, packed.container.inner_dimensions)
        except RouteSequenceError as error:
            stuck = ", ".join(placements[index].instance.id for index in sorted(error.stuck))
            return ValidationIssue(
                "unloading_order_violation",
                f"{packed.id}: stop {error.stop} cannot be fully unloaded ({stuck} still blocked)",
            )
        return None

    @staticmethod
    def _envelope_matches(placement, clearance_ticks: int) -> bool:
        expected = placement.dimensions.expand(Length(clearance_ticks)) if clearance_ticks else placement.dimensions
        return (placement.envelope_dimensions == expected
                and placement.position.x == placement.envelope_origin.x + clearance_ticks
                and placement.position.y == placement.envelope_origin.y + clearance_ticks
                and placement.position.z == placement.envelope_origin.z + clearance_ticks)

    @staticmethod
    def _check_groups(containers, issues: list) -> None:
        located: dict[str, set[str]] = {}
        for packed in containers:
            for placement in packed.placements:
                group = placement.instance.item.group
                if group is None: continue
                located.setdefault(group, set()).add(packed.id)
        for group, where in sorted(located.items()):
            if len(where) > 1: issues.append(ValidationIssue("group_split", f"{group}: {', '.join(sorted(where))}"))

    @staticmethod
    def _check_group_accounting(request: PackingRequest, containers, issues: list) -> None:
        expected: dict[str, set[str]] = {}
        placed: dict[str, set[str]] = {}
        for instance in request.instances:
            if instance.item.group is not None:
                expected.setdefault(instance.item.group, set()).add(instance.id)
        for packed in containers:
            for placement in packed.placements:
                group = placement.instance.item.group
                if group is not None:
                    placed.setdefault(group, set()).add(placement.instance.id)
        for group, members in sorted(expected.items()):
            count = len(placed.get(group, set()))
            if 0 < count < len(members):
                issues.append(ValidationIssue("group_partial", f"{group}: {count}/{len(members)} packed"))
