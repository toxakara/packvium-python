"""The three extension points: placement constraints, item orderings and solvers.

An extension may narrow what the library accepts but never widen it — the built-in
physical rules are applied alongside a custom constraint, not replaced by it. That
boundary is what these tests protect.
"""

from __future__ import annotations

from packvium import ExtensionRegistry, Packer, PackingConfig
from packvium.constraints import ConstraintResult
from packvium.solvers import ExtremePointSolver
from support import assert_sound, container, item


class RejectEverything:
    def evaluate(self, context):
        return ConstraintResult.reject("custom_rejection", context.item.id)


class FloorOnly:
    def evaluate(self, context):
        if context.point.z != 0:
            return ConstraintResult.reject("custom_floor_only")
        return ConstraintResult.allow()


class ReverseOrder:
    name = "reverse"

    def __init__(self):
        self.calls = []

    def order(self, items, seed):
        self.calls.append(seed)
        return tuple(reversed(items))


class RecordingSolver:
    """Delegates to the built-in solver but notes that it was consulted."""

    name = "recording"

    def __init__(self):
        self.calls = 0
        self._inner = ExtremePointSolver()

    def pack_one(self, container, sequence, items, config, stats, deadline):
        self.calls += 1
        return self._inner.pack_one(container, sequence, items, config, stats, deadline)


def packer(config=None, **extensions) -> Packer:
    return Packer(config or PackingConfig.balanced(), ExtensionRegistry(**extensions))


# ------------------------------------------------------------------- defaults

def test_an_empty_registry_registers_nothing():
    registry = ExtensionRegistry()
    assert registry.placement_constraints == ()
    assert registry.item_order_strategies == ()
    assert registry.solvers == ()
    assert registry.container_selector is None


# ---------------------------------------------------------------- constraints

def test_a_constraint_that_refuses_everything_leaves_the_order_unpacked():
    items = [item("a", 10, 10, 10)]
    containers = [container("b", 100, 100, 100)]
    result = packer(placement_constraints=(RejectEverything(),)).pack(items, containers)

    assert not result.complete
    assert result.containers == ()


def test_a_custom_constraint_is_applied_at_every_candidate_point():
    items = [item("a", 40, 40, 40, quantity=8)]
    containers = [container("c", 100, 100, 100, quantity=4)]
    result = packer(placement_constraints=(FloorOnly(),)).pack(items, containers)

    assert all(p.envelope_origin.z == 0 for c in result.containers for p in c.placements)
    assert_sound(result, items, containers)


def test_a_custom_constraint_cannot_widen_the_built_in_rules():
    """An always-allow extension still cannot make an over-large item fit: the physical
    checks run alongside it rather than being replaced by it."""
    class AllowEverything:
        def evaluate(self, context):
            return ConstraintResult.allow()

    items = [item("slab", 200, 200, 200)]
    containers = [container("c", 100, 100, 100)]
    assert not packer(placement_constraints=(AllowEverything(),)).pack(items, containers).complete


def test_a_custom_constraint_disables_the_lattice_fast_path():
    """The lattice places without consulting the constraint list, so a request carrying
    a custom rule must not be routed to it."""
    items = [item("a", 40, 40, 40, quantity=4)]
    containers = [container("c", 100, 100, 100)]
    result = packer(placement_constraints=(FloorOnly(),)).pack(items, containers)
    assert not result.algorithm.solver.startswith("grid")


# ------------------------------------------------------------------ orderings

def test_a_custom_ordering_is_consulted():
    strategy = ReverseOrder()
    items = [item("a", 10, 10, 10), item("b", 20, 20, 20)]
    containers = [container("box", 100, 100, 100)]
    result = Packer(PackingConfig.balanced(multi_start_orders=5, seed=13),
                    ExtensionRegistry(item_order_strategies=(strategy,))).pack(items, containers)

    assert strategy.calls == [13], "the configured seed is handed to the strategy"
    assert result.complete
    assert_sound(result, items, containers)


def test_a_custom_ordering_that_duplicates_a_built_in_one_is_not_run_twice():
    class VolumeDescending:
        name = "volume_again"

        def order(self, items, seed):
            return tuple(sorted(items, key=lambda i: (-i.dimensions.volume, -i.dimensions.max_edge,
                                                      -i.weight.ticks, i.id)))

    items = [item("a", 10, 10, 10), item("b", 20, 20, 20)]
    containers = [container("box", 100, 100, 100)]
    result = Packer(PackingConfig.balanced(multi_start_orders=8),
                    ExtensionRegistry(item_order_strategies=(VolumeDescending(),))).pack(items, containers)
    assert "volume_again" not in result.algorithm.solver


# -------------------------------------------------------------------- solvers

def test_a_custom_solver_is_consulted():
    solver = RecordingSolver()
    items = [item("a", 40, 40, 40, quantity=4)]
    containers = [container("c", 100, 100, 100)]
    result = Packer(PackingConfig.balanced(), ExtensionRegistry(solvers=(solver,))).pack(items, containers)

    assert solver.calls > 0
    assert_sound(result, items, containers)


def test_a_custom_solver_competes_with_the_built_in_ones_rather_than_replacing_them():
    """A custom solver that packs nothing must not be able to make the whole request fail."""
    class PacksNothing:
        name = "nothing"

        def pack_one(self, container, sequence, items, config, stats, deadline):
            from packvium.solvers import ContainerState, SingleContainerSolution
            return SingleContainerSolution(ContainerState(container, sequence), tuple(items))

    items = [item("a", 40, 40, 40, quantity=4)]
    containers = [container("c", 100, 100, 100)]
    result = Packer(PackingConfig.balanced(), ExtensionRegistry(solvers=(PacksNothing(),))).pack(items, containers)

    assert result.complete
    assert not result.algorithm.solver.startswith("nothing")


# --------------------------------------------------------- container selection

def test_a_custom_container_selector_can_prefer_lower_dimensional_weight_over_cost():
    # "big" is cheaper by cost_minor and would win under the default selector, but a
    # shipping-cost-aware selector can prefer whichever container is lighter to bill --
    # dimensional weight, exposed exactly for this purpose.
    from packvium import dimensional_weight

    class ByDimensionalWeight:
        def __init__(self):
            self.calls = 0

        def score(self, container, solution):
            self.calls += 1
            return (dimensional_weight(container.inner_dimensions, 139, "in", "lb").ticks, container.id)

    selector = ByDimensionalWeight()
    items = [item("a", 90, 90, 90)]
    containers = [
        container("big", 200, 200, 200, cost_minor=100),
        container("small", 110, 110, 110, cost_minor=200),
    ]
    result = Packer(PackingConfig.balanced(), ExtensionRegistry(container_selector=selector)).pack(items, containers)

    assert selector.calls > 0
    assert result.complete
    assert result.containers[0].container.id == "small"


# ------------------------------------------------------------ solution scoring

def test_a_custom_solution_scorer_is_consulted():
    """Passed directly to `Packer`, on the same terms as PHP's constructor argument --
    not through `ExtensionRegistry`, since ranking finished solutions is a different
    concern from shaping how a single container is packed."""
    class CountingScorer:
        def __init__(self):
            self.calls = 0

        def score(self, solution):
            self.calls += 1
            return (len(solution.unpacked), len(solution.containers), 0, 0, 0)

    scorer = CountingScorer()
    items = [item("a", 40, 40, 40, quantity=4)]
    containers = [container("c", 100, 100, 100)]
    result = Packer(PackingConfig.balanced(), solution_scorer=scorer).pack(items, containers)

    assert scorer.calls > 0
    assert result.complete


def test_a_custom_solution_scorer_determines_the_reported_score():
    class ConstantScorer:
        def score(self, solution):
            return (0, 0, 0, 0, 42)

    items = [item("a", 40, 40, 40, quantity=4)]
    containers = [container("c", 100, 100, 100)]
    result = Packer(PackingConfig.balanced(), solution_scorer=ConstantScorer()).pack(items, containers)

    assert tuple(result.score) == (0, 0, 0, 0, 42)


def test_a_stable_solution_scorer_can_select_the_better_centre_of_mass():
    """The ranking proof: the scorer changes the selected *packing*, not merely
    the score printed on an unchanged winner."""
    class BalanceScorer:
        def score(self, solution):
            offset = sum(
                container.centre_of_mass_offset_ppm
                for container in solution.containers
            )
            return (
                len(solution.unpacked),
                offset,
                len(solution.containers),
                0,
                0,
            )

    items = [
        item("a", 34, 58, 36, weight=42, quantity=3),
        item("b", 42, 41, 35, weight=941, quantity=3),
        item("c", 40, 32, 47, weight=914, quantity=2),
    ]
    containers = [container("c", 100, 100, 100, quantity=2)]
    config = PackingConfig.quality(time_limit_ms=10_000, seed=0, top_k=20)
    default = Packer(config).pack(items, containers)
    balanced = Packer(config, solution_scorer=BalanceScorer()).pack(items, containers)
    default_offset = sum(c.centre_of_mass_offset_ppm for c in default.containers)
    balanced_offset = sum(c.centre_of_mass_offset_ppm for c in balanced.containers)

    assert balanced_offset < default_offset
    assert tuple(balanced.score) == (0, balanced_offset, 1, 0, 0)
