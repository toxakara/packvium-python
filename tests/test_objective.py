"""The objective vector — five lexicographic integer keys, lower is better.

Its arithmetic is pinned here rather than left to whatever the solvers happen to
produce, because every implementation of this library must compute the identical
formula: a container volume in cubic ticks is far past the range where a double is
exact, so a floating-point score would silently disagree across languages.
See docs/OBJECTIVE.md.
"""

from __future__ import annotations

import pytest

from packvium import (Container, Dimensions, Item, PackedContainer, PackingConfig, Placement,
                         Point, Rotation, UnpackedItem)
from packvium.extensions import DefaultSolutionScorer
from support import assert_sound, container, item, pack

MM = 16_000


def filled(box: Container, instances, positions) -> PackedContainer:
    return PackedContainer(box, 1, tuple(
        Placement(instance, Point(*at), Rotation.LWH, instance.item.dimensions, Point(*at),
                  instance.item.dimensions)
        for instance, at in zip(instances, positions)
    ))


# ------------------------------------------------------------------------- shape

def test_the_vector_has_five_exact_integer_keys():
    result = pack([item("a", 100, 100, 100)], [container("c", 200, 200, 200)])
    assert len(result.score) == 5
    assert all(isinstance(key, int) for key in result.score)


def test_an_empty_solution_scores_only_its_unpacked_items():
    lost = Item.create("a", Dimensions.mm(1, 1, 1)).instances()
    assert DefaultSolutionScorer.score_containers((), tuple(UnpackedItem(i, "no_feasible_placement") for i in lost)) == (1, 0, 0, 0, 0)


# ------------------------------------------------------------------- arithmetic

def test_a_perfectly_filled_container_wastes_no_volume():
    """Eight 100 mm cubes exactly fill a 200 mm cube, so the unused-volume key is zero
    and the stack reaches the full height."""
    cubes = Item.create("cube", Dimensions.mm(100, 100, 100), quantity=8)
    box = Container.create("box", Dimensions.mm(200, 200, 200))
    lattice = [(x * 100 * MM, y * 100 * MM, z * 100 * MM)
               for z in range(2) for y in range(2) for x in range(2)]
    assert DefaultSolutionScorer.score_containers((filled(box, cubes.instances(), lattice),), ()) == (0, 1, 0, 0, DefaultSolutionScorer.SCALE)


def test_the_ratios_are_hand_checkable():
    """One eighth of the volume used, half the height reached."""
    cube = Item.create("cube", Dimensions.mm(100, 100, 100))
    box = Container.create("box", Dimensions.mm(200, 200, 200))
    assert DefaultSolutionScorer.score_containers((filled(box, cube.instances(), [(0, 0, 0)]),), ()) == (0, 1, 0, 875_000, 500_000)


def test_container_cost_is_summed_in_minor_units():
    cubes = Item.create("cube", Dimensions.mm(100, 100, 100), quantity=2)
    box = Container.create("box", Dimensions.mm(200, 200, 200), cost_minor=395)
    first, second = cubes.instances()
    score = DefaultSolutionScorer.score_containers(
        (filled(box, [first], [(0, 0, 0)]), PackedContainer(box, 2, filled(box, [second], [(0, 0, 0)]).placements)),
        (),
    )
    assert score[1:3] == (2, 790)


def test_the_volume_ratio_is_floored_per_container():
    """Per-container rather than pooled, so a large carton is not preferred merely for
    being large; floored so the key stays an exact integer."""
    cube = Item.create("cube", Dimensions.mm(30, 30, 30))
    box = Container.create("box", Dimensions.mm(100, 100, 100))
    used = cube.dimensions.volume
    volume = box.inner_dimensions.volume
    _, _, _, unused, _ = DefaultSolutionScorer.score_containers((filled(box, cube.instances(), [(0, 0, 0)]),), ())
    assert unused == (volume - used) * DefaultSolutionScorer.SCALE // volume


# ------------------------------------------------------------------ ordering

def test_placing_everything_dominates_every_other_key():
    """Key 0 first: an answer that leaves an item behind loses to one that does not,
    however elegantly it fills its cartons."""
    complete = DefaultSolutionScorer.score_containers((), ())
    incomplete = (0, 0, 0, 0, 0)
    lost = Item.create("a", Dimensions.mm(1, 1, 1)).instances()
    with_leftover = DefaultSolutionScorer.score_containers((), tuple(UnpackedItem(i, "x") for i in lost))
    assert complete == incomplete
    assert complete < with_leftover


def test_fewer_containers_beat_more():
    cubes = Item.create("cube", Dimensions.mm(100, 100, 100), quantity=2)
    box = Container.create("box", Dimensions.mm(200, 200, 200))
    first, second = cubes.instances()
    together = DefaultSolutionScorer.score_containers((filled(box, [first, second], [(0, 0, 0), (100 * MM, 0, 0)]),), ())
    apart = DefaultSolutionScorer.score_containers(
        (filled(box, [first], [(0, 0, 0)]), PackedContainer(box, 2, filled(box, [second], [(0, 0, 0)]).placements)),
        (),
    )
    assert together < apart


def test_a_lower_centre_of_load_ranks_better():
    cubes = Item.create("cube", Dimensions.mm(100, 100, 100), quantity=2)
    box = Container.create("box", Dimensions.mm(200, 200, 200))
    side_by_side = DefaultSolutionScorer.score_containers((filled(box, cubes.instances(), [(0, 0, 0), (100 * MM, 0, 0)]),), ())
    stacked = DefaultSolutionScorer.score_containers((filled(box, cubes.instances(), [(0, 0, 0), (0, 0, 100 * MM)]),), ())
    assert side_by_side < stacked


# ------------------------------------------------------------- reported result

def test_the_reported_score_matches_a_recomputation_from_the_placements():
    """The score a caller receives must describe the packing they received. This is the
    same check the cross-language conformance runner makes of every implementation."""
    items = [item("a", 40, 30, 20, quantity=6, weight="500 g"), item("b", 60, 60, 60, quantity=2)]
    containers = [container("small", 100, 100, 100, cost_minor=200),
                  container("large", 200, 200, 200, cost_minor=350)]
    result = pack(items, containers, PackingConfig.quality(time_limit_ms=2_000))

    assert_sound(result, items, containers)
    assert tuple(result.score) == DefaultSolutionScorer.score_containers(result.containers, result.unpacked)


def test_alternatives_are_never_better_than_the_chosen_answer():
    items = [item("a", 40, 30, 20, quantity=8)]
    containers = [container("c", 100, 100, 100, quantity=4)]
    result = pack(items, containers, PackingConfig.quality(time_limit_ms=2_000, top_k=5))
    assert all(tuple(result.score) <= tuple(other.score) for other in result.alternatives)


@pytest.mark.parametrize("profile", [PackingConfig.fast, PackingConfig.balanced, PackingConfig.quality])
def test_every_profile_reports_a_score_it_can_defend(profile):
    items = [item("cube", 100, 100, 100, quantity=8)]
    containers = [container("box", 200, 200, 200)]
    result = pack(items, containers, profile())
    assert tuple(result.score) == DefaultSolutionScorer.score_containers(result.containers, result.unpacked)


# ------------------------------------------------------------- selectable objective

def test_lowest_cost_prefers_fewer_dollars_over_fewer_containers():
    """The default objective ranks by container count ahead of cost -- fewer boxes wins
    even if it costs more. lowest_cost inverts that: cost decides before count."""
    from packvium.extensions import LowestCostSolutionScorer

    big_expensive = Container.create("big", Dimensions.mm(200, 200, 200), cost_minor=1000)
    small_cheap = Container.create("small", Dimensions.mm(100, 100, 100), cost_minor=100)
    cubes = Item.create("cube", Dimensions.mm(100, 100, 100), quantity=2)
    first, second = cubes.instances()

    one_container = (filled(big_expensive, [first, second], [(0, 0, 0), (100 * MM, 0, 0)]),)
    two_containers = (
        filled(small_cheap, [first], [(0, 0, 0)]),
        PackedContainer(small_cheap, 2, filled(small_cheap, [second], [(0, 0, 0)]).placements),
    )

    default_winner = min([one_container, two_containers], key=lambda c: DefaultSolutionScorer.score_containers(c, ()))
    cost_winner = min([one_container, two_containers], key=lambda c: LowestCostSolutionScorer.score_containers(c, ()))

    assert default_winner is one_container   # fewer containers wins by default (cost 1000 > 200)
    assert cost_winner is two_containers      # lowest_cost prefers 200 over 1000 even though it's 2 boxes


def test_the_default_objective_name_reproduces_todays_ranking_exactly():
    items = [item("cube", 100, 100, 100, quantity=8)]
    containers = [container("box", 200, 200, 200)]
    explicit_default = pack(items, containers, PackingConfig.balanced(objective="default"))
    unspecified = pack(items, containers, PackingConfig.balanced())
    assert tuple(explicit_default.score) == tuple(unspecified.score)
    assert explicit_default.objective == unspecified.objective == "default"


def test_the_result_reports_which_objective_was_used():
    items = [item("a", 40, 40, 40)]
    containers = [container("c", 100, 100, 100)]
    result = pack(items, containers, PackingConfig.balanced(objective="lowest_cost"))
    assert result.objective == "lowest_cost"


def test_an_unknown_objective_name_is_rejected():
    from packvium.extensions import UnknownObjectiveError

    items = [item("a", 40, 40, 40)]
    containers = [container("c", 100, 100, 100)]
    with pytest.raises(UnknownObjectiveError):
        pack(items, containers, PackingConfig.balanced(objective="cheapest_shipping_ever"))


# --------------------------------------------------------- shipping cost

def test_shipping_cost_is_hand_checkable_from_a_carrier_divisor():
    """A 200mm cube container (20cm/side) with divisor 5000 (cm/kg): dimensional
    weight is 20*20*20/5000 = 1.6kg. A 1g item's actual gross weight is far below
    that, so the billable weight is the dimensional weight, in weight ticks
    (1 tick = 1/8 microgram, so 1.6kg = 1,600g = 1,600 * 8,000,000 ticks)."""
    from packvium.extensions import ShippingCostSolutionScorer

    cube = Item.create("cube", Dimensions.mm(100, 100, 100), weight="1g")
    box = Container.create("box", Dimensions.mm(200, 200, 200))
    scorer = ShippingCostSolutionScorer(divisor=5000, length_unit="cm", weight_unit="kg")
    score = scorer.score_containers((filled(box, cube.instances(), [(0, 0, 0)]),), ())
    assert score == (0, 1_600 * 8_000_000, 1, 875_000, 500_000)


def test_shipping_cost_bills_the_greater_of_actual_and_dimensional_weight():
    """A heavy, tightly-fitted item bills its actual weight; dimensional weight only
    takes over once the shipment is light for its size."""
    from packvium.extensions import ShippingCostSolutionScorer
    from packvium.units import Weight

    cube = Item.create("cube", Dimensions.mm(100, 100, 100), weight="5kg")
    box = Container.create("box", Dimensions.mm(200, 200, 200))
    scorer = ShippingCostSolutionScorer(divisor=5000, length_unit="cm", weight_unit="kg")
    score = scorer.score_containers((filled(box, cube.instances(), [(0, 0, 0)]),), ())
    assert score[1] == Weight.parse("5kg").ticks  # actual weight beats the 1.6kg dimensional figure


def test_shipping_cost_uses_outer_dimensions_when_declared_not_usable_capacity():
    """A carrier measures the box that moves, not its usable interior -- a declared
    outer size must win over the (smaller) inner capacity used everywhere else."""
    from packvium.extensions import ShippingCostSolutionScorer

    cube = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    box = Container.create("box", Dimensions.mm(20, 20, 20), outer_dimensions=Dimensions.mm(30, 30, 30))
    scorer = ShippingCostSolutionScorer(divisor=5000, length_unit="cm", weight_unit="kg")
    score = scorer.score_containers((filled(box, cube.instances(), [(0, 0, 0)]),), ())
    # 30mm = 3cm/side -> 27cm^3 / 5000 = 5.4g, not the 20mm inner's 1.6g.
    assert score[1] == 43_200_000


def test_shipping_cost_prefers_the_smaller_bulky_container():
    """Same light item, two container sizes: the smaller one has less dimensional
    weight, so shipping_cost prefers it even though default (fewest containers, no
    tie-break on size) is indifferent between them."""
    from packvium.extensions import ShippingCostSolutionScorer

    light_item = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    snug = Container.create("snug", Dimensions.mm(20, 20, 20))
    oversized = Container.create("oversized", Dimensions.mm(30, 30, 30))
    scorer = ShippingCostSolutionScorer(divisor=5000, length_unit="cm", weight_unit="kg")

    in_snug = (filled(snug, light_item.instances(), [(0, 0, 0)]),)
    in_oversized = (filled(oversized, light_item.instances(), [(0, 0, 0)]),)
    winner = min([in_snug, in_oversized], key=lambda c: scorer.score_containers(c, ()))
    assert winner is in_snug


def test_shipping_cost_requires_a_divisor_and_reports_the_objective_used():
    from packvium.extensions import ShippingCostSolutionScorer, UnknownObjectiveError

    items = [item("a", 40, 40, 40)]
    containers = [container("c", 100, 100, 100)]
    with pytest.raises(
        UnknownObjectiveError,
        match="the shipping_cost objective requires configuration.dimensional_weight_divisor",
    ):
        ShippingCostSolutionScorer.from_config(None)
    with pytest.raises(UnknownObjectiveError):
        pack(items, containers, PackingConfig.balanced(objective="shipping_cost"))

    result = pack(
        items, containers,
        PackingConfig.balanced(objective="shipping_cost", dimensional_weight_divisor=5000),
    )
    assert result.objective == "shipping_cost"
    assert len(result.score) == 5


# ----------------------------------------------------------- landed cost

# Every assertion here is about money, and every number is hand-checkable from the table
# above it. These mirror the PHP suite's landed-cost tests case for case: the objective
# was only ever exercised through the cross-language fixture, which proves four engines
# agree but says nothing about which of this scorer's own branches ever ran.

def landed(box: Container, thing: Item, divisor: int = 5_000) -> tuple[int, ...]:
    from packvium.extensions import LandedCostSolutionScorer

    scorer = LandedCostSolutionScorer(divisor=divisor, length_unit="cm", weight_unit="kg")
    return scorer.score_containers((filled(box, thing.instances(), [(0, 0, 0)]),), ())


def test_landed_cost_is_hand_checkable_from_a_bracket_table():
    """200mm cube = 20cm/side -> 8,000cm^3 / 5,000 = 1.6kg = 1,600g billed, which beats
    the 1g item's actual weight. 1,600 is above the 1,000g bracket and inside the
    2,000g one, so the price is 900 minor units."""
    from packvium.models import RateTable

    cube = Item.create("cube", Dimensions.mm(100, 100, 100), weight="1g")
    box = Container.create("box", Dimensions.mm(200, 200, 200),
                           rate_table=RateTable((1_000, 2_000), (500, 900)))
    assert landed(box, cube) == (0, 900, 1, 875_000, 500_000)


def test_landed_cost_and_shipping_cost_disagree_when_a_price_band_dips():
    """The one case that justifies a second objective. The snug box bills 2g and the
    oversized one 6g, so `shipping_cost` -- which ranks in grams -- always prefers snug.
    This tariff's second band is cheaper than its first, so the money runs the other
    way. A rate card that dips is a real promotional band, which is why RateTable
    accepts one rather than rejecting it as malformed."""
    from packvium.extensions import LandedCostSolutionScorer, ShippingCostSolutionScorer
    from packvium.models import RateTable

    tariff = RateTable((4, 10), (500, 300))
    light = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    snug = Container.create("snug", Dimensions.mm(20, 20, 20), rate_table=tariff)
    oversized = Container.create("oversized", Dimensions.mm(30, 30, 30), rate_table=tariff)
    assert landed(snug, light)[1] == 500
    assert landed(oversized, light)[1] == 300

    in_snug = (filled(snug, light.instances(), [(0, 0, 0)]),)
    in_oversized = (filled(oversized, light.instances(), [(0, 0, 0)]),)
    by_weight = ShippingCostSolutionScorer(divisor=5_000, length_unit="cm", weight_unit="kg")
    by_money = LandedCostSolutionScorer(divisor=5_000, length_unit="cm", weight_unit="kg")
    candidates = [in_snug, in_oversized]
    assert min(candidates, key=lambda c: by_weight.score_containers(c, ())) is in_snug
    assert min(candidates, key=lambda c: by_money.score_containers(c, ())) is in_oversized


def test_landed_cost_bills_whole_grams_rounded_up():
    """20mm cube -> 8cm^3 / 5,000 = 1.6g. A carrier reads a scale upward, so this is 2g
    and lands in the second band. Rounding down would price it at 100 -- below what the
    carrier charges, which is the one direction that must never happen."""
    from packvium.models import RateTable

    light = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    box = Container.create("box", Dimensions.mm(20, 20, 20), rate_table=RateTable((1, 5), (100, 700)))
    assert landed(box, light)[1] == 700


def test_landed_cost_prices_the_outer_box_a_carrier_actually_moves():
    """30mm outer -> 5.4g -> 6g, in the second band; the 20mm interior would have billed
    2g and priced in the first."""
    from packvium.models import RateTable

    light = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    box = Container.create("box", Dimensions.mm(20, 20, 20),
                           outer_dimensions=Dimensions.mm(30, 30, 30),
                           rate_table=RateTable((4, 10), (500, 300)))
    assert landed(box, light)[1] == 300


def test_a_container_without_a_rate_table_is_refused_rather_than_ranked_free():
    """Rating some containers and not others would compare a priced packing against an
    unpriced one as though the unpriced were free -- and the objective would then prefer
    exactly the answer nobody quoted. Naming the container is the point of the refusal:
    a caller with thirty containers needs to know which rate card is missing."""
    from packvium.extensions import UnknownObjectiveError

    cube = Item.create("cube", Dimensions.mm(100, 100, 100), weight="1g")
    unrated = Container.create("unrated", Dimensions.mm(200, 200, 200))
    with pytest.raises(UnknownObjectiveError, match="'unrated'"):
        landed(unrated, cube)


def test_a_weight_above_the_last_bracket_has_no_price_and_says_so():
    """Clamping to the top price would under-quote every oversize shipment silently."""
    from packvium.models import RateTable, UnratedWeightError

    with pytest.raises(UnratedWeightError, match="no published price"):
        RateTable((1,), (100,)).charge_minor(2)


def test_unpriceable_container_answers_for_a_container_with_no_tariff_at_all():
    """`unpriceable_container` is exported, so it cannot assume the packer's admission ran.

    `Packer` never reaches this branch: `LandedCostSolutionScorer` refuses a missing
    `rate_table` while scoring, before the final guard is consulted. The function is
    public API all the same, and a caller checking a result it assembled itself must get
    an answer rather than an `AttributeError` off a `None` table.
    """
    from packvium.config import PackingConfig
    from packvium.extensions import unpriceable_container

    cube = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    box = Container.create("box", Dimensions.mm(20, 20, 20))
    packed = (filled(box, cube.instances(), [(0, 0, 0)]),)
    config = PackingConfig(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=5_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
    )
    # 20 mm cube = 2 cm a side -> 8 cm^3 / 5000 = 1.6 kg... of a gram, rounded up to 2 g,
    # which beats the item's own 1 g. No table means no bracket to name, hence 0.
    assert unpriceable_container(packed, config) == ("box", 2, 0)


def test_the_scorer_ranks_an_unpriceable_packing_worst_rather_than_raising():
    """The search has to *compare* an unpriceable candidate, not abort on one.

    Raising here is what made a request with one short tariff and one perfectly good
    alternative fail outright instead of shipping. Ranking it `UNPRICEABLE_MINOR`
    is what lets the priceable container win the round; `Packer.pack` is where the
    refusal belongs, once nothing priceable is left to prefer.
    """
    from packvium.extensions import UNPRICEABLE_MINOR
    from packvium.models import RateTable

    light = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    box = Container.create("box", Dimensions.mm(20, 20, 20), rate_table=RateTable((1,), (100,)))
    assert landed(box, light)[1] == UNPRICEABLE_MINOR


def test_packing_refuses_when_no_container_on_offer_can_price_the_load():
    """The sentinel is a search device; reaching a result with it still standing would
    quote a number the carrier never published."""
    from packvium.config import PackingConfig
    from packvium.models import RateTable, UnratedWeightError
    from packvium.packer import Packer

    light = Item.create("cube", Dimensions.mm(10, 10, 10), weight="1g")
    box = Container.create("box", Dimensions.mm(20, 20, 20), rate_table=RateTable((1,), (100,)))
    packer = Packer(PackingConfig(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=5_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
    ))
    with pytest.raises(UnratedWeightError, match="'box' bills at .* no published price"):
        packer.pack([light], [box])


def test_an_unpriceable_container_loses_to_a_priceable_one_in_the_greedy_round():
    """The general per-round choice, not the closed-form path.

    Eight units is past the single-item shape the fast paths take. The round key ranked
    by billed weight, and `alpha` bills lighter (5400 g of dimensional weight against
    12800 g) while its tariff stops at 2000 g -- so the objective chose the one shipment
    the caller cannot buy over one available at 1500. Ranking the round by the money the
    finished score will charge is what fixes it; Rust, PHP and the JavaScript fallback
    reach the identical answer.
    """
    from packvium.config import PackingConfig
    from packvium.models import RateTable
    from packvium.packer import Packer

    box = Item.create("box", Dimensions.mm(100, 100, 100), weight="500g", quantity=8)
    alpha = Container.create(
        "alpha_unpriceable", Dimensions.mm(300, 300, 300),
        rate_table=RateTable((2_000,), (900,)),
    )
    beta = Container.create(
        "beta_priceable", Dimensions.mm(400, 400, 400),
        rate_table=RateTable((20_000,), (1_500,)),
    )
    packer = Packer(PackingConfig(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=5_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
    ))
    result = packer.pack([box], [alpha, beta])
    assert [c.container.id for c in result.containers] == ["beta_priceable"]
    assert result.score[1] == 1_500
    assert result.unpacked == ()


def test_a_bracket_step_makes_the_cheaper_shipment_the_heavier_one():
    """Grams and money order candidates alike only while price rises smoothly with
    weight. Here the heavier container is the cheaper one, which is the whole reason
    this objective exists next to `shipping_cost`."""
    from packvium.config import PackingConfig
    from packvium.models import RateTable
    from packvium.packer import Packer

    box = Item.create("box", Dimensions.mm(100, 100, 100), weight="500g", quantity=8)
    dear = Container.create(
        "light_but_dear", Dimensions.mm(300, 300, 300),
        rate_table=RateTable((20_000,), (900,)),
    )
    cheap = Container.create(
        "heavy_but_cheap", Dimensions.mm(400, 400, 400),
        rate_table=RateTable((20_000,), (400,)),
    )
    packer = Packer(PackingConfig(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=5_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
    ))
    result = packer.pack([box], [dear, cheap])
    assert [c.container.id for c in result.containers] == ["heavy_but_cheap"]
    assert result.score[1] == 400


def test_a_quantity_compressed_round_still_prices_the_true_payload():
    """compact states carry no per-item `Placement`s, so a round key that sums
    `state.placements` prices a quantity-compressed trial as tare alone. Eight 2000 g
    cubes bill 16000 g -- past alpha's last bracket -- but alpha's dimensional 5400 g
    is not, so a tare-only key committed alpha and refused a request that beta ships
    at 1500. The key reads the lattice-aware `payload_ticks`/`placement_count`; the
    fast profile with coordinates waived is the configuration that takes this path.
    """
    from packvium.config import PackingConfig, SolverProfile
    from packvium.models import RateTable
    from packvium.packer import Packer

    box = Item.create("box", Dimensions.mm(100, 100, 100), weight="2000g", quantity=8)
    alpha = Container.create(
        "alpha_unpriceable", Dimensions.mm(300, 300, 300),
        rate_table=RateTable((10_000,), (800,)),
    )
    beta = Container.create(
        "beta_priceable", Dimensions.mm(400, 400, 400),
        rate_table=RateTable((20_000,), (1_500,)),
    )
    packer = Packer(PackingConfig(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=5_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
        profile=SolverProfile.FAST,
        require_placement_coordinates=False,
    ))
    result = packer.pack([box], [alpha, beta])
    assert [c.container.id for c in result.containers] == ["beta_priceable"]
    assert result.score[1] == 1_500
    assert result.unpacked == ()


# --------------------------------------------------------- maximum value

def test_maximum_value_ranks_forgone_value_ahead_of_container_count():
    from packvium.extensions import MaximumValueSolutionScorer

    lost_cheap = Item.create("foam", Dimensions.mm(1, 1, 1), value=1).instances()
    lost_pricey = Item.create("goods", Dimensions.mm(1, 1, 1), value=1000).instances()
    scorer = MaximumValueSolutionScorer()
    cheap_left_behind = scorer.score_containers((), tuple(UnpackedItem(i, "no_feasible_placement") for i in lost_cheap))
    pricey_left_behind = scorer.score_containers((), tuple(UnpackedItem(i, "no_feasible_placement") for i in lost_pricey))
    assert cheap_left_behind < pricey_left_behind
    assert cheap_left_behind == (1, 1, 0, 0, 0)
    assert pricey_left_behind == (1, 1000, 0, 0, 0)


def test_maximum_value_never_prefers_an_incomplete_answer_over_a_complete_one():
    """Completeness (unpacked_count) always leads, this objective included: leaving
    zero value behind never outranks leaving zero items behind."""
    from packvium.extensions import MaximumValueSolutionScorer

    complete = MaximumValueSolutionScorer().score_containers((), ())
    lost_worthless = Item.create("a", Dimensions.mm(1, 1, 1)).instances()  # value unset -> 0
    incomplete_but_worthless = MaximumValueSolutionScorer().score_containers(
        (), tuple(UnpackedItem(i, "no_feasible_placement") for i in lost_worthless)
    )
    assert complete < incomplete_but_worthless


def test_maximum_value_selects_the_higher_value_item_to_keep_regardless_of_input_order():
    only_room_for_one = container("c", 50, 50, 50, quantity=1)
    config = PackingConfig.balanced(objective="maximum_value")

    for items in (
        [item("cheap", 50, 50, 50, value=1), item("pricey", 50, 50, 50, value=100)],
        [item("pricey", 50, 50, 50, value=100), item("cheap", 50, 50, 50, value=1)],
    ):
        result = pack(items, [only_room_for_one], config)
        kept = {p.instance.item.id for c in result.containers for p in c.placements}
        assert kept == {"pricey"}


def test_maximum_value_selects_the_higher_value_item_under_a_single_start_profile():
    """The test above passes under `balanced` for the wrong reason: the
    ordering never read the objective, and the multi-start portfolio happened to contain
    a start that packed the valuable item, so the best-scoring result was right by luck.
    `fast` runs exactly one ordering, which is where the defect was visible -- identical
    dimensions made every ordering key tie and fall through to the item id, so `cheap`
    was packed and 500 of value was left behind."""
    only_room_for_one = container("c", 50, 50, 50, quantity=1)
    config = PackingConfig.fast(objective="maximum_value")

    for items in (
        [item("cheap", 50, 50, 50, value=1), item("precious", 50, 50, 50, value=500)],
        [item("precious", 50, 50, 50, value=500), item("cheap", 50, 50, 50, value=1)],
    ):
        result = pack(items, [only_room_for_one], config)
        kept = {p.instance.item.id for c in result.containers for p in c.placements}
        assert kept == {"precious"}
        assert result.score[1] == 1


def test_maximum_value_leaves_an_explicit_priority_bias_in_charge():
    """Value sits behind priority, not ahead of it: a caller who explicitly ranks the
    cheap item first still gets it packed, because priority is the caller's own say
    over the search and this key must not silently override it."""
    only_room_for_one = container("c", 50, 50, 50, quantity=1)
    result = pack(
        [item("cheap", 50, 50, 50, value=1, priority=5),
         item("precious", 50, 50, 50, value=500)],
        [only_room_for_one],
        PackingConfig.fast(objective="maximum_value"),
    )
    kept = {p.instance.item.id for c in result.containers for p in c.placements}
    assert kept == {"cheap"}


def test_a_request_that_omits_value_reproduces_the_default_result_byte_for_byte():
    """`value` is read only by the `maximum_value` objective; every other objective's
    result (score, containers, placements) must be completely unaffected by whether
    an item happens to declare one."""
    containers = [container("c", 120, 40, 40)]
    without_value = pack([item("a", 40, 40, 40, quantity=3)], containers, PackingConfig.balanced())
    with_value = pack([item("a", 40, 40, 40, quantity=3, value=7)], containers, PackingConfig.balanced())
    assert without_value.score == with_value.score
    assert len(without_value.containers) == len(with_value.containers)
    for left, right in zip(without_value.containers, with_value.containers):
        assert [(p.position, p.rotation) for p in left.placements] == [
            (p.position, p.rotation) for p in right.placements
        ]


def test_a_negative_value_fails_admission_instead_of_being_ignored():
    with pytest.raises(ValueError, match="value must be non-negative"):
        item("a", 10, 10, 10, value=-1)


def test_an_explicit_solution_scorer_overrides_the_named_objective():
    from packvium import Packer

    class ConstantScorer:
        def score(self, solution):
            return (0, 0, 0, 0, 99)

    items = [item("a", 40, 40, 40)]
    containers = [container("c", 100, 100, 100)]
    result = Packer(
        PackingConfig.balanced(objective="lowest_cost"), solution_scorer=ConstantScorer()
    ).pack(items, containers)
    assert tuple(result.score) == (0, 0, 0, 0, 99)


# --------------------------------------------------- open-dimension objective

def test_open_dimension_ranks_by_raw_achieved_height_ahead_of_container_count():
    """The default objective checks `container_count` before height, so one tall
    container of 300mm beats two shorter containers summing to 200mm even though the
    second answer is objectively less tall overall -- exactly backwards from what an
    "open dimension" caller ("how tall does this end up") wants.
    `open_dimension_height` reorders the vector so the raw summed achieved height
    decides before container count."""
    from packvium.extensions import OpenDimensionSolutionScorer

    tall_box = Container.create("tall", Dimensions.mm(100, 100, 1000))
    short_box = Container.create("short", Dimensions.mm(100, 100, 1000))
    stacked = Item.create("stacked", Dimensions.mm(100, 100, 300))
    a, b = Item.create("half", Dimensions.mm(100, 100, 100), quantity=2).instances()

    one_tall_container = (filled(tall_box, stacked.instances(), [(0, 0, 0)]),)
    two_short_containers = (
        filled(short_box, [a], [(0, 0, 0)]),
        PackedContainer(short_box, 2, filled(short_box, [b], [(0, 0, 0)]).placements),
    )

    candidates = [one_tall_container, two_short_containers]
    winner_by_default = min(candidates, key=lambda c: DefaultSolutionScorer.score_containers(c, ()))
    assert winner_by_default is one_tall_container  # fewer containers wins by default

    winner_by_open_dimension = min(
        candidates, key=lambda c: OpenDimensionSolutionScorer.score_containers(c, ())
    )
    assert winner_by_open_dimension is two_short_containers  # lower summed height wins here


def test_open_dimension_height_is_reported_and_selectable_end_to_end():
    items = [item("cube", 100, 100, 100, quantity=4)]
    containers = [container("pallet", 200, 200, 1_000)]  # generous, "open" height
    result = pack(items, containers, PackingConfig.balanced(objective="open_dimension_height"))
    assert result.objective == "open_dimension_height"
    assert len(result.score) == 5
    assert_sound(result, items, containers)


def test_open_dimension_height_matches_the_exact_solver_on_a_small_instance():
    """Cross-check against `exact_small`: for a small enough instance the heuristic
    profile must reach the same, provably minimal achieved height the exhaustive
    solver finds -- not merely a valid one."""
    from packvium.extensions import OpenDimensionSolutionScorer

    items = [item("cube", 100, 100, 100, quantity=4)]
    containers = [container("pallet", 200, 200, 1_000)]
    config_kwargs = dict(objective="open_dimension_height")

    heuristic = pack(items, containers, PackingConfig.balanced(**config_kwargs))
    exact = pack(items, containers, PackingConfig.exact_small(**config_kwargs))

    assert_sound(heuristic, items, containers)
    assert_sound(exact, items, containers)
    assert not heuristic.unpacked and not exact.unpacked
    heuristic_height = OpenDimensionSolutionScorer.score_containers(heuristic.containers, ())[1]
    exact_height = OpenDimensionSolutionScorer.score_containers(exact.containers, ())[1]
    # Four 100mm cubes fit in one 200x200 layer of two by two: the exact minimum
    # achievable height is exactly one cube's height, 100mm (1_600_000 ticks).
    assert exact_height == 1_600_000
    assert heuristic_height == exact_height


def test_exact_small_does_not_prune_a_heavier_promotional_rate_band():
    """A tariff may dip: two 100 g parcels cost 100, while a 100 g + 800 g pair
    costs 10. Exact search must retain the heavier equal-count branch rather than use
    volume/weight monotonicity the public rate-table contract does not promise."""
    from packvium.models import RateTable

    parcels = [
        Item.create("a-light", Dimensions.mm(100, 100, 100), weight="100g"),
        Item.create("b-light", Dimensions.mm(100, 100, 100), weight="100g"),
        Item.create("z-heavy", Dimensions.mm(100, 100, 100), weight="800g"),
    ]
    bin_type = Container.create(
        "bin",
        Dimensions.mm(200, 100, 100),
        rate_table=RateTable((200, 900), (100, 10)),
        quantity=1,
    )
    result = pack(parcels, [bin_type], PackingConfig.exact_small(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=10_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
        max_containers=1,
    ))
    assert result.score[:2] == (1, 10)
    assert "z-heavy#1" in {
        placement.instance.id
        for packed_container in result.containers
        for placement in packed_container.placements
    }


def test_a_missing_rate_table_is_refused_at_admission_even_when_unused():
    """A missing tariff is a static property of the request: rating some containers and
    not others would rank a priced packing against an unpriced one as though the
    unpriced were free. Rust and the JavaScript fallback already refused up front;
    Python enforced this only if the search happened to touch the untabled container,
    so the same request answered or aborted depending on search internals (
    review)."""
    from packvium.config import PackingConfig
    from packvium.extensions import UnknownObjectiveError
    from packvium.models import RateTable
    from packvium.packer import Packer

    tiny = Item.create("tiny", Dimensions.mm(10, 10, 10), weight="1g")
    tabled = Container.create(
        "tabled", Dimensions.mm(100, 100, 100), rate_table=RateTable((1_000,), (100,)),
    )
    untabled = Container.create("untabled", Dimensions.mm(500, 500, 500))
    packer = Packer(PackingConfig(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=5_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
    ))
    with pytest.raises(UnknownObjectiveError, match="requires a rate_table on every container; 'untabled'"):
        packer.pack([tiny], [tabled, untabled])


def test_a_missing_divisor_is_refused_at_admission_naming_the_right_objective():
    """The late scorer check inherits shipping_cost's sentence, so a landed-cost
    request used to run the whole search and then die blaming the wrong objective."""
    from packvium.config import PackingConfig
    from packvium.extensions import UnknownObjectiveError
    from packvium.models import RateTable
    from packvium.packer import Packer

    tiny = Item.create("tiny", Dimensions.mm(10, 10, 10), weight="1g")
    tabled = Container.create(
        "tabled", Dimensions.mm(100, 100, 100), rate_table=RateTable((1_000,), (100,)),
    )
    with pytest.raises(
        UnknownObjectiveError,
        match="the lowest_landed_cost objective requires configuration.dimensional_weight_divisor",
    ):
        Packer(PackingConfig(objective="lowest_landed_cost")).pack([tiny], [tabled])


def test_alternatives_never_quote_the_sentinel():
    """The refusal guarded only the winner; `alternatives` (top_k defaults to 3) could
    carry a feasible-status packing of an unpriceable container with the sentinel as
    its landed cost -- the exact number this objective exists to never invent. Runner-
    ups the tariff cannot price are dropped before the slice (review)."""
    from packvium.config import PackingConfig, SolverProfile
    from packvium.extensions import UNPRICEABLE_MINOR, unpriceable_container
    from packvium.models import RateTable
    from packvium.packer import Packer

    box = Item.create("box", Dimensions.mm(100, 100, 100), weight="500g", quantity=8)
    alpha = Container.create(
        "alpha_unpriceable", Dimensions.mm(300, 300, 300),
        rate_table=RateTable((2_000,), (900,)),
    )
    beta = Container.create(
        "beta_priceable", Dimensions.mm(400, 400, 400),
        rate_table=RateTable((20_000,), (1_500,)),
    )
    config = PackingConfig(
        objective="lowest_landed_cost",
        dimensional_weight_divisor=5_000,
        dimensional_weight_length_unit="cm",
        dimensional_weight_weight_unit="kg",
        profile=SolverProfile.QUALITY,
    )
    result = Packer(config).pack([box], [alpha, beta])
    assert result.score[1] == 1_500
    for alternative in result.alternatives:
        assert alternative.score[1] != UNPRICEABLE_MINOR
        assert unpriceable_container(alternative.containers, config) is None
