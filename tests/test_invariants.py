"""Randomised instances checked against the properties that must hold for all of them.

Hand-written scenarios test the cases somebody thought of. This suite generates
orders from a fixed seed and asserts the guarantees the library owes every caller,
which is how the awkward interactions — a group of non-stackable items under a
payload ceiling — get exercised without anyone having to imagine them first.

The seed is fixed on purpose: a failure here must be reproducible from the test name
alone, not only on the machine that first saw it.
"""

from __future__ import annotations

import random

import pytest
import packvium.solvers as solvers_module

from packvium import (Dimensions, EffortBudget, Length, PackingConfig, PackingStatus,
                         pack_from_dict)
from packvium.extensions import DefaultSolutionScorer
from support import assert_sound, container, item, pack

TRIALS = 40


def generate(seed: int):
    """One plausible order: a handful of item types and one or two container types."""
    rng = random.Random(seed)
    items = []
    for index in range(rng.randint(1, 5)):
        length, width, height = (rng.randrange(10, 70, 5) for _ in range(3))
        items.append(item(
            f"i{index}", length, width, height,
            quantity=rng.randint(1, 4),
            weight=f"{rng.randrange(50, 900)} g",
            stackable=rng.random() > 0.2,
            must_be_on_floor=rng.random() > 0.85,
            keep_upright=rng.random() > 0.8,
            minimum_support_ratio=rng.choice([0.0, 0.0, 0.5, 0.75]),
            max_top_load=None if rng.random() > 0.25 else f"{rng.randrange(500, 4000)} g",
            group=None if rng.random() > 0.2 else f"g{index % 2}",
        ))
    containers = []
    for index in range(rng.randint(1, 2)):
        length, width, height = (rng.randrange(80, 220, 10) for _ in range(3))
        containers.append(container(
            f"c{index}", length, width, height,
            quantity=rng.randint(1, 3),
            cost_minor=rng.randrange(0, 900),
            max_payload=None if rng.random() > 0.3 else f"{rng.randrange(2, 12)} kg",
        ))
    return items, containers


def instance_ids(items) -> set[str]:
    return {instance.id for one in items for instance in one.instances()}


#: Fields describing how much work the search did rather than what it decided. The
#: time budget is wall clock, so how far a start gets before its slice runs out is
#: machine- and load-dependent. The answer is reproducible; the effort spent reaching
#: it is not, and asserting otherwise would make this suite flaky by design.
EFFORT_FIELDS = ("duration_ms", "candidates_evaluated", "placements_attempted", "metrics")


def chosen_answer(report: dict) -> dict:
    """The answer the caller receives, with the effort diagnostics removed.

    `alternatives` is dropped as well. A start that was cut off mid-search reports the
    placements it had managed by then, and how far it got is a wall-clock question — so
    a truncated runner-up may legitimately differ between two runs of the same request
    even though the winning packing does not.
    """
    for field in EFFORT_FIELDS:
        report["algorithm"].pop(field, None)
    report.pop("alternatives", None)
    return report


@pytest.mark.parametrize("seed", range(TRIALS))
def test_every_generated_order_is_answered_soundly(seed):
    items, containers = generate(seed)
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250))

    assert_sound(result, items, containers)
    assert result.complete == (result.unpacked == ())
    assert tuple(result.score) == DefaultSolutionScorer.score_containers(result.containers, result.unpacked)


@pytest.mark.parametrize("seed", range(TRIALS))
def test_no_item_is_lost_or_duplicated(seed):
    """The strongest bookkeeping property there is: every instance the caller asked for
    comes back exactly once, either placed or explained."""
    items, containers = generate(seed)
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250))

    placed = [p.instance.id for c in result.containers for p in c.placements]
    unplaced = [u.instance.id for u in result.unpacked]
    assert len(placed) + len(unplaced) == len(placed + unplaced)
    assert sorted(placed + unplaced) == sorted(instance_ids(items))
    assert len(set(placed)) == len(placed)


@pytest.mark.parametrize("seed", range(TRIALS))
def test_container_stock_is_never_overdrawn(seed):
    items, containers = generate(seed)
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250))

    used: dict[str, int] = {}
    for packed in result.containers:
        used[packed.container.id] = used.get(packed.container.id, 0) + 1
    for one in containers:
        if one.quantity is not None:
            assert used.get(one.id, 0) <= one.quantity


@pytest.mark.parametrize("seed", range(TRIALS))
def test_every_unpacked_item_carries_a_recognised_reason(seed):
    """A caller acts on these strings, so an unexplained failure is a failure."""
    known = {"no_compatible_container_dimensions", "payload_exceeded", "no_feasible_placement",
             "group_cannot_fit_together", "time_limit", "insufficient_support"}
    items, containers = generate(seed)
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250))
    assert {u.reason for u in result.unpacked} <= known


#: These reproducibility checks exercise the entire portfolio with a frozen monotonic
#: clock. Deadline correctness has separate check-budget tests; using real time here
#: would make scheduler load part of the expected answer.
FROZEN_CLOCK = lambda: 0


@pytest.mark.parametrize("seed", range(12))
def test_repeating_a_generated_order_reproduces_the_same_answer(seed):
    """Same request, same seed, adequate budget — same answer, every time."""
    items, containers = generate(seed)
    config = PackingConfig.balanced(time_limit_ms=1, seed=1234)
    assert (chosen_answer(pack(items, containers, config, clock=FROZEN_CLOCK).to_dict())
            == chosen_answer(pack(items, containers, config, clock=FROZEN_CLOCK).to_dict()))


@pytest.mark.parametrize("seed", range(12))
def test_a_budget_large_enough_to_finish_reproduces_the_alternatives_too(seed):
    """With deterministic time, the whole report — runners-up included — is reproducible."""
    items, containers = generate(seed)
    config = PackingConfig.balanced(time_limit_ms=1, seed=1234)
    once, twice = (pack(items, containers, config, clock=FROZEN_CLOCK).to_dict() for _ in range(2))

    assert ([chosen_answer(report) for report in (once, *once["alternatives"])]
            == [chosen_answer(report) for report in (twice, *twice["alternatives"])])


@pytest.mark.parametrize("seed", range(12))
def test_a_biting_deadline_promises_soundness_but_not_reproducibility(seed):
    """What a tight budget still owes the caller.

    Measured, not assumed: at 250 ms the chosen packing differed between two runs of
    one request on 1 of 12 generated orders, and at 500 ms on 1 of 12. Equality is
    therefore not a property of a budget that bites. Validity, deadline reporting and
    item accounting still are, and those are what a caller may rely on at any budget.
    """
    items, containers = generate(seed)
    config = PackingConfig.balanced(time_limit_ms=50, seed=1234)
    for _ in range(2):
        result = pack(items, containers, config)
        assert_sound(result, items, containers)


def _ticking_clock(ns_per_call: int):
    """A deterministic stand-in for `monotonic_ns`: advances by a fixed amount on every
    call rather than by real elapsed time. Two instances with different `ns_per_call`
    simulate "two machines under different load" without the test itself depending on
    real wall-clock timing -- exactly the injected-clock discipline the rest of the suite
    already established for this suite, applied to a search that actually runs to
    completion or to a budget rather than to a frozen instant."""
    state = [0]

    def clock() -> int:
        state[0] += ns_per_call
        return state[0]

    return clock


# A wall clock is not a reproducible measure of work -- the search does
# genuinely different amounts of it between two runs depending on scheduling, CPU
# frequency and load, and `test_a_biting_deadline_promises_soundness_but_not_
# reproducibility` above measures the result: a wall-clock budget that actually bites
# is sound but not bit-identical. An `EffortBudget` counts work instead, so a caller
# who needs reproducibility even when the search is genuinely cut short can buy it.

@pytest.mark.parametrize("seed", range(12))
def test_a_biting_effort_budget_is_bit_identical_regardless_of_simulated_clock_speed(seed):
    """The scenario `test_a_biting_deadline_promises_soundness_but_not_reproducibility`
    shows is not reproducible under a wall clock alone -- promoted to exact here by
    counting work instead. The wall-clock ceiling stays generous enough that only the
    effort budget can plausibly trip first; the two ticking clocks (1 ns and 1 ms per
    call) stand in for very different machine speeds."""
    items, containers = generate(seed)
    effort = EffortBudget(max_candidates_evaluated=40, max_placement_attempts=40, max_search_nodes=20, max_restarts=2)
    config = PackingConfig.balanced(time_limit_ms=600_000, seed=1234, effort_budget=effort)

    fast_machine = chosen_answer(pack(items, containers, config, clock=_ticking_clock(1)).to_dict())
    slow_machine = chosen_answer(pack(items, containers, config, clock=_ticking_clock(1_000_000)).to_dict())
    assert fast_machine == slow_machine


# `SolverOrchestrator` can run every start after the first one concurrently,
# in separate worker processes (`PackingConfig.parallel_starts > 1`), instead of one
# after another sharing a single sliced deadline. The two tests below are the actual
# proof the ticket asks for -- not an injected clock standing in for scheduling, but
# real `ProcessPoolExecutor` workers on the real system clock, repeated enough times
# that genuine OS scheduling jitter (process start order, which worker the kernel runs
# first, background load on the test machine) gets a real chance to vary which start
# happens to finish first. Every repetition must still choose the same packing.
CONCURRENT_REPEATS = 8


@pytest.mark.parametrize("seed", range(4))
def test_concurrent_starts_reproduce_the_sequential_answer_under_an_ample_budget(seed):
    """Given a budget large enough to finish, the documented promise is that the whole
    report is reproducible (docs/ALGORITHMS-AND-COMPLEXITY.md, "What deterministic
    covers"). This asserts that promise still holds when the starts making up that
    report actually run at the same time in different processes: the chosen packing
    must match plain sequential execution (`parallel_starts=1`) and must not move
    across repeated real-process runs."""
    items, containers = generate(seed)
    sequential = PackingConfig.balanced(
        time_limit_ms=10_000, seed=1234, multi_start_orders=6,
        solvers=("extreme_points", "layer"), parallel_starts=1,
    )
    concurrent = PackingConfig.balanced(
        time_limit_ms=10_000, seed=1234, multi_start_orders=6,
        solvers=("extreme_points", "layer"), parallel_starts=4,
    )

    baseline = chosen_answer(pack(items, containers, sequential).to_dict())
    for _ in range(CONCURRENT_REPEATS):
        answer = chosen_answer(pack(items, containers, concurrent).to_dict())
        assert answer == baseline


def test_concurrent_starts_are_bit_identical_under_a_biting_effort_budget():
    """The counted `EffortBudget` is what turns reproducibility from a probabilistic
    promise into an exact one under a budget that bites -- this repeats that
    proof under real concurrent worker processes rather than sequential execution with
    an injected clock, since the concurrent path explicitly must not widen that determinism
    boundary. Each worker's own effort counters are private to its own process, so
    which one the OS happens to schedule first can never change what any individual
    start decides, and the parent's ranking is a pure, order-independent function of
    the finished set (see `Packer.pack`'s full-key sort)."""
    items, containers = generate(7)
    effort = EffortBudget(max_candidates_evaluated=40, max_placement_attempts=40,
                          max_search_nodes=20, max_restarts=4)
    config = PackingConfig.balanced(
        time_limit_ms=10_000, seed=1234, effort_budget=effort, multi_start_orders=4,
        solvers=("extreme_points", "layer"), parallel_starts=4,
    )

    baseline = chosen_answer(pack(items, containers, config).to_dict())
    for _ in range(CONCURRENT_REPEATS):
        assert chosen_answer(pack(items, containers, config).to_dict()) == baseline


def test_parallel_starts_falls_back_to_sequential_with_an_injected_clock():
    """An injected test clock has no meaning across a process boundary (see
    `Deadline.uses_real_clock`), so `parallel_starts > 1` must silently fall back to
    the exact sequential path rather than trying -- and failing -- to hand a closure to
    a worker process. This is the gate itself, not just its effect: the call must
    succeed and match plain sequential execution exactly."""
    items, containers = generate(2)
    sequential = PackingConfig.balanced(time_limit_ms=1, seed=1234, parallel_starts=1)
    concurrent = PackingConfig.balanced(time_limit_ms=1, seed=1234, parallel_starts=8)

    baseline = chosen_answer(pack(items, containers, sequential, clock=FROZEN_CLOCK).to_dict())
    answer = chosen_answer(pack(items, containers, concurrent, clock=FROZEN_CLOCK).to_dict())
    assert answer == baseline


def test_parallel_starts_falls_back_when_the_host_denies_process_pool_access(monkeypatch):
    """A restricted host may reject the semaphore capability check performed by
    `ProcessPoolExecutor` before any worker starts. Parallel execution is only an
    optimisation, so that platform failure must preserve the sequential answer
    instead of escaping through the public packing API."""
    items, containers = generate(2)
    sequential = PackingConfig.balanced(
        time_limit_ms=10_000, seed=1234, multi_start_orders=4,
        solvers=("extreme_points", "layer"), parallel_starts=1,
    )
    concurrent = PackingConfig.balanced(
        time_limit_ms=10_000, seed=1234, multi_start_orders=4,
        solvers=("extreme_points", "layer"), parallel_starts=4,
    )
    baseline = chosen_answer(pack(items, containers, sequential).to_dict())

    def denied_process_pool(*, max_workers):
        raise PermissionError("process semaphores are unavailable")

    monkeypatch.setattr(solvers_module, "ProcessPoolExecutor", denied_process_pool)

    assert chosen_answer(pack(items, containers, concurrent).to_dict()) == baseline


def test_concurrent_portfolio_is_exercised_without_host_semaphores(monkeypatch):
    """The process-pool branch must remain testable on restricted CI hosts.

    Real-process tests above provide the scheduling proof when the host permits worker
    processes.  This deterministic executor exercises the parent's submission,
    position-preserving collection and ranking logic even when the host denies the
    semaphore probe, so a coverage run cannot silently skip that whole code path.
    """
    worker_counts = []

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class InlineProcessPool:
        def __init__(self, *, max_workers):
            worker_counts.append(max_workers)

        def shutdown(self, *, wait):
            assert wait is True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, function, *args):
            return ImmediateFuture(function(*args))

    items, containers = generate(2)
    sequential = PackingConfig.balanced(
        time_limit_ms=10_000, seed=1234, multi_start_orders=4,
        solvers=("extreme_points", "layer"), parallel_starts=1,
    )
    concurrent = PackingConfig.balanced(
        time_limit_ms=10_000, seed=1234, multi_start_orders=4,
        solvers=("extreme_points", "layer"), parallel_starts=4,
    )
    baseline = chosen_answer(pack(items, containers, sequential).to_dict())

    monkeypatch.setattr(solvers_module, "ProcessPoolExecutor", InlineProcessPool)

    assert chosen_answer(pack(items, containers, concurrent).to_dict()) == baseline
    assert worker_counts[0] == 1  # capability probe
    assert 1 < worker_counts[1] <= concurrent.parallel_starts


def test_the_effort_budget_above_actually_bites_on_at_least_one_generated_order():
    """Guards the reproducibility test above against being vacuously true because a
    budget that never actually constrains the search would of course be reproducible.
    Measured, not assumed -- matching the discipline of the wall-clock test it answers."""
    effort = EffortBudget(max_candidates_evaluated=40, max_placement_attempts=40, max_search_nodes=20, max_restarts=2)
    config = PackingConfig.balanced(time_limit_ms=600_000, seed=1234, effort_budget=effort)
    bit_at_least_once = False
    for seed in range(12):
        items, containers = generate(seed)
        result = pack(items, containers, config, clock=_ticking_clock(1))
        bit_at_least_once = bit_at_least_once or result.algorithm.effort_limit_reached
    assert bit_at_least_once


def test_the_wall_clock_still_stops_a_search_when_the_effort_budget_is_looser():
    """An effort budget is an additional bound, not a replacement for the wall clock:
    with the simulated clock advancing far faster than a tight time budget and an
    effort budget generous enough never to bite first, the wall clock is still what
    stops the search -- the safety net kept alongside the new
    counted budgets, expressible together in one config."""
    items, containers = generate(0)
    loose_effort = EffortBudget(max_candidates_evaluated=1_000_000, max_placement_attempts=1_000_000,
                                 max_search_nodes=1_000_000, max_restarts=1_000)
    config = PackingConfig.balanced(time_limit_ms=1, seed=1234, effort_budget=loose_effort)
    result = pack(items, containers, config, clock=_ticking_clock(10_000_000))

    assert_sound(result, items, containers)
    assert result.algorithm.time_limit_reached


# A longer time budget must never make the packer choose a solution the
# active objective ranks worse than what a shorter budget, same request and same seed,
# would have chosen. `_ticking_clock` stands in for a slow, heavily-loaded machine (the
# same technique the effort-budget tests above use) so a tiny, fast-to-run fixture can
# still exercise genuine truncation at millisecond-scale budgets instead of needing a
# real BR1-sized corpus and second-scale deadlines -- and a slow simulated machine is
# exactly the regime (bookkeeping overhead competing against search time) where the
# violation this guards against was actually found.
def test_raising_the_time_limit_never_lowers_the_chosen_rank():
    """Hand-built fixture: seven differently-sized items in a container that cannot
    hold all of them, so `exact_small`'s branch-and-bound search genuinely has several
    distinct, differently-scored candidates to choose between at different budgets
    rather than converging on one obvious answer immediately."""
    items = [
        item("i0", 85, 50, 80, weight="265 g"),
        item("i1", 20, 40, 60, weight="298 g"),
        item("i2", 50, 80, 85, weight="205 g"),
        item("i3", 55, 45, 65, weight="161 g"),
        item("i4", 60, 30, 40, weight="121 g"),
        item("i5", 80, 25, 65, weight="459 g"),
        item("i6", 40, 60, 75, weight="464 g"),
    ]
    containers = [container("box", 240, 170, 190, quantity=1)]
    scorer = DefaultSolutionScorer()
    time_limits_ms = (1, 2, 3, 5, 8, 12, 18, 25, 35)

    scores = []
    for time_limit_ms in time_limits_ms:
        config = PackingConfig.exact_small(time_limit_ms=time_limit_ms, seed=103561)
        result = pack(items, containers, config, clock=_ticking_clock(50_000))
        scores.append(scorer.score_containers(result.containers, result.unpacked))
        if time_limit_ms == time_limits_ms[0]:
            # Guards against vacuous success: the smallest budget must actually be
            # too tight to finish, or every budget trivially finding the same
            # complete answer would pass this test without exercising anything.
            assert result.algorithm.time_limit_reached

    assert len(set(scores)) > 1, "fixture never produced distinct candidates across budgets"
    for previous, current, time_limit_ms in zip(scores, scores[1:], time_limits_ms[1:]):
        assert current <= previous, (
            f"raising the time limit to {time_limit_ms}ms chose a worse-ranked "
            f"solution ({current}) than a shorter budget had already found ({previous})"
        )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("profile", [PackingConfig.fast, PackingConfig.quality, PackingConfig.exact_small],
                         ids=["fast", "quality", "exact_small"])
def test_every_profile_answers_soundly(seed, profile):
    items, containers = generate(seed)
    result = pack(items, containers, profile(time_limit_ms=300))

    assert_sound(result, items, containers)
    assert result.status is not PackingStatus.INVALID_RESULT


@pytest.mark.parametrize("seed", range(12))
def test_a_container_budget_is_never_exceeded(seed):
    items, containers = generate(seed)
    for budget in (1, 2):
        result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250, max_containers=budget))
        assert len(result.containers) <= budget


@pytest.mark.parametrize("seed", range(12))
def test_clearance_is_honoured_on_generated_orders(seed):
    """A gap that the solver applies but the validator does not know about would go
    unnoticed on hand-written cases where everything fits comfortably."""
    gap = Length.mm(1)
    items, containers = generate(seed)
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=250, clearance=gap))

    assert_sound(result, items, containers, clearance=gap)
    for placement in (p for c in result.containers for p in c.placements):
        assert placement.envelope_dimensions == placement.dimensions.expand(gap)


@pytest.mark.parametrize("seed", range(12))
def test_a_tight_budget_never_produces_an_unsound_answer(seed):
    """Partial work must still be a valid packing; an escaping deadline used to discard
    it entirely."""
    items, containers = generate(seed)
    result = pack(items, containers, PackingConfig.balanced(time_limit_ms=1))
    assert_sound(result, items, containers)


@pytest.mark.parametrize("seed", range(8))
def test_the_dictionary_api_agrees_with_the_object_api(seed):
    """Two entry points, one library: the wire contract must not be a second algorithm.

    Both calls must be governed by counted effort, not a bare wall clock: under load,
    the object-API call and the dict-API call can each get a different distance into
    the same anytime search within an identical millisecond window and legitimately
    settle on different tied answers."""
    items, containers = generate(seed)
    config = PackingConfig.balanced(
        time_limit_ms=60_000,
        effort_budget=EffortBudget(max_search_nodes=500, max_restarts=9),
    )
    direct = pack(items, containers, config).to_dict()
    through_dict = pack_from_dict({
        "units": {"length": "mm"},
        "items": [serialise_item(one) for one in items],
        "containers": [serialise_container(one) for one in containers],
        "configuration": {
            "solver_profile": "balanced",
            "time_limit_ms": 60_000,
            "effort_budget": {"max_search_nodes": 500, "max_restarts": 9},
        },
    })

    assert through_dict["score"] == direct["score"]
    assert through_dict["summary"] == direct["summary"]


def measurement(dimensions: Dimensions) -> dict:
    return {axis: str(getattr(dimensions, axis).ticks) + " ticks" for axis in ("length", "width", "height")}


def serialise_item(one) -> dict:
    payload = {
        "id": one.id, "dimensions": measurement(one.dimensions), "quantity": one.quantity,
        "weight": f"{one.weight.ticks} ticks", "stackable": one.stackable,
        "must_be_on_floor": one.must_be_on_floor,
        "minimum_support_ratio": one.minimum_support_ratio,
        "allowed_rotations": [r.value for r in one.allowed_rotations],
    }
    if one.max_top_load is not None:
        payload["max_top_load"] = f"{one.max_top_load.ticks} ticks"
    if one.group is not None:
        payload["group"] = one.group
    return payload


def serialise_container(one) -> dict:
    payload = {"id": one.id, "inner_dimensions": measurement(one.inner_dimensions),
               "cost_minor": one.cost_minor, "quantity": one.quantity}
    if one.max_payload is not None:
        payload["max_payload"] = f"{one.max_payload.ticks} ticks"
    return payload
