"""Determinism under real concurrent CPU load, not simulated jitter.

`test_invariants.py`/`test_packing.py` already prove reproducibility under an effort
budget using the real clock (multiple runs of one hand-built scenario) and a fake
ticking clock across many generated orders (the injected-clock contribution) -- both fast,
deterministic and safe to run on every commit. Neither puts the machine under the kind
of contention this targets: "concurrent compilation or CI load
changes how far QUALITY search progresses". This script closes that gap directly by
spawning real OS processes that burn CPU for the duration of the run, then repeating a
fixed request 100 times against the real wall clock while they compete for cores.

Kept out of the default test run on purpose, for the same reason
`benchmarks/run_baselines.py` is: it depends on real OS scheduling under real
concurrent load, not only this repository's own state, so it cannot be as fast or as
guaranteed-quiet as the deterministic suite. Run by hand with `make stress-determinism`.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import container, item, pack  # noqa: E402

from packvium import EffortBudget, IndependentSolutionValidator, PackingConfig, PackingRequest  # noqa: E402

RUNS = 100
LOAD_WORKERS = 4


def _burn_cpu(stop_at: float) -> None:
    total = 0
    while time.monotonic() < stop_at:
        total += sum(i * i for i in range(2_000))
    # Touch `total` so the loop body is not optimised away by anything watching the
    # process; the value itself is meaningless.
    if total < 0:
        raise AssertionError("unreachable")


def _scenario():
    items = [
        item("a", 40, 30, 20, quantity=6),
        item("b", 60, 50, 40, quantity=4),
        item("c", 25, 25, 25, quantity=8),
    ]
    containers = [container("box", 150, 150, 150, quantity=3)]
    return items, containers


EFFORT_FIELDS = ("duration_ms", "candidates_evaluated", "placements_attempted", "metrics")


def _chosen_answer(report: dict) -> dict:
    for field in EFFORT_FIELDS:
        report["algorithm"].pop(field, None)
    report.pop("alternatives", None)
    return report


def _is_sound(items, containers, result) -> bool:
    issues = IndependentSolutionValidator().validate(
        PackingRequest(tuple(items), tuple(containers)), tuple(result.containers)
    ).issues
    return not issues


def main() -> int:
    load_deadline = time.monotonic() + 30.0
    workers = [multiprocessing.Process(target=_burn_cpu, args=(load_deadline,)) for _ in range(LOAD_WORKERS)]
    for worker in workers:
        worker.start()
    try:
        items, containers = _scenario()

        # The actual acceptance: bounded by counted work alone, real clock, real
        # concurrent load -- every run must land on the exact same answer.
        effort_config = PackingConfig.balanced(
            time_limit_ms=60_000, seed=1234,
            effort_budget=EffortBudget(max_search_nodes=200, max_restarts=4),
        )
        answers = [_chosen_answer(pack(items, containers, effort_config).to_dict()) for _ in range(RUNS)]
        mismatches = sum(1 for answer in answers if answer != answers[0])
        if mismatches:
            print(f"FAIL: {mismatches}/{RUNS} effort-budget runs disagreed with the first under concurrent load",
                  file=sys.stderr)
            return 1
        print(f"OK: {RUNS}/{RUNS} effort-budget runs were bit-identical under {LOAD_WORKERS}-process concurrent load")

        # The contrast the acceptance also names: a truncated wall-clock-only run makes
        # no bit-identical claim. Soundness is still checked -- that promise never
        # lapses -- but equality across runs is deliberately not asserted here.
        tight_wall_clock = PackingConfig.balanced(time_limit_ms=5, seed=1234)
        unsound = 0
        wall_clock_answers = set()
        for _ in range(RUNS):
            result = pack(items, containers, tight_wall_clock)
            if not _is_sound(items, containers, result):
                unsound += 1
            wall_clock_answers.add(repr(_chosen_answer(result.to_dict())))
        if unsound:
            print(f"FAIL: {unsound}/{RUNS} wall-clock-only runs under load were unsound", file=sys.stderr)
            return 1
        print(f"OK: {RUNS}/{RUNS} wall-clock-only runs under load stayed sound "
              f"({len(wall_clock_answers)} distinct answer(s) seen -- no equality claimed or required)")
        return 0
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.join()


if __name__ == "__main__":
    raise SystemExit(main())
