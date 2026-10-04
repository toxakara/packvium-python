# Packvium for Python

Deterministic 3D cartonization and rectangular bin packing. Pure Python, **no runtime
dependencies**, exact integer geometry.

Use it to pick the smallest carton for an order, build a pallet, load a shipping container,
or load a truck within its axle ratings and delivery-stop order. Every answer comes back as
coordinates and rotations for each item, with a reason for anything that did not fit.

Full documentation, the constraint reference and benchmarks live at
[packvium.com](https://packvium.com).

> **Version 1.5.0 — the public API is frozen.** Field names, status codes and the
> objective vector do not change without a major version, so any `1.x` is a safe upgrade
> from any earlier `1.x`.
> Read [docs/GUARANTEES.md](https://github.com/toxakara/packvium-python/blob/main/docs/GUARANTEES.md) before relying on a result.

```bash
pip install packvium
```

## Quick start

```python
from packvium import Container, Dimensions, Item, Packer, PackingConfig

result = Packer(PackingConfig.balanced()).pack(
    items=[Item.create("book", Dimensions.mm("210", "140", "30"), quantity=4)],
    containers=[Container.create("box", Dimensions.mm("400", "300", "250"))],
)

print(result.status.value)                # feasible
for container in result.containers:
    for placement in container.placements:
        print(placement.instance.id, placement.position, placement.rotation.value)
```

The same request as JSON, through the function every Packvium engine shares, or through the
command line, which reads a request on standard input:

```python
from packvium import pack_from_dict

result = pack_from_dict({
    "items": [{"id": "book", "quantity": 4, "dimensions": {"length": "210", "width": "140", "height": "30"}}],
    "containers": [{"id": "box", "inner_dimensions": {"length": "400", "width": "300", "height": "250"}}],
})
```

```bash
echo '{"items":[{"id":"box","quantity":8,"dimensions":{"length":"50","width":"50","height":"50"}}],
       "containers":[{"id":"carton","inner_dimensions":{"length":"100","width":"100","height":"100"}}]}' \
  | python -m packvium
```

## Units

Lengths are stored in ticks of 1/16000 mm and weights in ticks of 1/8 µg, as integers.
Inputs are integers or decimal strings with a unit (`mm`, `cm`, `m`, `in`, `ft`; `mg`, `g`,
`kg`, `oz`, `lb`), and fractional inches are exact, not approximated:

```python
Dimensions.inches("12 3/8", "8 1/2", "3/4")
Item.create("mug", Dimensions.mm("100", "100", "120"), "12 oz")
```

A JSON request refuses a float measure rather than rounding it: send `"160.5"`, not
`160.5`. See
[docs/UNITS-AND-NUMERICS.md](https://github.com/toxakara/packvium-python/blob/main/docs/UNITS-AND-NUMERICS.md).

## Getting the same answer every time

The search runs several starts and keeps the best. By default a wall clock decides when it
stops (`time_limit_ms`, one second), so a search the clock cuts short keeps whatever it had
reached, and a busy machine can reach less. The same request can then give a different
answer on a different run.

For an answer you can store, compare or replay, bound the search by counted work instead,
and keep the time limit only as a generous fuse:

```python
from packvium import EffortBudget, PackingConfig

config = PackingConfig.balanced(
    time_limit_ms=60_000,
    effort_budget=EffortBudget(max_candidates_evaluated=50_000),
)
# JSON: "configuration": {"time_limit_ms": 60000, "effort_budget": {"max_candidates_evaluated": 50000}}
```

The result says what stopped the search: `result.termination.code` is `complete`,
`effort_limit` (reproducible), or `time_limit` (not reproducible).
`packvium.artifacts.replay_level(result.to_dict())` gives the same verdict as `exact` or
`not_guaranteed`. `reproducibility.py` shows four simulated machines giving four
different answers under a clock, and one answer under a budget.

## Solver profiles

`solver_profile` (JSON) or the matching `PackingConfig` constructor picks how hard the search
tries:

| Profile | What it runs |
| --- | --- |
| `fast` | One ordering, one solver. No runners-up. |
| `balanced` (default) | Several solvers over several item orderings, keeping up to two runners-up. |
| `quality` | More solvers and orderings, a beam over container choices, and exact search for small orders. Slower. |
| `exact_small` | Bounded exact search when the order has at most `exact_item_limit` items (7, counting quantity); otherwise the same search as `balanced`. |

`PackingConfig.fast()`, `.balanced()`, `.quality()` and `.exact_small()` default to time
limits of 0.2, 1, 5 and 10 seconds; a JSON request defaults to 1 second whatever the
profile. An `effort_budget` makes any of them reproducible.

## Trucks, axles and delivery routes

A container can describe a vehicle, not only a box:

- `axles`: `[front, rear]`, each a position along the length and a `max_load`. Every
  placement is checked against both ratings using the gross load, tare included, and the
  result reports the exact `axle_reactions`.
- `access_directions`: the walls cargo leaves through, such as `["+x"]` for rear doors.
- `stop_index` on an item: the stop it is delivered at, `0` first. Nothing due later may
  sit on top of something due earlier, or, when `access_directions` is set, block its
  last way out.

After the solve, `safe_route_removal_order` gives an unloading order stop by stop,
`replay_loading_order` checks the reverse as a loading order, and `placement_reachability`
says what can be reached when the doors open. An item the axle ratings cannot carry is
reported as `no_feasible_placement`, not as an axle problem. See `trucking.py`.

## Errors

A request that no engine may answer raises `packvium.InvalidRequestError`, a `ValueError`,
before anything is solved. It names the problem instead of describing it:

```python
from packvium import InvalidRequestError, pack_from_dict

try:
    result = pack_from_dict(request)
except InvalidRequestError as error:
    error.code     # "invalid_request"
    error.reason   # "below_minimum"
    error.field    # "/items/0/quantity" -- a JSON Pointer into your request
    str(error)     # "invalid_request: /items/0/quantity: must be at least 1"
```

`reason` is one of `missing_field`, `wrong_type`, `below_minimum`, `above_maximum`,
`negative_measure`, `invalid_unit`, `duplicate_id`, `not_allowed` or `invalid_value`.
`FixedPlacementError` is a subclass, with code `invalid_fixed_placement` and reason
`malformed` or `cannot_hold`. The message is the same in every Packvium engine. Branch on
`reason` and `field`; show the message to a person.

An unknown `objective` or `access_directions` value in a request is refused with
reason `not_allowed` and the JSON Pointer of the bad value (direct domain model constructors
raise `UnknownObjectiveError` or `InvalidDirectionError`). A request that is
valid but does not fit completely is not an error: the result lists what was left out, and
why, in `unpacked_items`. `errors.py` turns all of these into HTTP responses.

## Examples

Runnable, in [`examples/`](https://github.com/toxakara/packvium-python/tree/main/examples),
and included in the source distribution. Each one is a single file you can read top to
bottom. Every one of them is executed by the test suite on each release, so none of them
can quietly stop working.

New here? Read `basic.py`, then `objectives.py` — between them they cover what most
callers need. `units.py` and `serialization.py` explain the two design choices that
surprise people. `extensions.py` is last on purpose: reach for it only after the fields
in `constraints.py` and `limits.py` have failed you.

| File | What it shows |
| --- | --- |
| [`basic.py`](https://github.com/toxakara/packvium-python/blob/main/examples/basic.py) | The smallest useful call: items in, placements out — and the three details in it that are easy to miss. |
| [`objectives.py`](https://github.com/toxakara/packvium-python/blob/main/examples/objectives.py) | All six objectives on scenes where they genuinely disagree, including the rate card that makes the heavier shipment the cheaper one. |
| [`constraints.py`](https://github.com/toxakara/packvium-python/blob/main/examples/constraints.py) | Upright-only, floor-only, non-stackable, top-load limits, and tags that keep two items out of the same box — plus how to read the reason an item was refused. |
| [`limits.py`](https://github.com/toxakara/packvium-python/blob/main/examples/limits.py) | A courier fleet: a van with a fridge unit and wheel arches as obstacles, an item cap per bike, one dangerous-goods item per van, frozen goods only in a refrigerated vehicle, and a cap on vehicles. |
| [`units.py`](https://github.com/toxakara/packvium-python/blob/main/examples/units.py) | Why there are no floats anywhere: fractional inches, exact ticks, and the one-tick difference between a fit and a refusal. |
| [`serialization.py`](https://github.com/toxakara/packvium-python/blob/main/examples/serialization.py) | The same request as JSON, the result in full, and exactly which mistakes are refused and which are silently ignored. |
| [`errors.py`](https://github.com/toxakara/packvium-python/blob/main/examples/errors.py) | A web handler: refused requests turned into a 422 with a pointer and a message a person can act on, fixed placements that cannot hold, and a partial fit that is not an error. |
| [`reproducibility.py`](https://github.com/toxakara/packvium-python/blob/main/examples/reproducibility.py) | Why a clock-limited search answers differently on a busy machine, how an effort budget fixes it, and how to read `termination`. |
| [`trucking.py`](https://github.com/toxakara/packvium-python/blob/main/examples/trucking.py) | A three-stop delivery van within its axle ratings: loading and unloading order, what the driver can reach at each stop, and the same goods packed without a route. |
| [`fixed_placements.py`](https://github.com/toxakara/packvium-python/blob/main/examples/fixed_placements.py) | A trailer that arrives partly loaded: new pallets packed around the ones already on board, and a record of what is on board that cannot be true. |
| [`rebalancing.py`](https://github.com/toxakara/packvium-python/blob/main/examples/rebalancing.py) | Two pallets of very different weight evened out after packing, one validated move at a time. |
| [`shapes.py`](https://github.com/toxakara/packvium-python/blob/main/examples/shapes.py) | Items that are not their box: complementary wedges sharing one crate as `convex_hull`, and a cushion that compresses under load until the crush limit refuses it. |
| [`nested.py`](https://github.com/toxakara/packvium-python/blob/main/examples/nested.py) | Units into cartons, cartons onto a pallet, in one call. |
| [`commerce.py`](https://github.com/toxakara/packvium-python/blob/main/examples/commerce.py) | Rate a shipment, apply an eligibility rule, and pin a catalog version. |
| [`execution.py`](https://github.com/toxakara/packvium-python/blob/main/examples/execution.py) | Turn a result into dock instructions: solver facts kept apart from screen text, a step order that is injected or honestly absent, and an operator lock that yields a second plan rather than editing the approved one. |
| [`artifacts.py`](https://github.com/toxakara/packvium-python/blob/main/examples/artifacts.py) | Hand a result to a system with no engine: one document with the plan, geometry and the request that produced it, exported as CSV and a printable work order, with an honest replay level. |
| [`revisions.py`](https://github.com/toxakara/packvium-python/blob/main/examples/revisions.py) | Replan a half-loaded job: a missing item and a locked placement recorded against the approved plan, a replan that keeps the locked item in place, and a hash-chained record that notices an edit. |
| [`intelligence.py`](https://github.com/toxakara/packvium-python/blob/main/examples/intelligence.py) | Prove a carton change is worth publishing: two scenarios compared order by order, a proposal that refuses to exist on thin evidence, and a replay against held-out history where a cheaper packing your validator rejects still counts as a regression. |
| [`extensions.py`](https://github.com/toxakara/packvium-python/blob/main/examples/extensions.py) | A rule the schema has no field for — and an honest account of what you give up by writing one. |

With `packvium` installed, run any of them from a copy of the `examples/` directory:

```bash
python3 examples/trucking.py
```

From a checkout of the repository without installing, point Python at the source tree
instead: `PYTHONPATH=src python3 examples/trucking.py`.

## What it does

- **Exact arithmetic.** Length is measured in ticks of 1/16000 mm and weight in 1/8 µg.
  No coordinate is ever a float, so no placement decision depends on rounding.
- **Real constraints.** Weight and payload limits, permitted rotations, keep-upright,
  floor-only, non-stackable, top-load limits, minimum support ratio, tag incompatibility,
  eligible container tags, per-container item and tag limits, clearance, obstacles, two-axle
  load limits and multi-stop route order.
- **A solver portfolio, not one algorithm.** Regular-grid, layer, extreme-point,
  maximal-space and bounded exact search, selected by problem shape and profile.
- **Answers you can check.** Every solution is re-validated by logic independent of the
  search. Unplaced items come back with a reason code, not silently missing.
- **Deterministic when you ask for it.** The same request with an `effort_budget` gives the
  same result on every run and machine. A search stopped by `time_limit_ms` keeps what it
  reached, which can differ between runs; `termination` says which happened.
- **Multi-container and nested.** Split across containers, or pack containers into
  containers. `rebalance_weight` evens out the payload across containers afterwards.
- **Extensible.** Register your own constraints, item orderings, candidate scorers,
  container selectors or complete solvers.
- **Work orders, not just coordinates.** `packvium.execution` turns a validated result
  into a plan: solver facts kept apart from the text that cites them, an injected loading
  order or an honest `unavailable`, and a canonical form four engines emit byte for byte.
  `packvium.locks` lets an operator pin a placement and get a *second* plan beside the
  approved one — never an edit of it, and never a placement the engine would refuse.
- **Artifacts a warehouse system can use without an engine.** `packvium.artifacts` wraps the
  plan with geometry, display values and provenance, including the request itself and an
  honest replay level. `packvium.artifact_exports` writes it as RFC 8785 JSON, CSV or a
  self-contained HTML work order, byte for byte what the other engines write.
- **Items already in place, and replanning around them.** `fixed_placements` pins items to
  known positions before the solve: they keep their place, carry weight and support, and come
  back marked `fixed: true`. `packvium.revisions` records what changed on the dock — a missing
  item, a substituted container, a lock, a verification — as an append-only chain linked by
  SHA-256, and derives the request the next plan solves. `packvium.revision_outcomes` feeds
  those events to the historical evaluation (Python only).
- **Decide before you publish.** `packvium.simulation` and `packvium.recommendations`
  compare two catalog scenarios order by order and propose a change only when the paired
  cohort supports it; `packvium.holdout` replays that proposal against history it has not
  seen. Python-only, and supported but not yet signature-frozen — see
  [PUBLIC-API.md](https://github.com/toxakara/packvium-python/blob/main/docs/PUBLIC-API.md).

## Documentation

| Document | Covers |
| --- | --- |
| [docs/GUARANTEES.md](https://github.com/toxakara/packvium-python/blob/main/docs/GUARANTEES.md) | What is promised and what is not. Start here. |
| [docs/PUBLIC-API.md](https://github.com/toxakara/packvium-python/blob/main/docs/PUBLIC-API.md) | Inputs, outputs and status semantics. |
| [docs/COMMERCE-API.md](https://github.com/toxakara/packvium-python/blob/main/docs/COMMERCE-API.md) | Carrier rating, eligibility rules and catalog versions. |
| [docs/UNITS-AND-NUMERICS.md](https://github.com/toxakara/packvium-python/blob/main/docs/UNITS-AND-NUMERICS.md) | Units, accepted input forms, rounding policy. |

## Requirements

Python 3.9 or newer. No dependencies.

## The Packvium family

One request and result contract, implemented independently in four engines (Rust,
Python, PHP, JavaScript) and held to identical placements on a shared fixture set.
Pick the package for your stack; mixing them in one system is safe.

Documentation, the constraint reference and the benchmarks are at
[packvium.com](https://packvium.com).

| Package | Install | Source |
| --- | --- | --- |
| Python — [`packvium`](https://pypi.org/project/packvium/) | `pip install packvium` | [packvium-python](https://github.com/toxakara/packvium-python) |
| PHP — [`packvium/packvium`](https://packagist.org/packages/packvium/packvium) | `composer require packvium/packvium` | [packvium-php](https://github.com/toxakara/packvium-php) |
| Rust — [`packvium`](https://crates.io/crates/packvium) | `packvium = "1.0"` | [packvium-rust](https://github.com/toxakara/packvium-rust) |
| Node.js — [`@packvium/engine`](https://www.npmjs.com/package/@packvium/engine) | `npm install @packvium/engine` | [packvium-node](https://github.com/toxakara/packvium-node) |
| Browser / WebAssembly — [`@packvium/browser`](https://www.npmjs.com/package/@packvium/browser) | `npm install @packvium/browser` | [packvium-wasm](https://github.com/toxakara/packvium-wasm) |
| PHP FFI bridge — [`packvium/native-bridge`](https://packagist.org/packages/packvium/native-bridge) | `composer require packvium/native-bridge` | [packvium-php-bridge](https://github.com/toxakara/packvium-php-bridge) |
| Python native selector — `packvium-native` | from source until the native wheels ship | [packvium-python-adapter](https://github.com/toxakara/packvium-python-adapter) |

## Contributing

See [CONTRIBUTING.md](https://github.com/toxakara/packvium-python/blob/main/CONTRIBUTING.md). Security reports go through the process in
[SECURITY.md](https://github.com/toxakara/packvium-python/blob/main/SECURITY.md), not public issues.

## Citation

If Packvium supports your research, cite it as software. GitHub's **Cite this repository**
button reads [`CITATION.cff`](https://github.com/toxakara/packvium-python/blob/main/CITATION.cff), and
[`codemeta.json`](https://github.com/toxakara/packvium-python/blob/main/codemeta.json) carries the same record in
CodeMeta form.

```bibtex
@software{packvium_python,
  author  = {{Packvium contributors}},
  title   = {Packvium for Python},
  version = {1.5.0},
  license = {MIT},
  url     = {https://packvium.com}
}
```

## License

MIT. See [LICENSE](https://github.com/toxakara/packvium-python/blob/main/LICENSE).
