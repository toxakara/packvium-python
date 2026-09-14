# Packvium for Python

Deterministic 3D cartonization and rectangular bin packing. Pure Python, **no runtime
dependencies**, exact integer geometry.

Full documentation, the constraint reference and benchmarks live at
[packvium.com](https://packvium.com).

> **Version 1.2.0 — the public API is frozen.** Field names, status codes and the
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

print(result.status)                      # feasible
for container in result.containers:
    for placement in container.placements:
        print(placement.item_id, placement.position, placement.orientation)
```

Fractional inches are exact, not approximated:

```python
Dimensions.inches("12 3/8", "8 1/2", "3/4")
```

There is also a CLI that reads a JSON request on standard input:

```bash
echo '{"items":[{"id":"box","quantity":8,"dimensions":{"length":"50","width":"50","height":"50"}}],
       "containers":[{"id":"carton","inner_dimensions":{"length":"100","width":"100","height":"100"}}]}' \
  | python -m packvium
```

## Examples

Runnable, in [`examples/`](https://github.com/toxakara/packvium-python/tree/main/examples). Each one is a single file you can read top to bottom
and execute without a project around it. Every one of them is executed by the test suite
on each release, so none of them can quietly stop working.

New here? Read `basic.py`, then `objectives.py` — between them they cover what most
callers need. `units.py` and `serialization.py` explain the two design choices that
surprise people. `extensions.py` is last on purpose: reach for it only after the fields
in `constraints.py` have failed you.

| File | What it shows |
| --- | --- |
| [`basic.py`](https://github.com/toxakara/packvium-python/blob/main/examples/basic.py) | The smallest useful call: items in, placements out — and the three details in it that are easy to miss. |
| [`objectives.py`](https://github.com/toxakara/packvium-python/blob/main/examples/objectives.py) | All six objectives on scenes where they genuinely disagree, including the rate card that makes the heavier shipment the cheaper one. |
| [`constraints.py`](https://github.com/toxakara/packvium-python/blob/main/examples/constraints.py) | Upright-only, floor-only, non-stackable, top-load limits, and tags that keep two items out of the same box — plus how to read the reason an item was refused. |
| [`units.py`](https://github.com/toxakara/packvium-python/blob/main/examples/units.py) | Why there are no floats anywhere: fractional inches, exact ticks, and the one-tick difference between a fit and a refusal. |
| [`serialization.py`](https://github.com/toxakara/packvium-python/blob/main/examples/serialization.py) | The same request as JSON, the result in full, and exactly which mistakes are refused and which are silently ignored. |
| [`shapes.py`](https://github.com/toxakara/packvium-python/blob/main/examples/shapes.py) | Items that are not their box: complementary wedges sharing one crate as `convex_hull`, and a cushion that compresses under load until the crush limit refuses it. |
| [`nested.py`](https://github.com/toxakara/packvium-python/blob/main/examples/nested.py) | Units into cartons, cartons onto a pallet, in one call. |
| [`commerce.py`](https://github.com/toxakara/packvium-python/blob/main/examples/commerce.py) | Rate a shipment, apply an eligibility rule, and pin a catalog version. |
| [`execution.py`](https://github.com/toxakara/packvium-python/blob/main/examples/execution.py) | Turn a result into dock instructions: solver facts kept apart from screen text, a step order that is injected or honestly absent, and an operator lock that yields a second plan rather than editing the approved one. |
| [`intelligence.py`](https://github.com/toxakara/packvium-python/blob/main/examples/intelligence.py) | Prove a carton change is worth publishing: two scenarios compared order by order, a proposal that refuses to exist on thin evidence, and a replay against held-out history where a cheaper packing your validator rejects still counts as a regression. |
| [`extensions.py`](https://github.com/toxakara/packvium-python/blob/main/examples/extensions.py) | A rule the schema has no field for — and an honest account of what you give up by writing one. |

```bash
PYTHONPATH=src python3 examples/objectives.py
```

## What it does

- **Exact arithmetic.** Length is measured in ticks of 1/16000 mm and weight in 1/8 µg.
  No coordinate is ever a float, so no placement decision depends on rounding.
- **Real constraints.** Weight and payload limits, permitted rotations, keep-upright,
  floor-only, non-stackable, top-load limits, minimum support ratio, tag incompatibility,
  clearance and rectangular obstacles.
- **A solver portfolio, not one algorithm.** Regular-grid, layer, extreme-point,
  maximal-space and bounded exact search, selected by problem shape and profile.
- **Answers you can check.** Every solution is re-validated by logic independent of the
  search. Unplaced items come back with a reason code, not silently missing.
- **Deterministic.** The same input and seed produce the same result, always.
- **Multi-container and nested.** Split across containers, or pack containers into
  containers.
- **Extensible.** Register your own constraints, item orderings, candidate scorers,
  container selectors or complete solvers.
- **Work orders, not just coordinates.** `packvium.execution` turns a validated result
  into a plan: solver facts kept apart from the text that cites them, an injected loading
  order or an honest `unavailable`, and a canonical form four engines emit byte for byte.
  `packvium.locks` lets an operator pin a placement and get a *second* plan beside the
  approved one — never an edit of it, and never a placement the engine would refuse.
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

## License

MIT. See [LICENSE](https://github.com/toxakara/packvium-python/blob/main/LICENSE).
