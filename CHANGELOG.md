# Changelog

What changed in `packvium` on PyPI, release by release. The format follows
[Keep a Changelog](https://keepachangelog.com/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0]

Portable operational artifacts: hand a packing result to a WMS, a TMS or a printer that runs
no engine. Nothing breaks 1.2.0.

### Added

- **`packvium.artifacts`** — `build_operational_artifact(request, result, loading_orders=)` and
  `canonical_artifact_json()`. One `packvium-operational-artifact/v1` document carries the
  execution plan, exact geometry, the values a work order shows, and the request that produced
  it. Every Packvium engine builds the same bytes from the same result.
- **`packvium.artifact_exports`** — `export_json`, `export_csv` (RFC 4180, 18 columns) and
  `export_work_order_html`: a printable work order in one HTML file with no scripts and no
  external resources.
- `examples/artifacts.py`.

### Changed

- **`packvium.execution`, `packvium.artifacts` and `packvium.artifact_exports` are in the frozen
  API snapshot**, with the same guarantee as the rest of the public surface.
- **`canonical_plan_json` writes RFC 8785.** Every plan the engine produces keeps its 1.2.0
  bytes. A float, U+2028 or a key outside the Basic Multilingual Plane is now spelled as the
  other engines spell it, and an integer beyond 2^53 − 1 raises instead of being written.
- **Faster commerce lookups, same answers.** A pinned catalog or tariff version resolves
  without scanning the history (1000 lookups over 16000 versions: 374 ms → 1.2 ms), a
  snapshot's SKU lookup builds its index on first use, and `packvium.pareto` finds a two-axis
  frontier with one sort instead of comparing every pair.

## [1.2.0]

Execution plans, operator locks, and the intelligence API. Nothing breaks 1.1.0.

### Added

- **`packvium.execution`** — `build_execution_plan()` and `canonical_plan_json()` turn a
  validated result into a work order: what to lift, why a carton was chosen, what was not
  packed. Step order is taken from `loading_orders` or reported as `"unavailable"`.
- **`packvium.locks`** — pin a placement and get a constrained re-solve beside the approved
  plan (`resolve_with_locks`, `locks_from_plan`). A lock can reserve its slot but never force a
  placement the engine would refuse; a lock that cannot be honoured is reported, not raised.
- **`packvium.simulation`, `.recommendations`, `.outcomes`, `.holdout`, `.pareto`** — compare
  two scenarios order by order, propose a carton or rule change only on enough paired
  evidence, publish it through an explicit approval, and replay it against held-out history.
  Supported, but not yet signature-frozen: pin the version if you rely on exact shapes.
- `pack_from_dict(..., extensions=)` — keyword-only, defaults to `None`.
- `examples/execution.py`, `examples/intelligence.py`.

### Fixed

- The new modules imported on Python 3.10+ only; they now import on 3.9, the declared minimum.
- A `NaN` metric is refused with a named error instead of being ranked Pareto-optimal.

## Earlier releases

Up to 1.1.0 one changelog covered every Packvium language. Those entries are kept in
this repository's GitHub Releases for each tag.
