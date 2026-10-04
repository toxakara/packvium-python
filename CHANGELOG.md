# Changelog

What changed in `packvium` on PyPI, release by release. The format follows
[Keep a Changelog](https://keepachangelog.com/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0]

Refusals you can name by type, and the replay promise of a result without building an
artifact. A misspelt configuration key is now refused (see *Fixed*).

### Added

- **`UnsupportedFeatureError`, `UnknownSolverError`, `UnknownObjectiveError` and
  `PolicyError` on `packvium`.** `pack_from_dict` already raised them; they are now importable
  from the package root, so a refusal can be caught by type instead of by message.
- **`packvium.artifacts.replay_level(result)`** — `"exact"` or `"not_guaranteed"`, the promise an
  operational artifact of this result would make, without building the artifact.

### Changed

- **Faster beam and maximal-space search, same results.** The beam skips work on expansions
  that cannot beat the best plan so far, and the maximal-space solver stops checking for
  containing spaces once none can exist. Measured: beam requests 3–6% faster, `maximal_spaces`
  up to 20% faster on large orders. Every result is byte-identical.

### Fixed

- **A block answer reports the support each item really has.** `HomogeneousBlockSolver` wrote
  `support_ratio` 1.0 for every item, including those of a block set on a smaller one, which
  overhang it. The bottom layer of each block now reports the share of its base that rests on
  something. With `minimum_support_ratio` above zero the block solver is no longer tried at
  all; its answer was built and then discarded by validation.
- **A misspelt `configuration` key is refused.** A key the request schema does not declare in
  `configuration` or its `effort_budget` (`profile` for `solver_profile`, `top_k` for
  `alternatives`) was silently ignored, so the request ran on defaults. `pack_from_dict` now
  raises `InvalidRequestError` with reason `not_allowed` and the key's pointer, for example
  `/configuration/profile`.
- **An unknown `objective` or access direction is a named request error.** `pack_from_dict`
  raised `UnknownObjectiveError` / `InvalidDirectionError` with no reason or field; it now raises
  `InvalidRequestError` with reason `not_allowed` and the value's pointer, like every other
  refusal. The typed `Packer` API still raises `UnknownObjectiveError`.
- **`Length` refuses a fractional float, as `Weight` already did.** `Length.mm(0.1)` silently
  became 1600 ticks and `Length.mm(1e-9)` became 0; both now raise `TypeError`. An integral float
  such as `100.0` is exact and still accepted.
- **Compressible items are no longer laid out as rigid blocks.** `HomogeneousBlockSolver` now
  hands them to the general solver, as `GridSolver` does.
- **Refusal messages match `docs/ERRORS.md`**, e.g. a fixed placement with unknown keys says it
  `cannot carry` them.

## [1.4.0]

Replanning a job that has already started: items already loaded stay where they are, and every
change is recorded against the plan it changed. A request the schema never allowed is now
refused (see *Fixed*).

### Added

- **`fixed_placements`** in a request, and `FixedPlacement` for typed requests. Each entry pins
  an item type to a container type and instance, at the origin a result reports, in one
  orientation. Fixed items keep their place, count toward weight, support and top load, and
  come back marked `"fixed": true`. A fixed set that is not a valid packing on its own raises
  `FixedPlacementError` (`invalid_fixed_placement: ...`) before any search.
- **`packvium.revisions`** — `root_revision`, `derive_revision`, `apply_events`,
  `verify_revision_chain`, `document_digest` and `canonical_revision_json`. A
  `packvium-plan-revision/v1` chain records what happened on the dock — `item_missing`,
  `container_substituted`, `placement_locked`, `placement_verified` — against the artifact it
  changed, linked by SHA-256, and carries the request the next plan solves. Every Packvium
  engine computes the same bytes from the same inputs.
- **`packvium.revision_outcomes`** — turns a chain's events into outcome events for the
  historical evaluation. Supported, not yet signature-frozen, like the rest of that surface.
- **`InvalidRequestError`.** A malformed request names what is wrong: `code` is
  `invalid_request`, `reason` one of a closed set (`missing_field`, `wrong_type`,
  `below_minimum`, `above_maximum`, `negative_measure`, `invalid_unit`, `duplicate_id`,
  `not_allowed`, `invalid_value`), `field` the JSON Pointer of the bad value, and the message
  reads `invalid_request: /items/0/quantity: must be at least 1`. It is a `ValueError`, so
  existing handlers still catch it, and `FixedPlacementError` is its subclass.
- `examples/revisions.py`.

### Changed

- **`packvium.revisions`, `FixedPlacement` and `FixedPlacementError` are in the frozen API
  snapshot.**

### Fixed

- **Limits below their floor are refused.** A zero or negative `max_items`, `max_containers`
  or effort-budget limit was accepted; each now raises `ValueError`.

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
