"""The execution-plan adapter.

`docs/EXECUTION-PLAN.md` is the contract. What is checked here is what the design says the
adapter must not do, because those are the properties that decay quietly:

  * it must not invent a step order when it was not given one;
  * it must not reference a placement by an identifier the cross-language projection drops;
  * it must not blend a score vector into one number, or claim a cause the solver never
    recorded;
  * presentation text must cite the authoritative fields it came from.

The alternatives assertions run on a constructed result, and the reason is not that
nothing produces them: all four engines do, and 165 of the 399 fixtures carry a non-empty
list on a live solve. It is that `conformance/golden/` stores the *projection*, which has no
`alternatives` key at all -- so the corpus this suite is built around cannot exercise these
rules, however many of its requests would produce one. `TestARealEngineProducesAlternatives`
below solves for real instead.
"""

from __future__ import annotations

import pytest

from packvium.execution import (
    FORMAT,
    ExecutionPlanError,
    build_execution_plan,
    canonical_plan_json,
    placement_reference,
)


def _placement(item_type: str, x: int, y: int, z: int, orientation: str = "LWH") -> dict:
    def scalar(ticks: int) -> dict:
        # Both spellings, as a real result carries them: `value` is a rendering and
        # `ticks` is the number. A test that supplied only one could not catch a reference
        # built on the wrong one.
        return {"ticks": ticks, "value": str(ticks // 16000), "unit": "mm"}
    return {
        "item_id": f"{item_type}#{x}{y}{z}",
        "item_type": item_type,
        "orientation": orientation,
        "position": {"x": scalar(x), "y": scalar(y), "z": scalar(z)},
        "dimensions": {"length": scalar(1600000), "width": scalar(1600000),
                       "height": scalar(1600000)},
        "support_ratio": 1.0,
        "top_load": 0,
    }


def _result(**overrides) -> dict:
    base = {
        "status": "feasible",
        "objective": "default",
        "score": [0, 1, 0, 0, 1000000],
        "feasibility": {"code": "feasible"},
        "optimality": {"code": "not_proven"},
        "containers": [{
            "id": "box#1",
            "container_type": "box",
            "volume_utilization": 0.5,
            "placements": [_placement("cube", 0, 0, 0), _placement("cube", 1600000, 0, 0)],
        }],
        "unpacked_items": [],
        "alternatives": [],
    }
    base.update(overrides)
    return base


class TestTheStepOrderIsNeverInvented:
    def test_without_an_injected_order_the_plan_says_so(self):
        plan = build_execution_plan({}, _result())
        container = plan["containers"][0]
        assert container["order"] == "unavailable"
        # Every placement is still listed, so nothing is hidden -- but no step numbers are
        # attached, because array position is an artifact of candidate iteration.
        assert len(container["steps"]) == 2
        assert all("sequence" not in step for step in container["steps"])

    def test_an_injected_order_is_used_verbatim(self):
        plan = build_execution_plan({}, _result(), loading_orders={0: [1, 0]})
        container = plan["containers"][0]
        assert container["order"] == "loading"
        assert [step["sequence"] for step in container["steps"]] == [1, 2]
        # Step 1 is the placement the caller put first, not the one the result listed first.
        assert container["steps"][0]["placement"]["position_ticks"]["x"] == 1600000

    def test_an_order_that_is_not_a_permutation_is_refused(self):
        with pytest.raises(ExecutionPlanError, match="permutation"):
            build_execution_plan({}, _result(), loading_orders={0: [0, 0]})


class TestThePlacementReferenceIsCrossLanguageSafe:
    def test_it_does_not_use_item_id(self):
        """`item_id` exists and is dropped by conformance/canonical.py as an instance count.

        Referencing it would make a plan that two correct engines disagree about.
        """
        reference = placement_reference(0, _placement("cube", 0, 0, 0))
        assert "item_id" not in reference
        assert set(reference) == {"container_index", "item_type", "orientation", "position_ticks"}

    def test_it_reads_ticks_and_not_the_rendered_value(self):
        placement = _placement("cube", 12345, 0, 0)
        placement["position"]["x"]["value"] = "wrong"
        assert placement_reference(0, placement)["position_ticks"]["x"] == 12345

    def test_it_refuses_a_placement_it_cannot_reference(self):
        placement = _placement("cube", 0, 0, 0)
        del placement["orientation"]
        with pytest.raises(ExecutionPlanError, match="missing a field"):
            placement_reference(0, placement)


class TestAnAlternativeIsExplainedWithoutInventingOne:
    """Constructed results: the golden corpus has no ranked runners-up to use."""

    def test_the_loss_is_the_first_differing_index_and_never_a_blend(self):
        loser = {"status": "feasible", "score": [0, 2, 0, 0, 900000]}
        plan = build_execution_plan({}, _result(alternatives=[loser]))
        facts = plan["alternatives"][0]["facts"]
        assert facts["first_difference"] == {
            "index": 1, "winner": 1, "alternative": 2, "difference": 1,
        }
        # Nothing anywhere sums or weights the vector.
        assert "total" not in facts and "weighted" not in facts

    def test_the_sentence_names_an_axis_and_claims_no_cause(self):
        loser = {"status": "feasible", "score": [0, 2, 0, 0, 900000]}
        plan = build_execution_plan({}, _result(alternatives=[loser]))
        presentation = plan["alternatives"][0]["presentation"]
        assert "axis 1" in presentation["summary"]
        assert presentation["cites"] == ["score", "alternatives[].score"]
        for causal in ("because", "due to", "caused"):
            assert causal not in presentation["summary"].lower()

    def test_an_identical_score_is_reported_as_undetermined(self):
        twin = {"status": "feasible", "score": [0, 1, 0, 0, 1000000]}
        plan = build_execution_plan({}, _result(alternatives=[twin]))
        assert plan["alternatives"][0]["facts"]["first_difference"] is None
        assert "does not record why" in plan["alternatives"][0]["presentation"]["summary"]

    def test_score_vectors_of_different_length_are_refused(self):
        with pytest.raises(ExecutionPlanError, match="different length"):
            build_execution_plan({}, _result(alternatives=[{"score": [0, 1]}]))

    def test_a_result_with_no_alternatives_is_well_formed(self):
        # The common case, and today the only one.
        plan = build_execution_plan({}, _result())
        assert plan["alternatives"] == []
        assert plan["format"] == FORMAT


class TestARealEngineProducesAlternatives:
    """Evidence for the claim the constructed tests above stand in for.

    Worth a real solve rather than another fixture, because the golden corpus stores the
    projection and cannot answer a question about this field -- reading it is what produced
    two wrong claims about `alternatives` in a row.

    Live, 165 of 399 fixtures carry a non-empty list. The empty ones have four causes, and
    the ones this class pins are the two a reader is most likely to misread: a request the
    grid lattice completes has no runners-up, and `configuration.alternatives`
    counts the winner, so the schema's own minimum of `1` returns nothing.

    The last two tests are 's decision made enforceable. No part of this field's
    content is in the cross-language contract, but three invariants hold of every engine
    independently, and two of them are checkable here.
    """

    @staticmethod
    def _pack(items):
        from packvium import pack_from_dict

        return pack_from_dict({
            "configuration": {"solver_profile": "quality", "alternatives": 4,
                              "time_limit_ms": 30000},
            "items": items,
            "containers": [{"id": "box",
                            "inner_dimensions": {"length": "400", "width": "300",
                                                 "height": "300"}}],
        })

    def test_a_lattice_the_grid_completes_has_no_runners_up(self):
        tiling = [{"id": "cube", "quantity": 8,
                   "dimensions": {"length": "100", "width": "100", "height": "100"}}]
        assert self._pack(tiling).get("alternatives") == []

    def test_a_request_the_lattice_cannot_complete_ranks_several(self):
        mixed = [
            {"id": "a", "quantity": 3,
             "dimensions": {"length": "170", "width": "110", "height": "90"}},
            {"id": "b", "quantity": 4,
             "dimensions": {"length": "130", "width": "70", "height": "50"}},
            {"id": "c", "quantity": 2,
             "dimensions": {"length": "90", "width": "90", "height": "210"}},
        ]
        result = self._pack(mixed)
        alternatives = result.get("alternatives") or []
        assert alternatives, "the engine ranked no runners-up on a mixed-size request"
        # Each is a full result with its own integer score vector -- which is what the
        # adapter's ranking rules read, and why they are implementable at all.
        for alternative in alternatives:
            assert alternative["score"] and all(
                isinstance(value, int) for value in alternative["score"])

    def test_no_alternative_outranks_the_winner(self):
        """'s second invariant. If an alternative scored better it would *be* the
        winner, so a violation means the ranking and the reported result disagree."""
        mixed = [
            {"id": "a", "quantity": 3,
             "dimensions": {"length": "170", "width": "110", "height": "90"}},
            {"id": "b", "quantity": 4,
             "dimensions": {"length": "130", "width": "70", "height": "50"}},
        ]
        result = self._pack(mixed)
        assert result.get("alternatives"), "no runners-up to check the invariant against"
        for alternative in result["alternatives"]:
            assert alternative["score"] >= result["score"], alternative["score"]

    def test_the_configured_count_includes_the_winner(self):
        """The caller-visible surprise found, pinned so it cannot drift.

        `configuration.alternatives` is the size of the ranked set the portfolio keeps, not
        the number of runners-up returned. The schema types it `minimum: 1`, and `1` yields
        an empty list -- so a request asking for alternatives can be named for them and
        still never carry one. Documented in `docs/PUBLIC-API.md`; changing the meaning
        would be a contract break, so the documentation is the fix.
        """
        from packvium import pack_from_dict

        mixed = [
            {"id": "a", "quantity": 3,
             "dimensions": {"length": "170", "width": "110", "height": "90"}},
            {"id": "b", "quantity": 4,
             "dimensions": {"length": "130", "width": "70", "height": "50"}},
        ]

        def pack(cap):
            return pack_from_dict({
                "configuration": {"solver_profile": "quality", "alternatives": cap,
                                  "time_limit_ms": 30000},
                "items": mixed,
                "containers": [{"id": "box", "inner_dimensions": {
                    "length": "400", "width": "300", "height": "300"}}],
            })

        assert pack(1).get("alternatives") == []
        for cap in (2, 3, 4):
            assert len(pack(cap).get("alternatives") or []) <= cap - 1

    def test_the_adapter_explains_a_real_alternative(self):
        """The ranking rules against engine output rather than a constructed dict."""
        mixed = [
            {"id": "a", "quantity": 3,
             "dimensions": {"length": "170", "width": "110", "height": "90"}},
            {"id": "b", "quantity": 4,
             "dimensions": {"length": "130", "width": "70", "height": "50"}},
            {"id": "c", "quantity": 2,
             "dimensions": {"length": "90", "width": "90", "height": "210"}},
        ]
        result = self._pack(mixed)
        plan = build_execution_plan({}, result)
        assert plan["alternatives"], "no alternative reached the plan"
        for entry in plan["alternatives"]:
            difference = entry["facts"]["first_difference"]
            # Either they differ at some index, or they tie and the plan says the score
            # does not record why one was taken. There is no third answer.
            assert difference is None or difference["index"] >= 0
            assert entry["presentation"]["cites"]


class TestFactsAndPresentationStaySeparate:
    def test_an_unpacked_item_keeps_its_proof_level_unsoftened(self):
        result = _result(unpacked_items=[{
            "item_id": "ladder#1", "item_type": "ladder", "reason": "no_container_fits",
            "details": ["longest dimension exceeds every container"],
            "proof": {"level": "observed", "observations": [{"code": "too_long"}]},
        }])
        entry = build_execution_plan({}, result)["unplaced"][0]
        assert entry["facts"]["proof_level"] == "observed"
        # The level appears in the sentence too: a reader must not be told "cannot fit"
        # when the engine only observed that it did not.
        assert "observed" in entry["presentation"]["summary"]
        assert entry["presentation"]["cites"] == [
            "unpacked_items[].reason", "unpacked_items[].proof.level",
        ]

    def test_every_presentation_block_cites_at_least_one_field(self):
        result = _result(
            alternatives=[{"status": "feasible", "score": [0, 2, 0, 0, 900000]}],
            unpacked_items=[{
                "item_id": "x#1", "item_type": "x", "reason": "no_container_fits",
                "details": [], "proof": {"level": "proven", "observations": [{"code": "c"}]},
            }],
        )
        plan = build_execution_plan({}, result)
        blocks = [entry["presentation"] for entry in plan["alternatives"] + plan["unplaced"]]
        assert blocks, "the fixture must produce at least one presentation block"
        for block in blocks:
            assert block["cites"], block


class TestThePlanIsDeterministicAndByteComparable:
    def test_the_same_inputs_produce_the_same_bytes(self):
        first = canonical_plan_json(build_execution_plan({}, _result()))
        second = canonical_plan_json(build_execution_plan({}, _result()))
        assert first == second
        # Sorted keys and no incidental whitespace, so PHP can be diffed against this.
        assert first.startswith('{"alternatives":') and ", " not in first

    def test_a_result_without_a_status_is_not_a_validated_result(self):
        with pytest.raises(ExecutionPlanError, match="validated result"):
            build_execution_plan({}, {"containers": []})


class TestPythonAndPhpAgreeToTheByte:
    """'s actual bar, under test rather than observed once.

    A committed expected string would only prove PHP still agrees with a string. Running
    both adapters over the same result is what proves the two implementations agree with
    each other, which is the claim being made.
    """

    @staticmethod
    def _php_plan(result: dict, orders: dict | None = None) -> str:
        import json
        import shutil
        import subprocess
        from pathlib import Path

        if shutil.which("php") is None:
            pytest.skip("php is not available in this environment")
        root = Path(__file__).resolve().parents[2]
        # The PHP engine sits beside this package in the workspace; a published copy does
        # not carry it, and the cross-language equality is proven where both exist.
        if not (root / "packvium-php" / "autoload.php").is_file():
            pytest.skip("the PHP engine is not part of this package")
        script = (
            'require $argv[1] . "/autoload.php";'
            '$r = json_decode($argv[2], true, 512, JSON_THROW_ON_ERROR);'
            '$o = json_decode($argv[3], true, 512, JSON_THROW_ON_ERROR);'
            'echo Packvium\\Execution\\Plan::canonicalJson('
            'Packvium\\Execution\\Plan::build([], $r, $o));'
        )
        finished = subprocess.run(
            ["php", "-r", script, str(root / "packvium-php"),
             json.dumps(result), json.dumps({str(k): v for k, v in (orders or {}).items()})],
            capture_output=True, text=True, check=False,
        )
        assert finished.returncode == 0, finished.stderr[-2000:]
        return finished.stdout

    def test_a_plain_result_agrees(self):
        plan = canonical_plan_json(build_execution_plan({}, _result()))
        assert plan == self._php_plan(_result())

    def test_floats_unpacked_items_and_alternatives_agree(self):
        """The case most likely to diverge: a repeating float, a proof level and a ranking.

        `volume_utilization` is a double, and two languages rendering it differently is
        the classic way a byte-identity claim quietly stops being true.
        """
        result = _result(
            containers=[{
                "id": "box#1", "container_type": "box",
                "volume_utilization": 0.3333333333333333,
                "placements": [_placement("crate", 0, 0, 0)],
            }],
            unpacked_items=[{
                "item_id": "ladder#1", "item_type": "ladder", "reason": "no_container_fits",
                "details": ["longest dimension exceeds every container"],
                "proof": {"level": "observed", "observations": [{"code": "too_long"}]},
            }],
            alternatives=[{"status": "feasible", "score": [0, 3, 0, 250000, 900000]}],
        )
        plan = canonical_plan_json(build_execution_plan({}, result, loading_orders={0: [0]}))
        assert plan == self._php_plan(result, {0: [0]})


class TestTheAdapterCannotBecomeABypass:
    def test_it_imports_no_solver_and_no_validator(self):
        """The dependency direction from docs/EXECUTION-PLAN.md, as the cheapest test of it.

        The adapter has no way to produce a placement, because it has no solver, and no way
        to bless one, because it has no validator. It can only describe what a validated
        result already says.
        """
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1]
                  / "src/packvium/execution.py").read_text()
        for forbidden in ("from .solvers", "from .packer", "from .validation",
                          "import solvers", "import packer", "import validation"):
            assert forbidden not in source, forbidden
