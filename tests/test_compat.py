import ast
import os
import sys
from pathlib import Path

#: Extra source trees to hold to the same Python floor, supplied by whatever project
#: this package is checked out inside. The published package is a single tree; naming a
#: sibling here would bake in a path that does not exist for anyone who installed it.
EXTRA_SOURCE_ROOTS = "PACKVIUM_EXTRA_PYTHON_SOURCE_ROOTS"

from packvium._compat import _dataclass_options, _reinstate_state_methods, dataclass


def test_python_39_drops_only_the_unsupported_slots_option():
    options = {"frozen": True, "order": True, "slots": True}

    assert _dataclass_options(options, (3, 9)) == {"frozen": True, "order": True}
    assert options == {"frozen": True, "order": True, "slots": True}


def test_python_310_keeps_slots_enabled():
    assert _dataclass_options({"frozen": True, "slots": True}, (3, 10)) == {
        "frozen": True,
        "slots": True,
    }


def test_compat_dataclass_preserves_immutability_and_values():
    @dataclass(frozen=True, slots=True)
    class Value:
        amount: int

    assert Value(7).amount == 7


def test_compat_dataclass_keeps_the_state_methods_a_class_defines():
    # Python 3.10's slotted frozen dataclass replaced both with list-only versions, so a
    # dictionary state -- what a non-slotted runtime pickles -- restored garbage there.
    @dataclass(frozen=True, slots=True)
    class Pair:
        left: int
        right: int

        def __getstate__(self):
            return [self.left, self.right]

        def __setstate__(self, state):
            values = [state["left"], state["right"]] if isinstance(state, dict) else state
            object.__setattr__(self, "left", values[0])
            object.__setattr__(self, "right", values[1])

    restored = Pair.__new__(Pair)
    restored.__setstate__({"left": 1, "right": 2})
    assert (restored.left, restored.right) == (1, 2)
    assert Pair(3, 4).__getstate__() == [3, 4]


def test_a_replaced_state_method_is_put_back_and_an_untouched_one_is_left_alone():
    # What Python 3.10's decorator does, reproduced on any interpreter.
    class Holder:
        def __setstate__(self, state):
            return "own"

    own = {"__setstate__": Holder.__dict__["__setstate__"]}
    Holder.__setstate__ = lambda self, state: "replaced"
    assert _reinstate_state_methods(Holder, own) is Holder
    assert Holder().__setstate__({}) == "own"
    assert _reinstate_state_methods(Holder, own) is Holder
    assert Holder().__setstate__({}) == "own"


def test_all_consumer_python_sources_parse_with_the_python_39_grammar():
    package = Path(__file__).resolve().parents[1]
    roots = [package / "src"]
    roots += [Path(entry) for entry in
              os.environ.get(EXTRA_SOURCE_ROOTS, "").split(os.pathsep) if entry]
    failures = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            source = path.read_text()
            try:
                if sys.version_info[:2] == (3, 9):
                    ast.parse(source, filename=str(path))
                else:
                    ast.parse(source, filename=str(path), feature_version=(3, 9))
            except SyntaxError as error:
                failures.append(f"{path.relative_to(workspace)}:{error.lineno}: {error.msg}")
    assert failures == []
