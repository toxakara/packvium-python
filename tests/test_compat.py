import ast
import os
import sys
from pathlib import Path

#: Extra source trees to hold to the same Python floor, supplied by whatever project
#: this package is checked out inside. The published package is a single tree; naming a
#: sibling here would bake in a path that does not exist for anyone who installed it.
EXTRA_SOURCE_ROOTS = "PACKVIUM_EXTRA_PYTHON_SOURCE_ROOTS"

from packvium._compat import _dataclass_options, dataclass


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
