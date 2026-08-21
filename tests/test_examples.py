"""Every shipped example actually runs.

An example nobody executes is documentation that rots silently: the API moves, the
example keeps compiling in a reader's head, and the first person to paste it discovers
it stopped working three releases ago. So the whole `examples/` directory is a test
target -- each file is run as a real subprocess, exactly the way the README tells a
reader to run it.

The checks beyond "exit 0" are deliberate. An example that prints nothing has nothing to
teach, and one whose docstring does not say how to run it makes the reader guess at the
PYTHONPATH.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
EXAMPLES = PACKAGE / "examples"
EXAMPLE_FILES = sorted(EXAMPLES.glob("*.py"))


def example_ids() -> list[str]:
    return [path.name for path in EXAMPLE_FILES]


def test_the_examples_directory_is_not_empty():
    """Guards the parametrization itself: a glob that matched nothing would make every
    test below vacuously pass."""
    assert EXAMPLE_FILES, f"no examples found under {EXAMPLES}"


@pytest.fixture(scope="module")
def run_example():
    environment = {**os.environ, "PYTHONPATH": str(PACKAGE / "src"), "PYTHONIOENCODING": "utf-8"}

    def run(path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(path)], cwd=PACKAGE, env=environment,
            capture_output=True, text=True, timeout=180,
        )

    return run


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=example_ids())
def test_an_example_runs_to_completion(path, run_example):
    finished = run_example(path)
    assert finished.returncode == 0, f"{path.name} exited {finished.returncode}:\n{finished.stderr}"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=example_ids())
def test_an_example_prints_something(path, run_example):
    assert run_example(path).stdout.strip(), f"{path.name} produced no output"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=example_ids())
def test_an_example_raises_nothing_it_did_not_mean_to(path, run_example):
    """Several examples deliberately catch and print a refusal -- that is the lesson.
    What must not appear is an *uncaught* one."""
    assert "Traceback (most recent call last)" not in run_example(path).stderr


#: `duration_ms` is a measurement of the run, not part of the answer, and it is the one
#: number in a result that legitimately differs between two identical solves.
TIMING = re.compile(r'("(?:duration_ms|elapsed_ms)":\s*)\d+|(\bin )\d+( ms\b)')


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=example_ids())
def test_an_example_is_deterministic(path, run_example):
    """This library promises the same answer for the same input, forever. An example
    whose output moves between two runs is either demonstrating something it should not
    be, or has found a determinism bug worth knowing about.

    Elapsed times are masked rather than asserted on -- they are the one part of a result
    that is a fact about the machine instead of about the packing.
    """
    first, second = run_example(path).stdout, run_example(path).stdout
    assert TIMING.sub(r"\1\2<masked>\3", first) == TIMING.sub(r"\1\2<masked>\3", second)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=example_ids())
def test_an_example_says_how_to_run_it(path):
    docstring = ast.get_docstring(ast.parse(path.read_text()))
    assert docstring, f"{path.name} has no module docstring"
    assert "Run it:" in docstring, f"{path.name} does not tell the reader how to run it"
    assert f"examples/{path.name}" in docstring, f"{path.name}'s run instruction names another file"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=example_ids())
def test_an_example_imports_only_the_public_package(path):
    """A reader copies an example verbatim. If it reaches into a private module, they
    inherit a dependency on something that is free to change without notice."""
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    # `__future__` is a language directive, not a private module.
    private = [
        name for name in imported
        if name.split(".")[-1].startswith("_") and not name.startswith("__")
    ]
    assert not private, f"{path.name} imports private module(s) {private}"


def test_the_readme_lists_every_example():
    """Examples that are not linked are examples nobody finds."""
    readme = (PACKAGE / "README.md").read_text()
    missing = [path.name for path in EXAMPLE_FILES if path.name not in readme]
    assert not missing, f"README.md does not mention {missing}"
