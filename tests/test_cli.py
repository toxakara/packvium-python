"""Argument handling for `python -m packvium`.

The 354 conformance fixtures already exercise the CLI end to end through
`python -m packvium`, so the packing logic behind it is covered many times over.
What they never touch is the CLI's *own* surface -- which source it reads from, where
it writes, and what it does with input that is not valid JSON at all -- because every
one of those invocations passes a well-formed file and lets the result go to stdout.
"""

from __future__ import annotations

import json

import pytest

from packvium.__main__ import main


def request_json() -> dict:
    return {
        "items": [{"id": "cube", "dimensions": {"length": "40", "width": "40", "height": "40"}}],
        "containers": [{"id": "box", "inner_dimensions": {"length": "100", "width": "100", "height": "100"}}],
    }


def test_reads_from_stdin_and_writes_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["packvium"])
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(request_json())))
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is True


def test_reads_from_a_named_input_file(tmp_path, monkeypatch, capsys):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request_json()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["packvium", str(request_path)])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is True


def test_writes_to_the_named_output_file_instead_of_stdout(tmp_path, monkeypatch, capsys):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request_json()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["packvium", str(request_path), "-o", str(output_path)]
    )
    assert main() == 0
    assert capsys.readouterr().out == ""
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["complete"] is True


def test_long_output_flag_is_equivalent_to_the_short_one(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request_json()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["packvium", str(request_path), "--output", str(output_path)]
    )
    assert main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["complete"] is True


def test_output_is_stable_pretty_printed_json_with_a_trailing_newline(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request_json()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["packvium", str(request_path), "-o", str(output_path)]
    )
    assert main() == 0
    text = output_path.read_text(encoding="utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")
    reencoded = json.dumps(json.loads(text), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert text == reencoded


def test_malformed_json_input_raises_rather_than_producing_a_result(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    request_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["packvium", str(request_path)])
    with pytest.raises(json.JSONDecodeError):
        main()


def test_a_missing_input_file_raises_rather_than_producing_a_result(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["packvium", str(tmp_path / "missing.json")])
    with pytest.raises(FileNotFoundError):
        main()


def test_an_unknown_flag_is_rejected_before_any_input_is_read(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["packvium", "--not-a-real-flag"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
