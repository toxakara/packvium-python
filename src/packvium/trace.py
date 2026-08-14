"""Opt-in decision trace events (see spec/explainability/decision-trace.md).

A `contextvars.ContextVar` rather than a parameter threaded through every solver
signature: tracing is a cross-cutting diagnostic concern, not a parameter any solver's
own logic needs to know about, and adding it to `SingleContainerSolver.pack_one`'s
signature would touch every implementation of a protocol none of them actually uses.
`use_trace` is the only way to set it, and always resets it afterward (even on an
exception), so a trace from one `Packer.pack()` call can never leak into another.

The one invariant this whole feature exists to keep: enabling a trace must never change
what gets packed. Every call site in `solvers.py` that emits an event guards it behind
`trace.active()` first, so the *cost* of tracing (not just its visible output) is zero
when no one is listening -- not merely "no events," but no event-payload construction
either.
"""

from __future__ import annotations

import contextvars
from typing import Callable

TraceSink = Callable[[dict], None]

_current: contextvars.ContextVar[TraceSink | None] = contextvars.ContextVar(
    "packvium_trace", default=None
)


def active() -> bool:
    return _current.get() is not None


def emit(event: dict) -> None:
    sink = _current.get()
    if sink is not None:
        sink(event)


class use_trace:
    """`with use_trace(sink): packer.pack(...)` -- or `use_trace(None)` for a no-op."""

    def __init__(self, sink: TraceSink | None):
        self._sink = sink
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "use_trace":
        self._token = _current.set(self._sink)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _current.reset(self._token)
