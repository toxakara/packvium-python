"""Small runtime compatibility helpers for the public Python package.

The canonical models use ``dataclass(slots=True)`` on Python 3.10+ to reduce
per-instance memory.  Python 3.9 has the same dataclass semantics but does not
accept the ``slots`` keyword.  Keeping the version branch in one lower-layer
module lets every model retain one definition and one public API.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass as _stdlib_dataclass
from typing import Any, Callable, TypeVar, overload


_T = TypeVar("_T")


def _dataclass_options(options: dict[str, Any], version: tuple[int, int]) -> dict[str, Any]:
    compatible = dict(options)
    if version < (3, 10):
        compatible.pop("slots", None)
    return compatible


@overload
def dataclass(cls: type[_T]) -> type[_T]: ...


@overload
def dataclass(**options: Any) -> Callable[[type[_T]], type[_T]]: ...


#: State methods a class may define for itself. Python 3.10's ``slots=True, frozen=True``
#: replaces both with list-only versions; 3.11 and later keep the class's own. A snapshot
#: that must restore a dictionary state from a non-slotted runtime depends on its own.
_OWN_STATE_METHODS = ("__getstate__", "__setstate__")


def dataclass(cls: type[_T] | None = None, **options: Any) -> Any:
    """Delegate to stdlib dataclass, omitting only unsupported 3.10 keywords, and keeping
    any state method the class defines itself, as Python 3.11 and later do."""

    compatible = _dataclass_options(options, sys.version_info[:2])

    def decorate(target: type[_T]) -> type[_T]:
        own = {name: target.__dict__[name] for name in _OWN_STATE_METHODS if name in target.__dict__}
        return _reinstate_state_methods(_stdlib_dataclass(target, **compatible), own)

    return decorate if cls is None else decorate(cls)


def _reinstate_state_methods(decorated: type[_T], own: dict[str, Any]) -> type[_T]:
    """Put back each state method the class defined that the decorator replaced."""
    for name, method in own.items():
        if decorated.__dict__.get(name) is not method:
            setattr(decorated, name, method)
    return decorated
