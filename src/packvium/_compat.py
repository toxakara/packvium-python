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


def dataclass(cls: type[_T] | None = None, **options: Any) -> Any:
    """Delegate to stdlib dataclass, omitting only unsupported 3.10 keywords."""

    compatible = _dataclass_options(options, sys.version_info[:2])

    def decorate(target: type[_T]) -> type[_T]:
        return _stdlib_dataclass(target, **compatible)

    return decorate if cls is None else decorate(cls)
