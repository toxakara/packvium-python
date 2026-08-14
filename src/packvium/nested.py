from __future__ import annotations

from ._compat import dataclass

from .config import PackingConfig
from .packer import Packer
from .result import PackingResult


@dataclass(frozen=True, slots=True)
class PackingLevel:
    name: str
    containers: tuple
    config: PackingConfig | None = None


@dataclass(frozen=True, slots=True)
class NestedPackingResult:
    levels: tuple[PackingResult, ...]


class NestedPacker:
    def pack(self, items, levels) -> NestedPackingResult:
        current = tuple(items); results = []
        for level in levels:
            result = Packer(level.config or PackingConfig.balanced()).pack(current, level.containers)
            results.append(result)
            if not result.complete: break
            current = tuple(c.as_item() for c in result.containers)
        return NestedPackingResult(tuple(results))
