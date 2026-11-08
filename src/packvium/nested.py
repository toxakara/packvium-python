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
    """Packs levels in order, each level's packed containers becoming the next level's items.

    `level_beam_width` K carries the K best chains from one level to the next. A
    chain is a sequence of level results; it extends by packing its handoff against the
    level's whole container list and, when K > 1 and the level offers more than one, against
    each container alone. Chains rank by their outermost result's score, then each earlier
    level's outward in, then generation order; two chains handing up the same multiset of
    (container type, billed dimensions, gross weight) keep only the better. A level with no
    complete candidate ends every chain, and the best incomplete one is the answer. K = 1 is
    exactly the greedy chain. At most
    L * K * (T + 1) level packs for L levels offering at most T containers each.
    """

    def __init__(self, level_beam_width: int = 1) -> None:
        if isinstance(level_beam_width, bool) or not isinstance(level_beam_width, int) or level_beam_width < 1:
            raise ValueError("level_beam_width must be a positive integer")
        self.level_beam_width = level_beam_width

    def pack(self, items, levels) -> NestedPackingResult:
        beam: list[tuple[PackingResult, ...]] = [()]
        handoffs: list[tuple] = [tuple(items)]
        for level in levels:
            ranked = self._ranked_extensions(level, beam, handoffs)
            complete = [chain for chain in ranked if chain[-1].complete]
            if not complete:
                return NestedPackingResult(ranked[0])
            beam = self._distinct_handoffs(complete)
            handoffs = [tuple(c.as_item() for c in chain[-1].containers) for chain in beam]
        return NestedPackingResult(beam[0])

    def _ranked_extensions(self, level: PackingLevel, beam, handoffs) -> list[tuple[PackingResult, ...]]:
        packer = Packer(level.config or PackingConfig.balanced())
        extensions = [chain + (result,)
                      for chain, items in zip(beam, handoffs)
                      for result in self._candidates(packer, items, tuple(level.containers))]
        # Sorting is stable, so generation order is the final tie-break.
        return sorted(extensions, key=lambda chain: tuple(result.score for result in reversed(chain)))

    def _candidates(self, packer: Packer, items: tuple, containers: tuple) -> list[PackingResult]:
        offers = [containers]
        if self.level_beam_width > 1 and len(containers) > 1:
            offers.extend((container,) for container in containers)
        return [packer.pack(items, offer) for offer in offers]

    def _distinct_handoffs(self, ranked: list[tuple[PackingResult, ...]]) -> list[tuple[PackingResult, ...]]:
        kept: list[tuple[PackingResult, ...]] = []
        seen: set[tuple] = set()
        for chain in ranked:
            handoff = self._handoff_key(chain[-1])
            if handoff in seen:
                continue
            seen.add(handoff)
            kept.append(chain)
            if len(kept) == self.level_beam_width:
                break
        return kept

    @staticmethod
    def _handoff_key(result: PackingResult) -> tuple:
        # A multiset: instance numbers and emission order are not part of a packing's
        # contract (conformance/canonical.py), so two engines must collapse the same chains.
        return tuple(sorted((c.container.id, c.billed_dimensions.length.ticks, c.billed_dimensions.width.ticks,
                             c.billed_dimensions.height.ticks, c.gross_weight.ticks) for c in result.containers))
