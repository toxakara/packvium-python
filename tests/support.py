"""Shared construction helpers for the test suite.

Tests read better when the interesting fact is the only thing on the line, so the
boilerplate of building an item, a container or a packing request lives here.
"""

from __future__ import annotations

from packvium import (Container, Dimensions, IndependentSolutionValidator, Item, Packer,
                         PackingConfig, PackingRequest)


def item(id: str, length: int, width: int, height: int, **kwargs) -> Item:
    """An item sized in whole millimetres."""
    return Item.create(id, Dimensions.mm(length, width, height), kwargs.pop("weight", 0), **kwargs)


def container(id: str, length: int, width: int, height: int, **kwargs) -> Container:
    """A container sized in whole millimetres."""
    return Container.create(id, Dimensions.mm(length, width, height), **kwargs)


def pack(items, containers, config: PackingConfig | None = None, *, clock=None):
    return Packer(config or PackingConfig.balanced(), clock=clock).pack(items, containers)


def issues_for(items, containers, packed, **kwargs) -> list[str]:
    """Codes the independent validator raises against a hand-built solution."""
    report = IndependentSolutionValidator().validate(
        PackingRequest(tuple(items), tuple(containers)), tuple(packed), **kwargs
    )
    return [issue.code for issue in report.issues]


def assert_sound(result, items, containers, **kwargs) -> None:
    """Every guarantee the library owes a caller, re-derived from the placements alone.

    A solver that reports a placement it never verified is caught here rather than by
    a test asserting only on counts.
    """
    codes = issues_for(items, containers, result.containers, **kwargs)
    assert codes == [], f"independent validator rejected the solution: {codes}"
    placed = [p.instance.id for c in result.containers for p in c.placements]
    assert len(placed) == len(set(placed)), "an item instance was packed twice"
    expected = {instance.id for one in items for instance in one.instances()}
    assert set(placed) | {u.instance.id for u in result.unpacked} == expected, "items were lost"
