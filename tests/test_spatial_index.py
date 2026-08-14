"""The spatial index is a broad-phase filter, not the source of truth.

Every test here compares `SpatialIndex.query()` against a naive O(n) scan using the
*exact same* intersection predicate `find_candidates` uses (see solvers.py). The one
property that must never break is that the index's result is a superset of the naive
scan's -- an index that reports too many candidates only costs a few wasted exact
checks downstream, but one that misses a real collision would let two items overlap.
"""

from __future__ import annotations

import random

import pytest

from packvium.spatial_index import SpatialIndex, build


def exact_intersects(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2 and az1 < bz2 and bz1 < az2


def naive_overlaps(bounds: list[tuple[int, ...]], query: tuple[int, ...]) -> set[int]:
    return {i for i, bound in enumerate(bounds) if exact_intersects(bound, query)}


def random_bound(rng: random.Random, span: int) -> tuple[int, int, int, int, int, int]:
    x1, y1, z1 = (rng.randrange(0, span) for _ in range(3))
    x2, y2, z2 = x1 + rng.randrange(1, span // 4 + 1), y1 + rng.randrange(1, span // 4 + 1), z1 + rng.randrange(1, span // 4 + 1)
    return (x1, y1, z1, x2, y2, z2)


def test_an_empty_index_finds_nothing():
    index = SpatialIndex(1000, 1000, 1000)
    assert list(index.query(0, 0, 0, 100, 100, 100)) == []


def test_a_single_box_is_found_by_an_overlapping_query():
    index = SpatialIndex(1000, 1000, 1000)
    index.add(0, (0, 0, 0, 100, 100, 100))
    assert 0 in set(index.query(50, 50, 50, 150, 150, 150))


def test_a_disjoint_box_far_away_is_not_returned():
    index = SpatialIndex(1000, 1000, 1000)
    index.add(0, (0, 0, 0, 10, 10, 10))
    # Far enough that it cannot possibly share a grid cell at any reasonable sizing.
    assert list(index.query(900, 900, 900, 950, 950, 950)) == []


def test_edge_touching_boxes_are_still_candidates_for_the_exact_check():
    """Touching (zero overlap) is not a collision, but the index only has to avoid
    missing real ones -- it is allowed to hand the exact check a touching pair."""
    index = SpatialIndex(1000, 1000, 1000)
    index.add(0, (0, 0, 0, 100, 100, 100))
    candidates = set(index.query(100, 0, 0, 200, 100, 100))
    # Whether or not the touching box is returned, the exact predicate (not the index)
    # is what must say "not blocked" -- confirm that predicate agrees separately.
    assert not exact_intersects((0, 0, 0, 100, 100, 100), (100, 0, 0, 200, 100, 100))
    naive = naive_overlaps([(0, 0, 0, 100, 100, 100)], (100, 0, 0, 200, 100, 100))
    assert naive <= candidates


@pytest.mark.parametrize("seed", range(30))
def test_the_index_never_misses_a_true_collision(seed):
    rng = random.Random(seed)
    span = 2000
    bounds = [random_bound(rng, span) for _ in range(rng.randint(1, 150))]
    index = build(bounds, span, span, span)

    for _ in range(20):
        query = random_bound(rng, span)
        expected = naive_overlaps(bounds, query)
        found = set(index.query(*query))
        assert expected <= found, (query, expected - found)


@pytest.mark.parametrize("seed", range(10))
def test_incremental_add_matches_a_bulk_build(seed):
    rng = random.Random(seed)
    span = 1000
    bounds = [random_bound(rng, span) for _ in range(rng.randint(1, 80))]

    incremental = SpatialIndex(span, span, span)
    for position, bound in enumerate(bounds):
        incremental.add(position, bound)
    bulk = build(bounds, span, span, span)

    for _ in range(10):
        query = random_bound(rng, span)
        assert set(incremental.query(*query)) == set(bulk.query(*query))


def test_copy_is_independent_of_the_original():
    index = SpatialIndex(1000, 1000, 1000)
    index.add(0, (0, 0, 0, 100, 100, 100))
    clone = index.copy()

    clone.add(1, (500, 500, 500, 600, 600, 600))
    index.add(2, (10, 10, 10, 20, 20, 20))

    assert 1 in set(clone.query(500, 500, 500, 600, 600, 600))
    assert 1 not in set(index.query(500, 500, 500, 600, 600, 600))
    assert 2 in set(index.query(10, 10, 10, 20, 20, 20))
    assert 2 not in set(clone.query(10, 10, 10, 20, 20, 20))


def test_a_box_spanning_many_cells_is_found_from_any_of_them():
    index = SpatialIndex(800, 800, 800, cells_per_axis=8)
    big = (0, 0, 0, 800, 100, 100)  # spans every x cell
    index.add(0, big)
    for x in (0, 200, 400, 600, 790):
        assert 0 in set(index.query(x, 0, 0, x + 5, 5, 5)), x
