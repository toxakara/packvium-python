from __future__ import annotations

from ._compat import dataclass
from typing import TYPE_CHECKING, Sequence

from .geometry import Dimensions, Point, Rotation

if TYPE_CHECKING:
    from .models import ItemInstance, Placement


@dataclass(frozen=True, slots=True)
class LatticeSummary:
    """Compact, O(1)-to-build description of one `GridSolver` regular-lattice run.

    `GridSolver` already computes every field here in `O(r)` (at most six rotations)
    before it ever places a single item -- rotation, physical/envelope dimensions,
    per-axis capacity, layer step, clearance, and how many instances actually fit.
    Building the summary costs nothing beyond what the solver already pays; what it
    *replaces* is the `O(n)` loop that used to construct one `Placement` domain object
    per item purely to fill in per-item coordinates.

    `expand` reconstructs the exact `Placement` tuple that loop would have built --
    same order, same coordinates, same rotation -- so opting into the compact form
    never loses information, only defers materializing it until a caller actually asks
    for per-item coordinates.

    Restricted to the case `Item.nesting_height` is unset: a nested column's used
    volume and overlap bookkeeping (`nesting.py`) depends on which adjacent pair of
    layers actually touch, which is exact but not worth the closed-form derivation
    risk for a still-uncommon feature. `GridSolver` only builds a `LatticeSummary`
    when the prototype item has no `nesting_height`; nesting keeps using the O(n)
    materializing loop unchanged, regardless of the config flag.
    """

    item_type: str
    rotation: Rotation
    physical: Dimensions
    envelope: Dimensions
    nx: int
    ny: int
    layer_step: int
    clearance_ticks: int
    count: int
    weight_ticks: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("a lattice summary must describe at least one placed instance")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("a lattice summary requires positive per-axis capacity")

    @property
    def _per_layer(self) -> int:
        return self.nx * self.ny

    @property
    def full_layers(self) -> int:
        return self.count // self._per_layer

    @property
    def remainder(self) -> int:
        return self.count % self._per_layer

    @property
    def layers_used(self) -> int:
        return self.full_layers + (1 if self.remainder else 0)

    @property
    def total_weight_ticks(self) -> int:
        return self.count * self.weight_ticks

    @property
    def used_volume_ticks(self) -> int:
        """Physical volume occupied. No nesting overlap term: `GridSolver` only ever
        builds a summary when the prototype has no `nesting_height` (see class
        docstring), so `used_volume` reduces to a plain per-item volume sum -- no
        `_nesting_overlap_volume` term is needed or computed."""
        return self.count * self.physical.volume

    @property
    def max_z_ticks(self) -> int:
        """Highest envelope `z2`, matching `AxisAlignedBox.z2` of the topmost placed
        item's envelope box in the original per-item loop."""
        top_layer_index = self.layers_used - 1
        return top_layer_index * self.layer_step + self.envelope.height.ticks

    def _row_triangular(self, n: int) -> int:
        return n * (n - 1) // 2

    def centre_of_mass_offset_ppm(self, inner_length_ticks: int, inner_width_ticks: int) -> int:
        """Closed-form equivalent of `centre_of_mass.centre_of_mass_offset_ppm` for a
        uniform lattice of identical items, without expanding a single `Placement`.

        The per-item formula there is
        `doubled_weighted_axis = sum(weight * (2*position + dimension))`; because
        every instance shares the same weight and physical dimensions in a lattice,
        this reduces to `weight * (2*envelope_step*sum(index) + count*(2*clearance +
        dimension))`, where `sum(index)` is the sum of the x (or y) grid index across
        every placed item -- itself a closed form over full layers plus one partial
        layer, since the grid fills `x` fastest, then `y`, then `z` (see `GridSolver`).
        """
        if self.weight_ticks == 0 or self.count == 0:
            return 0
        nx, ny = self.nx, self.ny
        full_layers, rows_in_partial, extra_cols = self.full_layers, self.remainder // nx, self.remainder % nx
        sum_x = full_layers * ny * self._row_triangular(nx) + rows_in_partial * self._row_triangular(nx) + self._row_triangular(extra_cols)
        sum_y = (full_layers * nx * self._row_triangular(ny)
                 + nx * self._row_triangular(rows_in_partial)
                 + extra_cols * rows_in_partial)
        clearance = self.clearance_ticks
        doubled_weighted_x = self.weight_ticks * (2 * self.envelope.length.ticks * sum_x + self.count * (2 * clearance + self.physical.length.ticks))
        doubled_weighted_y = self.weight_ticks * (2 * self.envelope.width.ticks * sum_y + self.count * (2 * clearance + self.physical.width.ticks))
        total_weight = self.total_weight_ticks
        numerator_x = doubled_weighted_x - total_weight * inner_length_ticks
        numerator_y = doubled_weighted_y - total_weight * inner_width_ticks
        offset_x_ppm = abs(numerator_x) * 1_000_000 // (total_weight * inner_length_ticks)
        offset_y_ppm = abs(numerator_y) * 1_000_000 // (total_weight * inner_width_ticks)
        return max(offset_x_ppm, offset_y_ppm)

    def expand(self, items: Sequence["ItemInstance"]) -> tuple["Placement", ...]:
        """Rebuild the exact `Placement` tuple the original per-item loop would have
        produced, in the same order, from `items[0:count]`."""
        from .models import Placement

        nx, ny, layer_step, clearance = self.nx, self.ny, self.layer_step, self.clearance_ticks
        out = []
        for index in range(self.count):
            x, y, z = index % nx, (index // nx) % ny, index // (nx * ny)
            point = Point(x * self.envelope.length.ticks, y * self.envelope.width.ticks, z * layer_step)
            position = Point(point.x + clearance, point.y + clearance, point.z + clearance)
            out.append(Placement(items[index], position, self.rotation, self.physical, point, self.envelope, 1.0))
        return tuple(out)
