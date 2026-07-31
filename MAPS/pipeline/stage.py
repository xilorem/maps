"""Stage IR for scheduled execution units on the mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from maps.planning.submesh import Submesh
from MAPS.pipeline.layer import Layer

if TYPE_CHECKING:
    from MAPS.core.tensor import Tensor


@dataclass(frozen=True)
class Stage:
    """One scheduled Execution Plan Stage on a placed submesh."""

    name: str
    submesh: Submesh
    layers: tuple[Layer, ...] = field(default_factory=tuple)
    virtual_to_physical: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not self.layers:
            raise ValueError("stages must contain at least one layer")

    @property
    def physical_to_virtual(self) -> dict[int, int]:
        """Return physical tile ids keyed to their virtual tile ids."""

        return {
            physical_tile_id: virtual_tile_id
            for virtual_tile_id, physical_tile_id in self.virtual_to_physical.items()
        }

    def validate_tensors(self, tensors: tuple["Tensor", ...]) -> None:
        """Validate layer tensor ids and output layout compatibility."""

        for layer in self.layers:
            layer.validate_tensors(tensors)
