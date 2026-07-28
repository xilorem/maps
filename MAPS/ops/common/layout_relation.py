"""Bidirectional input/output layout contracts."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.layout import LayoutAxis, TensorLayout
from MAPS.core.tensor import Tensor


@dataclass(frozen=True)
class LayoutRelation:
    """One immutable, indexed input-to-output layout relationship.

    ``input_axis_for_output_axis`` maps every output tensor axis to the
    corresponding input tensor axis. Relations are deliberately opt-in.
    """

    input_index: int
    output_index: int
    input_axis_for_output_axis: tuple[int, ...]
    guarantees_slice_containment: bool

    @classmethod
    def exact(
        cls,
        *,
        input_index: int,
        output_index: int,
        tensor: Tensor,
    ) -> "LayoutRelation":
        return cls(
            input_index=input_index,
            output_index=output_index,
            input_axis_for_output_axis=tuple(range(tensor.rank)),
            guarantees_slice_containment=True,
        )

    def output_layout_from_input_layout(
        self,
        input_layout: TensorLayout,
    ) -> TensorLayout:
        output_axis_for_input_axis = {
            input_axis: output_axis
            for output_axis, input_axis in enumerate(self.input_axis_for_output_axis)
        }
        return TensorLayout(
            submesh=input_layout.submesh,
            mesh_x=self._retarget_axis(input_layout.mesh_x, output_axis_for_input_axis),
            mesh_y=self._retarget_axis(input_layout.mesh_y, output_axis_for_input_axis),
            logical_width=input_layout.logical_width,
            logical_height=input_layout.logical_height,
        )

    def input_layout_from_output_layout(
        self,
        output_layout: TensorLayout,
    ) -> TensorLayout:
        mapping = {
            output_axis: input_axis
            for output_axis, input_axis in enumerate(self.input_axis_for_output_axis)
        }
        return TensorLayout(
            submesh=output_layout.submesh,
            mesh_x=self._retarget_axis(output_layout.mesh_x, mapping),
            mesh_y=self._retarget_axis(output_layout.mesh_y, mapping),
            logical_width=output_layout.logical_width,
            logical_height=output_layout.logical_height,
        )

    @staticmethod
    def _retarget_axis(axis: LayoutAxis, mapping: dict[int, int]) -> LayoutAxis:
        if axis.tensor_axis is None:
            return axis
        if axis.tensor_axis not in mapping:
            raise ValueError(
                f"layout relation does not map sharded tensor axis {axis.tensor_axis}"
            )
        return LayoutAxis(mode=axis.mode, tensor_axis=mapping[axis.tensor_axis])


def payload_layout_relations(payload: object) -> tuple[LayoutRelation, ...]:
    """Return one payload's declared relations, conservatively defaulting empty."""

    relations = getattr(payload, "layout_relations", ())
    return tuple(relations)


def find_layout_relation(
    payload: object,
    *,
    input_index: int,
    output_index: int,
) -> LayoutRelation | None:
    """Find one unambiguous indexed relation."""

    matches = tuple(
        relation
        for relation in payload_layout_relations(payload)
        if relation.input_index == input_index
        and relation.output_index == output_index
    )
    if len(matches) > 1:
        raise ValueError(
            f"payload declares duplicate layout relation for input {input_index}, "
            f"output {output_index}"
        )
    return matches[0] if matches else None
