from dataclasses import dataclass
from typing import ClassVar

from MAPS.arch import L1Memory, L2Memory, Mesh, WorkKind
from MAPS.core.dtype import TensorDType
from MAPS.core.graph import Node, OpKind
from maps.planning.layouts import TensorLayout, tile_tensor_slice
from maps.planning.submesh import Submesh
from MAPS.core.tensor import Tensor
from maps.operations import OpCostModel
from maps.operations import LayoutRelation
from maps.operations import OpPayload, sharded_layout
from maps.operations.elementwise import ElementwiseTileWork, UnaryElementwisePayload
import maps.planning.allocation.candidates as candidates_module
from maps.planning.allocation.candidates import StageCandidateAnalyzer
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _mesh(tile_count: int, l1_size: int = 4096) -> Mesh:
    return Mesh(
        width=tile_count,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(tile_count, 1),
        tiles=rectangular_test_tiles(
            tile_count,
            1,
            memory=L1Memory(size=l1_size, bandwidth=1),
        ),
    )


def _unary_node(name: str, length: int = 8) -> Node:
    x = Tensor(f"{name}_input", 1, (length,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor(
        f"{name}_output", 1, (length,), 2, dtype=TensorDType.FLOAT16
    )
    return Node(
        name,
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Relu", x, output),
    )


@dataclass(frozen=True)
class _FixedCostModel(OpCostModel):
    tile_cycles: tuple[int, ...]
    placement_cycles: int

    def cost(self, tile_work, tile, assigned_device) -> int:
        del tile_work, assigned_device
        return self.tile_cycles[tile.tile_id]

    def placement_cost(self, *, node, output_layouts) -> int:
        del node, output_layouts
        return self.placement_cycles


@dataclass(frozen=True)
class _FixedCostPayload(OpPayload):
    x: Tensor
    output: Tensor
    tile_cycles: tuple[int, ...]
    placement_cycles: int
    work_kind: WorkKind = WorkKind.RELU

    @property
    def layout_relations(self) -> tuple[LayoutRelation, ...]:
        return (
            LayoutRelation.exact(
                input_index=0,
                output_index=0,
                tensor=self.x,
            ),
        )

    @property
    def cost_model(self) -> OpCostModel:
        return _FixedCostModel(self.tile_cycles, self.placement_cycles)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile,
    ) -> ElementwiseTileWork:
        output_layout = self.single_output_layout(output_layouts)
        output_slice = tile_tensor_slice(self.output, output_layout, tile)
        return ElementwiseTileWork(
            work_kind=WorkKind.RELU,
            output=self.output,
            output_slice=output_slice,
            inputs=(self.x,),
            input_tile_slices=(output_slice,),
        )


class _CountingUnaryPayload(UnaryElementwisePayload):
    build_calls: ClassVar[int] = 0

    def build_tile_work(self, output_layouts, tile):
        type(self).build_calls += 1
        return super().build_tile_work(output_layouts, tile)


@dataclass(frozen=True)
class _InverseSliceCostModel(OpCostModel):
    def cost(self, tile_work, tile, assigned_device) -> int:
        del tile, assigned_device
        return 100 // tile_work.output_slices[0].tensor_slice.num_elements


class _InverseSliceCostPayload(UnaryElementwisePayload):
    @property
    def cost_model(self) -> OpCostModel:
        return _InverseSliceCostModel()


def test_stage_candidate_contains_immutable_per_tile_intrinsic_facts() -> None:
    node = _unary_node("stage")
    analyzer = StageCandidateAnalyzer(
        stage_formation={7: (node,)},
        mesh=_mesh(2),
        initializer_tensors=frozenset(),
        num_token_slots=2,
    )

    candidate = analyzer.candidate(stage_id=7, tile_count=2)

    assert candidate is not None
    assert candidate.plan.stage_id == 7
    assert candidate.plan.tile_count == 2
    assert candidate.plan.logical_shape == (2, 1)
    assert candidate.plan.device_names == ("core",)
    assert tuple(
        (fact.tile_id, fact.compute_cycles, fact.permanent_l1_bytes)
        for fact in candidate.tile_facts
    ) == ((0, 4, 32), (1, 4, 32))
    assert candidate.stage_compute == 4


def test_stage_candidate_resolves_device_assignment_once(monkeypatch) -> None:
    node = _unary_node("stage")
    assignment_calls = 0
    assigned_device_name = candidates_module.assigned_device_name

    def count_assignment(*args, **kwargs):
        nonlocal assignment_calls
        assignment_calls += 1
        return assigned_device_name(*args, **kwargs)

    monkeypatch.setattr(candidates_module, "assigned_device_name", count_assignment)
    analyzer = StageCandidateAnalyzer(
        stage_formation={0: (node,)},
        mesh=_mesh(2),
        initializer_tensors=frozenset(),
    )

    assert analyzer.candidate(stage_id=0, tile_count=2) is not None
    assert assignment_calls == 1


def test_stage_compute_accumulates_layers_per_tile_before_finding_peak() -> None:
    x = Tensor("x", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    intermediate = Tensor(
        "intermediate", 1, (8,), 2, dtype=TensorDType.FLOAT16
    )
    output = Tensor("output", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    first = Node(
        "first",
        OpKind.CUSTOM,
        inputs=(x,),
        outputs=(intermediate,),
        payload=_FixedCostPayload(x, intermediate, (9, 1), 3),
    )
    second = Node(
        "second",
        OpKind.CUSTOM,
        inputs=(intermediate,),
        outputs=(output,),
        payload=_FixedCostPayload(intermediate, output, (1, 9), 5),
    )
    analyzer = StageCandidateAnalyzer(
        {0: (first, second)},
        _mesh(2),
        frozenset(),
    )

    candidate = analyzer.candidate(0, 2)

    assert candidate is not None
    assert tuple(fact.compute_cycles for fact in candidate.tile_facts) == (18, 18)
    assert candidate.stage_compute == 18


def test_equal_compute_shapes_prefer_smaller_logical_height() -> None:
    x = Tensor("input", 2, (8, 8), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, (8, 8), 2, dtype=TensorDType.FLOAT16)
    node = Node(
        "stage",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Relu", x, output),
    )
    analyzer = StageCandidateAnalyzer(
        {0: (node,)},
        _mesh(4),
        frozenset(),
    )

    candidate = analyzer.candidate(0, 4)

    assert candidate is not None
    assert candidate.plan.logical_shape == (4, 1)


def test_l1_infeasible_shape_is_skipped_for_a_feasible_alternative() -> None:
    x = Tensor("input", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    node = Node(
        "stage",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=_InverseSliceCostPayload("Relu", x, output),
    )
    analyzer = StageCandidateAnalyzer(
        {0: (node,)},
        _mesh(2, l1_size=48),
        frozenset(),
    )

    candidate = analyzer.candidate(0, 2)

    assert candidate is not None
    assert candidate.plan.logical_shape == (2, 1)
    assert candidate.stage_compute == 25


def test_stage_candidate_cache_reuses_successful_analysis() -> None:
    _CountingUnaryPayload.build_calls = 0
    x = Tensor("input", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    node = Node(
        "stage",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=_CountingUnaryPayload("Relu", x, output),
    )
    analyzer = StageCandidateAnalyzer(
        {0: (node,)},
        _mesh(2),
        frozenset(),
    )

    first = analyzer.candidate(0, 2)
    second = analyzer.candidate(0, 2)

    assert first is second
    assert _CountingUnaryPayload.build_calls == 4


def test_stage_candidate_cache_reuses_l1_infeasible_miss() -> None:
    _CountingUnaryPayload.build_calls = 0
    x = Tensor("input", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    node = Node(
        "stage",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(output,),
        payload=_CountingUnaryPayload("Relu", x, output),
    )
    analyzer = StageCandidateAnalyzer(
        {0: (node,)},
        _mesh(1, l1_size=1),
        frozenset(),
    )

    assert analyzer.candidate(0, 1) is None
    assert analyzer.candidate(0, 1) is None
    assert _CountingUnaryPayload.build_calls == 1
