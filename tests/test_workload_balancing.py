from typing import ClassVar

import pytest

from MAPS.arch import L1Memory, L2Memory, Mesh, Tile
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.layout import TensorLayout
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.ops.defs.elementwise import ElementwiseTileWork, UnaryElementwisePayload
from MAPS.planner.passes.stage_selection import form_stages
from MAPS.planner.passes.workload_balancing import balance_workload
from MAPS.planner.workload import allocation as allocation_module
from MAPS.planner.workload.candidates import (
    StageCandidate,
    StageCandidateAnalyzer,
)
from MAPS.planner.workload.metrics import (
    SelectionEvaluation,
    StageMetricBreakdown,
)
from MAPS.planner.workload.memory import permanent_l1_allocation_for_tile
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _gemm_node(name: str, m: int, k: int, n: int) -> Node:
    x = Tensor(name=f"{name}_x", rank=2, dims=(m, k), elem_bytes=2)
    w = Tensor(name=f"{name}_w", rank=2, dims=(k, n), elem_bytes=2)
    out = Tensor(name=f"{name}_out", rank=2, dims=(m, n), elem_bytes=2)
    op = GemmPayload(x=x, w=w, y=None, output=out)
    return Node(
        name=name,
        kind=OpKind.GEMM,
        inputs=(x, w),
        outputs=(out,),
        payload=op,
    )


def _batched_gemm_node(name: str, b: int, m: int, k: int, n: int) -> Node:
    x = Tensor(name=f"{name}_x", rank=3, dims=(b, m, k), elem_bytes=2)
    w = Tensor(name=f"{name}_w", rank=3, dims=(b, k, n), elem_bytes=2)
    out = Tensor(name=f"{name}_out", rank=3, dims=(b, m, n), elem_bytes=2)
    op = GemmPayload(x=x, w=w, y=None, output=out)
    return Node(
        name=name,
        kind=OpKind.GEMM,
        inputs=(x, w),
        outputs=(out,),
        payload=op,
    )


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height, memory=L1Memory(size=l1_size, bandwidth=1)),
    )


def test_balance_workload_uses_full_tile_budget() -> None:
    node0 = _gemm_node("gemm0", m=16, k=16, n=16)
    node1 = _gemm_node("gemm1", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node0, node1))
    mesh = _mesh_with_l1(4, 4, l1_size=4096)

    allocation = {
        stage_id: plan.tile_count
        for stage_id, plan in balance_workload(graph, mesh, form_stages(graph)).items()
    }

    assert allocation == {0: 8, 1: 8}
    assert sum(allocation.values()) == mesh.num_tiles


def test_balance_workload_gives_more_tiles_to_heavier_gemm() -> None:
    heavy = _gemm_node("heavy", m=64, k=64, n=64)
    light = _gemm_node("light", m=8, k=8, n=8)
    graph = Graph(name="g", nodes=(heavy, light))
    mesh = _mesh_with_l1(3, 2, l1_size=32768)

    allocation = {
        stage_id: plan.tile_count
        for stage_id, plan in balance_workload(graph, mesh, form_stages(graph)).items()
    }

    assert allocation[0] > allocation[1]
    assert sum(allocation.values()) == mesh.num_tiles


def test_workload_diagnostics_charge_inter_stage_writes_to_producer_tiles(
    capsys,
) -> None:
    producer = _gemm_node("producer", m=8, k=8, n=8)
    consumer = _gemm_node("consumer", m=8, k=8, n=8)
    consumer = Node(
        name=consumer.name,
        kind=consumer.kind,
        inputs=(producer.outputs[0], consumer.inputs[1]),
        outputs=consumer.outputs,
        payload=GemmPayload(
            x=producer.outputs[0],
            w=consumer.inputs[1],
            y=None,
            output=consumer.outputs[0],
        ),
    )
    graph = Graph(
        name="g",
        tensors=tuple(
            dict.fromkeys(
                producer.inputs
                + producer.outputs
                + consumer.inputs
                + consumer.outputs
            )
        ),
        nodes=(producer, consumer),
    )
    mesh = _mesh_with_l1(2, 1, l1_size=4096)
    stage_selection = {0: (producer,), 1: (consumer,)}
    balance_workload(graph, mesh, stage_selection, debug=True)

    output = capsys.readouterr().out
    assert "stage=0 nodes=producer compute=512 communication=128" in output
    assert "stage=1 nodes=consumer compute=512 communication=0" in output


def test_workload_diagnostics_include_graph_input_and_output_transitions(
    capsys,
) -> None:
    node = _gemm_node("gemm", m=8, k=8, n=8)
    graph = Graph(
        name="graph_io",
        tensors=node.inputs + node.outputs,
        nodes=(node,),
        inputs=node.inputs,
        outputs=node.outputs,
    )
    mesh = _mesh_with_l1(1, 1, l1_size=4096)

    balance_workload(graph, mesh, {0: (node,)}, debug=True)

    output = capsys.readouterr().out
    assert "stage=0 nodes=gemm compute=512 communication=384" in output


def test_balance_workload_preserves_layout_decisions() -> None:
    node = _gemm_node("gemm", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node,))
    mesh = _mesh_with_l1(4, 1, l1_size=32768)

    plans = balance_workload(graph, mesh, form_stages(graph))

    assert plans[0].tile_count == 4
    assert plans[0].logical_shape[0] * plans[0].logical_shape[1] == 4
    assert plans[0].node_output_layouts[-1][0].logical_width == plans[0].logical_shape[0]
    assert plans[0].node_output_layouts[-1][0].logical_height == plans[0].logical_shape[1]


def test_balance_workload_starts_from_minimum_l1_feasible_tile_count() -> None:
    node = _gemm_node("gemm", m=4, k=4, n=4)
    graph = Graph(name="g", nodes=(node,))
    mesh = _mesh_with_l1(2, 1, l1_size=128)

    plans = balance_workload(graph, mesh, form_stages(graph))

    assert plans[0].tile_count == 2


class _CountingUnaryPayload(UnaryElementwisePayload):
    build_calls: ClassVar[int] = 0

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        type(self).build_calls += 1
        return super().build_tile_work(output_layouts, tile)


class _PlateauCostModel(OpCostModel):
    def __init__(self, unsharded_cycles: int, sharded_cycles: int) -> None:
        self._unsharded_cycles = unsharded_cycles
        self._sharded_cycles = sharded_cycles

    def cost(self, tile_work: TileWork, tile: Tile) -> int:
        del tile
        if tile_work.output_slices[0].tensor_slice.num_elements == 8:
            return self._unsharded_cycles
        return self._sharded_cycles


class _PlateauCostPayload(UnaryElementwisePayload):
    def __init__(
        self,
        op_name: str,
        x: Tensor,
        output: Tensor,
        unsharded_cycles: int,
        sharded_cycles: int,
    ) -> None:
        super().__init__(op_name, x, output)
        object.__setattr__(self, "_unsharded_cycles", unsharded_cycles)
        object.__setattr__(self, "_sharded_cycles", sharded_cycles)

    @property
    def cost_model(self) -> OpCostModel:
        return _PlateauCostModel(
            self._unsharded_cycles,
            self._sharded_cycles,
        )


def _plateau_node(
    name: str,
    unsharded_cycles: int,
    sharded_cycles: int,
) -> tuple[Node, Tensor]:
    input_tensor = Tensor(f"{name}_input", 1, (8,), 2, is_initializer=True)
    output = Tensor(f"{name}_output", 1, (8,), 2)
    return (
        Node(
            name,
            OpKind.ELEMENTWISE,
            inputs=(input_tensor,),
            outputs=(output,),
            payload=_PlateauCostPayload(
                "Relu",
                input_tensor,
                output,
                unsharded_cycles,
                sharded_cycles,
            ),
        ),
        input_tensor,
    )


@pytest.mark.parametrize("debug", (False, True))
def test_balance_workload_reuses_candidates_across_growth_probes(
    debug: bool,
    capsys,
) -> None:
    _CountingUnaryPayload.build_calls = 0
    input_tensor = Tensor("input", 1, (8,), 2, is_initializer=True)
    output = Tensor("output", 1, (8,), 2)
    node = Node(
        "stage",
        OpKind.ELEMENTWISE,
        inputs=(input_tensor,),
        outputs=(output,),
        payload=_CountingUnaryPayload("Relu", input_tensor, output),
    )
    graph = Graph(
        "candidate_reuse",
        tensors=(input_tensor, output),
        nodes=(node,),
        initializers=(input_tensor,),
    )
    mesh = _mesh_with_l1(2, 1, l1_size=4096)

    plans = balance_workload(graph, mesh, {0: (node,)}, debug=debug)
    capsys.readouterr()

    assert plans[0].tile_count == 2
    assert _CountingUnaryPayload.build_calls == 5


def test_balance_workload_rejects_growth_that_worsens_global_objective(
    monkeypatch,
    capsys,
) -> None:
    first, first_initializer = _plateau_node("first", 10, 5)
    second, second_initializer = _plateau_node("second", 9, 8)
    graph = Graph(
        "global_objective",
        nodes=(first, second),
        initializers=(first_initializer, second_initializer),
    )
    mesh = _mesh_with_l1(3, 1, l1_size=4096)

    def evaluate_selection(
        candidates: dict[int, StageCandidate],
        **kwargs: object,
    ) -> SelectionEvaluation:
        del kwargs
        tile_counts = tuple(
            candidates[stage_id].plan.tile_count
            for stage_id in sorted(candidates)
        )
        metrics = {
            (1, 1): {0: 10.0, 1: 9.0},
            (2, 1): {0: 5.0, 1: 20.0},
            (1, 2): {0: 10.0, 1: 8.0},
        }[tile_counts]
        return SelectionEvaluation(
            {
                stage_id: StageMetricBreakdown(
                    compute_cycles=1000 + stage_id * 100 + tile_counts[stage_id],
                    communication_cycles=2000 + stage_id * 100 + tile_counts[stage_id],
                    weighted_bottleneck=metric,
                )
                for stage_id, metric in metrics.items()
            }
        )

    monkeypatch.setattr(
        allocation_module,
        "evaluate_candidate_selection",
        evaluate_selection,
    )

    plans = balance_workload(
        graph,
        mesh,
        {0: (first,), 1: (second,)},
        debug=True,
    )

    output = capsys.readouterr().out
    assert {stage_id: plan.tile_count for stage_id, plan in plans.items()} == {
        0: 1,
        1: 2,
    }
    assert "stage=0 nodes=first compute=1001 communication=2001" in output
    assert "stage=1 nodes=second compute=1102 communication=2102" in output


def test_planner_selected_token_slots_control_l1_feasibility() -> None:
    node = _gemm_node("gemm", m=4, k=4, n=4)
    graph = Graph(name="g", nodes=(node,))
    mesh = _mesh_with_l1(2, 1, l1_size=80)
    selection = form_stages(graph)

    plans = balance_workload(
        graph,
        mesh,
        selection,
        num_token_slots=1,
    )

    assert plans[0].tile_count == 2
    with pytest.raises(ValueError, match="has no L1-feasible layout"):
        balance_workload(
            graph,
            mesh,
            selection,
            num_token_slots=2,
        )


def test_balance_workload_accepts_explicit_stage_selection() -> None:
    node0 = _gemm_node("gemm0", m=16, k=16, n=16)
    node1 = _gemm_node("gemm1", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node0, node1))
    mesh = _mesh_with_l1(2, 2, l1_size=4096)

    plans = balance_workload(
        graph,
        mesh,
        stage_selection={0: (node0, node1)},
    )

    assert tuple(plans) == (0,)
    assert plans[0].tile_count == mesh.num_tiles
    assert plans[0].nodes == (node0, node1)
    assert len(plans[0].node_output_layouts) == 2


def _elementwise_chain() -> tuple[Graph, Node, Node]:
    input_tensor = Tensor("input", 1, (8,), 2)
    intermediate = Tensor("intermediate", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(input_tensor,),
        outputs=(intermediate,),
        payload=UnaryElementwisePayload("Relu", input_tensor, intermediate),
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(intermediate,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Neg", intermediate, output),
    )
    return (
        Graph(
            "elementwise_chain",
            tensors=(input_tensor, intermediate, output),
            nodes=(first, second),
            inputs=(input_tensor,),
            outputs=(output,),
        ),
        first,
        second,
    )


def test_balance_workload_accepts_automatically_formed_compatible_stage() -> None:
    graph, first, second = _elementwise_chain()
    mesh = _mesh_with_l1(2, 1, l1_size=4096)

    plans = balance_workload(graph, mesh, form_stages(graph))

    assert tuple(plans) == (0,)
    assert plans[0].nodes == (first, second)


def test_balance_workload_accepts_caller_supplied_compatible_stage() -> None:
    graph, first, second = _elementwise_chain()
    mesh = _mesh_with_l1(2, 1, l1_size=4096)

    plans = balance_workload(graph, mesh, {7: (first, second)})

    assert tuple(plans) == (7,)
    assert plans[7].nodes == (first, second)


def test_balance_workload_rejects_caller_supplied_incompatible_internal_edge() -> None:
    first = _gemm_node("first", m=4, k=4, n=4)
    second_weight = Tensor("second_weight", 2, (4, 4), 2)
    output = Tensor("output", 2, (4, 4), 2)
    second = Node(
        "second",
        OpKind.GEMM,
        inputs=(first.outputs[0], second_weight),
        outputs=(output,),
        payload=GemmPayload(first.outputs[0], second_weight, None, output),
    )
    graph = Graph(
        "incompatible_internal_edge",
        nodes=(first, second),
        initializers=(first.inputs[1], second_weight),
    )
    mesh = _mesh_with_l1(2, 1, l1_size=4096)

    with pytest.raises(
        ValueError,
        match=r"stage 0 has incompatible internal dependency first->second",
    ):
        balance_workload(graph, mesh, {0: (first, second)})


def test_balance_workload_rejects_caller_split_explicit_stage_group() -> None:
    graph, first, second = _elementwise_chain()
    first = Node(
        first.name,
        first.kind,
        inputs=first.inputs,
        outputs=first.outputs,
        payload=first.payload,
        attributes={"stage_group_id": "together"},
    )
    second = Node(
        second.name,
        second.kind,
        inputs=second.inputs,
        outputs=second.outputs,
        payload=second.payload,
        attributes={"stage_group_id": "together"},
    )
    graph = Graph(
        graph.name,
        tensors=graph.tensors,
        nodes=(first, second),
        inputs=graph.inputs,
        outputs=graph.outputs,
    )
    mesh = _mesh_with_l1(2, 1, l1_size=4096)

    with pytest.raises(
        ValueError,
        match=r"explicit stage group 'together' is split across stages 0 and 1",
    ):
        balance_workload(graph, mesh, {0: (first,), 1: (second,)})


class _FailingLayoutUnaryPayload(UnaryElementwisePayload):
    def output_layouts(self, submesh, logical_shape=None):
        raise ValueError("concrete layout invariant failed")


def test_balance_workload_propagates_concrete_layout_invariant_failure() -> None:
    graph, first, second = _elementwise_chain()
    failing_second = Node(
        second.name,
        second.kind,
        inputs=second.inputs,
        outputs=second.outputs,
        payload=_FailingLayoutUnaryPayload(
            "Neg",
            second.inputs[0],
            second.outputs[0],
        ),
    )
    graph = Graph(
        graph.name,
        tensors=graph.tensors,
        nodes=(first, failing_second),
        inputs=graph.inputs,
        outputs=graph.outputs,
    )
    mesh = _mesh_with_l1(2, 1, l1_size=4096)

    with pytest.raises(ValueError, match="concrete layout invariant failed"):
        balance_workload(graph, mesh, {0: (first, failing_second)})


def test_balance_workload_can_use_selected_stage_groups() -> None:
    node0 = _gemm_node("gemm0", m=16, k=16, n=16)
    node1 = _gemm_node(
        "gemm1",
        m=16,
        k=16,
        n=16,
    )
    node1 = Node(
        name=node1.name,
        kind=node1.kind,
        inputs=node1.inputs,
        outputs=node1.outputs,
        payload=node1.payload,
        attributes={"stage_group_id": "group0"},
    )
    node2 = _gemm_node("gemm2", m=16, k=16, n=16)
    node0 = Node(
        name=node0.name,
        kind=node0.kind,
        inputs=node0.inputs,
        outputs=node0.outputs,
        payload=node0.payload,
        attributes={"stage_group_id": "group0"},
    )
    graph = Graph(name="g", nodes=(node0, node1, node2))
    mesh = _mesh_with_l1(3, 2, l1_size=4096)

    del mesh
    try:
        form_stages(graph)
    except ValueError as exc:
        assert "dependency-connected" in str(exc)
    else:
        raise AssertionError("expected malformed explicit group to fail")


def test_stage_candidate_uses_l1_feasible_logical_shape_for_fixed_tile_count() -> None:
    node = _gemm_node("gemm", m=4, k=16, n=7)
    mesh = _mesh_with_l1(6, 1, l1_size=32768)

    candidate = StageCandidateAnalyzer(
        {0: (node,)},
        mesh,
        frozenset(),
    ).candidate(0, 6)

    assert candidate is not None
    assert candidate.plan.tile_count == 6
    assert candidate.plan.logical_shape[0] * candidate.plan.logical_shape[1] == 6


def test_balance_workload_grows_an_improving_stage_past_two_tiles() -> None:
    node = _gemm_node("gemm", m=32, k=32, n=32)
    graph = Graph(
        name="growth",
        tensors=node.inputs + node.outputs,
        nodes=(node,),
    )
    mesh = _mesh_with_l1(4, 4, l1_size=32768)

    plans = balance_workload(graph, mesh, {0: (node,)})

    assert plans[0].tile_count > 2


def test_growth_prunes_stage_when_doubling_current_count_does_not_improve(
    capsys,
) -> None:
    no_scale, no_scale_input = _plateau_node("no_scale", 30, 30)
    scales, scales_input = _plateau_node("scales", 20, 10)
    graph = Graph(
        "growth",
        nodes=(no_scale, scales),
        initializers=(no_scale_input, scales_input),
    )
    mesh = _mesh_with_l1(5, 1, l1_size=4096)

    plans = balance_workload(
        graph,
        mesh,
        {0: (no_scale,), 1: (scales,)},
        debug=True,
    )

    output = capsys.readouterr().out
    assert {stage_id: plan.tile_count for stage_id, plan in plans.items()} == {
        0: 1,
        1: 2,
    }
    assert "stage=0 doubled_current_tile_count=2 no_improvement prune_stage" in output
    assert "stage=1 doubled_current_tile_count=4 no_improvement prune_stage" in output


def test_stage_candidate_rejects_tile_work_that_does_not_fit() -> None:
    node = _batched_gemm_node("batched", b=4, m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_size=64)

    candidate = StageCandidateAnalyzer(
        {0: (node,)},
        mesh,
        frozenset(),
    ).candidate(0, 1)

    assert candidate is None


def test_stage_candidate_counts_outputs_in_l1_fit() -> None:
    node = _gemm_node("gemm", m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_size=80)

    candidate = StageCandidateAnalyzer(
        {0: (node,)},
        mesh,
        frozenset(),
    ).candidate(0, 1)

    assert candidate is None


def test_stage_l1_accounting_keeps_every_backend_allocation() -> None:
    x = Tensor("x", 1, (8,), 2)
    a = Tensor("a", 1, (8,), 2)
    early_output = Tensor("early_output", 1, (8,), 2)
    c = Tensor("c", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    nodes = (
        Node(
            "produce_a",
            OpKind.ELEMENTWISE,
            (x,),
            (a,),
            UnaryElementwisePayload("Relu", x, a),
        ),
        Node(
            "early_branch",
            OpKind.ELEMENTWISE,
            (a,),
            (early_output,),
            UnaryElementwisePayload("Exp", a, early_output),
        ),
        Node(
            "late_branch",
            OpKind.ELEMENTWISE,
            (a,),
            (c,),
            UnaryElementwisePayload("Neg", a, c),
        ),
        Node(
            "finish",
            OpKind.ELEMENTWISE,
            (c,),
            (output,),
            UnaryElementwisePayload("Sqrt", c, output),
        ),
    )
    mesh = _mesh_with_l1(1, 1, l1_size=4096)
    submesh = Submesh(mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    layouts = tuple(
        node.payload.output_layouts(submesh, logical_shape=(1, 1))
        for node in nodes
    )

    assert permanent_l1_allocation_for_tile(
        nodes,
        layouts,
        submesh.tiles[0],
        frozenset(),
    ) == 160


def test_l1_occupancy_token_buffers_runtime_tensors_but_not_initializers() -> None:
    weight = Tensor("weight", 1, (8,), 2, is_initializer=True)
    output = Tensor("output", 1, (8,), 2)
    node = Node(
        "op",
        OpKind.ELEMENTWISE,
        (weight,),
        (output,),
        UnaryElementwisePayload("Exp", weight, output),
    )
    mesh = _mesh_with_l1(1, 1, l1_size=4096)
    submesh = Submesh(mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    layouts = node.payload.output_layouts(submesh, logical_shape=(1, 1))

    assert permanent_l1_allocation_for_tile(
        (node,),
        (layouts,),
        submesh.tiles[0],
        frozenset((weight,)),
        num_token_slots=3,
    ) == 64


def test_permanent_l1_allocation_reproduces_tile_245_overflow() -> None:
    x = Tensor("x", 1, (71_552,), 2)
    intermediate = Tensor("intermediate", 1, (71_552,), 2)
    output = Tensor("output", 1, (71_552,), 2)
    nodes = (
        Node(
            "first",
            OpKind.ELEMENTWISE,
            (x,),
            (intermediate,),
            UnaryElementwisePayload("Relu", x, intermediate),
        ),
        Node(
            "second",
            OpKind.ELEMENTWISE,
            (intermediate,),
            (output,),
            UnaryElementwisePayload("Exp", intermediate, output),
        ),
    )
    mesh = _mesh_with_l1(1, 1, l1_size=0xD0000)
    submesh = Submesh(mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    layouts = tuple(
        node.payload.output_layouts(submesh, logical_shape=(1, 1))
        for node in nodes
    )

    assert permanent_l1_allocation_for_tile(
        nodes,
        layouts,
        submesh.tiles[0],
        frozenset(),
        num_token_slots=2,
    ) == 858_624

    assert StageCandidateAnalyzer(
        {0: nodes},
        mesh,
        frozenset(),
        num_token_slots=2,
    ).candidate(0, 1) is None


def test_permanent_l1_allocation_aligns_each_slice_start() -> None:
    x = Tensor("x", 1, (1,), 1)
    intermediate = Tensor("intermediate", 1, (1,), 1)
    output = Tensor("output", 1, (1,), 1)
    nodes = (
        Node(
            "first",
            OpKind.ELEMENTWISE,
            (x,),
            (intermediate,),
            UnaryElementwisePayload("Relu", x, intermediate),
        ),
        Node(
            "second",
            OpKind.ELEMENTWISE,
            (intermediate,),
            (output,),
            UnaryElementwisePayload("Exp", intermediate, output),
        ),
    )
    mesh = _mesh_with_l1(1, 1, l1_size=4096)
    submesh = Submesh(mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    layouts = tuple(
        node.payload.output_layouts(submesh, logical_shape=(1, 1))
        for node in nodes
    )

    assert permanent_l1_allocation_for_tile(
        nodes,
        layouts,
        submesh.tiles[0],
        frozenset(),
        num_token_slots=1,
    ) == 33
