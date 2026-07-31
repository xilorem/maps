from typing import ClassVar

import pytest

from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.cost import OpCostModel
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.ops.defs.elementwise import UnaryElementwisePayload
from MAPS.planner.passes.stage_selection import form_stages
from MAPS.planner.passes.workload_balancing import balance_workload
from MAPS.planner.workload.allocation import grow_tile_count_for_stage
from MAPS.planner.workload.metrics import (
    estimate_selection_metrics,
    virtual_communication_cycles,
)
from MAPS.planner.workload.plans import best_plan_for_stage, plan_all_stages
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


def test_virtual_traffic_charges_inter_stage_writes_to_producer_tiles() -> None:
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
    plans = plan_all_stages(
        stage_selection,
        mesh,
        tile_counts={0: 1, 1: 1},
        initializer_tensors=frozenset(),
        debug=False,
    )

    communication = virtual_communication_cycles(graph, mesh, plans)

    assert communication[0][0] > communication[1][0]


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

    def build_tile_work(self, output_layouts, tile):
        type(self).build_calls += 1
        return super().build_tile_work(output_layouts, tile)


class _PlateauCostModel(OpCostModel):
    def __init__(self, unsharded_cycles: int, sharded_cycles: int) -> None:
        self._unsharded_cycles = unsharded_cycles
        self._sharded_cycles = sharded_cycles

    def cost(self, tile_work, tile) -> int:
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


def test_balance_workload_reuses_candidates_across_growth_probes() -> None:
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

    plans = balance_workload(graph, mesh, {0: (node,)})

    assert plans[0].tile_count == 2
    assert _CountingUnaryPayload.build_calls == 5


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


def test_best_stage_plan_uses_l1_feasible_logical_shape_for_fixed_tile_count() -> None:
    node = _gemm_node("gemm", m=4, k=16, n=7)
    mesh = _mesh_with_l1(6, 1, l1_size=32768)

    plan = best_plan_for_stage(
        stage_nodes=(node,),
        mesh=mesh,
        stage_id=0,
        tile_count=6,
        initializer_tensors=frozenset(),
        debug=False,
    )

    assert plan.tile_count == 6
    assert plan.logical_shape[0] * plan.logical_shape[1] == 6


def test_growth_prefers_tile_count_with_more_physical_shape_options() -> None:
    node = _gemm_node("gemm", m=32, k=32, n=32)
    mesh = _mesh_with_l1(4, 4, l1_size=32768)
    stage_selection = {0: (node,)}
    current_plan = best_plan_for_stage(
        stage_nodes=(node,),
        mesh=mesh,
        stage_id=0,
        tile_count=2,
        initializer_tensors=frozenset(),
        debug=False,
    )
    graph = Graph(
        name="growth",
        tensors=node.inputs + node.outputs,
        nodes=(node,),
    )
    current_metric = estimate_selection_metrics(
        {0: current_plan},
        stage_selection,
        mesh=mesh,
        compute_weight=1.0,
        communication_weight=1.0,
        graph=graph,
    )[0]

    best_growth = grow_tile_count_for_stage(
        stage_id=0,
        stage_selection=stage_selection,
        mesh=mesh,
        tile_counts={0: 2},
        used_tiles=2,
        current_metric=current_metric,
        initializer_tensors=frozenset(),
        graph=graph,
        debug=False,
    )

    assert best_growth is not None
    assert best_growth > 2


def test_growth_prunes_stage_when_doubling_current_count_does_not_improve(
    capsys,
) -> None:
    no_scale_input = Tensor("no_scale_input", 1, (8,), 2, is_initializer=True)
    no_scale_output = Tensor("no_scale_output", 1, (8,), 2)
    no_scale = Node(
        "no_scale",
        OpKind.ELEMENTWISE,
        inputs=(no_scale_input,),
        outputs=(no_scale_output,),
        payload=_PlateauCostPayload(
            "Relu",
            no_scale_input,
            no_scale_output,
            unsharded_cycles=30,
            sharded_cycles=30,
        ),
    )
    scales_input = Tensor("scales_input", 1, (8,), 2, is_initializer=True)
    scales_output = Tensor("scales_output", 1, (8,), 2)
    scales = Node(
        "scales",
        OpKind.ELEMENTWISE,
        inputs=(scales_input,),
        outputs=(scales_output,),
        payload=_PlateauCostPayload(
            "Relu",
            scales_input,
            scales_output,
            unsharded_cycles=20,
            sharded_cycles=10,
        ),
    )
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


def test_best_stage_plan_rejects_tile_work_that_does_not_fit() -> None:
    node = _batched_gemm_node("batched", b=4, m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_size=64)

    try:
        best_plan_for_stage(
            stage_nodes=(node,),
            mesh=mesh,
            stage_id=0,
            tile_count=1,
            initializer_tensors=frozenset(),
            debug=False,
        )
    except ValueError as exc:
        assert "local stage layouts and permanent L1 allocation" in str(exc)
    else:
        raise AssertionError("expected tile-work L1 fit failure")


def test_best_stage_plan_counts_outputs_in_l1_fit() -> None:
    node = _gemm_node("gemm", m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_size=80)

    try:
        best_plan_for_stage(
            stage_nodes=(node,),
            mesh=mesh,
            stage_id=0,
            tile_count=1,
            initializer_tensors=frozenset(),
            debug=False,
        )
    except ValueError as exc:
        assert "local stage layouts and permanent L1 allocation" in str(exc)
    else:
        raise AssertionError("expected output slice to be included in L1 fit")


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

    with pytest.raises(ValueError, match="permanent L1 allocation"):
        best_plan_for_stage(
            stage_nodes=nodes,
            mesh=mesh,
            stage_id=0,
            tile_count=1,
            initializer_tensors=frozenset(),
            num_token_slots=2,
        )


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
