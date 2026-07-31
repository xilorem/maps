from dataclasses import dataclass

import pytest

from MAPS.arch import Tile
from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
    TensorSliceRef,
    tile_tensor_slice,
)
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.hw.chips import magia_mesh
from MAPS.ops.common import OperationPayload
from MAPS.planner.contracts.stages import StagePlacement, StagePlan
from MAPS.planner.spatial.traffic import build_virtual_traffic
from MAPS.transitions import (
    InputTransition,
    IntermediateTransition,
    OutputTransition,
    VirtualInputTransition,
    VirtualIntermediateTransition,
    VirtualOutputTransition,
    bind_transitions,
    build_virtual_transitions,
)


@dataclass(frozen=True)
class _TileWork:
    input_slices: tuple[TensorSliceRef, ...]


@dataclass(frozen=True)
class _Payload(OperationPayload):
    inputs: tuple[Tensor, ...]
    required_slices: dict[int, tuple[TensorSlice, ...]]

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> _TileWork:
        del output_layouts
        return _TileWork(
            input_slices=tuple(
                TensorSliceRef(tensor=tensor, tensor_slice=tensor_slice)
                for tensor, tensor_slice in zip(
                    self.inputs,
                    self.required_slices.get(tile.tile_id, ()),
                )
            )
        )


def _layout(submesh: Submesh, axis: int | None = 0) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(
            mode=LayoutAxisMode.REPLICATE if axis is None else LayoutAxisMode.SHARD,
            tensor_axis=axis,
        ),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )


def _node(
    name: str,
    inputs: tuple[Tensor, ...],
    output: Tensor,
    required_slices: dict[int, tuple[TensorSlice, ...]],
) -> Node:
    return Node(
        name=name,
        kind=OpKind.CUSTOM,
        inputs=inputs,
        outputs=(output,),
        payload=_Payload(inputs=inputs, required_slices=required_slices),
    )


def _plan(
    stage_id: int,
    nodes: tuple[Node, ...],
    layouts: tuple[tuple[TensorLayout, ...], ...],
) -> StagePlan:
    return StagePlan(
        stage_id=stage_id,
        tile_count=layouts[0][0].submesh.num_tiles,
        logical_shape=(layouts[0][0].submesh.num_tiles, 1),
        nodes=nodes,
        node_output_layouts=layouts,
        device_names=("core",) * len(nodes),
    )


def test_compiles_input_intermediate_and_output_transitions() -> None:
    mesh = magia_mesh(width=2, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids={0, 1})
    runtime_input = Tensor("input", 1, (4,), 2)
    initializer = Tensor("weight", 1, (4,), 2, is_initializer=True)
    intermediate = Tensor("middle", 1, (4,), 2)
    output = Tensor("output", 1, (4,), 2)
    full = TensorSlice(1, (TensorRange(0, 4),))
    halves = {
        tile.tile_id: (tile_tensor_slice(runtime_input, _layout(virtual), tile), full)
        for tile in virtual.tiles
    }
    producer = _node(
        "producer",
        (runtime_input, initializer),
        intermediate,
        halves,
    )
    consumer = _node(
        "consumer",
        (intermediate,),
        output,
        {
            tile.tile_id: (tile_tensor_slice(intermediate, _layout(virtual), tile),)
            for tile in virtual.tiles
        },
    )
    graph = Graph(
        name="graph",
        tensors=(runtime_input, initializer, intermediate, output),
        nodes=(producer, consumer),
        inputs=(runtime_input, initializer),
        outputs=(output,),
        initializers=(initializer,),
    )
    plans = {
        1: _plan(1, (consumer,), ((_layout(virtual),),)),
        0: _plan(0, (producer,), ((_layout(virtual),),)),
    }

    transitions = build_virtual_transitions(graph, plans)

    assert tuple(type(transition) for transition in transitions) == (
        VirtualInputTransition,
        VirtualIntermediateTransition,
        VirtualOutputTransition,
    )
    input_transition = transitions[0]
    assert isinstance(input_transition, VirtualInputTransition)
    assert input_transition.tensor is runtime_input
    assert input_transition.tensor_id == 0
    assert input_transition.destination_stage_id == 0
    assert input_transition.destination_input_index == 0
    assert tuple(destination.virtual_tile_id for destination in input_transition.destinations) == (0, 1)

    intermediate_transition = transitions[1]
    assert isinstance(intermediate_transition, VirtualIntermediateTransition)
    assert intermediate_transition.source_stage_id == 0
    assert intermediate_transition.source_output_index == 0
    assert intermediate_transition.destination_stage_id == 1
    assert intermediate_transition.destination_input_index == 0
    assert tuple(
        (transfer.source_virtual_tile_id, transfer.destination_virtual_tile_id)
        for transfer in intermediate_transition.transfers
    ) == ((0, 0), (1, 1))

    output_transition = transitions[2]
    assert isinstance(output_transition, VirtualOutputTransition)
    assert output_transition.tensor is output
    assert output_transition.tensor_id == 3
    assert output_transition.source_stage_id == 1
    assert output_transition.source_output_index == 0
    assert tuple(source.virtual_tile_id for source in output_transition.sources) == (0, 1)
    assert all(transition.tensor is not initializer for transition in transitions)

    traffic = build_virtual_traffic(transitions, plans)

    assert traffic.stage_comm == {(0, 1): 8}
    assert traffic.edge_matrices == {(0, 1): {(0, 0): 4, (1, 1): 4}}
    assert traffic.l2_read_weights == {0: {0: 4, 1: 4}, 1: {0: 0, 1: 0}}
    assert traffic.l2_write_weights == {0: {0: 0, 1: 0}, 1: {0: 4, 1: 4}}
    assert traffic.input_weights == {0: {0: 4, 1: 4}, 1: {0: 4, 1: 4}}
    assert traffic.output_weights == {0: {0: 4, 1: 4}, 1: {0: 4, 1: 4}}


def test_compiles_fanout_replication_offsets_and_empty_demand_deterministically() -> None:
    mesh = magia_mesh(width=2, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids={0, 1})
    source = Tensor("source", 1, (4,), 2)
    left_output = Tensor("left", 1, (4,), 2)
    right_output = Tensor("right", 1, (4,), 2)
    empty_output = Tensor("empty", 1, (4,), 2)
    source_node = _node("source", (), source, {})
    replicated_parent = TensorSlice(1, (TensorRange(0, 4),))
    replicated_demand = {
        0: (replicated_parent,),
        1: (replicated_parent,),
    }
    left = _node("left", (source,), left_output, replicated_demand)
    right = _node("right", (source,), right_output, replicated_demand)
    empty = _node("empty", (source,), empty_output, {})
    graph = Graph(
        name="fanout",
        tensors=(source, left_output, right_output, empty_output),
        nodes=(source_node, left, right, empty),
    )
    source_layout = TensorLayout(
        submesh=virtual,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )
    plans = {
        3: _plan(3, (empty,), ((_layout(virtual),),)),
        2: _plan(2, (right,), ((_layout(virtual),),)),
        0: _plan(0, (source_node,), ((source_layout,),)),
        1: _plan(1, (left,), ((_layout(virtual),),)),
    }

    transitions = build_virtual_transitions(graph, plans)

    assert tuple(
        transition.destination_stage_id
        for transition in transitions
        if isinstance(transition, VirtualIntermediateTransition)
    ) == (1, 2, 3)
    left_transition = transitions[0]
    assert isinstance(left_transition, VirtualIntermediateTransition)
    assert len(left_transition.transfers) == 4
    assert tuple(
        (transfer.source_virtual_tile_id, transfer.destination_virtual_tile_id)
        for transfer in left_transition.transfers
    ) == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert left_transition.transfers[0].source_subslice.dims == (
        TensorRange(start=0, length=2),
    )
    assert left_transition.transfers[0].destination_subslice.dims == (
        TensorRange(start=0, length=2),
    )
    assert left_transition.transfers[2].source_subslice.dims == (
        TensorRange(start=0, length=2),
    )
    assert left_transition.transfers[2].destination_subslice.dims == (
        TensorRange(start=2, length=2),
    )
    empty_transition = transitions[2]
    assert isinstance(empty_transition, VirtualIntermediateTransition)
    assert empty_transition.transfers == ()

    traffic = build_virtual_traffic(transitions, plans)

    assert traffic.stage_comm == {(0, 1): 16, (0, 2): 16}
    assert traffic.edge_matrices == {
        (0, 1): {(0, 0): 4, (0, 1): 4, (1, 0): 4, (1, 1): 4},
        (0, 2): {(0, 0): 4, (0, 1): 4, (1, 0): 4, (1, 1): 4},
        (0, 3): {},
    }
    assert traffic.input_weights[1] == {0: 8, 1: 8}
    assert traffic.input_weights[2] == {0: 8, 1: 8}
    assert traffic.input_weights[3] == {0: 0, 1: 0}
    assert traffic.output_weights[0] == {0: 16, 1: 16}


def test_same_stage_dependencies_produce_no_virtual_transition() -> None:
    mesh = magia_mesh(width=1, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})
    middle = Tensor("middle", 1, (4,), 2)
    output = Tensor("output", 1, (4,), 2)
    producer = _node("producer", (), middle, {})
    consumer = _node(
        "consumer",
        (middle,),
        output,
        {0: (TensorSlice(1, (TensorRange(0, 4),)),)},
    )
    graph = Graph(
        name="local",
        tensors=(middle, output),
        nodes=(producer, consumer),
    )
    plans = {
        0: _plan(
            0,
            (producer, consumer),
            ((_layout(virtual),), (_layout(virtual),)),
        )
    }

    assert build_virtual_transitions(graph, plans) == ()


def test_repeated_tensor_inputs_keep_their_positional_demands() -> None:
    mesh = magia_mesh(width=1, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})
    source = Tensor("source", 1, (4,), 2)
    output = Tensor("output", 1, (4,), 2)
    producer = _node("producer", (), source, {})
    first_half = TensorSlice(1, (TensorRange(0, 2),))
    second_half = TensorSlice(1, (TensorRange(2, 2),))
    consumer = _node(
        "consumer",
        (source, source),
        output,
        {0: (first_half, second_half)},
    )
    graph = Graph(
        name="repeated-input",
        tensors=(source, output),
        nodes=(producer, consumer),
    )
    plans = {
        0: _plan(0, (producer,), ((_layout(virtual, axis=None),),)),
        1: _plan(1, (consumer,), ((_layout(virtual),),)),
    }

    transitions = build_virtual_transitions(graph, plans)

    assert tuple(
        transition.destination_input_index
        for transition in transitions
        if isinstance(transition, VirtualIntermediateTransition)
    ) == (0, 1)
    assert isinstance(transitions[0], VirtualIntermediateTransition)
    assert isinstance(transitions[1], VirtualIntermediateTransition)
    assert transitions[0].transfers[0].destination_subslice.parent == first_half
    assert transitions[1].transfers[0].destination_subslice.parent == second_half


def test_input_and_output_transition_order_is_deterministic() -> None:
    mesh = magia_mesh(width=1, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})
    input0 = Tensor("input0", 1, (4,), 2)
    input1 = Tensor("input1", 1, (4,), 2)
    output0 = Tensor("output0", 1, (4,), 2)
    output1 = Tensor("output1", 1, (4,), 2)
    full = TensorSlice(1, (TensorRange(0, 4),))
    stage2 = _node("stage2", (input0,), output0, {0: (full,)})
    stage1 = _node(
        "stage1",
        (input1, input0),
        output1,
        {0: (full, full)},
    )
    graph = Graph(
        name="ordered",
        tensors=(input0, input1, output0, output1),
        nodes=(stage2, stage1),
        inputs=(input0, input1),
        outputs=(output0, output1),
    )
    plans = {
        2: _plan(2, (stage2,), ((_layout(virtual),),)),
        1: _plan(1, (stage1,), ((_layout(virtual),),)),
    }

    transitions = build_virtual_transitions(graph, plans)

    assert tuple(
        (
            transition.destination_stage_id,
            transition.destination_input_index,
        )
        for transition in transitions
        if isinstance(transition, VirtualInputTransition)
    ) == ((1, 0), (1, 1), (2, 0))
    assert tuple(
        transition.tensor
        for transition in transitions
        if isinstance(transition, VirtualOutputTransition)
    ) == (output0, output1)


def test_compilation_rejects_mismatched_slice_ranks() -> None:
    mesh = magia_mesh(width=1, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})
    source = Tensor("source", 1, (4,), 2)
    output = Tensor("output", 1, (4,), 2)
    producer = _node("producer", (), source, {})
    consumer = _node(
        "consumer",
        (source,),
        output,
        {
            0: (
                TensorSlice(
                    2,
                    (TensorRange(0, 2), TensorRange(0, 2)),
                ),
            )
        },
    )
    graph = Graph(
        name="mismatched-ranks",
        tensors=(source, output),
        nodes=(producer, consumer),
    )
    plans = {
        0: _plan(0, (producer,), ((_layout(virtual, axis=None),),)),
        1: _plan(1, (consumer,), ((_layout(virtual),),)),
    }

    with pytest.raises(ValueError, match="different ranks"):
        build_virtual_transitions(graph, plans)


def test_binding_changes_only_tile_endpoints_and_preserves_positions() -> None:
    mesh = magia_mesh(width=4, height=1)
    virtual = Submesh(mesh=mesh, submesh_id=0, tile_ids={0, 1})
    physical0 = Submesh(mesh=mesh, submesh_id=10, tile_ids={2, 3})
    physical1 = Submesh(mesh=mesh, submesh_id=11, tile_ids={0, 1})
    runtime_input = Tensor("input", 1, (4,), 2)
    x = Tensor("x", 1, (4,), 2)
    y = Tensor("y", 1, (4,), 2)
    full = TensorSlice(1, (TensorRange(0, 4),))
    producer = _node(
        "producer",
        (runtime_input,),
        x,
        {0: (full,), 1: (full,)},
    )
    consumer = _node("consumer", (x,), y, {0: (full,), 1: (full,)})
    graph = Graph(
        name="binding",
        tensors=(runtime_input, x, y),
        nodes=(producer, consumer),
        inputs=(runtime_input,),
        outputs=(y,),
    )
    plans = {
        0: _plan(0, (producer,), ((_layout(virtual),),)),
        1: _plan(1, (consumer,), ((_layout(virtual),),)),
    }
    virtual_transitions = build_virtual_transitions(graph, plans)
    placements = {
        0: StagePlacement(0, virtual, physical0, {0: 3, 1: 2}),
        1: StagePlacement(1, virtual, physical1, {0: 1, 1: 0}),
    }

    transitions = bind_transitions(virtual_transitions, placements)

    assert tuple(type(transition) for transition in transitions) == (
        InputTransition,
        IntermediateTransition,
        OutputTransition,
    )
    input_transition = transitions[0]
    assert isinstance(input_transition, InputTransition)
    assert tuple(
        destination.tile_id
        for destination in input_transition.destinations
    ) == (3, 2)
    intermediate = transitions[1]
    assert isinstance(intermediate, IntermediateTransition)
    virtual_intermediate = virtual_transitions[1]
    assert isinstance(virtual_intermediate, VirtualIntermediateTransition)
    assert tuple(
        (transfer.source_tile_id, transfer.destination_tile_id)
        for transfer in intermediate.transfers
    ) == ((3, 1), (3, 0), (2, 1), (2, 0))
    assert tuple(
        (transfer.source_subslice, transfer.destination_subslice)
        for transfer in intermediate.transfers
    ) == tuple(
        (transfer.source_subslice, transfer.destination_subslice)
        for transfer in virtual_intermediate.transfers
    )
    assert intermediate.tensor_id == virtual_intermediate.tensor_id
    output = transitions[2]
    assert isinstance(output, OutputTransition)
    assert tuple(source.tile_id for source in output.sources) == (1, 0)
