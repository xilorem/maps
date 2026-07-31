from MAPS.arch import L2Memory, Mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from maps.operations.gemm import GemmPayload
from MAPS.planner.contracts.stages import StagePlacement, StagePlan, virtual_submesh
from MAPS.planner.passes.spatial_mapping import map_spatially
from MAPS.planner.spatial.evaluation import MappingEvaluator, evaluate_mapping
from MAPS.planner.spatial.ownership import assign_stage_ownerships
import MAPS.planner.spatial.repair as spatial_repair
import MAPS.planner.spatial.topology as spatial_topology
from MAPS.planner.spatial.models import VirtualTraffic
from MAPS.planner.spatial.traffic import build_virtual_traffic
from MAPS.transitions import VirtualIntermediateTransition, build_virtual_transitions
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _test_mesh(width: int, height: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height),
    )


def _gemm_node(name: str, x: Tensor | None = None) -> Node:
    input_tensor = x if x is not None else Tensor(name=f"{name}_x", rank=2, dims=(8, 8), elem_bytes=2)
    weight_tensor = Tensor(name=f"{name}_w", rank=2, dims=(8, 8), elem_bytes=2)
    output_tensor = Tensor(name=f"{name}_out", rank=2, dims=(8, 8), elem_bytes=2)
    op = GemmPayload(x=input_tensor, w=weight_tensor, y=None, output=output_tensor)
    return Node(
        name=name,
        kind=OpKind.GEMM,
        inputs=(input_tensor, weight_tensor),
        outputs=(output_tensor,),
        payload=op,
    )


def _single_node_stage_plan(mesh: Mesh, stage_id: int, node: Node, tile_ids: set[int]) -> StagePlan:
    virtual_submesh = Submesh(mesh=mesh, submesh_id=stage_id, tile_ids=frozenset(tile_ids))
    output_layouts = node.payload.output_layouts(virtual_submesh, logical_shape=(len(tile_ids), 1))
    return StagePlan(
        stage_id=stage_id,
        tile_count=len(tile_ids),
        logical_shape=(len(tile_ids), 1),
        nodes=(node,),
        node_output_layouts=(output_layouts,),
        device_names=("core",),
    )


def _share_boundary(mesh: Mesh, left: set[int], right: set[int]) -> bool:
    for tile_id in left:
        x, y = mesh.coords(tile_id)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if mesh.contains_coord(nx, ny) and mesh.tile_id(nx, ny) in right:
                return True
    return False


def test_build_virtual_traffic_tracks_inter_stage_bytes() -> None:
    mesh = _test_mesh(4, 2)
    producer = _gemm_node("producer")
    consumer = _gemm_node("consumer", x=producer.outputs[0])
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
        edges=(Edge(tensor=producer.outputs[0], src=producer, dst=consumer),),
    )
    stage_plans = {
        0: _single_node_stage_plan(mesh, 0, producer, {0, 1}),
        1: _single_node_stage_plan(mesh, 1, consumer, {0, 1}),
    }

    virtual_transitions = build_virtual_transitions(graph, stage_plans)
    traffic = build_virtual_traffic(virtual_transitions, stage_plans)

    assert traffic.stage_comm[(0, 1)] > 0
    assert sum(traffic.input_weights[1].values()) > 0
    assert sum(traffic.output_weights[0].values()) > 0


def test_map_spatially_returns_connected_adjacent_mapping() -> None:
    mesh = _test_mesh(4, 2)
    producer = _gemm_node("producer")
    consumer = _gemm_node("consumer", x=producer.outputs[0])
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
        edges=(Edge(tensor=producer.outputs[0], src=producer, dst=consumer),),
    )
    stage_plans = {
        0: _single_node_stage_plan(mesh, 0, producer, {0, 1}),
        1: _single_node_stage_plan(mesh, 1, consumer, {0, 1}),
    }

    mapping = map_spatially(
        mesh=mesh,
        stage_plans=stage_plans,
        virtual_transitions=build_virtual_transitions(graph, stage_plans),
        print_mapping=False,
        show_progress=False,
    )

    assert set(mapping) == {0, 1}
    all_tile_ids = set()
    for stage_id, placement in mapping.items():
        assert placement.physical_submesh.num_tiles == stage_plans[stage_id].tile_count
        assert len(placement.virtual_to_physical) == stage_plans[stage_id].tile_count
        assert len(set(placement.virtual_to_physical.values())) == stage_plans[stage_id].tile_count
        all_tile_ids |= set(placement.physical_submesh.tile_ids)

    assert len(all_tile_ids) == 4
    assert _share_boundary(
        mesh,
        set(mapping[0].physical_submesh.tile_ids),
        set(mapping[1].physical_submesh.tile_ids),
    )


def test_mapping_charges_l1_communication_to_the_producer_tile() -> None:
    mesh = _test_mesh(2, 1)
    producer = _gemm_node("producer")
    consumer = _gemm_node("consumer", x=producer.outputs[0])
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
        edges=(Edge(tensor=producer.outputs[0], src=producer, dst=consumer),),
    )
    stage_plans = {
        0: _single_node_stage_plan(mesh, 0, producer, {0}),
        1: _single_node_stage_plan(mesh, 1, consumer, {0}),
    }
    placements = {
        stage_id: StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh(plan),
            physical_submesh=Submesh(mesh=mesh, submesh_id=stage_id, tile_ids=frozenset({stage_id})),
            virtual_to_physical={0: stage_id},
        )
        for stage_id, plan in stage_plans.items()
    }

    evaluation = evaluate_mapping(
        mesh=mesh,
        stage_plans=stage_plans,
        placements=placements,
        virtual_transitions=build_virtual_transitions(graph, stage_plans),
    )

    producer_score = evaluation.tile_scores[0]
    consumer_score = evaluation.tile_scores[1]
    assert producer_score.tile_to_tile_writes > 0
    assert producer_score.consumer_stage_writes == {1: producer_score.tile_to_tile_writes}
    assert consumer_score.tile_to_tile_writes == 0
    assert evaluation.stage_breakdowns[0].l1_write == producer_score.tile_to_tile_writes
    assert producer_score.score == (
        producer_score.l2_reads + producer_score.l2_writes + producer_score.tile_to_tile_writes
    )


def test_repair_region_skips_an_infeasible_growth_attempt(monkeypatch) -> None:
    mesh = _test_mesh(2, 1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids=frozenset({0}))
    placement = StagePlacement(
        stage_id=0,
        virtual_submesh=submesh,
        physical_submesh=submesh,
        virtual_to_physical={0: 0},
    )
    traffic = VirtualTraffic(
        stage_comm={},
        edge_matrices={},
        input_weights={},
        output_weights={},
        l2_read_weights={},
        l2_write_weights={},
        communication_degree={},
        bottleneck_risk={},
        l2_pressure={},
    )

    def fail_growth(**kwargs) -> set[int]:
        assert kwargs["exhaustive_future_feasibility"] is False
        raise ValueError("infeasible region")

    monkeypatch.setattr(spatial_repair, "grow_stage_region", fail_growth)

    assert spatial_repair.repair_region(
        mesh=mesh,
        stage_plans={
            0: StagePlan(
                stage_id=0,
                tile_count=1,
                logical_shape=(1, 1),
                nodes=(),
                node_output_layouts=(),
                device_names=(),
            )
        },
        current_placements={0: placement},
        traffic=traffic,
        affected_stages=frozenset({0}),
        focus_stage_id=0,
        debug=False,
    ) is None


def test_incremental_evaluation_rescores_moved_stages_and_predecessors() -> None:
    mesh = _test_mesh(4, 1)
    producer = _gemm_node("producer")
    consumer = _gemm_node("consumer", x=producer.outputs[0])
    neighbor = _gemm_node("neighbor")
    unrelated = _gemm_node("unrelated")
    graph = Graph(
        name="g",
        tensors=tuple(
            dict.fromkeys(
                tensor
                for node in (producer, consumer, neighbor, unrelated)
                for tensor in node.inputs + node.outputs
            )
        ),
        nodes=(producer, consumer, neighbor, unrelated),
        edges=(Edge(tensor=producer.outputs[0], src=producer, dst=consumer),),
        outputs=(consumer.outputs[0], unrelated.outputs[0]),
    )
    nodes = (producer, consumer, neighbor, unrelated)
    stage_plans = {
        stage_id: _single_node_stage_plan(mesh, stage_id, node, {0})
        for stage_id, node in enumerate(nodes)
    }
    placements = {
        stage_id: StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh(plan),
            physical_submesh=Submesh(
                mesh=mesh,
                submesh_id=stage_id,
                tile_ids=frozenset({stage_id}),
            ),
            virtual_to_physical={0: stage_id},
        )
        for stage_id, plan in stage_plans.items()
    }
    virtual_transitions = build_virtual_transitions(graph, stage_plans)
    evaluator = MappingEvaluator(mesh, stage_plans, virtual_transitions)
    initial = evaluator.evaluate(placements)

    trial = dict(placements)
    for stage_id, tile_id in ((1, 2), (2, 1)):
        trial[stage_id] = StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh(stage_plans[stage_id]),
            physical_submesh=Submesh(
                mesh=mesh,
                submesh_id=stage_id,
                tile_ids=frozenset({tile_id}),
            ),
            virtual_to_physical={0: tile_id},
        )

    incremental = evaluator.evaluate(
        trial,
        previous=initial,
        moved_stage_ids=frozenset({1, 2}),
    )
    complete = evaluate_mapping(
        mesh,
        stage_plans,
        trial,
        virtual_transitions,
    )

    assert incremental == complete
    assert incremental.tile_scores[3] is initial.tile_scores[3]
    assert incremental.tile_scores[0] is not initial.tile_scores[0]


def test_exact_mapping_ignores_initializers_absent_from_virtual_transitions() -> None:
    mesh = _test_mesh(1, 1)
    node = _gemm_node("only")
    graph = Graph(
        name="g",
        tensors=node.inputs + node.outputs,
        inputs=node.inputs,
        initializers=node.inputs,
        nodes=(node,),
    )
    stage_plans = {0: _single_node_stage_plan(mesh, 0, node, {0})}
    placement = StagePlacement(
        stage_id=0,
        virtual_submesh=virtual_submesh(stage_plans[0]),
        physical_submesh=Submesh(
            mesh=mesh,
            submesh_id=0,
            tile_ids=frozenset({0}),
        ),
        virtual_to_physical={0: 0},
    )
    virtual_transitions = build_virtual_transitions(graph, stage_plans)

    evaluation = evaluate_mapping(
        mesh,
        stage_plans,
        {0: placement},
        virtual_transitions,
    )

    assert virtual_transitions == ()
    assert evaluation.tile_scores[0].score == 0


def test_exact_mapping_charges_runtime_input_reads_and_graph_output_writes() -> None:
    mesh = _test_mesh(1, 1)
    node = _gemm_node("only")
    graph = Graph(
        name="g",
        tensors=node.inputs + node.outputs,
        inputs=node.inputs,
        outputs=node.outputs,
        initializers=(node.inputs[1],),
        nodes=(node,),
    )
    stage_plans = {0: _single_node_stage_plan(mesh, 0, node, {0})}
    placement = StagePlacement(
        stage_id=0,
        virtual_submesh=virtual_submesh(stage_plans[0]),
        physical_submesh=Submesh(
            mesh=mesh,
            submesh_id=0,
            tile_ids=frozenset({0}),
        ),
        virtual_to_physical={0: 0},
    )

    evaluation = evaluate_mapping(
        mesh,
        stage_plans,
        {0: placement},
        build_virtual_transitions(graph, stage_plans),
    )

    score = evaluation.tile_scores[0]
    assert score.l2_reads > 0
    assert score.l2_writes > 0
    assert score.tile_to_tile_writes == 0


def test_incremental_evaluation_reuses_source_of_empty_transition() -> None:
    mesh = _test_mesh(3, 1)
    nodes = tuple(_gemm_node(name) for name in ("source", "destination", "other"))
    stage_plans = {
        stage_id: _single_node_stage_plan(mesh, stage_id, node, {0})
        for stage_id, node in enumerate(nodes)
    }
    placements = {
        stage_id: StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh(plan),
            physical_submesh=Submesh(
                mesh=mesh,
                submesh_id=stage_id,
                tile_ids=frozenset({stage_id}),
            ),
            virtual_to_physical={0: stage_id},
        )
        for stage_id, plan in stage_plans.items()
    }
    empty_transition = VirtualIntermediateTransition(
        tensor=nodes[0].outputs[0],
        tensor_id=0,
        source_stage_id=0,
        source_output_index=0,
        destination_stage_id=1,
        destination_input_index=0,
    )
    evaluator = MappingEvaluator(mesh, stage_plans, (empty_transition,))
    initial = evaluator.evaluate(placements)
    trial = dict(placements)
    for stage_id, tile_id in ((1, 2), (2, 1)):
        trial[stage_id] = StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh(stage_plans[stage_id]),
            physical_submesh=Submesh(
                mesh=mesh,
                submesh_id=stage_id,
                tile_ids=frozenset({tile_id}),
            ),
            virtual_to_physical={0: tile_id},
        )

    incremental = evaluator.evaluate(
        trial,
        previous=initial,
        moved_stage_ids=frozenset({1, 2}),
    )

    assert incremental == evaluator.evaluate(trial)
    assert incremental.tile_scores[0] is initial.tile_scores[0]


def test_local_ownership_assignment_preserves_unmoved_stages() -> None:
    mesh = _test_mesh(3, 1)
    nodes = tuple(_gemm_node(f"stage_{stage_id}") for stage_id in range(3))
    stage_plans = {
        stage_id: _single_node_stage_plan(mesh, stage_id, node, {0})
        for stage_id, node in enumerate(nodes)
    }
    placements = {
        stage_id: StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh(plan),
            physical_submesh=Submesh(
                mesh=mesh,
                submesh_id=stage_id,
                tile_ids=frozenset({stage_id}),
            ),
            virtual_to_physical={0: stage_id},
        )
        for stage_id, plan in stage_plans.items()
    }
    traffic = VirtualTraffic(
        stage_comm={},
        edge_matrices={},
        input_weights={stage_id: {0: 0} for stage_id in stage_plans},
        output_weights={stage_id: {0: 0} for stage_id in stage_plans},
        l2_read_weights={stage_id: {0: 0} for stage_id in stage_plans},
        l2_write_weights={stage_id: {0: 0} for stage_id in stage_plans},
        communication_degree={stage_id: 0 for stage_id in stage_plans},
        bottleneck_risk={stage_id: 0 for stage_id in stage_plans},
        l2_pressure={stage_id: 0 for stage_id in stage_plans},
    )

    assigned = assign_stage_ownerships(
        mesh,
        stage_plans,
        placements,
        traffic,
        stage_ids=frozenset({1}),
    )

    assert assigned[0] is placements[0]
    assert assigned[2] is placements[2]


def test_non_exhaustive_future_feasibility_uses_component_sizes(monkeypatch) -> None:
    mesh = _test_mesh(5, 4)

    def fail_subset_enumeration(**kwargs) -> bool:
        del kwargs
        raise AssertionError("connected subsets must not be enumerated")

    monkeypatch.setattr(
        spatial_topology,
        "_can_partition_connected_regions",
        fail_subset_enumeration,
    )

    assert spatial_topology.remaining_counts_fit_free_components(
        mesh,
        set(range(mesh.num_tiles)),
        (10, 10),
        exhaustive=False,
    )
