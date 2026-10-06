import pytest

from maps.hardware import L2Memory, Mesh
from maps.graph import Edge, Graph, Node, OpKind, TensorDType, decompose_graph
from maps.planning.mapping import Submesh
from maps.graph import Tensor
from maps.operations.gemm import GemmPayload
from maps.operations.softmax import SoftmaxPayload
from maps.planning.allocation.candidates import StageCandidateAnalyzer
from maps.planning.stages import StagePlacement, StagePlan, virtual_submesh
from maps.planning.placement import place
from maps.planning.placement.evaluation import PlacementEvaluator, evaluate_placement
from maps.planning.placement.regions import assign_stage_ownerships
import maps.planning.placement.repair as placement_repair
import maps.planning.placement.regions as placement_topology
import maps.planning.allocation.candidates as candidates_module
from maps.planning.placement.evaluation import VirtualTraffic
from maps.planning.placement.evaluation import build_virtual_traffic
from maps.planning.transitions import VirtualIntermediateTransition, build_virtual_transitions
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles
from maps.target import magia


def test_anchor_precomputation_preserves_scores_and_refreshes_after_peer_moves() -> None:
    mesh = magia.build_mesh(width=4, height=4)
    traffic = VirtualTraffic(
        stage_comm={(0, 1): 7, (2, 0): 11, (0, 3): 0},
        edge_matrices={}, input_weights={}, output_weights={},
        l2_read_weights={}, l2_write_weights={}, communication_degree={},
        bottleneck_risk={}, l2_pressure={0: 13},
    )
    placed = {1: {0, 1}, 2: {8, 12}}
    before = placement_topology.stage_anchor_costs(mesh, 0, traffic, placed)
    assert before == tuple(
        placement_topology.stage_anchor_cost(mesh, 0, tile, traffic, placed)
        for tile in mesh.tiles
    )
    placed[1] = {14, 15}
    after = placement_topology.stage_anchor_costs(mesh, 0, traffic, placed)
    assert after != before
    assert after == tuple(
        placement_topology.stage_anchor_cost(mesh, 0, tile, traffic, placed)
        for tile in mesh.tiles
    )
    seeds = placement_topology.sorted_candidate_tiles(
        mesh, range(mesh.num_tiles), (1.5, 1.5), 0, traffic, placed,
        anchor_costs=after,
    )
    assert seeds == sorted(range(mesh.num_tiles), key=lambda tile_id: (
        placement_topology._seed_tile_score(
            0, mesh, mesh.tile_by_id(tile_id), (1.5, 1.5), traffic, placed,
        ), mesh.tile_by_id(tile_id).y, mesh.tile_by_id(tile_id).x, tile_id,
    ))


def test_serpentine_fallback_partitions_a_full_mesh_into_connected_regions() -> None:
    mesh = _test_mesh(4, 2)

    regions = placement_topology.snake_stage_regions(
        mesh,
        ordered_stage_ids=(3, 7, 9),
        tile_counts={3: 3, 7: 2, 9: 3},
    )

    assert regions == {3: {0, 1, 2}, 7: {3, 7}, 9: {4, 5, 6}}
    assert _share_boundary(mesh, regions[3], regions[7])
    assert _share_boundary(mesh, regions[7], regions[9])


def test_serpentine_fallback_leaves_unused_tiles_after_sparse_allocations() -> None:
    mesh = _test_mesh(4, 2)

    regions = placement_topology.snake_stage_regions(
        mesh,
        ordered_stage_ids=(0, 1),
        tile_counts={0: 2, 1: 2},
    )

    assert regions == {0: {0, 1}, 1: {2, 3}}
    assert set.union(*regions.values()) == {0, 1, 2, 3}


def test_initial_placement_uses_serpentine_fallback_for_sparse_allocation(
    monkeypatch,
) -> None:
    mesh = _test_mesh(3, 2)
    nodes = (_gemm_node("stage_0"), _gemm_node("stage_1"))
    stage_plans = {
        stage_id: _single_node_stage_plan(mesh, stage_id, node, {0, 1})
        for stage_id, node in enumerate(nodes)
    }
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
        del kwargs
        raise ValueError("heuristic growth failed")

    monkeypatch.setattr(placement_topology, "grow_stage_region", fail_growth)

    placements = placement_topology.build_initial_stage_placements(
        mesh=mesh,
        stage_plans=stage_plans,
        tile_counts={0: 2, 1: 2},
        traffic=traffic,
        debug=False,
    )

    placed_tile_ids = set().union(
        *(placement.physical_submesh.tile_ids for placement in placements.values())
    )
    assert len(placed_tile_ids) == 4


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


def test_place_returns_connected_adjacent_placements() -> None:
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

    placements = place(
        mesh=mesh,
        stage_plans=stage_plans,
        virtual_transitions=build_virtual_transitions(graph, stage_plans),
        print_placement=False,
        show_progress=False,
    )

    assert set(placements) == {0, 1}
    all_tile_ids = set()
    for stage_id, placement in placements.items():
        assert placement.physical_submesh.num_tiles == stage_plans[stage_id].tile_count
        assert len(placement.virtual_to_physical) == stage_plans[stage_id].tile_count
        assert len(set(placement.virtual_to_physical.values())) == stage_plans[stage_id].tile_count
        all_tile_ids |= set(placement.physical_submesh.tile_ids)

    assert len(all_tile_ids) == 4
    assert _share_boundary(
        mesh,
        set(placements[0].physical_submesh.tile_ids),
        set(placements[1].physical_submesh.tile_ids),
    )


def test_placement_charges_l1_communication_to_the_producer_tile() -> None:
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

    evaluation = evaluate_placement(
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
        producer_score.stage_latency
        + producer_score.l2_reads + producer_score.l2_writes + producer_score.tile_to_tile_writes
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

    monkeypatch.setattr(placement_repair, "grow_stage_region", fail_growth)

    assert placement_repair.repair_region(
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


@pytest.mark.parametrize("compute_weight", [1.0, 100.0])
def test_repair_moves_an_isolated_stage_into_unused_tiles_nearer_l2(compute_weight) -> None:
    mesh = _test_mesh(6, 3)
    producer = _gemm_node("producer")
    fixed = _gemm_node("fixed")
    graph = Graph(
        name="sparse",
        tensors=producer.inputs + producer.outputs + fixed.inputs + fixed.outputs,
        nodes=(producer, fixed),
        inputs=producer.inputs,
        outputs=producer.outputs,
        initializers=(producer.inputs[1],) + fixed.inputs,
    )
    plans = {
        0: _single_node_stage_plan(mesh, 0, producer, {0, 1}),
        1: _single_node_stage_plan(mesh, 1, fixed, {0}),
    }
    placements = placement_topology.placements_from_regions(
        mesh, plans, {0: {16, 17}, 1: {5}},
    )
    transitions = build_virtual_transitions(graph, plans)
    traffic = build_virtual_traffic(transitions, plans)
    evaluator = PlacementEvaluator(mesh, plans, transitions, stage_latency_weight=compute_weight)
    initial = evaluator.evaluate(placements)
    if compute_weight == 100.0:
        # The worst stage has no traffic and cannot improve.
        assert initial.tile_scores[initial.worst_tile_id].stage_id == 1

    repaired = placement_repair.improve_placement(
        mesh, plans, placements, traffic, transitions, initial, False,
        evaluator=evaluator,
    )

    assert evaluator.evaluate(repaired).objective < initial.objective
    assert repaired[1] is placements[1]
    region = repaired[0].physical_submesh.tile_ids
    assert len(region) == 2
    assert region.isdisjoint(placements[1].physical_submesh.tile_ids)
    assert region - placements[0].physical_submesh.tile_ids
    left, right = sorted(region)
    assert _share_boundary(mesh, {left}, {right})
    assert set(repaired[0].virtual_to_physical) == {0, 1}
    assert set(repaired[0].virtual_to_physical.values()) == set(region)
    assert placement_repair.improve_placement(
        mesh, plans, placements, traffic, transitions, initial, False,
        evaluator=evaluator,
    ) == repaired


@pytest.mark.parametrize("allow_unused_tiles", (False, True))
def test_region_repair_keeps_other_stages_reserved(allow_unused_tiles: bool) -> None:
    mesh = _test_mesh(4, 1)
    nodes = (_gemm_node("movable"), _gemm_node("fixed"))
    plans = {
        0: _single_node_stage_plan(mesh, 0, nodes[0], {0, 1}),
        1: _single_node_stage_plan(mesh, 1, nodes[1], {0}),
    }
    placements = placement_topology.placements_from_regions(
        mesh, plans, {0: {2, 3}, 1: {0}},
    )
    graph = Graph(
        name="reserved",
        tensors=nodes[0].inputs + nodes[0].outputs,
        nodes=(nodes[0],),
        inputs=nodes[0].inputs,
        outputs=nodes[0].outputs,
        initializers=(nodes[0].inputs[1],),
    )
    transitions = build_virtual_transitions(graph, plans)
    traffic = build_virtual_traffic(transitions, plans)

    repaired = placement_repair.repair_region(
        mesh, plans, placements, traffic, frozenset({0}), 0, False,
        allow_unused_tiles=allow_unused_tiles,
    )

    assert repaired is not None
    assert repaired[1] is placements[1]
    region = repaired[0].physical_submesh.tile_ids
    assert 0 not in region
    assert len(region) == 2
    if allow_unused_tiles:
        assert 1 in region
        assert evaluate_placement(mesh, plans, repaired, transitions).objective < (
            evaluate_placement(mesh, plans, placements, transitions).objective
        )
    else:
        assert region == placements[0].physical_submesh.tile_ids


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
    evaluator = PlacementEvaluator(mesh, stage_plans, virtual_transitions)
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
    complete = evaluate_placement(
        mesh,
        stage_plans,
        trial,
        virtual_transitions,
    )

    assert incremental == complete
    assert incremental.tile_scores[3] is initial.tile_scores[3]
    assert incremental.tile_scores[0] is not initial.tile_scores[0]


def test_exact_placement_ignores_initializers_absent_from_virtual_transitions() -> None:
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

    evaluation = evaluate_placement(
        mesh,
        stage_plans,
        {0: placement},
        virtual_transitions,
    )

    assert virtual_transitions == ()
    assert evaluation.tile_scores[0].l2_reads == 0
    assert evaluation.tile_scores[0].l2_writes == 0
    assert evaluation.tile_scores[0].stage_latency > 0
    assert evaluation.tile_scores[0].score == evaluation.tile_scores[0].stage_latency


def test_collective_stage_placement_prefers_nearer_physical_participants(
    monkeypatch,
) -> None:
    x = Tensor("x", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 1, (8,), 2, dtype=TensorDType.FLOAT16)
    softmax = Node(
        "softmax",
        OpKind.CUSTOM,
        inputs=(x,),
        outputs=(output,),
        payload=SoftmaxPayload(x, output, axis=0),
    )
    graph = decompose_graph(
        Graph(
            "softmax",
            tensors=(x, output),
            nodes=(softmax,),
            inputs=(x,),
            outputs=(output,),
        )
    )
    mesh = magia.build_mesh(width=4, height=2)
    monkeypatch.setattr(candidates_module, "logical_shape_options", lambda _: ((4, 1),))
    plan = StageCandidateAnalyzer(
        {0: graph.nodes},
        mesh,
        frozenset(),
    ).candidate(0, 4)
    assert plan is not None
    virtual = virtual_submesh(plan.plan)

    def placement(tile_ids: frozenset[int]) -> StagePlacement:
        return StagePlacement(
            stage_id=0,
            virtual_submesh=virtual,
            physical_submesh=Submesh(mesh, 0, tile_ids),
            virtual_to_physical={
                virtual_id: physical_id
                for virtual_id, physical_id in zip(
                    sorted(virtual.tile_ids), sorted(tile_ids)
                )
            },
        )

    adjacent = evaluate_placement(
        mesh,
        {0: plan.plan},
        {0: placement(frozenset((0, 1, 4, 5)))},
        (),
    )
    distant = evaluate_placement(
        mesh,
        {0: plan.plan},
        {0: placement(frozenset((0, 1, 2, 3)))},
        (),
    )

    assert adjacent.stage_breakdowns[0].stage_latency < (
        distant.stage_breakdowns[0].stage_latency
    )
    assert len(adjacent.objective) == 1
    assert adjacent.objective < distant.objective


def test_exact_placement_charges_runtime_input_reads_and_graph_output_writes() -> None:
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

    evaluation = evaluate_placement(
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
        destination_stage_id=1,
    )
    evaluator = PlacementEvaluator(mesh, stage_plans, (empty_transition,))
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
        placement_topology,
        "_can_partition_connected_regions",
        fail_subset_enumeration,
    )

    assert placement_topology.remaining_counts_fit_free_components(
        mesh,
        set(range(mesh.num_tiles)),
        (10, 10),
        exhaustive=False,
    )


def test_compact_seeding_is_independent_of_extra_mesh_space() -> None:
    from maps.planning.placement import compact_tile_domain
    positions = []
    for size in (8, 16):
        mesh = _test_mesh(size, size)
        nodes = (_gemm_node("a"), _gemm_node("b"))
        plans = {i: _single_node_stage_plan(mesh, i, node, set(range(13)))
                 for i, node in enumerate(nodes)}
        traffic = build_virtual_traffic((), plans)
        domain = compact_tile_domain(mesh, 26)
        placements = placement_topology.build_initial_stage_placements(
            mesh, plans, {0: 13, 1: 13}, traffic, False, allowed_tile_ids=domain,
        )
        occupied = set()
        for p in placements.values():
            assert p.physical_submesh.tile_ids <= domain
            assert not occupied & p.physical_submesh.tile_ids
            occupied.update(p.physical_submesh.tile_ids)
            assert p.physical_submesh.num_tiles == 13
        positions.append({i: {mesh.coords(t) for t in p.physical_submesh.tile_ids}
                          for i, p in placements.items()})
    assert positions[0] == positions[1]


def test_compact_seed_handles_thin_mesh_and_constrained_fallback(monkeypatch) -> None:
    from maps.planning.placement import compact_tile_domain
    mesh = _test_mesh(1, 16)
    plans = {0: _single_node_stage_plan(mesh, 0, _gemm_node("a"), {0, 1, 2})}
    domain = compact_tile_domain(mesh, 3)
    def fail_growth(**kwargs):
        raise ValueError("force fallback")
    monkeypatch.setattr(placement_topology, "grow_stage_region", fail_growth)
    placed = placement_topology.build_initial_stage_placements(
        mesh, plans, {0: 3}, build_virtual_traffic((), plans), False,
        allowed_tile_ids=domain,
    )
    assert placed[0].physical_submesh.tile_ids <= domain
    assert placed[0].physical_submesh.num_tiles == 3
