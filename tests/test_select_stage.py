from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.defs.elementwise import UnaryElementwisePayload
from MAPS.planner.contracts.options import StageSelectionOptions
from MAPS.planner.passes.stage_selection import select_stages


def test_select_stages_defaults_to_singleton_groups() -> None:
    node0 = Node(name="n0", kind=OpKind.CUSTOM)
    node1 = Node(name="n1", kind=OpKind.CUSTOM)
    node2 = Node(name="n2", kind=OpKind.CUSTOM)
    graph = Graph(
        name="singleton_groups",
        nodes=(node0, node1, node2),
    )

    groups = select_stages(graph)

    assert groups == {
        0: (node0,),
        1: (node1,),
        2: (node2,),
    }


def test_select_stages_groups_nodes_with_same_explicit_stage_group_id() -> None:
    intermediate = Tensor("intermediate", 1, (8,), 2)
    node0 = Node(
        name="reduce_max",
        kind=OpKind.REDUCTION,
        outputs=(intermediate,),
        attributes={"stage_group_id": "softmax_0"},
    )
    node1 = Node(
        name="allreduce_max",
        kind=OpKind.CUSTOM,
        inputs=(intermediate,),
        attributes={"stage_group_id": "softmax_0"},
    )
    node2 = Node(name="next_stage", kind=OpKind.CUSTOM)
    graph = Graph(
        name="grouped_nodes",
        nodes=(node0, node1, node2),
    )

    groups = select_stages(graph)

    assert groups == {
        0: (node0, node1),
        1: (node2,),
    }


def test_select_stages_coalesces_exact_elementwise_chain_left_to_right() -> None:
    x = Tensor("x", 2, (4, 8), 2)
    middle = Tensor("middle", 2, (4, 8), 2)
    almost_output = Tensor("almost_output", 2, (4, 8), 2)
    output = Tensor("output", 2, (4, 8), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Relu", x, middle),
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(middle,),
        outputs=(almost_output,),
        payload=UnaryElementwisePayload("Exp", middle, almost_output),
    )
    third = Node(
        "third",
        OpKind.ELEMENTWISE,
        inputs=(almost_output,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Neg", almost_output, output),
    )
    graph = Graph("chain", nodes=(first, second, third))

    assert select_stages(graph) == {0: (first, second, third)}
    assert select_stages(
        graph,
        StageSelectionOptions(max_stage_nodes=2),
    ) == {0: (first, second), 1: (third,)}
    assert select_stages(
        graph,
        StageSelectionOptions(max_stage_nodes=1),
    ) == {0: (first,), 1: (second,), 2: (third,)}


def test_select_stages_stops_at_fanout() -> None:
    x = Tensor("x", 1, (8,), 2)
    middle = Tensor("middle", 1, (8,), 2)
    left_out = Tensor("left_out", 1, (8,), 2)
    right_out = Tensor("right_out", 1, (8,), 2)
    producer = Node(
        "producer",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Relu", x, middle),
    )
    left = Node(
        "left",
        OpKind.ELEMENTWISE,
        inputs=(middle,),
        outputs=(left_out,),
        payload=UnaryElementwisePayload("Exp", middle, left_out),
    )
    right = Node(
        "right",
        OpKind.ELEMENTWISE,
        inputs=(middle,),
        outputs=(right_out,),
        payload=UnaryElementwisePayload("Neg", middle, right_out),
    )

    assert select_stages(Graph("fanout", nodes=(producer, left, right))) == {
        0: (producer,),
        1: (left,),
        2: (right,),
    }


def test_select_stages_rejects_unhashable_explicit_group_keys() -> None:
    graph = Graph(
        name="bad_group_key",
        nodes=(
            Node(
                name="softmax_step",
                kind=OpKind.CUSTOM,
                attributes={"stage_group_id": {"bad": "key"}},
            ),
        ),
    )

    try:
        select_stages(graph)
    except ValueError as exc:
        assert "unhashable stage_group_id" in str(exc)
    else:
        raise AssertionError("expected invalid stage_group_id to fail")
