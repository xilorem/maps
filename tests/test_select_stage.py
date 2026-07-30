from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.defs.elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from MAPS.planner.contracts.options import StageSelectionOptions
from MAPS.planner.passes.stage_selection import form_stages


def test_form_stages_defaults_to_singleton_groups() -> None:
    node0 = Node(name="n0", kind=OpKind.CUSTOM)
    node1 = Node(name="n1", kind=OpKind.CUSTOM)
    node2 = Node(name="n2", kind=OpKind.CUSTOM)
    graph = Graph(
        name="singleton_groups",
        nodes=(node0, node1, node2),
    )

    groups = form_stages(graph)

    assert groups == {
        0: (node0,),
        1: (node1,),
        2: (node2,),
    }


def test_form_stages_groups_nodes_with_same_explicit_stage_group_id() -> None:
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

    groups = form_stages(graph)

    assert groups == {
        0: (node0, node1),
        1: (node2,),
    }


def test_form_stages_coalesces_exact_elementwise_chain_left_to_right() -> None:
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

    assert form_stages(graph) == {0: (first, second, third)}
    assert form_stages(
        graph,
        StageSelectionOptions(max_stage_nodes=2),
    ) == {0: (first, second), 1: (third,)}
    assert form_stages(
        graph,
        StageSelectionOptions(max_stage_nodes=1),
    ) == {0: (first,), 1: (second,), 2: (third,)}


def test_form_stages_does_not_put_runtime_input_on_internal_layer() -> None:
    x = Tensor("x", 1, (8,), 2)
    runtime_input = Tensor("runtime_input", 1, (8,), 2)
    middle = Tensor("middle", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
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
        inputs=(middle, runtime_input),
        outputs=(output,),
        payload=BinaryElementwisePayload("Add", middle, runtime_input, output),
    )
    graph = Graph(
        "runtime_input",
        nodes=(first, second),
        inputs=(x, runtime_input),
    )

    assert form_stages(graph) == {0: (first,), 1: (second,)}


def test_form_stages_rejects_explicit_internal_runtime_input() -> None:
    x = Tensor("x", 1, (8,), 2)
    runtime_input = Tensor("runtime_input", 1, (8,), 2)
    middle = Tensor("middle", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Relu", x, middle),
        attributes={"stage_group_id": "explicit"},
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(middle, runtime_input),
        outputs=(output,),
        payload=BinaryElementwisePayload("Add", middle, runtime_input, output),
        attributes={"stage_group_id": "explicit"},
    )
    graph = Graph(
        "explicit_internal_runtime_input",
        nodes=(first, second),
        inputs=(x, runtime_input),
    )

    try:
        form_stages(graph)
    except ValueError as exc:
        assert "explicit stage group 'explicit'" in str(exc)
        assert "Runtime Input runtime_input reaches internal Layer second" in str(exc)
    else:
        raise AssertionError("expected invalid explicit stage group to fail")


def test_form_stages_rejects_explicit_internal_cross_stage_input() -> None:
    x = Tensor("x", 1, (8,), 2)
    external_value = Tensor("external_value", 1, (8,), 2)
    middle = Tensor("middle", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    producer = Node(
        "producer",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(external_value,),
        payload=UnaryElementwisePayload("Relu", x, external_value),
    )
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Exp", x, middle),
        attributes={"stage_group_id": "explicit"},
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(middle, external_value),
        outputs=(output,),
        payload=BinaryElementwisePayload("Add", middle, external_value, output),
        attributes={"stage_group_id": "explicit"},
    )
    graph = Graph(
        "explicit_internal_cross_stage_input",
        nodes=(producer, first, second),
        inputs=(x,),
    )

    try:
        form_stages(graph)
    except ValueError as exc:
        assert "explicit stage group 'explicit'" in str(exc)
        assert "cross-stage input external_value reaches internal Layer second" in str(exc)
    else:
        raise AssertionError("expected invalid explicit stage group to fail")


def test_form_stages_does_not_put_cross_stage_input_on_internal_layer() -> None:
    x = Tensor("x", 1, (8,), 2)
    external_value = Tensor("external_value", 1, (8,), 2)
    middle = Tensor("middle", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    producer = Node(
        "producer",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(external_value,),
        payload=UnaryElementwisePayload("Relu", x, external_value),
    )
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Exp", x, middle),
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(middle, external_value),
        outputs=(output,),
        payload=BinaryElementwisePayload("Add", middle, external_value, output),
    )
    graph = Graph(
        "internal_cross_stage_input",
        nodes=(producer, first, second),
        inputs=(x,),
    )

    assert form_stages(graph) == {
        0: (producer,),
        1: (first,),
        2: (second,),
    }


def test_form_stages_does_not_put_graph_output_on_internal_layer() -> None:
    x = Tensor("x", 1, (8,), 2)
    graph_output = Tensor("graph_output", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(graph_output,),
        payload=UnaryElementwisePayload("Relu", x, graph_output),
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(graph_output,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Exp", graph_output, output),
    )
    graph = Graph(
        "internal_graph_output",
        nodes=(first, second),
        inputs=(x,),
        outputs=(graph_output, output),
    )

    assert form_stages(graph) == {0: (first,), 1: (second,)}


def test_form_stages_rejects_explicit_internal_graph_output() -> None:
    x = Tensor("x", 1, (8,), 2)
    graph_output = Tensor("graph_output", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(graph_output,),
        payload=UnaryElementwisePayload("Relu", x, graph_output),
        attributes={"stage_group_id": "explicit"},
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(graph_output,),
        outputs=(output,),
        payload=UnaryElementwisePayload("Exp", graph_output, output),
        attributes={"stage_group_id": "explicit"},
    )
    graph = Graph(
        "explicit_internal_graph_output",
        nodes=(first, second),
        inputs=(x,),
        outputs=(graph_output, output),
    )

    try:
        form_stages(graph)
    except ValueError as exc:
        assert "explicit stage group 'explicit'" in str(exc)
        assert "graph output graph_output leaves internal Layer first" in str(exc)
    else:
        raise AssertionError("expected invalid explicit stage group to fail")


def test_form_stages_rejects_explicit_internal_cross_stage_output() -> None:
    x = Tensor("x", 1, (8,), 2)
    shared = Tensor("shared", 1, (8,), 2)
    internal_output = Tensor("internal_output", 1, (8,), 2)
    external_output = Tensor("external_output", 1, (8,), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(shared,),
        payload=UnaryElementwisePayload("Relu", x, shared),
        attributes={"stage_group_id": "explicit"},
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(shared,),
        outputs=(internal_output,),
        payload=UnaryElementwisePayload("Exp", shared, internal_output),
        attributes={"stage_group_id": "explicit"},
    )
    external_consumer = Node(
        "external_consumer",
        OpKind.CUSTOM,
        inputs=(shared,),
        outputs=(external_output,),
    )
    graph = Graph(
        "explicit_internal_cross_stage_output",
        nodes=(first, second, external_consumer),
        inputs=(x,),
    )

    try:
        form_stages(graph)
    except ValueError as exc:
        assert "explicit stage group 'explicit'" in str(exc)
        assert "cross-stage output shared leaves internal Layer first" in str(exc)
    else:
        raise AssertionError("expected invalid explicit stage group to fail")


def test_form_stages_extends_explicit_group_when_merge_makes_edges_local() -> None:
    x = Tensor("x", 1, (8,), 2)
    shared = Tensor("shared", 1, (8,), 2)
    middle = Tensor("middle", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(shared,),
        payload=UnaryElementwisePayload("Relu", x, shared),
        attributes={"stage_group_id": "explicit"},
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(shared,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Exp", shared, middle),
        attributes={"stage_group_id": "explicit"},
    )
    third = Node(
        "third",
        OpKind.ELEMENTWISE,
        inputs=(middle, shared),
        outputs=(output,),
        payload=BinaryElementwisePayload("Add", middle, shared, output),
    )
    graph = Graph(
        "extended_explicit_group",
        nodes=(first, second, third),
        inputs=(x,),
    )

    assert form_stages(graph) == {0: (first, second, third)}


def test_form_stages_prepends_to_explicit_group_when_merge_makes_edges_local() -> None:
    x = Tensor("x", 1, (8,), 2)
    shared = Tensor("shared", 1, (8,), 2)
    middle = Tensor("middle", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    producer = Node(
        "producer",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(shared,),
        payload=UnaryElementwisePayload("Relu", x, shared),
    )
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(shared,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Exp", shared, middle),
        attributes={"stage_group_id": "explicit"},
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(middle, shared),
        outputs=(output,),
        payload=BinaryElementwisePayload("Add", middle, shared, output),
        attributes={"stage_group_id": "explicit"},
    )
    graph = Graph(
        "prepended_explicit_group",
        nodes=(producer, first, second),
        inputs=(x,),
    )

    assert form_stages(graph) == {0: (producer, first, second)}


def test_form_stages_does_not_put_cross_stage_output_on_internal_layer() -> None:
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

    assert form_stages(Graph("fanout", nodes=(producer, left, right))) == {
        0: (producer,),
        1: (left,),
        2: (right,),
    }


def test_form_stages_allows_initializer_on_internal_layer() -> None:
    x = Tensor("x", 1, (8,), 2)
    initializer = Tensor("initializer", 1, (8,), 2, is_initializer=True)
    middle = Tensor("middle", 1, (8,), 2)
    output = Tensor("output", 1, (8,), 2)
    first = Node(
        "first",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(middle,),
        payload=UnaryElementwisePayload("Relu", x, middle),
        attributes={"stage_group_id": "explicit"},
    )
    second = Node(
        "second",
        OpKind.ELEMENTWISE,
        inputs=(middle, initializer),
        outputs=(output,),
        payload=BinaryElementwisePayload("Add", middle, initializer, output),
        attributes={"stage_group_id": "explicit"},
    )
    graph = Graph(
        "internal_initializer",
        nodes=(first, second),
        inputs=(x, initializer),
        initializers=(initializer,),
    )

    assert form_stages(graph) == {0: (first, second)}


def test_form_stages_rejects_unhashable_explicit_group_keys() -> None:
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
        form_stages(graph)
    except ValueError as exc:
        assert "unhashable stage_group_id" in str(exc)
    else:
        raise AssertionError("expected invalid stage_group_id to fail")
