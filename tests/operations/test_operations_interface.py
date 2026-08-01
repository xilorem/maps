"""Public contract tests for vertically organized maps Operations."""

from collections.abc import Callable

import pytest

from maps.graph import Graph, Node, OpKind, Tensor, TensorDType
from maps.graph.onnx.parser import parse_graph
from maps.graph.onnx.operations import ONNX_OPERATION_CONVERTERS
from maps.hardware import WorkKind, WorkSignature
from maps.operations import (
    LayoutRelation,
    OpCostModel,
    OperationPayload,
    OpPayload,
    TileWork,
    broadcast_input_slice,
    broadcast_shape,
    sharded_layout,
)
from maps.operations.cast import CastCostModel, CastPayload, CastTileWork
from maps.operations.elementwise import (
    BinaryElementwisePayload,
    ElementwiseCostModel,
    ElementwiseTileWork,
    UnaryElementwisePayload,
)
from maps.operations.gemm import GemmCostModel, GemmPayload, GemmTileWork
from maps.operations.collective import (
    AllReduceCostModel,
    AllReducePayload,
    CollectiveTileWork,
)
from maps.operations.convolution import (
    Conv2DCostModel,
    Conv2DPayload,
    Conv2DTileWork,
    ConvPayload,
)
from maps.operations.convolution_transforms import (
    ConvTransformCostModel,
    Im2ColPayload,
    OutputReformatPayload,
    TransformTileWork,
    WeightPackPayload,
)
from maps.operations.depthwise_convolution import (
    DepthwiseConvPayload,
    DepthwiseConvTileWork,
)
from maps.operations.normalization import (
    GroupNormalizationPayload,
    GroupNormalizeFromMomentsPayload,
    GroupReducePayload,
    GroupReduceTileWork,
)
from maps.operations.rearrangement import (
    RearrangeTileWork,
    ReshapePayload,
    TransposePayload,
)
from maps.operations.reduction import (
    GlobalAveragePoolPayload,
    ReduceSumPayload,
    ReductionCostModel,
    ReductionPayload,
    ReductionTileWork,
    ScalarMultiplyPayload,
)
from maps.operations.softmax import SoftmaxPayload
from maps.operations.split import SplitPayload, StaticSlicePayload, StaticSliceTileWork
from maps.planning.mapping import Submesh
from maps.target.magia import build_mesh as magia_mesh
from maps.planning import PlacementOptions, PlanningOptions, plan

from maps.deployment.serialization import execution_plan_json_payload


def test_operations_exposes_shared_planning_contracts() -> None:
    assert issubclass(OpPayload, OperationPayload)
    assert issubclass(OpPayload, object)
    assert issubclass(TileWork, object)
    assert issubclass(OpCostModel, object)
    assert LayoutRelation.__module__ == "maps.operations.contracts"
    assert broadcast_shape((2, 1), (1, 3)) == (2, 3)
    assert callable(broadcast_input_slice)
    assert callable(sharded_layout)


def test_gemm_family_colocates_operation_tile_work_and_cost() -> None:
    assert GemmPayload.__module__ == "maps.operations.gemm"
    assert GemmTileWork.__module__ == "maps.operations.gemm"
    assert GemmCostModel.__module__ == "maps.operations.gemm"


def test_cast_family_colocates_operation_tile_work_and_cost() -> None:
    assert CastPayload.__module__ == "maps.operations.cast"
    assert CastTileWork.__module__ == "maps.operations.cast"
    assert CastCostModel.__module__ == "maps.operations.cast"


def test_elementwise_family_colocates_operations_tile_work_and_cost() -> None:
    assert UnaryElementwisePayload.__module__ == "maps.operations.elementwise"
    assert BinaryElementwisePayload.__module__ == "maps.operations.elementwise"
    assert ElementwiseTileWork.__module__ == "maps.operations.elementwise"
    assert ElementwiseCostModel.__module__ == "maps.operations.elementwise"


@pytest.mark.parametrize(
    ("family_module", "members"),
    (
        (
            "maps.operations.collective",
            (AllReducePayload, CollectiveTileWork, AllReduceCostModel),
        ),
        (
            "maps.operations.convolution",
            (ConvPayload, Conv2DPayload, Conv2DTileWork, Conv2DCostModel),
        ),
        (
            "maps.operations.convolution_transforms",
            (
                Im2ColPayload,
                WeightPackPayload,
                OutputReformatPayload,
                TransformTileWork,
                ConvTransformCostModel,
            ),
        ),
        (
            "maps.operations.depthwise_convolution",
            (DepthwiseConvPayload, DepthwiseConvTileWork),
        ),
        (
            "maps.operations.normalization",
            (
                GroupNormalizationPayload,
                GroupNormalizeFromMomentsPayload,
                GroupReducePayload,
                GroupReduceTileWork,
            ),
        ),
        (
            "maps.operations.rearrangement",
            (ReshapePayload, TransposePayload, RearrangeTileWork),
        ),
        (
            "maps.operations.reduction",
            (
                ReductionPayload,
                ReductionTileWork,
                ReductionCostModel,
                ReduceSumPayload,
                GlobalAveragePoolPayload,
                ScalarMultiplyPayload,
            ),
        ),
        ("maps.operations.softmax", (SoftmaxPayload,)),
        (
            "maps.operations.split",
            (SplitPayload, StaticSlicePayload, StaticSliceTileWork),
        ),
    ),
)
def test_remaining_operation_families_are_vertically_owned(
    family_module: str,
    members: tuple[type[object], ...],
) -> None:
    assert all(member.__module__ == family_module for member in members)


def test_onnx_adapter_explicitly_maps_migrated_operation_families() -> None:
    assert set(ONNX_OPERATION_CONVERTERS) == {
        "Abs",
        "Add",
        "Cast",
        "Conv",
        "Div",
        "Exp",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "GroupNormalization",
        "Log",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "Relu",
        "ReduceSum",
        "Reshape",
        "Sigmoid",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
    }


def test_logical_gemm_produces_distinct_tile_work_with_exact_assignment_cost() -> None:
    x = Tensor("x", 2, (8, 16), 2, dtype=TensorDType.FLOAT16)
    weight = Tensor("weight", 2, (16, 12), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, (8, 12), 2, dtype=TensorDType.FLOAT16)
    operation = GemmPayload(x=x, w=weight, y=None, output=output)
    node = Node("gemm", OpKind.GEMM, (x, weight), (output,), operation)
    mesh = magia_mesh(width=2, height=2)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0, 1, 2, 3})

    tile_work = tuple(
        operation.build_tile_work(operation.output_layouts(submesh), tile)
        for tile in submesh.tiles
    )
    signature = WorkSignature.from_node(node)
    assigned_name = mesh.tiles[0].device_assignment.assignments[signature]
    assigned_device = mesh.tiles[0].device_by_name(assigned_name)

    assert signature == WorkSignature(
        WorkKind.GEMM,
        (TensorDType.FLOAT16, TensorDType.FLOAT16),
        (TensorDType.FLOAT16,),
    )
    assert len({id(work) for work in tile_work}) == 4
    assert all(work is not operation for work in tile_work)
    assert operation.cost_model.cost(tile_work[0], mesh.tiles[0], assigned_device) > 0
    assert tile_work[0].l1_bytes == sum(
        reference.num_bytes
        for reference in tile_work[0].input_slices + tile_work[0].output_slices
    )


def _assert_assigned_cost(
    graph: Graph,
    operation: OpPayload,
    expected_signature: WorkSignature,
    expected_device_name: str,
) -> None:
    mesh = magia_mesh(width=1, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids={0})
    tile = mesh.tiles[0]
    tile_work = operation.build_tile_work(operation.output_layouts(submesh), tile)

    assert WorkSignature.from_node(graph.nodes[0]) == expected_signature
    assert tile.device_assignment.assignments[expected_signature] == expected_device_name
    assert operation.cost_model.cost(
        tile_work,
        tile,
        tile.device_by_name(expected_device_name),
    ) > 0
    assert tile_work.l1_bytes == sum(
        reference.num_bytes
        for reference in tile_work.input_slices + tile_work.output_slices
    )


def test_cast_vertical_family_covers_signature_assignment_memory_and_cost() -> None:
    x = Tensor("x", 2, (2, 4), 4, dtype=TensorDType.FLOAT32)
    output = Tensor("output", 2, (2, 4), 2, dtype=TensorDType.FLOAT16)
    operation = CastPayload(x=x, output=output)
    graph = Graph(
        "cast",
        tensors=(x, output),
        nodes=(Node("cast", OpKind.TRANSFORM, (x,), (output,), operation),),
        inputs=(x,),
        outputs=(output,),
    )

    _assert_assigned_cost(
        graph,
        operation,
        WorkSignature(
            WorkKind.CAST,
            (TensorDType.FLOAT32,),
            (TensorDType.FLOAT16,),
        ),
        "spatz",
    )


def test_elementwise_vertical_family_covers_signature_assignment_memory_and_cost() -> None:
    x = Tensor("x", 2, (2, 4), 2, dtype=TensorDType.FLOAT16)
    output = Tensor("output", 2, (2, 4), 2, dtype=TensorDType.FLOAT16)
    operation = UnaryElementwisePayload("Relu", x, output)
    graph = Graph(
        "relu",
        tensors=(x, output),
        nodes=(Node("relu", OpKind.ELEMENTWISE, (x,), (output,), operation),),
        inputs=(x,),
        outputs=(output,),
    )

    _assert_assigned_cost(
        graph,
        operation,
        WorkSignature(
            WorkKind.RELU,
            (TensorDType.FLOAT16,),
            (TensorDType.FLOAT16,),
        ),
        "spatz",
    )


def _imported_gemm_graph() -> Graph:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 4])
    weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT16, [4, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 3])
    node = helper.make_node("MatMul", ("x", "weight"), ("output",), name="gemm")
    return parse_graph(helper.make_graph([node], "gemm", [x, weight], [output]))


def _imported_cast_graph() -> Graph:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 4])
    node = helper.make_node(
        "Cast",
        ("x",),
        ("output",),
        name="cast",
        to=TensorProto.FLOAT16,
    )
    return parse_graph(helper.make_graph([node], "cast", [x], [output]))


def _imported_elementwise_graph() -> Graph:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 4])
    node = helper.make_node("Relu", ("x",), ("output",), name="relu")
    return parse_graph(helper.make_graph([node], "elementwise", [x], [output]))


@pytest.mark.parametrize(
    ("graph_builder", "work_kind", "device_name", "serialized_operation_name"),
    (
        (_imported_gemm_graph, WorkKind.GEMM, "redmule", None),
        (_imported_cast_graph, WorkKind.CAST, "spatz", None),
        (_imported_elementwise_graph, WorkKind.RELU, "spatz", "Relu"),
    ),
)
def test_each_imported_vertical_family_plans_with_stable_observable_results(
    graph_builder: Callable[[], Graph],
    work_kind: WorkKind,
    device_name: str,
    serialized_operation_name: str | None,
) -> None:
    options = PlanningOptions(
        placement=PlacementOptions(print_placement=False),
        print_execution_plan_cost=False,
    )

    execution_plan = plan(
        graph_builder(),
        magia_mesh(width=1, height=1),
        options,
    )

    assert len(execution_plan.stages) == 1
    assert len(execution_plan.stages[0].layers) == 1
    layer = execution_plan.stages[0].layers[0]
    assert getattr(layer.node.payload, "work_kind") is work_kind
    assert layer.device_name == device_name
    assert tuple(layer.outputs[0].layout.submesh.tile_ids) == (0,)
    if serialized_operation_name is not None:
        serialized = execution_plan_json_payload(execution_plan)
        assert serialized["stages"][0]["layers"][0]["node"]["payload"][
            "op_name"
        ] == serialized_operation_name
