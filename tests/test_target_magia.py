from maps.graph import TensorDType
from maps.hardware import (
    DMAJob,
    DeviceKind,
    EndpointKind,
    Mesh,
    RoutingPolicy,
    TrafficKind,
    WorkKind,
    WorkSignature,
)
from maps.operations.convolution_transforms import Im2ColPayload, OutputReformatPayload
from maps.operations.cast import CastPayload
from maps.operations.gemm import GemmPayload
from maps.planning import PlacementOptions, PlanningOptions, plan
from maps.target import magia
from maps.target import SpecializationOptions

from tests.test_conv_to_gemm import _conv_model
from tests.test_precision_lowering import _gemm_model


def test_magia_builds_a_plain_mesh_with_target_owned_devices() -> None:
    mesh = magia.build_mesh(width=2, height=1)

    assert type(mesh) is Mesh
    assert mesh.shape == (2, 1)
    assert mesh.l2_memory.size == magia.L2_SIZE_BYTES
    assert mesh.l2_memory.bandwidth == magia.L2_BANDWIDTH_BYTES
    assert all(tile.memory.size == magia.L1_USABLE_BYTES for tile in mesh.tiles)
    assert all(
        tile.memory.bandwidth == magia.L1_BANDWIDTH_BYTES for tile in mesh.tiles
    )
    assert mesh.noc.routing_policy is RoutingPolicy.XY
    assert mesh.noc.traffic_policy is not None
    assert len(mesh.noc.nodes) == 2
    assert len(mesh.noc.links) == 1
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L1)) == 2
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L2)) == 1
    assert tuple(channel.tag for channel in mesh.noc.links[0].channels) == (
        "req",
        "rsp",
        "wide",
    )
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_REQ) == (0,)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_REQ) == (2,)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_RSP) == (2,)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_RSP) == (1,)
    assert not hasattr(mesh, "precision_lowering_recipes")
    assert not hasattr(mesh, "required_graph_rewrites")

    devices = {device.name: device for device in mesh.tiles[0].devices}
    assert set(devices) == {"idma_read", "idma_write", "core", "spatz", "redmule"}
    assert devices["idma_read"].kind is DeviceKind.DMA
    assert devices["idma_write"].kind is DeviceKind.DMA
    assert devices["core"].kind is DeviceKind.SCALAR
    assert devices["spatz"].kind is DeviceKind.VECTOR
    assert devices["redmule"].kind is DeviceKind.SYSTOLIC
    assert tuple(devices.values()) == magia.TILE_DEVICES
    assert devices["core"] is magia.CORE_DEVICE
    assert devices["spatz"] is magia.SPATZ_DEVICE
    assert devices["redmule"] is magia.REDMULE_DEVICE
    assert magia.IDMA_READ_DEVICE.job is DMAJob.READJOB
    assert magia.IDMA_WRITE_DEVICE.job is DMAJob.WRITEJOB
    assert mesh.tiles[0].assigned_device(
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT16, TensorDType.FLOAT16),
            (TensorDType.FLOAT16,),
        )
    ) is magia.REDMULE_DEVICE
    assert mesh.tiles[0].assigned_device(
        WorkSignature(
            WorkKind.RELU,
            (TensorDType.FLOAT32,),
            (TensorDType.FLOAT32,),
        )
    ) is magia.SPATZ_DEVICE

    configurable = magia.build_mesh(width=4, height=3)
    assert configurable.shape == (4, 3)
    assert len(configurable.noc.links) == 17
    assert tuple(
        endpoint.node_id
        for endpoint in configurable.noc.endpoints_of_kind(EndpointKind.L2)
    ) == (0, 4, 8)


def test_magia_specializes_convolution_deterministically_and_plans_it() -> None:
    model, _ = _conv_model(TensorDType.FLOAT16)
    options = SpecializationOptions(enable_precision_lowering=False)

    first = magia.specialize(model, magia.build_mesh(width=1, height=1), options)
    second = magia.specialize(model, magia.build_mesh(width=1, height=1), options)

    assert first == second
    im2col, gemm, output_reformat = first.model.graph.nodes
    assert isinstance(im2col.payload, Im2ColPayload)
    assert isinstance(gemm.payload, GemmPayload)
    assert isinstance(output_reformat.payload, OutputReformatPayload)
    assert [WorkSignature.from_node(node).work_kind for node in first.model.graph.nodes] == [
        WorkKind.IM2COL,
        WorkKind.GEMM,
        WorkKind.OUTPUT_REFORMAT,
    ]
    assert [event.rewrite_name for event in first.report.events] == ["conv_to_gemm"]

    execution_plan = plan(
        first.model.graph,
        magia.build_mesh(width=1, height=1),
        PlanningOptions(
            placement=PlacementOptions(print_mapping=False),
            print_execution_plan_cost=False,
        ),
    )
    assert [layer.device_name for layer in execution_plan.stages[0].layers] == [
        "core",
        "redmule",
        "core",
    ]


def test_magia_optionally_lowers_fp32_gemm_precision_and_initializers() -> None:
    result = magia.specialize(
        _gemm_model(),
        magia.build_mesh(width=1, height=1),
        SpecializationOptions(enable_precision_lowering=True),
    )

    input_cast, gemm, output_cast = result.model.graph.nodes
    assert isinstance(input_cast.payload, CastPayload)
    assert isinstance(gemm.payload, GemmPayload)
    assert isinstance(output_cast.payload, CastPayload)
    assert [tensor.dtype for tensor in gemm.inputs] == [TensorDType.FLOAT16] * 2
    assert result.model.constants.get("weight").dtype is TensorDType.FLOAT16
    assert [event.rewrite_name for event in result.report.events] == [
        "precision_lowering"
    ]
    assert result.report.events[0].converted_initializers == ("weight",)


def test_magia_composes_convolution_and_precision_specialization() -> None:
    model, _ = _conv_model(TensorDType.FLOAT32)

    result = magia.specialize(model, magia.build_mesh(width=1, height=1))

    assert [WorkSignature.from_node(node).work_kind for node in result.model.graph.nodes] == [
        WorkKind.IM2COL,
        WorkKind.CAST,
        WorkKind.GEMM,
        WorkKind.CAST,
        WorkKind.OUTPUT_REFORMAT,
    ]
    assert [event.rewrite_name for event in result.report.events] == [
        "conv_to_gemm",
        "precision_lowering",
    ]
    assert result.model.constants.get("weight").dtype is TensorDType.FLOAT16
    assert result.model.constants.get("bias").dtype is TensorDType.FLOAT16
