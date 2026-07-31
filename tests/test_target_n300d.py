import pytest

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
from maps.planning import PlacementOptions, PlanningOptions, plan
from maps.target import n300d

from tests.test_precision_lowering import _gemm_model


def test_n300d_builds_wormhole_mesh_with_target_owned_tensix_devices() -> None:
    mesh = n300d.build_mesh()

    assert type(mesh) is Mesh
    assert mesh.shape == (n300d.MESH_WIDTH, n300d.MESH_HEIGHT)
    assert mesh.l2_memory.size == n300d.L2_SIZE_BYTES
    assert mesh.l2_memory.bandwidth == n300d.L2_BANDWIDTH_BYTES
    assert all(tile.memory.size == n300d.L1_USABLE_BYTES for tile in mesh.tiles)
    assert all(
        tile.memory.bandwidth == n300d.L1_BANDWIDTH_BYTES for tile in mesh.tiles
    )
    assert len(mesh.noc.nodes) == n300d.NOC_WIDTH * n300d.NOC_HEIGHT
    assert mesh.noc.routing_policy is RoutingPolicy.TORUS_XY
    assert mesh.noc.traffic_policy is not None
    assert len(mesh.noc.links) == 4 * n300d.NOC_WIDTH * n300d.NOC_HEIGHT
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L2)) == len(
        n300d.L2_ENDPOINT_COORDS
    )
    l1_endpoints = mesh.noc.endpoints_of_kind(EndpointKind.L1)
    l2_endpoints = mesh.noc.endpoints_of_kind(EndpointKind.L2)
    assert tuple(
        mesh.noc.node_by_id(endpoint.node_id).coords for endpoint in l1_endpoints
    ) == n300d.TILE_NOC_COORDS
    assert tuple(
        mesh.noc.node_by_id(endpoint.node_id).coords for endpoint in l2_endpoints
    ) == n300d.L2_ENDPOINT_COORDS
    assert all(
        endpoint.ingress_latency_cycles == n300d.NIU_LATENCY_CYCLES
        and endpoint.egress_latency_cycles == n300d.NIU_LATENCY_CYCLES
        for endpoint in l1_endpoints + l2_endpoints
    )
    assert {link.channels[0].tag for link in mesh.noc.links} == {"noc0", "noc1"}
    assert all(
        link.channels[0].width_bytes == n300d.NOC_CHANNEL_WIDTH_BYTES
        and link.channels[0].hop_latency_cycles == n300d.NOC_HOP_LATENCY_CYCLES
        for link in mesh.noc.links
    )
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_REQ) == (0, 1)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_DATA) == (0, 1)

    devices = {device.name: device for device in mesh.tiles[0].devices}
    assert set(devices) == {
        "tensix_read_core",
        "tensix_write_core",
        "tensix_scalar",
        "tensix_vector",
        "tensix_matrix",
    }
    assert devices["tensix_read_core"].kind is DeviceKind.DMA
    assert devices["tensix_write_core"].kind is DeviceKind.DMA
    assert devices["tensix_scalar"].kind is DeviceKind.SCALAR
    assert devices["tensix_vector"].kind is DeviceKind.VECTOR
    assert devices["tensix_matrix"].kind is DeviceKind.MATRIX
    assert tuple(devices.values()) == n300d.TILE_DEVICES
    assert n300d.READ_CORE.job is DMAJob.READJOB
    assert n300d.WRITE_CORE.job is DMAJob.WRITEJOB
    assert mesh.tiles[0].assigned_device(
        WorkSignature(
            WorkKind.GEMM,
            (TensorDType.FLOAT32, TensorDType.FLOAT32),
            (TensorDType.FLOAT32,),
        )
    ) is n300d.MATRIX_DEVICE
    assert mesh.tiles[0].assigned_device(
        WorkSignature(
            WorkKind.RELU,
            (TensorDType.FLOAT16,),
            (TensorDType.FLOAT16,),
        )
    ) is n300d.VECTOR_DEVICE

    with pytest.raises(ValueError, match="fixed 8x8 compute Mesh"):
        n300d.build_mesh(width=4, height=4)


def test_n300d_specialization_is_deterministic_and_plans_tensix_work() -> None:
    model = _gemm_model()

    first = n300d.specialize(model, n300d.build_mesh())
    second = n300d.specialize(model, n300d.build_mesh())

    assert first == second
    assert first.model == model
    assert first.report.events == ()

    execution_plan = plan(
        first.model.graph,
        n300d.build_mesh(),
        PlanningOptions(
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    assert execution_plan.stages[0].layers[0].device_name == "tensix_matrix"
