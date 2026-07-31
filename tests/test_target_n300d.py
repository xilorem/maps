from MAPS.planner.contracts.options import PlannerOptions, SpatialMappingOptions
from MAPS.planner.plan import plan_graph
from maps.hardware import DeviceKind, EndpointKind, Mesh, RoutingPolicy
from maps.target import n300d

from tests.test_precision_lowering import _gemm_model


def test_n300d_builds_wormhole_mesh_with_target_owned_tensix_devices() -> None:
    mesh = n300d.build_mesh()

    assert type(mesh) is Mesh
    assert mesh.shape == (n300d.MESH_WIDTH, n300d.MESH_HEIGHT)
    assert len(mesh.noc.nodes) == n300d.NOC_WIDTH * n300d.NOC_HEIGHT
    assert mesh.noc.routing_policy is RoutingPolicy.TORUS_XY
    assert len(mesh.noc.links) == 4 * n300d.NOC_WIDTH * n300d.NOC_HEIGHT
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L2)) == len(
        n300d.L2_ENDPOINT_COORDS
    )

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


def test_n300d_specialization_is_deterministic_and_plans_tensix_work() -> None:
    model = _gemm_model()

    first = n300d.specialize(model, n300d.build_mesh())
    second = n300d.specialize(model, n300d.build_mesh())

    assert first == second
    assert first.model == model
    assert first.report.events == ()

    plan = plan_graph(
        first.model.graph,
        n300d.build_mesh(),
        PlannerOptions(
            spatial_mapping=SpatialMappingOptions(print_mapping=False),
            print_execution_plan_cost=False,
        ),
    )
    assert plan.stages[0].layers[0].device_name == "tensix_matrix"
