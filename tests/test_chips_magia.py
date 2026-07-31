from MAPS.arch import DMADevice, DMAJob, DeviceKind, EndpointKind, RoutingPolicy, ScalarDevice, SystolicDevice, TrafficKind, WorkKind, WorkSignature
from MAPS.core.dtype import TensorDType
from MAPS.hw.chips.magia import (
    MAGIA_L1_BANDWIDTH_BYTES,
    MAGIA_L1_DATA_BYTES,
    MAGIA_L1_USABLE_BYTES,
    MAGIA_L2_BANDWIDTH_BYTES,
    MAGIA_MESH_HEIGHT,
    MAGIA_MESH_WIDTH,
    MAGIA_NOC_CHANNEL_WIDTH_BYTES,
    MAGIA_NOC_HOP_LATENCY_CYCLES,
    MAGIA_NOC_WIDE_CHANNEL_WIDTH_BYTES,
    magia_mesh,
)
from MAPS.hw.devices import MAGIA_CORE_DEVICE, MAGIA_SPATZ_DEVICE, MAGIA_TILE_DEVICES, SpatzDevice


def test_magia_mesh_matches_paper_memory_map_and_shape() -> None:
    mesh = magia_mesh()

    assert mesh.shape == (MAGIA_MESH_WIDTH, MAGIA_MESH_HEIGHT)
    assert mesh.num_tiles == MAGIA_MESH_WIDTH * MAGIA_MESH_HEIGHT
    assert mesh.l2_memory.bandwidth == MAGIA_L2_BANDWIDTH_BYTES
    assert MAGIA_L1_DATA_BYTES == 0xD0000
    assert MAGIA_L1_USABLE_BYTES == MAGIA_L1_DATA_BYTES
    assert all(tile.memory.size == MAGIA_L1_USABLE_BYTES for tile in mesh.tiles)
    assert all(tile.memory.bandwidth == MAGIA_L1_BANDWIDTH_BYTES for tile in mesh.tiles)
    assert mesh.noc.routing_policy is RoutingPolicy.XY
    assert mesh.noc.traffic_policy is not None
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_REQ) == (0,)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_REQ) == (2,)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_RSP) == (2,)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_RSP) == (1,)


def test_magia_tiles_have_idma_core_spatz_and_redmule_devices() -> None:
    mesh = magia_mesh()

    for tile in mesh.tiles:
        devices = {device.name: device for device in tile.devices}
        assert set(devices) == {"idma_read", "idma_write", "core", "spatz", "redmule"}
        assert devices["idma_read"].kind is DeviceKind.DMA
        assert isinstance(devices["idma_read"], DMADevice)
        assert devices["idma_read"].job is DMAJob.READJOB
        assert devices["idma_write"].kind is DeviceKind.DMA
        assert isinstance(devices["idma_write"], DMADevice)
        assert devices["idma_write"].job is DMAJob.WRITEJOB
        assert devices["core"].kind is DeviceKind.SCALAR
        assert isinstance(devices["core"], ScalarDevice)
        assert devices["spatz"].kind is DeviceKind.VECTOR
        assert isinstance(devices["spatz"], SpatzDevice)
        assert devices["redmule"].kind is DeviceKind.SYSTOLIC
        assert isinstance(devices["redmule"], SystolicDevice)


def test_magia_tile_profile_is_used_by_the_mesh() -> None:
    mesh = magia_mesh()

    assert mesh.tiles[0].devices is MAGIA_TILE_DEVICES
    assert MAGIA_CORE_DEVICE.supports(
        WorkSignature(
            WorkKind.ADD,
            (TensorDType.FLOAT32, TensorDType.FLOAT32),
            (TensorDType.FLOAT32,),
        )
    )
    assert MAGIA_CORE_DEVICE.supports(
        WorkSignature(
            WorkKind.DIV,
            (TensorDType.FLOAT16, TensorDType.FLOAT16),
            (TensorDType.FLOAT16,),
        )
    )
    assert all(tile.devices[3] is MAGIA_SPATZ_DEVICE for tile in mesh.tiles)


def test_magia_mesh_accepts_custom_shape() -> None:
    mesh = magia_mesh(width=4, height=3)

    assert mesh.shape == (4, 3)
    assert mesh.num_tiles == 12
    assert [tile.tile_id for tile in mesh.tiles] == list(range(12))
    assert mesh.tile(3, 2).tile_id == 11
    assert len(mesh.noc.nodes) == 12
    assert len(mesh.noc.links) == 17
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L1)) == 12
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L2)) == 3
    assert tuple(endpoint.node_id for endpoint in mesh.noc.endpoints_of_kind(EndpointKind.L2)) == (0, 4, 8)
    assert all(link.bidirectional for link in mesh.noc.links)
    assert all(tuple(channel.tag for channel in link.channels) == ("req", "rsp", "wide") for link in mesh.noc.links)
    assert all(link.channels[0].width_bytes == MAGIA_NOC_CHANNEL_WIDTH_BYTES for link in mesh.noc.links)
    assert all(link.channels[1].width_bytes == MAGIA_NOC_CHANNEL_WIDTH_BYTES for link in mesh.noc.links)
    assert all(link.channels[2].width_bytes == MAGIA_NOC_WIDE_CHANNEL_WIDTH_BYTES for link in mesh.noc.links)
    assert all(
        all(channel.hop_latency_cycles == MAGIA_NOC_HOP_LATENCY_CYCLES for channel in link.channels)
        for link in mesh.noc.links
    )
