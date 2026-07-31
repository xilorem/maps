from maps.graph import TensorDType
from maps.hardware import (
    DMADevice,
    DMAJob,
    Device,
    DeviceKind,
    EndpointKind,
    FixedDeviceAssignment,
    L1Memory,
    NoC,
    NoCChannel,
    NoCEndpoint,
    NoCLink,
    NoCNode,
    ScalarDevice,
    Tile,
    WorkKind,
    same_dtype_signatures,
)

_TEST_WORK = tuple(work_kind for work_kind in WorkKind if work_kind is not WorkKind.DMA)
TEST_SCALAR_DEVICE = ScalarDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={work_kind: 1 for work_kind in _TEST_WORK},
    capabilities=same_dtype_signatures(
        _TEST_WORK,
        (1, 2, 3, 5),
        (TensorDType.FLOAT16, TensorDType.FLOAT32),
    ),
)
TEST_IDMA_READ_DEVICE = DMADevice(
    name="idma_read",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.READJOB,
)
TEST_IDMA_WRITE_DEVICE = DMADevice(
    name="idma_write",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1},
    job=DMAJob.WRITEJOB,
)
TEST_DEVICE_ASSIGNMENT = FixedDeviceAssignment(
    {
        signature: TEST_SCALAR_DEVICE.name
        for signature in TEST_SCALAR_DEVICE.capabilities
    }
)
DEFAULT_TEST_TILE_DEVICES = (
    TEST_IDMA_READ_DEVICE,
    TEST_IDMA_WRITE_DEVICE,
    TEST_SCALAR_DEVICE,
)


def rectangular_test_tiles(
    width: int,
    height: int,
    *,
    memory: L1Memory = L1Memory(size=1, bandwidth=1),
    devices: tuple[Device, ...] = DEFAULT_TEST_TILE_DEVICES,
) -> tuple[Tile, ...]:
    return tuple(
        Tile(
            tile_id=y * width + x,
            x=x,
            y=y,
            memory=memory,
            devices=devices,
            device_assignment=(
                TEST_DEVICE_ASSIGNMENT
                if devices == DEFAULT_TEST_TILE_DEVICES
                else FixedDeviceAssignment()
            ),
        )
        for y in range(height)
        for x in range(width)
    )


def rectangular_test_noc(width: int, height: int, *, include_l2: bool = True) -> NoC:
    nodes = tuple(
        NoCNode(node_id=y * width + x, x=x, y=y)
        for y in range(height)
        for x in range(width)
    )
    link_pairs = tuple(
        (y * width + x, y * width + x + 1)
        for y in range(height)
        for x in range(width - 1)
    ) + tuple(
        (y * width + x, (y + 1) * width + x)
        for y in range(height - 1)
        for x in range(width)
    )
    links = tuple(
        NoCLink(
            link_id=link_id,
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            channels=(NoCChannel(channel_id=0, width_bytes=1, hop_latency_cycles=1),),
            bidirectional=True,
        )
        for link_id, (src_node_id, dst_node_id) in enumerate(link_pairs)
    )
    l1_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=tile_id,
            kind=EndpointKind.L1,
            node_id=tile_id,
            tile_id=tile_id,
        )
        for tile_id in range(width * height)
    )
    l2_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=width * height + y,
            kind=EndpointKind.L2,
            node_id=y * width,
            name=f"l2_{y}",
        )
        for y in range(height)
    )
    return NoC(
        nodes=nodes,
        links=links,
        endpoints=l1_endpoints + (l2_endpoints if include_l2 else ()),
    )
