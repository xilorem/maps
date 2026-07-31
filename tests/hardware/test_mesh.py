import pytest

from maps.hardware import EndpointKind, L1Memory, L2Memory, Mesh, NoC, NoCChannel, NoCEndpoint, NoCLink, NoCNode
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def test_mesh_preserves_dimension_accessors() -> None:
    mesh = Mesh(
        width=3,
        height=2,
        l2_memory=L2Memory(size=8192, bandwidth=1),
        noc=rectangular_test_noc(3, 2),
        tiles=rectangular_test_tiles(3, 2),
    )

    assert mesh.width == 3
    assert mesh.height == 2
    assert mesh.shape == (3, 2)
    assert mesh.num_tiles == 6
    assert mesh.l2_memory.size == 8192


def test_mesh_can_use_tile_l1_capacity() -> None:
    mesh = Mesh(
        width=2,
        height=2,
        l2_memory=L2Memory(size=16384, bandwidth=1),
        noc=rectangular_test_noc(2, 2),
        tiles=rectangular_test_tiles(2, 2, memory=L1Memory(size=4096, bandwidth=1)),
    )

    assert tuple(tile.memory.size for tile in mesh.tiles) == (4096, 4096, 4096, 4096)
    assert mesh.tile(1, 1).memory.size == 4096


def test_mesh_exposes_memory_objects() -> None:
    mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=16384, bandwidth=128),
        noc=rectangular_test_noc(2, 1),
        tiles=rectangular_test_tiles(2, 1, memory=L1Memory(size=4096, bandwidth=64)),
    )

    assert mesh.l2_memory.size == 16384
    assert mesh.l2_memory.bandwidth == 128
    assert mesh.tile(0, 0).memory.bandwidth == 64
    assert tuple(tile.memory.size for tile in mesh.tiles) == (4096, 4096)


def test_mesh_rectangle_keeps_row_major_order() -> None:
    mesh = Mesh(
        width=4,
        height=3,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(4, 3),
        tiles=rectangular_test_tiles(4, 3),
    )

    rectangle = mesh.rectangle(x0=1, y0=1, width=2, height=2)

    assert [tile.tile_id for tile in rectangle] == [5, 6, 9, 10]


def test_mesh_accepts_attached_noc() -> None:
    noc = NoC(
        nodes=(
            NoCNode(node_id=0, x=0, y=0),
            NoCNode(node_id=1, x=1, y=0),
        ),
        links=(
            NoCLink(
                link_id=0,
                src_node_id=0,
                dst_node_id=1,
                channels=(NoCChannel(channel_id=0, width_bytes=4),),
                bidirectional=True,
            ),
        ),
        endpoints=(
            NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
            NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=1),
        ),
    )

    mesh = Mesh(width=2, height=1, l2_memory=L2Memory(size=4096, bandwidth=1), noc=noc, tiles=rectangular_test_tiles(2, 1))

    assert mesh.noc == noc
    assert mesh.noc.endpoint_by_id(0).tile_id == 0


def test_mesh_rejects_attached_noc_endpoint_tile_id_outside_mesh() -> None:
    noc = NoC(
        nodes=(NoCNode(node_id=0, x=0, y=0),),
        links=(),
        endpoints=(NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=4),),
    )

    with pytest.raises(ValueError, match="NoC endpoint tile_id out of bounds"):
        Mesh(width=2, height=2, l2_memory=L2Memory(size=4096, bandwidth=1), noc=noc, tiles=rectangular_test_tiles(2, 2))
