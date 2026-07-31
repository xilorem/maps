from MAPS.core.graph import Node, OpKind
from MAPS.core.layout import TensorRange
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.hw.chips import magia_mesh
from maps.operations.collective import AllReducePayload
from maps.operations.normalization import (
    GroupNormalizationPayload,
    GroupNormalizeFromMomentsPayload,
    GroupReducePayload,
    decompose_group_normalization_node,
)


def _tensor(name: str, dims: tuple[int, ...]) -> Tensor:
    return Tensor(name, len(dims), dims, 4)


def test_group_normalization_decomposes_like_softmax_with_collectives() -> None:
    x = _tensor("x", (1, 4, 2, 2))
    scale = _tensor("scale", (4,))
    bias = _tensor("bias", (4,))
    output = _tensor("output", (1, 4, 2, 2))
    payload = GroupNormalizationPayload(x, scale, bias, output, num_groups=2)
    node = Node(
        "group_norm",
        OpKind.CUSTOM,
        (x, scale, bias),
        (output,),
        payload,
    )

    tensors, nodes = decompose_group_normalization_node(node)

    assert len(tensors) == 7
    assert tuple(item.name for item in nodes) == (
        "group_norm__square",
        "group_norm__reduce_sum",
        "group_norm__reduce_sumsq",
        "group_norm__allreduce_sum_x",
        "group_norm__allreduce_sum_y",
        "group_norm__allreduce_sumsq_x",
        "group_norm__allreduce_sumsq_y",
        "group_norm__normalize",
    )
    assert isinstance(nodes[1].payload, GroupReducePayload)
    assert isinstance(nodes[3].payload, AllReducePayload)
    assert isinstance(nodes[-1].payload, GroupNormalizeFromMomentsPayload)
    assert all(
        item.attributes["stage_group_id"] == "group_norm::group_norm"
        for item in nodes
    )
    assert nodes[-1].payload.element_count_per_group == 8


def test_group_normalization_tile_work_uses_owned_values_and_channel_affine() -> None:
    mesh = magia_mesh(width=2, height=1)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    x = _tensor("x", (1, 4, 2, 4))
    stats = _tensor("stats", (1, 2, 1, 1))
    scale = _tensor("scale", (4,))
    bias = _tensor("bias", (4,))
    output = _tensor("output", x.dims)
    reduce = GroupReducePayload(x, stats, num_groups=2)
    normalize = GroupNormalizeFromMomentsPayload(
        x,
        stats,
        stats,
        scale,
        bias,
        output,
        num_groups=2,
        epsilon=1e-5,
    )

    reduce_work = reduce.build_tile_work(
        reduce.output_layouts(submesh),
        submesh.tiles[1],
    )
    normalize_work = normalize.build_tile_work(
        normalize.output_layouts(submesh),
        submesh.tiles[1],
    )

    assert reduce_work.input_slice.dims[-1] == TensorRange(2, 2)
    assert reduce_work.output_slice.dims == (
        TensorRange(0, 1),
        TensorRange(0, 2),
        TensorRange(0, 1),
        TensorRange(0, 1),
    )
    assert normalize_work.output_slice.dims[-1] == TensorRange(2, 2)
    assert normalize_work.input_tile_slices[3].dims[0] == TensorRange(0, 4)
    assert normalize_work.input_tile_slices[4].dims[0] == TensorRange(0, 4)
