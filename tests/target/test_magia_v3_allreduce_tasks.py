from maps.graph import TensorDType
from maps.hardware import WorkKind, WorkSignature
from maps.target import magia_v3


def test_magia_v3_assigns_fp16_allreduces_to_spatz() -> None:
    mesh = magia_v3.build_mesh(width=2, height=1)

    for work_kind in (WorkKind.ALL_REDUCE_SUM, WorkKind.ALL_REDUCE_MAX):
        signature = WorkSignature(
            work_kind, (TensorDType.FLOAT16,), (TensorDType.FLOAT16,)
        )
        assert magia_v3.SPATZ_DEVICE.supports(signature)
        assert mesh.tiles[0].assigned_device(signature) is magia_v3.SPATZ_DEVICE
        assert magia_v3.SPATZ_DEVICE.collective_cycles(
            work_kind, 16, mesh.tiles
        ) > 0
