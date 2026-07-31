import inspect

from maps.target import magia, n300d


def test_concrete_targets_expose_the_same_operations_without_chip_wrappers() -> None:
    assert inspect.signature(magia.build_mesh).return_annotation == inspect.signature(
        n300d.build_mesh
    ).return_annotation
    assert inspect.signature(magia.specialize) == inspect.signature(n300d.specialize)
    assert not hasattr(magia, "Chip")
    assert not hasattr(n300d, "Chip")
    assert not hasattr(n300d, "wormhole_n300d_mesh")
