from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from maps.graph import (
    Constant,
    ConstantStore,
    Edge,
    Graph,
    ImportedModel,
    Node,
    OpKind,
    Tensor,
    TensorDType,
    decompose_graph,
    import_onnx_model,
    validate_imported_model,
)


def test_graph_interface_owns_logical_models_and_onnx_import() -> None:
    assert Graph.__module__.startswith("maps.graph")
    assert Node.__module__.startswith("maps.graph")
    assert Edge.__module__.startswith("maps.graph")
    assert Tensor.__module__.startswith("maps.graph")
    assert TensorDType.__module__.startswith("maps.graph")
    assert ImportedModel.__module__.startswith("maps.graph")
    assert Constant.__module__.startswith("maps.graph")
    assert ConstantStore.__module__.startswith("maps.graph")
    assert import_onnx_model.__module__.startswith("maps.graph")


def test_imported_model_validation_matches_initializer_metadata_and_bytes() -> None:
    weight = Tensor(
        "weight",
        rank=1,
        dims=(2,),
        elem_bytes=4,
        dtype=TensorDType.FLOAT32,
        is_initializer=True,
    )
    model = ImportedModel(
        graph=Graph("model", tensors=(weight,), initializers=(weight,)),
        constants=ConstantStore(
            (Constant("weight", TensorDType.FLOAT32, (2,), b"too short"),)
        ),
    )

    with pytest.raises(ValueError, match="has 9 bytes; expected 8"):
        validate_imported_model(model)


def test_hardware_independent_decomposition_is_deterministic() -> None:
    tensor = Tensor("x", 1, (1,), 4, dtype=TensorDType.FLOAT32)
    graph = Graph("identity", tensors=(tensor,), inputs=(tensor,), outputs=(tensor,))

    assert decompose_graph(graph) == graph
    assert decompose_graph(graph) == decompose_graph(graph)


def test_graph_package_has_no_hardware_or_planning_imports() -> None:
    graph_package = Path(__file__).parents[1] / "maps" / "graph"
    forbidden = {
        "MAPS.arch",
        "MAPS.hw",
        "MAPS.planner",
        "maps.hardware",
        "maps.target",
        "maps.planning",
    }

    imports = set()
    for source_path in graph_package.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert not {
        imported
        for imported in imports
        if any(
            imported == name or imported.startswith(f"{name}.")
            for name in forbidden
        )
    }


def test_importing_logical_graph_does_not_load_hardware_or_planning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from maps.graph import import_onnx_model, run_graph_rewrites; "
            "import sys; "
            "forbidden = tuple(name for name in sys.modules "
            "if name.startswith(('MAPS.arch', 'MAPS.hw', 'MAPS.planner', "
            "'maps.hardware', 'maps.target', 'maps.planning'))); "
            "assert not forbidden, forbidden",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
