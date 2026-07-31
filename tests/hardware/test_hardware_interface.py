from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from maps.hardware import (
    Device,
    FixedDeviceAssignment,
    L1Memory,
    L2Memory,
    Mesh,
    NoC,
    Tile,
    WorkKind,
    WorkSignature,
)
from maps.hardware.reporting import print_mesh


def test_hardware_interface_owns_reusable_physical_contracts() -> None:
    assert Mesh.__module__.startswith("maps.hardware")
    assert Tile.__module__.startswith("maps.hardware")
    assert L1Memory.__module__.startswith("maps.hardware")
    assert L2Memory.__module__.startswith("maps.hardware")
    assert NoC.__module__.startswith("maps.hardware")
    assert Device.__module__.startswith("maps.hardware")
    assert WorkKind.__module__.startswith("maps.hardware")
    assert WorkSignature.__module__.startswith("maps.hardware")
    assert FixedDeviceAssignment.__module__.startswith("maps.hardware")
    assert print_mesh.__module__ == "maps.hardware.reporting"


def test_mesh_contract_contains_no_target_specialization_policy() -> None:
    assert "required_graph_rewrites" not in Mesh.__dataclass_fields__
    assert "precision_lowering_recipes" not in Mesh.__dataclass_fields__


def test_hardware_package_has_no_concrete_target_imports() -> None:
    hardware_package = Path(__file__).parents[2] / "maps" / "hardware"
    forbidden = {"MAPS.hw", "maps.target"}

    imports = set()
    for source_path in hardware_package.rglob("*.py"):
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


def test_importing_hardware_does_not_load_concrete_targets() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import maps.hardware, sys; "
            "forbidden = tuple(name for name in sys.modules "
            "if name.startswith(('MAPS.hw', 'maps.target'))); "
            "assert not forbidden, forbidden",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
