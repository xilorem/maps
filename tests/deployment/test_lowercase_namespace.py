from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_installed_package_exposes_only_the_lowercase_namespace(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    build_python = shutil.which("python")
    assert build_python is not None
    isolated_environment = os.environ.copy()
    isolated_environment.pop("PYTHONPATH", None)
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    shutil.copy2(repository / "pyproject.toml", source_directory)
    shutil.copytree(repository / "maps", source_directory / "maps")
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    subprocess.run(
        [
            build_python,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_directory),
            str(source_directory),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=isolated_environment,
    )
    wheel = next(wheel_directory.glob("*.whl"))
    environment_directory = tmp_path / "environment"
    subprocess.run(
        [
            build_python,
            "-m",
            "venv",
            str(environment_directory),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=isolated_environment,
    )
    installed_python = environment_directory / "bin" / "python"
    subprocess.run(
        [
            str(installed_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=isolated_environment,
    )

    result = subprocess.run(
        [
            str(installed_python),
            "-c",
            "import MAPS",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=isolated_environment,
    )

    assert result.returncode != 0
    assert "No module named 'MAPS'" in result.stderr

    lowercase_result = subprocess.run(
        [str(installed_python), "-c", "import maps"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=isolated_environment,
    )

    assert lowercase_result.returncode == 0, lowercase_result.stderr


def test_repository_contains_no_retired_package_paths() -> None:
    repository = Path(__file__).parents[2]
    retired_paths = (
        repository / "MAPS",
        repository / "maps" / "hw",
        repository / "maps" / "pipeline",
        repository / "maps" / "planner",
        repository / "maps" / "deployment" / "package.py",
    )

    assert not tuple(path for path in retired_paths if path.exists())


def test_repository_sources_do_not_import_retired_namespaces() -> None:
    repository = Path(__file__).parents[2]
    retired = (
        "MAPS",
        "maps.arch",
        "maps.core",
        "maps.hw",
        "maps.importers",
        "maps.ops",
        "maps.pipeline",
        "maps.planner",
        "maps.transforms",
        "maps.transitions",
        "maps.utils",
    )
    imported_by_path: dict[Path, set[str]] = {}
    for source_root in ("maps", "tests", "examples", "tutorials"):
        for source_path in (repository / source_root).rglob("*.py"):
            imports = set()
            for node in ast.walk(ast.parse(source_path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
            retired_imports = {
                name
                for name in imports
                if any(
                    name == namespace or name.startswith(f"{namespace}.")
                    for namespace in retired
                )
            }
            if retired_imports:
                imported_by_path[source_path.relative_to(repository)] = retired_imports

    assert not imported_by_path


def test_tests_are_grouped_once_by_public_role_without_duplicate_modules() -> None:
    tests_root = Path(__file__).parents[1]
    test_files = tuple(tests_root.rglob("test_*.py"))
    roles = {"graph", "operations", "hardware", "target", "planning", "deployment"}

    assert test_files
    assert {path.parent.name for path in test_files} <= roles
    assert all(len(path.relative_to(tests_root).parts) == 2 for path in test_files)
    module_names = tuple(path.name for path in test_files)
    assert len(module_names) == len(set(module_names))


def test_lowercase_cli_starts_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "maps.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "{build,plan,inspect,verify}" in result.stdout
