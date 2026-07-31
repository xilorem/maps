from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def test_lowercase_namespace_runs_a_representative_planning_workflow(
    tmp_path: Path,
) -> None:
    from maps.hw.chips import magia_mesh
    from maps.pipeline import ExecutionPlan
    from maps.planner.plan import build_execution_plan
    from MAPS.hw.chips import magia_mesh as legacy_magia_mesh
    from MAPS.pipeline import ExecutionPlan as LegacyExecutionPlan
    from MAPS.planner.plan import build_execution_plan as legacy_build_execution_plan

    assert magia_mesh is legacy_magia_mesh
    assert ExecutionPlan is LegacyExecutionPlan
    assert build_execution_plan is legacy_build_execution_plan

    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [1, 1], [1.0])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ("x", "weight"), ("output",), name="gemm")],
            "lowercase_namespace",
            (input_info,),
            (output_info,),
            initializer=(weight,),
        )
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    execution_plan = build_execution_plan(model_path, magia_mesh(width=1, height=1))

    assert isinstance(execution_plan, ExecutionPlan)
    assert execution_plan.name == "lowercase_namespace"


def test_installed_package_exposes_lowercase_and_migration_namespaces(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    build_python = shutil.which("python")
    assert build_python is not None
    isolated_environment = os.environ.copy()
    isolated_environment.pop("PYTHONPATH", None)
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
            str(repository),
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
            "--system-site-packages",
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
            "import MAPS, maps, maps.cli, maps.planner.plan",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=isolated_environment,
    )

    assert result.returncode == 0, result.stderr

    cli_result = subprocess.run(
        [str(environment_directory / "bin" / "maps"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=isolated_environment,
    )

    assert cli_result.returncode == 0, cli_result.stderr
    assert "{plan,package}" in cli_result.stdout


def test_lowercase_cli_starts_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "maps.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "{plan,package}" in result.stdout
