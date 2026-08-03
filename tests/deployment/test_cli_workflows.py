from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import maps.cli as cli_module
from maps.cli import main


@pytest.mark.parametrize(
    ("target_name", "mesh_shape"),
    (("magia", (2, 3)), ("n300d", (8, 8))),
)
def test_plan_command_composes_the_selected_target_workflow(
    target_name: str,
    mesh_shape: tuple[int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    imported = object()
    rewritten = object()
    graph = object()
    execution_plan = object()
    mesh = SimpleNamespace(width=mesh_shape[0], height=mesh_shape[1])
    target = getattr(cli_module, target_name)

    monkeypatch.setattr(
        cli_module,
        "import_onnx_model",
        lambda model: calls.append(("import", model)) or imported,
    )
    monkeypatch.setattr(
        cli_module,
        "run_graph_rewrites",
        lambda model: calls.append(("rewrite", model)) or rewritten,
    )
    monkeypatch.setattr(
        target,
        "build_mesh",
        lambda **shape: calls.append(("mesh", shape)) or mesh,
    )
    monkeypatch.setattr(
        target,
        "specialize",
        lambda model, selected_mesh, options: calls.append(
            ("specialize", model, selected_mesh, options)
        )
        or SimpleNamespace(model=SimpleNamespace(graph=graph)),
    )
    monkeypatch.setattr(
        cli_module,
        "plan",
        lambda selected_graph, selected_mesh, options: calls.append(
            ("plan", selected_graph, selected_mesh, options)
        )
        or execution_plan,
    )
    monkeypatch.setattr(
        cli_module,
        "write_execution_plan_json",
        lambda value, output: calls.append(("write", value, output)) or output,
    )
    model = tmp_path / "model.onnx"
    output = tmp_path / "execution-plan.json"

    assert main(
        [
            "plan",
            str(model),
            "--target",
            target_name,
            "--mesh",
            f"{mesh_shape[0]}x{mesh_shape[1]}",
            "--output",
            str(output),
        ]
    ) == 0

    assert calls[0:3] == [
        ("mesh", {"width": mesh_shape[0], "height": mesh_shape[1]}),
        ("import", model),
        ("rewrite", imported),
    ]
    assert calls[-1] == ("write", execution_plan, output)


def test_n300d_package_request_fails_before_deployment_work(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "package",
                str(tmp_path / "model.onnx"),
                "--target",
                "n300d",
                "--output",
                str(tmp_path / "model.maps"),
            ]
        )

    assert "n300d deployment backend is unsupported" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("target_name", "mesh_shape"),
    (("magia", (4, 4)), ("n300d", (8, 8))),
)
def test_plan_command_plans_a_representative_model_for_each_target(
    target_name: str,
    mesh_shape: tuple[int, int],
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / f"{target_name}.json"

    assert main(
        [
            "plan",
            str(repository / "examples" / "simple_three_stage.onnx"),
            "--target",
            target_name,
            "--mesh",
            f"{mesh_shape[0]}x{mesh_shape[1]}",
            "--max-stage-operations",
            "1",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["name"] == "simple_three_stage"
    assert (payload["mesh"]["width"], payload["mesh"]["height"]) == mesh_shape
    assert payload["stages"]


def test_make_exposes_target_neutral_cli_workflows() -> None:
    makefile = (Path(__file__).parents[2] / "Makefile").read_text(encoding="utf-8")

    for workflow in ("test", "plan", "package", "inspect", "verify"):
        assert f"{workflow}:" in makefile
    assert "-m maps.cli" in makefile
    assert "--target $(TARGET)" in makefile
    assert "examples/magia_example.py" not in makefile
    assert "-m MAPS.cli" not in makefile


@pytest.mark.parametrize(
    ("target_name", "mesh"),
    (("magia", "4x4"), ("n300d", "8x8")),
)
def test_make_plan_executes_the_cli_for_each_target(
    target_name: str,
    mesh: str,
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / f"{target_name}.json"

    result = subprocess.run(
        [
            "make",
            "plan",
            f"TARGET={target_name}",
            f"MESH={mesh}",
            f"EXECUTION_PLAN={output}",
            "MAX_STAGE_OPERATIONS=1",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()


def test_make_package_reports_the_n300d_deployment_limitation(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / "n300d.maps"

    result = subprocess.run(
        ["make", "package", "TARGET=n300d", f"PACKAGE={output}"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "n300d deployment backend is unsupported" in result.stderr
    assert not output.exists()
