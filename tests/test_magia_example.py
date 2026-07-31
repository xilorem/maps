from pathlib import Path
from types import SimpleNamespace

from examples import magia_example


def test_magia_example_builds_and_writes_one_execution_plan_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    execution_plan = SimpleNamespace(
        name="simple_three_stage",
        stages=(),
        transitions=(),
    )
    bundle = SimpleNamespace(execution_plan=execution_plan)
    calls: dict[str, object] = {}

    monkeypatch.setattr(magia_example, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        magia_example,
        "build_execution_plan_bundle",
        lambda model_path, mesh, options: calls.update(
            model_path=model_path,
            mesh=mesh,
            options=options,
        ) or bundle,
    )
    monkeypatch.setattr(
        magia_example,
        "validate_execution_plan",
        lambda execution_plan, constraints: SimpleNamespace(
            is_valid=True,
            violations=(),
        ),
    )
    monkeypatch.setattr(
        magia_example,
        "print_submeshes",
        lambda execution_plan: None,
    )

    def write_bundle(value, execution_plan_path, weights_path):
        calls["written_bundle"] = value
        calls["execution_plan_path"] = execution_plan_path
        calls["weights_path"] = weights_path
        return execution_plan_path, weights_path

    monkeypatch.setattr(
        magia_example,
        "write_execution_plan_bundle",
        write_bundle,
    )

    magia_example.main()

    assert calls["model_path"] == magia_example.DEFAULT_MODEL_PATH
    assert calls["written_bundle"] is bundle
    assert calls["execution_plan_path"] == (
        tmp_path / "generated" / "magia_example.execution-plan.json"
    )
    assert calls["weights_path"] == (
        tmp_path / "generated" / "magia_example.execution-plan.weights.bin"
    )
    options = calls["options"]
    assert options.execution.num_token_slots == 2
    assert options.stage_formation.max_stage_nodes == 1
