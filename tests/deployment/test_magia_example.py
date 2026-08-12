from pathlib import Path
from types import SimpleNamespace

from examples import magia_example


def test_magia_example_builds_and_writes_one_deployment_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    execution_plan = SimpleNamespace(
        name="simple_three_stage",
        stages=(),
        transitions=(),
    )
    imported_model = object()
    rewritten_model = object()
    specialized_graph = object()
    specialization = SimpleNamespace(
        model=SimpleNamespace(graph=specialized_graph),
    )
    bundle = SimpleNamespace(execution_plan=execution_plan)
    calls: dict[str, object] = {}

    monkeypatch.setattr(magia_example, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        magia_example,
        "import_onnx_model",
        lambda model_path: calls.update(
            model_path=model_path,
        ) or imported_model,
    )
    monkeypatch.setattr(
        magia_example,
        "run_graph_rewrites_with_effects",
        lambda model: (rewritten_model, ()),
    )
    monkeypatch.setattr(
        magia_example.magia,
        "specialize",
        lambda model, mesh, options: calls.update(
            specialized_model=model,
            mesh=mesh,
            specialization_options=options,
        ) or specialization,
    )
    monkeypatch.setattr(
        magia_example,
        "plan",
        lambda graph, mesh, options: calls.update(
            planned_graph=graph,
            planning_options=options,
        ) or execution_plan,
    )
    monkeypatch.setattr(
        magia_example,
        "build_deployment_bundle",
        lambda value, plan, graph_rewrite_effects: calls.update(
            specialization=value,
            planned_execution=plan,
            graph_rewrite_effects=graph_rewrite_effects,
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

    def write_bundle(value, bundle_path, initializers_path):
        calls["written_bundle"] = value
        calls["bundle_path"] = bundle_path
        calls["initializers_path"] = initializers_path
        return bundle_path, initializers_path

    monkeypatch.setattr(
        magia_example,
        "write_deployment_bundle",
        write_bundle,
    )

    magia_example.main()

    assert calls["model_path"] == magia_example.DEFAULT_MODEL_PATH
    assert calls["written_bundle"] is bundle
    assert calls["bundle_path"] == (
        tmp_path / "generated" / "magia_example.bundle.json"
    )
    assert calls["initializers_path"] == (
        tmp_path / "generated" / "magia_example.bundle.initializers.bin"
    )
    assert calls["specialized_model"] is rewritten_model
    assert calls["planned_graph"] is specialized_graph
    assert calls["specialization"] is specialization
    options = calls["planning_options"]
    assert options.execution.num_token_slots == 2
    assert options.stage_formation.max_stage_operations == 1
    assert calls["specialization_options"].enable_precision_lowering
