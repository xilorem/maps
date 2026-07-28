from pathlib import Path
from types import SimpleNamespace

from examples import magia_example


def test_magia_example_builds_and_writes_one_pipeline_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = SimpleNamespace(
        name="simple_three_stage",
        stages=(),
        transitions=(),
    )
    bundle = SimpleNamespace(pipeline=pipeline)
    calls: dict[str, object] = {}

    monkeypatch.setattr(magia_example, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        magia_example,
        "build_pipeline_bundle",
        lambda model_path, mesh, options: calls.update(
            model_path=model_path,
            mesh=mesh,
            options=options,
        ) or bundle,
    )
    monkeypatch.setattr(
        magia_example,
        "validate_constraints",
        lambda pipeline, constraints: SimpleNamespace(
            is_valid=True,
            violations=(),
        ),
    )
    monkeypatch.setattr(magia_example, "print_submeshes", lambda pipeline: None)

    def write_bundle(value, pipeline_path, weights_path):
        calls["written_bundle"] = value
        calls["pipeline_path"] = pipeline_path
        calls["weights_path"] = weights_path
        return pipeline_path, weights_path

    monkeypatch.setattr(magia_example, "write_pipeline_bundle", write_bundle)

    magia_example.main()

    assert calls["model_path"] == magia_example.DEFAULT_MODEL_PATH
    assert calls["written_bundle"] is bundle
    assert calls["pipeline_path"] == tmp_path / "generated" / "magia_example.pipeline.json"
    assert calls["weights_path"] == (
        tmp_path / "generated" / "magia_example.pipeline.weights.bin"
    )
    options = calls["options"]
    assert options.execution.num_token_slots == 2
