"""Command-line entry point for maps planning and deployment workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from maps.deployment import (
    package_summary,
    validate_deployment_package,
    write_deployment_package,
)
from maps.deployment.serialization import write_execution_plan_json
from maps.graph import import_onnx_model, run_graph_rewrites
from maps.planning import (
    ExecutionContract,
    PlacementOptions,
    PlanningOptions,
    StageFormationOptions,
    plan,
)
from maps.target import SpecializationOptions, magia, n300d


_TARGETS: dict[str, ModuleType] = {
    "magia": magia,
    "n300d": n300d,
}


def _mesh(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", maxsplit=1)
    try:
        width = int(parts[0])
        height = int(parts[1]) if len(parts) == 2 else width
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mesh must be N or WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("mesh dimensions must be positive")
    return width, height


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maps")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="plan a model and write an Execution Plan")
    commands.add_parser("package", help="build, inspect, or verify a deployment package")
    return parser


def _add_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", choices=tuple(_TARGETS), default="magia")


def _add_planning_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", type=Path)
    _add_target(parser)
    parser.add_argument("--mesh", type=_mesh)
    parser.add_argument("--token-slots", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)


def _plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maps plan")
    _add_planning_arguments(parser)
    parser.add_argument("--max-stage-nodes", type=int, default=0)
    return parser


def _package_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maps package")
    _add_planning_arguments(parser)
    parser.add_argument("--pipeline-token-capacity", type=int, default=1)
    parser.add_argument("--maps-translate", type=Path)
    return parser


def _package_read_parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"maps package {action}")
    parser.add_argument("package", type=Path)
    return parser


def _mesh_options(shape: tuple[int, int] | None) -> dict[str, int]:
    if shape is None:
        return {}
    return {"width": shape[0], "height": shape[1]}


def _run_package(arguments: list[str]) -> int:
    if arguments and arguments[0] in {"inspect", "verify"}:
        action = arguments[0]
        options = _package_read_parser(action).parse_args(arguments[1:])
        manifest = validate_deployment_package(options.package)
        if action == "inspect":
            print(package_summary(manifest))
        else:
            print(f"Valid deployment package: {options.package}")
        return 0

    options = _package_build_parser().parse_args(arguments)
    dimensions = _mesh_options(options.mesh)
    output = write_deployment_package(
        options.model,
        options.output,
        target=options.target,
        mesh_width=dimensions.get("width", magia.MESH_WIDTH),
        mesh_height=dimensions.get("height", magia.MESH_HEIGHT),
        num_token_slots=options.token_slots,
        pipeline_token_capacity=options.pipeline_token_capacity,
        maps_translate=options.maps_translate,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Deployment package: {output}")
    return 0


def _run_plan(arguments: list[str]) -> int:
    options = _plan_parser().parse_args(arguments)
    target = _TARGETS[options.target]
    mesh = target.build_mesh(**_mesh_options(options.mesh))
    imported = import_onnx_model(options.model)
    rewritten = run_graph_rewrites(imported)
    specialization = target.specialize(rewritten, mesh, SpecializationOptions())
    execution_plan = plan(
        specialization.model.graph,
        mesh,
        PlanningOptions(
            execution=ExecutionContract(num_token_slots=options.token_slots),
            stage_formation=StageFormationOptions(
                max_stage_nodes=options.max_stage_nodes
            ),
            placement=PlacementOptions(print_placement=False),
            print_execution_plan_cost=False,
        ),
    )
    write_execution_plan_json(execution_plan, options.output)
    print(f"Execution Plan: {options.output}")
    return 0


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser = _build_parser()
    try:
        commands = {
            "plan": _run_plan,
            "package": _run_package,
        }
        if arguments and (command := commands.get(arguments[0])) is not None:
            return command(arguments[1:])
        parser.parse_args(arguments)
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    sys.exit(main())
