"""Command-line entry point for MAPS deployment workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from MAPS.deployment import (
    package_summary,
    validate_deployment_package,
    write_deployment_package,
)
from MAPS.hw.chips import magia_mesh
from MAPS.planner.plan import build_execution_plan


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


def _plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maps plan")
    parser.add_argument("model", type=Path)
    parser.add_argument("--mesh", type=_mesh, default=(32, 32))
    parser.add_argument("--token-slots", type=int, default=2)
    parser.add_argument("--max-stage-nodes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _package_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maps package")
    parser.add_argument("model", type=Path)
    parser.add_argument("--target", default="magia-v2")
    parser.add_argument("--mesh", type=_mesh, default=(32, 32))
    parser.add_argument("--token-slots", type=int, default=2)
    parser.add_argument("--pipeline-token-capacity", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps-translate", type=Path)
    return parser


def _package_read_parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"maps package {action}")
    parser.add_argument("package", type=Path)
    return parser


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
    width, height = options.mesh
    output = write_deployment_package(
        options.model,
        options.output,
        target=options.target,
        mesh_width=width,
        mesh_height=height,
        num_token_slots=options.token_slots,
        pipeline_token_capacity=options.pipeline_token_capacity,
        maps_translate=options.maps_translate,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Deployment package: {output}")
    return 0


def _run_plan(arguments: list[str]) -> int:
    options = _plan_parser().parse_args(arguments)
    width, height = options.mesh
    build_execution_plan(
        options.model,
        magia_mesh(width=width, height=height),
        output_json_path=options.output,
        max_stage_nodes=options.max_stage_nodes,
        num_token_slots=options.token_slots,
    )
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
