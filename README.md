# MAPS

MAPS is a planner for distributed execution on homogeneous multi-tile architectures. It estimates compute and transport costs, and assembles a scheduled pipeline over a physical mesh.

## Directory layout

```text
MAPS/
  arch/        Hardware and NoC abstractions
  core/        Graph, layout, tensor, and submesh IR
  hw/          Concrete device and chip descriptions
  importers/   ONNX import path
  ops/         Operation payloads, tile work, and cost models
  pipeline/    Scheduled pipeline IR, layers, stages, and JSON export
  planner/     Workload balancing, spatial mapping, constraints, and pipeline build
  transforms/  Graph decomposition and graph utility transforms
  transitions/ Inter-stage transition building and transport costing
  utils/       Pipeline JSON and mesh/submesh printing helpers
examples/      Runnable examples
tests/         Unit and integration tests
tutorials/     Short development guides
```

## Download

Clone the repository and install it in a local virtual environment:

```bash
git clone --recursive https://github.com/xilorem/MAPS
cd MAPS
python -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
```

## Run tests

Run the full test suite with:

```bash
./.venv/bin/python -m pytest -q
```

## Configure the MAGIA backend

Deployment packages use the `maps-translate` executable from the `maps-ir`
submodule. Configure and build it once:

```bash
cmake -S maps-ir -B maps-ir/build \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm
cmake --build maps-ir/build --target maps-translate
```

## Build a deployment package

The supported handoff from MAPS to the MAGIA SDK is one verified, relocatable
deployment-package directory. First run the small end-to-end smoke test:

```bash
./.venv/bin/maps package examples/simple_three_stage.onnx \
  --target magia-v2 \
  --mesh 4 \
  --token-slots 2 \
  --output generated/simple_three_stage.maps

./.venv/bin/maps package verify generated/simple_three_stage.maps
```

This exercises planning, constant packing, maps-ir lowering, verification, and
atomic publication. The CLI prints planner and packaging progress while it
works.

`--mesh 4` selects a 4x4 mesh; rectangular meshes use syntax such as
`--mesh 4x8`. `--token-slots` fixes planner-owned concurrent tensor
residency. The optional `--pipeline-token-capacity` controls the number of
runtime pipeline tokens represented by generated L2 buffers and defaults to
one.

The command imports and specializes the ONNX model, plans it once, packs its
constants, invokes one maps-ir package operation, independently verifies the
result, and publishes:

```text
generated/simple_three_stage.maps/
  manifest.json
  model.h
  model_data.c
  model_weights.S
  model.weights.bin
```

Pipeline JSON and Maps MLIR are build intermediates and are not part of the
normal package. Existing output paths are rejected rather than overwritten,
and a failed build leaves no partial package at the requested path.

The equivalent Make target is:

```bash
make package \
  MODEL=examples/simple_three_stage.onnx \
  TARGET=magia-v2 \
  MESH=4 \
  TOKEN_SLOTS=2 \
  PACKAGE=generated/simple_three_stage.maps
```

## Inspect and verify a package

Inspection first performs complete independent verification, then prints the
model, target, tensor counts, token contract, and artifact count:

```bash
./.venv/bin/maps package inspect generated/simple_three_stage.maps
```

Verification is suitable for automation and does not need the source ONNX
model:

```bash
./.venv/bin/maps package verify generated/simple_three_stage.maps
```

The verifier checks the manifest schema and ABI versions, fixed tensor
contracts, target and memory requirements, safe package-relative paths, the
exact runtime file set, byte sizes, and SHA-256 checksums. A copied package can
therefore be verified independently:

```bash
cp -a generated/simple_three_stage.maps /tmp/simple_three_stage.maps
./.venv/bin/maps package verify /tmp/simple_three_stage.maps
```

The matching Make targets are:

```bash
make package-inspect PACKAGE=generated/simple_three_stage.maps
make package-verify PACKAGE=generated/simple_three_stage.maps
```

## Execute through the MAGIA SDK

The package manifest is the runtime contract. The MAGIA SDK binds raw input
and output tensors by the names, dtypes, shapes, and byte sizes recorded there;
callers do not supply L2 addresses, generated symbols, tile IDs, or operation
identifiers.

A run description is kept outside the immutable package:

```json
{
  "schema_version": 1,
  "pipeline_tokens": 1,
  "inputs": [
    {
      "name": "x",
      "path": "inputs/x.bin",
      "sha256": "..."
    }
  ],
  "outputs": [
    {
      "name": "add_out",
      "path": "outputs/add_out.bin"
    }
  ]
}
```

The intended SDK invocation is:

```bash
magia-run --package generated/simple_three_stage.maps --run run.json
```

This repository now completes and verifies the package handoff. The
corresponding manifest consumption, operation support, and GVSOC execution
path must be present in the separately maintained MAGIA SDK before full-model
execution succeeds.

## Backend development targets

The lower-level example and translation targets remain available for backend
development:

```bash
make pipeline-bundle
make maps-translate
make maps-mlir
make magia-header
make magia-data
make magia-package-artifacts
```
