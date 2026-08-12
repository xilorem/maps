# 01 — Expand the Deployment and expert compiler seams

**What to build:** Add the precise Execution Plan, Deployment Bundle, plan-import, and target-code-generation interfaces beside the current interfaces. Preserve current behavior while giving later tickets stable seams that distinguish plain Execution Plan serialization, serialized Deployment Bundles with packed Initializers, maps MLIR import, and Target application generation.

**Blocked by:** None — can start immediately

**Status:** complete

- [x] The public Python API can write a plain Execution Plan through `write_execution_plan`.
- [x] The public Python API can write and independently validate a serialized Deployment Bundle through `write_deployment_bundle` and `validate_deployment_bundle`.
- [x] Plain Execution Plan serialization remains observably distinct from Deployment Bundle serialization and does not gain packed Initializer or provenance metadata.
- [x] The serialized Deployment Bundle retains deterministic provenance, Initializer metadata, packed bytes, alignment, checksums, and L2-capacity validation.
- [x] An expert Execution Plan import entry point converts a serialized Execution Plan into maps MLIR.
- [x] An expert target-code-generation entry point accepts maps MLIR and an output location without relying on an unused translation output.
- [x] Existing public workflows and tests remain green during this expand phase.
- [x] Focused tests cover every new interface through its public seam rather than private helper functions.
