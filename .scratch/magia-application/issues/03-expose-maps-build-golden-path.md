# 03 — Expose the `maps build` golden path

**What to build:** Let an ordinary user build the minimal MAGIA Application directly from an Imported Model without seeing Deployment Bundle JSON, maps MLIR, compiler translation names, or temporary files. Expose the same intent through the high-level Python API and finish with concise SDK handoff guidance.

**Blocked by:** 02 — Generate a minimal MAGIA Application

**Status:** complete

- [x] `maps build <model>` composes import, Graph Rewrites, Target Specialization, Planning, Deployment Bundle construction, and MAGIA Application generation.
- [x] The default Target spelling is consistently `magia-v2`.
- [x] The default output is `build/<normalized-model-name>/`.
- [x] `--name` overrides the derived application identity consistently across paths, build identity, manifest identity, and generated C symbols.
- [x] `--output` overrides the application directory without changing application identity unless `--name` is also supplied.
- [x] The high-level Python API exposes `build_application` with the same observable behavior.
- [x] Temporary Deployment Bundle, packed-Initializer staging, and maps MLIR files are absent from the published application and normal command output.
- [x] Backend or validation failure leaves no partially published application.
- [x] Successful output reports application location, Target, Mesh, Execution Token count, active-tile count, and the SDK copy/registration action.
- [x] CLI and Python acceptance tests exercise the complete public build seam.
