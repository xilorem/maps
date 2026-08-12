# 02 — Generate a minimal MAGIA Application

**What to build:** Make expert MAGIA code generation produce a minimal, relocatable MAGIA Application that can be copied into the existing SDK structure and built after registration. The result presents one normalized application identity, one synthetic Execution Token, a small public interface, a generated runner, a user-owned customization point, embedded packed Initializers, and clear SDK integration instructions.

**Blocked by:** 01 — Expand the Deployment and expert compiler seams

**Status:** complete

- [x] Expert code generation produces an ordinary application directory with a build definition, README, manifest, include area, source area, and data area.
- [x] The application name is normalized once to snake case and controls the application directory, build target, model-specific filenames, and C symbol prefix.
- [x] The generated public header exposes a small model interface and contains no large Execution Plan descriptor tables.
- [x] The generated runner initializes required MAGIA devices, constructs the planned execution state, initializes data, synchronizes the Mesh, and executes one synthetic Execution Token.
- [x] The initial user-owned application source provides model-namespaced input and output hooks and is clearly identified as the only customization source.
- [x] Packed Initializers remain raw binary data embedded through a relocatable assembly/CMake mechanism with model-namespaced linker symbols.
- [x] The build definition links SDK-owned runtime, HAL, drivers, and required Operation Implementations without copying SDK sources into the application.
- [x] The README explains SDK copying, `add_subdirectory` registration, the SDK build action, and which source the developer may edit.
- [x] The manifest records at least application identity, `magia-v2`, planned Mesh, ABI versions, one Execution Token, Token Slots, Tensor interfaces, memory requirements, and initial file ownership.
- [x] Repeated generation from the same input produces byte-identical generated files.
