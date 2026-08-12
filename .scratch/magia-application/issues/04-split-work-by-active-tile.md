# 04 — Split generated work by active physical tile

**What to build:** Organize generated Execution Plan data for direct tile-level debugging. Emit one deterministic C source for each active physical tile, keep shared L2 data at model level, and dispatch runtime execution to the correct tile plan by physical tile ID.

**Blocked by:** 02 — Generate a minimal MAGIA Application

**Status:** complete

- [x] Every active physical tile has exactly one generated source named with only its zero-padded physical tile ID.
- [x] Inactive physical tiles have no generated tile source.
- [x] Tile coordinates do not appear in tile filenames.
- [x] Each tile source contains that tile's Slices and L1 offsets, Token Slot counts and strides, Layers and Operation Implementation descriptors, Transition receives and sends, collective participation, synchronization facts, and tile-plan construction.
- [x] Shared Runtime Input, graph-output, intermediate, and Initializer-related L2 storage remains in model-level generated source and is not duplicated across tile sources.
- [x] Model-level code dispatches from the runtime physical tile ID to every generated active-tile plan constructor.
- [x] Tile files are emitted in deterministic physical-ID order and remain byte-identical across equivalent builds.
- [x] Representative tests cover active and inactive tiles, Transitions, Layers, Token Slots, and a collective without snapshotting unrelated emitter formatting.
