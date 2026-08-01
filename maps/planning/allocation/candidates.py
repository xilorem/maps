"""Generate feasible virtual layout candidates for a Stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from maps.hardware import Mesh, Tile, WorkSignature
from maps.graph import Node
from maps.graph import Tensor
from maps.operations.contracts import OpPayload, TileWork, find_layout_relation
from maps.planning.mapping import TensorLayout, TensorSlice, tile_tensor_slice
from maps.planning.stages import StagePlan, StageFormation


@dataclass(frozen=True)
class StageTileFacts:
    """Intrinsic facts for one virtual tile of a Stage Candidate."""

    tile_id: int
    compute_cycles: int
    permanent_l1_bytes: int


@dataclass(frozen=True)
class StageCandidate:
    """The best feasible intrinsic configuration at one fixed tile count."""

    plan: StagePlan
    tile_facts: tuple[StageTileFacts, ...]

    @property
    def stage_compute(self) -> int:
        """Return the greatest accumulated Layer compute on one virtual tile."""

        return max(fact.compute_cycles for fact in self.tile_facts)


class StageCandidateAnalyzer:
    """Lazily analyze and cache candidates within one Allocation invocation."""

    def __init__(
        self,
        stage_formation: StageFormation,
        mesh: Mesh,
        initializer_tensors: frozenset[Tensor],
        num_token_slots: int = 2,
    ) -> None:
        self._stage_formation = {
            stage_id: tuple(stage_nodes)
            for stage_id, stage_nodes in stage_formation.items()
        }
        self._device_names = {
            stage_id: tuple(
                assigned_device_name(node, mesh.tiles)
                for node in stage_nodes
            )
            for stage_id, stage_nodes in self._stage_formation.items()
        }
        self._mesh = mesh
        self._initializer_tensors = initializer_tensors
        self._num_token_slots = num_token_slots
        self._cache: dict[tuple[int, int], StageCandidate | None] = {}

    def candidate(
        self,
        stage_id: int,
        tile_count: int,
    ) -> StageCandidate | None:
        """Return the cached best feasible candidate for one Stage size."""

        key = (stage_id, tile_count)
        if key not in self._cache:
            self._cache[key] = self._analyze(
                stage_id,
                self._stage_formation[stage_id],
                tile_count,
            )
        return self._cache[key]

    def _analyze(
        self,
        stage_id: int,
        stage_nodes: tuple[Node, ...],
        tile_count: int,
    ) -> StageCandidate | None:
        submesh = representative_connected_submesh(
            self._mesh,
            stage_id,
            tile_count,
        )
        best_candidate: StageCandidate | None = None
        payloads = tuple(cast(OpPayload, node.payload) for node in stage_nodes)
        device_names = self._device_names[stage_id]
        for logical_shape in logical_shape_options(tile_count):
            layouts = resolve_stage_layouts(stage_nodes, submesh, logical_shape)
            node_tile_work = tuple(
                tuple(
                    payload.build_tile_work(
                        output_layouts=output_layouts,
                        tile=tile,
                    )
                    for tile in submesh.tiles
                )
                for payload, output_layouts in zip(payloads, layouts)
            )
            verify_stage_locality(
                stage_nodes,
                layouts,
                submesh,
                node_tile_work,
            )
            cost_models = tuple(payload.cost_model for payload in payloads)
            placement_cycles = tuple(
                int(
                    cost_model.placement_cost(
                        node=node,
                        output_layouts=output_layouts,
                    )
                )
                for node, output_layouts, cost_model in zip(
                    stage_nodes,
                    layouts,
                    cost_models,
                )
            )
            tile_facts = tuple(
                StageTileFacts(
                    tile_id=tile.tile_id,
                    compute_cycles=sum(
                        _node_cost(
                            cost_models[node_index],
                            node_tile_work[node_index][tile_index],
                            tile,
                            device_names[node_index],
                        )
                        + placement_cycles[node_index]
                        for node_index in range(len(stage_nodes))
                    ),
                    permanent_l1_bytes=permanent_l1_allocation_for_tile_work(
                        tuple(
                            work_by_tile[tile_index]
                            for work_by_tile in node_tile_work
                        ),
                        self._initializer_tensors,
                        self._num_token_slots,
                    ),
                )
                for tile_index, tile in enumerate(submesh.tiles)
            )
            if any(
                fact.permanent_l1_bytes > tile.memory.size
                for fact, tile in zip(tile_facts, submesh.tiles)
            ):
                continue
            candidate = StageCandidate(
                plan=StagePlan(
                    stage_id=stage_id,
                    tile_count=tile_count,
                    logical_shape=logical_shape,
                    nodes=stage_nodes,
                    node_output_layouts=layouts,
                    device_names=device_names,
                ),
                tile_facts=tile_facts,
            )
            if best_candidate is None or (
                candidate.stage_compute,
                candidate.plan.logical_shape[1],
            ) < (
                best_candidate.stage_compute,
                best_candidate.plan.logical_shape[1],
            ):
                best_candidate = candidate
        return best_candidate


def _node_cost(cost_model, tile_work, tile, device_name: str) -> int:
    return cost_model.cost(
        tile_work,
        tile,
        tile.device_by_name(device_name),
    )


def assigned_device_name(node: Node, tiles: tuple[Tile, ...]) -> str:
    """Resolve one stable Device name for a Node across homogeneous Tiles."""

    signature = WorkSignature.from_node(node)
    try:
        assigned = tuple(tile.assigned_device(signature) for tile in tiles)
    except ValueError as exc:
        raise ValueError(f"node {node.name} with {signature}: {exc}") from exc
    device_names = {device.name for device in assigned}
    if len(device_names) != 1:
        raise ValueError(
            f"node {node.name} with {signature} has inconsistent fixed Device "
            f"assignments across tiles: {sorted(device_names)}"
        )
    return assigned[0].name


def logical_shape_options(tile_count: int) -> tuple[tuple[int, int], ...]:
    """Enumerate rectangular logical shapes whose area equals ``tile_count``."""

    return tuple(
        (tile_count // height, height)
        for height in range(1, tile_count + 1)
        if tile_count % height == 0
    )


def cost_estimator(
    node: Node,
    output_layouts: tuple[TensorLayout, ...],
) -> int:
    """Estimate one node's bottleneck compute cost for virtual planning."""

    cost_model = node.payload.cost_model
    output_layout = node.payload.single_output_layout(output_layouts)
    submesh = output_layout.submesh
    tile_work = tuple(
        (
            tile,
            node.payload.build_tile_work(output_layouts=output_layouts, tile=tile),
        )
        for tile in submesh.tiles
    )
    tile_cost = max(
        (
            cost_model.cost(
                work,
                tile,
                (
                    tile.assigned_device(WorkSignature.from_node(node))
                ),
            )
            for tile, work in tile_work
        ),
        default=0,
    )
    return tile_cost + int(
        cost_model.placement_cost(node=node, output_layouts=output_layouts)
    )


def placement_cost_estimator(
    node: Node,
    output_layouts: tuple[TensorLayout, ...],
) -> int:
    """Estimate the placement-specific component of one node cost."""

    return int(
        node.payload.cost_model.placement_cost(
            node=node,
            output_layouts=output_layouts,
        )
    )


def resolve_stage_layouts(
    stage_nodes: tuple[Node, ...],
    submesh,
    logical_shape: tuple[int, int],
) -> tuple[tuple[TensorLayout, ...], ...]:
    """Resolve one coherent set of output layouts in stage execution order."""

    producer_output_by_tensor: dict[object, tuple[Node, int]] = {}
    layouts_by_node: dict[int, tuple[TensorLayout, ...]] = {}
    resolved: list[tuple[TensorLayout, ...]] = []
    for node in stage_nodes:
        payload = cast(OpPayload, node.payload)
        standalone = list(
            payload.output_layouts(submesh, logical_shape=logical_shape)
        )
        derived_by_output: dict[int, TensorLayout] = {}
        for input_index, tensor in enumerate(node.inputs):
            producer_info = producer_output_by_tensor.get(tensor)
            if producer_info is None:
                continue
            producer, producer_output_index = producer_info
            relation = find_layout_relation(
                node.payload,
                input_index=input_index,
                output_index=0,
            )
            if relation is None:
                continue
            incoming_layout = layouts_by_node[id(producer)][producer_output_index]
            derived = relation.output_layout_from_input_layout(incoming_layout)
            previous = derived_by_output.get(relation.output_index)
            if previous is not None and previous != derived:
                raise ValueError(
                    f"node {node.name} has conflicting stage-local layout relations"
                )
            derived_by_output[relation.output_index] = derived
        for output_index, derived in derived_by_output.items():
            derived.validate_for(node.outputs[output_index])
            standalone[output_index] = derived
        node_layouts = tuple(standalone)
        layouts_by_node[id(node)] = node_layouts
        resolved.append(node_layouts)
        for output_index, tensor in enumerate(node.outputs):
            producer_output_by_tensor[tensor] = (node, output_index)
    return tuple(resolved)


def verify_stage_locality(
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple[TensorLayout, ...], ...],
    submesh,
    node_tile_work: tuple[tuple[TileWork, ...], ...],
) -> None:
    """Require every local consumer read to fit its same-tile producer slice."""

    producer_by_tensor: dict[object, tuple[Node, int, tuple]] = {}
    for node_index, (node, layouts) in enumerate(
        zip(stage_nodes, node_output_layouts)
    ):
        for tile_index, tile in enumerate(submesh.tiles):
            work = node_tile_work[node_index][tile_index]
            required_by_tensor = {
                reference.tensor: reference.tensor_slice
                for reference in work.input_slices
            }
            for tensor in node.inputs:
                producer_info = producer_by_tensor.get(tensor)
                if producer_info is None:
                    continue
                producer, output_index, producer_layouts = producer_info
                produced_slice = tile_tensor_slice(
                    tensor,
                    producer_layouts[output_index],
                    tile,
                )
                required_slice = required_by_tensor[tensor]
                if not _contains(produced_slice, required_slice):
                    raise ValueError(
                        f"stage-local edge {producer.name}->{node.name} is not "
                        f"tile-local on tile {tile.tile_id}"
                    )
        for output_index, tensor in enumerate(node.outputs):
            producer_by_tensor[tensor] = (node, output_index, layouts)


def _contains(container: TensorSlice, contained: TensorSlice) -> bool:
    if container.rank != contained.rank:
        return False
    return all(
        outer.start <= inner.start
        and inner.start + inner.length <= outer.start + outer.length
        for outer, inner in zip(container.dims, contained.dims)
    )
L1_ALLOCATION_ALIGNMENT_BYTES = 16


def permanent_l1_allocation_for_stage(
    stage_nodes: tuple,
    node_output_layouts: tuple[tuple, ...],
    submesh,
    initializer_tensors: frozenset[Tensor],
    num_token_slots: int = 2,
) -> int:
    """Return the greatest permanent L1 allocation on any virtual tile."""

    return max(
        (
            permanent_l1_allocation_for_tile(
                stage_nodes,
                node_output_layouts,
                tile,
                initializer_tensors,
                num_token_slots,
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def permanent_l1_allocation_for_tile(
    stage_nodes: tuple,
    node_output_layouts: tuple[tuple, ...],
    tile,
    initializer_tensors: frozenset[Tensor],
    num_token_slots: int = 2,
) -> int:
    """Mirror the backend's monotonic, non-reusing tile-L1 allocator."""

    works = tuple(
        node.payload.build_tile_work(output_layouts=layouts, tile=tile)
        for node, layouts in zip(stage_nodes, node_output_layouts)
    )
    return permanent_l1_allocation_for_tile_work(
        works,
        initializer_tensors,
        num_token_slots,
    )


def permanent_l1_allocation_for_tile_work(
    tile_work: tuple[TileWork, ...],
    initializer_tensors: frozenset[Tensor],
    num_token_slots: int = 2,
) -> int:
    """Return permanent L1 bytes from already constructed Layer Tile Work."""

    produced_tensors = set()
    allocation_sizes = []
    for work in tile_work:
        for reference in work.input_slices:
            if reference.tensor in produced_tensors:
                continue
            slot_count = (
                1
                if _is_initializer(reference.tensor, initializer_tensors)
                else num_token_slots
            )
            allocation_sizes.append(reference.num_bytes * slot_count)

        for reference in work.output_slices:
            allocation_sizes.append(reference.num_bytes * num_token_slots)
            produced_tensors.add(reference.tensor)

    return permanent_l1_allocation_bytes(allocation_sizes)


def permanent_l1_allocation_bytes(allocation_sizes) -> int:
    """Return the final offset of the backend's monotonic L1 allocator."""

    next_offset = 0
    for allocation_size in allocation_sizes:
        next_offset = _align_to(next_offset, L1_ALLOCATION_ALIGNMENT_BYTES)
        next_offset += allocation_size
    return next_offset


def _is_initializer(
    tensor: object,
    initializer_tensors: frozenset[Tensor],
) -> bool:
    return tensor in initializer_tensors or getattr(tensor, "is_initializer", False)


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class ConnectedSubmesh:
    """A connected tile set with a separate logical row-major shape."""

    mesh: Mesh
    submesh_id: int
    tile_ids: tuple[int, ...]
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate shape, mesh membership, uniqueness, and connectivity."""

        if self.submesh_id < 0:
            raise ValueError("submesh_id must be >= 0")
        if not self.tile_ids:
            raise ValueError("tile_ids must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")
        if self.width * self.height != len(self.tile_ids):
            raise ValueError("logical shape area must match tile count")
        if len(set(self.tile_ids)) != len(self.tile_ids):
            raise ValueError("tile_ids must be unique")
        if any(not self.mesh.contains_tile_id(tile_id) for tile_id in self.tile_ids):
            raise ValueError("tile_ids must be inside the mesh")
        if len(self.tile_ids) > 1 and not _is_connected(self.tile_ids, self.mesh):
            raise ValueError("tile_ids must form one connected component")

    @property
    def num_tiles(self) -> int:
        """Return the number of physical tiles in the submesh."""

        return len(self.tile_ids)

    @property
    def tiles(self) -> tuple[Tile, ...]:
        """Resolve tile ids to mesh tile objects in logical order."""

        return tuple(self.mesh.tile_by_id(tile_id) for tile_id in self.tile_ids)

    @property
    def tile_mask(self) -> int:
        """Return a bit mask with one bit per physical tile id."""

        mask = 0
        for tile_id in self.tile_ids:
            mask |= 1 << tile_id
        return mask

    def contains_tile_id(self, tile_id: int) -> bool:
        """Return whether a physical tile belongs to this submesh."""

        return tile_id in self.tile_ids

    def intersects_tile_ids(self, tile_ids: set[int]) -> bool:
        """Return whether this submesh intersects a supplied tile set."""

        return any(tile_id in tile_ids for tile_id in self.tile_ids)

    def global_to_local(self, tile_id: int) -> tuple[int, int]:
        """Translate a physical tile id to logical row-major coordinates."""

        if tile_id not in self.tile_ids:
            raise ValueError(f"tile_id {tile_id} is not inside submesh {self.submesh_id}")
        ordinal = self.tile_ids.index(tile_id)
        return ordinal % self.width, ordinal // self.width

    def local_to_global(self, local_x: int, local_y: int) -> int:
        """Translate logical row-major coordinates to a physical tile id."""

        if local_x < 0 or local_x >= self.width:
            raise ValueError(f"local_x out of bounds: {local_x}")
        if local_y < 0 or local_y >= self.height:
            raise ValueError(f"local_y out of bounds: {local_y}")
        return self.tile_ids[local_y * self.width + local_x]


def representative_connected_submesh(
    mesh: Mesh,
    submesh_id: int,
    tile_count: int,
) -> ConnectedSubmesh:
    """Return the deterministic virtual submesh used for layout planning."""

    if tile_count <= 0 or tile_count > mesh.num_tiles:
        raise ValueError("tile_count must be in [1, mesh.num_tiles]")
    return ConnectedSubmesh(
        mesh=mesh,
        submesh_id=submesh_id,
        tile_ids=tuple(range(tile_count)),
        width=tile_count,
        height=1,
    )


def connected_submesh_placements(
    tile_count: int,
    mesh: Mesh,
    submesh_id: int,
) -> tuple[ConnectedSubmesh, ...]:
    """Enumerate connected tile placements with one logical line shape."""

    return tuple(
        ConnectedSubmesh(
            mesh=mesh,
            submesh_id=submesh_id,
            tile_ids=tile_ids,
            width=tile_count,
            height=1,
        )
        for tile_ids in _connected_tile_id_sets(tile_count, mesh)
    )


def _connected_tile_id_sets(
    tile_count: int,
    mesh: Mesh,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate unique connected physical tile sets of one size."""

    if tile_count <= 0:
        raise ValueError("tile_count must be > 0")
    if tile_count > mesh.num_tiles:
        return ()
    neighbors = {
        tile_id: _cardinal_neighbors(tile_id, mesh)
        for tile_id in range(mesh.num_tiles)
    }
    results: set[tuple[int, ...]] = set()
    seen: set[frozenset[int]] = set()

    def expand(tile_ids: frozenset[int], frontier: frozenset[int]) -> None:
        """Depth-first enumerate connected supersets from one frontier."""

        if tile_ids in seen:
            return
        seen.add(tile_ids)
        if len(tile_ids) == tile_count:
            results.add(tuple(sorted(tile_ids)))
            return
        for next_tile_id in sorted(frontier):
            next_tile_ids = tile_ids | {next_tile_id}
            next_frontier = (
                frontier | neighbors[next_tile_id]
            ) - next_tile_ids
            expand(next_tile_ids, next_frontier)

    for start_tile_id in range(mesh.num_tiles):
        expand(frozenset({start_tile_id}), neighbors[start_tile_id])
    return tuple(sorted(results))


def _cardinal_neighbors(tile_id: int, mesh: Mesh) -> frozenset[int]:
    """Return in-mesh cardinal neighbors of one tile id."""

    x, y = mesh.coords(tile_id)
    neighbors = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = x + dx
        ny = y + dy
        if mesh.contains_coord(nx, ny):
            neighbors.add(mesh.tile_id(nx, ny))
    return frozenset(neighbors)


def _is_connected(tile_ids: tuple[int, ...], mesh: Mesh) -> bool:
    """Return whether tile ids form one cardinally connected component."""

    remaining = set(tile_ids)
    frontier = {tile_ids[0]}
    visited = set()
    while frontier:
        tile_id = frontier.pop()
        if tile_id in visited:
            continue
        visited.add(tile_id)
        frontier.update(_cardinal_neighbors(tile_id, mesh) & remaining)
    return visited == remaining
