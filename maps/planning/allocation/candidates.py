"""Generate feasible virtual layout candidates for a Stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from maps.hardware import Mesh, WorkSignature
from maps.graph import Node
from maps.graph import Tensor
from maps.operations.contracts import OpPayload
from maps.planning.stages import StagePlan, StageFormation
from maps.planning.allocation.layouts import resolve_stage_layouts, verify_stage_locality
from maps.planning.allocation.memory import permanent_l1_allocation_for_tile_work
from maps.planning.allocation.submesh import representative_connected_submesh


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


def assigned_device_name(node: Node, tiles: tuple) -> str:
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
