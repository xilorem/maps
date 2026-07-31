"""Generate feasible virtual layout candidates for a stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from MAPS.arch import Mesh
from MAPS.core.graph import Node
from MAPS.core.tensor import Tensor
from MAPS.ops.common.payload import OpPayload
from MAPS.planner.contracts.stages import StagePlan, StageSelection
from MAPS.planner.workload.layouts import resolve_stage_layouts, verify_stage_locality
from MAPS.planner.workload.memory import permanent_l1_allocation_for_tile_work
from MAPS.planner.workload.submesh import representative_connected_submesh


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
    """Lazily analyze and cache candidates within one workload invocation."""

    def __init__(
        self,
        stage_selection: StageSelection,
        mesh: Mesh,
        initializer_tensors: frozenset[Tensor],
        num_token_slots: int = 2,
    ) -> None:
        self._stage_selection = {
            stage_id: tuple(stage_nodes)
            for stage_id, stage_nodes in stage_selection.items()
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
                self._stage_selection[stage_id],
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
                        cost_models[node_index].cost(
                            node_tile_work[node_index][tile_index],
                            tile,
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


def logical_shape_options(tile_count: int) -> tuple[tuple[int, int], ...]:
    """Enumerate rectangular logical shapes whose area equals ``tile_count``."""

    return tuple(
        (tile_count // height, height)
        for height in range(1, tile_count + 1)
        if tile_count % height == 0
    )
