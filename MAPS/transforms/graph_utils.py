"""Migration bridge to Graph-owned edge construction."""

from maps.graph.graph_utils import (
    add_generated_tensor,
    build_graph_edges_from_nodes,
    reserve_generated_node_name,
)

__all__ = [
    "add_generated_tensor",
    "build_graph_edges_from_nodes",
    "reserve_generated_node_name",
]
