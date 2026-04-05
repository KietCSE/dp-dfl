"""N-regular undirected graph for DFL topology."""

from typing import Dict, Set


def create_regular_graph(n_nodes: int, n_neighbors: int, seed: int = 42) -> Dict[int, Set[int]]:
    """
    Create a deterministic N-regular undirected graph.

    Every node has exactly n_neighbors neighbors (degree = n_neighbors).
    Uses circular lattice: node i connects to i±1, i±2, ..., i±(n_neighbors//2).

    Requires n_neighbors to be even (for undirected regularity) and < n_nodes.
    Same (n_nodes, n_neighbors) always produces the same graph regardless of seed.
    """
    if n_neighbors >= n_nodes:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be < n_nodes ({n_nodes})"
        )
    if (n_nodes * n_neighbors) % 2 != 0:
        raise ValueError(
            f"n_nodes * n_neighbors must be even, got {n_nodes} * {n_neighbors} = {n_nodes * n_neighbors}. "
            f"Either n_nodes or n_neighbors must be even."
        )

    graph: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    half = n_neighbors // 2

    # Circular lattice: node i connects to i±1, i±2, ..., i±half
    for i in range(n_nodes):
        for k in range(1, half + 1):
            j = (i + k) % n_nodes
            graph[i].add(j)
            graph[j].add(i)

    # Odd degree: connect node i to node i + n_nodes//2 (diametrically opposite)
    if n_neighbors % 2 != 0:
        for i in range(n_nodes // 2):
            j = i + n_nodes // 2
            graph[i].add(j)
            graph[j].add(i)

    return graph


# Backward-compatible alias
create_random_graph = create_regular_graph
