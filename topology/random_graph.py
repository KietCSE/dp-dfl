"""Random undirected graph for DFL topology."""

import random
from typing import Dict, Set


def create_random_graph(n_nodes: int, n_neighbors: int, seed: int = 42) -> Dict[int, Set[int]]:
    """
    Create random undirected graph where each node has ~n_neighbors neighbors.
    Ensures symmetry: if i->j then j->i.
    """
    assert n_neighbors < n_nodes, f"n_neighbors ({n_neighbors}) must be < n_nodes ({n_nodes})"
    rng = random.Random(seed)
    graph: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}

    for i in range(n_nodes):
        while len(graph[i]) < n_neighbors:
            j = rng.randint(0, n_nodes - 1)
            if j != i and j not in graph[i]:
                graph[i].add(j)
                graph[j].add(i)

    return graph
