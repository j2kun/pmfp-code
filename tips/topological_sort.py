"""Compute a topological sort of a directed, acyclic graph."""

from collections import deque

# The key is the node's unique id, and the value for node N is the list of node
# ids that depend on N.
Graph = dict[int, list[int]]


def topological_sort(dag: Graph) -> list[int]:
    """Compute a topological sorting of the input graph.

    The edges in the graph are interpreted so that there is an edge from node i
    to node j if node j depends on node i. An edge from node i to node j is
    represented in the input by j being a member of dag[i].

    The input graph is required to have no cycles, i.e., be a directed, acyclic
    graph (DAG).

    Arguments:
        dag: a dictionary mapping node ids to their downstream dependencies.

    Returns:
        A list of node ids in topological sorted order, meaning upstream
        dependencies come first in the list.

    Raises:
        KeyError: if a node id cannot be found.
        ValueError: if the input graph contains cycles.
    """
    sorted_nodes: deque = deque([], maxlen=len(dag))
    visited_overall = set()
    visited_single_dfs = set()
    starting_nodes = list(dag.keys())

    def dfs_helper(node):
        visited_single_dfs.add(node)
        for dependent in dag[node]:
            if dependent in visited_single_dfs and dependent not in visited_overall:
                raise ValueError(
                    f"Input graph {dag} is not a DAG! "
                    f"Node {dependent} is part of a loop",
                )
            if dependent in visited_overall:
                continue
            dfs_helper(dependent)
        visited_overall.add(node)
        sorted_nodes.appendleft(node)

    # Because we select unvisited notes essentially at random, and because the
    # graph not be a single connected component, we must admit the possibility
    # that we need to run multiple DFS loops to find all the nodes.
    while starting_nodes:
        starting_node = starting_nodes.pop()
        if starting_node in visited_overall:
            continue
        visited_single_dfs = set()
        dfs_helper(starting_node)

    return list(sorted_nodes)
