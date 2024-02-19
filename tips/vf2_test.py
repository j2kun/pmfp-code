import igraph
import pytest

from tips.vf2 import contains_isomorphic_subgraph


@pytest.mark.parametrize(
    "edge_list",
    [
        [(0, 1), (0, 2)],
        [(1, 0), (1, 2)],
        [(2, 0), (2, 1)],
    ],
)
def test_star_graph_vs_smaller_star(edge_list):
    graph = igraph.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])

    query = igraph.Graph(n=3, edges=edge_list)
    assert contains_isomorphic_subgraph(graph, query)


def test_graph_vs_itself():
    graph = igraph.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])
    assert contains_isomorphic_subgraph(graph, graph)


def test_graph_without_triangle():
    graph = igraph.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])
    query = igraph.Graph(n=3, edges=[(0, 1), (0, 2), (1, 2)])
    assert not contains_isomorphic_subgraph(graph, query)
