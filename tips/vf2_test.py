import igraph
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tips.vf2 import contains_isomorphic_subgraph
from util.hypothesis_graph import unweighted_graph, unweighted_graph_and_subgraph


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


def test_graph_contains_itself():
    graph = igraph.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])
    assert contains_isomorphic_subgraph(graph, graph)


def test_graph_without_triangle():
    graph = igraph.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])
    query = igraph.Graph(n=3, edges=[(0, 1), (0, 2), (1, 2)])
    assert not contains_isomorphic_subgraph(graph, query)


@pytest.mark.parametrize(
    "query_n,edge_list",
    [
        (3, [(0, 1)]),  # an isolated node
        (4, [(1, 0), (2, 3)]),  # two isolated edges
    ],
)
def test_graph_with_subgraph_but_not_induced(query_n, edge_list):
    graph = igraph.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])
    query = igraph.Graph(n=query_n, edges=edge_list)
    assert not contains_isomorphic_subgraph(graph, query)


@given(
    unweighted_graph_and_subgraph(
        graph_drawer=unweighted_graph(
            num_vertices=st.integers(min_value=10, max_value=10),
        ),
        subgraph_size=st.integers(min_value=6, max_value=6),
    ),
    st.permutations(range(6)),
)
def test_random_graph_contains_random_subgraph(
    graph_and_subgraph,
    subgraph_permutation,
):
    graph, subgraph_indices = graph_and_subgraph
    subgraph = graph.induced_subgraph(subgraph_indices)
    subgraph = subgraph.permute_vertices(subgraph_permutation)
    # import ipdb; ipdb.set_trace()
    assert contains_isomorphic_subgraph(graph, subgraph)
