import igraph
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, floats, integers

from tips import louvain
from util.hypothesis_graph import unweighted_graph

# We tests against all parameters, but more loosely check that Louvain produces
# a partition that maximizes the objective as well as the expected partition.
# For some parameter choices, the resulting community set will look very different,
# but even for the main parameter choices, there can be multiple valid communities.
SPARSE_CUT_RESOLUTIONS = [0.8, 0.9, 0.99]
DENSE_SUBGRAPH_RESOLUTIONS = [0.01, 0.1, 0.2, 0.3]
ALL_RESOLUTIONS = SPARSE_CUT_RESOLUTIONS + DENSE_SUBGRAPH_RESOLUTIONS


def run_louvain(graph, resolution):
    """Run louvain and normalize the results for easier test assertions."""
    output = louvain.louvain(graph, resolution)
    return {frozenset(c.vertices) for c in sorted(output, key=lambda c: c.id)}


def check_equivalent(expected, actual, resolution, graph):
    """Assert that the objectives for the expected and actual communities are the
    same."""
    actual_objective = louvain._compute_objective(graph, resolution)

    g2 = graph.copy()
    community_vector = [0] * g2.vcount()
    for i, community in enumerate(expected):
        for vertex in community:
            community_vector[vertex] = i
    g2.vs["community"] = community_vector
    expected_objective = louvain._compute_objective(graph, resolution)

    assert expected_objective == actual_objective, (
        f"Expected {expected} with objective {expected_objective}, but got "
        f"{actual} with objective {actual_objective} for resolution {resolution}. "
        f"Note that the two communities may differ so long as their objectives agree."
    )


@pytest.mark.parametrize("resolution", ALL_RESOLUTIONS)
def test_two_stars_with_one_edge(resolution):
    graph = igraph.Graph(
        n=10,
        edges=[(0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7), (5, 8), (5, 9)],
    )
    actual = run_louvain(graph, resolution)
    expected = {frozenset([0, 1, 2, 3, 4]), frozenset([5, 6, 7, 8, 9])}
    check_equivalent(expected, actual, resolution, graph)


@pytest.mark.parametrize("resolution", ALL_RESOLUTIONS)
def test_two_near_cliques_sparsely_connected(resolution):
    clique_1 = [
        (0, 1),
        (0, 2),
        (0, 3),
        # (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        # (2, 3),
        (2, 4),
        (3, 4),
    ]
    clique_2 = [
        (5, 6),
        # (5, 7),
        (5, 8),
        (5, 9),
        # (6, 7),
        (6, 8),
        (6, 9),
        (7, 8),
        (7, 9),
        # (8, 9),
    ]
    crossing_edges = [
        (0, 5),
        (2, 7),
        (3, 9),
    ]
    graph = igraph.Graph(n=10, edges=clique_1 + clique_2 + crossing_edges)
    actual = run_louvain(graph, resolution)
    expected = {frozenset([0, 1, 2, 3, 4]), frozenset([5, 6, 7, 8, 9])}
    check_equivalent(expected, actual, resolution, graph)


@given(unweighted_graph(), floats(min_value=0.01, max_value=0.99))
def test_undirected_graph_converges(graph, resolution):
    run_louvain(graph, resolution)


@given(
    floats(min_value=0.7, max_value=0.99),
    floats(min_value=0.01, max_value=0.3),
    floats(min_value=0.01, max_value=0.99),
)
@settings(deadline=None, max_examples=500)
def test_stochastic_block_model(in_prob, out_prob, resolution):
    graph = igraph.Graph.SBM(
        100,
        block_sizes=[50, 50],
        pref_matrix=[[in_prob, out_prob], [out_prob, in_prob]],
    )
    actual = run_louvain(graph, resolution)
    expected = {frozenset(range(50)), frozenset(range(50, 100))}
    check_equivalent(expected, actual, resolution, graph)


@composite
def weighted_sbm(
    draw,
    in_prob=floats(min_value=0.7, max_value=0.99),
    out_prob=floats(min_value=0.01, max_value=0.3),
    num_vertices=integers(min_value=20, max_value=40),
    v_weight=floats(min_value=1, max_value=5),
    in_edge_weight=floats(min_value=1, max_value=5),
    out_edge_weight=floats(min_value=0.01, max_value=1),
):
    n = draw(num_vertices)
    if n % 2 == 1:
        n += 1

    p_in = draw(in_prob)
    p_out = draw(out_prob)
    graph = igraph.Graph.SBM(
        n,
        block_sizes=[n // 2, n // 2],
        pref_matrix=[[p_in, p_out], [p_out, p_in]],
    )

    graph.vs["weight"] = [draw(v_weight) for _ in range(n)]
    for edge in graph.es:
        in_edge = (edge.source < n // 2 and edge.target < n // 2) or (
            edge.source >= n // 2 and edge.target >= n // 2
        )
        edge["weight"] = draw(in_edge_weight if in_edge else out_edge_weight)

    return graph


i = 0


@given(weighted_sbm(), floats(min_value=0.01, max_value=0.99))
@settings(max_examples=500, deadline=None)
def test_weighted_sbm(graph, resolution):
    global i
    print(i)
    i += 1
    actual = run_louvain(graph, resolution)
    n = len(graph.vs)
    expected = {frozenset(range(n // 2)), frozenset(range(n // 2, n))}
    check_equivalent(expected, actual, resolution, graph)
