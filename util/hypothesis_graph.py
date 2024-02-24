import itertools

import igraph
from hypothesis.strategies import booleans, composite, integers, permutations


@composite
def unweighted_graph(
    draw,
    edge_decider=booleans(),
    num_vertices=integers(min_value=5, max_value=20),
):
    """Draw an unweighted, undirected graph."""
    graph = igraph.Graph(n=draw(num_vertices))
    for i, j in itertools.combinations(range(graph.vcount()), 2):
        if draw(edge_decider):
            graph.add_edge(i, j)
    return graph


@composite
def unweighted_graph_and_subgraph(
    draw,
    graph_drawer=unweighted_graph(),
    subgraph_size=integers(min_value=5, max_value=10),
):
    """Draw an unweighted, undirected graph and a randomly chosen subgraph."""
    graph = draw(graph_drawer)
    subgraph_vertex_count = draw(subgraph_size)
    subgraph_indices = draw(permutations(range(graph.vcount())))[:subgraph_vertex_count]
    return graph, subgraph_indices
