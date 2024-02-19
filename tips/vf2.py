import itertools
from typing import Iterable, Optional

import igraph
from bidict import bidict

# A Mapping is a partial function from the vertices of the query graph to the
# vertices of the target graph. It represents a potential matched isomorphism. The
# keys are indices of vertices in the subgraph query, and the values are indices
# of vertices in the target graph the subgraph is sought within.
Mapping = bidict[int, int]
Extension = tuple[int, int]


def flatten(iterable):
    return set(itertools.chain.from_iterable(iterable))


def strict_neighbors(graph: igraph.Graph, vertices: Iterable[int]) -> set[int]:
    """Return neighbors of any vertex in `vertices`, excluding  `vertices`."""
    if not vertices:
        return set()
    inclusive_neighbors = flatten(graph.neighborhood(vertices, order=1, mindist=0))
    return inclusive_neighbors - set(vertices)


def generate_extensions(
    graph: igraph.Graph,
    query: igraph.Graph,
    mapping: Mapping,
) -> list[Extension]:
    """Generate a list of the next extensions of `mapping` to try.

    Note that the extensions chosen correspond to specifying a preference order over
    which nodes to try to add to the mapping next. This "matching order" is noted in the
    literature as a cornerstone of VF2's optimizations.

    For this demonstration, I use the trivial matching order of selecting unmapped
    neighbors of currently mapped vertices. As such, this algorithm is quite slow. For a
    more sophisticated matching order, see

    VF2++—An improved subgraph isomorphism algorithm; Alpár Jüttner, Péter Madarasi;
    https://doi.org/10.1016/j.dam.2018.02.018
    """
    unmapped_graph_neighbors = strict_neighbors(graph, mapping.values())
    unmapped_query_neighbors = strict_neighbors(query, mapping.keys())
    if len(unmapped_graph_neighbors) > 0 and len(unmapped_query_neighbors) > 0:
        return [
            (u, v) for u in unmapped_query_neighbors for v in unmapped_graph_neighbors
        ]

    unmapped_graph_vertices = graph.vs.select(
        lambda w: w.index not in mapping.values(),
    ).indices
    unmapped_query_vertices = query.vs.select(lambda w: w.index not in mapping).indices
    return [(u, v) for u in unmapped_query_vertices for v in unmapped_graph_vertices]


def is_consistent(
    graph: igraph.Graph,
    query: igraph.Graph,
    mapping: Mapping,
    extension: Extension,
) -> bool:
    """Check if `extension` is consistent with `mapping`.

    Note that this consistency check can be tweaked depending on the problem. For
    example, if the graphs are molecules with labels representing atom types, the
    consistency check could assert that the target and source of the extension must have
    the same label.
    """
    u, v = extension
    for u_neighbor in query.neighbors(u):
        if u_neighbor in mapping and not graph.are_connected(v, mapping[u_neighbor]):
            # print(f"\tInconsistent mapping {extension}")
            return False

    for v_neighbor in graph.neighbors(v):
        if v_neighbor in mapping.values() and not query.are_connected(
            u,
            mapping.inverse[v_neighbor],
        ):
            # print(f"\tInconsistent mapping {extension}")
            return False

    # print(f"\tConsistent mapping {extension}")
    return True


def should_cut(
    graph: igraph.Graph,
    query: igraph.Graph,
    mapping: Mapping,
    extension: Extension,
) -> bool:
    """Check if `extension` should be cut from the search.

    Generally this routine would contain a suite of fast heuristics to help prune the
    search space.
    """
    u, v = extension

    unmapped_graph_neighbors = strict_neighbors(graph, mapping.values())
    unmapped_graph_neighbors_and_mapping = flatten(
        graph.neighborhood(
            list(mapping.values()),
            order=1,
            mindist=0,  # include mapping vertices
        ),
    )
    unmapped_graph_vertices_not_adjacent_to_mapping = set(
        graph.vs.select(
            lambda w: w.index not in unmapped_graph_neighbors_and_mapping,
        ).indices,
    )
    v_neighbors = set(graph.neighbors(v))

    unmapped_query_neighbors = strict_neighbors(query, mapping.keys())
    unmapped_query_neighbors_and_mapping = flatten(
        query.neighborhood(
            list(mapping.keys()),
            order=1,
            mindist=0,  # include mapping vertices
        ),
    )
    unmapped_query_vertices_not_adjacent_to_mapping = set(
        query.vs.select(
            lambda w: w.index not in unmapped_query_neighbors_and_mapping,
        ).indices,
    )
    u_neighbors = set(query.neighbors(u))

    if len(v_neighbors & unmapped_graph_neighbors) < len(
        u_neighbors & unmapped_query_neighbors,
    ) or len(v_neighbors & unmapped_graph_vertices_not_adjacent_to_mapping) < len(
        u_neighbors & unmapped_query_vertices_not_adjacent_to_mapping,
    ):
        # print(f"\tCutting {extension}")
        return True

    # other heuristics...

    # print(f"\tNot cutting {extension}")
    return False


# The main routine for contains_isomorphic_subgraph, after being initialized
# with an empty mapping and called recursively.
def vf2(
    graph: igraph.Graph,
    query: igraph.Graph,
    mapping: Mapping,
) -> Optional[Mapping]:
    if len(mapping) == len(query.vs):
        return mapping

    candidates = generate_extensions(graph, query, mapping)
    for extension in candidates:
        # print(f"Trying to map {extension}")
        if is_consistent(graph, query, mapping, extension) and not should_cut(
            graph,
            query,
            mapping,
            extension,
        ):
            # print(f"\tAdding mapping {extension}")
            source, target = extension
            mapping[source] = target
            result = vf2(graph, query, mapping)
            if result:
                return result
            # print(f"\tRemoving mapping {extension}")
            del mapping[source]

    return None


def contains_isomorphic_subgraph(graph: igraph.Graph, query: igraph.Graph) -> bool:
    """Check if `query` is isomorphic to a subgraph of `graph`."""
    result = vf2(graph, query, bidict())
    return result is not None


if __name__ == "__main__":
    import random

    graph = igraph.Graph.Erdos_Renyi(n=50, p=0.1)
    subgraph_indices = random.sample(range(graph.vcount()), 15)
    subgraph = graph.induced_subgraph(subgraph_indices)
    result = contains_isomorphic_subgraph(graph, subgraph)
    assert result, "incorrect!"
