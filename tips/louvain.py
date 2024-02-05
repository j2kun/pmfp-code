import itertools


def louvain(graph):
    """Partition the graph into communities using the sequential Louvain method.

    Args:
        graph: An igraph.Graph object.

    Returns:
        A set of sets of integers, where each set represents a community.
    """
    vertices = list(range(graph.vcount()))
    graph.vs["community"] = vertices

    output = set()
    for k, g in itertools.groupby(
        sorted(graph.vs, key=lambda v: v["community"]),
        key=lambda v: v["community"],
    ):
        output.add(frozenset([v.index for v in g]))

    return output
