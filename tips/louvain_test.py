import igraph

from tips import louvain


def test_two_stars_with_one_edge():
    graph = igraph.Graph(
        n=10,
        edges=[(0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7), (5, 8), (5, 9)],
    )
    clustering = louvain.louvain(graph)

    assert clustering == {frozenset([0, 1, 2, 3, 4]), frozenset([5, 6, 7, 8, 9])}
