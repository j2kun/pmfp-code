"""An algorithm to construct a stitch plan for an embroidery machine.

Based on https://github.com/inkstitch/inkstitch/blob/main/lib/stitches/auto_fill.py
"""

from collections import deque
from itertools import groupby
from itertools import pairwise  # type: ignore
from typing import Any
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from shapely import geometry as geo
import networkx

Point = Tuple[float, float]
GratingSegment = Tuple[Point, Point]
Edge = Tuple[Point, Point]


def outlines(shape: geo.Polygon) -> List[geo.LineString]:
    outlines = shape.boundary
    if isinstance(outlines, geo.MultiLineString):
        return outlines.geoms
    return [outlines]  # only other option is a single LineString


def sorted_grouped_by_outline(graph: networkx.MultiGraph) -> Dict[int, List[Any]]:
    nodes = list(graph.nodes(data=True))

    def sort_key(node_and_data):
        _, data = node_and_data
        return (data["outline"], data["order"])

    def grouping_key(node_and_data):
        _, data = node_and_data
        return data["outline"]

    nodes.sort(key=sort_key)
    return {
        outline_index: [node for node, _ in nodes]
        for outline_index, nodes in groupby(nodes, key=grouping_key)
    }


def pairwise_cyclic(itr):
    return list(pairwise(itr)) + [(itr[-1], itr[0])]


def pick_edge(edges, segment_edges):
    """Pick an edge to traverse next."""
    # The sort key is as follows: first prefer to always take a segment edge if
    # possible, then prefer the edge that is closest to an unvisited grating
    # segment, where closest is geometrically. Closest could be defined by
    # traversing graph edges instead, but this is good enough.
    def dist_to_closest_segment(edge):
        source, target, _ = edge
        edge_line = geo.LineString([source, target])
        return min(
            edge_line.distance(geo.LineString([e[0], e[1]])) for e in segment_edges
        )

    values = {e: 0 if e[2] == "segment" else dist_to_closest_segment(e) for e in edges}
    output = min(values, key=values.get)
    return output


def find_stitch_path(
    shape: geo.Polygon,
    grating_segments: Iterable[GratingSegment],
    starting_point: Point,
) -> List[Edge]:
    graph = networkx.MultiGraph()
    for segment in grating_segments:
        graph.add_edge(segment[0], segment[1], key="segment")

    # In order to add edges between grating segment end points, we need to
    # visit the graph nodes in order around the boundary of the shape. However,
    # the shape can be arbitrarily complex: non-convex and with holes in the
    # interior. The shapely library handles the geometry of distinguishing
    # the multiple boundary/hole outlines.
    shape_outlines = list(enumerate(outlines(shape)))
    for node in graph.nodes():
        pt = geo.Point(node)

        def distance(outline_index):
            index, outline = outline_index
            return outline.distance(pt)

        # All points should lie exactly on an outline, but floating point approximations
        # may break an exact intersection check.
        index, outline = min(shape_outlines, key=distance)
        # add_node just updates node data when the input node already exists
        # (as determined by its hash). `outline.project(pt)` returns the
        # distance along the outline from the start of the outline's path to the
        # closest point on the outline to `pt`.
        graph.add_node(node, outline=index, order=outline.project(pt))

    for nodes in sorted_grouped_by_outline(graph).values():
        for i, (node1, node2) in enumerate(pairwise_cyclic(nodes)):
            graph.add_edge(node1, node2, key="outline")
            # This extra edge ensures every node in the graph has degree 4, and
            # hence that an Eulerian path exists.
            if i % 2 == 0:
                graph.add_edge(node1, node2, key="extra")

    path: Deque[Edge] = deque([])
    vertex_stack = [starting_point]
    last_vertex = None

    while vertex_stack:
        current_vertex = vertex_stack[-1]
        if graph.degree(current_vertex) == 0:
            if last_vertex:
                path.appendleft((current_vertex, last_vertex))
            last_vertex = current_vertex
            vertex_stack.pop()
        else:
            # This could be made more efficient by keeping track of the
            # remaining segment edges as you go.
            segment_edges = {e for e in graph.edges(keys=True) if e[2] == "segment"}
            if not segment_edges:
                graph.clear_edges()
                continue

            source, target, key = pick_edge(
                graph.edges(current_vertex, keys=True), segment_edges,
            )
            if target:
                vertex_stack.append(target)
                graph.remove_edge(source, target, key=key)

    return list(path)
