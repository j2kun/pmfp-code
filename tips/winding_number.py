from dataclasses import dataclass
from typing import List

from more_itertools import pairwise


@dataclass(frozen=True)
class Point:
    x: float
    y: float


# An ordered list of vertices whose first and last entry are the same
Polygon = List[Point]


def is_left(point: Point, l0: Point, l1: Point) -> float:
    """Determine if a point is to the left of the line through l0 and l1.

    Returns 0 if the point is on the line, a positive number if the point is to the left
    of the line, and a negative number if the point is on the right of the line.

    The line is oriented in the direction from l0 to l1.
    """
    return (l1.x - l0.x) * (point.y - l0.y) - (point.x - l0.x) * (l1.y - l0.y)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Determine if a polygon contains a point.

    This implementation uses the winding number of the polygon around the point.
    The point is outside the polygon if and only if the winding number is zero.

    Note that this algorithm works even when the polygon is self-intersecting.

    Arguments:
      polygon: a polygon, closed, but not necessarily simple.
      p: the point to be tested.

    Returns true or false depending on whether the polygon contains the point.
    """
    winding_number = 0

    # each polygon edge is a (source, target) pair of subsequent vertices
    for source, target in pairwise(polygon):
        if source.y <= point.y:
            if target.y > point.y and is_left(point, source, target) > 0:
                winding_number += 1
        elif target.y <= point.y and is_left(point, source, target) < 0:
            winding_number -= 1

    return winding_number != 0
