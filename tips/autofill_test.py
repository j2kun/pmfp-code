import math
import random

import hypothesis
from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from matplotlib.patches import FancyArrowPatch
from shapely import geometry as geo
import matplotlib.pyplot as plt
import shapely
import shapely.affinity as affine

from tips.autofill import find_stitch_path


def test_fill_box():
    # The outline of a box with corners (10, 10) and (13, 13)
    shape = geo.box(10, 10, 13, 13, ccw=True)

    grating_segments = [
        ((10, 10), (13, 10)),
        ((10, 11), (13, 11)),
        ((10, 12), (13, 12)),
        ((10, 13), (13, 13)),
    ]
    random.seed(1)
    random.shuffle(grating_segments)

    starting_point = (10, 10)
    expected = [
        ((10, 10), (13, 10)),
        ((13, 10), (13, 11)),
        ((13, 11), (10, 11)),
        ((10, 11), (10, 12)),
        ((10, 12), (13, 12)),
        ((13, 12), (13, 13)),
        ((13, 13), (10, 13)),
    ]
    actual = find_stitch_path(shape, grating_segments, starting_point)

    assert expected == actual


def test_box_with_hole():
    box_with_hole = geo.box(10, 10, 16, 16, ccw=True).difference(
        geo.box(12, 12, 14, 14, ccw=True)
    )
    grating_segments = [
        ((10, 10), (16, 10)),
        ((10, 11), (16, 11)),
        ((10, 12), (12, 12)),
        ((14, 12), (16, 12)),
        ((10, 13), (12, 13)),
        ((14, 13), (16, 13)),
        ((10, 14), (12, 14)),
        ((14, 14), (16, 14)),
        ((10, 15), (16, 15)),
        ((10, 16), (16, 16)),
    ]
    random.seed(1)
    random.shuffle(grating_segments)

    starting_point = (10, 10)

    expected = [
        ((10, 10), (16, 10)),
        ((16, 10), (16, 11)),
        ((16, 11), (10, 11)),
        ((10, 11), (10, 12)),
        ((10, 12), (12, 12)),
        ((12, 12), (12, 13)),
        ((12, 13), (10, 13)),
        ((10, 13), (10, 14)),
        ((10, 14), (12, 14)),
        ((12, 14), (14, 14)),
        ((14, 14), (16, 14)),
        ((16, 14), (16, 15)),
        ((16, 15), (10, 15)),
        ((10, 15), (10, 16)),
        ((10, 16), (16, 16)),
        ((16, 16), (16, 15)),
        ((16, 15), (16, 14)),
        ((16, 14), (16, 13)),
        ((16, 13), (14, 13)),
        ((14, 13), (14, 12)),
        ((14, 12), (16, 12)),
    ]
    actual = find_stitch_path(box_with_hole, grating_segments, starting_point)
    assert expected == actual


def test_box_with_hole_non_aligned_grating():
    box_with_hole = geo.box(10, 10, 16, 16, ccw=True).difference(
        geo.box(12, 12, 14, 14, ccw=True)
    )
    rows = intersect_region_with_grating(
        shape=box_with_hole,
        angle=math.pi / 6,
        row_spacing=1,
    )
    segments = [tuple(point for point in segment) for row in rows for segment in row]
    starting_point = segments[0][0]
    # assert no error
    find_stitch_path(box_with_hole, segments, starting_point)
    # uncomment to matplotlib show the resulting stitch plan
    # output = find_stitch_path(box_with_hole, segments, starting_point)
    # draw_line_segments(output)


@composite
def random_shape_difference(draw, min_points=3, max_points=10):
    """Generate a matrix, and a kernel with strictly smaller dimension."""
    n_points = draw(integers(min_value=min_points, max_value=max_points))
    my_floats = floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False)
    poly1_points = [
        geo.Point(draw(my_floats), draw(my_floats)) for _ in range(n_points)
    ]
    poly2_points = [
        geo.Point(draw(my_floats), draw(my_floats)) for _ in range(n_points)
    ]
    angle = draw(
        floats(
            min_value=0, max_value=2 * math.pi, allow_nan=False, allow_infinity=False
        )
    )
    poly1 = geo.MultiPoint([[p.x, p.y] for p in poly1_points]).convex_hull
    poly2 = geo.MultiPoint([[p.x, p.y] for p in poly2_points]).convex_hull
    return poly1.difference(poly2), angle


@given(random_shape_difference())
def test_fuzz(shape_and_angle):
    shape, angle = shape_and_angle
    hypothesis.assume(shape.area > 1)
    rows = intersect_region_with_grating(
        shape=shape,
        angle=angle,
        row_spacing=1,
    )
    segments = [tuple(point for point in segment) for row in rows for segment in row]
    starting_segment = segments[0]
    starting_point = starting_segment[0]
    # assert no error
    find_stitch_path(shape, segments, starting_point)
    # uncomment to matplotlib show the resulting stitch plan
    # output = find_stitch_path(shape, segments, starting_point)
    # draw_line_segments(output)


def perpendicular_unit_vector(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    perp = [-dy, dx]
    norm = (perp[0] ** 2 + perp[1] ** 2) ** 0.5
    return [perp[0] / norm, perp[1] / norm]


def should_offset(D, segment):
    line1 = geo.LineString(segment)
    tol = 0.1
    count = 0
    for x in D:
        line2 = geo.LineString(x)
        if (
            line1.almost_equals(line2, decimal=tol)
            or line1.almost_equals(line2.reverse(), decimal=tol)
            or line1.overlaps(line2)
        ):
            count += 1
    return count


def draw_line_segments(segments):
    drawn_segments = set()
    for segment in segments:
        start, end = list(segment)
        count = should_offset(drawn_segments, segment)
        if count:
            start = list(start)
            end = list(end)
            shift = 0.15
            direction = perpendicular_unit_vector(start, end)
            end[0] += direction[0] * shift * count
            end[1] += direction[1] * shift * count
            start[0] += direction[0] * shift * count
            start[1] += direction[1] * shift * count
        arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=20)
        plt.gca().add_patch(arrow)
        drawn_segments.add(segment)
    plt.xlim(
        min([s[0] - 1 for s, e in segments]),
        max([e[0] + 1 for s, e in segments]),
    )
    plt.ylim(
        min([s[1] - 1 for s, e in segments]),
        max([e[1] + 1 for s, e in segments]),
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# The code below was copied from Ink/Stitch for the purpose of generating randomized tests.
# - https://github.com/inkstitch/inkstitch/blob/dbded7c9b15a652677c04264fe1c6ee281e114ce/lib/utils/geometry.py#L150
# - https://github.com/inkstitch/inkstitch/blob/dbded7c9b15a652677c04264fe1c6ee281e114ce/lib/stitches/fill.py#L95


class InkstitchPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def mul(self, scalar):
        return self.__class__(self.x * scalar, self.y * scalar)

    def __mul__(self, other):
        if isinstance(other, InkstitchPoint):
            # dot product
            return self.x * other.x + self.y * other.y
        elif isinstance(other, (int, float)):
            return self.__class__(self.x * other, self.y * other)
        else:
            raise ValueError("cannot multiply %s by %s" % (type(self), type(other)))

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise ValueError("cannot multiply %s by %s" % (type(self), type(other)))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        else:
            raise ValueError("cannot divide %s by %s" % (type(self), type(other)))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def length(self):
        return math.sqrt(math.pow(self.x, 2.0) + math.pow(self.y, 2.0))

    def rotate(self, angle):
        return self.__class__(
            self.x * math.cos(angle) - self.y * math.sin(angle),
            self.y * math.cos(angle) + self.x * math.sin(angle),
        )

    def as_tuple(self):
        return (self.x, self.y)

    def __getitem__(self, item):
        return self.as_tuple()[item]

    def __len__(self):
        return 2


def intersect_region_with_grating(shape, angle, row_spacing):
    (minx, miny, maxx, maxy) = shape.bounds
    upper_left = InkstitchPoint(minx, miny)
    lower_right = InkstitchPoint(maxx, maxy)
    length = (upper_left - lower_right).length()
    half_length = length / 2.0
    direction = InkstitchPoint(1, 0).rotate(-angle)
    normal = direction.rotate(math.pi / 2)
    center = InkstitchPoint((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    _, start, _, end = affine.rotate(
        shape, angle, origin="center", use_radians=True
    ).bounds
    start -= center.y
    end -= center.y

    height = abs(end - start)
    if height == 0:
        return []

    start -= (start + normal * center) % row_spacing
    current_row_y = start
    rows = []
    while current_row_y < end:
        p0 = center + normal * current_row_y + direction * half_length
        p1 = center + normal * current_row_y - direction * half_length
        endpoints = [p0.as_tuple(), p1.as_tuple()]
        grating_line = shapely.geometry.LineString(endpoints)

        res = grating_line.intersection(shape)

        if isinstance(res, shapely.geometry.MultiLineString) or isinstance(
            res, geo.GeometryCollection
        ):
            runs = [
                line_string.coords
                for line_string in res.geoms
                if isinstance(line_string, geo.LineString)
            ]
        else:
            if res.is_empty or len(res.coords) == 1:
                # ignore if we intersected at a single point or no points
                runs = []
            else:
                runs = [res.coords]

        if runs:
            runs.sort(key=lambda seg: (InkstitchPoint(*seg[0]) - upper_left).length())
            rows.append(runs)

        current_row_y += row_spacing

    return rows
