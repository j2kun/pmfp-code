import random

from shapely import geometry as geo

from autofill import find_stitch_path


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
    # FIXME: finish testing
    box_with_hole = geo.box(10, 10, 20, 20, ccw=True).difference(
        geo.box(14, 14, 16, 16, ccw=True)
    )
