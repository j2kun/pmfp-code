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
    box_with_hole = geo.box(10, 10, 16, 16, ccw=True).difference(
        geo.box(12, 12, 14, 14, ccw=True)
    )
    grating_segments = [
        ((10, 10), (16, 10)),
        ((10, 11), (16, 11)),
        ((10, 12), (16, 12)),
        ((10, 12), (12, 12)), ((14, 12), (16, 12)),
        ((10, 13), (12, 13)), ((14, 13), (16, 13)),
        ((10, 14), (12, 14)), ((14, 14), (16, 14)),
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

        ((10, 16), (13, 16)),
        ((13, 16), (13, 13)),
        ((13, 13), (10, 13)),
    ]
    actual = find_stitch_path(box_with_hole, grating_segments, starting_point)
    assert expected == actual
