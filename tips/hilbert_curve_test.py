from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers

from hilbert_curve import to_coordinates
from hilbert_curve import to_hilbert_index
from hilbert_curve import hilbert_iter


@composite
def dimension_and_hilbert_index(
    draw, log_dim=integers(min_value=0, max_value=20)
):
    log_dim = draw(log_dim)
    dim = 2**log_dim
    index = draw(integers(min_value=0, max_value=dim - 1))
    return (dim, index)


@composite
def dimension_and_coordinates(
    draw, log_dim=integers(min_value=0, max_value=10)
):
    log_dim = draw(log_dim)
    dim = 2**log_dim
    x = draw(integers(min_value=0, max_value=dim - 1))
    y = draw(integers(min_value=0, max_value=dim - 1))
    return (dim, (x, y))


@given(dimension_and_hilbert_index())
def test_right_inverse_composes_to_identity(dim_and_index):
    (dim, index) = dim_and_index

    coords = to_coordinates(index, dim)
    actual = to_hilbert_index(coords, dim)
    assert actual == index


@given(dimension_and_coordinates())
def test_left_inverse_composes_to_identity(dim_and_coords):
    (dim, coords) = dim_and_coords

    index = to_hilbert_index(coords, dim)
    actual = to_coordinates(index, dim)
    assert actual == coords


def test_hilbert_iter_16():
    depth = 2  # depth d -> 4^d cells, in this case it's 16 = 4^2
    expected = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (2, 1),
        (2, 2),
        (3, 2),
        (3, 3),
        (2, 3),
        (1, 3),
        (1, 2),
        (0, 2),
        (0, 3),
    ]
    actual = list(hilbert_iter(depth))
    assert len(actual) == len(expected)
    assert expected == actual


# TODO: fix this by reimplementing the to_coordinates and from_coordinates
# to match the iteration order of hilbert_iter
'''
@given(integers(min_value=0, max_value=6))
def test_hilbert_iter_uses_ascending_index_order(log_dim):
    dim = 2**log_dim
    coords_iter = hilbert_iter(log_dim)
    hilbert_index_iter = [to_hilbert_index(x, dim) for x in coords_iter]
    assert hilbert_index_iter == list(range(dim))
'''
