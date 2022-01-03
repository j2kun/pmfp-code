from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers

from hilbert_curve import to_coordinates
from hilbert_curve import to_hilbert_index
from hilbert_curve import HilbertIndex
from hilbert_curve import Coordinates


@composite
def dimension_and_hilbert_index(
    draw, log_dim=integers(min_value=0, max_value=20)
):
    log_dim = draw(log_dim)
    dim = 2**log_dim
    index = draw(integers(min_value=0, max_value=dim - 1))
    return (dim, HilbertIndex(index))


@composite
def dimension_and_coordinates(
    draw, log_dim=integers(min_value=0, max_value=10)
):
    log_dim = draw(log_dim)
    dim = 2**log_dim
    x = draw(integers(min_value=0, max_value=dim - 1))
    y = draw(integers(min_value=0, max_value=dim - 1))
    return (dim, Coordinates((x, y)))


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
