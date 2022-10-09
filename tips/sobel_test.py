from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import lists
import hypothesis
import numpy as np
import pytest

from sobel import convolve
from sobel import detect_edges
from sobel import np_convolve2d
from sobel import np_detect_edges
from sobel import sobel_optimized


def test_convolve_4by4_with_2by2():
    matrix = [
        [1, 2, 3, 4],
        [2, -3, 4, -1],
        [3, 3, 0, 2],
        [1, 9, -2, 2],
    ]
    kernel = [
        [1, 2],
        [-1, -2],
    ]
    expected = [
        [9, 3, 9],
        [-13, 2, -2],
        [-10, -2, 2],
    ]
    assert convolve(matrix, kernel) == expected


def test_np_convolve_4by4_with_2by2():
    matrix = np.array([
        [1, 2, 3, 4],
        [2, -3, 4, -1],
        [3, 3, 0, 2],
        [1, 9, -2, 2],
    ])
    kernel = np.array([
        [1, 2],
        [-1, -2],
    ])
    expected = np.array([
        [9, 3, 9],
        [-13, 2, -2],
        [-10, -2, 2],
    ])
    actual = np_convolve2d(matrix, kernel)
    np.testing.assert_array_equal(actual, expected)


@composite
def random_matrix_and_kernel(draw, min_dim=1, max_dim=10):
    """Generate a matrix, and a kernel with strictly smaller dimension."""
    matrix_dim = integers(
        min_value=min_dim,
        max_value=max_dim,
    )
    values = integers(min_value=-100, max_value=100)
    matrix_row_count = draw(matrix_dim)
    matrix_col_count = draw(matrix_dim)
    matrix = draw(
        lists(
            lists(values,
                  min_size=matrix_col_count,
                  max_size=matrix_col_count),
            min_size=matrix_row_count,
            max_size=matrix_row_count
        ))

    kernel_row_count = draw(integers(
        min_value=1,
        max_value=matrix_row_count,
    ))
    kernel_col_count = draw(integers(
        min_value=1,
        max_value=matrix_col_count,
    ))
    kernel = draw(
        lists(
            lists(values,
                  min_size=kernel_col_count,
                  max_size=kernel_col_count),
            min_size=kernel_row_count,
            max_size=kernel_row_count
        ))

    return (matrix, kernel)


@given(random_matrix_and_kernel(min_dim=1, max_dim=10))
def test_dimension_fuzz_test(matrix_and_kernel):
    # This test merely runs the convolve on matrices and kernels of varying
    # dimensions, to check whether the dimension calculations are off.
    matrix, kernel = matrix_and_kernel
    result = convolve(matrix, kernel)
    assert len(result) == len(matrix) - len(kernel) + 1
    assert len(result[0]) == len(matrix[0]) - len(kernel[0]) + 1


@given(random_matrix_and_kernel(min_dim=1, max_dim=10))
def test_np_dimension_fuzz_test(matrix_and_kernel):
    # This test merely runs the convolve on matrices and kernels of varying
    # dimensions, to check whether the dimension calculations are off.
    matrix, kernel = matrix_and_kernel
    matrix, kernel = np.array(matrix), np.array(kernel)
    result = np_convolve2d(matrix, kernel)
    assert len(result) == len(matrix) - len(kernel) + 1
    assert len(result[0]) == len(matrix[0]) - len(kernel[0]) + 1


@pytest.mark.parametrize("detect_edges_fn", [detect_edges, sobel_optimized])
def test_detect_edges_simple_vert(detect_edges_fn):
    matrix = [
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]
    expected = [
        [0, 4],
        [0, 4],
    ]
    assert detect_edges_fn(matrix) == expected


@pytest.mark.parametrize("detect_edges_fn", [detect_edges, sobel_optimized])
def test_detect_edges_simple_horizontal(detect_edges_fn):
    matrix = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ]
    expected = [
        [4, 4],
        [0, 0],
    ]
    assert detect_edges_fn(matrix) == expected


@composite
def random_matrix(draw, min_dim=1, max_dim=10):
    """Generate a matrix, and a kernel with strictly smaller dimension."""
    matrix_dim = integers(
        min_value=min_dim,
        max_value=max_dim,
    )
    values = integers(min_value=-100, max_value=100)
    matrix_row_count = draw(matrix_dim)
    matrix_col_count = draw(matrix_dim)
    matrix = draw(
        lists(
            lists(values,
                  min_size=matrix_col_count,
                  max_size=matrix_col_count),
            min_size=matrix_row_count,
            max_size=matrix_row_count
        ))
    return matrix


@hypothesis.settings(deadline=5000)
@given(random_matrix(min_dim=4, max_dim=20))
def test_consistency_between_impls(matrix):
    result1 = detect_edges(matrix)
    result2 = sobel_optimized(matrix)
    result3 = np_detect_edges(np.array(matrix))
    assert result1 == result2
    np.testing.assert_array_equal(np.array(result1), result3)
