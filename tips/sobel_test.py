from sobel import convolve


def test_convolve_3_by_3():
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
    # TODO: add expected output
    expected = []
    assert convolve(matrix, kernel) == expected
