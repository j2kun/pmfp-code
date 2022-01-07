from itertools import product
from typing import List

Matrix = List[List[int]]


SOBEL_HORIZONTAL_KERNEL = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
]
SOBEL_VERTICAL_KERNEL = [
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
]


def indices(num_rows: int, num_cols: int):
    return product(range(num_rows), range(num_cols))


def detect_edges(image: Matrix) -> Matrix:
    G_x = convolve(image, SOBEL_HORIZONTAL_KERNEL)
    G_y = convolve(image, SOBEL_VERTICAL_KERNEL)
    return [
        [abs(x) + abs(y) for (x, y) in zip(row_G_x, row_G_y)]
        for (row_G_x, row_G_y) in zip(G_x, G_y)
    ]


def convolve(matrix: Matrix, kernel: Matrix) -> Matrix:
    output_num_rows, output_num_cols = (
        len(matrix) - len(kernel) + 1,
        len(matrix[0]) - len(kernel[0]) + 1,
    )
    kernel_indices = list(indices(len(kernel), len(kernel[0])))

    output = [[0] * output_num_cols for _ in range(output_num_rows)]
    for (row, col) in indices(output_num_rows, output_num_cols):
        value = 0
        for (i, j) in kernel_indices:
            value += kernel[i][j] * matrix[row + i][col + j]
        output[row][col] = value

    return output
