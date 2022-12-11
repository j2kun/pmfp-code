"""Edge detection using the Sobel kernel.

This module contains three implementations:

    - A pure-python, generic convolution operator. It is quite slow.
    - A pure-python, loop-unrolled implementation of Sobel convolution.
    - A faster numpy-based implementation using a generic numpy-based
      convolution.

Running this file via `python tips/sobel.py path/to/file.jpg` applies the Sobel
kernel to an image, and saves the resulting image to disk.
"""
from itertools import product
from typing import List

import numpy as np

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
    """Return a generator over a 2D index space in row-major order.

    A pure-python version of the (much faster) numpy.ndindex.
    """
    return product(range(num_rows), range(num_cols))


def detect_edges(image: Matrix) -> Matrix:
    """Detect the edges in an image.

    Arguments:
      - image: a 2D array containing one color channel of the image to process

    Returns:
      A 2D array. Entries (i, j) with large magnitude detect the presence of an edge
      at image position (i+1, j+1).
    """
    G_x = convolve(image, SOBEL_HORIZONTAL_KERNEL)
    G_y = convolve(image, SOBEL_VERTICAL_KERNEL)

    # sum of absolute values is an approximation of the norm, sqrt(x^2 + y^2)
    return [
        [abs(x) + abs(y) for (x, y) in zip(row_G_x, row_G_y)]
        for (row_G_x, row_G_y) in zip(G_x, G_y)
    ]


def convolve(matrix: Matrix, kernel: Matrix) -> Matrix:
    """Convolve a matrix with a kernel.

    Arguments:
      - matrx: a 2D array to be convolved
      - kernel: a smaller 2D array to use as the convolution kernel.

    Returns:
      A 2D array. Entries (i, j) contain the sum
          sum_{i_2, j_2} (kernel[i_2][j_j] * matrix[i+i_2][j+j_2])
    """
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


def sobel_optimized(matrix: Matrix) -> Matrix:
    # Use numpy or a compiled language if you want it to be truly fast. This
    # simply unrolls the generic convolution loop for the Sobel operators.
    output_num_rows, output_num_cols = (len(matrix) - 2, len(matrix[0]) - 2)

    output = [[0] * output_num_cols for _ in range(output_num_rows)]
    for (row, col) in indices(output_num_rows, output_num_cols):
        m00 = matrix[row][col]
        m01 = matrix[row][col + 1]
        m02 = matrix[row][col + 2]
        m10 = matrix[row + 1][col]
        m12 = matrix[row + 1][col + 2]
        m20 = matrix[row + 2][col]
        m21 = matrix[row + 2][col + 1]
        m22 = matrix[row + 2][col + 2]
        x = m00 - m02 + 2 * (m10 - m12) + m20 - m22
        y = m00 - m20 + 2 * (m01 - m21) + m02 - m22
        output[row][col] = abs(x) + abs(y)

    return output


def np_convolve2d(matrix: np.ndarray, kernel: np.ndarray):
    """An optimized 2d convolution routine using numpy.

    Kudos to the many answers on StackOverflow describing the use of
    as_strided and einsum.
    """
    sub_shape = kernel.shape
    view_shape = tuple(np.subtract(matrix.shape, sub_shape) + 1) + sub_shape
    strides = 2 * matrix.strides
    sub_matrix = np.lib.stride_tricks.as_strided(
        matrix, shape=view_shape, strides=strides
    )
    return np.einsum("kl,ijkl->ij", kernel, sub_matrix)


def np_detect_edges(image: np.ndarray) -> np.ndarray:
    """A numpy version of detect_edges."""
    np_horizontal = np.array(SOBEL_HORIZONTAL_KERNEL)
    np_vertical = np.array(SOBEL_VERTICAL_KERNEL)
    G_x = np_convolve2d(image, np_horizontal)
    G_y = np_convolve2d(image, np_vertical)
    return np.abs(G_x) + np.abs(G_y)


if __name__ == "__main__":
    import argparse
    import os

    from PIL import Image

    parser = argparse.ArgumentParser(description="Apply the Sobel filter to an image.")
    parser.add_argument("source_file")
    args = parser.parse_args()

    with Image.open(args.source_file) as im:
        # Convert to grayscale, one value per pixel. Could alternatively
        # compute Sobel on each color channel and recombine.
        pixel_array = np.array(im.convert("L"))
        print(pixel_array.shape)

    print("detecting edges")
    sobel_pixel_array = np_detect_edges(pixel_array)
    sobel_pixel_array = np.floor(255 * sobel_pixel_array / np.max(sobel_pixel_array))

    print("converting back to image")
    edge_image = Image.fromarray(sobel_pixel_array.astype(np.uint8), mode="L")
    # edge_image.show()
    source_dir, source_name = os.path.split(args.source_file)
    outfile = os.path.join(source_dir, f"edges_{source_name}")
    edge_image.save(outfile)
