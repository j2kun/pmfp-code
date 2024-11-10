import numpy as np
import pytest

from tips.image_alignment import align_images


@pytest.mark.parametrize(
    "row_shift, col_shift",
    # I manually tested with the cross product of range(50)xrange(40), but to
    # reduce the obscenely large number of tests, now I just test a
    # pseudorandomly chosen subset of those.
    list({((2 * x + 1) % 50, (3 * x + 5) % 40) for x in range(100)}),
)
def test_align_delta_functions(row_shift, col_shift):
    num_rows = 50
    num_cols = 40
    image1 = np.zeros((num_rows, num_cols))
    image2 = np.zeros((num_rows, num_cols))

    row1, col1 = (10, 15)
    row_shift = row_shift if row1 + row_shift < num_rows else row_shift - num_rows
    col_shift = col_shift if col1 + col_shift < num_cols else col_shift - num_cols
    image1[row1, col1] = 255
    row2, col2 = (row1 + row_shift, col1 + col_shift)
    image2[row2, col2] = 255

    actual_row_shift, actual_col_shift = align_images(image1, image2)
    assert (row1, col1) == (
        (row2 + actual_row_shift) % num_rows,
        (col2 + actual_col_shift) % num_cols,
    )
