'''Algorithms for converting 2D coordinates to and from the Hilbert index.

Here the Hilbert curve has been scaled and discretized, so that the
range {0, 1, ..., n^2 - 1} is mapped to coordinates 
{0, 1, ..., n-1} x {0, 1, ..., n-1}. In the classical Hilbert curve,
the continuous interval [0,1] is mapped to the unit square [0,1]^2.
'''
import time
import math

from collections import deque
from typing import Deque
from typing import Iterator
from typing import Tuple

HilbertIndex = int
Coordinates = Tuple[int, int]


def to_coordinates(index: HilbertIndex, n: int) -> Coordinates:
    '''Convert a Hilbert index to a 2d-coordinate.

    The Hilbert curve defines a partition of a square into subsquares indexed
    as
    
      1 | 2
      -----
      0 | 3
    
    each subsquare corresponds to a rotation and/or reflection of the
    perspective, followed by a shift (an affine map). If these transformations
    are denoted H_0, H_1, H_2, H_3 for each subsquare, and if the input index
    is represented in base-4 digits (b_1, b_2, ..., b_k), then the the mapping from
    coordinate to the Hilbert index is given as follows (where * denotes
    function composition)

      (H_{b_1} * H_{b_2} *  ...  * H_{b_k})(0, 0)

    Hence, the algorithm proceeds by applying the transformation for the
    least significant base-4 digit first.

    Args:
      - index: the Hilbert-curve index of the point
      - n: a power of 2 representing the width of the square grid of
           coordinates.

    Returns: 
      The (i, j) coordinate of the corresponding data point.
    '''
    i, j = (0, 0)
    # the side_length indexes both the level of recursion and the length of the
    # side of one subsquare.
    side_length = 1

    while side_length < n:
        subsquare = 3 & index  # least-significant base-4 digit

        if subsquare == 0:
            (i, j) = (j, i)
        elif subsquare == 1:
            (i, j) = (i + side_length, j)
        elif subsquare == 2:
            (i, j) = (i + side_length, j + side_length)
        else:  # subsquare == 3
            (i, j) = (side_length - 1 - j, 2 * side_length - 1 - i)

        index >>= 2  # get the next lowest two bits ready for masking
        side_length *= 2

    return (i, j)


def to_hilbert_index(coords: Coordinates, n: int) -> HilbertIndex:
    '''Convert 2d-coordinates to the Hilbert index.

    Args:
      - coords: the 2D coordinates of a data point.
      - n: a power of 2 representing the width of the square grid of
           coordinates.

    Returns: 
      The Hilbert index of the data point.
    '''
    # the side_length indexes both the level of recursion and the length of the
    # side of one subsquare.
    side_length = n // 2
    index = 0
    (i, j) = coords

    while side_length > 0:
        subsquare_i = int((i & side_length) > 0)
        subsquare_j = int((j & side_length) > 0)

        # The Hilbert curve defines a partition of a square into subsquares
        # indexed as
        #
        #   1 | 2
        #   -----
        #   0 | 3
        #
        # subsquare_i is 1 if the i coordinate is in the upper half,
        # which implies the subsquare is 1 or 2.
        # subsquare_j is 1 if the j coordinate is in the upper half,
        # which implies the subsquare is 0 or 3.
        #
        # So we need a function implementing the table
        #
        # subsquare_i | subsquare_j | subsquare
        # ------------+-------------+-----------
        #           0 |           0 |    00 = 0
        #           0 |           1 |    11 = 3
        #           1 |           0 |    01 = 1
        #           1 |           1 |    10 = 2
        #
        # The second bit of the rightmost column is the xor of the two inputs,
        # and the first bit is subsquare_j.
        subsquare = (subsquare_j << 1) + (subsquare_i ^ subsquare_j)

        if subsquare == 0:
            (i, j) = (j, i)
        elif subsquare == 1:
            (i, j) = (i - side_length, j)
        elif subsquare == 2:
            (i, j) = (i - side_length, j - side_length)
        else:  # subsquare == 3
            (i, j) = (2 * side_length - 1 - j, side_length - 1 - i)

        # each subsquare contains side_length^2 many index points, so recursing
        # to one subsquare causes the index skip over all those points.
        index += subsquare * side_length * side_length
        side_length >>= 1

    return index


def naive_matrix_vector_product(A, v, output, n):
    for i in range(n):
        for j in range(n):
            output[i] += A[i][j] * v[j]


def hilbert_matrix_vector_product(flattened_A, v, output, coordinate_iter):
    for t, (i, j) in coordinate_iter:
        output[i] += flattened_A[t] * v[j]


def hilbert_iter(depth: int) -> Iterator[Coordinates]:
    queue: Deque[Tuple[str, int]] = deque([('H', depth)])
    i, j = 0, 0
    non_terminals = set('HABC')
    yield (i, j)

    while queue:
        (symbol, depth) = queue.popleft()
        if depth == 0 and symbol not in non_terminals:
            if symbol == '↑':
                i += 1
            if symbol == '↓':
                i -= 1
            if symbol == '→':
                j += 1
            if symbol == '←':
                j -= 1
            yield (i, j)
        if depth > 0:
            if symbol == 'H':
                queue.append(('A', depth - 1))
                queue.append(('↑', depth - 1))
                queue.append(('H', depth - 1))
                queue.append(('→', depth - 1))
                queue.append(('H', depth - 1))
                queue.append(('↓', depth - 1))
                queue.append(('B', depth - 1))
            elif symbol == 'A':
                queue.append(('H', depth - 1))
                queue.append(('→', depth - 1))
                queue.append(('A', depth - 1))
                queue.append(('↑', depth - 1))
                queue.append(('A', depth - 1))
                queue.append(('←', depth - 1))
                queue.append(('C', depth - 1))
            elif symbol == 'B':
                queue.append(('C', depth - 1))
                queue.append(('←', depth - 1))
                queue.append(('B', depth - 1))
                queue.append(('↓', depth - 1))
                queue.append(('B', depth - 1))
                queue.append(('→', depth - 1))
                queue.append(('H', depth - 1))
            elif symbol == 'C':
                queue.append(('B', depth - 1))
                queue.append(('↓', depth - 1))
                queue.append(('C', depth - 1))
                queue.append(('←', depth - 1))
                queue.append(('C', depth - 1))
                queue.append(('↑', depth - 1))
                queue.append(('A', depth - 1))
            else:
                # terminal up/down/left/right symbols
                # must be preserved until the end
                queue.append((symbol, depth - 1))


if __name__ == '__main__':
    import random
    import timeit
    random.seed(10)
    n = 2**11
    start = time.time()
    A = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
    v = [random.randint(1, 10) for _ in range(n)]
    output1 = [0] * n
    output2 = [0] * n
    end = time.time()
    print(f'initial data generation: {end-start}s')

    timeit_count = 20
    total_n_seconds = timeit.timeit(
        lambda: naive_matrix_vector_product(A, v, output1, n),
        number=timeit_count
    )

    # reorder data
    start = time.time()

    flattened_A = [0] * (n * n)
    depth = int(math.log2(n))
    coordinate_iter = list(enumerate(hilbert_iter(depth)))
    for (t, (i, j)) in coordinate_iter:
        flattened_A[t] = A[i][j]

    end = time.time()
    print(f'hilbert data preprocessing: {end-start}s')

    # this would show an improvement, but Python lists basically get
    # no benefit from spaital locality. It is shown only as a complete
    # example of how it might work, if ported to another language.
    total_h_seconds = timeit.timeit(
        lambda:
        hilbert_matrix_vector_product(flattened_A, v, output2, coordinate_iter),
        number=timeit_count
    )

    assert output1 == output2

    print(f'Naive: {total_n_seconds}s ({total_n_seconds/timeit_count}s per)')
    print(f'Hilbert: {total_h_seconds}s ({total_h_seconds/timeit_count}s per)')
    print(f'Improvement: {100 * (1 - total_h_seconds / total_n_seconds)}%')
