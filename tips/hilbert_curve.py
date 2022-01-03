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

    Args:
      - index: the Hilbert-curve index of the point
      - n: a power of 2 representing the width of the square grid of
           coordinates.

    Returns: 
      The (x,y) coordinate of the corresponding data point.
    '''
    s, t = (1, int(index))
    x, y = (0, 0)

    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        (x, y) = rotate(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2

    return (x, y)


def to_hilbert_index(coords: Coordinates, n: int) -> HilbertIndex:
    '''Convert 2d-coordinates to the Hilbert index.

    Args:
      - coords: the 2D coordinates of a data point.
      - n: a power of 2 representing the width of the square grid of
           coordinates.

    Returns: 
      The Hilbert index of the data point.
    '''
    rx, ry, s, t = (0, 0, n // 2, 0)
    (x, y) = coords

    while s > 0:
        rx = int((x & s) > 0)
        ry = int((y & s) > 0)
        t += s * s * ((3 * rx) ^ ry)
        (x, y) = rotate(n, x, y, rx, ry)
        s //= 2

    return t


def rotate(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if (ry == 0):
        if (rx == 1):
            (x, y) = (n - 1 - x, n - 1 - y)
        (x, y) = (y, x)

    return (x, y)


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

    total_h_seconds = timeit.timeit(
        lambda:
        hilbert_matrix_vector_product(flattened_A, v, output2, coordinate_iter),
        number=timeit_count
    )

    assert output1 == output2

    print(f'Naive: {total_n_seconds}s ({total_n_seconds/timeit_count}s per)')
    print(f'Hilbert: {total_h_seconds}s ({total_h_seconds/timeit_count}s per)')
    print(f'Improvement: {100 * (1 - total_h_seconds / total_n_seconds)}%')
