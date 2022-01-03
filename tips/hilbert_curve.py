from typing import Tuple
from typing import NewType

HilbertIndex = NewType('HilbertIndex', int)
Coordinates = NewType('Coordinates', Tuple[int, int])


def to_coordinates(index: HilbertIndex, n: int) -> Coordinates:
    '''
    Convert a Hilbert index to a 2d-coordinate.

    Here the Hilbert curve has been scaled and discretized, so that the
    range {0, 1, ..., n^2 - 1} is mapped to coordinates 
    {0, 1, ..., n-1} x {0, 1, ..., n-1}. In the classical Hilbert curve,
    the continuous interval [0,1] is mapped to the unit square [0,1]^2.

    Args:
      - index: the Hilbert-curve index of the point
      - n: a power of 2 representing the width of the square grid of
           coordinates.

    Returns: 
      the (x,y) coordinate of the corresponding data point.
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

    return Coordinates((x, y))


def to_hilbert_index(coords: Coordinates, n: int) -> HilbertIndex:
    '''
    Convert 2d-coordinates to the Hilbert index.
    '''
    rx, ry, s, t = (0, 0, n // 2, 0)
    (x, y) = coords

    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        t += s * s * ((3 * rx) ^ ry)
        (x, y) = rotate(n, x, y, rx, ry)
        s //= 2

    return HilbertIndex(t)


def rotate(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if (ry == 0):
        if (rx == 1):
            x = n - 1 - x
            y = n - 1 - y

        # Swap x and y
        t = x
        x = y
        y = t

    return (x, y)
