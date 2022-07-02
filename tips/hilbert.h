/*
 * Algorithms for converting 2D coordinates to and from the Hilbert index. Here
 * the Hilbert curve has been scaled and discretized, so that the range {0, 1,
 * ..., n^2 - 1} is mapped to coordinates {0, 1, ..., n-1} x {0, 1, ..., n-1}.
 * In the classical Hilbert curve, the continuous interval [0,1] is mapped to
 * the unit square [0,1]^2.
 */

/*
 * Convert 2D coordinates to the Hilbert index.
 *
 *  Args:
 *    - x: the x coordinate to convert.
 *    - y: the y coordinate to convert.
 *    - n: a power of 2 representing the width of the square grid of
 *         coordinates.
 *  Returns: 
 *    The Hilbert index of the data point.
 */
int to_hilbert_index(int x, int y, int n);

/* 
 * Convert a Hilbert index to a 2D coordinate.
 *
 * The Hilbert curve defines a partition of a square into subsquares indexed
 * as
 * 
 *   1 | 2
 *   -----
 *   0 | 3
 * 
 * each subsquare corresponds to a rotation and/or reflection of the
 * perspective, followed by a shift (an affine map). If these transformations
 * are denoted H_0, H_1, H_2, H_3 for each subsquare, and if the input index
 * is represented in base-4 digits (b_1, b_2, ..., b_k), then the the mapping from
 * coordinate to the Hilbert index is given as follows (where * denotes
 * function composition)
 *
 *   (H_{b_1} * H_{b_2} *  ...  * H_{b_k})(0, 0)
 *
 * Hence, the algorithm proceeds by applying the transformation for the
 * least significant base-4 digit first.
 *
 * Args:
 *   - index: the Hilbert-curve index of the point
 *   - n: a power of 2 representing the width of the square grid of
 *        coordinates.
 *   - *x: an out-parameter for the resulting x coordinate.
 *   - *y: an out-parameter for the resulting y coordinate.
 */
void to_coordinates(int index, int n, int *x, int *y);

