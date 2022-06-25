#include "hilbert.h"

int to_hilbert_index(int x, int y, int n) {
    // The side_length indexes both the level of recursion and the length of
    // the side of one subsquare.
    int side_length = n / 2;
    int index = 0;
    int i = x;
    int j = y;
    int subsquare_i, subsquare_j, subsquare, tmp;

    while (side_length > 0) {
        subsquare_i = (int) ((i & side_length) > 0);
        subsquare_j = (int) ((j & side_length) > 0);

        /* The Hilbert curve defines a partition of a square into subsquares
         * indexed as
         *
         *   1 | 2
         *   -----
         *   0 | 3
         *
         * subsquare_i is 1 if the i coordinate is in the upper half,
         * which implies the subsquare is 1 or 2.
         * subsquare_j is 1 if the j coordinate is in the upper half,
         * which implies the subsquare is 0 or 3.
         *
         * So we need a function implementing the table
         *
         * subsquare_i | subsquare_j | subsquare
         * ------------+-------------+-----------
         *           0 |           0 |    00 = 0
         *           0 |           1 |    11 = 3
         *           1 |           0 |    01 = 1
         *           1 |           1 |    10 = 2
         *
         * The second bit of the rightmost column is the xor of the two inputs,
         * and the first bit is subsquare_j.
         */
        subsquare = (subsquare_j << 1) + (subsquare_i ^ subsquare_j);

        if (subsquare == 0) {
            tmp = i;
            i = j;
            j = tmp;
        } else if (subsquare == 1) {
            i -= side_length;
        } else if (subsquare == 2) {
            i -= side_length;
            j -= side_length;
        } else {  // subsquare == 3
            tmp = 2 * side_length - 1 - j;
            j =  side_length - 1 - i;
            i = tmp;
        }

        // Each subsquare contains side_length^2 many index points, so
        // recursing to one subsquare causes the index skip over all those
        // points.
        index += subsquare * side_length * side_length;
        side_length >>= 1;
    }

    return index;
}

void to_coordinates(int index, int n, int *x, int *y) {
    int i = 0;
    int j = 0;
    int tmp, subsquare;

    // the side_length indexes both the level of recursion and the length of the
    // side of one subsquare.
    int side_length = 1;

    while (side_length < n) {
        subsquare = 3 & index;  // least-significant base-4 digit

        if (subsquare == 0) {
            tmp = i;
            i = j;
            j = tmp;
        } else if (subsquare == 1) {
            i += side_length;
        } else if (subsquare == 2) {
            i += side_length;
            j += side_length;
        } else {  // subsquare == 3
            tmp = side_length - 1 - j;
            j = 2 * side_length - 1 - i;
            i = tmp;
        }

        index >>= 2;  // get the next lowest two bits ready for masking
        side_length *= 2;
    }

    *x = i;
    *y = j;
}
