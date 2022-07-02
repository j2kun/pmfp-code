/*
 * A program that demonstratest the runtime improvement of a matrix-vector
 * multiplication when using the Hilbert curve ordering.
 */
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <valgrind/callgrind.h>
#include "hilbert.h"

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec*1000LL + te.tv_usec/1000;
}

int main(int argc, char *argv[]) {
    long long start_t;
    long long end_t;
    srand(time(NULL));
    int rows;

    if (argc == 2) {
      rows = 1 << atoi(argv[1]);
    } else {
      // Pick a default dimension
      rows = 1 << 13;
    }
    printf("Running with dim %d\n", rows);

    // Allocate a matrix, vector, and output
    double **A = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        A[i] = (double *)malloc(rows * sizeof(double));
    }
    double *v = (double *)malloc(rows * sizeof(double));
    double *output = (double *)malloc(rows * sizeof(double));

    // Fill the matrix and vector with random integers in [0, 10]
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < rows; j++) {
        A[i][j] = rand() % 10; 
      }
    }
    for (int j = 0; j < rows; j++) {
      v[j] = (double)rand() / (double)(RAND_MAX);
    }
    for (int i = 0; i < rows; i++) {
      output[i] = 0;
    }

    // Preprocessing time for best case: store entire Hilbert coordinate lookup
    // table in memory. Then flatten the matrix to a single array in the
    // Hilbert order.
    start_t = current_timestamp();
    int *x_coord_lookup = (int *)malloc(rows * rows * sizeof(int));
    int *y_coord_lookup = (int *)malloc(rows * rows * sizeof(int));
    int *flattened_A = (int *)malloc(rows * rows * sizeof(int));
    for (int d = 0; d < rows*rows; d++) {
        int x = 0, y = 0;
        to_coordinates(d, rows, &x, &y);
        x_coord_lookup[d] = x;
        y_coord_lookup[d] = y;
        flattened_A[d] = A[x][y];
    }
    end_t = current_timestamp();

    printf("preprocessing time = %lldms\n", end_t - start_t);

    // Wall clock time of Hilbert matrix-vector multiplication (best case)
    start_t = current_timestamp();
    CALLGRIND_START_INSTRUMENTATION;
    for (int d = 0; d < rows*rows; d++) {
        output[x_coord_lookup[d]] += flattened_A[d] * v[y_coord_lookup[d]];
    }
    CALLGRIND_STOP_INSTRUMENTATION;
    end_t = current_timestamp();

    printf("Matrix-vector mul time = %lldms\n", end_t - start_t);

    return 0;
}
