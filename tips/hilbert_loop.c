/*
 * A program that demonstratest the runtime improvement of a matrix-vector
 * multiplication when using the Hilbert curve ordering.
 */
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "hilbert.h"

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec*1000LL + te.tv_usec/1000;
}

int main() {
    long long start_t;
    long long end_t;
    srand(time(NULL));

    int rows = 1 << 14;

    // Allocate a matrix, vector, and output
    int **A = (int **)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        A[i] = (int *)malloc(rows * sizeof(int));
    }
    int *v = (int *)malloc(rows * sizeof(int));
    int *output1 = (int *)malloc(rows * sizeof(int));
    int *output2 = (int *)malloc(rows * sizeof(int));

    // Fill the matrix and vector with random integers in [0, 10]
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < rows; j++) {
        A[i][j] = rand() % 10; 
      }
    }
    for (int j = 0; j < rows; j++) {
      v[j] = rand() % 10; 
    }
    for (int i = 0; i < rows; i++) {
      output1[i] = 0;
      output2[i] = 0;
    }

    // Wall clock time of naive matrix-vector multiplication
    start_t = current_timestamp();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < rows; j++) {
        output1[i] = A[i][j] * v[j];
      }
    }
    end_t = current_timestamp();

    printf("Naive matrix-vector multiplication time = %dms\n", 
        end_t - start_t);

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

    printf("Hilbert preprocessing time = %dms\n", 
        end_t - start_t);

    // Wall clock time of Hilbert matrix-vector multiplication (best case)
    start_t = current_timestamp();
    for (int d = 0; d < rows*rows; d++) {
        output2[x_coord_lookup[d]] += flattened_A[d] * v[y_coord_lookup[d]];
    }
    end_t = current_timestamp();

    printf("Hilbert matrix-vector multiplication time = %dms\n", 
        end_t - start_t);

    return 0;
}
