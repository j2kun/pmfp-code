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

    // Wall clock time of naive matrix-vector multiplication
    start_t = current_timestamp();
    CALLGRIND_START_INSTRUMENTATION;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < rows; j++) {
        output[i] = A[i][j] * v[j];
      }
    }
    CALLGRIND_STOP_INSTRUMENTATION;
    end_t = current_timestamp();

    printf("Naive matrix-vector multiplication time = %lldms\n", 
        end_t - start_t);

    return 0;
}
