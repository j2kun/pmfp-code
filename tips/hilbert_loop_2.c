#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <valgrind/callgrind.h>
#include "hilbert.h"
#include "hilbert_loop_2.h"

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec*1000LL + te.tv_usec/1000;
}

void mulvH(int n, double **A, double *v, double *output, int i, int j) {
  if (n == 1) {
     output[i] += A[n * i + j] * v[j];
  } else {
    mulvA(n / 2, A, v, output, i, j); i++; // up
    mulvH(n / 2, A, v, output, i, j); j++; // right
    mulvH(n / 2, A, v, output, i, j); i--; // down
    mulvB(n / 2, A, v, output, i, j);
  }
}

void mulvA(int n, double **A, double *v, double *output, int i, int j) {
  if (n == 1) {
     output[i] += A[n * i + j] * v[j];
  } else {
    mulvH(n / 2, A, v, output, i, j); j++; // right
    mulvA(n / 2, A, v, output, i, j); i++; // up
    mulvA(n / 2, A, v, output, i, j); j--; // left
    mulvC(n / 2, A, v, output, i, j);
  }
}

void mulvB(int n, double **A, double *v, double *output, int i, int j) {
  if (n == 1) {
     output[i] += A[n * i + j] * v[j];
  } else {
    mulvC(n / 2, A, v, output, i, j); j--; // left
    mulvB(n / 2, A, v, output, i, j); i--; // down
    mulvB(n / 2, A, v, output, i, j); j++; // right
    mulvH(n / 2, A, v, output, i, j);
  }
}

void mulvC(int n, double **A, double *v, double *output, int i, int j) {
  if (n == 1) {
     output[i] += A[n * i + j] * v[j];
  } else {
    mulvB(n / 2, A, v, output, i, j); i--; // down
    mulvC(n / 2, A, v, output, i, j); j--; // left
    mulvC(n / 2, A, v, output, i, j); i++; // up
    mulvA(n / 2, A, v, output, i, j);
  }
}

void flattenMatrix(int n, double **A, double **flattenedA) {
  int d = 0;
  flattenH(n, A, flattenedA, 0, 0, &d);
}

void flattenH(int n, double **A, double **flattenedA, int i, int j, int *d) {
  if (n == 1) {
     printf("%d = (%d, %d)\n", *d, i, j);
     flattenedA[*d] += A[i][j];
     *d += 1;
  } else {
    flattenA(n / 2, A, flattenedA, i, j, d); i++; // up
    flattenH(n / 2, A, flattenedA, i, j, d); j++; // right
    flattenH(n / 2, A, flattenedA, i, j, d); i--; // down
    flattenB(n / 2, A, flattenedA, i, j, d);
  }
}

void flattenA(int n, double **A, double **flattenedA, int i, int j, int *d) {
  if (n == 1) {
     printf("%d = (%d, %d)\n", *d, i, j);
     flattenedA[*d] += A[i][j];
     *d += 1;
  } else {
    flattenH(n / 2, A, flattenedA, i, j, d); j++; // right
    flattenA(n / 2, A, flattenedA, i, j, d); i++; // up
    flattenA(n / 2, A, flattenedA, i, j, d); j--; // left
    flattenC(n / 2, A, flattenedA, i, j, d);
  }
}

void flattenB(int n, double **A, double **flattenedA, int i, int j, int *d) {
  if (n == 1) {
     printf("%d = (%d, %d)\n", *d, i, j);
     flattenedA[*d] += A[i][j];
     *d += 1;
  } else {
    flattenC(n / 2, A, flattenedA, i, j, d); j--; // left
    flattenB(n / 2, A, flattenedA, i, j, d); i--; // down
    flattenB(n / 2, A, flattenedA, i, j, d); j++; // right
    flattenH(n / 2, A, flattenedA, i, j, d);
  }
}

void flattenC(int n, double **A, double **flattenedA, int i, int j, int *d) {
  if (n == 1) {
     printf("%d = (%d, %d)\n", *d, i, j);
     flattenedA[*d] += A[i][j];
     *d += 1;
  } else {
    flattenB(n / 2, A, flattenedA, i, j, d); i--; // down
    flattenC(n / 2, A, flattenedA, i, j, d); j--; // left
    flattenC(n / 2, A, flattenedA, i, j, d); i++; // up
    flattenA(n / 2, A, flattenedA, i, j, d);
  }
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

    // Reorder matrix in Hilbert curve order
    start_t = current_timestamp();
    double *flattenedA = (double *)malloc(rows * rows * sizeof(double));
    flattenMatrix(rows, A, flattenedA);
    end_t = current_timestamp();

    printf("preprocessing time = %lldms\n", end_t - start_t);

    // Wall clock time of Hilbert matrix-vector multiplication (best case)
    start_t = current_timestamp();
    CALLGRIND_START_INSTRUMENTATION;
    mulvH(rows, flattenedA, v, output, 0, 0);
    CALLGRIND_STOP_INSTRUMENTATION;
    end_t = current_timestamp();

    printf("Matrix-vector mul time = %lldms\n", end_t - start_t);

    return 0;
}
