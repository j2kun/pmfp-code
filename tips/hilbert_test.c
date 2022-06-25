#include <stdio.h>
#include <stdlib.h>

#include "minunit.h"
#include "hilbert.h"

int tests_run = 0;

static int hilbert_16_n = 4;
static int hilbert_16_x[] = { 0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0 };
static int hilbert_16_y[] = { 0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3 };

static char * test_to_coordinates_16() {
    char *err_msg = (char*)malloc(256 * sizeof(char)); 
    for (int ndx = 0; ndx < hilbert_16_n*hilbert_16_n; ndx++) {
      int x = 0, y = 0; 
      to_coordinates(ndx, hilbert_16_n, &x, &y); 
      sprintf(err_msg, 
          "Hilbert16 to_coordinates failed at %d, " 
          "expected (%d, %d), got (%d, %d)\n",
          ndx, hilbert_16_x[ndx], hilbert_16_y[ndx], x, y); 
      mu_assert(err_msg, hilbert_16_x[ndx] == x); 
      mu_assert(err_msg, hilbert_16_y[ndx] == y); 
    }
    return 0; 
}

static char * test_to_hilbert_index_16() {
    for (int ndx = 0; ndx < hilbert_16_n*hilbert_16_n; ndx++) {
      int x = hilbert_16_x[ndx];
      int y = hilbert_16_y[ndx];
      int actual = to_hilbert_index(x, y, hilbert_16_n); 
      char *err_msg = (char*)malloc(256 * sizeof(char)); 
      sprintf(err_msg, 
          "Hilbert16 to_hilbert_index failed at (%d, %d), " 
          "expected %d, got %d\n", x, y, ndx, actual); 
      mu_assert(err_msg, ndx == actual);
    }
    return 0; 
}

static char * all_tests() {
    mu_run_test(test_to_coordinates_16);
    mu_run_test(test_to_hilbert_index_16);
    return 0;
}

int main(int argc, char **argv) {
   char *result = all_tests();
   if (result != 0) {
       printf("%s\n", result);
   }
   else {
       printf("ALL TESTS PASSED\n");
   }
   printf("Tests run: %d\n", tests_run);

   return result != 0;
 }
