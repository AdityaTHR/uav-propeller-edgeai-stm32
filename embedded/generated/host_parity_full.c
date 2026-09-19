#include <stdio.h>
#include "adaboost_model.h"
#include "test_vectors_full.h"

int main(void) {
    int passed = 0;
    int first_fail = -1;

    for (int i = 0; i < FULL_TEST_VECTOR_COUNT; ++i) {
        int pred = adaboost_predict(FULL_TEST_VECTORS[i]);
        if (pred == FULL_EXPECTED[i]) {
            passed++;
        } else if (first_fail < 0) {
            first_fail = i;
        }
    }

    printf("FULL HOLDOUT PARITY: %d/%d\n", passed, FULL_TEST_VECTOR_COUNT);

    if (first_fail >= 0) {
        int pred = adaboost_predict(FULL_TEST_VECTORS[first_fail]);
        printf("First failure: index=%d predicted=%d expected=%d\n",
               first_fail, pred, FULL_EXPECTED[first_fail]);
    }

    return (passed == FULL_TEST_VECTOR_COUNT) ? 0 : 1;
}
