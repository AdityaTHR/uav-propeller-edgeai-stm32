#include <stdio.h>
#include "adaboost_model.h"
#include "test_vectors.h"

int main(void) {
    int passed = 0;

    for (int i = 0; i < TEST_VECTOR_COUNT; ++i) {
        int pred = adaboost_predict(TEST_VECTORS[i]);
        int expected = EXPECTED_PREDICTIONS[i];

        printf("vector=%02d predicted=%d expected=%d %s\n",
               i, pred, expected, (pred == expected) ? "PASS" : "FAIL");

        if (pred == expected) {
            passed++;
        }
    }

    printf("\nPARITY: %d/%d\n", passed, TEST_VECTOR_COUNT);
    return (passed == TEST_VECTOR_COUNT) ? 0 : 1;
}
