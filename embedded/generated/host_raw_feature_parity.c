#include <stdio.h>
#include <math.h>
#include "adaboost_model.h"
#include "uav_features_host.h"
#include "raw_test_vectors.h"

static int feature_close(float a, float b) {
    float diff = fabsf(a - b);
    float tol = 1e-3f + 5e-4f * fabsf(b);
    return diff <= tol;
}

int main(void) {
    int prediction_pass = 0;
    int feature_vectors_pass = 0;
    float global_max_abs_error = 0.0f;

    for (int i = 0; i < RAW_TEST_COUNT; ++i) {
        float features[RAW_FEATURE_COUNT];
        uav_extract_features_host(RAW_X[i], RAW_Y[i], RAW_FS_HZ, features);

        int all_features_ok = 1;

        for (int j = 0; j < RAW_FEATURE_COUNT; ++j) {
            float err = fabsf(features[j] - PY_EXPECTED_FEATURES[i][j]);
            if (err > global_max_abs_error) global_max_abs_error = err;
            if (!feature_close(features[j], PY_EXPECTED_FEATURES[i][j])) {
                all_features_ok = 0;
            }
        }

        if (all_features_ok) feature_vectors_pass++;

        int pred = adaboost_predict(features);
        if (pred == RAW_EXPECTED_PREDICTIONS[i]) prediction_pass++;

        printf(
            "raw=%02d feature_check=%s predicted=%d expected=%d %s\n",
            i,
            all_features_ok ? "PASS" : "FAIL",
            pred,
            RAW_EXPECTED_PREDICTIONS[i],
            (pred == RAW_EXPECTED_PREDICTIONS[i]) ? "PASS" : "FAIL"
        );
    }

    printf("\nFEATURE PARITY VECTORS: %d/%d\n", feature_vectors_pass, RAW_TEST_COUNT);
    printf("RAW->FEATURES->MODEL PREDICTION PARITY: %d/%d\n", prediction_pass, RAW_TEST_COUNT);
    printf("MAX ABS FEATURE ERROR: %.9g\n", global_max_abs_error);

    return (prediction_pass == RAW_TEST_COUNT) ? 0 : 1;
}
