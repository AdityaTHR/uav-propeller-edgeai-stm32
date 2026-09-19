#include "uav_config.h"
#include "uav_app.h"
#include "uav_features.h"
#include "adaboost_model.h"
#include <math.h>

#if UAV_SELF_TEST
#include "raw_test_vectors.h"
#endif

int uav_app_init(void)
{
    return uav_features_init();
}

int uav_predict_window(const float *x, const float *y, float *features_out)
{
    uav_extract_features(x, y, features_out);
    return adaboost_predict(features_out);
}

const char *uav_class_name(int class_id)
{
    switch (class_id) {
        case 0: return "Healthy";
        case 1: return "Damaged Bottom Right Blade";
        case 2: return "Damaged Top Right Blade";
        case 3: return "Unbalanced Bottom Right Blade";
        case 4: return "Unbalanced Top Right Blade";
        default: return "Unknown";
    }
}

#if UAV_SELF_TEST
int uav_run_self_test(void)
{
    int passed = 0;
    float features[UAV_FEATURE_COUNT];

    for (int i = 0; i < RAW_TEST_COUNT; ++i) {
        int pred = uav_predict_window(RAW_X[i], RAW_Y[i], features);
        if (pred == RAW_EXPECTED_PREDICTIONS[i]) passed++;
    }

    return passed;
}
#endif
