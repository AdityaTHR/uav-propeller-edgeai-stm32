#ifndef UAV_FEATURES_H
#define UAV_FEATURES_H

#include "adaboost_model.h"
#include "uav_config.h"

int uav_features_init(void);
void uav_extract_features(
    const float x[UAV_WINDOW_SAMPLES],
    const float y[UAV_WINDOW_SAMPLES],
    float out[UAV_FEATURE_COUNT]
);

#endif
