#ifndef UAV_FEATURES_HOST_H
#define UAV_FEATURES_HOST_H

#define UAV_WINDOW_SAMPLES 500
#define UAV_FFT_LEN 512
#define UAV_FEATURE_COUNT 16

void uav_extract_features_host(
    const float x[UAV_WINDOW_SAMPLES],
    const float y[UAV_WINDOW_SAMPLES],
    float fs_hz,
    float out[UAV_FEATURE_COUNT]
);

#endif
