#include <math.h>
#include <float.h>
#include "uav_features_host.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void axis_time_features(
    const float *s,
    float *mean_out,
    float *std_out,
    float *rms_out,
    float *ptp_out,
    float *crest_out,
    float *skew_out,
    float *kurt_out
) {
    const int n = UAV_WINDOW_SAMPLES;
    float sum = 0.0f;
    float sumsq_raw = 0.0f;
    float minv = FLT_MAX;
    float maxv = -FLT_MAX;
    float maxabs = 0.0f;

    for (int i = 0; i < n; ++i) {
        float v = s[i];
        sum += v;
        sumsq_raw += v * v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        float av = fabsf(v);
        if (av > maxabs) maxabs = av;
    }

    float mean = sum / (float)n;
    float m2sum = 0.0f;
    float m3sum = 0.0f;
    float m4sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        float d = s[i] - mean;
        float d2 = d * d;
        m2sum += d2;
        m3sum += d2 * d;
        m4sum += d2 * d2;
    }

    float std = sqrtf(m2sum / (float)(n - 1));
    float rms = sqrtf(sumsq_raw / (float)n);

    float m2 = m2sum / (float)n;
    float m3 = m3sum / (float)n;
    float m4 = m4sum / (float)n;

    float skew = 0.0f;
    float kurt = 0.0f;

    if (m2 > 1e-20f) {
        float g1 = m3 / powf(m2, 1.5f);
        skew = sqrtf((float)n * (float)(n - 1)) / (float)(n - 2) * g1;

        float nf = (float)n;
        kurt =
            (((nf * nf - 1.0f) * (m4 / (m2 * m2))) -
             (3.0f * (nf - 1.0f) * (nf - 1.0f))) /
            ((nf - 2.0f) * (nf - 3.0f));
    }

    *mean_out = mean;
    *std_out = std;
    *rms_out = rms;
    *ptp_out = maxv - minv;
    *crest_out = (rms > 1e-20f) ? (maxabs / rms) : 0.0f;
    *skew_out = skew;
    *kurt_out = kurt;
}

static float spectral_centroid_dft512(const float *s, float fs_hz) {
    const int n = UAV_WINDOW_SAMPLES;
    const int nfft = UAV_FFT_LEN;

    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += s[i];
    mean /= (float)n;

    double power_total = 0.0;
    double weighted = 0.0;

    for (int k = 0; k <= nfft / 2; ++k) {
        double re = 0.0;
        double im = 0.0;

        for (int i = 0; i < n; ++i) {
            double sample = (double)(s[i] - mean);
            double angle = -2.0 * M_PI * (double)k * (double)i / (double)nfft;
            re += sample * cos(angle);
            im += sample * sin(angle);
        }

        double power = re * re + im * im;
        double freq = (double)k * (double)fs_hz / (double)nfft;
        power_total += power;
        weighted += freq * power;
    }

    if (power_total <= 1e-30) return 0.0f;
    return (float)(weighted / power_total);
}

static void fill_axis(const float *s, float fs_hz, float *out) {
    axis_time_features(
        s,
        &out[0], &out[1], &out[2], &out[3],
        &out[4], &out[5], &out[6]
    );
    out[7] = spectral_centroid_dft512(s, fs_hz);
}

void uav_extract_features_host(
    const float x[UAV_WINDOW_SAMPLES],
    const float y[UAV_WINDOW_SAMPLES],
    float fs_hz,
    float out[UAV_FEATURE_COUNT]
) {
    fill_axis(x, fs_hz, &out[0]);
    fill_axis(y, fs_hz, &out[8]);
}
