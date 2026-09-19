#include "uav_features.h"
#include "arm_math.h"
#include <float.h>
#include <math.h>

static arm_rfft_fast_instance_f32 g_rfft;
static float g_fft_in[UAV_FFT_LEN];
static float g_fft_out[UAV_FFT_LEN];

static void axis_time_features(
    const float *s,
    float *mean_out,
    float *std_out,
    float *rms_out,
    float *ptp_out,
    float *crest_out,
    float *skew_out,
    float *kurt_out)
{
    const float n = (float)UAV_WINDOW_SAMPLES;
    float sum = 0.0f;
    float sumsq_raw = 0.0f;
    float minv = FLT_MAX;
    float maxv = -FLT_MAX;
    float maxabs = 0.0f;

    for (int i = 0; i < UAV_WINDOW_SAMPLES; ++i) {
        float v = s[i];
        sum += v;
        sumsq_raw += v * v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        float av = fabsf(v);
        if (av > maxabs) maxabs = av;
    }

    float mean = sum / n;
    float m2sum = 0.0f;
    float m3sum = 0.0f;
    float m4sum = 0.0f;

    for (int i = 0; i < UAV_WINDOW_SAMPLES; ++i) {
        float d = s[i] - mean;
        float d2 = d * d;
        m2sum += d2;
        m3sum += d2 * d;
        m4sum += d2 * d2;
    }

    float std = sqrtf(m2sum / (n - 1.0f));
    float rms = sqrtf(sumsq_raw / n);

    float m2 = m2sum / n;
    float m3 = m3sum / n;
    float m4 = m4sum / n;
    float skew = 0.0f;
    float kurt = 0.0f;

    if (m2 > 1e-20f) {
        float g1 = m3 / powf(m2, 1.5f);
        skew = sqrtf(n * (n - 1.0f)) / (n - 2.0f) * g1;

        /* scipy.stats.kurtosis(..., fisher=True, bias=False) equivalent */
        kurt = (((n * n - 1.0f) * (m4 / (m2 * m2)))
                - (3.0f * (n - 1.0f) * (n - 1.0f)))
               / ((n - 2.0f) * (n - 3.0f));
    }

    *mean_out = mean;
    *std_out = std;
    *rms_out = rms;
    *ptp_out = maxv - minv;
    *crest_out = (rms > 1e-20f) ? (maxabs / rms) : 0.0f;
    *skew_out = skew;
    *kurt_out = kurt;
}

static float spectral_centroid_512(const float *s)
{
    float mean = 0.0f;
    for (int i = 0; i < UAV_WINDOW_SAMPLES; ++i) mean += s[i];
    mean /= (float)UAV_WINDOW_SAMPLES;

    for (int i = 0; i < UAV_WINDOW_SAMPLES; ++i) g_fft_in[i] = s[i] - mean;
    for (int i = UAV_WINDOW_SAMPLES; i < UAV_FFT_LEN; ++i) g_fft_in[i] = 0.0f;

    arm_rfft_fast_f32(&g_rfft, g_fft_in, g_fft_out, 0);

    double power_total = 0.0;
    double weighted = 0.0;

    /* CMSIS packed RFFT: out[0]=DC real, out[1]=Nyquist real,
       for k=1..255: out[2k]=real, out[2k+1]=imag. */
    double p0 = (double)g_fft_out[0] * (double)g_fft_out[0];
    power_total += p0;

    for (int k = 1; k < UAV_FFT_LEN / 2; ++k) {
        double re = (double)g_fft_out[2 * k];
        double im = (double)g_fft_out[2 * k + 1];
        double p = re * re + im * im;
        double f = ((double)k * (double)UAV_SAMPLE_RATE_HZ) / (double)UAV_FFT_LEN;
        power_total += p;
        weighted += f * p;
    }

    double pny = (double)g_fft_out[1] * (double)g_fft_out[1];
    double fny = 0.5 * (double)UAV_SAMPLE_RATE_HZ;
    power_total += pny;
    weighted += fny * pny;

    if (power_total <= 1e-30) return 0.0f;
    return (float)(weighted / power_total);
}

static void fill_axis(const float *s, float *out)
{
    axis_time_features(
        s,
        &out[0], &out[1], &out[2], &out[3],
        &out[4], &out[5], &out[6]);
    out[7] = spectral_centroid_512(s);
}

int uav_features_init(void)
{
    return (arm_rfft_fast_init_512_f32(&g_rfft) == ARM_MATH_SUCCESS) ? 0 : -1;
}

void uav_extract_features(
    const float x[UAV_WINDOW_SAMPLES],
    const float y[UAV_WINDOW_SAMPLES],
    float out[UAV_FEATURE_COUNT])
{
    fill_axis(x, &out[0]);
    fill_axis(y, &out[8]);
}
