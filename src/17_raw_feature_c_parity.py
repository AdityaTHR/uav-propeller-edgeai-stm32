from __future__ import annotations

import subprocess
import joblib
import numpy as np
import pandas as pd

from config import RAW_DIR, MODELS_DIR, PROJECT_ROOT
from data_utils import (
    CLASS_MAP,
    list_data_files,
    read_table,
    detect_columns,
    estimate_sampling_rate,
    split_contiguous_segments,
)

WINDOW = 500
FFT_LEN = 512
VECTORS_PER_CLASS = 3


def f32_literal(x):
    return f"{np.float32(x):.9g}f"


def collect_raw_vectors():
    vectors = []

    for path in list_data_files(RAW_DIR):
        class_name = path.stem
        label = CLASS_MAP[class_name]

        df = read_table(path)
        cols = detect_columns(df)
        _, fs = estimate_sampling_rate(df[cols['time']])

        starts, ends, _ = split_contiguous_segments(df[cols['time']].to_numpy())
        full = [(s, e, i) for i, (s, e) in enumerate(zip(starts, ends)) if e - s == WINDOW]
        total = len(full)

        count = 0
        for order, (s, e, segment_idx) in enumerate(full):
            domain = min(4, int(5 * order / total))
            if domain != 4:
                continue

            vectors.append({
                'label': label,
                'class_name': class_name,
                'fs': fs,
                'x': df[cols['x']].iloc[s:e].to_numpy(np.float64),
                'y': df[cols['y']].iloc[s:e].to_numpy(np.float64),
            })
            count += 1
            if count >= VECTORS_PER_CLASS:
                break

    return vectors


def py_spec_centroid(signal, fs):
    centered = signal - np.mean(signal)
    padded = np.zeros(FFT_LEN, dtype=np.float64)
    padded[: len(centered)] = centered
    spectrum = np.fft.rfft(padded, n=FFT_LEN)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(FFT_LEN, d=1.0 / fs)
    total = power.sum()
    return float(np.sum(freqs * power) / total) if total > 1e-20 else 0.0


def py_axis_features(signal, fs):
    from scipy.stats import skew, kurtosis

    signal = np.asarray(signal, dtype=np.float64)
    mean = float(np.mean(signal))
    std = float(np.std(signal, ddof=1))
    rms = float(np.sqrt(np.mean(signal * signal)))
    ptp = float(np.ptp(signal))
    crest = float(np.max(np.abs(signal)) / rms) if rms > 1e-12 else 0.0

    return [
        mean,
        std,
        rms,
        ptp,
        crest,
        float(skew(signal, bias=False)),
        float(kurtosis(signal, fisher=True, bias=False)),
        py_spec_centroid(signal, fs),
    ]


def main():
    model = joblib.load(MODELS_DIR / 'adaboost_xy_fft512_final.joblib')
    vectors = collect_raw_vectors()

    out_dir = PROJECT_ROOT / 'embedded' / 'generated'
    out_dir.mkdir(parents=True, exist_ok=True)

    x_rows = []
    y_rows = []
    expected_feat_rows = []
    expected_pred = []

    for v in vectors:
        x_rows.append('    {' + ', '.join(f32_literal(z) for z in v['x']) + '}')
        y_rows.append('    {' + ', '.join(f32_literal(z) for z in v['y']) + '}')

        feats = py_axis_features(v['x'], v['fs']) + py_axis_features(v['y'], v['fs'])
        expected_feat_rows.append('    {' + ', '.join(f32_literal(z) for z in feats) + '}')
        pred = int(model.predict(pd.DataFrame([feats], columns=model.feature_names_in_))[0])
        expected_pred.append(pred)

    fs = vectors[0]['fs']

    header = f'''#ifndef RAW_TEST_VECTORS_H
#define RAW_TEST_VECTORS_H

#define RAW_TEST_COUNT {len(vectors)}
#define RAW_WINDOW_SAMPLES {WINDOW}
#define RAW_FEATURE_COUNT 16
#define RAW_FS_HZ {f32_literal(fs)}

static const float RAW_X[RAW_TEST_COUNT][RAW_WINDOW_SAMPLES] = {{
{',\n'.join(x_rows)}
}};

static const float RAW_Y[RAW_TEST_COUNT][RAW_WINDOW_SAMPLES] = {{
{',\n'.join(y_rows)}
}};

static const float PY_EXPECTED_FEATURES[RAW_TEST_COUNT][RAW_FEATURE_COUNT] = {{
{',\n'.join(expected_feat_rows)}
}};

static const int RAW_EXPECTED_PREDICTIONS[RAW_TEST_COUNT] = {{
    {', '.join(str(v) for v in expected_pred)}
}};

#endif
'''
    (out_dir / 'raw_test_vectors.h').write_text(header, encoding='utf-8')

    features_h = '''#ifndef UAV_FEATURES_HOST_H
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
'''
    (out_dir / 'uav_features_host.h').write_text(features_h, encoding='utf-8')

    features_c = r'''#include <math.h>
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
'''
    (out_dir / 'uav_features_host.c').write_text(features_c, encoding='utf-8')

    test_c = r'''#include <stdio.h>
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
'''
    (out_dir / 'host_raw_feature_parity.c').write_text(test_c, encoding='utf-8')

    cmd = [
        'gcc', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
        'host_raw_feature_parity.c',
        'uav_features_host.c',
        'adaboost_model.c',
        '-lm',
        '-o', 'host_raw_feature_parity',
    ]

    print('=' * 88)
    print('RAW-SIGNAL -> C FEATURE EXTRACTION -> C MODEL PARITY')
    print('=' * 88)
    print(f'Raw vectors: {len(vectors)}')

    subprocess.run(cmd, cwd=out_dir, check=True)
    result = subprocess.run(
        ['./host_raw_feature_parity'],
        cwd=out_dir,
        check=False,
        text=True,
        capture_output=True,
    )

    print(result.stdout, end='')

    if result.returncode != 0:
        raise SystemExit('FAIL: raw-signal C pipeline does not reproduce Python predictions.')

    print('PASS: raw-signal host-C pipeline reproduced Python predictions for all test vectors.')


if __name__ == '__main__':
    main()
