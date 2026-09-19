from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from config import RAW_DIR, PROCESSED_DIR, TABLES_DIR, MODELS_DIR
from data_utils import (
    CLASS_MAP,
    list_data_files,
    read_table,
    detect_columns,
    estimate_sampling_rate,
    split_contiguous_segments,
)

FFT_LEN = 512
WINDOW_SAMPLES = 500
RANDOM_STATE = 42

FEATURES = [
    'y_mean', 'y_std', 'y_rms', 'y_ptp', 'y_crest', 'y_skew', 'y_kurtosis', 'y_spec_centroid',
    'z_mean', 'z_std', 'z_rms', 'z_ptp', 'z_crest', 'z_skew', 'z_kurtosis', 'z_spec_centroid',
]

PERTURBATIONS = [
    'clean',
    'gaussian_1pct',
    'gaussian_3pct',
    'gaussian_5pct',
    'gain_plus_5pct',
    'gain_minus_5pct',
    'dc_offset_2pct_std',
]


def make_xgb(seed=RANDOM_STATE):
    return XGBClassifier(
        n_estimators=75,
        max_depth=2,
        learning_rate=0.10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        random_state=seed,
        n_jobs=4,
        reg_lambda=1.0,
    )


def spectral_centroid_fft512(signal, fs):
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - np.mean(signal)
    padded = np.zeros(FFT_LEN, dtype=np.float64)
    padded[: len(centered)] = centered
    spectrum = np.fft.rfft(padded, n=FFT_LEN)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(FFT_LEN, d=1.0 / fs)
    total = float(power.sum())
    if total <= 1e-20:
        return 0.0
    return float(np.sum(freqs * power) / total)


def axis_features(signal, fs, prefix):
    signal = np.asarray(signal, dtype=np.float64)
    mean = float(np.mean(signal))
    std = float(np.std(signal, ddof=1))
    rms = float(np.sqrt(np.mean(signal * signal)))
    ptp = float(np.ptp(signal))
    peak = float(np.max(np.abs(signal)))
    crest = float(peak / rms) if rms > 1e-12 else 0.0

    return {
        f'{prefix}_mean': mean,
        f'{prefix}_std': std,
        f'{prefix}_rms': rms,
        f'{prefix}_ptp': ptp,
        f'{prefix}_crest': crest,
        f'{prefix}_skew': float(skew(signal, bias=False)),
        f'{prefix}_kurtosis': float(kurtosis(signal, fisher=True, bias=False)),
        f'{prefix}_spec_centroid': spectral_centroid_fft512(signal, fs),
    }


def perturb(signal, condition, rng):
    s = np.asarray(signal, dtype=np.float64).copy()
    sd = float(np.std(s, ddof=1))
    if condition == 'clean':
        return s
    if condition == 'gaussian_1pct':
        return s + rng.normal(0.0, 0.01 * sd, size=s.shape)
    if condition == 'gaussian_3pct':
        return s + rng.normal(0.0, 0.03 * sd, size=s.shape)
    if condition == 'gaussian_5pct':
        return s + rng.normal(0.0, 0.05 * sd, size=s.shape)
    if condition == 'gain_plus_5pct':
        return 1.05 * s
    if condition == 'gain_minus_5pct':
        return 0.95 * s
    if condition == 'dc_offset_2pct_std':
        return s + 0.02 * sd
    raise ValueError(condition)


def load_windows():
    windows = []
    for path in list_data_files(RAW_DIR):
        class_name = path.stem
        df = read_table(path)
        cols = detect_columns(df)
        _, fs = estimate_sampling_rate(df[cols['time']])

        starts, ends, _ = split_contiguous_segments(df[cols['time']].to_numpy())
        full = [(s, e, i) for i, (s, e) in enumerate(zip(starts, ends)) if e - s == WINDOW_SAMPLES]
        total = len(full)

        for order, (s, e, segment_idx) in enumerate(full):
            domain = min(4, int(5 * order / total))
            windows.append({
                'label': CLASS_MAP[class_name],
                'class_name': class_name,
                'domain': domain,
                'segment_order': order,
                'segment_index': segment_idx,
                'fs': fs,
                'y': df[cols['y']].iloc[s:e].to_numpy(np.float64),
                'z': df[cols['z']].iloc[s:e].to_numpy(np.float64),
            })
    return windows


def build_table(windows):
    rows = []
    for w in windows:
        row = {}
        row.update(axis_features(w['y'], w['fs'], 'y'))
        row.update(axis_features(w['z'], w['fs'], 'z'))
        row.update({
            'label': w['label'],
            'class_name': w['class_name'],
            'domain': w['domain'],
            'segment_order': w['segment_order'],
            'segment_index': w['segment_index'],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def cv_metrics(dev):
    groups = dev['domain']
    cv = GroupKFold(n_splits=4)
    scores, accs = [], []

    for tr, va in cv.split(dev[FEATURES], dev['label'], groups):
        model = make_xgb()
        model.fit(dev.iloc[tr][FEATURES], dev.iloc[tr]['label'])
        pred = model.predict(dev.iloc[va][FEATURES])
        scores.append(f1_score(dev.iloc[va]['label'], pred, average='macro'))
        accs.append(accuracy_score(dev.iloc[va]['label'], pred))

    return {
        'cv_macro_f1_mean': float(np.mean(scores)),
        'cv_macro_f1_std': float(np.std(scores, ddof=1)),
        'cv_accuracy_mean': float(np.mean(accs)),
        'cv_macro_f1_folds': [float(v) for v in scores],
    }


def holdout_metrics(model, holdout):
    pred = model.predict(holdout[FEATURES])
    report = classification_report(holdout['label'], pred, output_dict=True, zero_division=0)
    return {
        'macro_f1': float(f1_score(holdout['label'], pred, average='macro')),
        'accuracy': float(accuracy_score(holdout['label'], pred)),
        'min_class_recall': float(min(report[str(i)]['recall'] for i in range(5))),
        'per_class_recall': {str(i): float(report[str(i)]['recall']) for i in range(5)},
        'confusion_matrix': confusion_matrix(holdout['label'], pred).tolist(),
    }


def raw_robustness(model, windows):
    holdout_windows = [w for w in windows if w['domain'] == 4]
    rng = np.random.default_rng(20260918)
    rows = []

    for condition in PERTURBATIONS:
        feats, labels = [], []
        for w in holdout_windows:
            y = perturb(w['y'], condition, rng)
            z = perturb(w['z'], condition, rng)
            row = {}
            row.update(axis_features(y, w['fs'], 'y'))
            row.update(axis_features(z, w['fs'], 'z'))
            feats.append(row)
            labels.append(w['label'])

        X = pd.DataFrame(feats, columns=FEATURES)
        y_true = np.asarray(labels)
        pred = model.predict(X)
        report = classification_report(y_true, pred, output_dict=True, zero_division=0)
        rows.append({
            'condition': condition,
            'macro_f1': float(f1_score(y_true, pred, average='macro')),
            'accuracy': float(accuracy_score(y_true, pred)),
            'min_class_recall': float(min(report[str(i)]['recall'] for i in range(5))),
        })

    return pd.DataFrame(rows)


def main():
    print('=' * 96)
    print('APPLES-TO-APPLES FFT512 CHECK — COMPACT XGBOOST')
    print('=' * 96)

    windows = load_windows()
    df = build_table(windows)
    df.to_csv(PROCESSED_DIR / 'features_xgboost_yz_fft512.csv', index=False)

    dev = df[df['domain'] < 4].copy()
    holdout = df[df['domain'] == 4].copy()

    cv = cv_metrics(dev)
    model = make_xgb()
    model.fit(dev[FEATURES], dev['label'])
    holdout_result = holdout_metrics(model, holdout)
    robustness = raw_robustness(model, windows)

    model_path = MODELS_DIR / 'xgboost_yz_fft512_reference.joblib'
    joblib.dump(model, model_path)
    robustness.to_csv(TABLES_DIR / 'xgboost_yz_fft512_raw_robustness.csv', index=False)

    ada = json.loads(
        (TABLES_DIR / 'adaboost_xy_fft512_final_metrics.json').read_text(encoding='utf-8')
    )

    comparison = pd.DataFrame([
        {
            'model': 'AdaBoost_XY_FFT512',
            'cv_macro_f1': ada['cv']['cv_macro_f1_mean'],
            'cv_macro_f1_std': ada['cv']['cv_macro_f1_std'],
            'holdout_macro_f1': ada['holdout']['macro_f1'],
            'holdout_accuracy': ada['holdout']['accuracy'],
            'holdout_min_class_recall': ada['holdout']['min_class_recall'],
            'trees': ada['n_estimators'],
            'feature_count': len(ada['feature_order']),
        },
        {
            'model': 'XGBoost_YZ_FFT512_reference',
            'cv_macro_f1': cv['cv_macro_f1_mean'],
            'cv_macro_f1_std': cv['cv_macro_f1_std'],
            'holdout_macro_f1': holdout_result['macro_f1'],
            'holdout_accuracy': holdout_result['accuracy'],
            'holdout_min_class_recall': holdout_result['min_class_recall'],
            'trees': 375,
            'feature_count': len(FEATURES),
        },
    ])
    comparison.to_csv(TABLES_DIR / 'fft512_apples_to_apples_comparison.csv', index=False)

    payload = {
        'cv': cv,
        'holdout': holdout_result,
        'features': FEATURES,
        'model_path': str(model_path),
    }
    (TABLES_DIR / 'xgboost_yz_fft512_reference_metrics.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8'
    )

    print('\nComparison:')
    print(comparison.to_string(index=False))
    print('\nXGBoost raw-signal robustness:')
    print(robustness.to_string(index=False))
    print('\nThis is a fairness check, not a new hyperparameter search.')


if __name__ == '__main__':
    main()
