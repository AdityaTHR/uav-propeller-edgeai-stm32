
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from config import RAW_DIR, PROCESSED_DIR, TABLES_DIR, MODELS_DIR
from data_utils import (
    CLASS_MAP,
    list_data_files,
    read_table,
    detect_columns,
    estimate_sampling_rate,
    split_contiguous_segments,
)

RANDOM_STATE = 42
FFT_LEN = 512
WINDOW_SAMPLES = 500

FEATURES = [
    "x_mean", "x_std", "x_rms", "x_ptp", "x_crest", "x_skew", "x_kurtosis", "x_spec_centroid",
    "y_mean", "y_std", "y_rms", "y_ptp", "y_crest", "y_skew", "y_kurtosis", "y_spec_centroid",
]

PERTURBATIONS = [
    "clean",
    "gaussian_1pct",
    "gaussian_3pct",
    "gaussian_5pct",
    "gain_plus_5pct",
    "gain_minus_5pct",
    "dc_offset_2pct_std",
]


def make_model(seed=RANDOM_STATE):
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=5,
            random_state=seed,
        ),
        n_estimators=50,
        learning_rate=0.5,
        random_state=seed,
    )


def spectral_centroid_fft512(signal, fs):
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - np.mean(signal)
    padded = np.zeros(FFT_LEN, dtype=np.float64)
    padded[:len(centered)] = centered

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
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_ptp": ptp,
        f"{prefix}_crest": crest,
        f"{prefix}_skew": float(skew(signal, bias=False)),
        f"{prefix}_kurtosis": float(kurtosis(signal, fisher=True, bias=False)),
        f"{prefix}_spec_centroid": spectral_centroid_fft512(signal, fs),
    }


def perturb(signal, condition, rng):
    s = np.asarray(signal, dtype=np.float64).copy()
    sd = float(np.std(s, ddof=1))

    if condition == "clean":
        return s
    if condition == "gaussian_1pct":
        return s + rng.normal(0.0, 0.01 * sd, size=s.shape)
    if condition == "gaussian_3pct":
        return s + rng.normal(0.0, 0.03 * sd, size=s.shape)
    if condition == "gaussian_5pct":
        return s + rng.normal(0.0, 0.05 * sd, size=s.shape)
    if condition == "gain_plus_5pct":
        return 1.05 * s
    if condition == "gain_minus_5pct":
        return 0.95 * s
    if condition == "dc_offset_2pct_std":
        return s + 0.02 * sd
    raise ValueError(condition)


def load_windows():
    windows = []

    for path in list_data_files(RAW_DIR):
        class_name = path.stem
        if class_name not in CLASS_MAP:
            raise ValueError(f"Unexpected class filename: {path.name}")

        df = read_table(path)
        cols = detect_columns(df)
        _, fs = estimate_sampling_rate(df[cols["time"]])

        starts, ends, _ = split_contiguous_segments(df[cols["time"]].to_numpy())
        full = [
            (s, e, idx)
            for idx, (s, e) in enumerate(zip(starts, ends))
            if (e - s) == WINDOW_SAMPLES
        ]
        total = len(full)

        for order, (s, e, segment_idx) in enumerate(full):
            domain = min(4, int(5 * order / total))
            windows.append({
                "label": CLASS_MAP[class_name],
                "class_name": class_name,
                "domain": domain,
                "segment_order": order,
                "segment_index": segment_idx,
                "fs": fs,
                "x": df[cols["x"]].iloc[s:e].to_numpy(np.float64),
                "y": df[cols["y"]].iloc[s:e].to_numpy(np.float64),
            })

    return windows


def feature_table_from_windows(windows):
    rows = []
    for w in windows:
        row = {}
        row.update(axis_features(w["x"], w["fs"], "x"))
        row.update(axis_features(w["y"], w["fs"], "y"))
        row.update({
            "label": w["label"],
            "class_name": w["class_name"],
            "domain": w["domain"],
            "segment_order": w["segment_order"],
            "segment_index": w["segment_index"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_cv(dev):
    groups = dev["domain"]
    cv = GroupKFold(n_splits=4)
    f1s = []
    accs = []

    for tr, va in cv.split(dev[FEATURES], dev["label"], groups):
        model = make_model()
        model.fit(dev.iloc[tr][FEATURES], dev.iloc[tr]["label"])
        pred = model.predict(dev.iloc[va][FEATURES])
        f1s.append(f1_score(dev.iloc[va]["label"], pred, average="macro"))
        accs.append(accuracy_score(dev.iloc[va]["label"], pred))

    return {
        "cv_macro_f1_mean": float(np.mean(f1s)),
        "cv_macro_f1_std": float(np.std(f1s, ddof=1)),
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_macro_f1_folds": [float(x) for x in f1s],
    }


def evaluate_holdout(model, holdout):
    pred = model.predict(holdout[FEATURES])
    report = classification_report(
        holdout["label"], pred, output_dict=True, zero_division=0
    )

    return {
        "macro_f1": float(f1_score(holdout["label"], pred, average="macro")),
        "accuracy": float(accuracy_score(holdout["label"], pred)),
        "min_class_recall": float(min(report[str(i)]["recall"] for i in range(5))),
        "per_class_recall": {str(i): float(report[str(i)]["recall"]) for i in range(5)},
        "confusion_matrix": confusion_matrix(holdout["label"], pred).tolist(),
        "classification_report": report,
    }


def evaluate_raw_robustness(model, windows):
    holdout_windows = [w for w in windows if w["domain"] == 4]
    rng = np.random.default_rng(20260918)
    results = []

    for condition in PERTURBATIONS:
        rows = []
        labels = []

        for w in holdout_windows:
            x = perturb(w["x"], condition, rng)
            y = perturb(w["y"], condition, rng)

            row = {}
            row.update(axis_features(x, w["fs"], "x"))
            row.update(axis_features(y, w["fs"], "y"))
            rows.append(row)
            labels.append(w["label"])

        X = pd.DataFrame(rows, columns=FEATURES)
        y_true = np.asarray(labels)
        pred = model.predict(X)

        report = classification_report(
            y_true, pred, output_dict=True, zero_division=0
        )

        results.append({
            "condition": condition,
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "accuracy": float(accuracy_score(y_true, pred)),
            "min_class_recall": float(min(report[str(i)]["recall"] for i in range(5))),
        })

    return pd.DataFrame(results)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("FINAL EMBEDDED-PARITY PREPROCESSING REBUILD")
    print("500 real samples -> mean removal -> zero-pad to 512 -> spectral centroid")
    print("=" * 92)

    windows = load_windows()
    df = feature_table_from_windows(windows)
    df.to_csv(PROCESSED_DIR / "features_adaboost_xy_fft512.csv", index=False)

    dev = df[df["domain"] < 4].copy()
    holdout = df[df["domain"] == 4].copy()

    cv_metrics = evaluate_cv(dev)

    model = make_model()
    model.fit(dev[FEATURES], dev["label"])
    holdout_metrics = evaluate_holdout(model, holdout)

    model_path = MODELS_DIR / "adaboost_xy_fft512_final.joblib"
    joblib.dump(model, model_path)

    robustness = evaluate_raw_robustness(model, windows)
    robustness.to_csv(
        TABLES_DIR / "adaboost_xy_fft512_raw_robustness.csv",
        index=False,
    )

    payload = {
        "model": "AdaBoost_XY_FFT512",
        "window_samples": WINDOW_SAMPLES,
        "fft_len": FFT_LEN,
        "feature_order": FEATURES,
        "axes": "XY",
        "n_estimators": int(len(model.estimators_)),
        "cv": cv_metrics,
        "holdout": holdout_metrics,
        "model_path": str(model_path),
    }

    (TABLES_DIR / "adaboost_xy_fft512_final_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nDevelopment CV:")
    print(json.dumps(cv_metrics, indent=2))
    print("\nTemporal holdout:")
    print(
        f"Macro-F1={holdout_metrics['macro_f1']:.6f} | "
        f"Accuracy={holdout_metrics['accuracy']:.6f} | "
        f"Min class recall={holdout_metrics['min_class_recall']:.6f}"
    )
    print("Per-class recall:", holdout_metrics["per_class_recall"])
    print("\nRaw-signal robustness:")
    print(robustness.to_string(index=False))
    print("\nSaved final embedded-compatible model and metrics.")


if __name__ == "__main__":
    main()
