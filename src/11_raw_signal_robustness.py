
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import RAW_DIR, TABLES_DIR
from data_utils import (
    CLASS_MAP,
    list_data_files,
    read_table,
    detect_columns,
    estimate_sampling_rate,
    split_contiguous_segments,
)


RNG = np.random.default_rng(20260918)


def spectral_features(signal, fs):
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)

    if len(power) > 1:
        dominant_freq = float(freqs[1 + np.argmax(power[1:])])
    else:
        dominant_freq = 0.0

    total = float(np.sum(power))
    if total <= 0:
        return dominant_freq, 0.0, 0.0

    centroid = float(np.sum(freqs * power) / total)
    p = power / total
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum() / np.log2(len(power)))
    return dominant_freq, centroid, entropy


def all_axis_features(signal, fs):
    signal = np.asarray(signal, dtype=float)
    mean = float(np.mean(signal))
    std = float(np.std(signal, ddof=1))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    ptp = float(np.ptp(signal))
    peak = float(np.max(np.abs(signal)))
    crest = float(peak / rms) if rms > 1e-12 else 0.0
    dom, centroid, sent = spectral_features(signal, fs)

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "ptp": ptp,
        "crest": crest,
        "skew": float(skew(signal, bias=False)),
        "kurtosis": float(kurtosis(signal, fisher=True, bias=False)),
        "dom_freq": dom,
        "spec_centroid": centroid,
        "spec_entropy": sent,
    }


def perturb_axis(signal, condition):
    s = np.asarray(signal, dtype=float).copy()
    sd = float(np.std(s, ddof=1))

    if condition == "clean":
        return s
    if condition == "gaussian_1pct":
        return s + RNG.normal(0.0, 0.01 * sd, size=s.shape)
    if condition == "gaussian_3pct":
        return s + RNG.normal(0.0, 0.03 * sd, size=s.shape)
    if condition == "gaussian_5pct":
        return s + RNG.normal(0.0, 0.05 * sd, size=s.shape)
    if condition == "gain_plus_5pct":
        return 1.05 * s
    if condition == "gain_minus_5pct":
        return 0.95 * s
    if condition == "dc_offset_2pct_std":
        return s + 0.02 * sd

    raise ValueError(condition)


def build_feature_row(window_axes, fs, requested_features, condition):
    axis_cache = {}

    for axis in ("x", "y", "z"):
        if any(f.startswith(axis + "_") for f in requested_features):
            perturbed = perturb_axis(window_axes[axis], condition)
            axis_cache[axis] = all_axis_features(perturbed, fs)

    row = {}
    for feature in requested_features:
        axis, suffix = feature.split("_", 1)
        row[feature] = axis_cache[axis][suffix]
    return row


def collect_domain4_windows():
    rows = []

    for path in list_data_files(RAW_DIR):
        class_name = path.stem
        label = CLASS_MAP[class_name]

        df = read_table(path)
        cols = detect_columns(df)
        _, fs = estimate_sampling_rate(df[cols["time"]])

        starts, ends, _ = split_contiguous_segments(
            df[cols["time"]].to_numpy()
        )
        full = [
            (s, e, idx)
            for idx, (s, e) in enumerate(zip(starts, ends))
            if (e - s) == 500
        ]
        total = len(full)

        for order, (s, e, segment_index) in enumerate(full):
            domain = min(4, int(5 * order / total))
            if domain != 4:
                continue

            rows.append({
                "label": label,
                "class_name": class_name,
                "fs": fs,
                "x": df[cols["x"]].iloc[s:e].to_numpy(float),
                "y": df[cols["y"]].iloc[s:e].to_numpy(float),
                "z": df[cols["z"]].iloc[s:e].to_numpy(float),
            })

    return rows


def evaluate_candidate(name, meta, windows):
    model = joblib.load(meta["model_path"])
    features = meta["features"]

    conditions = [
        "clean",
        "gaussian_1pct",
        "gaussian_3pct",
        "gaussian_5pct",
        "gain_plus_5pct",
        "gain_minus_5pct",
        "dc_offset_2pct_std",
    ]

    results = []

    for condition in conditions:
        X_rows = []
        y = []

        for w in windows:
            X_rows.append(
                build_feature_row(
                    w,
                    w["fs"],
                    features,
                    condition,
                )
            )
            y.append(w["label"])

        X = pd.DataFrame(X_rows, columns=features)
        y = np.asarray(y)

        pred = model.predict(X)
        report = classification_report(
            y, pred, output_dict=True, zero_division=0
        )

        results.append({
            "model": name,
            "condition": condition,
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred)),
            "min_class_recall": float(
                min(report[str(i)]["recall"] for i in range(5))
            ),
        })

    return pd.DataFrame(results)


def main():
    candidates = json.loads(
        (TABLES_DIR / "embedded_feature_candidates.json").read_text(encoding="utf-8")
    )

    print("Loading raw domain-4 windows...")
    windows = collect_domain4_windows()
    print(f"Loaded {len(windows)} held-out raw windows.")

    all_results = []

    for name, meta in candidates.items():
        print("\n" + "=" * 88)
        print(f"RAW-SIGNAL ROBUSTNESS — {name}")
        print("=" * 88)

        result = evaluate_candidate(name, meta, windows)
        print(result.to_string(index=False))
        all_results.append(result)

    out = pd.concat(all_results, ignore_index=True)
    out.to_csv(
        TABLES_DIR / "raw_signal_robustness.csv",
        index=False,
    )

    print("\nIMPORTANT:")
    print("These are synthetic perturbations of the recorded held-out signal.")
    print("They do NOT prove performance on another UAV, another sensor, or outdoor flight.")


if __name__ == "__main__":
    main()
