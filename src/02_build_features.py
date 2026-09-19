
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from config import RAW_DIR, PROCESSED_DIR, TABLES_DIR
from data_utils import CLASS_MAP, list_data_files, read_table, detect_columns, estimate_sampling_rate, split_contiguous_segments

def spectral_features(signal, fs):
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)

    if len(power) > 1:
        dominant_freq = float(freqs[1 + np.argmax(power[1:])])
    else:
        dominant_freq = 0.0

    total = float(np.sum(power))
    if total <= 0:
        return dominant_freq, 0.0, 0.0

    spectral_centroid = float(np.sum(freqs * power) / total)
    p = power / total
    p = p[p > 0]
    spectral_entropy = float(-(p * np.log2(p)).sum() / np.log2(len(power)))
    return dominant_freq, spectral_centroid, spectral_entropy

def axis_features(signal, fs, prefix):
    signal = np.asarray(signal, dtype=float)
    mean = float(np.mean(signal))
    std = float(np.std(signal, ddof=1))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    ptp = float(np.ptp(signal))
    peak = float(np.max(np.abs(signal)))
    crest = float(peak / rms) if rms > 1e-12 else 0.0
    dom, centroid, sent = spectral_features(signal, fs)

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_ptp": ptp,
        f"{prefix}_crest": crest,
        f"{prefix}_skew": float(skew(signal, bias=False)),
        f"{prefix}_kurtosis": float(kurtosis(signal, fisher=True, bias=False)),
        f"{prefix}_dom_freq": dom,
        f"{prefix}_spec_centroid": centroid,
        f"{prefix}_spec_entropy": sent,
    }

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = []

    for path in list_data_files(RAW_DIR):
        class_name = path.stem
        if class_name not in CLASS_MAP:
            raise ValueError(f"Unexpected filename/class: {path.name}")

        df = read_table(path)
        cols = detect_columns(df)
        _, fs = estimate_sampling_rate(df[cols["time"]])
        starts, ends, _ = split_contiguous_segments(df[cols["time"]].to_numpy())

        full = [(s,e,i) for i,(s,e) in enumerate(zip(starts,ends)) if (e-s) == 500]
        total = len(full)

        for order, (s,e,segment_index) in enumerate(full):
            row = {}
            for axis in ("x","y","z"):
                row.update(axis_features(df[cols[axis]].iloc[s:e].to_numpy(float), fs, axis))

            # Five large chronological domains. Domain 4 is the first temporal holdout.
            domain = min(4, int(5 * order / total))

            row.update({
                "label": CLASS_MAP[class_name],
                "class_name": class_name,
                "segment_order": order,
                "segment_index": segment_index,
                "domain": domain,
            })
            rows.append(row)

        summary.append({
            "class_name": class_name,
            "label": CLASS_MAP[class_name],
            "sampling_hz": fs,
            "full_500_sample_segments": total,
        })

    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED_DIR/"features_500sample.csv", index=False)
    pd.DataFrame(summary).to_csv(TABLES_DIR/"feature_build_summary.csv", index=False)

    print(out.groupby(["class_name","domain"]).size())
    print(f"\nFeature table: {out.shape[0]} windows x {out.shape[1]} columns")
    print("Saved data/processed/features_500sample.csv")

if __name__ == "__main__":
    main()
