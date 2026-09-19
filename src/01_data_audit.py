
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RAW_DIR, PLOTS_DIR, TABLES_DIR
from data_utils import list_data_files, read_table, detect_columns, estimate_sampling_rate, split_contiguous_segments

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    files = list_data_files(RAW_DIR)
    if not files:
        print(f"No data files found in {RAW_DIR}")
        return

    results = []
    print("=" * 72)
    print("UAV PROPELLER VIBRATION DATA AUDIT")
    print("=" * 72)

    for path in files:
        df = read_table(path)
        cols = detect_columns(df)
        median_dt, fs = estimate_sampling_rate(df[cols["time"]])
        starts, ends, _ = split_contiguous_segments(df[cols["time"]].to_numpy())
        lengths = ends - starts

        sensor_missing = int(df[[cols["x"], cols["y"], cols["z"]]].isna().sum().sum())

        info = {
            "file": path.name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "column_names": [str(c) for c in df.columns],
            "sensor_missing_total": sensor_missing,
            "duplicate_rows": int(df.duplicated().sum()),
            "column_map": {k: str(v) for k,v in cols.items()},
            "median_dt": median_dt,
            "estimated_sampling_hz": fs,
            "contiguous_segments": int(len(lengths)),
            "segment_length_mode": int(pd.Series(lengths).mode().iloc[0]),
            "full_500_sample_segments": int(np.sum(lengths == 500)),
            "large_gap_count": int(len(lengths)-1),
        }
        results.append(info)

        print(f"\nFILE: {path.name}")
        print(f"Shape: {len(df)} x {df.shape[1]}")
        print(f"Mapped columns: {info['column_map']}")
        print(f"Sensor missing values: {sensor_missing}")
        print(f"Estimated sampling rate: {fs:.3f} Hz")
        print(f"Contiguous acquisition segments: {len(lengths)}")
        print(f"Most common segment length: {info['segment_length_mode']} samples")

        n = min(5000, len(df))
        plt.figure(figsize=(11,6))
        t = df[cols["time"]].iloc[:n]
        for axis in ("x","y","z"):
            plt.plot(t, df[cols[axis]].iloc[:n], label=axis.upper(), linewidth=0.8)
        plt.title(f"Raw vibration preview — {path.stem}")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR/f"{path.stem}_raw_preview.png", dpi=150)
        plt.close()

    (TABLES_DIR/"data_audit_summary_v2.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nSaved outputs/tables/data_audit_summary_v2.json")

if __name__ == "__main__":
    main()
