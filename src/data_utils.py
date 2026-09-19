
from pathlib import Path
import re
import numpy as np
import pandas as pd

SUPPORTED = {".xlsx", ".xls", ".csv"}

CLASS_MAP = {
    "Healthy": 0,
    "Damaged Bottom Right Blade": 1,
    "Damaged Top Right Blade": 2,
    "Unbalanced Bottom Right Blade": 3,
    "Unbalanced Top Right Blade": 4,
}

def list_data_files(raw_dir: Path):
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

def detect_columns(df: pd.DataFrame):
    out = {"time": None, "x": None, "y": None, "z": None}
    normalized = {c: normalize(c) for c in df.columns}

    for c, n in normalized.items():
        if n in {"time", "times", "timestamp", "timesec", "times"} or n.startswith("time"):
            out["time"] = c
            break

    axis_aliases = {
        "x": {"x", "ax", "accx", "xaxis", "xacceleration", "accelerationx", "accinxaxis"},
        "y": {"y", "ay", "accy", "yaxis", "yacceleration", "accelerationy", "accinyaxis"},
        "z": {"z", "az", "accz", "zaxis", "zacceleration", "accelerationz", "accinzaxis"},
    }

    for axis, aliases in axis_aliases.items():
        for c, n in normalized.items():
            if n in aliases or (("acc" in n or "acceleration" in n) and axis in n):
                out[axis] = c
                break

    missing = [k for k,v in out.items() if v is None]
    if missing:
        raise ValueError(f"Could not detect columns {missing}. Found columns: {list(df.columns)}")
    return out

def estimate_sampling_rate(time_values):
    t = pd.to_numeric(pd.Series(time_values), errors="coerce").dropna().to_numpy(float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("No positive timestamp differences.")
    median_dt = float(np.median(dt))
    return median_dt, 1.0 / median_dt

def split_contiguous_segments(time_values, gap_factor=2.0):
    t = np.asarray(time_values, dtype=float)
    dt = np.diff(t)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive))
    gap_idx = np.where(dt > gap_factor * median_dt)[0]
    starts = np.r_[0, gap_idx + 1]
    ends = np.r_[gap_idx + 1, len(t)]
    return starts, ends, median_dt
