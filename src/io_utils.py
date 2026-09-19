from pathlib import Path
import pandas as pd

SUPPORTED = {".xlsx", ".xls", ".csv"}

def list_data_files(raw_dir: Path):
    return sorted(p for p in raw_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED)

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")
