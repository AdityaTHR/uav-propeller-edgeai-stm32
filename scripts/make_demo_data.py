from pathlib import Path
import argparse
import pandas as pd
import numpy as np

WINDOW = 500

EXPECTED = [
    "Healthy",
    "Damaged Bottom Right Blade",
    "Damaged Top Right Blade",
    "Unbalanced Bottom Right Blade",
    "Unbalanced Top Right Blade",
]

def locate(source: Path, stem: str):
    matches = [p for p in source.rglob("*.xlsx") if stem.lower() in p.stem.lower()]
    if not matches:
        raise FileNotFoundError(f"Could not locate {stem}.xlsx under {source}")
    return matches[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder containing the full XLSX recordings")
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--windows", type=int, default=30, help="Number of 500-sample windows per class")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for stem in EXPECTED:
        path = locate(source, stem)
        df = pd.read_excel(path)

        n_full = len(df) // WINDOW
        if n_full == 0:
            raise ValueError(f"{path.name} has fewer than {WINDOW} rows")

        take = min(args.windows, n_full)

        # Spread demo windows across the full recording rather than only taking the first windows.
        ids = np.linspace(0, n_full - 1, take, dtype=int)
        chunks = [df.iloc[i*WINDOW:(i+1)*WINDOW].copy() for i in ids]
        demo = pd.concat(chunks, ignore_index=True)

        out = output / f"{stem}.xlsx"
        demo.to_excel(out, index=False)
        print(f"{stem}: {len(demo)} rows -> {out}")

if __name__ == "__main__":
    main()
