
from __future__ import annotations

import json
import pandas as pd

from config import TABLES_DIR


CLASS_NAMES = {
    "0": "Healthy",
    "1": "Damaged Bottom Right Blade",
    "2": "Damaged Top Right Blade",
    "3": "Unbalanced Bottom Right Blade",
    "4": "Unbalanced Top Right Blade",
}


def main():
    candidates = json.loads(
        (TABLES_DIR / "embedded_feature_candidates.json").read_text(encoding="utf-8")
    )
    raw = pd.read_csv(TABLES_DIR / "raw_signal_robustness.csv")

    rows = []

    for name, meta in candidates.items():
        clean = raw[
            (raw["model"] == name) &
            (raw["condition"] == "clean")
        ].iloc[0]

        worst_perturbed = raw[
            (raw["model"] == name) &
            (raw["condition"] != "clean")
        ].sort_values("macro_f1").iloc[0]

        recalls = meta["holdout"]["per_class_recall"]
        weakest_class_id = min(recalls, key=recalls.get)

        rows.append({
            "model": name,
            "axes": meta["axes"],
            "feature_tier": meta["tier"],
            "n_features": len(meta["features"]),
            "uses_fft": meta["uses_fft"],
            "dev_cv_macro_f1": meta["cv_macro_f1_mean"],
            "holdout_macro_f1": meta["holdout"]["macro_f1"],
            "holdout_min_class_recall": meta["holdout"]["min_class_recall"],
            "weakest_holdout_class": CLASS_NAMES[weakest_class_id],
            "weakest_holdout_class_recall": recalls[weakest_class_id],
            "raw_clean_macro_f1": clean["macro_f1"],
            "worst_raw_stress_condition": worst_perturbed["condition"],
            "worst_raw_stress_macro_f1": worst_perturbed["macro_f1"],
        })

    report = pd.DataFrame(rows)
    report.to_csv(TABLES_DIR / "final_ml_gate_report.csv", index=False)

    print("=" * 100)
    print("FINAL ML GATE REPORT — DO NOT AUTO-DECLARE A WINNER")
    print("=" * 100)
    print(report.to_string(index=False))
    print("\nDecision must balance:")
    print("1. Macro-F1")
    print("2. Per-class recall")
    print("3. Temporal/domain stability")
    print("4. Raw-signal perturbation robustness")
    print("5. Feature-extraction cost")
    print("6. Tree/node complexity")
    print("\nExternal generalization is still NOT proven.")


if __name__ == "__main__":
    main()
