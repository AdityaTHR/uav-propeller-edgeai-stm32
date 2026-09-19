
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import PROCESSED_DIR, TABLES_DIR, MODELS_DIR

RANDOM_STATE = 42
TOLERANCE = 0.01  # 1 percentage point of Macro-F1

# Embedded cost tiers are ordered from cheapest to most expensive.
# This is intentionally coarse, not fake cycle-counting.
FEATURE_TIERS = {
    "base_time": ["mean", "std", "rms", "ptp", "crest"],
    "time_plus_skew": ["mean", "std", "rms", "ptp", "crest", "skew"],
    "time_all": ["mean", "std", "rms", "ptp", "crest", "skew", "kurtosis"],
    "time_plus_centroid": ["mean", "std", "rms", "ptp", "crest", "skew", "kurtosis", "spec_centroid"],
    "all_features": ["mean", "std", "rms", "ptp", "crest", "skew", "kurtosis",
                     "dom_freq", "spec_centroid", "spec_entropy"],
}

TIER_ORDER = list(FEATURE_TIERS.keys())


def make_adaboost(seed=RANDOM_STATE):
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


def make_compact_xgb(cfg, seed=RANDOM_STATE):
    return XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=4,
        reg_lambda=1.0,
    )


def tier_columns(all_features, axes, feature_suffixes):
    prefixes = tuple(f"{axis.lower()}_" for axis in axes)
    wanted_suffixes = set(feature_suffixes)
    cols = []
    for c in all_features:
        if not c.startswith(prefixes):
            continue
        suffix = c.split("_", 1)[1]
        if suffix in wanted_suffixes:
            cols.append(c)
    return cols


def evaluate_candidate(dev, cols, model_factory):
    groups = dev["domain"]
    cv = GroupKFold(n_splits=4)

    f1s = []
    accs = []

    for tr, va in cv.split(dev[cols], dev["label"], groups):
        model = model_factory()
        model.fit(dev.iloc[tr][cols], dev.iloc[tr]["label"])
        pred = model.predict(dev.iloc[va][cols])

        f1s.append(f1_score(dev.iloc[va]["label"], pred, average="macro"))
        accs.append(accuracy_score(dev.iloc[va]["label"], pred))

    return (
        float(np.mean(f1s)),
        float(np.std(f1s, ddof=1)),
        float(np.mean(accs)),
    )


def holdout_metrics(model, holdout, cols):
    pred = model.predict(holdout[cols])
    report = classification_report(
        holdout["label"], pred, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(holdout["label"], pred, average="macro")),
        "accuracy": float(accuracy_score(holdout["label"], pred)),
        "min_class_recall": float(min(report[str(i)]["recall"] for i in range(5))),
        "per_class_recall": {str(i): float(report[str(i)]["recall"]) for i in range(5)},
        "confusion_matrix": confusion_matrix(holdout["label"], pred).tolist(),
    }


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "features_500sample.csv")
    dev = df[df["domain"] < 4].copy()
    holdout = df[df["domain"] == 4].copy()

    all_features = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]

    compact_cfg = json.loads(
        (TABLES_DIR / "compact_xgboost_chosen.json").read_text(encoding="utf-8")
    )

    experiments = {
        "AdaBoost_XY": {
            "axes": "XY",
            "factory": lambda: make_adaboost(),
        },
        "XGBoost_YZ_compact": {
            "axes": "YZ",
            "factory": lambda: make_compact_xgb(compact_cfg),
        },
    }

    rows = []
    selection = {}

    for model_name, spec in experiments.items():
        print("\n" + "=" * 92)
        print(f"EMBEDDED FEATURE-COST GATE — {model_name}")
        print("=" * 92)

        local_rows = []

        for tier_index, tier_name in enumerate(TIER_ORDER):
            cols = tier_columns(
                all_features,
                spec["axes"],
                FEATURE_TIERS[tier_name],
            )

            mean_f1, std_f1, mean_acc = evaluate_candidate(
                dev, cols, spec["factory"]
            )

            row = {
                "model": model_name,
                "axes": spec["axes"],
                "tier": tier_name,
                "tier_index": tier_index,
                "n_features": len(cols),
                "uses_fft": int(
                    any(
                        f.endswith(("dom_freq", "spec_centroid", "spec_entropy"))
                        for f in cols
                    )
                ),
                "cv_macro_f1_mean": mean_f1,
                "cv_macro_f1_std": std_f1,
                "cv_accuracy_mean": mean_acc,
            }
            local_rows.append(row)
            rows.append(row)

        local = pd.DataFrame(local_rows)
        print(local.to_string(index=False))

        best_f1 = float(local["cv_macro_f1_mean"].max())
        eligible = local[
            local["cv_macro_f1_mean"] >= best_f1 - TOLERANCE
        ].copy()

        # Lowest feature-cost tier within 1 percentage point of best dev CV.
        chosen = eligible.sort_values(
            ["tier_index", "n_features", "cv_macro_f1_mean"],
            ascending=[True, True, False],
        ).iloc[0]

        chosen_tier = chosen["tier"]
        chosen_cols = tier_columns(
            all_features,
            spec["axes"],
            FEATURE_TIERS[chosen_tier],
        )

        model = spec["factory"]()
        model.fit(dev[chosen_cols], dev["label"])
        metrics = holdout_metrics(model, holdout, chosen_cols)

        out_name = model_name.lower()
        model_path = MODELS_DIR / f"{out_name}_embedded_feature_candidate.joblib"
        joblib.dump(model, model_path)

        selection[model_name] = {
            "axes": spec["axes"],
            "tier": chosen_tier,
            "features": chosen_cols,
            "uses_fft": bool(chosen["uses_fft"]),
            "cv_macro_f1_mean": float(chosen["cv_macro_f1_mean"]),
            "cv_macro_f1_std": float(chosen["cv_macro_f1_std"]),
            "holdout": metrics,
            "model_path": str(model_path),
        }

        print("\nCHOSEN EMBEDDED FEATURE TIER:")
        print(f"  Tier: {chosen_tier}")
        print(f"  Features ({len(chosen_cols)}): {chosen_cols}")
        print(f"  Uses FFT-derived feature(s): {bool(chosen['uses_fft'])}")
        print(
            f"  Holdout Macro-F1={metrics['macro_f1']:.6f} | "
            f"Accuracy={metrics['accuracy']:.6f} | "
            f"Minimum class recall={metrics['min_class_recall']:.6f}"
        )
        print("  Per-class recall:", metrics["per_class_recall"])

    pd.DataFrame(rows).to_csv(
        TABLES_DIR / "embedded_feature_cost_gate.csv", index=False
    )
    (TABLES_DIR / "embedded_feature_candidates.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 92)
    print("WHY THIS GATE EXISTS")
    print("=" * 92)
    print("A model may be small while its feature extraction is expensive.")
    print("FFT-derived features are therefore treated as a higher-cost tier.")
    print("No fake MCU cycle counts are claimed here; actual Flash/RAM/runtime come later.")


if __name__ == "__main__":
    main()
