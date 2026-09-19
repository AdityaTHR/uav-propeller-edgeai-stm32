
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from config import PROCESSED_DIR, TABLES_DIR, MODELS_DIR

RANDOM_STATE = 42
TOLERANCE = 0.01  # 1 percentage point of Macro-F1


def make_xgb(n_estimators, max_depth, learning_rate):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
        reg_lambda=1.0,
    )


def tree_complexity(model):
    dumps = model.get_booster().get_dump(dump_format="json")

    def count_nodes(node):
        if "children" not in node:
            return 1, 1, 0
        nodes = 1
        leaves = 0
        depths = []
        for child in node["children"]:
            n, l, d = count_nodes(child)
            nodes += n
            leaves += l
            depths.append(d)
        return nodes, leaves, 1 + max(depths)

    total_nodes = 0
    total_leaves = 0
    max_depth = 0
    for raw in dumps:
        tree = json.loads(raw)
        n, l, d = count_nodes(tree)
        total_nodes += n
        total_leaves += l
        max_depth = max(max_depth, d)

    return len(dumps), total_nodes, total_leaves, max_depth


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "features_500sample.csv")
    details = json.loads(
        (TABLES_DIR / "reduced_candidate_details.json").read_text(encoding="utf-8")
    )

    selected = details["XGBoost"]["selected_features"]

    dev = df[df["domain"] < 4].copy()
    holdout = df[df["domain"] == 4].copy()

    groups = dev["domain"]
    cv = GroupKFold(n_splits=4)

    configs = []
    for n_estimators in [20, 30, 50, 75, 100]:
        for max_depth in [1, 2, 3]:
            for learning_rate in [0.05, 0.10]:
                configs.append((n_estimators, max_depth, learning_rate))

    rows = []

    print("=" * 88)
    print("COMPACT XGBOOST RESOURCE FRONTIER — DEVELOPMENT CV ONLY")
    print("=" * 88)

    for n_estimators, max_depth, learning_rate in configs:
        fold_f1 = []
        fold_acc = []

        for tr, va in cv.split(dev[selected], dev["label"], groups):
            model = make_xgb(n_estimators, max_depth, learning_rate)
            model.fit(dev.iloc[tr][selected], dev.iloc[tr]["label"])
            pred = model.predict(dev.iloc[va][selected])

            fold_f1.append(
                f1_score(dev.iloc[va]["label"], pred, average="macro")
            )
            fold_acc.append(
                accuracy_score(dev.iloc[va]["label"], pred)
            )

        # Fit on all development data only to measure structural complexity.
        structural_model = make_xgb(n_estimators, max_depth, learning_rate)
        structural_model.fit(dev[selected], dev["label"])
        trees, nodes, leaves, actual_depth = tree_complexity(structural_model)

        rows.append({
            "n_estimators": n_estimators,
            "max_depth_setting": max_depth,
            "learning_rate": learning_rate,
            "cv_macro_f1_mean": float(np.mean(fold_f1)),
            "cv_macro_f1_std": float(np.std(fold_f1, ddof=1)),
            "cv_accuracy_mean": float(np.mean(fold_acc)),
            "trees": trees,
            "nodes": nodes,
            "leaves": leaves,
            "actual_max_depth": actual_depth,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["cv_macro_f1_mean", "nodes"],
        ascending=[False, True],
    )
    result.to_csv(TABLES_DIR / "compact_xgboost_frontier.csv", index=False)

    print(result.head(15).to_string(index=False))

    best = float(result["cv_macro_f1_mean"].max())
    eligible = result[
        result["cv_macro_f1_mean"] >= best - TOLERANCE
    ].copy()

    chosen = eligible.sort_values(
        ["nodes", "trees", "cv_macro_f1_mean"],
        ascending=[True, True, False],
    ).iloc[0]

    print("\nSelection rule:")
    print("Choose the LOWEST structural complexity within 1 percentage point of best dev CV Macro-F1.")
    print("\nCHOSEN COMPACT XGBOOST:")
    print(chosen.to_string())

    final_model = make_xgb(
        int(chosen["n_estimators"]),
        int(chosen["max_depth_setting"]),
        float(chosen["learning_rate"]),
    )
    final_model.fit(dev[selected], dev["label"])
    pred = final_model.predict(holdout[selected])

    holdout_f1 = float(
        f1_score(holdout["label"], pred, average="macro")
    )
    holdout_acc = float(
        accuracy_score(holdout["label"], pred)
    )

    model_path = MODELS_DIR / "xgboost_compact_candidate.joblib"
    joblib.dump(final_model, model_path)

    chosen_payload = {
        "selected_features": selected,
        "n_estimators": int(chosen["n_estimators"]),
        "max_depth": int(chosen["max_depth_setting"]),
        "learning_rate": float(chosen["learning_rate"]),
        "cv_macro_f1_mean": float(chosen["cv_macro_f1_mean"]),
        "cv_macro_f1_std": float(chosen["cv_macro_f1_std"]),
        "trees": int(chosen["trees"]),
        "nodes": int(chosen["nodes"]),
        "leaves": int(chosen["leaves"]),
        "actual_max_depth": int(chosen["actual_max_depth"]),
        "holdout_macro_f1": holdout_f1,
        "holdout_accuracy": holdout_acc,
        "model_path": str(model_path),
    }

    (TABLES_DIR / "compact_xgboost_chosen.json").write_text(
        json.dumps(chosen_payload, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nTemporal holdout after selection: "
        f"Macro-F1={holdout_f1:.6f}, Accuracy={holdout_acc:.6f}"
    )
    print("\nDo not choose another configuration because of the holdout score.")


if __name__ == "__main__":
    main()
