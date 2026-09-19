
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import PROCESSED_DIR, TABLES_DIR, MODELS_DIR


RANDOM_STATE = 42
TOLERANCE = 0.01  # choose smallest k within 1 percentage point of best development CV score


def make_adaboost():
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        n_estimators=50,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    )


def make_xgboost():
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
        reg_lambda=1.0,
    )


def feature_importance(model, feature_names):
    values = np.asarray(model.feature_importances_, dtype=float)
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def evaluate_reduction(dev, cols, model_factory, k_values, model_name, axes_name):
    groups = dev["domain"]
    cv = GroupKFold(n_splits=4)

    rows = []
    selected_by_k = {k: Counter() for k in k_values}

    for k in k_values:
        fold_scores = []
        fold_acc = []

        for fold, (tr_idx, va_idx) in enumerate(
            cv.split(dev[cols], dev["label"], groups),
            start=1,
        ):
            train = dev.iloc[tr_idx]
            valid = dev.iloc[va_idx]

            # IMPORTANT:
            # Feature ranking is learned ONLY from the training fold.
            # This avoids selecting features using the validation fold.
            ranking_model = model_factory()
            ranking_model.fit(train[cols], train["label"])
            ranking = feature_importance(ranking_model, cols)

            selected = ranking.head(k).index.tolist()
            selected_by_k[k].update(selected)

            model = model_factory()
            model.fit(train[selected], train["label"])
            pred = model.predict(valid[selected])

            fold_scores.append(
                f1_score(valid["label"], pred, average="macro")
            )
            fold_acc.append(
                accuracy_score(valid["label"], pred)
            )

        rows.append({
            "model": model_name,
            "axes": axes_name,
            "k_features": k,
            "cv_macro_f1_mean": float(np.mean(fold_scores)),
            "cv_macro_f1_std": float(np.std(fold_scores, ddof=1)),
            "cv_accuracy_mean": float(np.mean(fold_acc)),
        })

    result = pd.DataFrame(rows).sort_values("k_features", ascending=False)

    best_score = result["cv_macro_f1_mean"].max()
    eligible = result[
        result["cv_macro_f1_mean"] >= best_score - TOLERANCE
    ].copy()

    # Embedded-oriented rule:
    # among models within 1 percentage point of the best CV result,
    # choose the one with the FEWEST features.
    chosen_k = int(eligible["k_features"].min())

    return result, chosen_k, selected_by_k


def stable_feature_list(counter, k):
    return [name for name, _ in counter.most_common(k)]


def fit_final_reduced_candidate(dev, holdout, cols, model_factory, chosen_k):
    # Rank once using ALL development data only.
    ranking_model = model_factory()
    ranking_model.fit(dev[cols], dev["label"])
    ranking = feature_importance(ranking_model, cols)
    selected = ranking.head(chosen_k).index.tolist()

    final_model = model_factory()
    final_model.fit(dev[selected], dev["label"])
    pred = final_model.predict(holdout[selected])

    metrics = {
        "holdout_macro_f1": float(
            f1_score(holdout["label"], pred, average="macro")
        ),
        "holdout_accuracy": float(
            accuracy_score(holdout["label"], pred)
        ),
        "confusion_matrix": confusion_matrix(
            holdout["label"], pred
        ).tolist(),
        "classification_report": classification_report(
            holdout["label"], pred, output_dict=True
        ),
    }

    return final_model, selected, ranking, metrics


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "features_500sample.csv")
    dev = df[df["domain"] < 4].copy()
    holdout = df[df["domain"] == 4].copy()

    all_features = [
        c for c in df.columns
        if c.startswith(("x_", "y_", "z_"))
    ]

    # These axis choices came from the PREVIOUS development-CV ablation:
    # XGBoost: YZ was nearly as strong as XYZ.
    # AdaBoost: XY was nearly as strong as XYZ.
    xgb_cols = [c for c in all_features if c.startswith(("y_", "z_"))]
    ada_cols = [c for c in all_features if c.startswith(("x_", "y_"))]

    experiments = [
        {
            "name": "XGBoost",
            "axes": "YZ",
            "cols": xgb_cols,
            "factory": make_xgboost,
            "k_values": [20, 15, 12, 10, 8, 6, 5, 4, 3],
        },
        {
            "name": "AdaBoost",
            "axes": "XY",
            "cols": ada_cols,
            "factory": make_adaboost,
            "k_values": [20, 15, 12, 10, 8, 6, 5, 4, 3],
        },
    ]

    all_results = []
    details = {}

    for exp in experiments:
        print("=" * 72)
        print(f"{exp['name']} — {exp['axes']} FEATURE REDUCTION")
        print("=" * 72)

        result, chosen_k, counters = evaluate_reduction(
            dev=dev,
            cols=exp["cols"],
            model_factory=exp["factory"],
            k_values=exp["k_values"],
            model_name=exp["name"],
            axes_name=exp["axes"],
        )

        print(result.sort_values("k_features", ascending=False).to_string(index=False))
        print(f"\nChosen k by 1%-tolerance rule: {chosen_k}")

        final_model, selected, ranking, metrics = fit_final_reduced_candidate(
            dev=dev,
            holdout=holdout,
            cols=exp["cols"],
            model_factory=exp["factory"],
            chosen_k=chosen_k,
        )

        print("Selected features:")
        for i, f in enumerate(selected, start=1):
            print(f"  {i:02d}. {f}")

        print(
            f"Holdout Macro-F1: {metrics['holdout_macro_f1']:.6f} | "
            f"Accuracy: {metrics['holdout_accuracy']:.6f}"
        )

        model_path = (
            MODELS_DIR /
            f"{exp['name'].lower()}_{exp['axes'].lower()}_{chosen_k}features.joblib"
        )
        joblib.dump(final_model, model_path)

        ranking_path = (
            TABLES_DIR /
            f"{exp['name'].lower()}_{exp['axes'].lower()}_feature_ranking.csv"
        )
        ranking.rename("importance").to_csv(ranking_path, header=True)

        details[exp["name"]] = {
            "axes": exp["axes"],
            "chosen_k": chosen_k,
            "selected_features": selected,
            "holdout_metrics": metrics,
            "model_path": str(model_path),
        }

        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(TABLES_DIR / "feature_reduction_cv.csv", index=False)

    (TABLES_DIR / "reduced_candidate_details.json").write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("IMPORTANT")
    print("=" * 72)
    print("Feature count was selected using DEVELOPMENT CV only.")
    print("The temporal holdout was evaluated only after k was chosen.")
    print("Do not keep changing k based on the holdout result.")


if __name__ == "__main__":
    main()
