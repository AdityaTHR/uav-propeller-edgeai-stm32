
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import PROCESSED_DIR, TABLES_DIR


SEEDS = [42, 123, 777, 2026, 3407]


def make_adaboost(seed):
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


def make_compact_xgb(seed, cfg):
    return XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=4,
        reg_lambda=1.0,
    )


def evaluate_domain_rotation(df, features, factory):
    rows = []

    for test_domain in sorted(df["domain"].unique()):
        train = df[df["domain"] != test_domain]
        test = df[df["domain"] == test_domain]

        model = factory(42)
        model.fit(train[features], train["label"])
        pred = model.predict(test[features])

        report = classification_report(
            test["label"],
            pred,
            output_dict=True,
            zero_division=0,
        )

        rows.append({
            "test_domain": int(test_domain),
            "macro_f1": float(
                f1_score(test["label"], pred, average="macro")
            ),
            "accuracy": float(
                accuracy_score(test["label"], pred)
            ),
            "min_class_recall": float(
                min(report[str(i)]["recall"] for i in range(5))
            ),
        })

    return pd.DataFrame(rows)


def feature_noise_stress(model, X, y, train_std, levels=(0.01, 0.03, 0.05)):
    rng = np.random.default_rng(2026)
    rows = []

    base_pred = model.predict(X)
    rows.append({
        "stress": "none",
        "level": 0.0,
        "macro_f1": float(f1_score(y, base_pred, average="macro")),
        "accuracy": float(accuracy_score(y, base_pred)),
    })

    for level in levels:
        for repetition in range(5):
            noise = rng.normal(
                loc=0.0,
                scale=train_std.to_numpy() * level,
                size=X.shape,
            )
            perturbed = X.to_numpy(dtype=float) + noise
            pred = model.predict(perturbed)

            rows.append({
                "stress": "feature_gaussian_noise",
                "level": level,
                "repetition": repetition,
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "accuracy": float(accuracy_score(y, pred)),
            })

    return pd.DataFrame(rows)


def missing_feature_stress(model, X, y, train_median):
    rows = []

    for feature in X.columns:
        modified = X.copy()
        modified[feature] = float(train_median[feature])
        pred = model.predict(modified)

        rows.append({
            "feature_replaced_by_train_median": feature,
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred)),
        })

    return pd.DataFrame(rows)


def seed_stability(dev, holdout, features, factory):
    rows = []

    for seed in SEEDS:
        model = factory(seed)
        model.fit(dev[features], dev["label"])
        pred = model.predict(holdout[features])

        rows.append({
            "seed": seed,
            "macro_f1": float(
                f1_score(holdout["label"], pred, average="macro")
            ),
            "accuracy": float(
                accuracy_score(holdout["label"], pred)
            ),
        })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(PROCESSED_DIR / "features_500sample.csv")
    dev = df[df["domain"] < 4].copy()
    holdout = df[df["domain"] == 4].copy()

    reduced = json.loads(
        (TABLES_DIR / "reduced_candidate_details.json").read_text(encoding="utf-8")
    )
    compact = json.loads(
        (TABLES_DIR / "compact_xgboost_chosen.json").read_text(encoding="utf-8")
    )

    ada_features = reduced["AdaBoost"]["selected_features"]
    xgb_features = compact["selected_features"]

    candidates = {
        "AdaBoost_reduced": {
            "features": ada_features,
            "factory": make_adaboost,
        },
        "XGBoost_compact": {
            "features": xgb_features,
            "factory": lambda seed: make_compact_xgb(seed, compact),
        },
    }

    summary_rows = []

    for name, spec in candidates.items():
        print("\n" + "=" * 88)
        print(f"GENERALIZATION / ROBUSTNESS GATE — {name}")
        print("=" * 88)

        features = spec["features"]
        factory = spec["factory"]

        model = factory(42)
        model.fit(dev[features], dev["label"])

        holdout_pred = model.predict(holdout[features])
        holdout_report = classification_report(
            holdout["label"],
            holdout_pred,
            output_dict=True,
            zero_division=0,
        )

        domain_df = evaluate_domain_rotation(df, features, factory)
        domain_df.to_csv(
            TABLES_DIR / f"{name.lower()}_domain_rotation.csv",
            index=False,
        )

        train_std = dev[features].std()
        noise_df = feature_noise_stress(
            model,
            holdout[features],
            holdout["label"],
            train_std,
        )
        noise_df.to_csv(
            TABLES_DIR / f"{name.lower()}_feature_noise_stress.csv",
            index=False,
        )

        train_median = dev[features].median()
        missing_df = missing_feature_stress(
            model,
            holdout[features],
            holdout["label"],
            train_median,
        )
        missing_df.to_csv(
            TABLES_DIR / f"{name.lower()}_missing_feature_stress.csv",
            index=False,
        )

        seed_df = seed_stability(
            dev,
            holdout,
            features,
            factory,
        )
        seed_df.to_csv(
            TABLES_DIR / f"{name.lower()}_seed_stability.csv",
            index=False,
        )

        holdout_f1 = float(
            f1_score(
                holdout["label"],
                holdout_pred,
                average="macro",
            )
        )

        min_holdout_recall = float(
            min(holdout_report[str(i)]["recall"] for i in range(5))
        )

        noise5 = noise_df[
            noise_df["level"].fillna(-1) == 0.05
        ]["macro_f1"]

        summary_rows.append({
            "model": name,
            "n_features": len(features),
            "holdout_macro_f1": holdout_f1,
            "holdout_min_class_recall": min_holdout_recall,
            "domain_rotation_macro_f1_mean": float(domain_df["macro_f1"].mean()),
            "domain_rotation_macro_f1_std": float(domain_df["macro_f1"].std(ddof=1)),
            "domain_rotation_macro_f1_min": float(domain_df["macro_f1"].min()),
            "seed_holdout_f1_mean": float(seed_df["macro_f1"].mean()),
            "seed_holdout_f1_std": float(seed_df["macro_f1"].std(ddof=1)),
            "feature_noise_5pct_f1_mean": float(noise5.mean()),
            "worst_single_missing_feature_f1": float(missing_df["macro_f1"].min()),
        })

        print("\nDomain rotation:")
        print(domain_df.to_string(index=False))

        print("\nSeed stability:")
        print(seed_df.to_string(index=False))

        print(
            "\nNOTE: feature-noise stress is a NUMERICAL feature-space stress test. "
            "It is NOT a substitute for real sensor-noise / different-UAV validation."
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        TABLES_DIR / "generalization_gate_summary.csv",
        index=False,
    )

    print("\n" + "=" * 88)
    print("GENERALIZATION GATE SUMMARY")
    print("=" * 88)
    print(summary.to_string(index=False))

    print("\nInterpretation rule:")
    print("Do not call this external/general UAV generalization.")
    print("It measures temporal/domain stability and numerical robustness inside this dataset.")


if __name__ == "__main__":
    main()
