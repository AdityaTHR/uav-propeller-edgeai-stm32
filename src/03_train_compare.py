
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from config import PROCESSED_DIR, TABLES_DIR, MODELS_DIR

def make_models():
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2, min_samples_leaf=5, random_state=42
            ),
            n_estimators=50, learning_rate=0.5, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
            reg_lambda=1.0,
        ),
    }

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR/"features_500sample.csv")
    feature_cols = [c for c in df.columns if c.startswith(("x_","y_","z_"))]

    dev = df[df["domain"] < 4].copy()
    holdout = df[df["domain"] == 4].copy()

    X_dev, y_dev = dev[feature_cols], dev["label"]
    X_holdout, y_holdout = holdout[feature_cols], holdout["label"]
    groups = dev["domain"]

    cv = GroupKFold(n_splits=4)
    result_rows = []
    detailed = {}

    for name, model in make_models().items():
        fold_f1, fold_acc = [], []

        for fold, (tr, va) in enumerate(cv.split(X_dev, y_dev, groups), start=1):
            fold_model = make_models()[name]
            fold_model.fit(X_dev.iloc[tr], y_dev.iloc[tr])
            pred = fold_model.predict(X_dev.iloc[va])

            fold_f1.append(f1_score(y_dev.iloc[va], pred, average="macro"))
            fold_acc.append(accuracy_score(y_dev.iloc[va], pred))

        # One chronological holdout check after the predeclared model comparison.
        model.fit(X_dev, y_dev)
        pred = model.predict(X_holdout)

        row = {
            "model": name,
            "cv_macro_f1_mean": float(np.mean(fold_f1)),
            "cv_macro_f1_std": float(np.std(fold_f1, ddof=1)),
            "cv_accuracy_mean": float(np.mean(fold_acc)),
            "holdout_macro_f1": float(f1_score(y_holdout, pred, average="macro")),
            "holdout_accuracy": float(accuracy_score(y_holdout, pred)),
        }
        result_rows.append(row)

        detailed[name] = {
            "cv_macro_f1_folds": fold_f1,
            "holdout_confusion_matrix": confusion_matrix(y_holdout, pred).tolist(),
            "holdout_classification_report": classification_report(
                y_holdout, pred, output_dict=True
            ),
        }

        joblib.dump(model, MODELS_DIR/f"{name.lower()}_full_features.joblib")

    result_df = pd.DataFrame(result_rows).sort_values(
        "cv_macro_f1_mean", ascending=False
    )
    result_df.to_csv(TABLES_DIR/"model_comparison.csv", index=False)
    (TABLES_DIR/"model_comparison_details.json").write_text(
        json.dumps(detailed, indent=2), encoding="utf-8"
    )

    print(result_df.to_string(index=False))
    print("\nSaved outputs/tables/model_comparison.csv")
    print("Important: select later reductions using development CV, not by repeatedly tuning against holdout.")

if __name__ == "__main__":
    main()
