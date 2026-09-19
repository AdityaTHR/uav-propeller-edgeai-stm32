
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from config import PROCESSED_DIR, TABLES_DIR

def ada():
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=42),
        n_estimators=50, learning_rate=0.5, random_state=42
    )

def xgb():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=5,
        eval_metric="mlogloss", random_state=42, n_jobs=4, reg_lambda=1.0
    )

def main():
    df = pd.read_csv(PROCESSED_DIR/"features_500sample.csv")
    dev = df[df["domain"] < 4].copy()
    groups = dev["domain"]
    cv = GroupKFold(n_splits=4)

    all_features = [c for c in df.columns if c.startswith(("x_","y_","z_"))]
    axis_sets = {
        "X": [c for c in all_features if c.startswith("x_")],
        "Y": [c for c in all_features if c.startswith("y_")],
        "Z": [c for c in all_features if c.startswith("z_")],
        "XY": [c for c in all_features if c.startswith(("x_","y_"))],
        "XZ": [c for c in all_features if c.startswith(("x_","z_"))],
        "YZ": [c for c in all_features if c.startswith(("y_","z_"))],
        "XYZ": all_features,
    }

    rows = []
    for model_name, factory in [("AdaBoost", ada), ("XGBoost", xgb)]:
        for axes, cols in axis_sets.items():
            scores = []
            for tr, va in cv.split(dev[cols], dev["label"], groups):
                model = factory()
                model.fit(dev.iloc[tr][cols], dev.iloc[tr]["label"])
                pred = model.predict(dev.iloc[va][cols])
                scores.append(f1_score(dev.iloc[va]["label"], pred, average="macro"))

            rows.append({
                "model": model_name,
                "axes": axes,
                "n_features": len(cols),
                "cv_macro_f1_mean": float(np.mean(scores)),
                "cv_macro_f1_std": float(np.std(scores, ddof=1)),
            })

    out = pd.DataFrame(rows).sort_values(
        ["model","cv_macro_f1_mean"], ascending=[True,False]
    )
    out.to_csv(TABLES_DIR/"axis_ablation.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
