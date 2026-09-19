
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from config import PROCESSED_DIR, TABLES_DIR, MODELS_DIR


CLASS_NAMES = {
    0: "Healthy",
    1: "Damaged Bottom Right Blade",
    2: "Damaged Top Right Blade",
    3: "Unbalanced Bottom Right Blade",
    4: "Unbalanced Top Right Blade",
}


def export_vectors(df, model, features, name):
    # Deterministic small integration set: first 4 examples of each class from domain 4.
    parts = []
    for label in range(5):
        subset = df[
            (df["domain"] == 4) & (df["label"] == label)
        ].head(4).copy()
        parts.append(subset)

    vectors = pd.concat(parts, ignore_index=True)
    pred = model.predict(vectors[features])

    out = vectors[features + ["label", "class_name"]].copy()
    out["expected_prediction"] = pred
    out["expected_prediction_name"] = [
        CLASS_NAMES[int(x)] for x in pred
    ]

    out.to_csv(
        TABLES_DIR / f"{name}_integration_test_vectors.csv",
        index=False,
    )


def main():
    df = pd.read_csv(PROCESSED_DIR / "features_500sample.csv")

    reduced = json.loads(
        (TABLES_DIR / "reduced_candidate_details.json").read_text(encoding="utf-8")
    )
    compact = json.loads(
        (TABLES_DIR / "compact_xgboost_chosen.json").read_text(encoding="utf-8")
    )

    ada_features = reduced["AdaBoost"]["selected_features"]
    xgb_features = compact["selected_features"]

    ada_model = joblib.load(
        MODELS_DIR /
        f"adaboost_{reduced['AdaBoost']['axes'].lower()}_{reduced['AdaBoost']['chosen_k']}features.joblib"
    )
    xgb_model = joblib.load(
        MODELS_DIR / "xgboost_compact_candidate.joblib"
    )

    export_vectors(
        df, ada_model, ada_features, "adaboost_reduced"
    )
    export_vectors(
        df, xgb_model, xgb_features, "xgboost_compact"
    )

    contract = {
        "project": "UAV Propeller Edge-AI STM32",
        "sampling_rate_hz_observed": 1023.5414534523528,
        "window_samples": 500,
        "window_duration_seconds_approx": 500 / 1023.5414534523528,
        "input_signal_axes_original": ["X", "Y", "Z"],
        "classes": CLASS_NAMES,
        "candidate_models": {
            "AdaBoost_reduced": {
                "features_in_exact_order": ada_features,
                "numeric_type_for_embedded_target": "float32",
                "axes": reduced["AdaBoost"]["axes"],
                "trees": 50,
            },
            "XGBoost_compact": {
                "features_in_exact_order": xgb_features,
                "numeric_type_for_embedded_target": "float32",
                "axes": "YZ",
                "n_estimators": compact["n_estimators"],
                "trees": compact["trees"],
                "max_depth": compact["actual_max_depth"],
            },
        },
        "integration_rule": (
            "Embedded implementation must reproduce the Python predictions "
            "for the exported integration test vectors before STM32 simulation."
        ),
    }

    (TABLES_DIR / "embedded_inference_contract.json").write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("INFERENCE CONTRACT / TEST VECTORS EXPORTED")
    print("=" * 88)
    print("Saved:")
    print("  outputs/tables/adaboost_reduced_integration_test_vectors.csv")
    print("  outputs/tables/xgboost_compact_integration_test_vectors.csv")
    print("  outputs/tables/embedded_inference_contract.json")
    print("\nThese files become the Python -> C parity contract.")


if __name__ == "__main__":
    main()
