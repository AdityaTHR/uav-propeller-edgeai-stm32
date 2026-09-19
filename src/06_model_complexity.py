
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from config import TABLES_DIR, MODELS_DIR


def sklearn_tree_stats(tree):
    t = tree.tree_
    return {
        "trees": 1,
        "nodes": int(t.node_count),
        "max_depth": int(t.max_depth),
        "leaves": int(t.n_leaves),
    }


def adaboost_stats(model):
    trees = list(model.estimators_)
    return {
        "trees": len(trees),
        "nodes": int(sum(t.tree_.node_count for t in trees)),
        "max_depth": int(max(t.tree_.max_depth for t in trees)),
        "leaves": int(sum(t.tree_.n_leaves for t in trees)),
    }


def count_xgb_nodes(node):
    if "children" not in node:
        return 1, 1, 0

    total_nodes = 1
    total_leaves = 0
    child_depths = []

    for child in node["children"]:
        n, l, d = count_xgb_nodes(child)
        total_nodes += n
        total_leaves += l
        child_depths.append(d)

    return total_nodes, total_leaves, 1 + max(child_depths)


def xgboost_stats(model):
    dumps = model.get_booster().get_dump(dump_format="json")

    nodes = 0
    leaves = 0
    depths = []

    for raw in dumps:
        tree = json.loads(raw)
        n, l, d = count_xgb_nodes(tree)
        nodes += n
        leaves += l
        depths.append(d)

    return {
        "trees": len(dumps),
        "nodes": int(nodes),
        "max_depth": int(max(depths)),
        "leaves": int(leaves),
    }


def main():
    details_path = TABLES_DIR / "reduced_candidate_details.json"
    details = json.loads(details_path.read_text(encoding="utf-8"))

    rows = []

    for model_name, meta in details.items():
        path = Path(meta["model_path"])
        model = joblib.load(path)

        if model_name == "AdaBoost":
            stats = adaboost_stats(model)
        elif model_name == "XGBoost":
            stats = xgboost_stats(model)
        else:
            raise ValueError(model_name)

        artifact_bytes = path.stat().st_size

        rows.append({
            "model": model_name,
            "axes": meta["axes"],
            "features": meta["chosen_k"],
            "trees": stats["trees"],
            "nodes": stats["nodes"],
            "leaves": stats["leaves"],
            "max_depth": stats["max_depth"],
            "host_artifact_kb": artifact_bytes / 1024.0,
            "holdout_macro_f1": meta["holdout_metrics"]["holdout_macro_f1"],
            "holdout_accuracy": meta["holdout_metrics"]["holdout_accuracy"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "reduced_model_complexity.csv", index=False)

    print(out.to_string(index=False))
    print("\nNOTE:")
    print("host_artifact_kb is NOT STM32 Flash usage.")
    print("Trees/nodes/features are only complexity proxies.")
    print("Actual Flash/RAM comes later from the compiled STM32 firmware.")


if __name__ == "__main__":
    main()
