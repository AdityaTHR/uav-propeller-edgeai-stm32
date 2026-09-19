
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, TABLES_DIR, MODELS_DIR, PROJECT_ROOT


def f32_literal(x):
    return f"{np.float32(x):.9g}f"


def leaf_class(estimator, node_id):
    values = estimator.tree_.value[node_id][0]
    class_index = int(np.argmax(values))
    return int(estimator.classes_[class_index])


def emit_node(estimator, node_id, indent="    "):
    tree = estimator.tree_
    left = int(tree.children_left[node_id])
    right = int(tree.children_right[node_id])

    if left == right:
        return f"{indent}return {leaf_class(estimator, node_id)};\n"

    feature = int(tree.feature[node_id])
    threshold = f32_literal(tree.threshold[node_id])

    text = f"{indent}if (x[{feature}] <= {threshold}) {{\n"
    text += emit_node(estimator, left, indent + "    ")
    text += f"{indent}}} else {{\n"
    text += emit_node(estimator, right, indent + "    ")
    text += f"{indent}}}\n"
    return text


def generate_model_c(model, feature_names, out_dir):
    header = f"""#ifndef ADABOOST_MODEL_H
#define ADABOOST_MODEL_H

#define UAV_FEATURE_COUNT {len(feature_names)}
#define UAV_CLASS_COUNT 5
#define UAV_TREE_COUNT {len(model.estimators_)}

int adaboost_predict(const float x[UAV_FEATURE_COUNT]);

#endif
"""

    c_lines = ['#include "adaboost_model.h"\n\n']

    for i, estimator in enumerate(model.estimators_):
        c_lines.append(f"static int tree_{i}(const float *x) {{\n")
        c_lines.append(emit_node(estimator, 0))
        c_lines.append("}\n\n")

    weights = ", ".join(f32_literal(w) for w in model.estimator_weights_)
    c_lines.append(
        f"static const float TREE_WEIGHTS[UAV_TREE_COUNT] = {{{weights}}};\n\n"
    )

    c_lines.append("int adaboost_predict(const float x[UAV_FEATURE_COUNT]) {\n")
    c_lines.append("    float votes[UAV_CLASS_COUNT] = {0};\n")
    for i in range(len(model.estimators_)):
        c_lines.append(f"    votes[tree_{i}(x)] += TREE_WEIGHTS[{i}];\n")
    c_lines.append("    int best = 0;\n")
    c_lines.append("    for (int c = 1; c < UAV_CLASS_COUNT; ++c) {\n")
    c_lines.append("        if (votes[c] > votes[best]) best = c;\n")
    c_lines.append("    }\n")
    c_lines.append("    return best;\n")
    c_lines.append("}\n")

    (out_dir / "adaboost_model.h").write_text(header, encoding="utf-8")
    (out_dir / "adaboost_model.c").write_text("".join(c_lines), encoding="utf-8")


def make_test_vectors(model, feature_names, df, out_dir):
    parts = []
    for label in range(5):
        parts.append(df[(df["domain"] == 4) & (df["label"] == label)].head(4))

    test_df = pd.concat(parts, ignore_index=True)
    X = test_df[feature_names]
    expected = model.predict(X).astype(int)

    csv = test_df[feature_names + ["label", "class_name"]].copy()
    csv["expected_prediction"] = expected
    csv.to_csv(TABLES_DIR / "adaboost_fft512_c_parity_vectors.csv", index=False)

    vector_rows = []
    for row in X.to_numpy(dtype=np.float32):
        vals = ", ".join(f32_literal(v) for v in row)
        vector_rows.append("    {" + vals + "}")

    expected_rows = ", ".join(str(int(v)) for v in expected)

    header = f"""#ifndef TEST_VECTORS_H
#define TEST_VECTORS_H

#include "adaboost_model.h"

#define TEST_VECTOR_COUNT {len(test_df)}

static const float TEST_VECTORS[TEST_VECTOR_COUNT][UAV_FEATURE_COUNT] = {{
{",\n".join(vector_rows)}
}};

static const int EXPECTED_PREDICTIONS[TEST_VECTOR_COUNT] = {{{expected_rows}}};

#endif
"""
    (out_dir / "test_vectors.h").write_text(header, encoding="utf-8")


def make_host_test(out_dir):
    code = """#include <stdio.h>
#include "adaboost_model.h"
#include "test_vectors.h"

int main(void) {
    int passed = 0;

    for (int i = 0; i < TEST_VECTOR_COUNT; ++i) {
        int pred = adaboost_predict(TEST_VECTORS[i]);
        int expected = EXPECTED_PREDICTIONS[i];

        printf("vector=%02d predicted=%d expected=%d %s\\n",
               i, pred, expected, (pred == expected) ? "PASS" : "FAIL");

        if (pred == expected) {
            passed++;
        }
    }

    printf("\\nPARITY: %d/%d\\n", passed, TEST_VECTOR_COUNT);
    return (passed == TEST_VECTOR_COUNT) ? 0 : 1;
}
"""
    (out_dir / "host_parity_test.c").write_text(code, encoding="utf-8")

    makefile = """CC=gcc
CFLAGS=-O2 -std=c11 -Wall -Wextra -Werror

all: host_parity_test

host_parity_test: host_parity_test.c adaboost_model.c
\t$(CC) $(CFLAGS) host_parity_test.c adaboost_model.c -o host_parity_test

test: host_parity_test
\t./host_parity_test

clean:
\trm -f host_parity_test
"""
    (out_dir / "Makefile").write_text(makefile, encoding="utf-8")


def main():
    metrics = json.loads(
        (TABLES_DIR / "adaboost_xy_fft512_final_metrics.json").read_text(encoding="utf-8")
    )
    feature_names = metrics["feature_order"]

    model = joblib.load(MODELS_DIR / "adaboost_xy_fft512_final.joblib")
    df = pd.read_csv(PROCESSED_DIR / "features_adaboost_xy_fft512.csv")

    out_dir = PROJECT_ROOT / "embedded" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_model_c(model, feature_names, out_dir)
    make_test_vectors(model, feature_names, df, out_dir)
    make_host_test(out_dir)

    meta = {
        "model": "AdaBoost_XY_FFT512",
        "feature_order": feature_names,
        "classes": {
            "0": "Healthy",
            "1": "Damaged Bottom Right Blade",
            "2": "Damaged Top Right Blade",
            "3": "Unbalanced Bottom Right Blade",
            "4": "Unbalanced Top Right Blade",
        },
        "rule": "All exported C parity vectors must match Python before CubeIDE integration.",
    }

    (out_dir / "model_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("ADABOOST -> C EXPORT COMPLETE")
    print("=" * 88)
    print(f"Trees: {len(model.estimators_)}")
    print(f"Features: {len(feature_names)}")
    print(f"Generated in: {out_dir}")
    print("\\nNow run:")
    print(f"  cd {out_dir}")
    print("  make clean && make test")


if __name__ == "__main__":
    main()
