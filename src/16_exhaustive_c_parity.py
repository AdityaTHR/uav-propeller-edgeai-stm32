from __future__ import annotations

import subprocess
import joblib
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT


def f32_literal(x):
    return f"{np.float32(x):.9g}f"


def main():
    model = joblib.load(MODELS_DIR / 'adaboost_xy_fft512_final.joblib')
    df = pd.read_csv(PROCESSED_DIR / 'features_adaboost_xy_fft512.csv')

    feature_names = list(model.feature_names_in_)
    holdout = df[df['domain'] == 4].copy()
    X = holdout[feature_names]
    expected = model.predict(X).astype(int)

    out_dir = PROJECT_ROOT / 'embedded' / 'generated'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in X.to_numpy(dtype=np.float32):
        vals = ', '.join(f32_literal(v) for v in row)
        rows.append('    {' + vals + '}')

    expected_text = ', '.join(str(int(v)) for v in expected)

    header = f'''#ifndef TEST_VECTORS_FULL_H
#define TEST_VECTORS_FULL_H

#include "adaboost_model.h"

#define FULL_TEST_VECTOR_COUNT {len(X)}

static const float FULL_TEST_VECTORS[FULL_TEST_VECTOR_COUNT][UAV_FEATURE_COUNT] = {{
{',\n'.join(rows)}
}};

static const int FULL_EXPECTED[FULL_TEST_VECTOR_COUNT] = {{{expected_text}}};

#endif
'''
    (out_dir / 'test_vectors_full.h').write_text(header, encoding='utf-8')

    test_c = '''#include <stdio.h>
#include "adaboost_model.h"
#include "test_vectors_full.h"

int main(void) {
    int passed = 0;
    int first_fail = -1;

    for (int i = 0; i < FULL_TEST_VECTOR_COUNT; ++i) {
        int pred = adaboost_predict(FULL_TEST_VECTORS[i]);
        if (pred == FULL_EXPECTED[i]) {
            passed++;
        } else if (first_fail < 0) {
            first_fail = i;
        }
    }

    printf("FULL HOLDOUT PARITY: %d/%d\\n", passed, FULL_TEST_VECTOR_COUNT);

    if (first_fail >= 0) {
        int pred = adaboost_predict(FULL_TEST_VECTORS[first_fail]);
        printf("First failure: index=%d predicted=%d expected=%d\\n",
               first_fail, pred, FULL_EXPECTED[first_fail]);
    }

    return (passed == FULL_TEST_VECTOR_COUNT) ? 0 : 1;
}
'''
    (out_dir / 'host_parity_full.c').write_text(test_c, encoding='utf-8')

    cmd = [
        'gcc', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
        'host_parity_full.c', 'adaboost_model.c', '-o', 'host_parity_full'
    ]

    print('=' * 88)
    print('EXHAUSTIVE C CLASSIFIER PARITY — ALL TEMPORAL HOLDOUT WINDOWS')
    print('=' * 88)
    print(f'Vectors: {len(X)}')

    subprocess.run(cmd, cwd=out_dir, check=True)
    result = subprocess.run(
        ['./host_parity_full'],
        cwd=out_dir,
        check=False,
        text=True,
        capture_output=True,
    )
    print(result.stdout, end='')

    if result.returncode != 0:
        raise SystemExit('FAIL: C classifier does not match Python on the full holdout.')

    print('PASS: exhaustive holdout classifier parity.')


if __name__ == '__main__':
    main()
