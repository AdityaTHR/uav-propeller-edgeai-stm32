from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

# Never assume sampling rate or column names before auditing the real files.
COLUMN_MAP = {"time": None, "x": None, "y": None, "z": None}
EXPECTED_SAMPLING_HZ = None

# Model ladder:
# 0: simple/statistical sanity baseline
# 1: Decision Tree
# 2: AdaBoost with shallow trees
# 3: XGBoost challenger (accepted only if gain justifies embedded cost)
MODEL_CANDIDATES = ["decision_tree", "adaboost", "xgboost"]
