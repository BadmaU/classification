from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "rusentitweet.csv"
DEMO_DATA_PATH = RAW_DATA_DIR / "sample.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "clean.csv"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"
BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_report.json"
BASELINE_PREDICTIONS_PATH = REPORTS_DIR / "baseline_predictions.csv"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
ERRORS_PATH = REPORTS_DIR / "errors.csv"
CLASS_METRICS_PATH = REPORTS_DIR / "class_metrics.csv"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
COMPARISON_PLOT_PATH = FIGURES_DIR / "model_comparison.png"

TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.25
