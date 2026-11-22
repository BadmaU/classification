import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from .config import (
    BEST_MODEL_PATH,
    CLASS_METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    ERRORS_PATH,
    LABEL_COL,
    TEXT_COL,
)
from .load_data import split_data
from .predict import load_model
from .preprocess import get_training_data


def build_error_report(model=None, model_path=BEST_MODEL_PATH):
    if model is None:
        model = load_model(model_path)

    _, test_df = split_data(get_training_data())
    y_true = test_df[LABEL_COL]
    y_pred = pd.Series(model.predict(test_df[TEXT_COL]), index=test_df.index)

    errors = test_df.copy()
    errors["predicted"] = y_pred
    errors = errors[y_true != y_pred].copy()

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_df[TEXT_COL])
        errors["confidence"] = [round(float(p.max()), 4) for p in probabilities[y_true != y_pred]]

    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERRORS_PATH, index=False)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_metrics = pd.DataFrame(report).T
    class_metrics.to_csv(CLASS_METRICS_PATH)

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Матрица ошибок")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)

    confusions = (
        pd.DataFrame({"true": y_true, "predicted": y_pred})
        .query("true != predicted")
        .value_counts()
        .head(5)
    )

    print(f"Ошибок: {len(errors)} из {len(test_df)}")
    print("Частые пары ошибок:")
    print(confusions.to_string() if not confusions.empty else "нет")
    print(f"Отчёт: {ERRORS_PATH}")
    print(f"Матрица ошибок: {CONFUSION_MATRIX_PATH}")
    return errors, class_metrics


def main():
    parser = argparse.ArgumentParser(description="Анализ ошибок модели")
    parser.add_argument("--model", default=str(BEST_MODEL_PATH), help="Путь к сохранённой модели")
    args = parser.parse_args()
    build_error_report(model_path=Path(args.model))


if __name__ == "__main__":
    main()
