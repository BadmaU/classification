import argparse
import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from .config import (
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    BASELINE_PREDICTIONS_PATH,
    BASELINE_REPORT_PATH,
    CLEAN_DATA_PATH,
    LABEL_COL,
    RANDOM_STATE,
    ROOT_DIR,
    TEST_SIZE,
    TEXT_COL,
)
from .load_data import split_data
from .preprocess import get_training_data


def build_baseline():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def evaluate(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    }


def relative_path(path):
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def train_baseline():
    df = get_training_data()
    train_df, test_df = split_data(df)

    model = build_baseline()
    model.fit(train_df[TEXT_COL], train_df[LABEL_COL])
    y_pred = model.predict(test_df[TEXT_COL])
    metrics = evaluate(test_df[LABEL_COL], y_pred)

    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASELINE_MODEL_PATH)

    BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions = pd.DataFrame(
        {
            TEXT_COL: test_df[TEXT_COL],
            "label_true": test_df[LABEL_COL],
            "label_pred": y_pred,
        }
    )
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_df[TEXT_COL])
        predictions["confidence"] = probabilities.max(axis=1).round(4)
    BASELINE_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(BASELINE_PREDICTIONS_PATH, index=False)

    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    report = {
        "reference": "tfidf_logreg_baseline",
        "dataset": CLEAN_DATA_PATH.name,
        "rows": len(df),
        "split": {
            "type": "stratified",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "classes": sorted(train_df[LABEL_COL].unique().tolist()),
        "vectorization": {
            "name": type(tfidf).__name__,
            "analyzer": tfidf.analyzer,
            "ngram_range": list(tfidf.ngram_range),
            "min_df": tfidf.min_df,
        },
        "classifier": {
            "name": type(classifier).__name__,
            "max_iter": classifier.max_iter,
            "random_state": classifier.random_state,
        },
        "metrics": metrics,
        "artifacts": {
            "model": relative_path(BASELINE_MODEL_PATH),
            "metrics": relative_path(BASELINE_METRICS_PATH),
            "predictions": relative_path(BASELINE_PREDICTIONS_PATH),
        },
    }
    BASELINE_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(classification_report(test_df[LABEL_COL], y_pred, zero_division=0))
    print(f"Метрики baseline: {metrics}")
    print(f"Модель сохранена: {BASELINE_MODEL_PATH}")
    print(f"Отчёт сохранён: {BASELINE_REPORT_PATH}")
    return model, metrics


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Обучение baseline-модели")
    parser.parse_args()
    train_baseline()


if __name__ == "__main__":
    main()
