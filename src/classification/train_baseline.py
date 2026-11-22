import argparse
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from .config import (
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    LABEL_COL,
    RANDOM_STATE,
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

    print(classification_report(test_df[LABEL_COL], y_pred, zero_division=0))
    print(f"Метрики baseline: {metrics}")
    print(f"Модель сохранена: {BASELINE_MODEL_PATH}")
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Обучение baseline-модели")
    parser.parse_args()
    train_baseline()


if __name__ == "__main__":
    main()
