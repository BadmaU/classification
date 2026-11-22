import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import (
    BEST_MODEL_PATH,
    COMPARISON_PATH,
    COMPARISON_PLOT_PATH,
    LABEL_COL,
    RANDOM_STATE,
    TEXT_COL,
)
from .load_data import split_data
from .preprocess import get_training_data
from .train_baseline import build_baseline, evaluate


def candidate_models():
    return {
        "baseline_tfidf_logreg": build_baseline(),
        "bigram_tfidf_svc": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "char_tfidf_logreg": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1),
                ),
                ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "word_char_union_logreg": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb", ngram_range=(3, 5), min_df=1
                                ),
                            ),
                        ]
                    ),
                ),
                ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
            ]
        ),
    }


def train_improved():
    df = get_training_data()
    train_df, test_df = split_data(df)
    fit_df, val_df = split_data(train_df)

    rows = []
    fitted = {}
    for name, model in candidate_models().items():
        model.fit(fit_df[TEXT_COL], fit_df[LABEL_COL])
        y_val = model.predict(val_df[TEXT_COL])
        rows.append(
            {
                "model": name,
                "val_accuracy": round(float(accuracy_score(val_df[LABEL_COL], y_val)), 4),
                "val_f1_macro": round(float(f1_score(val_df[LABEL_COL], y_val, average="macro")), 4),
            }
        )
        fitted[name] = model

    comparison = pd.DataFrame(rows)
    best_name = comparison.sort_values(
        ["val_f1_macro", "val_accuracy"], ascending=False
    ).iloc[0]["model"]

    results = []
    for name, model in fitted.items():
        model.fit(train_df[TEXT_COL], train_df[LABEL_COL])
        y_pred = model.predict(test_df[TEXT_COL])
        metrics = evaluate(test_df[LABEL_COL], y_pred)
        results.append(
            {
                **metrics,
                "model": name,
                "selected": name == best_name,
                "val_accuracy": comparison.loc[comparison["model"] == name, "val_accuracy"].iloc[0],
                "val_f1_macro": comparison.loc[comparison["model"] == name, "val_f1_macro"].iloc[0],
            }
        )

    comparison = pd.DataFrame(results)[
        [
            "model",
            "val_accuracy",
            "val_f1_macro",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "selected",
        ]
    ].rename(columns={"accuracy": "test_accuracy", "f1_macro": "test_f1_macro", "f1_weighted": "test_f1_weighted"})

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    COMPARISON_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison["model"], comparison["val_f1_macro"])
    ax.set_ylabel("F1-macro (val)")
    ax.set_title("Сравнение подходов")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT_PATH)
    plt.close(fig)

    best_model = fitted[best_name]
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)

    print(comparison.to_string(index=False))
    print(f"Лучшая модель: {best_name} -> {BEST_MODEL_PATH}")
    return best_model, comparison


def main():
    parser = argparse.ArgumentParser(description="Сравнение улучшенных моделей")
    parser.parse_args()
    train_improved()


if __name__ == "__main__":
    main()
