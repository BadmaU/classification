import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import BEST_MODEL_PATH, TEXT_COL
from .preprocess import clean_text


def load_model(path=BEST_MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}. Сначала запустите обучение")
    return joblib.load(path)


def predict_texts(texts, model=None, model_path=BEST_MODEL_PATH):
    if model is None:
        model = load_model(model_path)

    cleaned = [clean_text(text) for text in texts]
    labels = model.predict(cleaned)
    results = [{"text": text, "label": label} for text, label in zip(texts, labels)]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(cleaned)
        for row, proba in zip(results, probabilities):
            row["confidence"] = round(float(proba.max()), 4)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Предсказание классов для новых текстов")
    parser.add_argument("texts", nargs="*", help="Тексты в кавычках")
    parser.add_argument("--model", default=str(BEST_MODEL_PATH), help="Путь к сохранённой модели")
    parser.add_argument("--input-file", help="CSV с колонкой text для пакетного предсказания")
    args = parser.parse_args()

    if args.input_file:
        input_df = pd.read_csv(args.input_file)
        if TEXT_COL not in input_df.columns:
            parser.error(f"В файле нет колонки {TEXT_COL}")
        texts = input_df[TEXT_COL].dropna().astype(str).tolist()
    else:
        texts = args.texts

    if not texts:
        parser.error("Передайте тексты или используйте --input-file")

    print(predict_texts(texts, model_path=Path(args.model)).to_string(index=False))


if __name__ == "__main__":
    main()
