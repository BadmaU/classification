import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from .config import CLEAN_DATA_PATH, LABEL_COL, RAW_DATA_PATH, TEXT_COL
from .load_data import load_dataset

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\b(?:t\.co|pic\.twitter\.com)/\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@[\w.]+", re.UNICODE)
HTML_RE = re.compile(r"<[^>]+>")
REPEAT_RE = re.compile(r"(.)\1{2,}", re.DOTALL)
PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
MIN_TEXT_LEN = 2
SENTIMENT_LABELS = ("negative", "neutral", "positive")


def clean_text(text):
    if text is None or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("ё", "е")
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_label(label):
    if pd.isna(label):
        return ""
    return str(label).strip().lower()


def is_valid_text(text):
    return len(text) >= MIN_TEXT_LEN and LETTER_RE.search(text) is not None


def encode_labels(df, label_col=LABEL_COL):
    result = df.copy()
    labels = result[label_col].map(normalize_label)
    present = set(labels)
    classes = [label for label in SENTIMENT_LABELS if label in present]
    classes += sorted(present - set(classes))
    mapping = {label: index for index, label in enumerate(classes)}

    id_col = f"{label_col}_id"
    result[label_col] = labels
    result[id_col] = labels.map(mapping).astype(int)
    return result, mapping


def preprocess_dataframe(df, text_col=TEXT_COL, label_col=LABEL_COL):
    result = df.copy()
    result[text_col] = result[text_col].map(clean_text)
    result[label_col] = result[label_col].map(normalize_label)
    valid = result[text_col].map(is_valid_text) & (result[label_col] != "")
    result = result[valid].reset_index(drop=True)
    result, _ = encode_labels(result, label_col)
    return result


def get_training_data(path=None):
    if path is None:
        path = CLEAN_DATA_PATH
    if path.exists():
        return load_dataset(path)
    df = preprocess_dataframe(load_dataset(RAW_DATA_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Предобработка текста")
    parser.add_argument("--input", default=str(RAW_DATA_PATH), help="Путь к исходному CSV")
    parser.add_argument("--output", default=str(CLEAN_DATA_PATH), help="Путь к очищенному CSV")
    args = parser.parse_args()

    raw = load_dataset(args.input)
    df = preprocess_dataframe(raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Сохранено текстов: {len(df)} из {len(raw)} (удалено {len(raw) - len(df)})")
    print(f"Классы: {df[LABEL_COL].value_counts().sort_index().to_dict()}")
    print(f"label_id: {df[f'{LABEL_COL}_id'].value_counts().sort_index().to_dict()}")
    print(f"Файл: {output}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
