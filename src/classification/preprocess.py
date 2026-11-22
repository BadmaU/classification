import argparse
import re
from pathlib import Path

from .config import CLEAN_DATA_PATH, RAW_DATA_PATH, TEXT_COL
from .load_data import load_dataset

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def preprocess_dataframe(df, text_col=TEXT_COL):
    result = df.copy()
    result[text_col] = result[text_col].map(clean_text)
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
    parser = argparse.ArgumentParser(description="Предобработка текста")
    parser.add_argument("--input", default=str(RAW_DATA_PATH), help="Путь к исходному CSV")
    parser.add_argument("--output", default=str(CLEAN_DATA_PATH), help="Путь к очищенному CSV")
    args = parser.parse_args()

    df = preprocess_dataframe(load_dataset(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Сохранено текстов: {len(df)} -> {output}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
