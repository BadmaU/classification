import argparse
import time
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from .config import RAW_DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/sismetanin/rusentitweet/main"
SOURCE_URLS = {
    "train": f"{BASE_URL}/rusentitweet_train.csv",
    "test": f"{BASE_URL}/rusentitweet_test.csv",
    "full": f"{BASE_URL}/rusentitweet_full.csv",
}
DEFAULT_LABELS = "positive,neutral,negative"


def download_csv(url, retries=3):
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except URLError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def prepare_frame(raw, labels):
    df = pd.read_csv(BytesIO(raw))
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"В исходном CSV отсутствуют колонки: {sorted(missing)}")

    df = df[df["label"].isin(labels)][["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    return df.drop_duplicates().reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Загрузка RuSentiTweet из оригинального репозитория")
    parser.add_argument("--split", choices=SOURCE_URLS, default="train", help="Сплит исходного датасета")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Классы через запятую")
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DATA_DIR / "rusentitweet.csv",
        help="Путь к выходному CSV",
    )
    args = parser.parse_args()

    raw = download_csv(SOURCE_URLS[args.split])
    source = pd.read_csv(BytesIO(raw))
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    df = prepare_frame(raw, labels)
    if df.empty:
        raise ValueError("После фильтрации классов датасет пуст")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Сохранено: {args.output}")
    print(f"Строк: {len(df)} (из {len(source)} в сплите {args.split})")
    print(df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
