import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LABEL_COL, RANDOM_STATE, RAW_DATA_PATH, TEXT_COL, TEST_SIZE


def load_dataset(path=RAW_DATA_PATH, text_col=TEXT_COL, label_col=LABEL_COL):
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {sorted(missing)}")

    df = df[[text_col, label_col]].dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[(df[text_col] != "") & (df[label_col] != "")].reset_index(drop=True)

    if df.empty:
        raise ValueError("Датасет пуст после проверки данных")
    return df


def class_distribution(df, label_col=LABEL_COL):
    return df[label_col].value_counts().sort_index()


def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, label_col=LABEL_COL):
    counts = df[label_col].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Для стратифицированного сплита нужно минимум 2 класса по 2 примера")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def describe_dataset(df, label_col=LABEL_COL):
    print(f"Строк: {len(df)}")
    print("Распределение классов:")
    print(class_distribution(df, label_col).to_string())


def main():
    parser = argparse.ArgumentParser(description="Загрузка и проверка датасета")
    parser.add_argument("--path", default=str(RAW_DATA_PATH), help="Путь к CSV с колонками text,label")
    args = parser.parse_args()

    df = load_dataset(args.path)
    describe_dataset(df)


if __name__ == "__main__":
    main()
