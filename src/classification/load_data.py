import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LABEL_COL, RANDOM_STATE, RAW_DATA_PATH, TEXT_COL, TEST_SIZE


def load_dataset(path=RAW_DATA_PATH, text_col=TEXT_COL, label_col=LABEL_COL):
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {sorted(missing)}")

    id_col = f"{label_col}_id"
    columns = [text_col, label_col] + ([id_col] if id_col in df.columns else [])
    df = df[columns].dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[(df[text_col] != "") & (df[label_col] != "")].reset_index(drop=True)

    if df.empty:
        raise ValueError("Датасет пуст после проверки данных")
    return df


def dataset_info(df, text_col=TEXT_COL, label_col=LABEL_COL):
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "duplicate_texts": int(df.duplicated(subset=[text_col]).sum()),
        "duplicate_rows": int(df.duplicated(subset=[text_col, label_col]).sum()),
        "empty_texts": int((df[text_col].astype(str).str.strip() == "").sum()),
    }


def class_distribution(df, label_col=LABEL_COL):
    return df[label_col].value_counts().sort_index()


def text_lengths(df, text_col=TEXT_COL):
    text = df[text_col].fillna("").astype(str)
    return pd.DataFrame({"chars": text.str.len(), "words": text.str.split().str.len()})


def length_stats_by_class(df, text_col=TEXT_COL, label_col=LABEL_COL):
    lengths = text_lengths(df, text_col)
    return (
        pd.concat([df[[label_col]], lengths], axis=1)
        .groupby(label_col)
        .agg(
            chars_min=("chars", "min"),
            chars_median=("chars", "median"),
            chars_max=("chars", "max"),
            words_min=("words", "min"),
            words_median=("words", "median"),
            words_max=("words", "max"),
        )
    )


def show_examples(df, per_class=2, text_col=TEXT_COL, label_col=LABEL_COL, random_state=RANDOM_STATE):
    samples = [
        group.sample(min(per_class, len(group)), random_state=random_state)
        for _, group in df.groupby(label_col, sort=True)
    ]
    return pd.concat(samples)[[label_col, text_col]].reset_index(drop=True)


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


def describe_dataset(df, text_col=TEXT_COL, label_col=LABEL_COL):
    info = dataset_info(df, text_col, label_col)
    print(f"Строк: {info['rows']}")
    print(f"Колонки: {info['columns']}")
    print("Типы колонок:")
    print(pd.Series(info["dtypes"], name="dtype").to_string())
    print("Пропуски:")
    print(pd.Series(info["missing"], name="missing").to_string())
    print(f"Дубликаты text: {info['duplicate_texts']}, text+label: {info['duplicate_rows']}")
    print(f"Пустые text: {info['empty_texts']}")

    counts = class_distribution(df, label_col)
    distribution = pd.DataFrame({"count": counts, "share": counts / counts.sum()})
    print("Распределение классов:")
    print(distribution.to_string(formatters={"share": "{:.2%}".format}))

    print("Длины текстов (символы и слова):")
    print(text_lengths(df, text_col).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.to_string())
    print("Длины по классам:")
    print(length_stats_by_class(df, text_col, label_col).to_string())


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Загрузка и первичный анализ датасета")
    parser.add_argument("--path", default=str(RAW_DATA_PATH), help="Путь к CSV с колонками text,label")
    parser.add_argument("--examples", type=int, default=2, help="Примеров на класс")
    args = parser.parse_args()

    df = load_dataset(args.path)
    describe_dataset(df)
    if args.examples:
        print("Примеры текстов:")
        print(show_examples(df, per_class=args.examples).to_string(index=False))


if __name__ == "__main__":
    main()
