import argparse

from .config import CLEAN_DATA_PATH, RAW_DATA_PATH
from .error_analysis import build_error_report
from .load_data import describe_dataset, load_dataset
from .predict import predict_texts
from .preprocess import preprocess_dataframe
from .train_baseline import train_baseline
from .train_improved import train_improved

DEMO_TEXTS = [
    "Команда выиграла финал чемпионата и получила золотые медали",
    "Новый процессор показал рекордные результаты в тестах производительности",
]


def run_pipeline(skip_training=False):
    df = load_dataset(RAW_DATA_PATH)
    print("=== 1. Загрузка данных ===")
    describe_dataset(df)

    print("\n=== 2. Предобработка ===")
    clean_df = preprocess_dataframe(df)
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Сохранено: {CLEAN_DATA_PATH}")
    print(clean_df.head(3).to_string(index=False))

    if not skip_training:
        print("\n=== 3. Baseline ===")
        train_baseline()
        print("\n=== 4. Улучшения и сравнение ===")
        train_improved()
    else:
        print("\n=== 3-4. Обучение пропущено ===")

    print("\n=== 5. Анализ ошибок ===")
    build_error_report()

    print("\n=== 6. Inference ===")
    print(predict_texts(DEMO_TEXTS).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Полный ML workflow классификации текстов")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Использовать уже сохранённую модель без переобучения",
    )
    args = parser.parse_args()
    run_pipeline(skip_training=args.skip_training)


if __name__ == "__main__":
    main()
