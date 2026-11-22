# Классификация текстов — ML mini-product

Учебный single-task pipeline полного цикла: **data → preprocessing → vectorization → baseline → improvement → evaluation → inference**.

Решение задач 1–7 дней и финальная интеграция в единый workflow.

## Что умеет система

- загружать CSV с текстами и метками и проверять его структуру;
- очищать тексты (нижний регистр, URL, HTML, пунктуация, лишние пробелы);
- строить TF-IDF-признаки (слово, char_wb, FeatureUnion word+char);
- обучать baseline: TF-IDF + LogisticRegression;
- сравнивать 3 улучшения baseline на валидации и тесте;
- оценивать качество: accuracy, F1-macro, F1-weighted, classification report, confusion matrix;
- сохранять лучшую модель и отчёты через joblib/CSV/PNG;
- анализировать ошибки и делать предсказания на новых текстах.

## Структура проекта

```
.
├── data/
│   ├── raw/rusentitweet.csv    # основной датасет: 3 класса тональности (text,label)
│   ├── raw/sample.csv          # демо-датасет для smoke-test (text,label)
│   └── processed/clean.csv     # очищенный датасет (создаётся pipeline)
├── models/
│   ├── baseline.joblib         # baseline-модель
│   └── best_model.joblib       # лучшая модель по валидации
├── reports/
│   ├── baseline_metrics.json
│   ├── model_comparison.csv
│   ├── errors.csv
│   ├── class_metrics.csv
│   └── figures/
│       ├── model_comparison.png
│       └── confusion_matrix.png
├── notebooks/                  # исследования и визуализации
│   └── classification_day01.ipynb  # EDA дня 1
└── src/classification/
    ├── config.py                # пути и константы
    ├── fetch_data.py            # загрузка RuSentiTweet в CSV
    ├── load_data.py             # этап 1: загрузка и проверка данных
    ├── preprocess.py            # этап 2: очистка текста
    ├── train_baseline.py        # этап 3: vectorization + baseline
    ├── train_improved.py        # этап 4: улучшения и сравнение
    ├── error_analysis.py        # этап 5: разбор ошибок
    ├── predict.py               # этап 6: inference
    ├── pipeline.py              # этап 7: финальная интеграция
    └── __init__.py
```

## 7 этапов разработки

| День | Этап | Модуль | Результат |
|------|------|--------|-----------|
| 1 | Загрузка и валидация датасета | `load_data.py` | проверенный DataFrame, распределение классов |
| 2 | Предобработка текста | `preprocess.py` | `data/processed/clean.csv` |
| 3 | Baseline-модель | `train_baseline.py` | TF-IDF + LogisticRegression, метрики на тесте |
| 4 | Улучшения и сравнение | `train_improved.py` | 3 варианта признаков/классификаторов, таблица сравнения |
| 5 | Оценка и анализ ошибок | `error_analysis.py` | `errors.csv`, метрики по классам, confusion matrix |
| 6 | Inference | `predict.py` | предсказание для новых текстов и CSV-файла |
| 7 | Интеграция | `pipeline.py` | один запуск всего workflow |

## Установка

```bash
uv sync
```

Требуется Python ≥ 3.14. Зависимости: pandas, scikit-learn, matplotlib, joblib, jupyter.

Подготовка данных (скачивает train-сплит RuSentiTweet, оставляет `positive`, `neutral`, `negative`):

```bash
uv run python -m classification.fetch_data
uv run python -m classification.load_data
```

Оригинальный test-сплит авторов не используется при обучении: для оценки проект делает свой стратифицированный сплит из train.

## Запуск

Весь workflow одной командой:

```bash
uv run classification
```

Пошагово:

```bash
uv run python -m classification.load_data
uv run python -m classification.preprocess
uv run python -m classification.train_baseline
uv run python -m classification.train_improved
uv run python -m classification.error_analysis
uv run python -m classification.predict "это просто космос, очень понравилось!" "ужасно, не рекомендую"
```

Повторный запуск без переобучения:

```bash
uv run classification --skip-training
```

Пакетное предсказание из CSV (нужна колонка `text`):

```bash
uv run python -m classification.predict --input-file data/raw/rusentitweet.csv
```

## Формат датасета

CSV с двумя колонками:

```csv
text,label
"это просто космос, очень понравилось!",positive
"ужасно, не рекомендую",negative
```

Основной датасет — [RuSentiTweet](https://github.com/sismetanin/rusentitweet): 8842 твита трёх классов тональности после исключения служебных меток `skip` и `speech_act`. Распределение неравномерное: `neutral` ~48%, `negative` ~30%, `positive` ~22%. Демо-файл `sample.csv` (60 текстов классов `sport`, `tech`, `business`) оставлен для smoke-test. Можно подключить свой CSV через `--path` / `--input`; классов должно быть минимум 2, в каждом — минимум 2 примера (стратифицированный сплит).

## Методика оценки

- сплит train/test: 75/25, стратификация, `random_state=42`;
- выбор лучшей модели: F1-macro и accuracy на валидации (25% train);
- финальное качество: accuracy, F1-macro, F1-weighted на тесте;
- одна и та же модель и сплит используются для inference и error analysis.

Результаты сравнения сохраняются в `reports/model_comparison.csv`, ошибки — в `reports/errors.csv`, графики — в `reports/figures/`.

## Критерии завершения

- [x] загрузка датасета
- [x] очистка текстов
- [x] построение признаков
- [x] обучение и сохранение модели
- [x] сравнение нескольких подходов
- [x] анализ ошибок
- [x] предсказание на новых примерах
