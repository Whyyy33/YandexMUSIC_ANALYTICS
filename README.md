# YandexMusic Analytics

Курсовой проект по дисциплине «Наука о данных и аналитика больших объёмов информации».
Аналитический пайплайн обработки датасета **Yambda-500M** от Яндекса —
500 миллионов событий взаимодействия пользователей с музыкальным сервисом
за 24 месяца.

## Быстрый старт (рекомендуемый способ)

Готовый дистрибутив доступен для скачивания. Не требует установки Python
или настройки окружения вручную.

1. Скачайте `YandexMUSIC_ANALYTICS_v1.0.zip` (~360 МБ)
2. Распакуйте архив в любую папку
3. Двойной клик по `setup.bat` — выберите режим:
   - **[1] Демо** — 5000 пользователей, оффлайн, готово за 2-3 минуты
   - **[2] Полный** — полный датасет с эмбеддингами (~15 ГБ скачивания)
4. Двойной клик по `run.bat` — запускаются все 7 аналитических задач
5. Результаты — 7 PNG-графиков — появятся в `data/results/`

Время от распаковки до первых графиков в демо-режиме: **~5 минут**.

## Аналитические задачи

1. Динамика прослушиваний по месяцам
2. Распределение сессионной активности
3. Граф ко-лайков и музыкальные сообщества (Louvain)
4. Энтропия Шеннона разнообразия слушания
5. Тепловая карта активности по часам и дням недели
6. Граф ко-прослушиваний с UMAP-валидацией по аудио-эмбеддингам
7. Прогноз оттока пользователей (Random Forest + Logistic Regression)

## Архитектура проекта

Пятислойный аналитический конвейер:

````
HuggingFace (yandex/yambda)
        │
        ▼
scripts/download_data.py  → загрузка parquet
        │
        ▼
data/raw/ + analytics.duckdb (VIEW поверх parquet)
        │
        ▼
src/tasks/task1..task7 + общие модули (config, data_loader, rank_labels)
        │
        ▼
data/results/  → 7 PNG-графиков
````

**СУБД:** DuckDB — встраиваемая аналитическая база данных, оптимизированная
под колоночные агрегации. Развёртывание автоматическое:
`pip install duckdb` + распаковка `analytics.duckdb` из дистрибутива.

## Технологический стек

- **Python 3.11+**
- **Polars** — табличная обработка с ленивой загрузкой
- **DuckDB** — аналитическая БД, SQL поверх parquet
- **NetworkX + python-louvain** — графовый анализ и кластеризация
- **UMAP-learn** — нелинейное снижение размерности эмбеддингов
- **scikit-learn** — модели прогноза оттока
- **Matplotlib + Seaborn + adjustText** — визуализация

## Запуск из исходников (для разработчиков)

Этот раздел нужен только если вы хотите модифицировать код проекта или
работать без bat-файлов.

```bash
git clone https://github.com/Whyyy33/YandexMUSIC_ANALYTICS.git
cd YandexMUSIC_ANALYTICS
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# Скачивание данных (основные, ~1.3 ГБ):
python scripts/download_data.py

# Опционально, для task6 UMAP-валидации (~13.8 ГБ):
python scripts/download_data.py --with-embeddings

# Подготовка БД:
python scripts/prepare_db.py

# Запуск всех 7 задач:
python scripts/run_all.py

# Или одной задачи:
python -m src.tasks.task1_monthly
```

## Сборка дистрибутива (для мейнтейнеров)

Воспроизводимая сборка `YandexMUSIC_ANALYTICS_v1.0.zip`:

```bash
# 1. Сгенерировать демо-сэмпл из полных данных
python scripts/build_demo.py

# 2. Скачать портативный Python для дистрибутива
python scripts/download_embeddable_python.py

# 3. Собрать финальный zip
python scripts/pack_dist.py
```

Результат: `dist/YandexMUSIC_ANALYTICS_v1.0.zip`.

## Структура репозитория

````
src/
  config.py                — параметры пайплайна
  data_loader.py           — базовые функции чтения parquet
  rank_labels.py           — глобальные ранг-метки A_k / T_k
  tasks/
    task1_monthly.py       — динамика прослушиваний
    task2_sessions.py      — сессионная активность
    task3_colikes.py       — граф ко-лайков + Louvain
    task4_diversity.py     — энтропия Шеннона
    task5_heatmap.py       — тепловая карта час/день
    task6_colistening.py   — граф ко-прослушиваний + UMAP
    task7_churn.py         — прогноз оттока (RF + LR)

scripts/
  download_data.py             — загрузка Yambda с HuggingFace
  prepare_db.py                — создание VIEW в DuckDB
  run_all.py                   — оркестратор всех задач
  build_demo.py                — сборка демо-сэмпла для дистрибутива
  pack_dist.py                 — упаковка финального zip
  download_embeddable_python.py — Windows embeddable Python

data/
  raw/                     — полные данные Yambda (gitignored)
  demo/                    — демо-сэмпл для дистрибутива (gitignored)
  processed/               — кеши rank_labels (gitignored)
  results/                 — финальные PNG

setup.bat / run.bat        — пользовательские скрипты дистрибутива
requirements.txt           — Python-зависимости
````

## Особенности данных

Yambda — анонимизированный датасет: артисты и треки представлены числовыми
идентификаторами без названий. Жанровых меток нет. В графиках используются
глобальные ранг-метки `A_k` для артистов и `T_k` для треков, отсортированные
по суммарной популярности в датасете (A1 — самый популярный артист).

Аудио-эмбеддинги (`embeddings.parquet`, 13.8 ГБ) позволяют валидировать
поведенческие сообщества артистов через акустическое подобие — реализовано
в task6 как UMAP-проекция в 2D с раскраской по сообществам Louvain.

## Источник данных

[Yandex Yambda на HuggingFace](https://huggingface.co/datasets/yandex/yambda)

## Авторы

Курсовая работа команды разработки, 2026.
