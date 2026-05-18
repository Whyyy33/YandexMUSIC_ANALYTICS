"""
Конфигурация проекта.
"""
from pathlib import Path

# Корень проекта (определяется автоматически от расположения этого файла)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папки с данными
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DEMO_DATA_DIR = DATA_DIR / "demo"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Файлы датасета Yambda-500M
LISTENS_PARQUET = RAW_DATA_DIR / "listens.parquet"
LIKES_PARQUET = RAW_DATA_DIR / "likes.parquet"
DISLIKES_PARQUET = RAW_DATA_DIR / "dislikes.parquet"
MULTI_EVENT_PARQUET = RAW_DATA_DIR / "multi_event.parquet"

# Файл агрегированной БД
ANALYTICS_DB = PROCESSED_DATA_DIR / "analytics.duckdb"

# Параметры датасета Yambda
YAMBDA_DATASET = "yandex/yambda"
YAMBDA_VERSION = "flat-multievent-500m"

# Константы для расчётов
# ВАЖНО: timestamp в Yambda — абсолютное значение в 5-секундных тиках от начала датасета,
# НЕ дельта. Умножай timestamp * TIMESTAMP_UNIT_SECONDS напрямую, без cum_sum.
TIMESTAMP_UNIT_SECONDS = 5  # 1 тик = 5 секунд реального времени
SESSION_GAP_MINUTES = 30    # пауза, после которой считаем сессию новой

# Лимиты памяти / сэмпла — общий потолок для тяжёлых тасков.
# Машина: 16 ГБ RAM, оставляем запас под ОС и Polars/Python.
DUCKDB_MEMORY_LIMIT = "6GB"
DUCKDB_THREADS = 4
MAX_USERS_TASK6 = 5_000
MAX_USERS_TASK7 = 90_000
RAM_SOFT_CAP_GB = 2.0  # df.estimated_size() выше этого — падаем с явным сообщением

# Создаём папки если их нет
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_parquet(name: str) -> Path:
    """Ищет parquet по имени. Приоритет: raw -> raw/flat/500m -> demo.
    Если ничего не найдено — FileNotFoundError с понятным сообщением.
    """
    candidates = [
        RAW_DATA_DIR / f"{name}.parquet",
        RAW_DATA_DIR / "flat" / "500m" / f"{name}.parquet",
        DEMO_DATA_DIR / f"{name}.parquet",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1024:
            return p
    raise FileNotFoundError(
        f"Parquet '{name}' не найден ни в data/raw/, ни в data/demo/. "
        f"Запустите: python scripts/download_data.py (полные данные) "
        f"или распакуйте data_demo.zip (демо для дистрибутива)."
    )