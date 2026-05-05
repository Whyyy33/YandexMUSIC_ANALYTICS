"""
Конфигурация проекта.
"""
from pathlib import Path

# Корень проекта (определяется автоматически от расположения этого файла)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папки с данными
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
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

# Создаём папки если их нет
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_parquet(name: str) -> Path:
    """Ищет parquet по имени: сначала data/raw/<name>.parquet, потом flat/500m/."""
    candidates = [
        RAW_DATA_DIR / f"{name}.parquet",
        RAW_DATA_DIR / "flat" / "500m" / f"{name}.parquet",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1024:
            return p
    raise FileNotFoundError(f"Parquet '{name}' не найден. Сначала запусти download_data.py")