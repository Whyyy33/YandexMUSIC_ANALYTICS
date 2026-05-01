"""
Базовые функции для загрузки и первичной обработки данных Yambda-500M.
"""

import time
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl

from src.config import (
    ANALYTICS_DB,
    DISLIKES_PARQUET,
    LIKES_PARQUET,
    LISTENS_PARQUET,
    MULTI_EVENT_PARQUET,
    TIMESTAMP_UNIT_SECONDS,
)


def get_connection(db_path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Возвращает подключение к DuckDB.

    TODO: добавить параметр use_motherduck: bool = False и ветку
      if use_motherduck: return duckdb.connect(f"md:{db_name}")
    """
    path = db_path or ANALYTICS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


# ---------------------------------------------------------------------------
# Чтение parquet — ленивый режим Polars (данные не грузятся в память до collect())
# ---------------------------------------------------------------------------

def read_listens() -> pl.LazyFrame:
    """uid, item_id, timestamp (дельта ×5 сек), is_organic, played_ratio_pct, track_length_seconds"""
    return pl.scan_parquet(LISTENS_PARQUET)


def read_likes() -> pl.LazyFrame:
    """uid, item_id, timestamp (дельта ×5 сек), is_organic"""
    return pl.scan_parquet(LIKES_PARQUET)


def read_dislikes() -> pl.LazyFrame:
    """uid, item_id, timestamp (дельта ×5 сек), is_organic"""
    return pl.scan_parquet(DISLIKES_PARQUET)


def read_multi_event() -> pl.LazyFrame:
    """Все события в едином формате."""
    return pl.scan_parquet(MULTI_EVENT_PARQUET)


# ---------------------------------------------------------------------------
# Восстановление времени
# ---------------------------------------------------------------------------

def restore_absolute_time(lf: pl.LazyFrame, uid_col: str = "uid", ts_col: str = "timestamp") -> pl.LazyFrame:
    """Добавляет колонку ts_seconds — время с первого события пользователя в секундах.

    timestamp в Yambda — дельта между соседними событиями одного пользователя,
    в единицах 5 секунд. Чтобы получить относительное время:
        ts_seconds = cumsum(timestamp) × 5  (по каждому uid отдельно)

    Абсолютные календарные даты из этих данных восстановить нельзя —
    глобальная эпоха в датасете не задана. Для task5 (heatmap по часам/дням)
    используем остаток от деления на длину недели/суток в секундах.
    """
    return lf.with_columns(
        (
            pl.col(ts_col)
            .cum_sum()
            .over(uid_col)
            * TIMESTAMP_UNIT_SECONDS
        ).alias("ts_seconds")
    )


# ---------------------------------------------------------------------------
# Базовая валидация
# ---------------------------------------------------------------------------

def validate_data(lf: pl.LazyFrame, name: str = "dataset") -> dict:
    """Собирает базовую статистику по датафрейму.

    Возвращает словарь с ключами: rows, columns, nulls, schema.
    Печатает сводку в stdout.
    """
    t0 = time.perf_counter()

    df = lf.collect()
    elapsed = time.perf_counter() - t0

    row_count = len(df)
    schema = df.schema

    null_counts = {col: df[col].null_count() for col in df.columns}
    total_nulls = sum(null_counts.values())

    dup_count = row_count - df.n_unique()

    print(f"\n=== {name} ===")
    print(f"Строк:       {row_count:,}")
    print(f"Колонок:     {len(df.columns)}  {list(df.columns)}")
    print(f"Дубликатов:  {dup_count:,}")
    print(f"Нулей:       {total_nulls:,}  (по колонкам: {null_counts})")
    print(f"Время сбора: {elapsed:.1f} с")

    if "played_ratio_pct" in df.columns:
        over_100 = (df["played_ratio_pct"] > 100).sum()
        print(f"played_ratio_pct > 100: {over_100:,}  (перемотки/повторы)")

    if "is_organic" in df.columns:
        organic_share = df["is_organic"].mean()
        print(f"Доля organic (is_organic=1): {organic_share:.1%}")

    print()

    return {
        "rows": row_count,
        "columns": len(df.columns),
        "nulls": null_counts,
        "schema": schema,
        "duplicates": dup_count,
    }
