"""
Глобальный rank-based справочник меток для артистов и треков.

В Yambda нет имён — только числовые ID. Чтобы графики были читаемыми,
строим компактные метки на основе суммарной популярности:
    artist_id с наибольшим числом прослушиваний → "A1"
    следующий                                    → "A2"
    ...
    item_id (трек) аналогично                    → "T1", "T2", ...

Метки строятся ОДИН раз на полном датасете, кешируются в data/processed/.
Все задачи используют один и тот же маппинг, поэтому артист "A1" в task1
и task3 — это один и тот же физический артист.

Память: агрегация и сортировка делаются в DuckDB с явным memory_limit
и пишутся напрямую в parquet через COPY (...) TO ... (FORMAT PARQUET).
В Python приходит только финальный маленький файл (artist_id+label,
item_id+label), без материализации полного 500M-row датасета в RAM.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import polars as pl

from src.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    PROCESSED_DATA_DIR,
    find_parquet,
)

ARTIST_LABELS_PARQUET = PROCESSED_DATA_DIR / "artist_rank_labels.parquet"
TRACK_LABELS_PARQUET  = PROCESSED_DATA_DIR / "track_rank_labels.parquet"


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute("PRAGMA temp_directory='data/processed/.duckdb_tmp'")
    return con


def _build_artist_labels_parquet() -> None:
    """Пишет artist_rank_labels.parquet через DuckDB COPY, БЕЗ материализации в Python."""
    print(f"  Строим глобальный rank артистов (DuckDB memory_limit={DUCKDB_MEMORY_LIMIT})...")
    con = _duckdb_connect()
    listens = str(find_parquet("listens")).replace("\\", "/")
    artmap  = str(find_parquet("artist_item_mapping")).replace("\\", "/")
    out     = str(ARTIST_LABELS_PARQUET).replace("\\", "/")

    con.execute(f"""
        COPY (
            WITH agg AS (
                SELECT a.artist_id, COUNT(*) AS listens
                FROM read_parquet('{listens}') l
                JOIN read_parquet('{artmap}')  a ON l.item_id = a.item_id
                GROUP BY a.artist_id
            )
            SELECT
                artist_id,
                listens,
                'A' || ROW_NUMBER() OVER (ORDER BY listens DESC, artist_id) AS label
            FROM agg
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()


def _build_track_labels_parquet() -> None:
    """Пишет track_rank_labels.parquet через DuckDB COPY, БЕЗ материализации в Python."""
    print(f"  Строим глобальный rank треков (DuckDB memory_limit={DUCKDB_MEMORY_LIMIT})...")
    con = _duckdb_connect()
    listens = str(find_parquet("listens")).replace("\\", "/")
    out     = str(TRACK_LABELS_PARQUET).replace("\\", "/")

    con.execute(f"""
        COPY (
            WITH agg AS (
                SELECT item_id, COUNT(*) AS listens
                FROM read_parquet('{listens}')
                GROUP BY item_id
            )
            SELECT
                item_id,
                listens,
                'T' || ROW_NUMBER() OVER (ORDER BY listens DESC, item_id) AS label
            FROM agg
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()


def get_artist_labels(force_rebuild: bool = False) -> pl.DataFrame:
    """Возвращает df(artist_id, listens, label). Кеш в data/processed/.

    Файл собирается через DuckDB COPY и читается в Polars готовым —
    промежуточная агрегация на 500M строк не материализуется в Python.
    """
    if force_rebuild or not ARTIST_LABELS_PARQUET.exists():
        t0 = time.perf_counter()
        _build_artist_labels_parquet()
        print(f"  Артисты — кеш записан в {ARTIST_LABELS_PARQUET.name} "
              f"({time.perf_counter() - t0:.1f} сек)")
    df = pl.read_parquet(ARTIST_LABELS_PARQUET)
    print(f"  Артистов: {df.height:,}, размер фрейма: {df.estimated_size() / 1e6:.1f} МБ")
    return df


def get_track_labels(force_rebuild: bool = False) -> pl.DataFrame:
    """Возвращает df(item_id, listens, label). Кеш в data/processed/."""
    if force_rebuild or not TRACK_LABELS_PARQUET.exists():
        t0 = time.perf_counter()
        _build_track_labels_parquet()
        print(f"  Треки — кеш записан в {TRACK_LABELS_PARQUET.name} "
              f"({time.perf_counter() - t0:.1f} сек)")
    df = pl.read_parquet(TRACK_LABELS_PARQUET)
    print(f"  Треков: {df.height:,}, размер фрейма: {df.estimated_size() / 1e6:.1f} МБ")
    return df


def artist_label_map() -> dict[int, str]:
    df = get_artist_labels()
    return dict(zip(df["artist_id"].to_list(), df["label"].to_list()))


def track_label_map() -> dict[int, str]:
    df = get_track_labels()
    return dict(zip(df["item_id"].to_list(), df["label"].to_list()))


if __name__ == "__main__":
    print("Building rank labels (full dataset scan)...")
    a = get_artist_labels(force_rebuild=True)
    t = get_track_labels(force_rebuild=True)
    print(a.head(5))
    print(t.head(5))
