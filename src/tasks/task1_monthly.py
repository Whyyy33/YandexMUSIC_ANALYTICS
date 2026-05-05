"""
Task 1 — Динамика прослушиваний по месяцам.

timestamp в Yambda — абсолютное время в 5-секундных тиках от точки отсчёта датасета
(значения монотонно возрастают и не требуют cum_sum).
month_offset = floor(timestamp * 5 / 2592000) даёт глобальный месяц от начала наблюдений.
Сравниваем органику vs рекомендации (is_organic).

Вывод: data/results/task1_monthly.png
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, TIMESTAMP_UNIT_SECONDS, find_parquet

SECONDS_PER_MONTH = 30 * 24 * 3600


def run() -> None:
    t0 = time.perf_counter()
    print("Task 1: Динамика прослушиваний по месяцам...")

    path = find_parquet("listens")
    artist_map_path = find_parquet("artist_item_mapping")
    df_monthly = (
        pl.scan_parquet(path)
        .select(["uid", "item_id", "timestamp", "is_organic"])
        .with_columns(
            (pl.col("timestamp").cast(pl.Int64) * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
        )
        .with_columns(
            (pl.col("ts_seconds") // SECONDS_PER_MONTH).cast(pl.Int32).alias("month_offset")
        )
        .group_by(["month_offset", "is_organic"])
        .agg(
            pl.len().alias("events"),
            pl.col("uid").n_unique().alias("unique_users"),
        )
        .sort("month_offset")
        .collect(engine="streaming")
    )
    print("  Месячная динамика готова")

    # Ветка 2 — топ артистов (streaming + join)
    df_artists = (
        pl.scan_parquet(path)
        .select(["item_id", "is_organic"])
        .join(pl.scan_parquet(artist_map_path), on="item_id", how="left")
        .filter(pl.col("artist_id").is_not_null())
        .group_by(["artist_id", "is_organic"])
        .agg(pl.len().alias("listens"))
        .collect(engine="streaming")
    )
    print("  Топ артистов готов")

    # --- Подготовка данных ---
    df_monthly = df_monthly.filter(pl.col("month_offset").is_between(0, 23))
    organic_m = df_monthly.filter(pl.col("is_organic") == 1)
    reco_m = df_monthly.filter(pl.col("is_organic") == 0)

    top_organic = (
        df_artists.filter(pl.col("is_organic") == 1)
        .sort("listens", descending=True)
        .head(15)
        .sort("listens")
    )
    top_reco = (
        df_artists.filter(pl.col("is_organic") == 0)
        .sort("listens", descending=True)
        .head(15)
        .sort("listens")
    )

    # --- Графики 2×2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Динамика прослушиваний (месячные срезы)", fontsize=14, fontweight="bold")

    # (0,0) Число событий
    ax = axes[0, 0]
    ax.plot(organic_m["month_offset"], organic_m["events"], marker="o", ms=4, label="Органика")
    ax.plot(reco_m["month_offset"], reco_m["events"], marker="s", ms=4, label="Рекомендации")
    ax.set_xlabel("Месяц с начала наблюдений")
    ax.set_ylabel("Число событий")
    ax.set_title("Число событий")
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,1) Уникальных пользователей
    ax = axes[0, 1]
    ax.plot(organic_m["month_offset"], organic_m["unique_users"], marker="o", ms=4, label="Органика")
    ax.plot(reco_m["month_offset"], reco_m["unique_users"], marker="s", ms=4, label="Рекомендации")
    ax.set_xlabel("Месяц с начала наблюдений")
    ax.set_ylabel("Уникальных пользователей")
    ax.set_title("Уникальных пользователей")
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,0) Топ-15 артистов — органика
    ax = axes[1, 0]
    labels = [str(a) for a in top_organic["artist_id"].to_list()]
    ax.barh(labels, top_organic["listens"].to_list(), color="steelblue")
    ax.set_xlabel("Прослушиваний")
    ax.set_title("Топ-15 артистов — органика")
    ax.grid(axis="x", alpha=0.3)

    # (1,1) Топ-15 артистов — рекомендации
    ax = axes[1, 1]
    labels = [str(a) for a in top_reco["artist_id"].to_list()]
    ax.barh(labels, top_reco["listens"].to_list(), color="tomato")
    ax.set_xlabel("Прослушиваний")
    ax.set_title("Топ-15 артистов — рекомендации")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "task1_monthly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
