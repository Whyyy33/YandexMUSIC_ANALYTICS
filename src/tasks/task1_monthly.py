"""
Task 1 — Динамика прослушиваний по месяцам.

Временна́я шкала относительная: month_offset = floor(ts_seconds / 2592000).
Сравниваем органику vs рекомендации (is_organic).

Вывод: data/results/task1_monthly.png
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, TIMESTAMP_UNIT_SECONDS, find_parquet

SECONDS_PER_MONTH = 30 * 24 * 3600


def run() -> None:
    t0 = time.perf_counter()
    print("Task 1: Динамика прослушиваний по месяцам...")

    path = find_parquet("listens")

    df = (
        pl.scan_parquet(path)
        .select(["uid", "timestamp", "is_organic"])
        .with_columns(
            (pl.col("timestamp").cum_sum().over("uid") * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
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
        .collect()
    )

    # Ограничим первые 24 месяца (убираем хвост с малым числом пользователей)
    df = df.filter(pl.col("month_offset").is_between(0, 23))

    organic = df.filter(pl.col("is_organic") == 1)
    reco = df.filter(pl.col("is_organic") == 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Динамика прослушиваний (месячные срезы)", fontsize=14, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["events", "unique_users"],
        ["Число событий", "Уникальных пользователей"],
    ):
        ax.plot(organic["month_offset"], organic[col], marker="o", ms=4, label="Органика")
        ax.plot(reco["month_offset"], reco[col], marker="s", ms=4, label="Рекомендации")
        ax.set_xlabel("Месяц с начала наблюдений")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "task1_monthly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
