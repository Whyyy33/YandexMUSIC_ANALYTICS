"""
Task 5 — Тепловая карта активности: час × день недели.

Абсолютное время неизвестно, поэтому берём ts_seconds mod 604800 (1 неделя).
Предположение: данные не имеют систематического смещения по дню недели,
поэтому паттерн отражает реальные циклические предпочтения.

Вывод: data/results/task5_heatmap.png
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, TIMESTAMP_UNIT_SECONDS, find_parquet

SECONDS_PER_WEEK = 7 * 24 * 3600
SECONDS_PER_DAY  = 24 * 3600

DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def compute_heatmap(df: pl.DataFrame) -> np.ndarray:
    """Возвращает матрицу 7×24 с числом событий."""
    mat = np.zeros((7, 24), dtype=np.float64)
    for day, hour, cnt in df.iter_rows():
        mat[int(day), int(hour)] += cnt
    return mat


def run() -> None:
    t0 = time.perf_counter()
    print("Task 5: Тепловая карта час/день недели...")

    path = find_parquet("listens")

    df = (
        pl.scan_parquet(path)
        .select(["uid", "timestamp", "is_organic"])
        .with_columns(
            (pl.col("timestamp").cum_sum().over("uid") * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
        )
        .with_columns(
            ((pl.col("ts_seconds") % SECONDS_PER_WEEK) // SECONDS_PER_DAY).cast(pl.Int8).alias("day_of_week"),
            ((pl.col("ts_seconds") % SECONDS_PER_DAY) // 3600).cast(pl.Int8).alias("hour"),
        )
        .group_by(["is_organic", "day_of_week", "hour"])
        .agg(pl.len().alias("events"))
        .collect()
    )

    organic_agg = df.filter(pl.col("is_organic") == 1).select(["day_of_week", "hour", "events"])
    reco_agg    = df.filter(pl.col("is_organic") == 0).select(["day_of_week", "hour", "events"])

    mat_o = compute_heatmap(organic_agg)
    mat_r = compute_heatmap(reco_agg)

    # Нормируем по строкам (дням) для сравнимости
    mat_o_norm = mat_o / mat_o.sum(axis=1, keepdims=True)
    mat_r_norm = mat_r / mat_r.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Активность: час × день недели (доля от дня)", fontsize=14, fontweight="bold")

    for ax, mat, title in [
        (axes[0], mat_o_norm, "Органика"),
        (axes[1], mat_r_norm, "Рекомендации"),
    ]:
        sns.heatmap(
            mat, ax=ax,
            xticklabels=range(24), yticklabels=DAYS,
            cmap="YlOrRd", fmt=".3f", annot=False,
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Доля событий"}
        )
        ax.set_xlabel("Час суток")
        ax.set_ylabel("День недели")
        ax.set_title(title)

    plt.tight_layout()
    out = RESULTS_DIR / "task5_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
