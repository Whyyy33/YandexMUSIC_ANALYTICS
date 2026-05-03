"""
Task 2 — Распределение длин сессий.

Сессия = последовательность событий одного пользователя с паузами < 30 мин.
Смотрим: события на сессию, длительность (сек), сравниваем органику vs рекомендации.

Вывод: data/results/task2_sessions.png
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, SESSION_GAP_MINUTES, TIMESTAMP_UNIT_SECONDS, find_parquet

SESSION_GAP_SEC = SESSION_GAP_MINUTES * 60
MAX_SESSION_EVENTS = 200  # хвост обрезаем для читаемости гистограммы


def run() -> None:
    t0 = time.perf_counter()
    print("Task 2: Распределение сессий...")

    path = find_parquet("listens")

    lf = (
        pl.scan_parquet(path)
        .select(["uid", "timestamp", "is_organic"])
        .with_columns(
            (pl.col("timestamp").cum_sum().over("uid") * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
        )
    )

    # Определяем начало новой сессии по паузе > SESSION_GAP_SEC
    lf = lf.with_columns(
        (
            (pl.col("ts_seconds") - pl.col("ts_seconds").shift(1).over("uid")).fill_null(0)
            > SESSION_GAP_SEC
        )
        .cast(pl.Int32)
        .alias("new_sess")
    ).with_columns(
        pl.col("new_sess").cum_sum().over("uid").alias("session_id")
    )

    # Агрегируем по сессии
    sessions = (
        lf.group_by(["uid", "session_id"])
        .agg(
            pl.len().alias("n_events"),
            (pl.col("ts_seconds").max() - pl.col("ts_seconds").min()).alias("duration_sec"),
            pl.col("is_organic").mean().alias("organic_ratio"),
        )
        .with_columns(
            (pl.col("organic_ratio") >= 0.5).alias("mostly_organic")
        )
        .collect()
    )

    organic_sess = sessions.filter(pl.col("mostly_organic"))
    reco_sess = sessions.filter(~pl.col("mostly_organic"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Распределение длин сессий", fontsize=14, fontweight="bold")

    for ax, col, xlabel, title in [
        (axes[0], "n_events", "Событий в сессии", "Событий на сессию"),
        (axes[1], "duration_sec", "Длительность (сек)", "Длительность сессии"),
    ]:
        data_o = organic_sess[col].filter(organic_sess[col] <= organic_sess[col].quantile(0.99))
        data_r = reco_sess[col].filter(reco_sess[col] <= reco_sess[col].quantile(0.99))

        ax.hist(data_o.to_numpy(), bins=50, alpha=0.6, label=f"Органика (med={data_o.median():.0f})", density=True)
        ax.hist(data_r.to_numpy(), bins=50, alpha=0.6, label=f"Рекомендации (med={data_r.median():.0f})", density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Доля сессий")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "task2_sessions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
