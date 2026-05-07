"""
Task 5 — Тепловая карта активности.

Три среза:
  (а) Глобальный цикл: ts_seconds mod 1 неделя — час × день относительно точки
      отсчёта датасета. Показывает есть ли макро-паттерн в анонимизированном времени.
  (б) Жизненный цикл пользователя: (ts - first_event_user) — сутки×час с момента
      первого события пользователя. Здесь могут быть видны реальные паттерны
      онбординга / удержания.
  (в) Внимательность прослушки: средний played_ratio_pct по день×час
      (RdYlGn, красное = плохо слушают, зелёное = хорошо).

Вывод: data/results/task5_heatmap.png
"""

import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    RAM_SOFT_CAP_GB,
    RESULTS_DIR,
    TIMESTAMP_UNIT_SECONDS,
    find_parquet,
)


def _check_size(df: pl.DataFrame, name: str) -> None:
    sz_gb = df.estimated_size() / 1e9
    print(f"  [mem] {name}: {df.height:,} строк, {sz_gb:.2f} ГБ")
    if sz_gb > RAM_SOFT_CAP_GB:
        raise MemoryError(
            f"{name} = {sz_gb:.2f} ГБ превышает RAM_SOFT_CAP_GB={RAM_SOFT_CAP_GB} ГБ."
        )

SECONDS_PER_WEEK = 7 * 24 * 3600
SECONDS_PER_DAY  = 24 * 3600

DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def compute_heatmap(df: pl.DataFrame, rows: int, cols: int,
                    row_col: str, col_col: str, value_col: str = "events") -> np.ndarray:
    mat = np.zeros((rows, cols), dtype=np.float64)
    for r, c, v in df.select([row_col, col_col, value_col]).iter_rows():
        if 0 <= r < rows and 0 <= c < cols:
            mat[int(r), int(c)] = v
    return mat


def run() -> None:
    t0 = time.perf_counter()
    print("Task 5: Тепловая карта активности...")

    path = str(find_parquet("listens")).replace("\\", "/")

    # --- (а) + (в) Глобальный цикл недели + средний played_ratio ---
    df_global = (
        pl.scan_parquet(path)
        .select(["uid", "timestamp", "is_organic", "played_ratio_pct"])
        .with_columns(
            (pl.col("timestamp").cast(pl.Int64) * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
        )
        .with_columns(
            ((pl.col("ts_seconds") % SECONDS_PER_WEEK) // SECONDS_PER_DAY).cast(pl.Int8).alias("day_of_week"),
            ((pl.col("ts_seconds") % SECONDS_PER_DAY) // 3600).cast(pl.Int8).alias("hour"),
        )
        .group_by(["is_organic", "day_of_week", "hour"])
        .agg(
            pl.len().alias("events"),
            pl.col("played_ratio_pct").mean().alias("avg_played_ratio"),
        )
        .collect(engine="streaming")
    )
    _check_size(df_global, "df_global")

    mat_g_o = compute_heatmap(
        df_global.filter(pl.col("is_organic") == 1), 7, 24, "day_of_week", "hour"
    )
    mat_g_r = compute_heatmap(
        df_global.filter(pl.col("is_organic") == 0), 7, 24, "day_of_week", "hour"
    )
    mat_g_o_norm = mat_g_o / mat_g_o.sum(axis=1, keepdims=True)
    mat_g_r_norm = mat_g_r / mat_g_r.sum(axis=1, keepdims=True)

    mat_pr_o = compute_heatmap(
        df_global.filter(pl.col("is_organic") == 1), 7, 24, "day_of_week", "hour",
        value_col="avg_played_ratio",
    )
    mat_pr_r = compute_heatmap(
        df_global.filter(pl.col("is_organic") == 0), 7, 24, "day_of_week", "hour",
        value_col="avg_played_ratio",
    )

    spread_g_o = mat_g_o_norm.max() - mat_g_o_norm.min()
    spread_g_r = mat_g_r_norm.max() - mat_g_r_norm.min()
    print(f"  (а) глобальный цикл: размах органика={spread_g_o*100:.2f}пп, реко={spread_g_r*100:.2f}пп")

    # --- (б) Жизненный цикл пользователя (sec от первого события) ---
    print(f"  Считаем offset от первого события (DuckDB memory_limit={DUCKDB_MEMORY_LIMIT})...")
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute("PRAGMA temp_directory='data/processed/.duckdb_tmp'")
    df_life = con.execute(f"""
        WITH base AS (
            SELECT
                uid,
                is_organic,
                timestamp * {TIMESTAMP_UNIT_SECONDS} AS ts_seconds
            FROM read_parquet('{path}')
        ),
        with_off AS (
            SELECT
                is_organic,
                ts_seconds - MIN(ts_seconds) OVER (PARTITION BY uid) AS ts_off
            FROM base
        ),
        bucketed AS (
            SELECT
                is_organic,
                CAST(ts_off /  {SECONDS_PER_DAY}              AS INTEGER) AS day_idx,
                CAST((ts_off % {SECONDS_PER_DAY}) / 3600       AS INTEGER) AS hour
            FROM with_off
            WHERE ts_off < 14 * {SECONDS_PER_DAY}
        )
        SELECT is_organic, day_idx, hour, COUNT(*) AS events
        FROM bucketed
        GROUP BY is_organic, day_idx, hour
    """).pl()
    con.close()
    _check_size(df_life, "df_life")

    mat_l_o = compute_heatmap(
        df_life.filter(pl.col("is_organic") == 1), 14, 24, "day_idx", "hour"
    )
    mat_l_r = compute_heatmap(
        df_life.filter(pl.col("is_organic") == 0), 14, 24, "day_idx", "hour"
    )
    mat_l_o_norm = mat_l_o / mat_l_o.sum(axis=1, keepdims=True)
    mat_l_r_norm = mat_l_r / mat_l_r.sum(axis=1, keepdims=True)

    spread_l_o = mat_l_o_norm.max() - mat_l_o_norm.min()
    print(f"  (б) первые 14 дней юзера: размах органика={spread_l_o*100:.2f}пп")

    # --- Вердикт по внимательности: средний played_ratio ночью (0-6) vs днём (10-20) ---
    night_o = mat_pr_o[:, 0:7].mean()
    day_o   = mat_pr_o[:, 10:21].mean()
    night_r = mat_pr_r[:, 0:7].mean()
    day_r   = mat_pr_r[:, 10:21].mean()
    avg_o   = mat_pr_o.mean()
    avg_r   = mat_pr_r.mean()
    delta = ((night_o - day_o) + (night_r - day_r)) / 2
    if delta > 2:
        attention_verdict = "растёт ночью"
    elif delta < -2:
        attention_verdict = "растёт днём"
    else:
        attention_verdict = "равномерна"
    print(f"  (в) внимательность: ночь_o={night_o:.1f}%, день_o={day_o:.1f}% | "
          f"ночь_r={night_r:.1f}%, день_r={day_r:.1f}%, вердикт={attention_verdict}")

    # --- Рисуем 3×2: (а) глобальный, (б) жизненный, (в) played_ratio ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(
        "Активность по часам и дням\n"
        f"(а) Глобальный цикл анонимизирован — размах ~{max(spread_g_o, spread_g_r)*100:.1f}пп, "
        f"паттерн почти равномерен.  (б) В первые 2 недели юзера видна жизненная динамика.\n"
        f"Внимательность {attention_verdict} по часам. "
        f"Реко дослушивают на {avg_r-avg_o:.0f}пп лучше: {avg_r:.0f}% vs {avg_o:.0f}%",
        fontsize=13, fontweight="bold",
    )

    # (а) Общая шкала
    g_vmin = min(mat_g_o_norm.min(), mat_g_r_norm.min())
    g_vmax = max(mat_g_o_norm.max(), mat_g_r_norm.max())
    for ax, mat, title in [
        (axes[0, 0], mat_g_o_norm, "(а) Органика — цикл недели"),
        (axes[0, 1], mat_g_r_norm, "(а) Рекомендации — цикл недели"),
    ]:
        sns.heatmap(
            mat, ax=ax, vmin=g_vmin, vmax=g_vmax,
            xticklabels=range(24), yticklabels=DAYS,
            cmap="YlOrRd", fmt=".3f", annot=True, annot_kws={"size": 6},
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Доля событий в дне"},
        )
        ax.set_xlabel("Час (сдвиг от 00:00 датасета)")
        ax.set_ylabel("День (сдвиг от начала датасета)")
        ax.set_title(title)

    # (б) Жизненный цикл — общая шкала
    l_vmin = min(mat_l_o_norm.min(), mat_l_r_norm.min())
    l_vmax = max(mat_l_o_norm.max(), mat_l_r_norm.max())
    day_labels = [f"+{i}д" for i in range(14)]
    for ax, mat, title in [
        (axes[1, 0], mat_l_o_norm, "(б) Органика — первые 14 дней юзера"),
        (axes[1, 1], mat_l_r_norm, "(б) Рекомендации — первые 14 дней юзера"),
    ]:
        sns.heatmap(
            mat, ax=ax, vmin=l_vmin, vmax=l_vmax,
            xticklabels=range(24), yticklabels=day_labels,
            cmap="YlOrRd", annot=False,
            linewidths=0.2, linecolor="white",
            cbar_kws={"label": "Доля событий в сутках"},
        )
        ax.set_xlabel("Час (от полуночи дня)")
        ax.set_ylabel("Сутки от первого события юзера")
        ax.set_title(title)

    # (в) Played ratio heatmap — общая шкала, RdYlGn
    pr_vmin = min(mat_pr_o.min(), mat_pr_r.min())
    pr_vmax = max(mat_pr_o.max(), mat_pr_r.max())
    for ax, mat, title in [
        (axes[2, 0], mat_pr_o, "(в) Внимательность ~равномерна по часам — органика"),
        (axes[2, 1], mat_pr_r, "(в) Внимательность ~равномерна по часам — рекомендации"),
    ]:
        sns.heatmap(
            mat, ax=ax, vmin=pr_vmin, vmax=pr_vmax,
            xticklabels=range(24), yticklabels=DAYS,
            cmap="RdYlGn", fmt=".0f", annot=True, annot_kws={"size": 6},
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Средний played_ratio, %"},
        )
        ax.set_xlabel("Час (сдвиг от 00:00 датасета)")
        ax.set_ylabel("День (сдвиг от начала датасета)")
        ax.set_title(title)

    plt.tight_layout()
    out = RESULTS_DIR / "task5_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
