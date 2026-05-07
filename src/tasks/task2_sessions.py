"""
Task 2 — Распределение длин сессий и качество сессий.

Сессия = последовательность событий одного пользователя с паузами < 30 мин.
Смотрим: события на сессию, длительность (сек), медианный played_ratio
(вдумчиво слушают или фоном), сравниваем органику vs рекомендации.

Используем DuckDB — он читает parquet чанками и не грузит 466M строк в RAM целиком.
Сэмплируем пользователей через uid % N, чтобы избежать отдельного scan для списка uid.

Вывод: data/results/task2_sessions.png
"""

import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, SESSION_GAP_MINUTES, TIMESTAMP_UNIT_SECONDS, find_parquet

SESSION_GAP_SEC = SESSION_GAP_MINUTES * 60
# Все uid в Yambda кратны 10 → uid % N даёт перекошенное распределение.
# Делим на 10 перед взятием остатка: (uid//10) % UID_MOD равномерно по 10 бакетам.
UID_MOD = 10


def run() -> None:
    t0 = time.perf_counter()
    print("Task 2: Распределение сессий...")

    path = str(find_parquet("listens"))
    con = duckdb.connect()

    query = f"""
    WITH base AS (
        SELECT
            uid,
            timestamp * {TIMESTAMP_UNIT_SECONDS} AS ts_seconds,
            is_organic,
            played_ratio_pct
        FROM read_parquet('{path}')
        WHERE (uid // 10) % {UID_MOD} = 0
    ),
    with_gap AS (
        SELECT
            uid,
            ts_seconds,
            is_organic,
            played_ratio_pct,
            CASE
                WHEN ts_seconds - LAG(ts_seconds, 1, ts_seconds)
                     OVER (PARTITION BY uid ORDER BY ts_seconds) > {SESSION_GAP_SEC}
                THEN 1 ELSE 0
            END AS new_sess
        FROM base
    ),
    with_session AS (
        SELECT
            uid,
            ts_seconds,
            is_organic,
            played_ratio_pct,
            SUM(new_sess) OVER (PARTITION BY uid ORDER BY ts_seconds) AS session_id
        FROM with_gap
    )
    SELECT
        uid,
        session_id,
        COUNT(*)                                     AS n_events,
        MAX(ts_seconds) - MIN(ts_seconds)            AS duration_sec,
        AVG(CAST(is_organic AS DOUBLE))              AS organic_ratio,
        MEDIAN(CAST(played_ratio_pct AS DOUBLE))     AS median_played_ratio
    FROM with_session
    GROUP BY uid, session_id
    """

    print(f"  Читаем каждого {UID_MOD}-го пользователя (uid % {UID_MOD} = 0)...")
    result = con.execute(query).pl()
    con.close()

    sessions = result.with_columns(
        (pl.col("organic_ratio") >= 0.5).alias("mostly_organic")
    )
    print(f"  Сессий: {len(sessions):,}")

    organic_sess = sessions.filter(pl.col("mostly_organic"))
    reco_sess    = sessions.filter(~pl.col("mostly_organic"))

    # Главные числа для подзаголовка
    med_n_o = organic_sess["n_events"].median()
    med_n_r = reco_sess["n_events"].median()
    med_d_o = organic_sess["duration_sec"].median()
    med_d_r = reco_sess["duration_sec"].median()
    med_pr_o = organic_sess["median_played_ratio"].median()
    med_pr_r = reco_sess["median_played_ratio"].median()
    full_share_o = float(
        organic_sess.filter(pl.col("median_played_ratio") >= 80).height
        / max(1, organic_sess.height)
    ) * 100
    full_share_r = float(
        reco_sess.filter(pl.col("median_played_ratio") >= 80).height
        / max(1, reco_sess.height)
    ) * 100

    # "Длинные сессии вдумчивее или фоновее коротких?"
    # Делим внутри каждой группы по медиане n_events, сравниваем средний played_ratio
    def long_vs_short(df: pl.DataFrame) -> tuple[float, float]:
        cutoff = df["n_events"].median()
        long_pr  = df.filter(pl.col("n_events") >  cutoff)["median_played_ratio"].mean()
        short_pr = df.filter(pl.col("n_events") <= cutoff)["median_played_ratio"].mean()
        return float(short_pr or 0), float(long_pr or 0)

    short_pr_o, long_pr_o = long_vs_short(organic_sess)
    short_pr_r, long_pr_r = long_vs_short(reco_sess)
    # Среднее по двум группам
    delta_long_short = ((long_pr_o - short_pr_o) + (long_pr_r - short_pr_r)) / 2
    if delta_long_short > 1.5:
        long_verdict = "вдумчивее"
    elif delta_long_short < -1.5:
        long_verdict = "фоновее"
    else:
        long_verdict = "сопоставимы по вдумчивости с"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Распределение длин и качество сессий\n"
        f"Сессии с рекомендациями длиннее: med {med_d_r:.0f} vs {med_d_o:.0f} сек "
        f"(×{med_d_r/max(1,med_d_o):.1f}), событий med {med_n_r:.0f} vs {med_n_o:.0f}\n"
        f"Доля сессий с полной прослушкой (med ≥ 80%): "
        f"органика {full_share_o:.0f}% vs реко {full_share_r:.0f}%. "
        f"Бимодальное распределение: либо скип, либо дослушка.\n"
        f"Длинные сессии {long_verdict} коротких (Δ={delta_long_short:+.1f} п.п.)",
        fontsize=13, fontweight="bold",
    )

    # (0,0) События на сессию (обрезка по 99-му перцентилю)
    ax = axes[0, 0]
    q99_o = organic_sess["n_events"].quantile(0.99)
    q99_r = reco_sess["n_events"].quantile(0.99)
    data_o = organic_sess["n_events"].filter(organic_sess["n_events"] <= q99_o)
    data_r = reco_sess["n_events"].filter(reco_sess["n_events"] <= q99_r)
    ax.hist(data_o.to_numpy(), bins=50, alpha=0.6,
            label=f"Органика (med={med_n_o:.0f})", density=True, color="steelblue")
    ax.hist(data_r.to_numpy(), bins=50, alpha=0.6,
            label=f"Рекомендации (med={med_n_r:.0f})", density=True, color="tomato")
    ax.set_xlabel("Событий в сессии")
    ax.set_ylabel("Доля сессий")
    ax.set_title("Событий на сессию (обрезано по 99-му перцентилю)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,1) Длительность с лог-шкалой по X (сохраняет хвост)
    ax = axes[0, 1]
    dur_o = organic_sess["duration_sec"].filter(organic_sess["duration_sec"] > 0).to_numpy()
    dur_r = reco_sess["duration_sec"].filter(reco_sess["duration_sec"] > 0).to_numpy()
    bins = np.logspace(0, np.log10(max(dur_o.max(), dur_r.max())), 60)
    ax.hist(dur_o, bins=bins, alpha=0.6,
            label=f"Органика (med={med_d_o:.0f}с)", density=True, color="steelblue")
    ax.hist(dur_r, bins=bins, alpha=0.6,
            label=f"Рекомендации (med={med_d_r:.0f}с)", density=True, color="tomato")
    ax.set_xscale("log")
    ax.set_xlabel("Длительность сессии (сек, log)")
    ax.set_ylabel("Доля сессий")
    ax.set_title("Длительность сессии (лог-шкала по X, весь хвост)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # (1,0) Распределение медианного played_ratio внутри сессии
    ax = axes[1, 0]
    pr_o = organic_sess["median_played_ratio"].drop_nulls().to_numpy()
    pr_r = reco_sess["median_played_ratio"].drop_nulls().to_numpy()
    bins_pr = np.linspace(0, 110, 56)
    ax.hist(pr_o, bins=bins_pr, alpha=0.6,
            label=f"Органика (med={med_pr_o:.0f}%)", density=True, color="steelblue")
    ax.hist(pr_r, bins=bins_pr, alpha=0.6,
            label=f"Рекомендации (med={med_pr_r:.0f}%)", density=True, color="tomato")
    ax.axvline(30, color="dimgray", ls="--", lw=1, alpha=0.6)
    ax.axvline(80, color="dimgray", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Медианный played_ratio внутри сессии, %")
    ax.set_ylabel("Доля сессий")
    ax.set_title("Качество сессии: фоновое скипание vs внимательная прослушка")
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,1) Hexbin: длина сессии (события) vs медианный played_ratio
    ax = axes[1, 1]
    # Обрезаем хвост по событиям (99-й перцентиль) чтобы не растягивать ось
    q99_all = float(sessions["n_events"].quantile(0.99))
    sub = sessions.filter(
        (pl.col("n_events") <= q99_all) &
        pl.col("median_played_ratio").is_not_null()
    )
    # Чтобы цвета по типу сессии работали — рисуем два полупрозрачных hexbin
    sub_o = sub.filter(pl.col("mostly_organic"))
    sub_r = sub.filter(~pl.col("mostly_organic"))
    extent = (1, q99_all, 0, 110)
    hb_o = ax.hexbin(
        sub_o["n_events"].to_numpy(), sub_o["median_played_ratio"].to_numpy(),
        gridsize=40, extent=extent, cmap="Blues", mincnt=1, alpha=0.7,
        bins="log",
    )
    hb_r = ax.hexbin(
        sub_r["n_events"].to_numpy(), sub_r["median_played_ratio"].to_numpy(),
        gridsize=40, extent=extent, cmap="Reds", mincnt=1, alpha=0.55,
        bins="log",
    )
    ax.set_xlabel("Событий в сессии (обрезано по 99-му перцентилю)")
    ax.set_ylabel("Медианный played_ratio внутри сессии, %")
    ax.set_title("Длина сессии vs внимательность (синий=органика, красный=реко)")
    ax.grid(alpha=0.3)
    cb = fig.colorbar(hb_o, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Сессий (log, органика)")

    plt.tight_layout()
    out = RESULTS_DIR / "task2_sessions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
