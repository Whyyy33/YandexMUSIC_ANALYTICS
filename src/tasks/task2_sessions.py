"""
Task 2 — Распределение длин сессий.

Сессия = последовательность событий одного пользователя с паузами < 30 мин.
Смотрим: события на сессию, длительность (сек), сравниваем органику vs рекомендации.

Используем DuckDB — он читает parquet чанками и не грузит 466M строк в RAM целиком.
Сэмплируем пользователей через uid % N, чтобы избежать отдельного scan для списка uid.

Вывод: data/results/task2_sessions.png
"""

import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import polars as pl

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
            is_organic
        FROM read_parquet('{path}')
        WHERE (uid // 10) % {UID_MOD} = 0
    ),
    with_gap AS (
        SELECT
            uid,
            ts_seconds,
            is_organic,
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
            SUM(new_sess) OVER (PARTITION BY uid ORDER BY ts_seconds) AS session_id
        FROM with_gap
    )
    SELECT
        uid,
        session_id,
        COUNT(*)                              AS n_events,
        MAX(ts_seconds) - MIN(ts_seconds)     AS duration_sec,
        AVG(CAST(is_organic AS DOUBLE))       AS organic_ratio
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Распределение длин сессий", fontsize=14, fontweight="bold")

    for ax, col, xlabel, title in [
        (axes[0], "n_events",     "Событий в сессии",  "Событий на сессию"),
        (axes[1], "duration_sec", "Длительность (сек)", "Длительность сессии"),
    ]:
        q99_o = organic_sess[col].quantile(0.99)
        q99_r = reco_sess[col].quantile(0.99)
        data_o = organic_sess[col].filter(organic_sess[col] <= q99_o)
        data_r = reco_sess[col].filter(reco_sess[col] <= q99_r)

        ax.hist(data_o.to_numpy(), bins=50, alpha=0.6,
                label=f"Органика (med={data_o.median():.0f})", density=True)
        ax.hist(data_r.to_numpy(), bins=50, alpha=0.6,
                label=f"Рекомендации (med={data_r.median():.0f})", density=True)
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
