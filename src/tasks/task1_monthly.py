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

import duckdb
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, TIMESTAMP_UNIT_SECONDS, find_parquet
from src.rank_labels import get_artist_labels

SECONDS_PER_MONTH = 30 * 24 * 3600
SKIP_THRESHOLD = 30
FULL_THRESHOLD = 80


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

    # Скип-рейт и полные прослушивания по месяцам — через DuckDB чтобы стримом считать
    print("  Скип-рейт по месяцам (DuckDB)...")
    con = duckdb.connect()
    quality_df = con.execute(f"""
        SELECT
            CAST((CAST(timestamp AS BIGINT) * {TIMESTAMP_UNIT_SECONDS}) / {SECONDS_PER_MONTH} AS INTEGER) AS month_offset,
            is_organic,
            CAST(COUNT(*) AS BIGINT)                                                                AS n_events,
            CAST(SUM(CASE WHEN played_ratio_pct < {SKIP_THRESHOLD} THEN 1 ELSE 0 END) AS BIGINT)     AS n_skips,
            CAST(SUM(CASE WHEN played_ratio_pct >= {FULL_THRESHOLD} THEN 1 ELSE 0 END) AS BIGINT)    AS n_full
        FROM read_parquet('{str(path).replace(chr(92), '/')}')
        WHERE played_ratio_pct IS NOT NULL
        GROUP BY month_offset, is_organic
        ORDER BY month_offset, is_organic
    """).pl()
    con.close()
    quality_df = quality_df.with_columns(
        (pl.col("n_skips").cast(pl.Float64) / pl.col("n_events") * 100).alias("skip_pct"),
        (pl.col("n_full").cast(pl.Float64)  / pl.col("n_events") * 100).alias("full_pct"),
    ).filter(pl.col("month_offset").is_between(0, 23))

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

    organic_q = quality_df.filter(pl.col("is_organic") == 1).sort("month_offset")
    reco_q    = quality_df.filter(pl.col("is_organic") == 0).sort("month_offset")

    # Подмешиваем глобальные rank-метки артистов (A1, A2, ... по полной популярности)
    artist_labels = get_artist_labels().select(["artist_id", "label"])
    df_artists = df_artists.join(artist_labels, on="artist_id", how="left")

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

    # Метрики для main takeaway
    total_events_first = int(df_monthly.filter(pl.col("month_offset") == 0)["events"].sum())
    total_events_last  = int(df_monthly.filter(pl.col("month_offset") == 23)["events"].sum())
    growth_pct = (total_events_last / total_events_first - 1) * 100 if total_events_first else 0
    reco_share_last = (
        reco_m.filter(pl.col("month_offset") == 23)["events"].sum() /
        max(1, df_monthly.filter(pl.col("month_offset") == 23)["events"].sum())
    )

    last_org_skip = float(organic_q.filter(pl.col("month_offset") == 23)["skip_pct"][0]) if len(organic_q.filter(pl.col("month_offset") == 23)) else float("nan")
    last_rec_skip = float(reco_q.filter(pl.col("month_offset") == 23)["skip_pct"][0]) if len(reco_q.filter(pl.col("month_offset") == 23)) else float("nan")

    # --- Графики 3×2 ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        "Динамика прослушиваний (месячные срезы)\n"
        f"За 24 месяца: {total_events_first/1e6:.1f}M → {total_events_last/1e6:.1f}M событий "
        f"(+{growth_pct:.0f}%), доля рекомендаций в финальном месяце {reco_share_last:.0%}\n"
        f"Скип-рейт в финальном месяце: реко {last_rec_skip:.1f}% vs органика {last_org_skip:.1f}%",
        fontsize=13, fontweight="bold",
    )

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

    # Единая шкала X для двух bar-чартов — берём общий максимум
    bar_xmax = max(top_organic["listens"].max() or 0, top_reco["listens"].max() or 0) * 1.05

    # (1,0) Топ-15 артистов — органика
    ax = axes[1, 0]
    labels = [
        f"{lab} ({n/1e6:.2f}M)" if lab is not None else f"#{aid} ({n/1e6:.2f}M)"
        for aid, lab, n in zip(
            top_organic["artist_id"].to_list(),
            top_organic["label"].to_list(),
            top_organic["listens"].to_list(),
        )
    ]
    ax.barh(labels, top_organic["listens"].to_list(), color="steelblue")
    ax.set_xlabel("Прослушиваний")
    ax.set_xlim(0, bar_xmax)
    ax.set_title("Топ-15 артистов — органика (метки A_k — глобальный ранг)")
    ax.grid(axis="x", alpha=0.3)

    # (1,1) Топ-15 артистов — рекомендации
    ax = axes[1, 1]
    labels = [
        f"{lab} ({n/1e6:.2f}M)" if lab is not None else f"#{aid} ({n/1e6:.2f}M)"
        for aid, lab, n in zip(
            top_reco["artist_id"].to_list(),
            top_reco["label"].to_list(),
            top_reco["listens"].to_list(),
        )
    ]
    ax.barh(labels, top_reco["listens"].to_list(), color="tomato")
    ax.set_xlabel("Прослушиваний")
    ax.set_xlim(0, bar_xmax)
    ax.set_title("Топ-15 артистов — рекомендации (метки A_k — глобальный ранг)")
    ax.grid(axis="x", alpha=0.3)

    # (2,0) Скип-рейт по месяцам
    ax = axes[2, 0]
    ax.plot(organic_q["month_offset"], organic_q["skip_pct"], marker="o", ms=4,
            color="steelblue", label="Органика")
    ax.plot(reco_q["month_offset"], reco_q["skip_pct"], marker="s", ms=4,
            color="tomato", label="Рекомендации")
    ax.set_xlabel("Месяц с начала наблюдений")
    ax.set_ylabel(f"% событий со скипом (played_ratio < {SKIP_THRESHOLD}%)")
    ax.set_title("Скип-рейт по месяцам")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2,1) Полные прослушивания по месяцам
    ax = axes[2, 1]
    ax.plot(organic_q["month_offset"], organic_q["full_pct"], marker="o", ms=4,
            color="steelblue", label="Органика")
    ax.plot(reco_q["month_offset"], reco_q["full_pct"], marker="s", ms=4,
            color="tomato", label="Рекомендации")
    ax.set_xlabel("Месяц с начала наблюдений")
    ax.set_ylabel(f"% событий с полной прослушкой (played_ratio ≥ {FULL_THRESHOLD}%)")
    ax.set_title("Полные прослушивания по месяцам")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.07)
    fig.text(
        0.5, 0.01,
        "Метки A_k обозначают артистов по их глобальному рангу популярности "
        "(A1 = самый популярный артист в датасете). "
        "Имена артистов не раскрываются: датасет Yambda анонимизирован Яндексом.",
        ha="center", fontsize=8, color="dimgray", wrap=True,
    )
    out = RESULTS_DIR / "task1_monthly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
