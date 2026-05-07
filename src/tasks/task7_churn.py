"""
Task 7 — Прогноз оттока (churn prediction).

Определение оттока: пользователь считается «отточившим», если в последней
трети своего временного окна у него нет активности.

Сэмплируем uid % UID_MOD = 0 через DuckDB.

Вывод: data/results/task7_churn.png
"""

import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    MAX_USERS_TASK7,
    RAM_SOFT_CAP_GB,
    RESULTS_DIR,
    TIMESTAMP_UNIT_SECONDS,
    find_parquet,
)

MIN_EVENTS = 50
# 10 чанков → ~46M строк на чанк, всё помещается в DuckDB streaming
CHUNKS = 10
# Метка churn: пользователь не вернулся за ПОСЛЕДНИЙ месяц глобального окна
CHURN_TAIL_DAYS = 30


def _check_size(df: pl.DataFrame, name: str) -> None:
    sz_gb = df.estimated_size() / 1e9
    print(f"  [mem] {name}: {df.height:,} строк, {sz_gb:.2f} ГБ")
    if sz_gb > RAM_SOFT_CAP_GB:
        raise MemoryError(
            f"{name} = {sz_gb:.2f} ГБ превышает RAM_SOFT_CAP_GB={RAM_SOFT_CAP_GB} ГБ. "
            f"Уменьши MAX_USERS_TASK7 в config.py или увеличь CHUNKS."
        )


def run() -> None:
    t0 = time.perf_counter()
    print("Task 7: Прогноз оттока...")

    path        = str(find_parquet("listens"))
    artist_path = str(find_parquet("artist_item_mapping"))
    album_path  = str(find_parquet("album_item_mapping"))

    # Глобальный момент времени T_end — общий для всех чанков, чтобы метка churn
    # была согласованной. Считаем один раз перед циклом по чанкам.
    print("  Считаем глобальный ts_max...")
    con0 = duckdb.connect()
    global_ts_max = con0.execute(
        f"SELECT MAX(timestamp) * {TIMESTAMP_UNIT_SECONDS} FROM read_parquet('{path}')"
    ).fetchone()[0]
    con0.close()
    cutoff_ts = global_ts_max - CHURN_TAIL_DAYS * 86400
    cutoff_7d  = cutoff_ts -  7 * 86400
    cutoff_30d = cutoff_ts - 30 * 86400
    print(f"  global ts_max={global_ts_max:,} sec, churn cutoff={cutoff_ts:,} sec "
          f"(последние {CHURN_TAIL_DAYS} дней — метка)")

    feat_chunks = []

    for chunk_id in range(CHUNKS):
        print(f"  Чанк {chunk_id + 1}/{CHUNKS}...", end=" ", flush=True)
        con = duckdb.connect()
        con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
        con.execute("PRAGMA temp_directory='data/processed/.duckdb_tmp'")

        # Шаг 1 — фильтруем listens по uid
        con.execute(f"""
            CREATE TEMP TABLE listens_chunk AS
            SELECT
                uid,
                item_id,
                timestamp * {TIMESTAMP_UNIT_SECONDS}      AS ts_seconds,
                is_organic,
                LEAST(played_ratio_pct, 100)              AS played_ratio,
                track_length_seconds
            FROM read_parquet('{path}')
            WHERE (uid // 10) % {CHUNKS} = {chunk_id}
        """)

        # Шаг 2 — метка churn: пользователь, у которого MAX(ts) < cutoff_ts
        con.execute(f"""
            CREATE TEMP TABLE user_label AS
            SELECT uid,
                   MAX(ts_seconds) AS ts_max,
                   CASE WHEN MAX(ts_seconds) < {cutoff_ts} THEN 1 ELSE 0 END AS churn
            FROM listens_chunk
            GROUP BY uid
        """)

        # Шаг 3 — фичи на событиях ДО cutoff_ts.
        # Recency: последний ts юзера, медиана/95-перц интервалов между соседними событиями,
        # активность за последние 7/30 дней, активность за первые 30 дней (для тренда).
        chunk = con.execute(f"""
            WITH train AS (
                SELECT l.uid,
                       l.item_id,
                       l.is_organic,
                       l.played_ratio,
                       l.track_length_seconds,
                       l.ts_seconds,
                       a.artist_id,
                       al.album_id
                FROM listens_chunk l
                LEFT JOIN read_parquet('{artist_path}') a  ON l.item_id = a.item_id
                LEFT JOIN read_parquet('{album_path}')  al ON l.item_id = al.item_id
                WHERE l.ts_seconds <= {cutoff_ts}
            ),
            with_gap AS (
                SELECT
                    uid,
                    ts_seconds,
                    item_id,
                    is_organic,
                    played_ratio,
                    track_length_seconds,
                    artist_id,
                    album_id,
                    ts_seconds - LAG(ts_seconds) OVER (PARTITION BY uid ORDER BY ts_seconds) AS gap_sec
                FROM train
            ),
            user_first AS (
                SELECT uid, MIN(ts_seconds) AS first_ts FROM train GROUP BY uid
            )
            SELECT
                t.uid,
                COUNT(*)                                    AS n_events,
                COUNT(DISTINCT t.item_id)                   AS n_unique_items,
                AVG(CAST(t.is_organic AS DOUBLE))           AS organic_ratio,
                AVG(t.played_ratio)                         AS avg_played_ratio,
                COALESCE(STDDEV(t.played_ratio), 0)         AS std_played_ratio,
                AVG(t.track_length_seconds)                 AS avg_track_length,
                COUNT(DISTINCT t.artist_id)                 AS n_unique_artists,
                COUNT(DISTINCT t.album_id)                  AS n_unique_albums,
                CAST(COUNT(DISTINCT t.item_id) AS DOUBLE)
                    / NULLIF(COUNT(*), 0)                   AS repeat_ratio,
                -- Recency:
                CAST(({cutoff_ts} - MAX(t.ts_seconds)) AS DOUBLE) / 86400.0
                                                             AS days_since_last_event,
                COALESCE(APPROX_QUANTILE(t.gap_sec, 0.5),  0) AS gap_p50,
                COALESCE(APPROX_QUANTILE(t.gap_sec, 0.95), 0) AS gap_p95,
                SUM(CASE WHEN t.ts_seconds BETWEEN {cutoff_7d}  AND {cutoff_ts} THEN 1 ELSE 0 END) AS events_last_7d,
                SUM(CASE WHEN t.ts_seconds BETWEEN {cutoff_30d} AND {cutoff_ts} THEN 1 ELSE 0 END) AS events_last_30d,
                CAST(SUM(CASE WHEN t.ts_seconds <= uf.first_ts + 30*86400 THEN 1 ELSE 0 END) AS BIGINT)
                                                             AS events_first_30d,
                u.churn
            FROM with_gap t
            JOIN user_label u  USING (uid)
            JOIN user_first uf USING (uid)
            GROUP BY t.uid, u.churn
            HAVING COUNT(*) >= {MIN_EVENTS}
        """).pl()

        con.close()
        feat_chunks.append(chunk)
        print(f"users={len(chunk):,} churn={chunk['churn'].mean():.1%}", flush=True)

    feats_df = pl.concat(feat_chunks)
    _check_size(feats_df, "feats_df")

    # Производные фичи в Polars
    feats_df = feats_df.with_columns(
        (pl.col("events_last_7d").cast(pl.Float64) /
         pl.when(pl.col("n_events") > 0).then(pl.col("n_events")).otherwise(1)
        ).alias("ratio_recent_7d"),
        (pl.col("events_last_30d").cast(pl.Float64) /
         pl.when(pl.col("events_first_30d") > 0).then(pl.col("events_first_30d")).otherwise(1)
        ).alias("trend_ratio"),
        pl.col("repeat_ratio").fill_null(0),
        pl.col("std_played_ratio").fill_null(0),
        pl.col("gap_p50").cast(pl.Float64).fill_null(0),
        pl.col("gap_p95").cast(pl.Float64).fill_null(0),
        pl.col("days_since_last_event").cast(pl.Float64).fill_null(0),
    )

    # Усечение до MAX_USERS_TASK7 — стабилизирует память на debug-прогонах
    if feats_df.height > MAX_USERS_TASK7:
        feats_df = feats_df.sample(n=MAX_USERS_TASK7, seed=42)
        print(f"  Урезаем до MAX_USERS_TASK7={MAX_USERS_TASK7:,}")
    churn_rate = feats_df["churn"].mean()
    print(f"  Итого пользователей: {len(feats_df):,}, churn rate: {churn_rate:.1%}")

    feature_cols = [
        "n_events", "n_unique_items", "organic_ratio",
        "avg_played_ratio", "std_played_ratio", "avg_track_length",
        "n_unique_artists", "n_unique_albums", "repeat_ratio",
        "days_since_last_event", "gap_p50", "gap_p95",
        "events_last_7d", "events_last_30d",
        "ratio_recent_7d", "trend_ratio",
    ]
    feature_labels_ru = {
        "n_events":              "Число событий",
        "n_unique_items":        "Уникальных треков",
        "organic_ratio":         "Доля органики",
        "avg_played_ratio":      "Средняя доля прослушки",
        "std_played_ratio":      "Разброс прослушки",
        "avg_track_length":      "Средняя длина трека",
        "n_unique_artists":      "Уникальных артистов",
        "n_unique_albums":       "Уникальных альбомов",
        "repeat_ratio":          "Доля повторов",
        "days_since_last_event": "Дней с послед. события",
        "gap_p50":               "Медиана интервалов (с)",
        "gap_p95":               "P95 интервалов (с)",
        "events_last_7d":        "События за 7 дней",
        "events_last_30d":       "События за 30 дней",
        "ratio_recent_7d":       "Доля активности за 7д",
        "trend_ratio":           "Тренд 30д / первые 30д",
    }
    volume_cols = ["n_events", "n_unique_items", "n_unique_artists", "n_unique_albums"]

    # Корреляционная матрица фичей
    print(f"\n  Корреляция (Pearson), {len(feature_cols)} фичей:")
    feat_np = feats_df[feature_cols].to_numpy().astype(np.float64)
    corr = np.corrcoef(feat_np, rowvar=False)
    print("\n  Пары |r| > 0.85:")
    high_corr_volume = False
    vol_idx = [feature_cols.index(c) for c in volume_cols]
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if abs(corr[i, j]) > 0.85:
                print(f"    {feature_cols[i]} <-> {feature_cols[j]}: r={corr[i, j]:.3f}")
                if i in vol_idx and j in vol_idx:
                    high_corr_volume = True

    X = feat_np
    y = feats_df["churn"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    print("  Обучаем модели...")
    lr.fit(X_train_s, y_train)
    rf.fit(X_train,   y_train)

    lr_proba = lr.predict_proba(X_test_s)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    rf_auc = roc_auc_score(y_test, rf_proba)

    rf_pred = rf.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, rf_pred, labels=[0, 1])
    print(f"  LR ROC-AUC: {lr_auc:.4f}")
    print(f"  RF ROC-AUC: {rf_auc:.4f}")
    print(f"  RF класс 'отток': precision={p[1]:.3f}, recall={r[1]:.3f}, f1={f[1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    importances = rf.feature_importances_
    idx_full = np.argsort(importances)
    top10_idx = idx_full[-10:]
    top_feat_ru = feature_labels_ru[feature_cols[idx_full[-1]]]

    suptitle_text = (
        "Прогноз оттока пользователей\n"
        f"Churn rate {churn_rate:.1%}, AUC≈{rf_auc:.2f}, "
        f"recall на оттоке={r[1]:.2f}, precision={p[1]:.2f}, F1={f[1]:.2f}, "
        f"ключевая фича — {top_feat_ru}"
    )
    if high_corr_volume:
        suptitle_text += (
            "\nЧасть фичей сильно коррелируют (r>0.85) — отражают общий объём активности"
        )
    fig.suptitle(suptitle_text, fontsize=13, fontweight="bold")

    # ROC
    ax = axes[0]
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax,
                                   name=f"Случайный лес (AUC={rf_auc:.3f})")
    RocCurveDisplay.from_estimator(lr, X_test_s, y_test, ax=ax,
                                   name=f"ЛогРег (AUC={lr_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Случайное угадывание")
    ax.set_xlabel("Доля ложноположительных (отток=1)")
    ax.set_ylabel("Доля верноположительных (отток=1)")
    ax.set_title("ROC-кривая")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Топ-10 feature importance
    ax2 = axes[1]
    ax2.barh(
        [feature_labels_ru[feature_cols[i]] for i in top10_idx],
        importances[top10_idx], color="steelblue", alpha=0.8,
    )
    ax2.set_xlabel("Важность признака (Случайный лес)")
    ax2.set_title("Топ-10 признаков (Случайный лес)")
    ax2.grid(axis="x", alpha=0.3)

    # Корреляционная матрица
    ax3 = axes[2]
    short_labels = [feature_labels_ru[c][:18] for c in feature_cols]
    im = ax3.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(feature_cols)))
    ax3.set_yticks(range(len(feature_cols)))
    ax3.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax3.set_yticklabels(short_labels, fontsize=8)
    # Числа в клетках
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            v = corr[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6, color=color)
    ax3.set_title("Корреляционная матрица фичей")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    fig.text(
        0.5, 0.01,
        "Отток (churn) = пользователь без активности в последние 30 дней датасета. "
        "Recency-фичи: дней с последнего события, медиана/p95 интервалов, активность за 7/30 дней, "
        "тренд (30д vs первые 30д). AUC = площадь под ROC.",
        ha="center", fontsize=8, color="dimgray", wrap=True,
    )
    out = RESULTS_DIR / "task7_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")

    print("\n  Отчёт (RandomForest):")
    print(classification_report(y_test, rf_pred, target_names=["Активен", "Отток"]))
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
