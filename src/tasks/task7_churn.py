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
from sklearn.metrics import RocCurveDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, TIMESTAMP_UNIT_SECONDS, find_parquet

MIN_EVENTS = 50
# 10 чанков → ~46M строк на чанк, всё помещается в DuckDB streaming
CHUNKS = 10
# Метка churn: пользователь не вернулся за ПОСЛЕДНИЙ месяц глобального окна
CHURN_TAIL_DAYS = 30


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
    print(f"  global ts_max={global_ts_max:,} sec, churn cutoff={cutoff_ts:,} sec "
          f"(последние {CHURN_TAIL_DAYS} дней — метка)")

    feat_chunks = []

    for chunk_id in range(CHUNKS):
        print(f"  Чанк {chunk_id + 1}/{CHUNKS}...", end=" ", flush=True)
        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='6GB'")
        con.execute("PRAGMA threads=4")

        # Шаг 1 — фильтруем listens по uid, материализуем БЕЗ маппингов
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

        # Шаг 3 — присоединяем маппинги ОДИН раз, фичи строятся на событиях ДО cutoff
        chunk = con.execute(f"""
            WITH train AS (
                SELECT l.uid,
                       l.item_id,
                       l.is_organic,
                       l.played_ratio,
                       l.track_length_seconds,
                       a.artist_id,
                       al.album_id
                FROM listens_chunk l
                LEFT JOIN read_parquet('{artist_path}') a  ON l.item_id = a.item_id
                LEFT JOIN read_parquet('{album_path}')  al ON l.item_id = al.item_id
                WHERE l.ts_seconds <= {cutoff_ts}
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
                u.churn
            FROM train t
            JOIN user_label u USING (uid)
            GROUP BY t.uid, u.churn
            HAVING COUNT(*) >= {MIN_EVENTS}
        """).pl()

        con.close()
        feat_chunks.append(chunk)
        print(f"users={len(chunk):,} churn={chunk['churn'].mean():.1%}", flush=True)

    feats_df = pl.concat(feat_chunks)
    print(f"  Итого пользователей: {len(feats_df):,}, churn rate: {feats_df['churn'].mean():.1%}")

    feature_cols = ["n_events", "n_unique_items", "organic_ratio",
                    "avg_played_ratio", "std_played_ratio", "avg_track_length",
                    "n_unique_artists", "n_unique_albums", "repeat_ratio"]

    feats_df = feats_df.with_columns(
        pl.col("repeat_ratio").fill_null(0),
        pl.col("std_played_ratio").fill_null(0),
    )

    X = feats_df[feature_cols].to_numpy()
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

    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"  LR ROC-AUC: {lr_auc:.4f}")
    print(f"  RF ROC-AUC: {rf_auc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Прогноз оттока пользователей", fontsize=14, fontweight="bold")

    ax = axes[0]
    RocCurveDisplay.from_estimator(lr, X_test_s, y_test, ax=ax, name="LogReg")
    RocCurveDisplay.from_estimator(rf, X_test,   y_test, ax=ax, name="RandomForest")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_title("ROC-кривые")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    importances = rf.feature_importances_
    idx = np.argsort(importances)
    ax2.barh([feature_cols[i] for i in idx], importances[idx], color="steelblue", alpha=0.8)
    ax2.set_xlabel("Важность признака (RF)")
    ax2.set_title("Важность признаков (RandomForest)")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "task7_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")

    print("\n  Отчёт (RandomForest):")
    print(classification_report(y_test, rf.predict(X_test),
                                 target_names=["Активен", "Отток"]))
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
