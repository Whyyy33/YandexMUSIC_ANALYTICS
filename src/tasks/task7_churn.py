"""
Task 7 — Прогноз оттока (churn prediction).

Определение оттока: пользователь считается «отточившим», если в последней
трети своего временного окна у него нет активности.

Схема:
  1. Вычисляем ts_seconds для каждого события
  2. Разбиваем историю пользователя на 2/3 (признаки) и 1/3 (метка)
  3. Метка churn = 1, если в последней трети событий нет
  4. Признаки: всего событий, уникальных треков, avg/std played_ratio,
               доля органики, энтропия треков
  5. Модели: LogisticRegression + RandomForestClassifier
  6. Метрики: ROC-AUC, F1, feature importance

Вывод: data/results/task7_churn_roc.png, task7_churn_features.png
"""

import sys
import time
from pathlib import Path

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

from src.config import RESULTS_DIR, SESSION_GAP_MINUTES, TIMESTAMP_UNIT_SECONDS, find_parquet

SESSION_GAP_SEC = SESSION_GAP_MINUTES * 60
MIN_EVENTS = 50          # минимум событий у пользователя
N_SAMPLE_USERS = 100_000  # сэмплируем для скорости


def shannon_entropy(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Строит матрицу признаков по первым 2/3 истории каждого пользователя."""

    # Граница: 2/3 от max ts_seconds пользователя
    user_max = df.group_by("uid").agg(pl.col("ts_seconds").max().alias("ts_max"))
    df = df.join(user_max, on="uid")
    df = df.with_columns((pl.col("ts_max") * 2 / 3).alias("ts_cutoff"))

    train_part = df.filter(pl.col("ts_seconds") <= pl.col("ts_cutoff"))
    future_part = df.filter(pl.col("ts_seconds") > pl.col("ts_cutoff"))

    # Пользователи без событий в будущем = churn
    future_uids = future_part["uid"].unique()

    # Признаки из train_part
    feats = (
        train_part
        .group_by("uid")
        .agg([
            pl.len().alias("n_events"),
            pl.col("item_id").n_unique().alias("n_unique_items"),
            pl.col("is_organic").mean().alias("organic_ratio"),
            pl.col("played_ratio_pct").clip(0, 100).mean().alias("avg_played_ratio"),
            pl.col("played_ratio_pct").clip(0, 100).std().fill_null(0).alias("std_played_ratio"),
            pl.col("track_length_seconds").mean().alias("avg_track_length"),
            pl.col("artist_id").n_unique().alias("n_unique_artists"),
            pl.col("album_id").n_unique().alias("n_unique_albums"),
            (pl.col("item_id").n_unique() / pl.col("item_id").len()).alias("repeat_ratio"),
        ])
        .filter(pl.col("n_events") >= MIN_EVENTS)
    )

    # Метка
    churn_series = (~feats["uid"].is_in(future_uids)).cast(pl.Int8).rename("churn")
    feats = feats.with_columns(churn_series)

    return feats


def run() -> None:
    t0 = time.perf_counter()
    print("Task 7: Прогноз оттока...")

    path = find_parquet("listens")
    artist_map_path = find_parquet("artist_item_mapping")
    album_map_path = find_parquet("album_item_mapping")

    df = (
        pl.scan_parquet(path)
        .select(["uid", "item_id", "timestamp", "is_organic",
                 "played_ratio_pct", "track_length_seconds"])
        .with_columns(
            (pl.col("timestamp").cum_sum().over("uid") * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
        )
        .collect()
    )

    # Сэмплируем пользователей
    uids = df["uid"].unique().sample(min(N_SAMPLE_USERS, df["uid"].n_unique()), seed=42)
    df = df.filter(pl.col("uid").is_in(uids))

    # Обогащаем маппингами artist_id и album_id
    artist_map = pl.read_parquet(artist_map_path)
    album_map = pl.read_parquet(album_map_path)
    df = df.join(artist_map, on="item_id", how="left")
    df = df.join(album_map, on="item_id", how="left")

    feats = build_features(df)
    print(f"  Пользователей: {len(feats):,}, churn rate: {feats['churn'].mean():.1%}")

    feature_cols = ["n_events", "n_unique_items", "organic_ratio",
                    "avg_played_ratio", "std_played_ratio", "avg_track_length",
                    "n_unique_artists", "n_unique_albums", "repeat_ratio"]
    X = feats[feature_cols].to_numpy()
    y = feats["churn"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    lr.fit(X_train_s, y_train)
    rf.fit(X_train, y_train)

    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"  LR ROC-AUC:  {lr_auc:.4f}")
    print(f"  RF ROC-AUC:  {rf_auc:.4f}")

    # --- ROC-кривые ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Прогноз оттока пользователей", fontsize=14, fontweight="bold")

    ax = axes[0]
    RocCurveDisplay.from_estimator(lr, X_test_s, y_test, ax=ax,
                                   name=f"LogReg (AUC={lr_auc:.3f})")
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax,
                                   name=f"RandomForest (AUC={rf_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_title("ROC-кривые")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Feature importance (RF) ---
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
    print(classification_report(y_test, rf.predict(X_test), target_names=["Активен", "Отток"]))
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
