"""
Task 4 — Разнообразие слушания: Shannon entropy.

Для каждого пользователя считаем энтропию двумя способами:
    - по item_id:   H = -sum(p_i * log2(p_i))
    - по artist_id: та же формула, группировка — артист

Читаем данные чанками по uid % CHUNKS чтобы не грузить 466M строк в RAM.
Энтропию считаем ВНУТРИ цикла на каждом чанке — наружу выходит только массив
значений per-uid (по float64 на пользователя), большие промежуточные таблицы
не накапливаются.

Вывод: data/results/task4_diversity.png
"""

import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, find_parquet

MIN_EVENTS_PER_USER = 10
CHUNKS = 20  # читаем 1/20 за раз → ~23M строк на чанк → ~1-2 ГБ RAM


def _entropy_per_user(counts: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Shannon entropy по пользователям из (uid, <item|artist>, is_organic, cnt).
       Возвращает (h_organic, h_reco)."""
    results = []
    for is_org in [1, 0]:
        sub = counts.filter(pl.col("is_organic") == is_org)
        if sub.is_empty():
            results.append(np.array([], dtype=np.float64))
            continue
        user_totals = sub.group_by("uid").agg(pl.col("cnt").sum().alias("total"))
        merged = sub.join(user_totals, on="uid").filter(pl.col("total") >= MIN_EVENTS_PER_USER)
        h = (
            merged
            .with_columns((pl.col("cnt").cast(pl.Float64) / pl.col("total")).alias("p"))
            .with_columns((-pl.col("p") * pl.col("p").log(base=2)).alias("hc"))
            .group_by("uid")
            .agg(pl.col("hc").sum().alias("entropy"))
        )["entropy"].to_numpy()
        results.append(h)
    return results[0], results[1]


def _plot_entropy_row(axes_row, h_organic, h_reco, label):
    ax = axes_row[0]
    ax.hist(h_organic, bins=60, alpha=0.6, density=True,
            label=f"Органика (med={np.median(h_organic):.2f})", color="steelblue")
    ax.hist(h_reco, bins=60, alpha=0.6, density=True,
            label=f"Рекомендации (med={np.median(h_reco):.2f})", color="coral")
    ax.set_xlabel("Shannon entropy (биты)")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Распределение энтропии — {label}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes_row[1]
    bp = ax2.boxplot([h_organic, h_reco], tick_labels=["Органика", "Рекомендации"],
                     patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], ["steelblue", "coral"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Shannon entropy (биты)")
    ax2.set_title(f"Органика vs рекомендации — {label}")
    ax2.grid(axis="y", alpha=0.3)


def run() -> None:
    t0 = time.perf_counter()
    print("Task 4: Shannon entropy разнообразия...")

    path        = str(find_parquet("listens"))
    artist_path = str(find_parquet("artist_item_mapping"))

    # На каждом чанке считаем энтропию сразу — наружу выходят только массивы float
    item_h_org_parts:   list[np.ndarray] = []
    item_h_reco_parts:  list[np.ndarray] = []
    artist_h_org_parts: list[np.ndarray] = []
    artist_h_reco_parts:list[np.ndarray] = []

    for chunk_id in range(CHUNKS):
        print(f"  Чанк {chunk_id + 1}/{CHUNKS}...", end=" ", flush=True)
        con = duckdb.connect()

        item_chunk = con.execute(f"""
            SELECT uid, item_id, is_organic, COUNT(*) AS cnt
            FROM read_parquet('{path}')
            WHERE (uid // 10) % {CHUNKS} = {chunk_id}
            GROUP BY uid, item_id, is_organic
        """).pl()

        artist_chunk = con.execute(f"""
            SELECT l.uid, a.artist_id, l.is_organic, COUNT(*) AS cnt
            FROM read_parquet('{path}') l
            JOIN read_parquet('{artist_path}') a ON l.item_id = a.item_id
            WHERE (l.uid // 10) % {CHUNKS} = {chunk_id}
            GROUP BY l.uid, a.artist_id, l.is_organic
        """).pl()

        con.close()

        h_io, h_ir = _entropy_per_user(item_chunk)
        h_ao, h_ar = _entropy_per_user(artist_chunk)
        item_h_org_parts.append(h_io)
        item_h_reco_parts.append(h_ir)
        artist_h_org_parts.append(h_ao)
        artist_h_reco_parts.append(h_ar)

        # Освобождаем тяжёлые DataFrame до следующей итерации
        del item_chunk, artist_chunk
        print(f"item users(org/reco)={len(h_io):,}/{len(h_ir):,} "
              f"artist={len(h_ao):,}/{len(h_ar):,}", flush=True)

    item_h_org   = np.concatenate(item_h_org_parts)   if item_h_org_parts   else np.array([])
    item_h_reco  = np.concatenate(item_h_reco_parts)  if item_h_reco_parts  else np.array([])
    artist_h_org = np.concatenate(artist_h_org_parts) if artist_h_org_parts else np.array([])
    artist_h_reco= np.concatenate(artist_h_reco_parts)if artist_h_reco_parts else np.array([])

    print(f"  item-энтропия:   organic med={np.median(item_h_org):.2f}, reco med={np.median(item_h_reco):.2f}")
    print(f"  artist-энтропия: organic med={np.median(artist_h_org):.2f}, reco med={np.median(artist_h_reco):.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Разнообразие слушания (Shannon entropy)", fontsize=14, fontweight="bold")

    _plot_entropy_row(axes[0], item_h_org,   item_h_reco,   "по трекам")
    _plot_entropy_row(axes[1], artist_h_org, artist_h_reco, "по артистам")

    plt.tight_layout()
    out = RESULTS_DIR / "task4_diversity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
