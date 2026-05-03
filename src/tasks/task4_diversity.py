"""
Task 4 — Разнообразие слушания: Shannon entropy.

Для каждого пользователя считаем энтропию двумя способами:
    - по item_id:   H = -sum(p_i * log2(p_i)), p_i = доля прослушиваний трека i
    - по artist_id: та же формула, единица группировки — артист (через artist_item_mapping)

Верхний ряд: item-энтропия. Нижний ряд: artist-энтропия.
Сравниваем органику vs рекомендации.

Вывод: data/results/task4_diversity.png
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, find_parquet

MIN_EVENTS_PER_USER = 10  # пользователи с малым числом событий дают нестабильную энтропию


def _entropy_from_counts(counts: pl.DataFrame) -> np.ndarray:
    """Считает Shannon entropy по пользователям. counts: (uid, <id_col>, cnt)."""
    user_totals = counts.group_by("uid").agg(pl.col("cnt").sum().alias("total"))
    merged = counts.join(user_totals, on="uid").filter(pl.col("total") >= MIN_EVENTS_PER_USER)
    entropies = (
        merged
        .with_columns((pl.col("cnt").cast(pl.Float64) / pl.col("total")).alias("p"))
        .with_columns((-pl.col("p") * pl.col("p").log(base=2)).alias("h_contrib"))
        .group_by("uid")
        .agg(pl.col("h_contrib").sum().alias("entropy"))
    )
    return entropies["entropy"].to_numpy()


def _plot_entropy_row(axes_row, h_organic: np.ndarray, h_reco: np.ndarray, label: str) -> None:
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
    ax2.boxplot(
        [h_organic, h_reco], labels=["Органика", "Рекомендации"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    ax2.set_ylabel("Shannon entropy (биты)")
    ax2.set_title(f"Органика vs рекомендации — {label}")
    ax2.grid(axis="y", alpha=0.3)


def run() -> None:
    t0 = time.perf_counter()
    print("Task 4: Shannon entropy разнообразия...")

    path = find_parquet("listens")
    artist_map_path = find_parquet("artist_item_mapping")

    # Общий корень — comm_subplan_elim устранит дублирование scan для всех 4 веток
    lf_base = pl.scan_parquet(path).select(["uid", "item_id", "is_organic"])

    # Ветки по item_id (без join)
    item_org_lf = (
        lf_base
        .filter(pl.col("is_organic") == 1)
        .group_by(["uid", "item_id"])
        .agg(pl.len().alias("cnt"))
    )
    item_reco_lf = (
        lf_base
        .filter(pl.col("is_organic") == 0)
        .group_by(["uid", "item_id"])
        .agg(pl.len().alias("cnt"))
    )

    # Ветки по artist_id — join тоже является общим subplan для двух веток
    lf_with_artist = (
        lf_base
        .join(pl.scan_parquet(artist_map_path), on="item_id", how="left")
        .filter(pl.col("artist_id").is_not_null())
    )
    artist_org_lf = (
        lf_with_artist
        .filter(pl.col("is_organic") == 1)
        .group_by(["uid", "artist_id"])
        .agg(pl.len().alias("cnt"))
    )
    artist_reco_lf = (
        lf_with_artist
        .filter(pl.col("is_organic") == 0)
        .group_by(["uid", "artist_id"])
        .agg(pl.len().alias("cnt"))
    )

    # Один проход по диску — Polars объединяет 4 плана через comm_subplan_elim
    counts_item_org, counts_item_reco, counts_artist_org, counts_artist_reco = pl.collect_all(
        [item_org_lf, item_reco_lf, artist_org_lf, artist_reco_lf]
    )

    item_h_org = _entropy_from_counts(counts_item_org)
    item_h_reco = _entropy_from_counts(counts_item_reco)
    artist_h_org = _entropy_from_counts(counts_artist_org)
    artist_h_reco = _entropy_from_counts(counts_artist_reco)

    print(f"  item-энтропия:   organic med={np.median(item_h_org):.2f}, reco med={np.median(item_h_reco):.2f}")
    print(f"  artist-энтропия: organic med={np.median(artist_h_org):.2f}, reco med={np.median(artist_h_reco):.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Разнообразие слушания (Shannon entropy)", fontsize=14, fontweight="bold")

    _plot_entropy_row(axes[0], item_h_org, item_h_reco, "по трекам")
    _plot_entropy_row(axes[1], artist_h_org, artist_h_reco, "по артистам")

    plt.tight_layout()
    out = RESULTS_DIR / "task4_diversity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
