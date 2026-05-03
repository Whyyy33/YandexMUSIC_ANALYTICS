"""
Task 4 — Разнообразие слушания: Shannon entropy.

Для каждого пользователя считаем энтропию по item_id:
    H = -sum(p_i * log2(p_i))  где p_i = доля прослушиваний трека i

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


def run() -> None:
    t0 = time.perf_counter()
    print("Task 4: Shannon entropy разнообразия...")

    path = find_parquet("listens")
    lazy = pl.scan_parquet(path).select(["uid", "item_id", "is_organic"])

    results = []
    for flag, label in [(1, "organic"), (0, "reco")]:
        # collect только после агрегации — в памяти оседают uid×item_id, а не 466M строк
        counts = (
            lazy
            .filter(pl.col("is_organic") == flag)
            .group_by(["uid", "item_id"])
            .agg(pl.len().alias("cnt"))
            .collect()
        )

        user_totals = counts.group_by("uid").agg(pl.col("cnt").sum().alias("total"))

        merged = counts.join(user_totals, on="uid").filter(pl.col("total") >= MIN_EVENTS_PER_USER)

        entropies = (
            merged
            .with_columns(
                (pl.col("cnt").cast(pl.Float64) / pl.col("total")).alias("p")
            )
            .with_columns(
                (-pl.col("p") * pl.col("p").log(base=2)).alias("h_contrib")
            )
            .group_by("uid")
            .agg(pl.col("h_contrib").sum().alias("entropy"))
        )

        entropies = entropies.with_columns(pl.lit(label).alias("type"))
        results.append(entropies)

    all_entropies = pl.concat(results)

    organic_h = all_entropies.filter(pl.col("type") == "organic")["entropy"].to_numpy()
    reco_h = all_entropies.filter(pl.col("type") == "reco")["entropy"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Разнообразие слушания (Shannon entropy)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(organic_h, bins=60, alpha=0.6, density=True,
            label=f"Органика (med={np.median(organic_h):.2f})", color="steelblue")
    ax.hist(reco_h, bins=60, alpha=0.6, density=True,
            label=f"Рекомендации (med={np.median(reco_h):.2f})", color="coral")
    ax.set_xlabel("Shannon entropy (биты)")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределение энтропии по пользователям")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.boxplot([organic_h, reco_h], labels=["Органика", "Рекомендации"],
                patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6),
                medianprops=dict(color="black", linewidth=2))
    ax2.set_ylabel("Shannon entropy (биты)")
    ax2.set_title("Сравнение органика vs рекомендации")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "task4_diversity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
