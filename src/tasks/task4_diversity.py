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

QUARTILE_LABELS = ["Q1 (мало)", "Q2", "Q3", "Q4 (много)"]


def _entropy_per_user(counts: pl.DataFrame) -> pl.DataFrame:
    """Shannon entropy per user из (uid, <item|artist>, is_organic, cnt).
       Возвращает фрейм uid, is_organic, entropy, n_unique."""
    parts = []
    for is_org in [1, 0]:
        sub = counts.filter(pl.col("is_organic") == is_org)
        if sub.is_empty():
            continue
        user_totals = sub.group_by("uid").agg(
            pl.col("cnt").sum().alias("total"),
            pl.len().alias("n_unique"),
        )
        merged = sub.join(user_totals, on="uid").filter(pl.col("total") >= MIN_EVENTS_PER_USER)
        h = (
            merged
            .with_columns((pl.col("cnt").cast(pl.Float64) / pl.col("total")).alias("p"))
            .with_columns((-pl.col("p") * pl.col("p").log(base=2)).alias("hc"))
            .group_by("uid")
            .agg(
                pl.col("hc").sum().alias("entropy"),
                pl.col("n_unique").first().alias("n_unique"),
            )
            .with_columns(pl.lit(is_org).cast(pl.UInt8).alias("is_organic"))
        )
        parts.append(h.select(["uid", "is_organic", "entropy", "n_unique"]))
    if not parts:
        return pl.DataFrame(schema={
            "uid": pl.UInt32, "is_organic": pl.UInt8,
            "entropy": pl.Float64, "n_unique": pl.UInt32,
        })
    return pl.concat(parts)


def _plot_hist(ax, h_org, h_reco, label):
    ax.hist(h_org, bins=60, alpha=0.6, density=True,
            label=f"Органика (med={np.median(h_org):.2f})", color="steelblue")
    ax.hist(h_reco, bins=60, alpha=0.6, density=True,
            label=f"Рекомендации (med={np.median(h_reco):.2f})", color="coral")
    ax.set_xlabel("Энтропия Шеннона (биты)")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Распределение энтропии — {label}")
    ax.legend()
    ax.grid(alpha=0.3)


def _plot_box_orgreco(ax, h_org, h_reco, label):
    bp = ax.boxplot([h_org, h_reco], tick_labels=["Органика", "Рекомендации"],
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], ["steelblue", "coral"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Энтропия Шеннона (биты)")
    ax.set_title(f"Органика vs рекомендации — {label}")
    ax.grid(axis="y", alpha=0.3)


def _plot_box_quartiles(ax, df: pl.DataFrame, label_unit: str):
    """Boxplot энтропии по квартилям активности (n_unique). Параллельно органика и реко."""
    boxes_data = []
    positions = []
    colors = []
    width = 0.36
    q_medians: dict[int, dict[str, float]] = {}

    for grp_idx, (is_org, color) in enumerate([(1, "steelblue"), (0, "coral")]):
        sub = df.filter(pl.col("is_organic") == is_org)
        if sub.is_empty():
            continue
        # Квартили считаются в пределах группы (org/reco отдельно)
        edges = sub["n_unique"].quantile(np.linspace(0, 1, 5).tolist())
        q_edges = [sub["n_unique"].quantile(q) for q in [0, 0.25, 0.5, 0.75, 1.0]]
        q_edges = [float(x) for x in q_edges]
        for q in range(4):
            lo = q_edges[q]
            hi = q_edges[q + 1]
            if q < 3:
                bucket = sub.filter((pl.col("n_unique") >= lo) & (pl.col("n_unique") < hi))
            else:
                bucket = sub.filter((pl.col("n_unique") >= lo) & (pl.col("n_unique") <= hi))
            vals = bucket["entropy"].to_numpy()
            if len(vals) == 0:
                continue
            offset = (-1 if is_org == 1 else 1) * width / 2
            boxes_data.append(vals)
            positions.append(q + 1 + offset)
            colors.append(color)
            q_medians.setdefault(q, {})[("org" if is_org == 1 else "reco")] = float(np.median(vals))

    bp = ax.boxplot(
        boxes_data, positions=positions, widths=width,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(QUARTILE_LABELS)
    ax.set_xlabel(f"Квартили по {label_unit}")
    ax.set_ylabel("Энтропия Шеннона (биты)")
    ax.set_title(f"Энтропия по квартилям активности — {label_unit}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="steelblue", alpha=0.65),
        plt.Rectangle((0, 0), 1, 1, fc="coral",     alpha=0.65),
    ], labels=["Органика", "Рекомендации"], loc="lower right", fontsize=9)
    return q_medians


def run() -> None:
    t0 = time.perf_counter()
    print("Task 4: Shannon entropy разнообразия...")

    path        = str(find_parquet("listens"))
    artist_path = str(find_parquet("artist_item_mapping"))

    item_parts:   list[pl.DataFrame] = []
    artist_parts: list[pl.DataFrame] = []

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

        item_parts.append(_entropy_per_user(item_chunk))
        artist_parts.append(_entropy_per_user(artist_chunk))

        del item_chunk, artist_chunk
        print(
            f"item users(org/reco)={item_parts[-1].filter(pl.col('is_organic')==1).height:,}"
            f"/{item_parts[-1].filter(pl.col('is_organic')==0).height:,}",
            flush=True,
        )

    item_df   = pl.concat(item_parts)   if item_parts   else pl.DataFrame()
    artist_df = pl.concat(artist_parts) if artist_parts else pl.DataFrame()

    item_h_org   = item_df.filter(pl.col("is_organic") == 1)["entropy"].to_numpy()
    item_h_reco  = item_df.filter(pl.col("is_organic") == 0)["entropy"].to_numpy()
    artist_h_org = artist_df.filter(pl.col("is_organic") == 1)["entropy"].to_numpy()
    artist_h_reco= artist_df.filter(pl.col("is_organic") == 0)["entropy"].to_numpy()

    item_med_o,   item_med_r   = float(np.median(item_h_org)),   float(np.median(item_h_reco))
    artist_med_o, artist_med_r = float(np.median(artist_h_org)), float(np.median(artist_h_reco))
    print(f"  item-энтропия:   organic med={item_med_o:.2f}, reco med={item_med_r:.2f}")
    print(f"  artist-энтропия: organic med={artist_med_o:.2f}, reco med={artist_med_r:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    _plot_hist(axes[0, 0], item_h_org,   item_h_reco,   "по трекам")
    _plot_box_orgreco(axes[0, 1], item_h_org, item_h_reco, "по трекам")
    item_qm = _plot_box_quartiles(axes[0, 2], item_df, "n_unique_tracks")

    _plot_hist(axes[1, 0], artist_h_org, artist_h_reco, "по артистам")
    _plot_box_orgreco(axes[1, 1], artist_h_org, artist_h_reco, "по артистам")
    art_qm = _plot_box_quartiles(axes[1, 2], artist_df, "n_unique_artists")

    # Вывод про рост разнообразия с активностью — берём средние Q1/Q4 по двум режимам
    def _q_summary(qm: dict[int, dict[str, float]]) -> tuple[float, float]:
        q1 = np.mean([v for v in qm.get(0, {}).values()])
        q4 = np.mean([v for v in qm.get(3, {}).values()])
        return float(q1), float(q4)

    item_q1, item_q4 = _q_summary(item_qm)
    art_q1,  art_q4  = _q_summary(art_qm)
    delta_item = item_q4 - item_q1
    delta_art  = art_q4 - art_q1
    if max(delta_item, delta_art) >= 0.5:
        diversity_line = (
            f"Разнообразие растёт с активностью: треки Q1 med {item_q1:.2f} → Q4 med {item_q4:.2f} бит "
            f"(+{delta_item:.2f}), артисты Q1 {art_q1:.2f} → Q4 {art_q4:.2f} (+{delta_art:.2f})"
        )
    else:
        diversity_line = "Разнообразие слабо зависит от объёма прослушки"

    fig.suptitle(
        "Разнообразие слушания (энтропия Шеннона)\n"
        f"Рекомендации повышают энтропию: треки +{item_med_r-item_med_o:.2f} бита "
        f"(med {item_med_o:.2f}→{item_med_r:.2f}), артисты +{artist_med_r-artist_med_o:.2f} "
        f"(med {artist_med_o:.2f}→{artist_med_r:.2f})\n"
        f"{diversity_line}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    out = RESULTS_DIR / "task4_diversity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
