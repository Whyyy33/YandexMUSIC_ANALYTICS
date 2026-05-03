"""
Task 6 — Граф ко-прослушиваний.

Аналог task3, но для прослушиваний (listens).
Чтобы граф не был сверхплотным, берём только пары треков,
прослушанных в одной сессии одним пользователем.

Для управляемости: сэмплируем N случайных пользователей.

Вывод: data/results/task6_colistening.png
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
from community import best_partition

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, SESSION_GAP_MINUTES, TIMESTAMP_UNIT_SECONDS, find_parquet

SESSION_GAP_SEC  = SESSION_GAP_MINUTES * 60
MAX_ITEMS_PER_SESSION = 20   # не строим пары для сверхдлинных сессий
MIN_EDGE_WEIGHT  = 5
TOP_N_NODES      = 150
N_SAMPLE_USERS   = 5_000     # берём подвыборку пользователей (уменьшено для первого запуска)


def run() -> None:
    t0 = time.perf_counter()
    print("Task 6: Граф ко-прослушиваний...")

    path = find_parquet("listens")

    df = (
        pl.scan_parquet(path)
        .select(["uid", "item_id", "timestamp"])
        .with_columns(
            (pl.col("timestamp").cum_sum().over("uid") * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
        )
        .with_columns(
            (
                (pl.col("ts_seconds") - pl.col("ts_seconds").shift(1).over("uid")).fill_null(0)
                > SESSION_GAP_SEC
            ).cast(pl.Int32).alias("new_sess")
        )
        .with_columns(
            pl.col("new_sess").cum_sum().over("uid").alias("session_id")
        )
        .collect()
    )

    # Сэмплируем пользователей
    unique_uids = df["uid"].unique().sample(min(N_SAMPLE_USERS, df["uid"].n_unique()), seed=42)
    df = df.filter(pl.col("uid").is_in(unique_uids))

    # Строим граф
    edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    for (uid, sess_id), grp in df.group_by(["uid", "session_id"]):
        items = grp["item_id"].unique().to_list()
        if len(items) < 2 or len(items) > MAX_ITEMS_PER_SESSION:
            continue
        items_s = sorted(items)
        for i in range(len(items_s)):
            for j in range(i + 1, len(items_s)):
                edge_weights[(items_s[i], items_s[j])] += 1

    G_full = nx.Graph()
    for (u, v), w in edge_weights.items():
        if w >= MIN_EDGE_WEIGHT:
            G_full.add_edge(u, v, weight=w)

    print(f"  Граф: {G_full.number_of_nodes()} вершин, {G_full.number_of_edges()} рёбер")

    node_strength = {n: sum(d["weight"] for _, _, d in G_full.edges(n, data=True))
                     for n in G_full.nodes()}
    top_nodes = sorted(node_strength, key=node_strength.get, reverse=True)[:TOP_N_NODES]
    G = G_full.subgraph(top_nodes).copy()

    partition = best_partition(G, weight="weight", random_state=42)
    n_communities = len(set(partition.values()))
    print(f"  Louvain: {n_communities} сообществ (топ-{TOP_N_NODES} треков)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Граф ко-прослушиваний (топ-{TOP_N_NODES} треков, {N_SAMPLE_USERS:,} пользователей)\n"
        f"Louvain: {n_communities} сообществ",
        fontsize=13, fontweight="bold"
    )

    pos = nx.spring_layout(G, seed=42, k=0.5)
    colors = [partition[n] for n in G.nodes()]
    weights = [G[u][v]["weight"] * 0.2 for u, v in G.edges()]
    sizes = [node_strength[n] * 0.3 + 20 for n in G.nodes()]

    ax = axes[0]
    ax.set_title("Граф с Louvain-сообществами")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=weights, edge_color="gray")
    sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                                cmap=plt.cm.tab20, alpha=0.85)
    ax.axis("off")
    plt.colorbar(sc, ax=ax, label="Сообщество", shrink=0.7)

    ax2 = axes[1]
    comm_sizes = defaultdict(int)
    for c in partition.values():
        comm_sizes[c] += 1
    sizes_sorted = sorted(comm_sizes.values(), reverse=True)
    ax2.bar(range(len(sizes_sorted)), sizes_sorted, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Сообщество")
    ax2.set_ylabel("Число треков")
    ax2.set_title("Размеры Louvain-сообществ")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "task6_colistening.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
