"""
Task 3 — Граф ко-лайков + Louvain-сообщества.

Два трека связаны, если их лайкнул один и тот же пользователь.
Вес ребра = число пользователей, лайкнувших оба трека.
Louvain выявляет кластеры треков со схожей аудиторией.

Вывод: data/results/task3_colikes.png
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
from community import best_partition  # python-louvain

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, find_parquet

MAX_LIKES_PER_USER = 30   # cap: игнорируем пользователей с аномально большим числом лайков
MIN_EDGE_WEIGHT = 3       # ребро в граф только если ≥3 пользователей лайкнули оба трека
TOP_N_NODES = 200         # берём топ-N треков по суммарному весу рёбер для визуализации


def build_colikes_graph(df: pl.DataFrame) -> nx.Graph:
    user_items: dict[int, list[int]] = defaultdict(list)
    for uid, item_id in df.iter_rows():
        user_items[uid].append(item_id)

    edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    for items in user_items.values():
        if len(items) < 2 or len(items) > MAX_LIKES_PER_USER:
            continue
        items_sorted = sorted(set(items))
        for i in range(len(items_sorted)):
            for j in range(i + 1, len(items_sorted)):
                edge_weights[(items_sorted[i], items_sorted[j])] += 1

    G = nx.Graph()
    for (u, v), w in edge_weights.items():
        if w >= MIN_EDGE_WEIGHT:
            G.add_edge(u, v, weight=w)

    return G


def run() -> None:
    print("Task 3: Граф ко-лайков...")

    path = find_parquet("likes")
    df = pl.read_parquet(path, columns=["uid", "item_id"])

    print(f"  Лайков: {len(df):,}, уникальных пользователей: {df['uid'].n_unique():,}")

    G_full = build_colikes_graph(df)
    print(f"  Граф: {G_full.number_of_nodes()} вершин, {G_full.number_of_edges()} рёбер")

    # Берём топ-N узлов по суммарному весу
    node_strength = {n: sum(d["weight"] for _, _, d in G_full.edges(n, data=True))
                     for n in G_full.nodes()}
    top_nodes = sorted(node_strength, key=node_strength.get, reverse=True)[:TOP_N_NODES]
    G = G_full.subgraph(top_nodes).copy()

    # Louvain
    partition = best_partition(G, weight="weight", random_state=42)
    n_communities = len(set(partition.values()))
    print(f"  Louvain: {n_communities} сообществ (топ-{TOP_N_NODES} треков)")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Граф ко-лайков (топ-{TOP_N_NODES} треков)\n"
        f"Рёбра: вес ≥{MIN_EDGE_WEIGHT} | Louvain: {n_communities} сообществ",
        fontsize=13, fontweight="bold"
    )

    pos = nx.spring_layout(G, seed=42, k=0.5)
    colors = [partition[n] for n in G.nodes()]
    weights = [G[u][v]["weight"] * 0.3 for u, v in G.edges()]
    sizes = [node_strength[n] * 0.5 + 20 for n in G.nodes()]

    ax = axes[0]
    ax.set_title("Граф с Louvain-сообществами")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=weights, edge_color="gray")
    sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                                cmap=plt.cm.tab20, alpha=0.85)
    ax.axis("off")
    plt.colorbar(sc, ax=ax, label="Сообщество", shrink=0.7)

    # Гистограмма размеров сообществ
    ax2 = axes[1]
    comm_sizes = defaultdict(int)
    for c in partition.values():
        comm_sizes[c] += 1
    sizes_sorted = sorted(comm_sizes.values(), reverse=True)
    ax2.bar(range(len(sizes_sorted)), sizes_sorted, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Сообщество (по убыванию размера)")
    ax2.set_ylabel("Число треков")
    ax2.set_title("Размеры Louvain-сообществ")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "task3_colikes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")


if __name__ == "__main__":
    run()
