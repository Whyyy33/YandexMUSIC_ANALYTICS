"""
Task 3 — Граф ко-лайков + Louvain-сообщества.

Верхний ряд: граф треков — два трека связаны, если их лайкнул один пользователь.
Нижний ряд: граф артистов — два артиста связаны, если их лайкнул один пользователь.
Вес ребра = число пользователей, лайкнувших оба объекта.

Вывод: data/results/task3_colikes.png
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
from community import best_partition  # python-louvain

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, find_parquet

# Параметры графа треков
MAX_LIKES_PER_USER = 30
MIN_EDGE_WEIGHT = 3
TOP_N_NODES = 200

# Параметры графа артистов
MAX_LIKED_ARTISTS_PER_USER = 50
MIN_EDGE_WEIGHT_ARTIST = 5
TOP_N_ARTISTS = 100


def _build_graph(uid_to_ids: dict[int, list[int]], max_per_user: int, min_weight: int) -> nx.Graph:
    edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    for ids in uid_to_ids.values():
        if len(ids) < 2 or len(ids) > max_per_user:
            continue
        ids_sorted = sorted(set(ids))
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                edge_weights[(ids_sorted[i], ids_sorted[j])] += 1

    G = nx.Graph()
    for (u, v), w in edge_weights.items():
        if w >= min_weight:
            G.add_edge(u, v, weight=w)
    return G


def build_colikes_graph(df: pl.DataFrame) -> nx.Graph:
    """Граф треков: df с колонками (uid, item_id)."""
    uid_items: dict[int, list[int]] = defaultdict(list)
    for uid, item_id in df.iter_rows():
        uid_items[uid].append(item_id)
    return _build_graph(uid_items, MAX_LIKES_PER_USER, MIN_EDGE_WEIGHT)


def build_artist_colikes_graph(df: pl.DataFrame) -> nx.Graph:
    """Граф артистов: df с колонками (uid, artist_id), уже дедуплицированный."""
    uid_artists: dict[int, list[int]] = defaultdict(list)
    for uid, artist_id in df.iter_rows():
        uid_artists[uid].append(artist_id)
    return _build_graph(uid_artists, MAX_LIKED_ARTISTS_PER_USER, MIN_EDGE_WEIGHT_ARTIST)


def _plot_graph_row(
    G_full: nx.Graph,
    top_n: int,
    axes_row,
    title_graph: str,
    title_hist: str,
    node_unit: str,
) -> int:
    """Рисует пару (граф + гистограмма сообществ) в одну строку axes."""
    node_strength = {
        n: sum(d["weight"] for _, _, d in G_full.edges(n, data=True))
        for n in G_full.nodes()
    }
    top_nodes = sorted(node_strength, key=node_strength.get, reverse=True)[:top_n]
    G = G_full.subgraph(top_nodes).copy()

    partition = best_partition(G, weight="weight", random_state=42)
    n_comm = len(set(partition.values()))

    pos = nx.spring_layout(G, seed=42, k=0.5)
    colors = [partition[n] for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] * 0.3 for u, v in G.edges()]
    node_sizes = [node_strength[n] * 0.5 + 20 for n in G.nodes()]

    # Граф
    ax = axes_row[0]
    ax.set_title(f"{title_graph}\nLouvain: {n_comm} сообществ")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=edge_widths, edge_color="gray")
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=node_sizes,
        cmap=plt.cm.tab20, alpha=0.85,
    )
    ax.axis("off")
    plt.colorbar(sc, ax=ax, label="Сообщество", shrink=0.7)

    # Гистограмма размеров сообществ
    ax2 = axes_row[1]
    comm_sizes = defaultdict(int)
    for c in partition.values():
        comm_sizes[c] += 1
    sizes_sorted = sorted(comm_sizes.values(), reverse=True)
    ax2.bar(range(len(sizes_sorted)), sizes_sorted, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Сообщество (по убыванию размера)")
    ax2.set_ylabel(f"Число {node_unit}")
    ax2.set_title(title_hist)
    ax2.grid(axis="y", alpha=0.3)

    return n_comm


def run() -> None:
    t0 = time.perf_counter()
    print("Task 3: Граф ко-лайков...")

    path = find_parquet("likes")
    artist_map_path = find_parquet("artist_item_mapping")

    # likes маленький (74 МБ) — читаем целиком
    df_likes = pl.read_parquet(path, columns=["uid", "item_id"])
    print(f"  Лайков: {len(df_likes):,}, уникальных пользователей: {df_likes['uid'].n_unique():,}")

    # --- Граф треков ---
    G_items = build_colikes_graph(df_likes)
    print(f"  Треки — граф: {G_items.number_of_nodes()} вершин, {G_items.number_of_edges()} рёбер")

    # --- Граф артистов ---
    artist_map = pl.read_parquet(artist_map_path)
    df_artist_likes = (
        df_likes
        .join(artist_map, on="item_id", how="left")
        .filter(pl.col("artist_id").is_not_null())
        .select(["uid", "artist_id"])
        .unique()  # один uid × один artist_id — убираем повторные лайки одному артисту
    )
    G_artists = build_artist_colikes_graph(df_artist_likes)
    print(f"  Артисты — граф: {G_artists.number_of_nodes()} вершин, {G_artists.number_of_edges()} рёбер")

    # --- Визуализация 2×2 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Граф ко-лайков: треки и артисты", fontsize=14, fontweight="bold")

    n_comm_items = _plot_graph_row(
        G_items, TOP_N_NODES, axes[0],
        title_graph=f"Топ-{TOP_N_NODES} треков (вес рёбер ≥{MIN_EDGE_WEIGHT})",
        title_hist="Размеры Louvain-сообществ (треки)",
        node_unit="треков",
    )
    n_comm_artists = _plot_graph_row(
        G_artists, TOP_N_ARTISTS, axes[1],
        title_graph=f"Топ-{TOP_N_ARTISTS} артистов (вес рёбер ≥{MIN_EDGE_WEIGHT_ARTIST})",
        title_hist="Размеры Louvain-сообществ (артисты)",
        node_unit="артистов",
    )
    print(f"  Louvain треки: {n_comm_items} сообществ | артисты: {n_comm_artists} сообществ")

    plt.tight_layout()
    out = RESULTS_DIR / "task3_colikes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
