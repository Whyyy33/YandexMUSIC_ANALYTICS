"""
Task 6 — Граф ко-прослушиваний.

Верхний ряд: граф треков — два трека связаны, если слушались в одной сессии.
Нижний ряд: граф артистов — та же логика на уровне артистов.
Вес ребра = число сессий, в которых оба объекта встречаются вместе.

Для управляемости: сэмплируем N случайных пользователей.

Вывод: data/results/task6_colistening.png
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from community import best_partition

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, SESSION_GAP_MINUTES, TIMESTAMP_UNIT_SECONDS, find_parquet

SESSION_GAP_SEC = SESSION_GAP_MINUTES * 60

# Параметры графа треков
MAX_ITEMS_PER_SESSION = 20
MIN_EDGE_WEIGHT = 5
TOP_N_NODES = 150
N_SAMPLE_USERS = 5_000

# Параметры графа артистов
MAX_LISTENED_ARTISTS_PER_SESSION = 30
MIN_EDGE_WEIGHT_ARTIST = 25
TOP_N_ARTISTS = 100


def _make_graph(edge_weights: dict[tuple[int, int], int], min_weight: int) -> nx.Graph:
    G = nx.Graph()
    for (u, v), w in edge_weights.items():
        if w >= min_weight:
            G.add_edge(u, v, weight=w)
    return G


def _plot_graph_row(
    G_full: nx.Graph,
    top_n: int,
    axes_row,
    title_graph: str,
    title_hist: str,
    node_unit: str,
) -> int:
    node_strength = {
        n: sum(d["weight"] for _, _, d in G_full.edges(n, data=True))
        for n in G_full.nodes()
    }
    top_nodes = sorted(node_strength, key=node_strength.get, reverse=True)[:top_n]
    G = G_full.subgraph(top_nodes).copy()

    partition = best_partition(G, weight="weight", random_state=42)
    n_comm = len(set(partition.values()))

    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=80)
    colors = [partition[n] for n in G.nodes()]

    # Рёбра: нормируем ширину, чтобы максимум был ~3
    edge_w = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    if edge_w.size:
        edge_widths = (edge_w / edge_w.max() * 2.5 + 0.3).tolist()
    else:
        edge_widths = []

    # Узлы: нормируем по максимальной strength → диапазон ~30..400
    sub_strength = np.array([node_strength[n] for n in G.nodes()], dtype=float)
    if sub_strength.size:
        node_sizes = (sub_strength / sub_strength.max() * 370 + 30).tolist()
    else:
        node_sizes = []

    ax = axes_row[0]
    ax.set_title(f"{title_graph}\nLouvain: {n_comm} сообществ")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=edge_widths, edge_color="gray")
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=node_sizes,
        cmap=plt.cm.tab20, alpha=0.85,
    )
    ax.axis("off")
    plt.colorbar(sc, ax=ax, label="Сообщество", shrink=0.7)

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
    print("Task 6: Граф ко-прослушиваний...")

    path = find_parquet("listens")
    artist_map_path = find_parquet("artist_item_mapping")

    # Сначала получаем uid для сэмпла — дёшево, без загрузки всего файла
    uid_series = pl.scan_parquet(path).select("uid").unique().collect()["uid"]
    sample_uids = uid_series.sample(min(N_SAMPLE_USERS, len(uid_series)), seed=42).to_list()

    # timestamp — абсолютное значение в 5-сек тиках, умножаем напрямую
    # Фильтр по uid стоит ДО collect — в память попадает только сэмпл
    df = (
        pl.scan_parquet(path)
        .select(["uid", "item_id", "timestamp"])
        .filter(pl.col("uid").is_in(sample_uids))
        .with_columns(
            (pl.col("timestamp").cast(pl.Int64) * TIMESTAMP_UNIT_SECONDS).alias("ts_seconds")
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

    # Добавляем artist_id заранее — чтобы в цикле не делать второй проход
    artist_map = pl.read_parquet(artist_map_path)
    df = df.join(artist_map, on="item_id", how="left")

    # Один цикл — два словаря рёбер одновременно
    item_edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    artist_edge_weights: dict[tuple[int, int], int] = defaultdict(int)

    for (uid, sess_id), grp in df.group_by(["uid", "session_id"]):
        # Треки в сессии
        items = grp["item_id"].unique().to_list()
        if 2 <= len(items) <= MAX_ITEMS_PER_SESSION:
            items_s = sorted(items)
            for i in range(len(items_s)):
                for j in range(i + 1, len(items_s)):
                    item_edge_weights[(items_s[i], items_s[j])] += 1

        # Артисты в сессии — drop_nulls убирает треки без маппинга, unique дедуплицирует
        artists = grp["artist_id"].drop_nulls().unique().to_list()
        if 2 <= len(artists) <= MAX_LISTENED_ARTISTS_PER_SESSION:
            artists_s = sorted(artists)
            for i in range(len(artists_s)):
                for j in range(i + 1, len(artists_s)):
                    artist_edge_weights[(artists_s[i], artists_s[j])] += 1

    G_items = _make_graph(item_edge_weights, MIN_EDGE_WEIGHT)
    G_artists = _make_graph(artist_edge_weights, MIN_EDGE_WEIGHT_ARTIST)

    print(f"  Треки — граф: {G_items.number_of_nodes()} вершин, {G_items.number_of_edges()} рёбер")
    print(f"  Артисты — граф: {G_artists.number_of_nodes()} вершин, {G_artists.number_of_edges()} рёбер")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Граф ко-прослушиваний ({N_SAMPLE_USERS:,} пользователей) — треки и артисты",
        fontsize=13, fontweight="bold",
    )

    n_comm_items = _plot_graph_row(
        G_items, TOP_N_NODES, axes[0],
        title_graph=f"Топ-{TOP_N_NODES} треков (вес ≥{MIN_EDGE_WEIGHT})",
        title_hist="Размеры Louvain-сообществ (треки)",
        node_unit="треков",
    )
    n_comm_artists = _plot_graph_row(
        G_artists, TOP_N_ARTISTS, axes[1],
        title_graph=f"Топ-{TOP_N_ARTISTS} артистов (вес ≥{MIN_EDGE_WEIGHT_ARTIST})",
        title_hist="Размеры Louvain-сообществ (артисты)",
        node_unit="артистов",
    )
    print(f"  Louvain треки: {n_comm_items} сообществ | артисты: {n_comm_artists} сообществ")

    plt.tight_layout()
    out = RESULTS_DIR / "task6_colistening.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
