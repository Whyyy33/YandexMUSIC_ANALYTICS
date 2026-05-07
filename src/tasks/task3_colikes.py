"""
Task 3 — Граф ко-лайков + Louvain-сообщества + матрица смежности.

Слева вверху: граф треков — два трека связаны, если их лайкнул один пользователь.
Слева внизу: граф артистов.
Справа: матрицы смежности тех же узлов, упорядочены по сообществам Louvain.

Веса рёбер очищены от отозванных лайков (likes - unlikes ≤ 0 → пара исключена).

Память: рёбра считаются в DuckDB через self-join `likes` по uid с условием
item_a < item_b и фильтром HAVING COUNT(*) >= MIN_EDGE_WEIGHT — то есть
полный edge-set НЕ материализуется в Python-dict, в pandas/networkx приходят
только рёбра выше порога. Юзеры с лайками вне диапазона [2, MAX_LIKES_PER_USER]
отбрасываются ДО self-join'а.

Вывод: data/results/task3_colikes.png
"""

import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from adjustText import adjust_text
from community import best_partition  # python-louvain
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    RAM_SOFT_CAP_GB,
    RESULTS_DIR,
    find_parquet,
)
from src.rank_labels import artist_label_map, track_label_map

# Параметры графа треков
MAX_LIKES_PER_USER = 30
MIN_EDGE_WEIGHT = 3
TOP_N_NODES = 200

# Параметры графа артистов
MAX_LIKED_ARTISTS_PER_USER = 50
MIN_EDGE_WEIGHT_ARTIST = 15
TOP_N_ARTISTS = 120

# Глобальная статистика для подзаголовка — заполняется в run()
_NET_STATS: dict[str, float] = {}


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    return con


def _check_size(df: pl.DataFrame, name: str) -> None:
    sz_gb = df.estimated_size() / 1e9
    print(f"  [mem] {name}: {df.height:,} строк, {sz_gb:.2f} ГБ")
    if sz_gb > RAM_SOFT_CAP_GB:
        raise MemoryError(
            f"{name} = {sz_gb:.2f} ГБ превышает RAM_SOFT_CAP_GB={RAM_SOFT_CAP_GB} ГБ. "
            f"Подними потолок в config.py или ужесточи фильтр."
        )


def _build_net_likes_view(
    con: duckdb.DuckDBPyConnection,
    likes_path: str,
    multi_event_path: str,
) -> str:
    """Создаёт VIEW net_user_items с парами (uid, item_id) у которых
    likes - unlikes > 0. unlike-события берутся из multi_event.parquet
    (event_type='unlike'). Заполняет _NET_STATS статистикой исключений."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _likes_agg AS
        SELECT uid, item_id, COUNT(*) AS n_likes
        FROM read_parquet('{likes_path}')
        GROUP BY uid, item_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _unlikes_agg AS
        SELECT uid, item_id, COUNT(*) AS n_unlikes
        FROM read_parquet('{multi_event_path}')
        WHERE event_type = 'unlike'
        GROUP BY uid, item_id
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _net_pairs AS
        SELECT
            l.uid,
            l.item_id,
            l.n_likes - COALESCE(u.n_unlikes, 0) AS net_likes
        FROM _likes_agg l
        LEFT JOIN _unlikes_agg u USING (uid, item_id)
    """)
    stats = con.execute("""
        SELECT
            COUNT(*)                                     AS total_pairs,
            SUM(CASE WHEN net_likes <= 0 THEN 1 ELSE 0 END) AS dropped_pairs
        FROM _net_pairs
    """).pl().to_dicts()[0]
    total = int(stats["total_pairs"])
    dropped = int(stats["dropped_pairs"])
    drop_pct = (dropped / total * 100) if total else 0.0
    _NET_STATS["total_pairs"] = total
    _NET_STATS["dropped_pairs"] = dropped
    _NET_STATS["drop_pct"] = drop_pct
    print(f"  Очистка: {dropped:,} / {total:,} пар отозваны (likes ≤ unlikes) — {drop_pct:.2f}%")

    con.execute("""
        CREATE OR REPLACE TEMP VIEW net_user_items AS
        SELECT uid, item_id FROM _net_pairs WHERE net_likes > 0
    """)
    return "net_user_items"


def _edges_for_likes(
    con: duckdb.DuckDBPyConnection,
    pairs_table: str,
    id_col: str,
    max_per_user: int,
    min_weight: int,
    extra_join_artist_path: str | None = None,
) -> pl.DataFrame:
    """Считает рёбра графа ко-лайков прямо в DuckDB.

    Возвращает фрейм (a, b, weight), уже отфильтрованный по min_weight.
    Юзеры с лайками вне [2, max_per_user] отбрасываются ДО self-join'а.
    """
    if extra_join_artist_path is None:
        base_select = f"SELECT DISTINCT uid, item_id AS {id_col} FROM {pairs_table}"
    else:
        base_select = f"""
            SELECT DISTINCT n.uid, m.artist_id AS {id_col}
            FROM {pairs_table} n
            JOIN read_parquet('{extra_join_artist_path}') m ON n.item_id = m.item_id
            WHERE m.artist_id IS NOT NULL
        """

    sql = f"""
    WITH base AS (
        {base_select}
    ),
    user_filter AS (
        SELECT uid
        FROM base
        GROUP BY uid
        HAVING COUNT(*) BETWEEN 2 AND {max_per_user}
    ),
    flat AS (
        SELECT b.uid, b.{id_col}
        FROM base b
        JOIN user_filter f USING (uid)
    )
    SELECT
        l1.{id_col} AS a,
        l2.{id_col} AS b,
        COUNT(*)   AS weight
    FROM flat l1
    JOIN flat l2 USING (uid)
    WHERE l1.{id_col} < l2.{id_col}
    GROUP BY l1.{id_col}, l2.{id_col}
    HAVING COUNT(*) >= {min_weight}
    """
    return con.execute(sql).pl()


def _graph_from_edges(edges: pl.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for a, b, w in edges.iter_rows():
        G.add_edge(a, b, weight=int(w))
    return G


def _plot_graph(
    G_full: nx.Graph,
    top_n: int,
    ax,
    title_graph: str,
    label_map: dict[int, str],
    n_label_nodes: int = 8,
) -> tuple[nx.Graph, dict[int, int], list[int], dict, plt.cm.ScalarMappable]:
    """Рисует граф с дискретной легендой сообществ и подписями топ-узлов.
    Возвращает (G_top, partition, comm_order, pos, cmap) — для построения матрицы рядом."""
    node_strength = {
        n: sum(d["weight"] for _, _, d in G_full.edges(n, data=True))
        for n in G_full.nodes()
    }
    top_nodes = sorted(node_strength, key=node_strength.get, reverse=True)[:top_n]
    G = G_full.subgraph(top_nodes).copy()

    partition = best_partition(G, weight="weight", random_state=42)
    comm_counts = Counter(partition.values())
    comm_order = [c for c, _ in comm_counts.most_common()]
    comm_index = {c: i for i, c in enumerate(comm_order)}
    n_comm = len(comm_order)
    sizes_sorted = [comm_counts[c] for c in comm_order]

    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=80)
    cmap = plt.cm.tab10 if n_comm <= 10 else plt.cm.tab20
    colors = [cmap(comm_index[partition[n]]) for n in G.nodes()]

    edge_w = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    edge_widths = (edge_w / edge_w.max() * 2.5 + 0.3).tolist() if edge_w.size else []

    sub_strength = np.array([node_strength[n] for n in G.nodes()], dtype=float)
    node_sizes = (sub_strength / sub_strength.max() * 370 + 30).tolist() if sub_strength.size else []

    sizes_str = "/".join(str(s) for s in sizes_sorted[:6]) + ("…" if len(sizes_sorted) > 6 else "")
    ax.set_title(f"{title_graph}\nLouvain: {n_comm} сообществ, размеры {sizes_str}")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=edge_widths, edge_color="gray")
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=node_sizes,
        alpha=0.9, edgecolors="black", linewidths=0.4,
    )

    label_nodes = sorted(G.nodes(), key=lambda n: node_strength[n], reverse=True)[:n_label_nodes]
    texts = []
    for n in label_nodes:
        x, y = pos[n]
        lab = label_map.get(int(n), f"#{int(n)}")
        texts.append(ax.text(
            x, y, lab, fontsize=9, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2),
            ha="center", va="center",
        ))
    if texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6, alpha=0.6),
            expand=(1.2, 1.4),
        )
    ax.axis("off")

    handles = [
        mpatches.Patch(color=cmap(i), label=f"Сообщество {i} (n={sizes_sorted[i]})")
        for i in range(min(n_comm, 8))
    ]
    if n_comm > 8:
        handles.append(mpatches.Patch(color="#777", label=f"… +{n_comm - 8} мелких"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85, frameon=True)

    return G, partition, comm_order, pos, cmap


def _plot_adj_matrix(
    G: nx.Graph,
    partition: dict[int, int],
    comm_order: list[int],
    cmap_louvain,
    ax,
    title: str,
) -> None:
    """Матрица смежности с упорядочением по сообществам и degree внутри."""
    degree = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}

    # Упорядочиваем узлы: сначала по comm (по comm_order), внутри — по degree desc
    ordered_nodes: list[int] = []
    comm_blocks: list[tuple[int, int, int]] = []  # (comm_label, start_idx, end_idx)
    pos_acc = 0
    for c in comm_order:
        nodes_in_comm = [n for n in G.nodes() if partition[n] == c]
        nodes_in_comm.sort(key=lambda n: degree[n], reverse=True)
        ordered_nodes.extend(nodes_in_comm)
        end = pos_acc + len(nodes_in_comm)
        comm_blocks.append((comm_order.index(c), pos_acc, end))
        pos_acc = end

    n = len(ordered_nodes)
    idx_of = {node: i for i, node in enumerate(ordered_nodes)}
    M = np.zeros((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        i, j = idx_of[u], idx_of[v]
        w = float(d.get("weight", 1))
        M[i, j] = w
        M[j, i] = w

    vmax = max(M.max(), 1.0)
    im = ax.imshow(
        M, cmap="YlOrRd",
        norm=LogNorm(vmin=1, vmax=vmax),
        interpolation="nearest", aspect="equal",
    )

    # Чёрные разделители блоков
    for _, _, end in comm_blocks[:-1]:
        ax.axhline(end - 0.5, color="black", lw=0.8)
        ax.axvline(end - 0.5, color="black", lw=0.8)

    # Тики посередине каждого блока
    tick_positions = [(s + e - 1) / 2 for _, s, e in comm_blocks]
    tick_labels = [f"С{idx} (n={e - s})" for idx, s, e in comm_blocks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    # Цветные полосы по краям — соответствуют цветам Louvain в графе
    # Реализуем через background patches на тиках
    for idx, s, e in comm_blocks:
        color = cmap_louvain(idx)
        ax.add_patch(plt.Rectangle(
            (-1.5, s - 0.5), 1.0, e - s,
            facecolor=color, edgecolor="none", clip_on=False, alpha=0.85,
        ))
        ax.add_patch(plt.Rectangle(
            (s - 0.5, -1.5), e - s, 1.0,
            facecolor=color, edgecolor="none", clip_on=False, alpha=0.85,
        ))

    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Сила связи (лог)")


def run() -> None:
    t0 = time.perf_counter()
    print("Task 3: Граф ко-лайков...")

    likes_path        = str(find_parquet("likes")).replace("\\", "/")
    multi_event_path  = str(find_parquet("multi_event")).replace("\\", "/")
    artist_path       = str(find_parquet("artist_item_mapping")).replace("\\", "/")

    con = _duckdb_connect()
    print(f"  DuckDB: memory_limit={DUCKDB_MEMORY_LIMIT}, threads={DUCKDB_THREADS}")

    # Очистка от отозванных лайков
    pairs_table = _build_net_likes_view(con, likes_path, multi_event_path)

    # --- Граф треков: рёбра считаются в DuckDB, фильтр min_weight внутри SQL ---
    print(f"  Считаем рёбра треков (min_weight={MIN_EDGE_WEIGHT})...")
    edges_items = _edges_for_likes(
        con, pairs_table,
        id_col="item_id",
        max_per_user=MAX_LIKES_PER_USER,
        min_weight=MIN_EDGE_WEIGHT,
    )
    _check_size(edges_items, "edges_items")
    G_items = _graph_from_edges(edges_items)
    print(f"  Треки — граф: {G_items.number_of_nodes()} вершин, {G_items.number_of_edges()} рёбер")

    # --- Граф артистов: те же рёбра, но id = artist_id из join'а ---
    print(f"  Считаем рёбра артистов (min_weight={MIN_EDGE_WEIGHT_ARTIST})...")
    edges_artists = _edges_for_likes(
        con, pairs_table,
        id_col="artist_id",
        max_per_user=MAX_LIKED_ARTISTS_PER_USER,
        min_weight=MIN_EDGE_WEIGHT_ARTIST,
        extra_join_artist_path=artist_path,
    )
    _check_size(edges_artists, "edges_artists")
    G_artists = _graph_from_edges(edges_artists)
    print(f"  Артисты — граф: {G_artists.number_of_nodes()} вершин, {G_artists.number_of_edges()} рёбер")

    con.close()

    track_lab  = track_label_map()
    artist_lab = artist_label_map()

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    G_t_top, part_t, order_t, _, cmap_t = _plot_graph(
        G_items, TOP_N_NODES, axes[0, 0],
        title_graph=f"Топ-{TOP_N_NODES} треков (вес ≥{MIN_EDGE_WEIGHT})",
        label_map=track_lab,
    )
    _plot_adj_matrix(
        G_t_top, part_t, order_t, cmap_t, axes[0, 1],
        title=f"Матрица совместных лайков — треки (n={G_t_top.number_of_nodes()})",
    )

    G_a_top, part_a, order_a, _, cmap_a = _plot_graph(
        G_artists, TOP_N_ARTISTS, axes[1, 0],
        title_graph=f"Топ-{TOP_N_ARTISTS} артистов (вес ≥{MIN_EDGE_WEIGHT_ARTIST})",
        label_map=artist_lab,
    )
    _plot_adj_matrix(
        G_a_top, part_a, order_a, cmap_a, axes[1, 1],
        title=f"Матрица совместных лайков — артисты (n={G_a_top.number_of_nodes()})",
    )

    n_comm_t = len(order_t)
    n_comm_a = len(order_a)
    sizes_a = [Counter(part_a.values())[c] for c in order_a]
    print(f"  Louvain треки: {n_comm_t} сообществ | артисты: {n_comm_a} сообществ")

    a_strength = {n: sum(d["weight"] for _, _, d in G_a_top.edges(n, data=True))
                  for n in G_a_top.nodes()}
    top_a_strength = sorted(a_strength, key=a_strength.get, reverse=True)[:2]
    top_a_names = ", ".join(artist_lab.get(int(a), f"#{int(a)}") for a in top_a_strength)

    drop_pct = _NET_STATS.get("drop_pct", 0.0)
    drop_line = ""
    if drop_pct > 5:
        drop_line = f"\nИз графа исключено {drop_pct:.1f}% пар (likes ≤ unlikes)"

    fig.suptitle(
        "Граф ко-лайков: треки и артисты\n"
        f"{n_comm_a} сообществ артистов (размеры {'/'.join(map(str, sizes_a[:5]))}), "
        f"ядро по числу связей — {top_a_names}"
        f"{drop_line}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.07)
    fig.text(
        0.5, 0.01,
        "Граф: размер узла ∝ числу связей, толщина ребра ∝ числу совместных лайков, "
        "цвет — сообщество Louvain. "
        "Матрица: тёмные квадраты на диагонали = плотные сообщества, "
        "тёмные клетки вне диагонали = мосты между сообществами. "
        "Граф очищен от отозванных лайков (unlikes). "
        "A_k / T_k — k-й по популярности артист/трек (Yambda анонимизирован).",
        ha="center", fontsize=8, color="dimgray", wrap=True,
    )
    out = RESULTS_DIR / "task3_colikes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
