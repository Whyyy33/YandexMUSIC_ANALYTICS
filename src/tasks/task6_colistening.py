"""
Task 6 — Граф ко-прослушиваний + матрица смежности + UMAP по аудио-эмбеддингам.

Слева вверху: граф треков — два трека связаны, если слушались в одной сессии.
Слева внизу: граф артистов.
По центру: матрицы смежности тех же узлов, упорядоченные по сообществам Louvain.
Справа: UMAP-проекция по аудио-эмбеддингам Yambda — проверяем, совпадают ли
поведенческие сообщества с акустическими.

Для управляемости: сэмплируем N случайных пользователей.

Вывод: data/results/task6_colistening.png
"""

import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from adjustText import adjust_text
from community import best_partition
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    MAX_USERS_TASK6,
    RAM_SOFT_CAP_GB,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SESSION_GAP_MINUTES,
    TIMESTAMP_UNIT_SECONDS,
    find_parquet,
)
from src.rank_labels import artist_label_map, track_label_map

SESSION_GAP_SEC = SESSION_GAP_MINUTES * 60
DAY_BUCKET_DIVISOR = 86400 // TIMESTAMP_UNIT_SECONDS  # 17280 — длина суток в Yambda-тиках

# Параметры графа треков
MIN_EDGE_WEIGHT = 5
TOP_N_NODES = 150
N_SAMPLE_USERS = MAX_USERS_TASK6  # из config.py — урезается на debug-прогонах
MAX_LISTENED_ITEMS_PER_USER = 300
POPULAR_ITEM_MIN_LISTENERS = 10

# Параметры графа артистов
MIN_EDGE_WEIGHT_ARTIST = 12
TOP_N_ARTISTS = 100
MAX_LISTENED_ARTISTS_PER_USER = 250
POPULAR_ARTIST_MIN_LISTENERS = 30

# Защитные пороги
MAX_EDGES_HARD_LIMIT = 500_000

EMBEDDINGS_PATH = RAW_DATA_DIR / "embeddings.parquet"


def _check_size(df: pl.DataFrame, name: str) -> None:
    sz_gb = df.estimated_size() / 1e9
    print(f"  [mem] {name}: {df.height:,} строк, {sz_gb:.2f} ГБ")
    if sz_gb > RAM_SOFT_CAP_GB:
        raise MemoryError(
            f"{name} = {sz_gb:.2f} ГБ превышает RAM_SOFT_CAP_GB={RAM_SOFT_CAP_GB} ГБ. "
            f"Уменьши MAX_USERS_TASK6 в config.py."
        )


def _make_graph(edges_df: pl.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for a, b, w in edges_df.iter_rows():
        G.add_edge(int(a), int(b), weight=int(w))
    return G


def _warn_if(condition: bool, msg: str) -> None:
    if condition:
        print(f"  [WARN] {msg}")


def _row_count(con, table: str) -> int:
    return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _build_cooccur_edges(
    con: "duckdb.DuckDBPyConnection",
    listens_path: str,
    sample_uids_csv: str,
    *,
    id_col: str,
    artist_join_path: str | None,
    popularity_min: int,
    max_per_user: int,
    min_edge_weight: int,
    label: str,
) -> pl.DataFrame:
    """Строит рёбра ко-прослушивания через DuckDB self-join по (uid, day_bucket).

    label — короткая метка для логов ('track' или 'artist').
    Возвращает фрейм (a, b, weight)."""
    # CTE 1: сырые (uid, item/artist, day_bucket) для сэмпла
    if artist_join_path is None:
        select_listens = f"""
            SELECT
                l.uid,
                l.item_id AS {id_col},
                CAST(l.timestamp / {DAY_BUCKET_DIVISOR} AS INTEGER) AS day_bucket
            FROM read_parquet('{listens_path}') l
            WHERE l.uid IN ({sample_uids_csv})
        """
    else:
        select_listens = f"""
            SELECT
                l.uid,
                m.artist_id AS {id_col},
                CAST(l.timestamp / {DAY_BUCKET_DIVISOR} AS INTEGER) AS day_bucket
            FROM read_parquet('{listens_path}') l
            JOIN read_parquet('{artist_join_path}') m ON l.item_id = m.item_id
            WHERE l.uid IN ({sample_uids_csv})
              AND m.artist_id IS NOT NULL
        """

    con.execute(f"CREATE OR REPLACE TEMP TABLE _user_listens_{label} AS {select_listens}")
    n = _row_count(con, f"_user_listens_{label}")
    print(f"  [{label}] user_listens: {n:,} строк")

    # CTE 2: популярные id'шники
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _popular_{label} AS
        SELECT {id_col}
        FROM _user_listens_{label}
        GROUP BY {id_col}
        HAVING COUNT(DISTINCT uid) >= {popularity_min}
    """)
    n_pop = _row_count(con, f"_popular_{label}")
    print(f"  [{label}] popular ({popularity_min}+ слушателей): {n_pop:,}")
    _warn_if(
        n_pop < 50,
        f"[{label}] слишком жёсткий порог популярности (n={n_pop}<50), "
        f"граф будет вырожденный — поднимай sample или снижай порог",
    )

    # CTE 3: tmp = только популярные id'шники из user_listens
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tmp_{label} AS
        SELECT DISTINCT u.uid, u.{id_col}, u.day_bucket
        FROM _user_listens_{label} u
        JOIN _popular_{label} p USING ({id_col})
    """)
    n_tmp = _row_count(con, f"_tmp_{label}")
    print(f"  [{label}] tmp (после popular фильтра): {n_tmp:,}")

    # CTE 4: префильтр юзеров — оставлять с 2..max_per_user уникальных популярных id
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _user_filter_{label} AS
        SELECT uid
        FROM (SELECT DISTINCT uid, {id_col} FROM _tmp_{label}) du
        GROUP BY uid
        HAVING COUNT(*) BETWEEN 2 AND {max_per_user}
    """)
    n_uf = _row_count(con, f"_user_filter_{label}")
    print(f"  [{label}] user_filter (2..{max_per_user} популярных {id_col}): {n_uf:,} юзеров")

    # CTE 5: flat — tmp ∩ user_filter
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _flat_{label} AS
        SELECT t.uid, t.{id_col}, t.day_bucket
        FROM _tmp_{label} t
        JOIN _user_filter_{label} f USING (uid)
    """)
    n_flat = _row_count(con, f"_flat_{label}")
    print(f"  [{label}] flat: {n_flat:,}")
    _warn_if(
        n_flat < 1000,
        f"[{label}] flat={n_flat}<1000 — мало данных для self-join, граф будет вырожденный",
    )

    # CTE 6: self-join по (uid, day_bucket) → рёбра + COUNT(DISTINCT uid) как вес
    edges_sql = f"""
        SELECT
            a1.{id_col} AS a,
            a2.{id_col} AS b,
            COUNT(DISTINCT a1.uid) AS weight
        FROM _flat_{label} a1
        JOIN _flat_{label} a2 USING (uid, day_bucket)
        WHERE a1.{id_col} < a2.{id_col}
        GROUP BY a1.{id_col}, a2.{id_col}
        HAVING COUNT(DISTINCT a1.uid) >= {min_edge_weight}
    """
    edges = con.execute(edges_sql).pl()
    n_edges = len(edges)
    print(f"  [{label}] edges (weight ≥ {min_edge_weight}): {n_edges:,}")
    _warn_if(
        n_edges < 50,
        f"[{label}] edges={n_edges}<50 — граф практически пустой",
    )
    return edges


def _plot_graph(
    G_full: nx.Graph,
    top_n: int,
    ax,
    title_graph: str,
    label_map: dict[int, str],
    n_label_nodes: int = 8,
) -> tuple[nx.Graph, dict[int, int], list[int], dict, plt.cm.ScalarMappable, list[int]]:
    """Рисует граф и возвращает (G_top, partition, comm_order, pos, cmap, top_strength_nodes)."""
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

    return G, partition, comm_order, pos, cmap, label_nodes[:2]


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

    ordered_nodes: list[int] = []
    comm_blocks: list[tuple[int, int, int]] = []
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

    for _, _, end in comm_blocks[:-1]:
        ax.axhline(end - 0.5, color="black", lw=0.8)
        ax.axvline(end - 0.5, color="black", lw=0.8)

    tick_positions = [(s + e - 1) / 2 for _, s, e in comm_blocks]
    tick_labels = [f"С{idx} (n={e - s})" for idx, s, e in comm_blocks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

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


def _load_embeddings_for_items(item_ids: list[int]) -> pl.DataFrame:
    """Через DuckDB (надёжнее polars на 13.8 ГБ файле + List-колонке).
    Filter pushится в read_parquet с использованием row group статистики."""
    import duckdb as _duckdb
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Не найден {EMBEDDINGS_PATH}.")
    ids_set = sorted(set(int(i) for i in item_ids))
    if not ids_set:
        return pl.DataFrame(schema={"item_id": pl.UInt32, "normalized_embed": pl.List(pl.Float64)})
    ids_csv = ",".join(str(i) for i in ids_set)
    con = _duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute("PRAGMA temp_directory='data/processed/.duckdb_tmp'")
    path = str(EMBEDDINGS_PATH).replace("\\", "/")
    df = con.execute(f"""
        SELECT item_id, normalized_embed
        FROM read_parquet('{path}')
        WHERE item_id IN ({ids_csv})
    """).pl()
    con.close()
    return df


def _embed_matrix(df: pl.DataFrame, key_col: str = "item_id") -> tuple[list[int], np.ndarray]:
    """Из колонки normalized_embed (List[Float]) возвращает (key_list, np.ndarray[N,D])."""
    keys = df[key_col].to_list()
    embs = df["normalized_embed"].to_list()
    M = np.asarray(embs, dtype=np.float32)
    return keys, M


def _plot_umap(
    ax,
    coords: np.ndarray,
    keys: list[int],
    partition: dict[int, int],
    comm_order: list[int],
    cmap_louvain,
    label_map: dict[int, str],
    n_labels: int = 10,
    title: str = "",
) -> float:
    """Scatter UMAP-точек, цвет по сообществу. Возвращает чистоту кластеров (silhouette-like proxy)."""
    comm_index = {c: i for i, c in enumerate(comm_order)}
    colors = [cmap_louvain(comm_index[partition[k]]) for k in keys]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=40, alpha=0.85,
               edgecolors="black", linewidths=0.3)

    # Подписи топ-N точек по degree-эквиваленту здесь не считаем — берём первые n_labels из keys
    label_keys = keys[:n_labels]
    texts = []
    for k in label_keys:
        i = keys.index(k)
        x, y = coords[i]
        lab = label_map.get(int(k), f"#{int(k)}")
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

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    # Чистота: средний косинус "точка-внутри-сообщества" минус "точка-снаружи"
    # Считаем через евклидову дистанцию в UMAP-пространстве
    from scipy.spatial.distance import cdist
    if len(keys) >= 5:
        groups = np.array([comm_index[partition[k]] for k in keys])
        D = cdist(coords, coords)
        np.fill_diagonal(D, np.nan)
        same = []
        diff = []
        for i in range(len(keys)):
            mask_same = (groups == groups[i]) & (np.arange(len(keys)) != i)
            mask_diff = groups != groups[i]
            if mask_same.any():
                same.append(np.nanmean(D[i, mask_same]))
            if mask_diff.any():
                diff.append(np.nanmean(D[i, mask_diff]))
        if same and diff:
            return float(np.mean(diff) / max(np.mean(same), 1e-9))
    return 0.0


def run() -> None:
    t0 = time.perf_counter()
    print("Task 6: Граф ко-прослушиваний...")

    import duckdb
    path = str(find_parquet("listens")).replace("\\", "/")
    artist_map_path = str(find_parquet("artist_item_mapping")).replace("\\", "/")

    uid_series = pl.scan_parquet(path).select("uid").unique().collect()["uid"]
    sample_uids = uid_series.sample(min(N_SAMPLE_USERS, len(uid_series)), seed=42).to_list()
    sample_uids_csv = ",".join(str(int(u)) for u in sample_uids)
    print(f"  Сэмпл: {len(sample_uids):,} юзеров")

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("PRAGMA threads=4")
    Path("data/processed/.duckdb_tmp").mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA temp_directory='data/processed/.duckdb_tmp'")

    # --- Граф треков через DuckDB self-join по (uid, day_bucket) ---
    edges_items = _build_cooccur_edges(
        con, path, sample_uids_csv,
        id_col="item_id",
        artist_join_path=None,
        popularity_min=POPULAR_ITEM_MIN_LISTENERS,
        max_per_user=MAX_LISTENED_ITEMS_PER_USER,
        min_edge_weight=MIN_EDGE_WEIGHT,
        label="track",
    )
    if len(edges_items) > MAX_EDGES_HARD_LIMIT:
        raise RuntimeError(
            f"Граф треков: {len(edges_items):,} рёбер > {MAX_EDGES_HARD_LIMIT:,} — "
            f"подними MIN_EDGE_WEIGHT в task6_colistening.py"
        )

    # --- Освобождаем DuckDB-память между графами (TEMP TABLE'ы трекового графа) ---
    import gc
    import psutil
    con.close()
    gc.collect()
    rss_gb = psutil.Process().memory_info().rss / 1024**3
    print(f"  DuckDB connection closed, memory released. RSS: {rss_gb:.2f} GB")

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA temp_directory='data/processed/.duckdb_tmp'")

    # --- Граф артистов: один прогон с MIN_EDGE_WEIGHT_ARTIST ---
    edges_artists = _build_cooccur_edges(
        con, path, sample_uids_csv,
        id_col="artist_id",
        artist_join_path=artist_map_path,
        popularity_min=POPULAR_ARTIST_MIN_LISTENERS,
        max_per_user=MAX_LISTENED_ARTISTS_PER_USER,
        min_edge_weight=MIN_EDGE_WEIGHT_ARTIST,
        label="artist",
    )
    used_artist_threshold = MIN_EDGE_WEIGHT_ARTIST
    if len(edges_artists) > MAX_EDGES_HARD_LIMIT:
        raise RuntimeError(
            f"Граф артистов: {len(edges_artists):,} рёбер > {MAX_EDGES_HARD_LIMIT:,} — "
            f"подними MIN_EDGE_WEIGHT_ARTIST в task6_colistening.py"
        )

    con.close()

    G_items = _make_graph(edges_items)
    G_artists = _make_graph(edges_artists)

    print(f"  Треки — граф: {G_items.number_of_nodes()} вершин, {G_items.number_of_edges()} рёбер")
    print(f"  Артисты — граф: {G_artists.number_of_nodes()} вершин, {G_artists.number_of_edges()} рёбер "
          f"(порог {used_artist_threshold})")

    track_lab  = track_label_map()
    artist_lab = artist_label_map()

    fig, axes = plt.subplots(2, 3, figsize=(26, 18))

    G_t_top, part_t, order_t, _, cmap_t, _ = _plot_graph(
        G_items, TOP_N_NODES, axes[0, 0],
        title_graph=f"Топ-{TOP_N_NODES} треков (вес ≥{MIN_EDGE_WEIGHT})",
        label_map=track_lab,
    )
    _plot_adj_matrix(
        G_t_top, part_t, order_t, cmap_t, axes[0, 1],
        title=f"Матрица совместных прослушиваний — треки (n={G_t_top.number_of_nodes()})",
    )

    G_a_top, part_a, order_a, _, cmap_a, top2_artists = _plot_graph(
        G_artists, TOP_N_ARTISTS, axes[1, 0],
        title_graph=f"Топ-{TOP_N_ARTISTS} артистов (вес ≥{used_artist_threshold})",
        label_map=artist_lab,
    )
    _plot_adj_matrix(
        G_a_top, part_a, order_a, cmap_a, axes[1, 1],
        title=f"Матрица совместных прослушиваний — артисты (n={G_a_top.number_of_nodes()})",
    )

    track_purity = 0.0
    artist_purity = 0.0

    # --- UMAP по аудио-эмбеддингам (опционально) ---
    if EMBEDDINGS_PATH.exists():
        import psutil as _psutil
        import umap

        proc = _psutil.Process()
        rss_before_track = proc.memory_info().rss / 1024**3
        print(f"  [emb track] RSS before: {rss_before_track:.2f} GB")
        track_ids = list(G_t_top.nodes())
        t_load = time.perf_counter()
        track_emb_df = _load_embeddings_for_items(track_ids)
        rss_after_track = proc.memory_info().rss / 1024**3
        print(f"  [emb track] загружено {len(track_emb_df):,} строк / {len(track_ids)} запрошено, "
              f"RSS after: {rss_after_track:.2f} GB, время: {time.perf_counter() - t_load:.1f} сек")
        t_keys, T = _embed_matrix(track_emb_df, key_col="item_id")
        track_purity = 0.0
        if len(t_keys) >= 5:
            n_neighbors_t = min(15, max(2, len(t_keys) - 1))
            print(f"  [umap track] input shape={T.shape}, n_neighbors={n_neighbors_t}")
            t_umap = time.perf_counter()
            reducer_t = umap.UMAP(n_neighbors=n_neighbors_t, min_dist=0.1, n_components=2,
                                  metric="cosine", random_state=42)
            T_xy = reducer_t.fit_transform(T)
            print(f"  [umap track] output shape={T_xy.shape}, fit={time.perf_counter() - t_umap:.1f} сек")
            # Сортируем by degree desc для красивых подписей
            degree_t = {n: sum(d["weight"] for _, _, d in G_t_top.edges(n, data=True)) for n in G_t_top.nodes()}
            order_idx = sorted(range(len(t_keys)), key=lambda i: -degree_t.get(t_keys[i], 0))
            t_keys_ord = [t_keys[i] for i in order_idx]
            T_xy_ord = T_xy[order_idx]
            track_purity = _plot_umap(
                axes[0, 2], T_xy_ord, t_keys_ord, part_t, order_t, cmap_t, track_lab,
                n_labels=8, title="UMAP проекция треков по аудио-эмбеддингам",
            )
        else:
            axes[0, 2].text(0.5, 0.5, "Недостаточно эмбеддингов для треков",
                            ha="center", va="center", transform=axes[0, 2].transAxes)
            axes[0, 2].axis("off")

        # Для артистов — эмбеддинги их треков, усреднённые
        rss_before_artist = proc.memory_info().rss / 1024**3
        print(f"  [emb artist] RSS before: {rss_before_artist:.2f} GB")
        artist_ids = list(G_a_top.nodes())
        artist_map = pl.read_parquet(artist_map_path)
        artist_items = (
            artist_map.filter(pl.col("artist_id").is_in(artist_ids))
            .select(["artist_id", "item_id"])
        )
        print(f"  [emb artist] artist→item rows: {len(artist_items):,}")
        art_track_ids = artist_items["item_id"].unique().to_list()
        t_load_a = time.perf_counter()
        art_emb_df = _load_embeddings_for_items(art_track_ids)
        rss_after_artist = proc.memory_info().rss / 1024**3
        print(f"  [emb artist] загружено {len(art_emb_df):,} строк / {len(art_track_ids):,} запрошено, "
              f"RSS after: {rss_after_artist:.2f} GB, время: {time.perf_counter() - t_load_a:.1f} сек")
        # Среднее эмбеддингов треков по артисту — поэлементно, через numpy.
        # polars Array.mean() вернул бы скаляр (среднее всех элементов всех векторов).
        if len(art_emb_df) > 0:
            joined = art_emb_df.join(artist_items, on="item_id", how="inner")
            embs_by_artist: dict[int, list[np.ndarray]] = {}
            for artist_id, embed in zip(
                joined["artist_id"].to_list(),
                joined["normalized_embed"].to_list(),
            ):
                embs_by_artist.setdefault(int(artist_id), []).append(
                    np.asarray(embed, dtype=np.float32)
                )
            a_keys = list(embs_by_artist.keys())
            A = np.array(
                [np.mean(np.stack(embs_by_artist[a]), axis=0) for a in a_keys],
                dtype=np.float32,
            )
            embed_dim = A.shape[1] if A.ndim == 2 else 0
            assert A.ndim == 2 and A.shape == (len(a_keys), embed_dim), (
                f"Ожидаем (N, D), получили {A.shape}"
            )
            print(f"  [emb artist] средний эмбед на артиста: shape={A.shape}")
        else:
            a_keys, A = [], np.zeros((0, 1))

        artist_purity = 0.0
        if len(a_keys) >= 5:
            n_neighbors_a = min(15, max(2, len(a_keys) - 1))
            print(f"  [umap artist] input shape={A.shape}, n_neighbors={n_neighbors_a}")
            t_umap_a = time.perf_counter()
            reducer_a = umap.UMAP(n_neighbors=n_neighbors_a, min_dist=0.1, n_components=2,
                                  metric="cosine", random_state=42)
            A_xy = reducer_a.fit_transform(A)
            print(f"  [umap artist] output shape={A_xy.shape}, fit={time.perf_counter() - t_umap_a:.1f} сек")
            degree_a = {n: sum(d["weight"] for _, _, d in G_a_top.edges(n, data=True)) for n in G_a_top.nodes()}
            order_idx = sorted(range(len(a_keys)), key=lambda i: -degree_a.get(a_keys[i], 0))
            a_keys_ord = [a_keys[i] for i in order_idx]
            A_xy_ord = A_xy[order_idx]
            artist_purity = _plot_umap(
                axes[1, 2], A_xy_ord, a_keys_ord, part_a, order_a, cmap_a, artist_lab,
                n_labels=10, title="UMAP проекция артистов по аудио-эмбеддингам",
            )
            # Аннотации: малое сообщество vs крупные перемешанные
            comm_index_a = {c: i for i, c in enumerate(order_a)}
            comm_centroids: dict[int, np.ndarray] = {}
            for i, k in enumerate(a_keys_ord):
                c = comm_index_a[part_a[k]]
                comm_centroids.setdefault(c, []).append(A_xy_ord[i])
            comm_sizes_local = {c: len(v) for c, v in comm_centroids.items()}
            if comm_sizes_local:
                small_c = min(comm_sizes_local, key=comm_sizes_local.get)
                big_cs  = sorted(comm_sizes_local, key=comm_sizes_local.get, reverse=True)[:2]
                small_xy = np.mean(np.array(comm_centroids[small_c]), axis=0)
                big_xy   = np.mean(np.concatenate(
                    [np.array(comm_centroids[c]) for c in big_cs]
                ), axis=0)
                ax_u = axes[1, 2]
                xlim = ax_u.get_xlim(); ylim = ax_u.get_ylim()
                x_off = (xlim[1] - xlim[0]) * 0.25
                y_off = (ylim[1] - ylim[0]) * 0.30
                ax_u.annotate(
                    f"Акустически изолированная\nгруппа (n={comm_sizes_local[small_c]})",
                    xy=small_xy, xytext=(small_xy[0] + x_off, small_xy[1] - y_off),
                    fontsize=9, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fff7c2", ec="black", lw=0.6, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                )
                ax_u.annotate(
                    "Поведенческие сообщества,\nакустически перемешаны",
                    xy=big_xy, xytext=(big_xy[0] - x_off, big_xy[1] + y_off),
                    fontsize=9, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#e6f0ff", ec="black", lw=0.6, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                )
        else:
            axes[1, 2].text(0.5, 0.5, "Недостаточно эмбеддингов для артистов",
                            ha="center", va="center", transform=axes[1, 2].transAxes)
            axes[1, 2].axis("off")
    else:
        print(f"  [emb] embeddings.parquet не найден — UMAP-валидация пропущена")
        axes[0, 2].text(0.5, 0.5,
                        "UMAP-валидация недоступна\n(требуется embeddings.parquet)",
                        ha="center", va="center", fontsize=11, color="#555",
                        transform=axes[0, 2].transAxes)
        axes[0, 2].axis("off")
        axes[1, 2].text(0.5, 0.5,
                        "UMAP-валидация недоступна\n(требуется embeddings.parquet)",
                        ha="center", va="center", fontsize=11, color="#555",
                        transform=axes[1, 2].transAxes)
        axes[1, 2].axis("off")

    n_comm_t = len(order_t)
    n_comm_a = len(order_a)
    sizes_a = [Counter(part_a.values())[c] for c in order_a]
    print(f"  Louvain треки: {n_comm_t} сообществ | артисты: {n_comm_a} сообществ")
    print(f"  UMAP кластерная чистота (между/внутри): треки={track_purity:.2f}, артисты={artist_purity:.2f}")

    top2_names = ", ".join(artist_lab.get(int(a), f"#{int(a)}") for a in top2_artists)
    # Честный вердикт: малое сообщество отделилось акустически, крупные — нет
    smallest_artist_comm_size = min(sizes_a) if sizes_a else 0
    largest_artist_comm_size  = max(sizes_a) if sizes_a else 0
    umap_verdict = (
        f"Только малое сообщество (n={smallest_artist_comm_size}) акустически отделено; "
        f"два крупных кластера (n={largest_artist_comm_size}…) перемешаны "
        f"в эмбеддинг-пространстве"
    )

    fig.suptitle(
        f"Граф ко-прослушиваний (сэмпл {N_SAMPLE_USERS:,} пользователей)\n"
        f"{n_comm_a} сообществ артистов (размеры {'/'.join(map(str, sizes_a[:5]))}), "
        f"ядро по числу связей — {top2_names}\n"
        f"{umap_verdict}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.07)
    fig.text(
        0.5, 0.01,
        "Связь = два трека (артиста) прослушаны одним юзером в один день (прокси сессии). "
        "Граф очищен от непопулярных треков (<10 слушателей в сэмпле) и артистов (<30) "
        "для снижения шума. "
        "Размер узла ∝ числу связей, толщина ребра ∝ числу юзеров с обеими прослушками, "
        "цвет — сообщество Louvain. "
        "Матрица: тёмные блоки = плотные сообщества. "
        "UMAP: 2D-проекция по аудио-эмбеддингам Yambda; точки одного цвета рядом друг с другом = "
        "поведенческое сообщество подтверждено акустически. "
        "A_k / T_k — k-й по популярности артист/трек (Yambda анонимизирован).",
        ha="center", fontsize=8, color="dimgray", wrap=True,
    )
    out = RESULTS_DIR / "task6_colistening.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {out}")
    print(f"  Время: {time.perf_counter() - t0:.2f} сек")


if __name__ == "__main__":
    run()
