"""
Сборка демо-сэмпла данных для дистрибутива.

Берёт полный датасет Yambda из data/raw/ и создаёт уменьшенную копию
в data/demo/ — выборка из 5000 случайных пользователей.

Размер результата: ~80 МБ (вместо 1.3 ГБ raw + 14 ГБ embeddings).
Время работы: 3-5 минут.

Использование:
    python scripts/build_demo.py
    python scripts/build_demo.py --users 1000   # меньше юзеров
    python scripts/build_demo.py --with-embeddings  # включить эмбеддинги

Запускается ОДИН РАЗ на машине разработчика перед сборкой дистрибутива.
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    RAW_DATA_DIR,
    find_parquet,
)

DEMO_DIR = ROOT / "data" / "demo"


def _connect() -> duckdb.DuckDBPyConnection:
    """Подключение к DuckDB с настроенными лимитами."""
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute("PRAGMA temp_directory='data/processed/.duckdb_tmp'")
    return con


def _sample_uids(n_users: int, seed: int = 42) -> list[int]:
    """Берёт случайную выборку uid из listens.

    В Yambda все uid кратны 10. Используем hash(uid) для равномерности и
    воспроизводимости. Seed фиксирован — результат стабилен между запусками.
    """
    print(f"[1/8] Выбираем случайную выборку из {n_users:,} пользователей...")
    listens_path = str(find_parquet("listens")).replace("\\", "/")
    con = _connect()
    con.execute(f"SELECT setseed({seed / 100.0})")

    df = con.execute(f"""
        SELECT DISTINCT uid
        FROM read_parquet('{listens_path}')
        ORDER BY hash(uid)
        LIMIT {n_users}
    """).pl()
    con.close()

    uids = df["uid"].to_list()
    print(f"      Выбрано: {len(uids):,} уникальных uid")
    return uids


def _filter_events(table_name: str, uids: list[int], out_path: Path) -> None:
    """Фильтрует факт-таблицу (listens/likes/dislikes) по списку uid."""
    try:
        src = find_parquet(table_name)
    except FileNotFoundError:
        print(f"      Пропускаем {table_name}: файл не найден")
        return

    src_str = str(src).replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    con = _connect()
    con.execute("CREATE TEMP TABLE selected_uids (uid UBIGINT)")
    con.executemany("INSERT INTO selected_uids VALUES (?)", [(u,) for u in uids])

    con.execute(f"""
        COPY (
            SELECT t.*
            FROM read_parquet('{src_str}') t
            JOIN selected_uids s ON t.uid = s.uid
        ) TO '{out_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_str}')").fetchone()[0]
    size_mb = out_path.stat().st_size / 1e6
    con.close()

    print(f"      {table_name}: {rows:,} строк, {size_mb:.1f} МБ")


def _filter_mapping(table_name: str, item_ids: set, out_path: Path) -> None:
    """Фильтрует таблицу-маппинг по списку использованных item_id."""
    try:
        src = find_parquet(table_name)
    except FileNotFoundError:
        print(f"      Пропускаем {table_name}: файл не найден")
        return

    src_str = str(src).replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    con = _connect()
    con.execute("CREATE TEMP TABLE selected_items (item_id UBIGINT)")
    con.executemany("INSERT INTO selected_items VALUES (?)", [(i,) for i in item_ids])

    con.execute(f"""
        COPY (
            SELECT t.*
            FROM read_parquet('{src_str}') t
            JOIN selected_items s ON t.item_id = s.item_id
        ) TO '{out_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_str}')").fetchone()[0]
    size_mb = out_path.stat().st_size / 1e6
    con.close()

    print(f"      {table_name}: {rows:,} строк, {size_mb:.1f} МБ")


def _get_item_ids(demo_listens_path: Path) -> set:
    """Собирает множество item_id, появившихся в демо-listens."""
    con = _connect()
    out_str = str(demo_listens_path).replace("\\", "/")
    df = con.execute(f"""
        SELECT DISTINCT item_id FROM read_parquet('{out_str}')
    """).pl()
    con.close()
    return set(df["item_id"].to_list())


def _filter_embeddings(item_ids: set, out_path: Path) -> bool:
    """Фильтрует embeddings.parquet по item_id (опционально)."""
    try:
        src = find_parquet("embeddings")
    except FileNotFoundError:
        return False

    print(f"[7/8] Фильтруем embeddings.parquet (это может занять 1-2 минуты)...")
    src_str = str(src).replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    con = _connect()
    con.execute("CREATE TEMP TABLE selected_items (item_id UBIGINT)")
    con.executemany("INSERT INTO selected_items VALUES (?)", [(i,) for i in item_ids])

    con.execute(f"""
        COPY (
            SELECT t.*
            FROM read_parquet('{src_str}') t
            JOIN selected_items s ON t.item_id = s.item_id
        ) TO '{out_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_str}')").fetchone()[0]
    size_mb = out_path.stat().st_size / 1e6
    con.close()
    print(f"      embeddings: {rows:,} строк, {size_mb:.1f} МБ")
    return True


def _build_analytics_db() -> None:
    """Создаёт data/demo/analytics.duckdb с VIEW на демо-parquet."""
    print("[8/8] Создаём analytics.duckdb с VIEW...")
    db_path = DEMO_DIR / "analytics.duckdb"
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    tables = ["listens", "likes", "dislikes", "multi_event",
              "artist_item_mapping", "album_item_mapping"]
    for name in tables:
        f = DEMO_DIR / f"{name}.parquet"
        if f.exists():
            path_str = str(f).replace("\\", "/")
            con.execute(f"""
                CREATE OR REPLACE VIEW {name} AS
                SELECT * FROM read_parquet('{path_str}')
            """)
            cnt = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"      VIEW {name}: {cnt:,} строк")

    f = DEMO_DIR / "embeddings.parquet"
    if f.exists():
        path_str = str(f).replace("\\", "/")
        con.execute(f"""
            CREATE OR REPLACE VIEW embeddings AS
            SELECT * FROM read_parquet('{path_str}')
        """)
        print(f"      VIEW embeddings: создан")

    con.close()
    print(f"      Файл: {db_path.name} ({db_path.stat().st_size / 1024:.1f} КБ)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка демо-сэмпла для дистрибутива")
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-embeddings", action="store_true")
    args = parser.parse_args()

    try:
        find_parquet("listens")
    except FileNotFoundError:
        print("ОШИБКА: data/raw/listens.parquet не найден.")
        print("Сначала запустите: python scripts/download_data.py")
        sys.exit(1)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nГенерация демо-сэмпла:")
    print(f"  Пользователей:  {args.users:,}")
    print(f"  Seed:           {args.seed}")
    print(f"  Эмбеддинги:     {'да' if args.with_embeddings else 'нет (опционально)'}")
    print(f"  Выход:          {DEMO_DIR}\n")

    t0 = time.perf_counter()

    uids = _sample_uids(args.users, args.seed)

    print(f"[2/8] Фильтруем listens.parquet...")
    _filter_events("listens", uids, DEMO_DIR / "listens.parquet")

    print(f"[3/8] Фильтруем likes.parquet...")
    _filter_events("likes", uids, DEMO_DIR / "likes.parquet")

    print(f"[4/8] Фильтруем dislikes.parquet...")
    _filter_events("dislikes", uids, DEMO_DIR / "dislikes.parquet")

    print(f"[4b/8] Фильтруем multi_event.parquet...")
    _filter_events("multi_event", uids, DEMO_DIR / "multi_event.parquet")

    print(f"[5/8] Собираем список использованных item_id...")
    item_ids = _get_item_ids(DEMO_DIR / "listens.parquet")
    print(f"      Уникальных треков в демо: {len(item_ids):,}")

    print(f"[6/8] Фильтруем таблицы-маппинги...")
    _filter_mapping("artist_item_mapping", item_ids,
                    DEMO_DIR / "artist_item_mapping.parquet")
    _filter_mapping("album_item_mapping", item_ids,
                    DEMO_DIR / "album_item_mapping.parquet")

    if args.with_embeddings:
        ok = _filter_embeddings(item_ids, DEMO_DIR / "embeddings.parquet")
        if not ok:
            print(f"[7/8] embeddings.parquet не найден — пропускаем")
    else:
        print(f"[7/8] embeddings.parquet — пропускаем (флаг --with-embeddings)")

    _build_analytics_db()

    total_size = sum(f.stat().st_size for f in DEMO_DIR.glob("*"))
    elapsed = time.perf_counter() - t0
    print(f"\nГотово за {elapsed:.1f} сек")
    print(f"Общий размер демо-данных: {total_size / 1e6:.1f} МБ")
    print(f"Папка: {DEMO_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(1)
