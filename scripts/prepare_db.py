"""
Создаёт data/processed/analytics.duckdb:
  - views на parquet-файлы (listens, likes, dislikes)
  - проверяет доступность каждого файла
  - запускается один раз после download_data.py

Использование:
    python scripts/prepare_db.py
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ANALYTICS_DB, find_parquet


PARQUET_NAMES = ["listens", "likes", "dislikes", "multi_event",
                 "album_item_mapping", "artist_item_mapping"]


def main() -> None:
    print(f"База данных: {ANALYTICS_DB}\n")
    con = duckdb.connect(str(ANALYTICS_DB))

    available = []
    for name in PARQUET_NAMES:
        try:
            path = find_parquet(name)
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
            count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            size_mb = path.stat().st_size / 1e6
            print(f"  [OK] {name:25s}  {count:>12,} строк  {size_mb:>8.1f} МБ")
            available.append(name)
        except FileNotFoundError:
            print(f"  [--] {name:25s}  файл не найден, пропущено")
        except Exception as e:
            print(f"  [!!] {name:25s}  ошибка: {e}")

    con.close()
    print(f"\nГотово. Доступно: {len(available)}/{len(PARQUET_NAMES)} таблиц.")


if __name__ == "__main__":
    main()
