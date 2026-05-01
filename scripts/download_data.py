"""
Скачивает parquet-файлы Yambda-500M с HuggingFace в data/raw/.
Пропускает файлы, которые уже скачаны (проверка по размеру).

Использование:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "yandex/yambda"
REPO_TYPE = "dataset"

# Основные файлы событий (flat/500m/)
FILES = {
    "listens":     "flat/500m/listens.parquet",
    "likes":       "flat/500m/likes.parquet",
    "dislikes":    "flat/500m/dislikes.parquet",
    "multi_event": "flat/500m/multi_event.parquet",
}

# Маппинги item_id → метаданные (в корне репо, маленькие файлы)
MAPPINGS = {
    "album_item_mapping":  "album_item_mapping.parquet",
    "artist_item_mapping": "artist_item_mapping.parquet",
}

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Минимальный размер для основных файлов (5 МБ — защита от обрыва)
MIN_FILE_SIZE = 5 * 1024 * 1024
# Маппинги маленькие — достаточно 100 КБ
MIN_MAPPING_SIZE = 100 * 1024


def already_downloaded(path: Path, min_size: int = MIN_FILE_SIZE) -> bool:
    return path.exists() and path.stat().st_size > min_size


def fetch_one(repo_path: str, dest: Path) -> None:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type=REPO_TYPE,
        local_dir=str(RAW_DIR),
    )
    src = Path(local_path)
    if src != dest:
        src.rename(dest)


def download_files() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    skipped = []
    downloaded = []

    print(f"Целевая папка: {RAW_DIR.resolve()}\n")

    # --- Основные файлы событий ---
    for name, repo_path in tqdm(FILES.items(), desc="Yambda-500M события", unit="файл"):
        dest = RAW_DIR / f"{name}.parquet"

        if already_downloaded(dest, MIN_FILE_SIZE):
            tqdm.write(f"  [пропуск] {name}.parquet уже есть ({dest.stat().st_size / 1e9:.2f} ГБ)")
            skipped.append(name)
            continue

        tqdm.write(f"  [загрузка] {repo_path} ...")
        fetch_one(repo_path, dest)
        tqdm.write(f"  [готово]   {name}.parquet ({dest.stat().st_size / 1e9:.2f} ГБ)")
        downloaded.append(name)

    # --- Маппинги item_id → метаданные ---
    for name, repo_path in tqdm(MAPPINGS.items(), desc="Маппинги", unit="файл"):
        dest = RAW_DIR / f"{name}.parquet"

        if already_downloaded(dest, MIN_MAPPING_SIZE):
            tqdm.write(f"  [пропуск] {name}.parquet уже есть ({dest.stat().st_size / 1e6:.1f} МБ)")
            skipped.append(name)
            continue

        tqdm.write(f"  [загрузка] {repo_path} ...")
        fetch_one(repo_path, dest)
        tqdm.write(f"  [готово]   {name}.parquet ({dest.stat().st_size / 1e6:.1f} МБ)")
        downloaded.append(name)

    print("\n--- Итог ---")
    if downloaded:
        print(f"Скачано:  {', '.join(downloaded)}")
    if skipped:
        print(f"Пропущено (уже были): {', '.join(skipped)}")


if __name__ == "__main__":
    try:
        download_files()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(1)
