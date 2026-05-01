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

# Реальные пути файлов в репо (flat/500m/, не flat-multievent-500m)
FILES = {
    "listens":     "flat/500m/listens.parquet",
    "likes":       "flat/500m/likes.parquet",
    "dislikes":    "flat/500m/dislikes.parquet",
    "multi_event": "flat/500m/multi_event.parquet",
}

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Минимальный ожидаемый размер файла в байтах (5 МБ — защита от неполной загрузки)
MIN_FILE_SIZE = 5 * 1024 * 1024


def already_downloaded(path: Path) -> bool:
    return path.exists() and path.stat().st_size > MIN_FILE_SIZE


def download_files() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    skipped = []
    downloaded = []

    print(f"Целевая папка: {RAW_DIR.resolve()}\n")

    for name, repo_path in tqdm(FILES.items(), desc="Yambda-500M", unit="файл"):
        dest = RAW_DIR / f"{name}.parquet"

        if already_downloaded(dest):
            tqdm.write(f"  [пропуск] {name}.parquet уже есть ({dest.stat().st_size / 1e9:.2f} ГБ)")
            skipped.append(name)
            continue

        tqdm.write(f"  [загрузка] {repo_path} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type=REPO_TYPE,
            local_dir=str(RAW_DIR),
        )

        # hf_hub_download кладёт файл в data/raw/flat/500m/name.parquet — перемещаем наверх
        src = Path(local_path)
        if src != dest:
            src.rename(dest)

        size_gb = dest.stat().st_size / 1e9
        tqdm.write(f"  [готово]   {name}.parquet ({size_gb:.2f} ГБ)")
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
