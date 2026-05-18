"""
Скачивает Windows embeddable Python в dist/python_embed.zip.
Запускается ОДИН РАЗ при подготовке дистрибутива.

Размер: ~10 МБ.
URL: python.org официальный.
"""

import sys
import urllib.request
from pathlib import Path

PYTHON_VERSION = "3.11.9"
URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
OUT_PATH = DIST_DIR / "python_embed.zip"


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 5 * 1024 * 1024:
        size_mb = OUT_PATH.stat().st_size / 1e6
        print(f"python_embed.zip уже скачан ({size_mb:.1f} МБ). Пропускаем.")
        return

    print(f"Скачиваем Python {PYTHON_VERSION} embeddable с python.org...")
    print(f"URL: {URL}")
    print(f"Это займёт ~10 секунд...")

    try:
        urllib.request.urlretrieve(URL, OUT_PATH)
    except Exception as e:
        print(f"ОШИБКА скачивания: {e}")
        sys.exit(1)

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Готово: {OUT_PATH.name} ({size_mb:.1f} МБ)")


if __name__ == "__main__":
    main()
