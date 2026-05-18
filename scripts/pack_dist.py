"""
Упаковщик финального дистрибутива.

Собирает в dist/YandexMUSIC_ANALYTICS_v1.0.zip:
  - код проекта (src/, scripts/)
  - bat-файлы (setup.bat, run.bat)
  - README дистрибутива
  - requirements.txt
  - data_demo.zip (упакованные демо-данные)
  - python_embed.zip (Windows embeddable Python)

Запускается ОДИН РАЗ перед раздачей дистрибутива.
Требует: data/demo/ заполнена (выполнен build_demo.py),
         dist/python_embed.zip существует (выполнен download_embeddable_python.py).
"""

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
DEMO_DIR = ROOT / "data" / "demo"
PYTHON_EMBED = DIST_DIR / "python_embed.zip"

DIST_NAME = "YandexMUSIC_ANALYTICS_v1.0"
OUTPUT_ZIP = DIST_DIR / f"{DIST_NAME}.zip"

# Файлы и папки для упаковки в финальный zip (внутри подпапки DIST_NAME/)
INCLUDE_PATHS = [
    "src",
    "scripts",
    "requirements.txt",
    "setup.bat",
    "run.bat",
    "README_DIST.md",
]

# Что НЕ включать в zip (внутри указанных папок)
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    "pack_dist.py",                  # сам упаковщик не нужен в дистрибутиве
    "download_embeddable_python.py", # тоже не нужен — это служебный скрипт
    "build_demo.py",                 # сборщик демо тоже не нужен
]


def should_skip(path: Path) -> bool:
    """Проверяет, нужно ли пропустить файл."""
    parts = path.parts
    name = path.name
    for pattern in EXCLUDE_PATTERNS:
        if pattern in parts or name == pattern or name.endswith(pattern):
            return True
    return False


def _build_data_demo_zip() -> Path:
    """Упаковывает data/demo/ в data_demo.zip."""
    if not DEMO_DIR.exists() or not any(DEMO_DIR.iterdir()):
        print("ОШИБКА: data/demo/ пуста. Сначала запустите build_demo.py")
        sys.exit(1)

    out = DIST_DIR / "data_demo.zip"
    print(f"[1/3] Упаковываем data/demo/ в data_demo.zip...")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in DEMO_DIR.iterdir():
            if f.is_file():
                zf.write(f, arcname=f.name)
                print(f"      + {f.name} ({f.stat().st_size / 1e6:.1f} МБ)")
    size_mb = out.stat().st_size / 1e6
    print(f"      Итого: {size_mb:.1f} МБ")
    return out


def _check_prerequisites() -> None:
    """Проверяет наличие всех нужных файлов перед упаковкой."""
    missing = []

    if not DEMO_DIR.exists() or not any(DEMO_DIR.iterdir()):
        missing.append("data/demo/ (запустите: python scripts/build_demo.py)")

    if not PYTHON_EMBED.exists():
        missing.append("dist/python_embed.zip (запустите: python scripts/download_embeddable_python.py)")

    for item in ["setup.bat", "run.bat", "README_DIST.md"]:
        if not (ROOT / item).exists():
            missing.append(item)

    if missing:
        print("ОШИБКА: отсутствуют файлы:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Сборка дистрибутива YandexMUSIC_ANALYTICS ===\n")
    _check_prerequisites()

    # Шаг 1 — упаковка демо-данных
    data_demo_zip = _build_data_demo_zip()

    # Шаг 2 — копирование Python embeddable из подготовленного zip
    print(f"\n[2/3] Используем готовый python_embed.zip "
          f"({PYTHON_EMBED.stat().st_size / 1e6:.1f} МБ)")

    # Шаг 3 — финальный zip
    print(f"\n[3/3] Собираем финальный {DIST_NAME}.zip...")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    files_added = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Код и bat-файлы
        for item in INCLUDE_PATHS:
            src = ROOT / item
            if not src.exists():
                print(f"      ВНИМАНИЕ: {item} не найден, пропускаем")
                continue
            if src.is_file():
                if should_skip(src):
                    continue
                if src.suffix.lower() == ".bat":
                    # Нормализуем line endings для bat-файлов (Windows требует CRLF)
                    content = src.read_bytes()
                    content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                    zf.writestr(f"{DIST_NAME}/{item}", content)
                else:
                    zf.write(src, arcname=f"{DIST_NAME}/{item}")
                files_added += 1
            else:
                for f in src.rglob("*"):
                    if f.is_file() and not should_skip(f):
                        rel = f.relative_to(ROOT)
                        if f.suffix.lower() == ".bat":
                            content = f.read_bytes()
                            content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                            zf.writestr(f"{DIST_NAME}/{rel}", content)
                        else:
                            zf.write(f, arcname=f"{DIST_NAME}/{rel}")
                        files_added += 1

        # data_demo.zip — внутри финального архива
        zf.write(data_demo_zip, arcname=f"{DIST_NAME}/data_demo.zip")
        files_added += 1

        # python_embed.zip — внутри финального архива
        zf.write(PYTHON_EMBED, arcname=f"{DIST_NAME}/python_embed.zip")
        files_added += 1

    size_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nГотово!")
    print(f"  Файл:           {OUTPUT_ZIP}")
    print(f"  Размер:         {size_mb:.1f} МБ")
    print(f"  Файлов внутри:  {files_added}")
    print(f"\nДистрибутив готов к раздаче.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(1)
