"""
Запускает все 7 аналитических задач последовательно.

Использование:
    python scripts/run_all.py            # все задачи
    python scripts/run_all.py 1 3 7      # только указанные номера
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tasks import task1_monthly, task2_sessions, task3_colikes
from src.tasks import task4_diversity, task5_heatmap, task6_colistening, task7_churn

TASKS = {
    1: ("Динамика прослушиваний по месяцам",        task1_monthly.run),
    2: ("Распределение сессий",                     task2_sessions.run),
    3: ("Граф ко-лайков + Louvain",                 task3_colikes.run),
    4: ("Shannon entropy разнообразия",              task4_diversity.run),
    5: ("Тепловая карта час/день недели",            task5_heatmap.run),
    6: ("Граф ко-прослушиваний + Louvain",           task6_colistening.run),
    7: ("Прогноз оттока (RandomForest + LogReg)",    task7_churn.run),
}


def main() -> None:
    if len(sys.argv) > 1:
        try:
            selected = [int(x) for x in sys.argv[1:]]
        except ValueError:
            print("Использование: python run_all.py [номера задач через пробел]")
            sys.exit(1)
    else:
        selected = list(TASKS.keys())

    total_start = time.perf_counter()
    results = {}

    for num in selected:
        if num not in TASKS:
            print(f"\n[ПРОПУСК] Задача {num} не существует")
            continue

        name, fn = TASKS[num]
        print(f"\n{'='*55}")
        print(f"ЗАДАЧА {num}: {name}")
        print("="*55)

        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            results[num] = ("OK", elapsed)
            print(f"  Готово за {elapsed:.1f} с")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results[num] = ("ОШИБКА", elapsed)
            print(f"  ОШИБКА: {e}")
            traceback.print_exc()

    total = time.perf_counter() - total_start
    print(f"\n{'='*55}")
    print("ИТОГ")
    print("="*55)
    for num, (status, elapsed) in results.items():
        name = TASKS[num][0]
        print(f"  Task {num}: [{status:^7}]  {elapsed:>6.1f} с  — {name}")
    print(f"\n  Всего: {total:.1f} с")


if __name__ == "__main__":
    main()
