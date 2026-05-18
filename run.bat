@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

title YandexMUSIC_ANALYTICS - Запуск

REM Принудительно включаем UTF-8 для stdout/stderr Python
set PYTHONIOENCODING=utf-8

echo.
echo ============================================================
echo   YandexMUSIC_ANALYTICS - Запуск всех 7 задач
echo ============================================================
echo.

REM Проверка что Python установлен
if not exist "python\python.exe" (
    echo ОШИБКА: Python не установлен.
    echo Сначала запустите setup.bat
    pause
    exit /b 1
)

REM Определяем режим (демо или полный) по наличию данных
set MODE=unknown
if exist "data\raw\listens.parquet" (
    set MODE=full
) else if exist "data\raw\flat\500m\listens.parquet" (
    set MODE=full
) else if exist "data\demo\listens.parquet" (
    set MODE=demo
)

if "%MODE%"=="unknown" (
    echo ОШИБКА: данные не найдены ни в data\raw\, ни в data\demo\
    echo Сначала запустите setup.bat
    pause
    exit /b 1
)

echo Режим работы: %MODE%
if "%MODE%"=="demo" (
    echo Данные:       data\demo\ (демо-сэмпл, 5000 пользователей)
    echo Время:        ~5-10 минут
) else (
    echo Данные:       data\raw\ (полный датасет Yambda-500M)
    echo Время:        ~30-60 минут
)
echo.

REM Подготовка БД (создаёт VIEW в analytics.duckdb)
echo Подготовка базы данных...
python\python.exe scripts\prepare_db.py
if errorlevel 1 (
    echo ОШИБКА: prepare_db.py упал.
    pause
    exit /b 1
)
echo.

REM Запуск всех задач
echo Запуск 7 аналитических задач...
echo.
python\python.exe scripts\run_all.py
if errorlevel 1 (
    echo.
    echo ============================================================
    echo   ВО ВРЕМЯ ВЫПОЛНЕНИЯ ВОЗНИКЛИ ОШИБКИ
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ
echo ============================================================
echo.
echo Графики сохранены в: data\results\
echo Открыть папку с результатами? [y/N]
set /p OPEN=
if /i "%OPEN%"=="y" start "" "data\results"
pause
exit /b 0
