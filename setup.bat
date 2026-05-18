@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

REM Установщик YandexMUSIC_ANALYTICS
REM Запускается двойным кликом или из cmd.exe

title YandexMUSIC_ANALYTICS - Установка

echo.
echo ============================================================
echo   YandexMUSIC_ANALYTICS - Установка
echo ============================================================
echo.
echo Выберите режим установки:
echo.
echo   [1] Демо-режим (~180 МБ, оффлайн, 5000 пользователей)
echo       Подходит для быстрого знакомства с проектом.
echo       Все 7 задач работают, task6 - без UMAP-валидации.
echo.
echo   [2] Полный режим (~15 ГБ скачивания, нужен интернет)
echo       Полный датасет Yambda-500M + аудио-эмбеддинги.
echo       Все 7 задач работают полностью.
echo.

:choice
set /p MODE="Ваш выбор [1/2]: "
if "%MODE%"=="1" goto :install_demo
if "%MODE%"=="2" goto :install_full
echo Неверный ввод. Введите 1 или 2.
goto :choice

:install_demo
echo.
echo === Установка в демо-режиме ===
echo.
call :setup_python
if errorlevel 1 goto :error
call :install_deps
if errorlevel 1 goto :error
call :unpack_demo
if errorlevel 1 goto :error
goto :success

:install_full
echo.
echo === Установка в полном режиме ===
echo.
echo ВНИМАНИЕ: будут скачаны ~15 ГБ данных с HuggingFace.
echo Скачивание может занять от 30 минут до нескольких часов.
echo.
set /p CONFIRM="Продолжить? [y/N]: "
if /i not "%CONFIRM%"=="y" (
    echo Установка отменена.
    pause
    exit /b 0
)
call :setup_python
if errorlevel 1 goto :error
call :install_deps
if errorlevel 1 goto :error
call :download_full
if errorlevel 1 goto :error
goto :success

REM ============================================================
REM Функции
REM ============================================================

:setup_python
echo [1/4] Распаковка Python embeddable...
if not exist "python_embed.zip" (
    echo ОШИБКА: python_embed.zip не найден.
    exit /b 1
)
if exist "python\" rmdir /s /q python
mkdir python
powershell -Command "Expand-Archive -Path 'python_embed.zip' -DestinationPath 'python' -Force"
if not exist "python\python.exe" (
    echo ОШИБКА: распаковка Python не удалась.
    exit /b 1
)
echo       Python распакован.
echo.

REM Включаем поддержку pip в embeddable Python
echo [2/4] Подготовка pip...
echo import site>> python\python311._pth
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'python\get-pip.py'"
python\python.exe python\get-pip.py --no-warn-script-location
if errorlevel 1 (
    echo ОШИБКА: установка pip не удалась.
    exit /b 1
)
echo       pip установлен.
echo.
exit /b 0

:install_deps
echo [3/4] Установка зависимостей из requirements.txt...
python\python.exe -m pip install -r requirements.txt --no-warn-script-location
if errorlevel 1 (
    echo ОШИБКА: установка зависимостей не удалась.
    exit /b 1
)
echo       Зависимости установлены.
echo.
exit /b 0

:unpack_demo
echo [4/4] Распаковка демо-данных...
if not exist "data_demo.zip" (
    echo ОШИБКА: data_demo.zip не найден.
    exit /b 1
)
if not exist "data\demo" mkdir data\demo
powershell -Command "Expand-Archive -Path 'data_demo.zip' -DestinationPath 'data\demo' -Force"
echo       Демо-данные распакованы в data\demo\
echo.
exit /b 0

:download_full
echo [4/4] Скачивание полных данных Yambda-500M с эмбеддингами...
echo       Это займёт от 30 минут до нескольких часов.
python\python.exe scripts\download_data.py --with-embeddings
if errorlevel 1 (
    echo ОШИБКА: скачивание данных не удалось.
    exit /b 1
)
echo       Данные скачаны в data\raw\
echo.
exit /b 0

:success
echo.
echo ============================================================
echo   УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО
echo ============================================================
echo.
echo Для запуска всех 7 задач выполните: run.bat
echo.
pause
exit /b 0

:error
echo.
echo ============================================================
echo   ОШИБКА УСТАНОВКИ
echo ============================================================
echo.
echo Установка прервана. См. сообщения выше для диагностики.
echo.
pause
exit /b 1
