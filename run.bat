@echo off
REM ======================================================
REM  SignBridge AI — Launcher (pakai venv bersih)
REM  Jalankan file ini dengan double-click atau dari terminal
REM ======================================================

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment belum dibuat!
    echo Jalankan dulu di terminal:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [INFO] Memulai SignBridge AI dengan GPU/CPU terbaik yang tersedia...
.venv\Scripts\python.exe realtime_inference.py %*

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Script keluar dengan error. Periksa pesan di atas.
    pause
)
