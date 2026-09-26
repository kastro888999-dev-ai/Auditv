@echo off
REM Arranca la GUI de AuditV en Windows (doble clic).
REM chcp 65001 + PYTHONUTF8=1: sin UTF-8, el log se corta con acentos y emojis.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Primero ejecuta instalar.bat
    pause
    exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
start "" cmd /k "venv\Scripts\python tools\auditv_gui.py"
