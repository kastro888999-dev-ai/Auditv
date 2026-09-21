@echo off
REM Arranca la GUI de AuditV en Windows (doble clic).
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Primero ejecuta instalar.bat
    pause
    exit /b 1
)
start "" cmd /k "venv\Scripts\python tools\auditv_gui.py"