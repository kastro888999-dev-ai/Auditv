@echo off
REM Instala AuditV en esta carpeta (modo portable, Windows).
REM Requiere: Python 3.10+ en el PATH y, opcionalmente, ffmpeg y Ollama.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 echo ERROR: Python no está en el PATH & pause & exit /b 1

echo -^> Python: 
python --version
python -m venv venv
if not exist venv\Scripts\pip.exe echo ERROR: no se pudo crear el venv & pause & exit /b 1

venv\Scripts\pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
venv\Scripts\pip install "gradio>=6"

echo.
where ffmpeg >nul 2>nul && (echo -^> ffmpeg: OK) || echo -^> ffmpeg: NO instalado ^(winget install ffmpeg o descárgalo y añádelo al PATH^)
if exist venv\Scripts\yt-dlp.exe (echo -^> yt-dlp: OK ^) o (echo -^> yt-dlp: NO instalado)
where ollama >nul 2>nul && (echo -^> ollama: OK) || echo -^> ollama: NO instalado ^(solo para el análisis con IA^)
echo.
echo Listo. Para arrancar la GUI: doble clic en iniciar_gui.bat
pause