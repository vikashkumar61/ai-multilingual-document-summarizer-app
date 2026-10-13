@echo off
setlocal
cd /d "%~dp0"
title AI Multilingual Document Summarizer - Localhost

echo ======================================================
echo   AI MULTILINGUAL DOCUMENT SUMMARIZER
echo ======================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python Launcher not found.
  echo Install Python 3.12 from https://www.python.org/downloads/release/python-31210/
  echo During installation, enable the Python Launcher and Add Python to PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating Python 3.12 virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 goto python_error
) else (
  echo [1/4] Virtual environment already exists.
)

echo.
echo [2/4] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_error

echo.
echo [3/4] Installing/checking project packages. First install may take several minutes...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_error

echo.
echo [4/4] Starting Streamlit. Keep this window open.
echo When ready, open http://localhost:8501 in your browser.
echo The first Transformer summary may download a large model and take time.
echo Press Ctrl+C in this window to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto launch_error
goto end

:python_error
echo.
echo Could not create the environment. Confirm Python 3.12 is installed by running: py -0p
pause
exit /b 1

:install_error
echo.
echo Package installation failed. Check internet access and the error above.
pause
exit /b 1

:launch_error
echo.
echo Streamlit stopped with an error. Check the terminal output above.
pause
exit /b 1

:end
endlocal
