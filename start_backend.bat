@echo off
echo ========================================
echo Thai-English Translator Backend
echo ========================================
echo.

cd /d "%~dp0backend"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies (first run only)...
pip install -q -r requirements.txt

echo.
echo ========================================
echo Server: http://localhost:8001
echo ========================================
echo Models preload at startup (~30s).
echo Expose the port with: ngrok http 8001
echo.

python main.py
