@echo off
echo ============================================
echo  Face Recognition Service Startup
echo ============================================
echo.

:: Change to the correct project directory
cd /d "%~dp0"

:: Activate the virtual environment (cmd.exe compatible)
call venv\Scripts\activate.bat

echo [1/2] Virtual environment activated.
echo [2/2] Starting face_service.py on http://127.0.0.1:5050
echo.
echo Keep this window open while the app is running.
echo Press Ctrl+C to stop the service.
echo.

python face_service.py
