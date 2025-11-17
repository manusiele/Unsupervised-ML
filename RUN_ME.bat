@echo off
echo ============================================================
echo   Plant Disease Detection - Quick Start
echo ============================================================
echo.
echo Checking for running servers on port 8000...
netstat -ano | findstr :8000 > nul
if %errorlevel% == 0 (
    echo.
    echo Port 8000 is in use. Starting on port 8080 instead...
    echo.
    python main_simple.py --port 8080
) else (
    echo.
    echo Port 8000 is free. Starting server...
    echo.
    python main_simple.py
)
pause
