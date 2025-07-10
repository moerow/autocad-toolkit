@echo off
REM Windows batch script to run the AutoCAD Toolkit GUI
echo AutoCAD Construction Toolkit - Starting GUI...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment and run GUI
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Running GUI...
python -m src.main gui

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause > nul
)