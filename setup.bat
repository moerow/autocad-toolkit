@echo off
REM Setup script for AutoCAD Construction Toolkit
echo AutoCAD Construction Toolkit - Setup
echo ====================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)

REM Activate and install dependencies
echo.
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo You can now run:
echo   - run_gui.bat          (Launch the GUI)
echo   - run_test.bat         (Run command-line test)
echo   - run_installation_test.bat (Test installation)
echo.
echo Make sure AutoCAD is running with a drawing open before testing!
echo.
pause