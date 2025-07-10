@echo off
REM Windows batch script to test the installation
echo AutoCAD Construction Toolkit - Installation Test...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment and run installation test
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Running installation test...
python test_installation.py

REM Keep window open
echo.
echo Installation test completed. Press any key to exit...
pause > nul