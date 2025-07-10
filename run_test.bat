@echo off
REM Windows batch script to run the AutoCAD Toolkit command-line test
echo AutoCAD Construction Toolkit - Running Test...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment and run test
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Running command-line test...
python -m src.main test

REM Keep window open
echo.
echo Test completed. Press any key to exit...
pause > nul