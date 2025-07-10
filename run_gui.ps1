# PowerShell script to run the AutoCAD Toolkit GUI
Write-Host "AutoCAD Construction Toolkit - Starting GUI..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run GUI using virtual environment
Write-Host "Running GUI..." -ForegroundColor Yellow
try {
    & "venv\Scripts\python.exe" -m src.main gui
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}