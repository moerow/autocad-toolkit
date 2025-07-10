# AutoCAD Construction Toolkit - Virtual Environment Setup

This project now includes a fully functional Python virtual environment with all dependencies pre-installed.

## 🚀 Quick Start

### Option 1: Easy Setup (Recommended)
1. **Double-click `setup.bat`** to automatically create the virtual environment and install all dependencies
2. **Double-click `run_gui.bat`** to launch the GUI application
3. **Done!** No manual Python configuration needed.

### Option 2: Manual Setup
If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m src.main gui
```

## 📁 Available Scripts

### Windows Batch Files (.bat)
- **`setup.bat`** - Complete setup: creates venv + installs dependencies
- **`run_gui.bat`** - Launch the GUI application  
- **`run_test.bat`** - Run command-line test with AutoCAD
- **`run_installation_test.bat`** - Test that everything is installed correctly

### PowerShell Scripts (.ps1)
- **`setup.ps1`** - PowerShell version of setup
- **`run_gui.ps1`** - PowerShell version of GUI launcher

## 🔧 Virtual Environment Details

### What's Included
- **Python Virtual Environment** in `venv/` directory
- **All Dependencies** pre-installed:
  - `pyautocad` - AutoCAD COM interface
  - `customtkinter` - Modern GUI framework
  - `pywin32` - Windows COM support
  - `click` - CLI framework
  - `pandas` - Data handling
  - `pytest` - Testing framework
  - And more...

### Directory Structure
```
autocad_toolkit/
├── venv/                    # Virtual environment
│   ├── Scripts/             # Windows executables
│   │   ├── python.exe       # Python interpreter
│   │   ├── pip.exe          # Package installer
│   │   └── activate.bat     # Activation script
│   └── Lib/                 # Installed packages
├── src/                     # Source code
├── setup.bat                # Easy setup script
├── run_gui.bat              # GUI launcher
└── requirements.txt         # Dependencies list
```

## 🎯 Usage Instructions

### 1. First Time Setup
```bash
# Run setup (creates venv + installs packages)
setup.bat

# Test installation
run_installation_test.bat
```

### 2. Running the Application
```bash
# Launch GUI (easiest way)
run_gui.bat

# Or run command-line test
run_test.bat
```

### 3. Using with AutoCAD
1. **Open AutoCAD** with a drawing containing lines
2. **Run the GUI**: `run_gui.bat`
3. **Click "Connect to AutoCAD"**
4. **Set layer filter** (optional, e.g., "WALL")
5. **Click "Add Dimensions"**
6. **Watch dimensions appear automatically!**

## 🛠️ Manual Virtual Environment Commands

If you need to work with the virtual environment manually:

### Activate Virtual Environment
```bash
# Windows Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1

# Git Bash / WSL
source venv/Scripts/activate
```

### Install New Packages
```bash
# Make sure virtual environment is activated first
venv\Scripts\python.exe -m pip install package_name
```

### Check Installed Packages
```bash
venv\Scripts\python.exe -m pip list
```

### Run Python Scripts
```bash
# Run with virtual environment Python
venv\Scripts\python.exe -m src.main gui
```

## 🧪 Testing

### Installation Test
```bash
run_installation_test.bat
```

### AutoCAD Integration Test
```bash
run_test.bat
```

### GUI Test
```bash
run_gui.bat
```

## 🔍 Troubleshooting

### Virtual Environment Issues
- **If `setup.bat` fails**: Make sure Python is installed and in PATH
- **If activation fails**: Try running from Command Prompt as Administrator
- **If packages fail to install**: Check your internet connection

### AutoCAD Connection Issues
- **"Failed to connect"**: Make sure AutoCAD is running
- **"No drawing open"**: Open a drawing in AutoCAD first
- **"COM errors"**: Try running as Administrator

### Permission Issues
- **"Access denied"**: Run Command Prompt as Administrator
- **"Script execution disabled"**: For PowerShell, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## 📋 System Requirements

- **Windows 10/11**
- **Python 3.8+** (3.10 recommended)
- **AutoCAD 2018+** (any version with COM support)
- **4GB RAM minimum**
- **Administrator privileges** (for AutoCAD COM access)

## 🎉 Success Indicators

When everything is working correctly:
- `setup.bat` completes without errors
- `run_installation_test.bat` shows all "[OK]" messages
- `run_gui.bat` opens the GUI window
- GUI shows "Status: Connected" when AutoCAD is running
- Dimensions appear automatically when you click "Add Dimensions"

That's it! The virtual environment is fully configured and ready to use.