#!/usr/bin/env python3
"""Test script to verify installation and basic functionality."""
import sys
import os
import importlib.util

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    required_modules = [
        'pyautocad',
        'tkinter',
        'pathlib',
        'logging',
        'threading',
        'math',
        'uuid'
    ]
    
    failed = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError as e:
            print(f"  [FAIL] {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    else:
        print("\nAll required modules imported successfully!")
        return True

def test_project_structure():
    """Test that the project structure is correct."""
    print("\nTesting project structure...")
    
    required_files = [
        'src/main.py',
        'src/application/services/dimension_service.py',
        'src/infrastructure/autocad/connection.py',
        'src/presentation/gui/main_window.py',
        'requirements.txt'
    ]
    
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path}")
            missing.append(file_path)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    else:
        print("\nProject structure is correct!")
        return True

def test_internal_imports():
    """Test that internal modules can be imported."""
    print("\nTesting internal imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from infrastructure.autocad.connection import AutoCADConnection
        print("  [OK] AutoCADConnection")
        
        from application.services.dimension_service import DimensionService
        print("  [OK] DimensionService")
        
        from presentation.gui.main_window import MainWindow
        print("  [OK] MainWindow")
        
        print("\nAll internal imports successful!")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Internal import error: {e}")
        return False

def main():
    """Run all tests."""
    print("AutoCAD Construction Toolkit - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test project structure
    if not test_project_structure():
        all_passed = False
    
    # Test internal imports
    if not test_internal_imports():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("SUCCESS: All tests passed! The toolkit is ready to use.")
        print("\nTo test with AutoCAD:")
        print("  1. Open AutoCAD with a drawing containing lines")
        print("  2. Run: python -m src.main gui")
        print("  3. Click 'Connect to AutoCAD' then 'Add Dimensions'")
        print("\nAlternatively, for command-line test:")
        print("  python -m src.main test")
    else:
        print("FAILED: Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())