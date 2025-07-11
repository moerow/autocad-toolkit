"""
Launch AI Model Dashboard

Simple script to launch the AI model performance dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install dashboard requirements."""
    requirements_file = Path(__file__).parent / "requirements_dashboard.txt"
    
    if requirements_file.exists():
        print("📦 Installing dashboard requirements...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_file = Path(__file__).parent / "ai_dashboard.py"
    
    if not dashboard_file.exists():
        print("❌ Dashboard file not found!")
        return False
    
    print("🚀 Launching AI Model Dashboard...")
    print("📊 Dashboard will open in your web browser at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_file),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True
    
    return True

def main():
    """Main function."""
    print("🧠 AI Model Dashboard Launcher")
    print("=" * 50)
    
    # Check if we need to install requirements
    try:
        import streamlit
        import plotly
        print("✅ Dashboard dependencies are installed")
    except ImportError:
        print("📦 Installing dashboard dependencies...")
        if not install_requirements():
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install -r requirements_dashboard.txt")
            return 1
    
    # Launch dashboard
    if launch_dashboard():
        print("✅ Dashboard launched successfully!")
        return 0
    else:
        print("❌ Failed to launch dashboard")
        return 1

if __name__ == "__main__":
    exit(main())