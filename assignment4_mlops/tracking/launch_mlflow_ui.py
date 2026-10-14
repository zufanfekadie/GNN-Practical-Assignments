"""
One-click runner for MLflow Experiment Tracking UI.
Click the VS Code Play (▷) button to launch the dashboard and automatically open the browser.
"""

import sys
import time
import threading
import webbrowser
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "assignment4_mlops" / "mlflow.db"

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"[ERROR] Database file not found at: {DB_PATH}")
        print("Please run train_tracked.py first to generate tracking experiments.")
        sys.exit(1)

    backend_uri = f"sqlite:///{DB_PATH.as_posix()}"
    
    print("=" * 75)
    print("STARTING MLFLOW EXPERIMENT TRACKING DASHBOARD")
    print(f"Tracking Database: {DB_PATH}")
    print("Dashboard URL:     http://127.0.0.1:5000")
    print("Opening browser automatically...")
    print("Press CTRL+C in this terminal to stop the server.")
    print("=" * 75)

    # Launch browser in a background thread after server starts
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch MLflow UI
    cmd = [
        sys.executable,
        "-m", "mlflow", "ui",
        "--backend-store-uri", backend_uri,
        "--port", "5000",
        "--host", "127.0.0.1"
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[STOPPED] MLflow server stopped.")
