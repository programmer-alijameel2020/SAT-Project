# main.py
import subprocess
import sys
import os

if __name__ == "__main__":
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    subprocess.run([sys.executable, "run.py"], cwd=backend_dir)