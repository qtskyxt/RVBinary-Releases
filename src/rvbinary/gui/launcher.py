"""Launch the optional browser UI using the active Python environment."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def launch_gui(port: int = 8501) -> int:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in [1, 65535]")
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError("Install the browser interface with: pip install 'rvbinary[gui]'")
    page = Path(__file__).with_name("app.py")
    print(f"RVBinary GUI: http://127.0.0.1:{port}", flush=True)
    try:
        return subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(page),
            "--server.address=127.0.0.1", f"--server.port={port}",
            "--server.headless=true", "--browser.gatherUsageStats=false",
        ], check=False).returncode
    except KeyboardInterrupt:
        return 130
