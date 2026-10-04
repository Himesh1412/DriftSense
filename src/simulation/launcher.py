# src/simulation/launcher.py
"""
Lets the main dashboard link to — and if needed start — the standalone
Simulation Engine app, which runs as its own Streamlit server on its own port.
"""
import os
import socket
import subprocess
import sys
import time

from src.config_loader import ROOT


def simulator_url(cfg: dict) -> str:
    return f"http://localhost:{cfg['simulator']['app_port']}"


def simulator_running(cfg: dict, timeout: float = 0.3) -> bool:
    """True if something is already listening on the simulator's port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex(("127.0.0.1", int(cfg["simulator"]["app_port"]))) == 0


def start_simulator(cfg: dict, wait_seconds: float = 30) -> bool:
    """
    Start the Simulation Engine as a detached local process (so it keeps running
    if the dashboard is closed) and wait until its port answers. Does nothing if it
    is already running, so repeated clicks can't pile up processes.
    """
    if simulator_running(cfg):
        return True
    args = [sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "simulation" / "app.py"),
            "--server.port", str(cfg["simulator"]["app_port"]), "--server.headless", "true"]
    detach = ({"creationflags": 0x00000008 | 0x00000200}  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
              if os.name == "nt" else {"start_new_session": True})
    subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **detach)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if simulator_running(cfg):
            return True
        time.sleep(0.5)
    return False
