# scripts/start_demo.py
"""
One command for presentations: starts the dashboard AND the Simulation Engine and opens both
in the browser.

    python -m scripts.start_demo            # start both and open them
    python -m scripts.start_demo --refresh  # also run one daily cycle first (needs internet)
    python -m scripts.start_demo --stop     # shut both down afterwards
"""
import os
import socket
import subprocess
import sys
import time
import webbrowser

from src.config_loader import ROOT, load_config
from src.simulation.launcher import start_simulator


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def start_dashboard(port: int, wait_seconds: float = 40) -> bool:
    if port_open(port):
        return True
    args = [sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "dashboard" / "app.py"),
            "--server.port", str(port), "--server.headless", "true"]
    detach = ({"creationflags": 0x00000008 | 0x00000200} if os.name == "nt" else {"start_new_session": True})
    subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **detach)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def stop_port(port: int):
    """Stop whatever is listening on this port (Windows: netstat + taskkill)."""
    if os.name != "nt":
        subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False)
        return
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    pids = {line.split()[-1] for line in out.splitlines()
            if f":{port} " in line and "LISTENING" in line}
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)


if __name__ == "__main__":
    cfg = load_config()
    dash_port, sim_port = cfg["simulator"]["dashboard_port"], cfg["simulator"]["app_port"]

    if "--stop" in sys.argv:
        for p in (dash_port, sim_port):
            stop_port(p)
        print("Dashboard and Simulation Engine stopped.")
        sys.exit(0)

    if "--refresh" in sys.argv:
        print("Running one daily cycle (fetching the latest prices)...", flush=True)
        try:
            from scripts.run_daily_cycle import run_daily_cycle
            run_daily_cycle()
        except Exception as e:  # no internet / Yahoo down must not block the demo
            print(f"  skipped - could not refresh ({e}). Showing the data already saved.")

    print("Starting the Simulation Engine...", flush=True)
    sim_ok = start_simulator(cfg)
    print("Starting the dashboard...", flush=True)
    dash_ok = start_dashboard(dash_port)

    print(f"  Dashboard:         http://localhost:{dash_port}  {'ready' if dash_ok else 'DID NOT START'}")
    print(f"  Simulation Engine: http://localhost:{sim_port}  {'ready' if sim_ok else 'DID NOT START'}")
    if dash_ok:
        webbrowser.open(f"http://localhost:{dash_port}")
    if sim_ok:
        webbrowser.open(f"http://localhost:{sim_port}")
    print("\nWhen finished:  python -m scripts.start_demo --stop")
