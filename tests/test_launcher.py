# tests/test_launcher.py
import socket
from unittest.mock import patch

from src.simulation import launcher


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_url_uses_the_configured_port():
    assert launcher.simulator_url({"simulator": {"app_port": 8502}}) == "http://localhost:8502"


def test_running_is_false_for_a_port_nobody_listens_on():
    assert launcher.simulator_running({"simulator": {"app_port": _free_port()}}) is False


def test_running_is_true_when_something_is_listening():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        cfg = {"simulator": {"app_port": server.getsockname()[1]}}
        assert launcher.simulator_running(cfg) is True


def test_start_does_nothing_if_it_is_already_running():
    with patch.object(launcher, "simulator_running", return_value=True), \
         patch.object(launcher.subprocess, "Popen") as popen:
        assert launcher.start_simulator({"simulator": {"app_port": 8502}}) is True
        popen.assert_not_called()  # repeated clicks must not pile up processes


def test_start_launches_the_simulation_app_on_the_configured_port():
    states = iter([False, False, True])  # not running, then it comes up
    with patch.object(launcher, "simulator_running", side_effect=lambda cfg: next(states)), \
         patch.object(launcher.subprocess, "Popen") as popen, \
         patch.object(launcher.time, "sleep"):
        assert launcher.start_simulator({"simulator": {"app_port": 9999}}) is True
    args = popen.call_args.args[0]
    assert "streamlit" in args and "9999" in args
    assert any(a.replace("\\", "/").endswith("src/simulation/app.py") for a in args)
