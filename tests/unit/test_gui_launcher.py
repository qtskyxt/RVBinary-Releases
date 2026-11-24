from typer.testing import CliRunner

from rvbinary.cli import app


def test_gui_missing_optional_dependency(monkeypatch):
    import importlib.util

    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: None if name == "streamlit" else original(name),
    )
    result = CliRunner().invoke(app, ["gui"])
    assert result.exit_code != 0
    assert "rvbinary[gui]" in result.output
    assert CliRunner().invoke(app, ["--help"]).exit_code == 0


def test_gui_launch_command_and_exit_status(monkeypatch):
    import importlib.util
    import subprocess
    from pathlib import Path

    seen = []

    def run(command, **kwargs):
        seen.extend(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    result = CliRunner().invoke(app, ["gui", "--port", "8502"])
    assert result.exit_code == 7
    assert "127.0.0.1:8502" in result.output
    assert seen[1:4] == ["-m", "streamlit", "run"]
    assert Path(seen[4]).name == "app.py"
    assert "--server.address=127.0.0.1" in seen
    assert "--server.port=8502" in seen


def test_gui_invalid_port():
    for port in ("0", "65536"):
        result = CliRunner().invoke(app, ["gui", "--port", port])
        assert result.exit_code != 0
