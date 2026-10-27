"""Setting up the hand tracker: a failed attempt is started afresh, and a Python its wheels don't fit is refused clearly."""

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cuesync.cli as cli
from cuesync.bending import READY_MARK


def checkout(tmp_path):
    (tmp_path / "cuesync").mkdir()
    (tmp_path / "cuesync" / "pyproject.toml").write_text("[project]\nname = 'cuesync'\n")
    return tmp_path / "cuesync" / ".venv-track"


def install(monkeypatch, tmp_path, uv):
    calls = []

    def run(cmd, check=False, **kw):
        calls.append([str(c) for c in cmd])
        if "venv" in cmd:  # the new environment and its Python
            py = tmp_path / "cuesync" / ".venv-track" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: uv)
    monkeypatch.setattr(subprocess, "run", run)
    return calls, lambda: cli.cmd_hands(SimpleNamespace(check=False, install=True, mask=False))


def test_an_unfinished_tracker_install_is_started_afresh(tmp_path, monkeypatch):
    venv = checkout(tmp_path)
    (venv / "lib").mkdir(parents=True)
    (venv / "lib" / "half-installed").write_text("")  # an earlier attempt, on an unsuitable Python, never marked ready
    calls, run = install(monkeypatch, tmp_path, uv="uv")
    assert run() == 0
    assert not (venv / "lib").exists()  # not reused
    assert calls[0][:2] == ["uv", "venv"] and calls[0][-2:] == ["--python", "3.12"]
    assert (venv / READY_MARK).exists()


def test_a_ready_tracker_is_reinstalled_in_place(tmp_path, monkeypatch):
    venv = checkout(tmp_path)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    py.parent.mkdir(parents=True)
    py.write_text("")
    (venv / READY_MARK).write_text("ok\n")
    calls, run = install(monkeypatch, tmp_path, uv="uv")
    assert run() == 0
    assert py.exists() and not any("venv" in c for c in calls)  # its environment kept: only the packages again


def test_without_uv_an_unsuitable_python_is_refused_before_anything_is_made(tmp_path, monkeypatch):
    venv = checkout(tmp_path)
    calls, run = install(monkeypatch, tmp_path, uv=None)
    monkeypatch.setattr(cli, "TRACKER_PYTHONS", ((3, 99), (3, 99)))  # as if this Python were too new for the wheels
    with pytest.raises(SystemExit, match=r"needs Python 3\.99 to 3\.99 .* or uv"):
        run()
    assert calls == [] and not venv.exists()


def test_without_uv_a_suitable_python_makes_the_environment_itself(tmp_path, monkeypatch):
    checkout(tmp_path)
    calls, run = install(monkeypatch, tmp_path, uv=None)
    monkeypatch.setattr(cli, "TRACKER_PYTHONS", (sys.version_info[:2], sys.version_info[:2]))
    assert run() == 0
    assert calls[0][:3] == [sys.executable, "-m", "venv"]
