"""Setting up the hand tracker: a failed attempt is started afresh, and a Python its wheels don't fit is refused clearly."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
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
    monkeypatch.setattr(shutil, "which", lambda name, path=None: uv)
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


def test_without_uv_an_unsuitable_python_is_refused_before_anything_is_made_or_removed(tmp_path, monkeypatch):
    venv = checkout(tmp_path)
    (venv / "lib").mkdir(parents=True)  # an earlier, unfinished attempt: kept, since nothing can replace it yet
    calls, run = install(monkeypatch, tmp_path, uv=None)
    monkeypatch.setattr(cli, "TRACKER_PYTHONS", ((3, 99), (3, 99)))  # as if this Python were too new for the wheels
    with pytest.raises(SystemExit, match=r"needs Python 3\.99 to 3\.99 .* -m pip install uv"):
        run()
    assert calls == [] and (venv / "lib").exists()


def test_uv_installed_into_the_studios_own_environment_is_found(tmp_path, monkeypatch):
    checkout(tmp_path)
    calls, run = install(monkeypatch, tmp_path, uv=None)
    here = str(Path(sys.executable).parent)
    monkeypatch.setattr(shutil, "which", lambda name, path=None: "venv-uv" if path == here else None)
    monkeypatch.setattr(cli, "TRACKER_PYTHONS", ((3, 99), (3, 99)))  # only uv could make it
    assert run() == 0
    assert calls[0][:2] == ["venv-uv", "venv"]


def test_a_locked_unfinished_install_says_what_to_do(tmp_path, monkeypatch):
    venv = checkout(tmp_path)
    (venv / "lib").mkdir(parents=True)
    calls, run = install(monkeypatch, tmp_path, uv="uv")

    def locked(path, *a, **k):
        raise PermissionError(32, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(shutil, "rmtree", locked)
    with pytest.raises(SystemExit, match="another install may still be running"):
        run()
    assert calls == []


def test_without_uv_a_suitable_python_makes_the_environment_itself(tmp_path, monkeypatch):
    checkout(tmp_path)
    calls, run = install(monkeypatch, tmp_path, uv=None)
    monkeypatch.setattr(cli, "TRACKER_PYTHONS", (sys.version_info[:2], sys.version_info[:2]))
    assert run() == 0
    assert calls[0][:3] == [sys.executable, "-m", "venv"]
