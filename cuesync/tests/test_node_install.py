"""The standalone ComfyUI pack must request its declared core dependency, even over an old install."""

import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "comfyui-cuesync" / "install.py"


@pytest.fixture
def installer(tmp_path, monkeypatch):
    pack = tmp_path / "custom_nodes" / "comfyui-cuesync"
    pack.mkdir(parents=True)
    script = pack / "install.py"
    script.write_text(INSTALL_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    calls = []
    monkeypatch.setattr(subprocess, "check_call", lambda args: calls.append(args))
    return script, calls


@pytest.mark.parametrize("importable", [False, True], ids=["missing-core", "arbitrary-installed-core"])
def test_standalone_install_requests_the_declared_core_with_comfys_python(installer, monkeypatch, importable):
    script, calls = installer
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if importable else None)

    state = runpy.run_path(str(script))

    assert calls == [
        [sys.executable, "-m", "pip", "install", state["CUESYNC"]],
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", state["CUESYNC"]],
    ]


def test_full_repository_uses_its_adjacent_core_without_network(installer, monkeypatch):
    script, calls = installer
    (script.parents[1] / "cuesync" / "src" / "cuesync").mkdir(parents=True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    runpy.run_path(str(script))

    assert calls == []


def test_an_empty_adjacent_directory_does_not_skip_the_core_install(installer):
    script, calls = installer
    (script.parents[1] / "cuesync").mkdir()

    state = runpy.run_path(str(script))

    assert len(calls) == 2 and all(call[-1] == state["CUESYNC"] for call in calls)


@pytest.mark.parametrize("fail_on", [1, 2], ids=["dependency-resolution", "exact-core-replacement"])
def test_dependency_install_failure_is_reported_to_the_node_manager(installer, monkeypatch, fail_on):
    script, _ = installer
    calls = []

    def fail_install(args):
        calls.append(args)
        if len(calls) == fail_on:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "check_call", fail_install)

    with pytest.raises(subprocess.CalledProcessError, match="non-zero exit status 1"):
        runpy.run_path(str(script))
    assert len(calls) == fail_on


def test_same_version_core_is_replaced_without_force_reinstalling_its_dependencies(installer, monkeypatch):
    """Pip skips a remote VCS replacement with the same version unless force-reinstall is requested."""
    script, _ = installer
    installed = {"core": "older-commit-at-version-0.1.0", "dependencies": "satisfied"}
    resolved = []

    def simulate_pip_same_version(args):
        if "--force-reinstall" not in args:
            # Dependency resolution still has to run for a fresh standalone environment.
            assert "--no-deps" not in args
            resolved.append(args[-1])
            return  # Same distribution version: pip may keep the old core here.
        assert resolved == [args[-1]], "Resolve dependencies before forcing the core replacement"
        assert "--no-deps" in args, "Do not reinstall ComfyUI's other packages with the core"
        installed["core"] = args[-1]

    monkeypatch.setattr(subprocess, "check_call", simulate_pip_same_version)

    state = runpy.run_path(str(script))

    assert installed == {"core": state["CUESYNC"], "dependencies": "satisfied"}
