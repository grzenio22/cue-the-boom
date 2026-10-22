"""Run the Space assembler on a small, real source tree without local build caches."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / "source"
    for part in ("cuesync", "effects", "deploy/space"):
        shutil.copytree(
            ROOT / part, source / part,
            ignore=shutil.ignore_patterns(
                ".venv*", "__pycache__", "*.pyc", "build", "dist", "*.egg-info",
                "effect_files", ".pytest_cache", ".ruff_cache", "tests",
            ),
        )
    for part in ("samples", "workflows", "web", "build"):
        (source / part).mkdir()
    for name in ("LICENSE", "NOTICE.md", "samples/CREDITS.md", "workflows/LICENSE-LTX-2.x", "build/model-files.md"):
        shutil.copy2(ROOT / name, source / name)
    return source


def assemble(source, out):
    return subprocess.run(
        [sys.executable, str(source / "deploy/space/make_space.py"), "--out", str(out)],
        capture_output=True, text=True,
    )


def test_assembled_space_loads_effects_without_checkout_or_build_cache(source_tree, tmp_path):
    # Omitting effects/ makes this fail even when the developer's checkout can load them.
    out = tmp_path / "space"
    result = assemble(source_tree, out)
    assert result.returncode == 0, result.stderr
    environment = os.environ.copy()
    environment.pop("CUESYNC_EFFECTS", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", "\n".join([
            "import sys",
            f"sys.path.insert(0, {str(out / 'cuesync/src')!r})",
            "from cuesync.effects import load_effect",
            "ids = ('blow_out', 'fire_breath', 'fire_burst', 'fireball', 'flame_plume', 'smoke_puff', 'steam_jet', 'wick_flare')",
            "assert all(load_effect(name).id == name for name in ids)",
        ])],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_assembled_space_preserves_license_notices(source_tree, tmp_path):
    # Missing root notices changes the license information delivered with the Space.
    out = tmp_path / "space"
    result = assemble(source_tree, out)
    assert result.returncode == 0, result.stderr
    for name in ("LICENSE", "NOTICE.md", "cuesync/LICENSE", "samples/CREDITS.md",
                 "workflows/LICENSE-LTX-2.x", "build/model-files.md"):
        assert (out / name).is_file(), f"Missing redistribution notice: {name}"
        assert (out / name).read_bytes() == (source_tree / name).read_bytes()


def test_assembled_space_excludes_stale_packaging_outputs(source_tree, tmp_path):
    # An old build or sdist copy must not hide missing or outdated effect definitions.
    stale_paths = (
        "cuesync/build/lib/cuesync/effect_files/stale.json",
        "cuesync/dist/stale.whl",
        "cuesync/src/cuesync.egg-info/PKG-INFO",
        "cuesync/src/cuesync/effect_files/stale.json",
    )
    for name in stale_paths:
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")
    out = tmp_path / "space"
    result = assemble(source_tree, out)
    assert result.returncode == 0, result.stderr
    assert not [name for name in stale_paths if (out / name).exists()]


def test_assembler_preserves_existing_output(source_tree, tmp_path):
    # A mistaken --out argument must not erase an existing checkout or build.
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")
    result = assemble(source_tree, out)
    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "keep me"


def test_assembled_space_excludes_private_files_but_keeps_public_examples(source_tree, tmp_path):
    private = (
        "web/.impeccable/hook.cache.json", "cuesync/.git/config",
        "web/components/.claude/settings.json", "samples/.codex/session.json",
        "workflows/.superpowers/notes.md", "web/.env", "web/.env.local",
        "cuesync/src/.env.production", "web/AGENTS.md", "effects/AGENTS.md",
        "web/out/index.html",
    )
    public = (
        "web/.env.example", "cuesync/src/.env.example",
        "samples/out/asset.json", "web/public/out/asset.json",
        "web/public/.well-known/config.json",
    )
    for name in private + public:
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture", encoding="utf-8")
    out = tmp_path / "space"
    result = assemble(source_tree, out)
    assert result.returncode == 0, result.stderr
    assert not [name for name in private if (out / name).exists()]
    for name in public:
        assert (out / name).read_bytes() == (source_tree / name).read_bytes()


def test_assembler_rejects_output_inside_copied_source(source_tree):
    # Copying a directory into itself would recurse or alter the source tree.
    out = source_tree / "effects/generated-space"
    result = assemble(source_tree, out)
    assert result.returncode != 0
    assert not out.exists()
