"""Ships the repository's effects/ inside the package, so `pip install "git+...#subdirectory=cuesync"` has them.

A wheel built straight from the checkout copies them in at build time; an sdist carries a copy in
src/cuesync/effect_files (not in git), so a wheel built from the sdist has them too."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

EFFECTS = Path(__file__).resolve().parent.parent / "effects"


def copy_effects(dst: Path) -> None:
    if EFFECTS.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in EFFECTS.glob("*.json"):
            shutil.copy2(f, dst / f.name)


class BuildPy(build_py):
    def run(self):
        super().run()
        copy_effects(Path(self.build_lib) / "cuesync" / "effect_files")


class Sdist(sdist):
    def run(self):
        copy_effects(Path(__file__).resolve().parent / "src" / "cuesync" / "effect_files")
        super().run()


setup(cmdclass={"build_py": BuildPy, "sdist": Sdist})
