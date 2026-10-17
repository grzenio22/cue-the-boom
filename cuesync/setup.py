"""Ships the repository's effects/ inside the package, so `pip install "git+...#subdirectory=cuesync"` has them."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        src = Path(__file__).resolve().parent.parent / "effects"
        if src.is_dir():
            dst = Path(self.build_lib) / "cuesync" / "effect_files"
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.json"):
                shutil.copy2(f, dst / f.name)


setup(cmdclass={"build_py": BuildPy})
