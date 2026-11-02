"""Install this standalone node pack's declared cuesync dependency with ComfyUI's Python.

A full repository checkout uses the bundled library next to the pack instead.
"""

import subprocess
import sys
from pathlib import Path

CUESYNC = "cuesync @ git+https://github.com/grzenio22/cue-the-boom@0b26a84c3e73c0e9862f5b7620022f70e1728c35#subdirectory=cuesync"

# An importable cuesync may belong to an older revision with the same package
# version. Resolve its dependencies normally, then replace only the core: pip can
# otherwise retain a same-version VCS install even when the requested commit differs.
if not (Path(__file__).resolve().parents[1] / "cuesync" / "src" / "cuesync").is_dir():
    subprocess.check_call([sys.executable, "-m", "pip", "install", CUESYNC])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", CUESYNC])
