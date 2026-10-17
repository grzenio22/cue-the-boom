"""ComfyUI-Manager and comfy-cli run this after installing the node pack on its own: it installs cuesync, the library
the nodes are built on, unless it is importable already or sits next to the pack in a checkout of the repository."""

import importlib.util
import subprocess
import sys
from pathlib import Path

CUESYNC = "cuesync @ git+https://github.com/grzenio22/cue-the-boom#subdirectory=cuesync"

if importlib.util.find_spec("cuesync") is None and not (Path(__file__).resolve().parents[1] / "cuesync" / "src" / "cuesync").is_dir():
    subprocess.check_call([sys.executable, "-m", "pip", "install", CUESYNC])
