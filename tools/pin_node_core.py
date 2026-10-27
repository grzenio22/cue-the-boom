"""Pin the standalone ComfyUI node pack to the newest commit that changed the cuesync core.

The pack's installer and metadata name an exact commit, so a history rewrite (or new core work) needs a new pin.
Run this after the last core change or rewrite and before pushing, then commit the two files it updates:

    python tools/pin_node_core.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED = (ROOT / "comfyui-cuesync" / "install.py", ROOT / "comfyui-cuesync" / "pyproject.toml")
PIN = re.compile(r"(github\.com/grzenio22/cue-the-boom@)[0-9a-f]{40}(#subdirectory=cuesync)")


def main():
    shallow = subprocess.check_output(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, text=True).strip()
    if shallow == "true":
        sys.exit("this is a shallow clone: its oldest commit isn't the core's; run this in the full repository")
    sha = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "cuesync/src", "cuesync/pyproject.toml", "cuesync/README.md"],
        cwd=ROOT, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        sys.exit("no commit found for the core")
    for path in PINNED:
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        text, n = PIN.subn(rf"\g<1>{sha}\g<2>", text)
        if n != 1:
            sys.exit(f"{path.relative_to(ROOT)}: expected one pinned core, found {n}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    print(sha)


if __name__ == "__main__":
    main()
