"""Assemble a Hugging Face Space (Docker, CPU basic) from this checkout: the studio with no GPU, a hand of your own
included. Writes the folder (default runs/space) with the Dockerfile, start.sh, a Space README and the parts of the
repository the image needs (no node_modules, no runs/). Then push it to your Space:

    python deploy/space/make_space.py
    cd runs/space && git init && git lfs install && git lfs track "*.mp4" "*.npz" && git add . && git commit -m space \\
      && git remote add origin https://huggingface.co/spaces/<you>/cue-the-boom && git push -u origin main

Set the Space secret CUE_PASSWORD to keep it behind a password.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
README = """---
title: Cue the Boom
emoji: 🔥
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Film your hand. Make the noise. Throw fire.
---

# Cue the Boom, with no GPU

Pick **Your hand: cupped palm**, or **Use your own clip**, then **Your hand**: tap your palm or fingertips, drag to aim,
make the noise. The fire is an element rendered ahead of time with LTX-2.5 + the ainvfx-fluid IC-LoRA, carried on your
tracked hand, timed and sized by your voice. A shot takes a minute or two on this CPU.

The full app, with renders of its own on a GPU: https://github.com/grzenio22/cue-the-boom
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "runs" / "space"))
    out = Path(ap.parse_args().out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    skip = shutil.ignore_patterns("node_modules", ".next", ".venv*", "__pycache__", "*.pyc", "tsconfig.tsbuildinfo",
                                  "runs", ".pytest_cache", "tests")
    for part in ("cuesync", "samples", "workflows", "web"):
        shutil.copytree(ROOT / part, out / part, ignore=skip)
    for name in ("Dockerfile", "start.sh"):
        shutil.copyfile(HERE / name, out / name)
    (out / "README.md").write_text(README, encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{out}: {size:.0f} MB, ready to push to a Docker Space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
