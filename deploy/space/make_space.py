"""Assemble a Hugging Face Space (Docker, CPU basic) from this checkout: the studio with no GPU, a hand of your own
included. Writes the folder (default runs/space) with the Dockerfile, start.sh, a Space README and the parts of the
repository the image needs (no node_modules, no runs/). Then push it to your Space:

    python deploy/space/make_space.py
    cd runs/space && git init && git lfs install && git lfs track "*.mp4" "*.npz" && git add . && git commit -m space \\
      && git remote add origin https://huggingface.co/spaces/<you>/cue-the-boom && git push -u origin main

The output must be a new directory; existing content is never removed. Set the Space secret CUE_PASSWORD
to keep it behind a password. See deploy/space/README.md for local preparation and smoke checks.
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
short_description: Generate the effect once. Perform it again with your voice.
---

# Cue the Boom, with no GPU

**Generate the effect once. Perform it again with your voice.** At the top, **One render. Three performances.** plays the
author's palm and one LTX-2.5 fire render in three performances using their own recorded voice;
two rhythms are edited from their recorded pf and BOOM sounds.

Pick one of the author's sample hands and a noise (or make your own), then **Perform it again** with another one. The
fire and the explosion's fireball were rendered ahead of time with LTX-2.5 and the ainvfx-fluid IC-LoRA; lightning,
water, air, earth and the explosion's blast are made on this CPU from the hand and the voice. **Use your own clip** to film or upload
your hand: tap your palm or fingertips, drag to aim, make the noise. Processing time depends on your clip, effect
and the available CPU.

The full app, with renders of its own on a GPU: https://github.com/grzenio22/cue-the-boom

See [LICENSE](LICENSE), [NOTICE.md](NOTICE.md), and [sample credits](samples/CREDITS.md) for redistribution terms.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "runs" / "space"))
    out = Path(ap.parse_args().out).expanduser().resolve()
    parts = ("cuesync", "effects", "samples", "workflows", "web")
    if out == ROOT or out in ROOT.parents or any(out.is_relative_to(ROOT / part) for part in parts):
        ap.error("--out must be outside the source directories being copied")
    if out.exists():
        ap.error(f"--out already exists: {out}; choose a new directory (existing content is never removed)")
    out.mkdir(parents=True)
    ignored = shutil.ignore_patterns("node_modules", ".next", ".venv*", "__pycache__", "*.pyc", "tsconfig.tsbuildinfo",
                                     "runs", ".pytest_cache", ".ruff_cache", "tests", "build", "dist", "*.egg-info",
                                     "effect_files", ".impeccable", ".git", ".claude", ".codex", ".superpowers",
                                     ".env", ".env.*", "AGENTS.md")

    def skip(directory, names):
        excluded = ignored(directory, names)
        excluded.discard(".env.example")  # public setup template, never local credentials
        if Path(directory) == ROOT / "web":
            excluded.add("out")  # only the frontend's generated static export
        return excluded

    for part in parts:
        shutil.copytree(ROOT / part, out / part, ignore=skip)
    # Media credits link this standalone recipe helper; keep the rest of the
    # development tools outside the Space context.
    for name in ("LICENSE", "NOTICE.md", "build/model-files.md", "tools/derive_fire_element.py"):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, out / name)
    for name in ("Dockerfile", "start.sh"):
        shutil.copyfile(HERE / name, out / name)
    (out / "README.md").write_text(README, encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{out}: {size:.0f} MB assembled; run the local checks in deploy/space/README.md before publishing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
