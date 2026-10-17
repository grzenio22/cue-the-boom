"""Make the sample shots ahead of time: every sample plate with every sample noise that suits it.

A judge's first click on a sample then returns at once (the studio says it was made earlier), and the made
shots fill the gallery. The results live outside the repository (videos are large); a deployment builds them
at start-up: instant mode on the CPU, full renders when a GPU is behind the render API.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .pipeline import ShotSpec, make_shot

SHOT_FILES = ("final.mp4", "composite.mp4", "breakdown.mp4", "guide.mp4", "element.mp4", "receipt.json", "cue.json",
              "guide.json", "guide_preview.mp4")


def cache_key(mode: str, plate: str, voice: str, effect: str, point: tuple[float, float], seed: int = 1) -> str:
    """The identity of a sample shot. Points are rounded to the hundredth: a sample's suggested spot."""
    return f"{mode}|{plate}|{voice}|{effect}|{point[0]:.2f},{point[1]:.2f}|{seed}"


def sample_pairs(samples_dir: Path) -> list[dict]:
    """(plate, voice) pairs worth making: a voice's effect must be one its plate suits."""
    meta = json.loads((samples_dir / "samples.json").read_text(encoding="utf-8"))
    pairs = []
    for plate, pm in meta.get("plates", {}).items():
        if not (samples_dir / "plates" / plate).exists() or "point" not in pm:
            continue
        for voice, vm in meta.get("voices", {}).items():
            effect = vm.get("effect")
            if effect and effect in pm.get("effects", [effect]) and (samples_dir / "voices" / voice).exists():
                pairs.append({"plate": plate, "voice": voice, "effect": effect, "point": tuple(pm["point"]),
                              "title": f"{vm.get('title', voice)} · {pm.get('title', plate)}"})
    return pairs


def precompute(renderer, mode: str, samples_dir: Path, out_root: Path, progress=print) -> dict:
    """Make every sample pair with ``renderer`` into ``out_root/<mode>/``; returns (and writes) the index."""
    out = Path(out_root) / mode
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    for p in sample_pairs(samples_dir):
        key = cache_key(mode, p["plate"], p["voice"], p["effect"], p["point"])
        name = f"{Path(p['plate']).stem}__{Path(p['voice']).stem}__{p['effect']}"
        if key in index and (out / index[key]["dir"] / "final.mp4").exists():
            continue
        t = time.perf_counter()
        make_shot(ShotSpec(plate=samples_dir / "plates" / p["plate"], voice=samples_dir / "voices" / p["voice"],
                           effect=p["effect"], point=p["point"], seed=1), renderer, out / name)
        index[key] = {"dir": name, "title": p["title"], "plate": p["plate"], "voice": p["voice"], "effect": p["effect"],
                      "mode": mode}
        index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
        progress(f"{mode}: {name} in {time.perf_counter() - t:.0f} s")
    return index


def load_cache(root: Path | None) -> dict:
    """key -> shot folder, for every mode found under ``root``."""
    found = {}
    if root is None or not Path(root).is_dir():
        return found
    for index_path in Path(root).glob("*/index.json"):
        for key, entry in json.loads(index_path.read_text(encoding="utf-8")).items():
            d = index_path.parent / entry["dir"]
            if (d / "final.mp4").exists():
                found[key] = entry | {"path": d}
    return found


def copy_shot(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in SHOT_FILES:
        if (src / name).exists():
            try:
                (dst / name).hardlink_to(src / name)  # no copy when the filesystem allows it
            except OSError:
                shutil.copyfile(src / name, dst / name)
