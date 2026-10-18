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

SHOT_FILES = ("final.mp4", "composite.mp4", "breakdown.mp4", "guide.mp4", "element.mp4", "render.mp4", "receipt.json",
              "cue.json", "guide.json", "guide_preview.mp4", "start_frame.png", "matte.png")


def effect_signature(effect: str, subject: str | None = None, direction: float | None = None) -> str:
    """A short hash of an effect's definition as it applies to a shot of ``subject`` aimed ``direction`` (its
    scene variants resolved, a flame taken as burning): a shot made with an older version of it is not that shot,
    but words added for other subjects (a hand) don't retire a jack-o'-lantern's."""
    import hashlib
    from dataclasses import asdict

    from .effects import load_effect

    eff = load_effect(effect)
    d = eff.to_json()
    if eff.scene is not None:
        d["scene"] = asdict(eff.scene.suited(subject, eff.scene.direction if direction is None else direction, lit=True))
        d["scene"].pop("variants", None)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def cache_key(mode: str, plate: str, voice: str, effect: str, point: tuple[float, float], seed: int = 1,
              direction: float | None = None, subject: str | None = None) -> str:
    """The identity of a sample shot. Points are rounded to the hundredth: a sample's suggested spot; a scene
    shot's direction to the degree; the effect by its definition, so editing an effect retires its old shots."""
    sig = effect_signature(effect, subject if mode == "scene" else None, direction)
    key = f"{mode}|{plate}|{voice}|{effect}@{sig}|{point[0]:.2f},{point[1]:.2f}|{seed}"
    return key + (f"|{round(direction)}" if direction is not None else "")


def sample_pairs(samples_dir: Path, mode: str = "render") -> list[dict]:
    """(plate, voice) pairs worth making. Element modes: a voice's effect must be one its plate suits. Scene
    mode: each plate says what each noise does to it ("scene": {voice: effect or {"effect", "seed"}})."""
    meta = json.loads((samples_dir / "samples.json").read_text(encoding="utf-8"))
    pairs = []
    for plate, pm in meta.get("plates", {}).items():
        if not (samples_dir / "plates" / plate).exists() or "point" not in pm:
            continue
        if mode == "scene":
            for voice, what in pm.get("scene", {}).items():
                if voice not in meta.get("voices", {}):
                    continue  # a gesture entry ("hit", "sustained"): it serves live takes, not a sample pair
                what = {"effect": what} if isinstance(what, str) else what
                vm = meta.get("voices", {}).get(voice, {})
                if (samples_dir / "voices" / voice).exists():
                    pairs.append({"plate": plate, "voice": voice, "effect": what["effect"],
                                  "point": tuple(what.get("point", pm["point"])), "seed": int(what.get("seed", 1)),
                                  "direction": what.get("direction", pm.get("direction")), "subject": pm.get("subject"),
                                  "title": f"{vm.get('title', voice)} · {pm.get('title', plate)}"})
            if pm.get("clip_cue"):  # the clip's own sound times it
                cc = pm["clip_cue"]
                pairs.append({"plate": plate, "voice": "clip", "effect": cc["effect"], "point": tuple(pm["point"]),
                              "seed": int(cc.get("seed", 1)), "direction": pm.get("direction"), "subject": pm.get("subject"),
                              "cue": "clip", "title": cc.get("title", f"the clip's own sound · {pm.get('title', plate)}")})
            for k, dr in enumerate(pm.get("drawn", [])):  # drawn showcase shots: a drawing and its cue
                voice = dr.get("voice")
                if voice and not (samples_dir / "voices" / voice).exists():
                    continue
                pairs.append({"plate": plate, "voice": voice or f"hold{k}", "effect": dr["effect"],
                              "point": tuple(dr["strokes"][0]["points"][0]), "seed": int(dr.get("seed", 1)),
                              "direction": None, "subject": pm.get("subject"), "strokes": dr["strokes"],
                              "describe": dr.get("describe"), "cue": "voice" if voice else "hold",
                              "holds": dr.get("holds"), "title": dr.get("title", f"drawn · {pm.get('title', plate)}")})
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
    for p in sample_pairs(samples_dir, mode):
        # the studio asks for a sample's first take with seed 1; a scene pair may be made with a chosen seed
        key = cache_key(mode, p["plate"], p["voice"], p["effect"], p["point"], 1, p.get("direction"), p.get("subject"))
        name = f"{Path(p['plate']).stem}__{Path(p['voice']).stem}__{p['effect']}"
        if key in index and (out / index[key]["dir"] / "final.mp4").exists():
            continue
        t = time.perf_counter()
        scene = mode == "scene"
        cue = p.get("cue", "voice")
        make_shot(ShotSpec(plate=samples_dir / "plates" / p["plate"],
                           voice=samples_dir / "voices" / p["voice"] if cue == "voice" else None,
                           effect=p["effect"], point=p["point"], seed=p.get("seed", 1),
                           mode="scene" if scene else "element", direction=p.get("direction") if scene else None,
                           subject=p.get("subject") if scene else None, cue=cue, holds=p.get("holds"),
                           strokes=p.get("strokes"), describe=p.get("describe")), renderer, out / name)
        index[key] = {"dir": name, "title": p["title"], "plate": p["plate"], "voice": p["voice"], "effect": p["effect"],
                      "mode": mode, "seed": p.get("seed", 1), "drawn": bool(p.get("strokes"))}
        index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
        progress(f"{mode}: {name} in {time.perf_counter() - t:.0f} s")
    return index


def load_cache(root: Path | None) -> dict:
    """key -> shot folder, for every mode found under ``root``."""
    found = {}
    if root is None or not Path(root).is_dir():
        return found
    for index_path in (Path(root) / m / "index.json" for m in ("instant", "render", "scene")):  # not old copies
        if not index_path.exists():
            continue
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
