"""Instant mode: pre-rendered LTX-2.5 elements, timed and placed by your performance on a CPU.

The full pipeline renders a new element from your voice on a GPU (20-50 s). Instant mode picks one from a
small library rendered the same way from reference performances (a hit, a short and a long noise per
effect), and does everything else from YOUR recording exactly as the full pipeline does: the element lands
on your hit or follows your onset, stops when you stop (the compositor's voice gate), sits where you tapped
and is a little bigger or smaller with how loud you were. What it cannot do is reshape the fluid itself to
your voice: that is what the GPU render adds.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .audio import analyze
from .effects import load_effect
from .guide import paint_guide, write_guide
from .media import read_video, write_video

LIBRARY_POINT = (0.5, 0.75)  # where every library element is emitted, normalised to LIBRARY_SIZE
LIBRARY_SIZE = (576, 1024)
SR = 48000
PERFORMANCES = {"hit": 0.12, "short": 0.45, "medium": 1.2, "long": 2.6}  # reference noise durations (s)
PLAN = {
    "fireball": ("hit", "short", "medium"),
    "smoke_puff": ("hit", "short", "medium"),
    "flame_plume": ("short", "medium", "long"),
    "steam_jet": ("short", "medium", "long"),
}
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "samples" / "library"


def reference_performance(name: str, seed: int = 0) -> np.ndarray:
    """A reference noise: breathy shaped noise with 1 s of silence either side. The library only needs its
    loudness over time; the timbre is irrelevant to the guide."""
    rng = np.random.default_rng(seed)
    dur = PERFORMANCES[name]
    t = np.arange(int(dur * SR)) / SR
    attack = np.minimum(t / 0.01, 1.0)
    if name == "hit":
        env = attack * np.exp(-t / 0.05)
    elif name == "short":
        env = attack * np.exp(-t / 0.3)
    elif name == "medium":  # a rising "fwoooOOOM"
        env = attack * (0.35 + 0.65 * t / dur) * np.minimum((dur - t) / 0.06, 1.0)
    else:  # a held noise with a little life in it
        env = attack * (0.75 + 0.1 * np.sin(2 * np.pi * 1.7 * t)) * np.minimum((dur - t) / 0.08, 1.0)
    noise = rng.standard_normal(t.size)
    smooth = np.empty_like(noise)  # one-pole low-pass: breath, not hiss
    acc = 0.0
    for i, v in enumerate(noise):
        acc += 0.15 * (v - acc)
        smooth[i] = acc
    body = 0.6 * env * smooth / max(np.abs(smooth).max(), 1e-9)
    pad = np.zeros(SR, np.float64)
    return np.concatenate([pad, body, pad]).astype(np.float32)


def render_library(renderer, out_dir: Path = DEFAULT_DIR, effects: dict | None = None, seeds=(1, 2),
                   progress=print) -> dict:
    """Render the library with ``renderer`` (a pipeline Renderer, e.g. ComfyRenderer) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cuesync-guides-"))
    entries = []
    for effect_id, perfs in (effects or PLAN).items():
        eff = load_effect(effect_id)
        for perf in perfs:
            wav = reference_performance(perf)
            cue = analyze(wav, SR, fps=24)
            plan = paint_guide(cue, eff, LIBRARY_POINT, size=LIBRARY_SIZE)
            guide = tmp / f"{effect_id}_{perf}.mp4"
            write_guide(plan, guide)
            for seed in seeds:
                name = f"{effect_id}_{perf}_s{seed}.mp4"
                seconds = renderer(guide, eff.prompt, seed, out_dir / name)
                entries.append({"effect": effect_id, "performance": perf, "seed": seed, "file": name,
                                "length": int(plan.frames.shape[0]), "anchor_frame": plan.anchor_frame,
                                "voice_end": plan.voice_end, "peak_level": round(float(np.max(cue.envelope)), 4),
                                "render_s": round(float(seconds), 1)})
                progress(f"{name}: {plan.frames.shape[0]} frames in {seconds:.0f} s")
            guide.unlink(missing_ok=True)
    manifest = {"point": list(LIBRARY_POINT), "size": list(LIBRARY_SIZE), "fps": 24, "entries": entries,
                "model": "LTX-2.5 22B distilled (int8) + AInVFX/ainvfx-fluid IC-LoRA"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def place(frames: np.ndarray, src_point: tuple[float, float], dst_point: tuple[float, float],
          size: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Element frames moved so ``src_point`` (pixels in them) lands on ``dst_point`` (pixels in a ``size``
    canvas), scaled by ``scale`` about that point; black where nothing lands."""
    w, h = size
    n, fh, fw = frames.shape[:3]
    out = np.zeros((n, h, w, 3), np.uint8)
    sw, sh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    ox, oy = int(round(dst_point[0] - src_point[0] * scale)), int(round(dst_point[1] - src_point[1] * scale))
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(w, ox + sw), min(h, oy + sh)
    if x1 <= x0 or y1 <= y0:
        return out
    for i in range(n):
        f = frames[i] if (sw, sh) == (fw, fh) else np.asarray(Image.fromarray(frames[i]).resize((sw, sh), Image.Resampling.BICUBIC))
        out[i, y0:y1, x0:x1] = f[y0 - oy: y1 - oy, x0 - ox: x1 - ox]
    return out


@dataclass
class LibraryRenderer:
    """A Renderer for make_shot that 'renders' by choosing a library element: the right effect, long enough
    for your performance, placed at your point and scaled with your loudness."""

    library_dir: Path = DEFAULT_DIR
    name: str = "library"

    def __post_init__(self) -> None:
        self.library_dir = Path(self.library_dir)
        self.manifest = json.loads((self.library_dir / "manifest.json").read_text(encoding="utf-8"))
        self.last: dict | None = None

    def choose(self, effect: str, length: int, seed: int) -> dict:
        entries = [e for e in self.manifest["entries"] if e["effect"] == effect]
        if not entries:
            raise ValueError(f"no library elements for {effect!r} in {self.library_dir}")
        long_enough = [e for e in entries if e["length"] >= length]
        pool = long_enough or entries
        best = min(e["length"] for e in pool) if long_enough else max(e["length"] for e in pool)
        candidates = sorted((e for e in pool if e["length"] == best), key=lambda e: e["seed"])
        return candidates[(seed - 1) % len(candidates)]

    def __call__(self, guide_path: Path, prompt: str, seed: int, out_path: Path) -> float:
        guide_path = Path(guide_path)
        plan = json.loads((guide_path.parent / "guide.json").read_text(encoding="utf-8"))
        cue = json.loads((guide_path.parent / "cue.json").read_text(encoding="utf-8"))
        entry = self.choose(plan["effect"], int(plan["length"]), seed)
        frames = read_video(self.library_dir / entry["file"]).frames
        size = tuple(plan["size"])
        lw, lh = self.manifest["size"]
        src = (self.manifest["point"][0] * lw, self.manifest["point"][1] * lh)
        dst = (plan["point"][0] * size[0], plan["point"][1] * size[1])
        loud = float(np.max(cue["envelope"])) if cue.get("envelope") else entry["peak_level"]
        scale = float(np.clip((loud / max(entry["peak_level"], 1e-3)) ** 0.5, 0.8, 1.25))
        write_video(out_path, place(frames, src, dst, size, scale), fps=24, crf=12)
        # the guide this element was rendered from (deterministic: repainted, not stored), placed the same way
        ref = paint_guide(analyze(reference_performance(entry["performance"]), SR, fps=24), load_effect(plan["effect"]),
                          LIBRARY_POINT, size=LIBRARY_SIZE)
        self.last = {"entry": entry["file"], "performance": entry["performance"], "scale": round(scale, 3),
                     "rendered_s": entry.get("render_s"),
                     "guide": {"frames": place(ref.frames, src, dst, size, scale), "keyframes": ref.keyframes}}
        return 0.0
