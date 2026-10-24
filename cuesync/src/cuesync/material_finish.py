"""An explicit LTX material pass over tracked stone; photographed pixels stay outside its mask.

The procedural performance supplies geometry, occlusion and timing. LTX supplies
surface detail. This is a fresh GPU render, never an instant replay. Air and water
are deliberately excluded: look-development trials changed their substance.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .media import read_audio, read_video, write_video

EFFECTS = ("earth_rock", "earth_shatter")
RECIPE = "tracked-stone-detail-v2"
MODEL_STRENGTH = 0.65
GUIDE_STRENGTH = 0.85
MATERIAL_SEED = 42  # a fixed, reviewed material recipe; the performance retains its own geometry seed


def prompt_for(effect: str) -> str:
    if effect not in EFFECTS:
        raise ValueError("material finish supports floating stone and stone shatter")
    action = ("rotates slowly, with small chips orbiting it" if effect == "earth_rock" else
              "breaks into jagged fragments on the hand strike, the fragments fly out and fall")
    return ("ainvfxfluid, A live action close up of an open human hand in a warm domestic room. "
            "A rough dark grey granite boulder the size of a fist levitates ten centimetres above the palm. "
            "The heavy irregular rock " + action + ", with broken angular faces, deep mineral cracks, "
            "coarse stony texture and mottled grey grains. A few jagged stone chips orbit it at different "
            "depths and fine dust falls down onto the palm. A soft contact shadow moves on the skin. "
            "The rock reflects the warm light already in the room. Natural room lighting, realistic skin, "
            "exactly five fingers. Preserve the room, wall switches, hand position and camera framing. "
            "No wood grain, no clay, no smooth sphere, no glow, no text.")


def replace_material(base: np.ndarray, generated: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Only covered stone pixels may change, including its existing antialiased edge.

    No dilation: it would let generated skin or background leak around the stone.
    No temporal expansion: a frame before the cue must remain unmodified.
    """
    if base.shape != generated.shape or coverage.shape != base.shape[:-1]:
        raise ValueError("material pass and mask must match the source frames")
    alpha = coverage.astype(np.float32)[..., None] / 255.0
    out = base.copy()
    inside = coverage > 0
    out[inside] = np.rint(base[inside] * (1 - alpha[inside]) + generated[inside] * alpha[inside]).astype(np.uint8)
    return out


def safe_material_coverage(base: np.ndarray, generated: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Keep generated skin, wall colours and translucent release frames out of stone.

    Geometry owns transparency. Model texture is useful on the opaque material;
    introducing it during the release would turn a fading stone into a flat patch.
    Colour is compared relative to the existing lit material, not a skin-tone rule.
    """
    a, b = base.astype(np.float32) + 8, generated.astype(np.float32) + 8
    chroma = np.max(np.abs(a / a.sum(-1, keepdims=True) - b / b.sum(-1, keepdims=True)), axis=-1)
    agreement = np.clip((.12 - chroma) / .06, 0, 1)
    opaque = np.clip((coverage.astype(np.float32) - 225) / 25, 0, 1)
    return np.rint(coverage * agreement * opaque).astype(np.uint8)


def stone_surface_detail(base: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Transfer fine mineral contrast, retaining the tracked geometry's colour and lighting.

    Even a masked RGB replacement can paint a dark hole or a piece of the room
    inside solid stone. Discard the model's broad shading and bound its detail:
    it can describe cracks and grains, but cannot replace a face of the rock.
    """
    if base.shape != generated.shape or base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError("stone detail requires matching RGB frames")
    luminance = generated.astype(np.float32) @ np.array([.2126, .7152, .0722], np.float32)
    sigma = max(1.0, min(base.shape[:2]) / 180)
    broad = cv2.GaussianBlur(luminance, (0, 0), sigma)
    detail = np.clip(((luminance + 12) / (broad + 12)) ** .65, .78, 1.22)
    return np.rint(np.clip(base.astype(np.float32) * detail[..., None], 0, 255)).astype(np.uint8)


def finish_shot(src: Path, renderer, out: Path, *, parent_id: str, progress=None):
    """Refine an existing stone shot, keeping its full duration, sound and cue ribbon."""
    start = time.perf_counter()
    src, out = Path(src), Path(out)
    receipt = json.loads((src / "receipt.json").read_text())
    effect = receipt.get("effect", {}).get("id")
    prompt = prompt_for(effect)
    if receipt.get("material_finish"):
        raise ValueError("this shot already has a material finish; refine the original performance")
    base = read_video(src / "composite.mp4")
    n, h, w = base.frames.shape[:3]
    if not 1 <= n <= 240 or abs(base.fps_nominal - 24) > .01:
        raise ValueError("material finish requires a shot of up to ten seconds at 24 fps")
    with np.load(src / "effect_mask.npz", allow_pickle=False) as data:
        coverage = data["alpha"]
    if coverage.dtype != np.uint8 or coverage.shape != base.frames.shape[:-1]:
        raise ValueError("stone coverage does not match the source shot")
    if not coverage.any():
        raise ValueError("there is no visible stone to refine in this performance")
    say = progress or (lambda *args: None)
    say("preparing", .05, "Preparing the tracked stone for its material pass")
    out.mkdir(parents=True, exist_ok=True)
    native = (576, 1024) if h > w else (1024, 576)
    guide = np.stack([cv2.resize(f, native, interpolation=cv2.INTER_AREA) for f in base.frames])
    guide_path = out / "guide.mp4"
    write_video(guide_path, guide, fps=24, audio=np.zeros(n * 2000, np.float32), preset="veryfast")
    # Fixed 81-frame contexts keep the reviewed recipe stable on longer takes.
    # Overlapping windows start from the photographed performance, never a
    # previously generated frame. Seventeen shared frames hide texture seams.
    starts = [0]
    while starts[-1] + 81 < n:
        starts.append(starts[-1] + 64)
    generated = np.zeros(guide.shape, np.float32)
    weights = np.zeros(n, np.float32)
    windows, seconds = [], 0.0
    for index, begin in enumerate(starts):
        part = out / f"window-{index:02d}"
        part.mkdir(exist_ok=True)
        count = min(81, n - begin)
        clip = guide[begin:begin + count]
        if count < 81:
            clip = np.concatenate([clip, np.repeat(clip[-1:], 81 - count, axis=0)])
        write_video(part / "guide.mp4", clip, fps=24, audio=np.zeros(81 * 2000, np.float32), preset="veryfast")
        Image.fromarray(clip[0]).save(part / "plate.png")
        say("rendering", .18 + .6 * index / len(starts), f"LTX stone material pass {index + 1} of {len(starts)}")
        seconds += renderer.refine(part / "guide.mp4", prompt, MATERIAL_SEED,
                                   part / "render.mp4", image=part / "plate.png")
        result = read_video(part / "render.mp4")
        if len(result.frames) != 81 or abs(result.fps_nominal - 24) > .01:
            raise ValueError("LTX material pass changed the guide's duration or frame rate")
        weight = np.ones(count, np.float32)
        if index:
            weight[:min(17, count)] = np.linspace(0, 1, min(17, count))
        if index < len(starts) - 1:
            weight[64:81] = np.linspace(1, 0, 17)
        generated[begin:begin + count] += result.frames[:count] * weight[:, None, None, None]
        weights[begin:begin + count] += weight
        windows.append({"start_frame": begin, "source_frames": count, "guide_frames": 81})
    generated = np.rint(generated / weights[:, None, None, None]).astype(np.uint8)
    write_video(out / "render.mp4", generated, fps=24, preset="veryfast")
    if (out / "window-00/material_workflow.json").exists():
        shutil.copyfile(out / "window-00/material_workflow.json", out / "material_workflow.json")
    del guide
    say("compositing", .8, "Keeping the original hand, room, timing and sound")
    frames = base.frames.copy()
    material_pixels, rejected_pixels = 0, 0
    for k in range(n):
        detail = cv2.resize(generated[k], (w, h), interpolation=cv2.INTER_LANCZOS4)
        safe = safe_material_coverage(base.frames[k], detail, coverage[k])
        core = coverage[k] >= 250
        material_pixels += int(core.sum())
        rejected_pixels += int((core & (safe < 128)).sum())
        frames[k] = replace_material(base.frames[k], stone_surface_detail(base.frames[k], detail), safe)
    rejected_fraction = rejected_pixels / max(1, material_pixels)
    if rejected_fraction > .25:
        raise ValueError("LTX changed too much of the stone material; the original performance is preserved")
    del generated
    audio = read_audio(src / "composite.mp4")[:n * 2000]
    write_video(out / "composite.mp4", frames, fps=24, audio=audio, crf=14, preset="veryfast")
    final = read_video(src / "final.mp4").frames
    if final.shape[0] != n or final.shape[2] != w or final.shape[1] < h:
        raise ValueError("source cue ribbon does not match the composite")
    final[:, :h] = frames
    write_video(out / "final.mp4", final, fps=24, audio=audio, crf=16, preset="veryfast")
    poster = int(np.argmax(np.count_nonzero(coverage, axis=(1, 2))))
    Image.fromarray(frames[poster]).save(out / "poster.jpg", quality=88)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt["material_finish"] = {
        "recipe": RECIPE, "parent_id": parent_id, "new_gpu_render": True,
        "native_size": list(native), "windows": windows, "output_frames": n,
        "source_sha256": digest(src / "composite.mp4"),
        "mask_sha256": digest(src / "effect_mask.npz"),
        "guide_sha256": digest(guide_path), "render_sha256": digest(out / "render.mp4"),
        "model_strength": MODEL_STRENGTH, "guide_strength": GUIDE_STRENGTH,
        "seed": MATERIAL_SEED,
        "transfer": "bounded fine luminance detail; procedural colour, geometry and broad shading retained",
        "scope": "tracked stone surface only; original photographed pixels outside its coverage",
        "rejected_material_fraction": round(rejected_fraction, 4),
    }
    receipt["render"] = {"seconds": round(seconds, 2), "renderer": "comfy", "model": "LTX-2.5",
                         "prompt": prompt, "guide_scale": 1.0}
    receipt["seconds_total"] = round(time.perf_counter() - start, 2)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    say("done", 1, "Stone material finished")
    return receipt
