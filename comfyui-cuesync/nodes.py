"""CueSync nodes for ComfyUI: a performed noise -> a voice-painted ainvfx-fluid guide -> a composite.

    Load Audio ──> CueSync Analyze Voice ──> CueSync Paint Guide ──> (guide video, prompt) ──> LTX-2.5 + ainvfx-fluid
    Load Video (plate) ─────────────────────────────────────────────> CueSync Composite <── element video
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

try:
    import cuesync  # noqa: F401
except ImportError:  # running from a checkout of the repo: <repo>/comfyui-cuesync next to <repo>/cuesync
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuesync" / "src"))

from cuesync.align import plan_alignment
from cuesync.audio import CueSheet, analyze
from cuesync.composite import composite, effect_options, first_shown_frame, voice_gates
from cuesync.conform import conform
from cuesync.effects import _candidate_dirs, load_effect
from cuesync.guide import paint_guide
from cuesync.media import VideoClip
from cuesync.score import measure_element

FPS = 24
SR = 48000
CATEGORY = "CueSync"


def _effect_ids() -> list[str]:
    for d in _candidate_dirs(Path("effects")):
        if d.is_dir() and any(d.glob("*.json")):
            return sorted(p.stem for p in d.glob("*.json"))
    return ["fireball", "flame_plume", "smoke_puff", "steam_jet"]


def _mono_48k(audio: dict) -> np.ndarray:
    wave = audio["waveform"][0].float().mean(0).cpu().numpy()  # [B, C, N] -> first item, channel mean
    sr = int(audio["sample_rate"])
    if sr != SR:
        t_old = np.arange(wave.size) / sr
        t_new = np.arange(int(round(wave.size * SR / sr))) / SR
        wave = np.interp(t_new, t_old, wave)
    return wave.astype(np.float32)


def _video(images_u8: np.ndarray, audio: dict | None = None):
    from comfy_api.latest import InputImpl, Types

    images = torch.from_numpy(images_u8.astype(np.float32) / 255.0)
    return InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(FPS)))


def _frames(video) -> tuple[np.ndarray, float]:
    c = video.get_components()
    return (c.images.cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8), float(c.frame_rate)


class CueSyncAnalyzeVoice:
    """A performed noise -> cue sheet: onsets, the hero hit, an absolute loudness envelope per frame, and
    when the performer is actually making the noise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "offset_s": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.001,
                                   "tooltip": "Plate time of the recording's first sample (Dub Mode timing.json)."}),
        }}

    RETURN_TYPES = ("CUESYNC_CUE", "IMAGE")
    RETURN_NAMES = ("cue", "envelope")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, offset_s):
        cue = analyze(_mono_48k(audio), SR, offset_s=offset_s, fps=FPS)
        env = np.asarray(cue.envelope)
        w, h = max(256, 4 * env.size), 128
        img = np.full((h, w, 3), 16, np.uint8)
        for i, v in enumerate(env):
            x0, x1 = int(i * w / env.size), int((i + 1) * w / env.size)
            img[h - 4 - int(v * (h - 8)): h - 4, x0:max(x0 + 1, x1)] = (255, 170, 40)
        for a, b in cue.intervals():
            img[:4, int(a * w / env.size): int(b * w / env.size)] = (255, 255, 255)
        return (cue.to_json(), torch.from_numpy(img.astype(np.float32) / 255.0)[None])


class CueSyncPaintGuide:
    """Cue sheet -> the ainvfx-fluid control video, painted from the voice (feed it to Get Video Components
    in place of the sample's painter), plus the effect's prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cue": ("CUESYNC_CUE",),
            "effect": (_effect_ids(),),
            "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "y": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "width": ("INT", {"default": 576, "min": 256, "max": 1920, "step": 64}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 1920, "step": 64}),
            "tail": (["wisp", "fade"],),
            "canvas": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING", "CUESYNC_PLAN")
    RETURN_NAMES = ("guide", "prompt", "plan")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, cue, effect, x, y, width, height, tail, canvas):
        eff = load_effect(effect)
        plan = paint_guide(CueSheet.from_json(cue), eff, (x, y), size=(width, height), tail=tail, canvas=canvas)
        n = plan.frames.shape[0]
        silence = {"waveform": torch.zeros(1, 1, int(round(n / FPS * SR))), "sample_rate": SR}  # the graph expects audio
        meta = plan.to_json() | {"x": x, "y": y}
        return (_video(plan.frames, silence), eff.prompt, meta)


class CueSyncComposite:
    """Plate + rendered element -> the shot: aligned to the voice, gated so it stops when the performer stops,
    composited onto the untouched plate (with the performance as its audio)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("VIDEO",),
            "element": ("VIDEO",),
            "cue": ("CUESYNC_CUE",),
            "plan": ("CUESYNC_PLAN",),
            "max_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.5}),
        }, "optional": {"voice": ("AUDIO",)}}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("shot",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, element, cue, plan, max_seconds, voice=None):
        eff = load_effect(plan["effect"])
        cue_s = CueSheet.from_json(cue)
        frames, fps = _frames(plate)
        clip = VideoClip(frames=frames, times_s=np.arange(len(frames)) / fps, fps_nominal=fps)
        h, w = frames.shape[1:3]
        size = (w - w % 2, h - h % 2)
        plate_f = conform(clip, size=size, fps=FPS, duration_s=max_seconds)
        elem, _ = _frames(element)
        timing = measure_element(elem, kind="fire" if eff.fluid == "fire" else "any")
        align = plan_alignment(int(plan["anchor_frame"]), timing, mode=eff.align, lead=eff.lead_frames)
        gates = voice_gates(len(plate_f), cue_s.intervals(), int(plan["anchor_frame"]), eff.voice_gate,
                            eff.release_fire, eff.release_smoke, open_from=first_shown_frame(align, timing))
        emission = (float(plan["x"]) * size[0], float(plan["y"]) * size[1])
        comp = composite(plate_f, elem, align, timing, gates=gates, emission=emission, **effect_options(eff))
        return (_video(comp, voice),)


NODE_CLASS_MAPPINGS = {
    "CueSyncAnalyzeVoice": CueSyncAnalyzeVoice,
    "CueSyncPaintGuide": CueSyncPaintGuide,
    "CueSyncComposite": CueSyncComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CueSyncAnalyzeVoice": "CueSync Analyze Voice",
    "CueSyncPaintGuide": "CueSync Paint Guide",
    "CueSyncComposite": "CueSync Composite",
}
