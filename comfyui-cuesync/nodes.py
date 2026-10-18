"""CueSync nodes for ComfyUI: a performed noise -> a voice-painted ainvfx-fluid guide -> the shot.

Scene mode (the effect rendered into the plate, the scene reacts):
    Load Audio ──> Analyze Voice ──> Paint Scene ──> (guide, start frame, prompt) ──> LTX-2.5 image-to-video + ainvfx-fluid
    Load Video (plate) ────────────────┘   └──────────────────────────> Scene Shot <── render

Element mode (rendered on black, composited onto the untouched plate):
    Load Audio ──> Analyze Voice ──> Paint Guide ──> (guide video, prompt) ──> LTX-2.5 + ainvfx-fluid
    Load Video (plate) ───────────────────────────────────────────────> Composite <── element video
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
from cuesync.scene import ScenePlan, Strokes, paint_scene_guide, plan_window, scene_prompt, scene_size, suited_effect
from cuesync.score import measure_element

FPS = 24
SR = 48000
CATEGORY = "CueSync"


NO_EFFECTS = "no effects found: reinstall cuesync"


def _effect_ids() -> list[str]:
    # every folder counts: your own effects (CUESYNC_EFFECTS) join the ones that ship with cuesync
    ids = {p.stem for d in _candidate_dirs(Path("effects")) if d.is_dir() for p in d.glob("*.json")}
    if not ids:
        print(f"[CueSync] {NO_EFFECTS} (searched {[str(d) for d in _candidate_dirs(Path('effects'))]})", file=sys.stderr)
    return sorted(ids) or [NO_EFFECTS]


def _effects_with(scene: bool) -> list[str]:
    """The effects a node can paint: scene mode needs a scene section; element mode can't paint one that only
    exists in a scene (blowing a flame out)."""
    ids = []
    for i in _effect_ids():
        try:
            eff = load_effect(i)
        except (FileNotFoundError, ValueError):
            continue
        if (eff.scene is not None) if scene else (eff.behaviour != "extinguish"):
            ids.append(i)
    return ids or [NO_EFFECTS]


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


def _placed(audio: dict | None, offset_s: float) -> dict | None:
    """The take on the plate's timeline: Analyze Sound's offset_s says where its first sample falls on the plate,
    so the voice the shot carries starts there too (silence before it, or its start cut)."""
    if audio is None or abs(offset_s) < 1e-6:
        return audio
    wave, sr = audio["waveform"], int(audio["sample_rate"])
    k = int(round(abs(offset_s) * sr))
    if offset_s > 0:
        wave = torch.cat([torch.zeros(*wave.shape[:-1], k, dtype=wave.dtype), wave], dim=-1)
    else:
        wave = wave[..., k:]
    return {"waveform": wave, "sample_rate": sr}


def _frames(video) -> tuple[np.ndarray, float]:
    c = video.get_components()
    return (c.images.cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8), float(c.frame_rate)


class CueSyncAnalyzeVoice:
    """A sound -> cue sheet: onsets, the hero hit, an absolute loudness envelope per frame, and when the
    sound is actually there. A performed noise, or the plate's own soundtrack (Get Video Components' audio:
    a finger snap, a lighter's click, a door slam) to lock the effect to a sound already in the clip."""

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
        if audio is None or audio.get("waveform") is None or audio["waveform"].numel() == 0:
            raise ValueError("Analyze Sound: there is no sound here (a clip without a soundtrack?); "
                             "use CueSync Hold Cue to time it without one")
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
            "effect": (_effects_with(scene=False),),
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


MAX_SIDE = 1920  # the render is the plate's size: a 4K phone plate would ask LTX-2.5 for 2176x3840


def _plate_frames(video, max_seconds: float) -> np.ndarray:
    """The plate at 24 fps, at most ``max_seconds`` long (never padded), its long side at most MAX_SIDE."""
    frames, fps = _frames(video)
    clip = VideoClip(frames=frames, times_s=np.arange(len(frames)) / fps, fps_nominal=fps)
    h, w = frames.shape[1:3]
    k = min(1.0, MAX_SIDE / max(w, h))
    size = (max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2))
    return conform(clip, size=size, fps=FPS, duration_s=min(max_seconds, len(frames) / fps))


def _strokes(text: str) -> Strokes | None:
    import json as _json

    if not text or not text.strip():
        return None
    try:
        data = _json.loads(text)
    except ValueError:
        raise ValueError('Paint Scene: strokes must be JSON like [{"points": [[0.4, 0.5], [0.6, 0.5]], "width": 0.05}] '
                         "(x and y from 0 to 1, width a share of the frame)") from None
    return Strokes.parse(data)


def _sources(text: str) -> list[tuple[float, float]] | None:
    import json as _json

    if not text or not text.strip():
        return None
    try:
        pts = [(float(a), float(b)) for a, b in _json.loads(text)]
    except (ValueError, TypeError):
        pts = []
    if not 2 <= len(pts) <= 4 or any(not (0 <= a <= 1 and 0 <= b <= 1) for a, b in pts):
        raise ValueError("Paint Scene: sources must be 2 to 4 points as JSON, like [[0.42, 0.72], [0.64, 0.58], [0.43, 0.3]] "
                         "(x and y from 0 to 1, one per hit of the rhythm, in order)")
    return pts


class CueSyncPaintScene:
    """Cue sheet + plate -> an in-scene guide: the plate painted in flat colour zones on every keyframe, the
    effect painted at its source going its way, timed and sized by the voice. Feed the guide to the fluid
    graph's Get Video Components and the start frame to LTXVImgToVideoInplace (strength 1.0, not bypassed):
    LTX-2.5 then renders the effect into the plate itself."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cue": ("CUESYNC_CUE",),
            "plate": ("VIDEO",),
            "effect": (_effects_with(scene=True),),
            "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip": "Where it comes from (a mouth, a wick)."}),
            "y": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.005}),
            "aim": ("BOOLEAN", {"default": False, "tooltip": "Use the direction below instead of the effect's own."}),
            "direction": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 5.0,
                                    "tooltip": "Degrees clockwise from straight up: -90 left, 90 right."}),
            "subject": ("STRING", {"default": "the object", "tooltip": "What it happens to, for the prompt."}),
            "max_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.5}),
        }, "optional": {
            "strokes": ("STRING", {"default": "", "multiline": True,
                                   "tooltip": 'Drawn strokes, JSON: [{"points": [[x, y], ...], "width": 0.05}]. '
                                              "The drawing is where the effect goes; the cue fills it."}),
            "describe": ("STRING", {"default": "", "multiline": True,
                                    "tooltip": "Your own words for what happens (replaces the effect's prompt)."}),
            "sources": ("STRING", {"default": "", "tooltip": "Call and response: a source per hit of a rhythm, JSON "
                                                             "[[x, y], ...] in order (2 to 4). Overrides x and y."}),
        }}

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING", "CUESYNC_SCENE")
    RETURN_NAMES = ("guide", "start_frame", "prompt", "plan")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, cue, plate, effect, x, y, aim, direction, subject, max_seconds, strokes="", describe="", sources=""):
        from cuesync.transfer import is_still

        eff = load_effect(effect)
        cue_s = CueSheet.from_json(cue)
        frames = _plate_frames(plate, max_seconds)
        still = is_still(frames)
        start, length = plan_window(cue_s, eff, None if still else len(frames))
        size = scene_size(frames.shape[2], frames.shape[1])
        drawn = _strokes(strokes)
        pts = _sources(sources)
        if pts and drawn:
            raise ValueError("Paint Scene: use sources or strokes, not both")
        if pts:
            x, y = pts[0]
        eff = suited_effect(eff, subject, drawn.direction if drawn else (direction if aim else None), frames[0],
                            drawn.source if drawn else (x, y), pts)
        sp = paint_scene_guide(cue_s, eff, (x, y), frames[:1] if still else frames, start=start, length=length,
                               direction=direction if aim else None, size=size, strokes=drawn, sources=pts)
        from PIL import Image as _Image
        first = np.asarray(_Image.fromarray(frames[min(max(start, 0), len(frames) - 1)]).resize(size, _Image.Resampling.LANCZOS))
        silence = {"waveform": torch.zeros(1, 1, int(round(length / FPS * SR))), "sample_rate": SR}
        meta = sp.to_json() | {"still": bool(still), "x": x, "y": y, "offset_s": float(cue_s.offset_s)}
        return (_video(sp.guide.frames, silence), torch.from_numpy(first.astype(np.float32) / 255.0)[None],
                scene_prompt(eff, subject, sp.direction, len(sp.events), max((e.size for e in sp.events), default=1.0),
                             describe or None), meta)


class CueSyncHoldCue:
    """Press-and-hold timing as a cue sheet, for when there is no sound: each press is a hit, each hold how
    long it lasts. Write them as "1.0-2.5, 3.2-3.6" (seconds on the plate)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "holds": ("STRING", {"default": "1.0-3.0", "tooltip": "press-release pairs in seconds, comma separated"}),
            "duration_s": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("CUESYNC_CUE",)
    RETURN_NAMES = ("cue",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, holds, duration_s):
        from cuesync.cues import cue_from_holds

        try:
            pairs = [tuple(float(v) for v in h.strip().split("-", 1)) for h in holds.split(",") if h.strip()]
        except ValueError:
            pairs = []
        if not pairs or any(len(p) != 2 for p in pairs):
            raise ValueError(f"Hold Cue: write each press and release in seconds, like 1.0-2.5, 3.2-3.6 (got {holds!r})")
        return (cue_from_holds(pairs, duration_s, FPS).to_json(),)


class CueSyncSceneShot:
    """Plate + in-scene render -> the shot: the effect's visible start measured and moved onto the voice; a
    still plate becomes the render, a moving plate keeps its pixels and gets only what the render changed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("VIDEO",),
            "render": ("VIDEO",),
            "plan": ("CUESYNC_SCENE",),
            "max_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.5}),
        }, "optional": {"voice": ("AUDIO",)}}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("shot", "report")
    OUTPUT_TOOLTIPS = ("the finished shot", "how far off the model was, how it was moved, and any warning (as the studio's receipt says)")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, render, plan, max_seconds, voice=None):
        from PIL import Image as _Image

        from cuesync.scene_shot import time_scene

        eff = load_effect(plan["effect"])
        frames = _plate_frames(plate, max_seconds)
        rend, _ = _frames(render)
        h, w = frames.shape[1:3]
        if rend.shape[1:3] != (h, w):
            rend = np.stack([np.asarray(_Image.fromarray(f).resize((w, h), _Image.Resampling.LANCZOS)) for f in rend])
        sp = ScenePlan.from_json(plan, np.zeros((1, 1, 1, 3), np.uint8))
        st = time_scene(frames, bool(plan.get("still", False)), rend, sp, eff, int(round(max_seconds * FPS)))
        return (_video(np.ascontiguousarray(st.frames), _placed(voice, float(plan.get("offset_s", 0.0)))), _report(st))


def _report(st) -> str:
    """The timing and the warnings of a scene shot, in plain words."""
    if st.raw is None:
        lines = ["No clear onset found where the effect was painted: the render was not re-timed."]
    elif st.warp:
        lines = [f"Each hit of the rhythm put on its own cue ({len(st.warp)} hits), the first one "
                 f"{abs(st.raw)} frames {'early' if st.raw > 0 else 'late'} as rendered."]
    else:
        lines = [f"LTX-2.5 put it {abs(st.raw)} frames {'early' if st.raw > 0 else 'late'}; moved by {abs(st.auto_shift)} "
                 f"frames onto the cue." if st.raw else "LTX-2.5 put it on the cue; not moved."]
    if st.transfer is not None:
        lines.append("Your clip's own pixels, plus what the render changed." if st.transfer.accepted
                     else f"Used the render instead of your clip's own pixels: {st.transfer.reason}.")
    lines += [f"Note: {w}" for w in st.warnings]
    return "\n".join(lines)


_SAM = {}
SAM_REVISIONS = {"facebook/sam2.1-hiera-small": "ee5bba1d82bb8749febdf90f45e84b687142ba03"}  # pinned weights


def _sam(model_id: str):
    """SAM 2.1 (Apache-2.0) through transformers, loaded once per process."""
    if model_id not in _SAM:
        from transformers import Sam2Model, Sam2Processor

        import comfy.model_management as mm
        device = mm.get_torch_device()
        rev = SAM_REVISIONS.get(model_id)
        _SAM[model_id] = (Sam2Processor.from_pretrained(model_id, revision=rev),
                          Sam2Model.from_pretrained(model_id, revision=rev).to(device).eval(), device)
    return _SAM[model_id]


def sam_matte(frame_u8: np.ndarray, point: tuple[float, float], model_id: str = "facebook/sam2.1-hiera-small"):
    """Float matte of the object under the tapped emission point, or None if no plausible mask comes out.

    Prompts: object just below the point (the point sits on the object's top edge or opening), background
    well above it (where the effect will be)."""
    from PIL import Image as _Image

    processor, model, device = _sam(model_id)
    h, w = frame_u8.shape[:2]
    x, y = point[0] * w, point[1] * h
    pts = [[[[x, min(h - 1, y + 0.03 * h)], [x, max(0, y - 0.12 * h)]]]]
    labels = [[[1, 0]]]
    inputs = processor(images=_Image.fromarray(frame_u8), input_points=pts, input_labels=labels, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].float().numpy()
    scores = out.iou_scores.squeeze().float().cpu().numpy().reshape(-1)
    best, best_score = None, -1.0
    for m, sc in zip(masks, scores):
        area = float((m > 0.5).mean())
        if 0.002 <= area <= 0.6 and m[int(min(h - 1, y + 0.03 * h)), int(min(w - 1, x))] > 0.5 and sc > best_score:
            best, best_score = m, float(sc)
    if best is None:
        return None
    from PIL import ImageFilter
    img = _Image.fromarray((np.clip(best, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(1.0, 0.002 * min(w, h))))
    return np.asarray(img, dtype=np.float32) / 255.0


class CueSyncObjectMatte:
    """The object the effect comes from (SAM 2.1, prompted by the emission point): it stays in front of the
    effect below that point. Feed it the plate's frames; the first one is used (plates are locked off)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "y": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "model": (["facebook/sam2.1-hiera-small", "facebook/sam2.1-hiera-tiny", "facebook/sam2.1-hiera-base-plus"],),
        }, "optional": {"plan": ("CUESYNC_PLAN", {"tooltip": "Take the point from CueSync Paint Guide (overrides x, y)."})}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("object",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, x, y, model, plan=None):
        if plan is not None:
            x, y = float(plan["x"]), float(plan["y"])
        frame = (images[0].cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        m = sam_matte(frame, (x, y), model)
        if m is None:  # nothing plausible: an empty matte, and the compositor's soft holdout takes over
            m = np.zeros(frame.shape[:2], np.float32)
        return (torch.from_numpy(m)[None],)


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
        }, "optional": {"voice": ("AUDIO",), "object_mask": ("MASK",)}}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("shot",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, element, cue, plan, max_seconds, voice=None, object_mask=None):
        eff = load_effect(plan["effect"])
        cue_s = CueSheet.from_json(cue)
        plate_f = _plate_frames(plate, max_seconds)
        size = (plate_f.shape[2], plate_f.shape[1])
        elem, _ = _frames(element)
        timing = measure_element(elem, kind="fire" if eff.fluid == "fire" else "any")
        align = plan_alignment(int(plan["anchor_frame"]), timing, mode=eff.align, lead=eff.lead_frames)
        gates = voice_gates(len(plate_f), cue_s.intervals(), int(plan["anchor_frame"]), eff.voice_gate,
                            eff.release_fire, eff.release_smoke, open_from=first_shown_frame(align, timing))
        emission = (float(plan["x"]) * size[0], float(plan["y"]) * size[1])
        occluder = None
        if object_mask is not None and float(object_mask.max()) > 0:
            m = object_mask[0].float().cpu().numpy()
            if m.shape != plate_f.shape[1:3]:
                from PIL import Image as _Image
                m = np.asarray(_Image.fromarray(m).resize((plate_f.shape[2], plate_f.shape[1]), _Image.Resampling.BILINEAR))
            occluder = m.astype(np.float32)
        comp = composite(plate_f, elem, align, timing, gates=gates, emission=emission, occluder=occluder,
                         **effect_options(eff))
        return (_video(comp, _placed(voice, float(cue_s.offset_s))),)


NODE_CLASS_MAPPINGS = {
    "CueSyncAnalyzeVoice": CueSyncAnalyzeVoice,
    "CueSyncPaintScene": CueSyncPaintScene,
    "CueSyncSceneShot": CueSyncSceneShot,
    "CueSyncHoldCue": CueSyncHoldCue,
    "CueSyncPaintGuide": CueSyncPaintGuide,
    "CueSyncObjectMatte": CueSyncObjectMatte,
    "CueSyncComposite": CueSyncComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CueSyncAnalyzeVoice": "CueSync Analyze Sound (voice or clip)",
    "CueSyncPaintScene": "CueSync Paint Scene",
    "CueSyncSceneShot": "CueSync Scene Shot",
    "CueSyncHoldCue": "CueSync Hold Cue (no sound)",
    "CueSyncPaintGuide": "CueSync Paint Guide",
    "CueSyncObjectMatte": "CueSync Object Matte",
    "CueSyncComposite": "CueSync Composite",
}
