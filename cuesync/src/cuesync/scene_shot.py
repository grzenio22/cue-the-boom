"""One shot in scene mode: the effect rendered into the plate itself (scene.py), timed by CueSync (transfer.py).

    plate frame + cue -> scene guide (the plate's zones, the effect at its source or where it was drawn) ->
    LTX-2.5 image-to-video from that plate frame -> measured onset -> re-timed onto the cue -> the shot

The cue is a voice, a sound already in the clip, or a press-and-hold (cues.py). A still plate (a photo held as
a clip) becomes the render itself; a moving plate keeps its own pixels and gets only what the render changed
(delta transfer), unless the transfer fails its check, and then the render is used and the shot says so.
Deliverables are the same as element mode's.
"""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import __version__
from .audio import CueSheet
from .conform import _crop_box, crop_point
from .effects import load_effect
from .guide import write_guide, write_guide_preview
from .matte import grabcut_matte
from .media import probe_video, read_video, write_video
from .pipeline import (FINISH_PRESET, FPS, Progress, Renderer, ShotResult, ShotSpec, _breakdown, _face_track,
                       _finalise, _sha, auto_scale, load_plate, place_audio, plate_notes, shot_cue)
from .scene import ScenePlan, Strokes, paint_scene_guide, plan_window, scene_prompt, scene_size
from .transfer import (TransferStats, _ecc, changed_share, effect_region, first_visible, gam, hit_onsets, is_still, lin,
                       onset_past_steps, transfer)

MAX_SHIFT = 16  # the model is trusted to land within this many frames of the painted hit
WARP = (0.75, 1.33)  # how much slower or faster a rhythm's render may play between two hits to put each on its cue
MODEL = "LTX-2.5 22B distilled (int8) + AInVFX/ainvfx-fluid IC-LoRA, image-to-video from the plate frame"
FIRE_STOP_SHARE = 0.15  # the fire has stopped when its visible area is below this share of its peak
GAS_STOP_SHARE = 0.3  # smoke or steam has settled when the change is below this share of its peak


@dataclass
class SceneTiming:
    frames: np.ndarray  # the shot
    index: list[int | None]  # render frame shown on each shot frame (None: the plate itself)
    plate_view: np.ndarray  # the plate on the shot's timeline (a still plate is held)
    onset: int | None  # render frame where the effect visibly starts (a rhythm: where its biggest hit lands)
    painted: int  # render frame where that was painted
    raw: int | None  # (painted - lead) - onset: how far the model was from the cue, in frames (+ early)
    auto_shift: int  # frames the render was moved so that onset lands on the cue
    shift: int  # auto_shift plus a hand re-time
    transfer: TransferStats | None  # delta transfer (moving plates only)
    hits: list[int] = field(default_factory=list)  # shot frames where each event shows
    warnings: list[str] = field(default_factory=list)
    merged: list[int] = field(default_factory=list)  # events the model merged into a neighbour (no flash, no shake)
    warp: list[tuple[int, int]] = field(default_factory=list)  # a rhythm's (render onset, cue) per hit put on its cue


def warp_anchors(onsets: list[int | None], wants: list[int]) -> list[tuple[int, int]]:
    """(render frame, cue frame) for each hit of a rhythm that can go on its own cue: shown on its own, within
    MAX_SHIFT of its cue, and reached from the hit before by playing the render at a speed within WARP. LTX-2.5
    spaces a rhythm's hits unevenly, so one shift for all of them leaves the later ones off their cues."""
    out: list[tuple[int, int]] = []
    for o, w in zip(onsets, wants):
        if o is None or abs(w - o) > MAX_SHIFT:
            continue
        if out:
            po, pw = out[-1]
            if w <= pw or o <= po or not WARP[0] <= (o - po) / (w - pw) <= WARP[1]:
                continue
        out.append((o, w))
    return out


def on_cue(r: float, anchors: list[tuple[int, int]]) -> float:
    """Where render frame ``r`` lands on the painted timeline: in straight lines through the anchors, at the
    render's own speed before the first and after the last."""
    rs, ws = zip(*anchors)
    if r <= rs[0] or len(anchors) == 1:
        return ws[0] + (r - rs[0]) if r <= rs[0] else ws[-1] + (r - rs[-1])
    if r >= rs[-1]:
        return ws[-1] + (r - rs[-1])
    return float(np.interp(r, rs, ws))


def from_cue(a: float, anchors: list[tuple[int, int]]) -> float:
    """The render frame that lands at ``a`` on the painted timeline (the inverse of ``on_cue``)."""
    return on_cue(a, [(w, r) for r, w in anchors])


def _fire_area(frame: np.ndarray, region: np.ndarray) -> float:
    """Share of the effect region that looks like flame (bright, warm, saturated)."""
    f = frame[::4, ::4].astype(np.int16)
    r = region[::4, ::4]
    fire = (f[..., 0] > 170) & (f[..., 0] - f[..., 2] > 70) & (f[..., 1] > 70)
    return float((fire & r).sum()) / max(1, int(r.sum()))


_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _gone(count: np.ndarray, early: slice, share: float) -> int | None:
    lit = float(np.median(count[early]))
    if lit < 4:
        return None
    low = count < share * lit
    for f in range(1, len(count) - 2):
        if low[f] and low[f + 1] and low[f + 2]:
            return f
    return None


def flame_out(render: np.ndarray, region: np.ndarray, ref_frames: int, share: float = 0.1) -> int | None:
    """The render frame where a flame that was burning goes out: almost none of its flame pixels are left
    (below ``share`` of the lit flame's) for three frames. Flame pixels are flame-coloured; where that can't
    tell (a white-hot flare, or wax the flame lit warm all along, as on a real candle) they are the pixels
    nearly as bright as the burning flame's brightest. None if no flame was visible."""
    small = render[:, ::2, ::2].astype(np.int16)
    reg = region[::2, ::2]
    early = slice(1, max(2, min(8, ref_frames)))  # before the model can have started
    warm = np.array([float((((f[..., 0] > 180) & (f[..., 0] - f[..., 2] > 50)) & reg).sum()) for f in small])
    out = _gone(warm, early, share)
    if out is None:
        luma = [f.astype(np.float32) @ _LUMA for f in small]
        top = 0.9 * float(np.percentile(np.median(np.stack(luma[early]), axis=0)[reg], 99.7))
        bright = np.array([float(((y > max(150.0, top)) & (f[..., 0] >= f[..., 2]) & reg).sum()) for y, f in zip(luma, small)])
        out = _gone(bright, early, share)
    return out


def measure_stop(frames: np.ndarray, region: np.ndarray, voice_end: int | None, fire: bool, first: int) -> int | None:
    """The shot frame where the effect has visibly stopped after the cue ends: flame area below
    FIRE_STOP_SHARE of its peak (fire), or the change from the scene before the effect below GAS_STOP_SHARE
    of its peak (smoke, steam). None when the cue never ends or it never stops within the shot."""
    if voice_end is None or voice_end >= len(frames):
        return None
    if fire:
        area = np.array([_fire_area(f, region) for f in frames])
        peak = float(area[max(0, first): voice_end + 1].max()) if voice_end >= first else 0.0
        if peak <= 0.002:
            return None
        below = np.nonzero(area[voice_end:] < FIRE_STOP_SHARE * peak)[0]
    else:
        ref = frames[max(0, first - 2)][::4, ::4].astype(np.float32)
        r = region[::4, ::4]
        ch = np.array([float(np.abs(f[::4, ::4].astype(np.float32) - ref).max(-1)[r].mean()) for f in frames])
        peak = float(ch[max(0, first): voice_end + 1].max()) if voice_end >= first else 0.0
        if peak <= 2.0:
            return None
        below = np.nonzero(ch[voice_end:] < GAS_STOP_SHARE * peak)[0]
    return int(voice_end + below[0]) if below.size else None


def _render_over_plate(plate: np.ndarray, render: np.ndarray, idx: list[int | None], fade: int = 6) -> np.ndarray:
    """When a transfer can't be trusted: the render inside its window, cross-faded in from the plate at the
    start and back to the plate after the render ends."""
    n, L = len(plate), render.shape[0]
    out = plate.copy()
    live = [f for f in range(n) if idx[f] is not None]
    if not live:
        return out
    f0 = live[0]
    end = next((f for f in live if idx[f] >= L - 1), live[-1])  # the render's last frame reaches the shot here
    for f in live:
        a = min(1.0, (f - f0 + 1) / fade)
        if f > end:
            a *= max(0.0, 1.0 - (f - end) / (2 * fade))
        if a <= 0:
            continue
        out[f] = gam(lin(plate[f]) * (1 - a) + lin(render[idx[f]]) * a)
    return out


def time_scene(plate: np.ndarray, still: bool, render: np.ndarray, sp: ScenePlan, eff, max_frames: int,
               hit_shift: int = 0, threads: int = 8, voice_end: int | None = None) -> SceneTiming:
    """Put an in-scene render on the plate's timeline: measure where the effect visibly starts, move it onto
    the cue (minus the effect's lead), then a still plate becomes the render and a moving plate gets the
    delta (or, if that fails its check, the render). ``render`` must already be at the plate's size."""
    n = plate.shape[0]
    L = render.shape[0]
    warnings: list[str] = []
    region = effect_region(sp.guide.blobs, sp.guide.size, (plate.shape[2], plate.shape[1]))
    at0 = sp.events[0].at if sp.events else 0
    ref_frames = max(4, at0 - EARLIEST)  # the settled scene: before the model can have started the effect
    steady = registered(render, region, ref_frames)  # camera drift must not read as the effect
    share = changed_share(steady, region, ref_frames)
    painted = at0
    shown = list(range(len(sp.events)))  # the events the render shows on their own (a blast flashes for these)
    warp: list[tuple[int, int]] = []  # a rhythm's hits, each put on its own cue
    if eff.behaviour == "extinguish":  # what must land on the cue is the flame going out, not the smoke after it
        onset = flame_out(steady, region, ref_frames)
        if onset is None:
            onset = first_visible(share, ref_frames)
            warnings.append("the flame doesn't go out in the render, so it was timed on the first change it shows")
    elif len(sp.events) >= 2:  # a rhythm: every hit it shows on its own, moved together by their middle error
        wants = [ev.at - eff.lead_frames for ev in sp.events]
        curve = changed_share(steady, region, ref_frames, flame=True) if eff.fluid == "fire" else share
        onsets = hit_onsets(curve, wants, ref_frames)
        found = [(w, o) for w, o in zip(wants, onsets) if o is not None]
        shown = [k for k, o in enumerate(onsets) if o is not None] or [0]
        warp = warp_anchors(onsets, wants)
        if len(warp) >= 2:  # each hit on its own cue: the render plays a little faster or slower between them
            onset = warp[0][0] - (warp[0][1] - (at0 - eff.lead_frames))  # the first hit, by the first anchor's error
        elif found:
            warp = []
            off = int(round(float(np.median([w - o for w, o in found]))))
            onset = at0 - eff.lead_frames - off  # where the first hit shows, by that error
        else:
            warp = []
            onset = first_visible(share, ref_frames)
        if len(found) < len(wants):
            warnings.append(f"the model merged {len(wants) - len(found)} of the {len(wants)} hits into their neighbours")
    else:
        onset = onset_past_steps(steady, region, ref_frames, share)
    desired = painted - eff.lead_frames  # light beats sound: a burst shows a frame before its "BOOM"
    raw = None if onset is None else desired - onset
    auto = 0 if raw is None else int(np.clip(raw, -MAX_SHIFT, MAX_SHIFT))
    if onset is None:
        warnings.append("the render shows no clear change where the effect was painted, so it was not re-timed")
    elif raw != auto:
        warnings.append(f"the model put it {abs(raw)} frames {'early' if raw > 0 else 'late'}; moved by {abs(auto)}")
    shift = auto + hit_shift
    start = sp.start
    # A still plate is held for as long as the render needs; a moving one is as long as it is.
    end = start + hit_shift + int(np.ceil(on_cue(L, warp))) if warp else start + shift + L
    n_out = min(max(n, end), max_frames) if still else n
    if warp:
        index = [int(round(from_cue(f - start - hit_shift, warp))) for f in range(n_out)]
    else:
        index = [f - start - shift for f in range(n_out)]
    # where each event now shows: the auto shift put the model's events on their painted times, except for what
    # the clamp could not take out (err), so the camera reacts to the burst, not to the cue
    err = 0 if raw is None else raw - auto
    hits = [start + hit_shift + ev.at - eff.lead_frames - err for ev in sp.events]
    if warp:  # a rhythm put on its cues hit by hit: each shows where its own onset now lands
        hits = [start + hit_shift + (int(round(on_cue(o, warp))) if o is not None else ev.at - eff.lead_frames)
                for o, ev in zip(onsets, sp.events)]
    strengths = [ev.size * (0.5 + 0.5 * ev.peak) for ev in sp.events]
    sc = eff.scene
    if sp.dropped:
        warnings.append(f"{len(sp.dropped)} later hit(s) fall outside the {L}-frame render and were left out")
    stats = None
    if still:
        pick: list[int | None] = [int(np.clip(e, 0, L - 1)) for e in index]
        frames = render[pick]
        plate_view = np.broadcast_to(plate[0], (n_out,) + plate.shape[1:])
        idx_out = pick
    else:
        idx = [None if e < 0 else min(e, L - 1) for e in index]
        frames, stats = transfer(plate, render, idx, region, ref_frames, start, threads=threads)
        if not stats.accepted:
            warnings.append(f"used the render instead of your clip's own pixels: {stats.reason}")
            frames = _render_over_plate(plate, render, idx)
        plate_view, idx_out = plate, idx
    if onset is not None:  # no flash or shake for a burst that can't be found, nor for a hit merged into another
        frames = impact(frames, [hits[k] for k in shown], [strengths[k] for k in shown],
                        sc.shake if sc else 0.0, sc.flash if sc else 0.0)
    return SceneTiming(frames=frames, index=idx_out, plate_view=plate_view, onset=onset, painted=painted, raw=raw,
                       auto_shift=auto, shift=shift, transfer=stats, hits=hits, warnings=warnings + list(sp.notes),
                       merged=[k for k in range(len(sp.events)) if k not in shown], warp=warp)


def registered(render: np.ndarray, region: np.ndarray, ref_frames: int) -> np.ndarray:
    """The render with each frame registered onto its own settled scene (the frames before the effect), so
    that a slow drift or zoom of the model's camera doesn't count as change."""
    ref = np.median(render[: max(1, ref_frames)], axis=0).astype(np.uint8)
    static = ~region
    if static.mean() < 0.05:
        return render
    h, w = ref.shape[:2]
    out = np.empty_like(render)
    for i, f in enumerate(render):
        A = _ecc(f, ref, static)
        out[i] = f if np.allclose(A, np.eye(2, 3)) else cv2.warpAffine(
            f, A, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return out


EARLIEST = 9  # LTX-2.5 starts an effect up to about 8 frames before its painted keyframe
SHAKE_FRAMES = 18  # a blast shakes the phone for three quarters of a second
SHAKE_DECAY = 4.5  # frames for the shake to fall to 1/e
PEAK_AFTER = 2  # a fireball is at its brightest about this many frames after it shows
SHAKE_AFTER = 4  # the camera shakes once the blast has peaked (2-3 frames after the peak)
FLASH_TINT = np.array([1.0, 0.86, 0.66], np.float32)  # a warm flash, not a white one


def impact(frames: np.ndarray, hits: list[int], strengths: list[float], shake_amount: float,
           flash_amount: float) -> np.ndarray:
    """What a blast does to the camera, timed from each hit (the frame the burst shows): a warm exposure
    flash at its peak, then the shake. Returns ``frames`` untouched when there is nothing to do."""
    if not hits or (shake_amount <= 0 and flash_amount <= 0):
        return frames
    out = frames if frames.flags.writeable else np.array(frames)
    if flash_amount > 0:
        for f0, s in zip(hits, strengths):
            for t, k in ((0, 1.0), (1, 0.45)):
                f = f0 + PEAK_AFTER + t
                if 0 <= f < len(out):
                    out[f] = gam(lin(out[f]) * (1.0 + flash_amount * s * k * FLASH_TINT))
    return shake(out, [h + SHAKE_AFTER for h in hits], strengths, shake_amount)


def shake(frames: np.ndarray, hits: list[int], strengths: list[float], amount: float) -> np.ndarray:
    """A blast shakes the camera: a decaying 8-11 Hz wobble from each of ``hits`` (offset ``amount`` x frame
    width x the hit's strength at its peak), with a slight zoom so no edge ever shows. Returns ``frames``
    untouched when there is nothing to shake."""
    if amount <= 0 or not hits:
        return frames
    n, h, w = frames.shape[:3]
    dx, dy = np.zeros(n), np.zeros(n)
    for k, (f0, s) in enumerate(zip(hits, strengths)):
        a = amount * w * s
        for t in range(SHAKE_FRAMES):
            f = f0 + t
            if 0 <= f < n:
                env = a * np.exp(-t / SHAKE_DECAY)
                dx[f] += env * np.sin(2 * np.pi * 10.5 * t / 24 + 1.3 * k)
                dy[f] += env * 0.7 * np.cos(2 * np.pi * 8.0 * t / 24 + 0.4 * k)
    moved = np.nonzero(np.hypot(dx, dy) > 0.25)[0]
    if moved.size == 0:
        return frames
    out = np.array(frames) if not frames.flags.writeable else frames
    for f in moved:
        z = 1.0 + 2.0 * max(abs(dx[f]) / w, abs(dy[f]) / h) + 1e-3
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, z)
        M[:, 2] += (dx[f], dy[f])
        out[f] = cv2.warpAffine(frames[f], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def _plate_strokes(spec: ShotSpec) -> Strokes | None:
    """The drawing, in the cropped plate's coordinates (the same crop ``load_plate`` makes around the point).
    Parts drawn outside the crop are left out (a stroke that leaves the frame is split), not squashed onto
    its edge."""
    strokes = Strokes.parse(spec.strokes)
    if strokes is None:
        return None
    info = probe_video(spec.plate)
    landscape = info.width > info.height
    size = spec.plate_size or ((1920, 1080) if landscape else (1080, 1920))
    box = _crop_box(info.width, info.height, size[0], size[1], spec.point)
    scale = info.width / max(1, box[2] - box[0])  # a brush width in source widths -> in crop widths
    mapped = []
    for s in strokes.strokes:
        run: list[list[float]] = []
        for p in s["points"]:
            x, y = crop_point(p, info.width, info.height, box)
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                run.append([x, y])
            elif run:
                mapped.append({"points": run, "width": s["width"] * scale})
                run = []
        if run:
            mapped.append({"points": run, "width": s["width"] * scale})
    return Strokes.parse(mapped)


def make_scene_shot(spec: ShotSpec, renderer: Renderer, out_dir: Path, progress: Progress | None = None) -> ShotResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    eff = load_effect(spec.effect)
    if eff.scene is None:
        raise ValueError(f"effect {eff.id!r} has no scene mode")
    if not getattr(renderer, "scenes", False):
        raise ValueError(f"the {getattr(renderer, 'name', '?')} renderer can't render into the scene")

    say("reading", 0.03, "Reading your plate")
    plate, point, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    n, ph, pw = plate.shape[:3]
    still = is_still(plate)

    say("listening", 0.08, {"voice": "Listening to your performance", "clip": "Finding the sound in your clip",
                            "hold": "Reading your timing"}.get(spec.cue, "Reading the cue"))
    c = shot_cue(spec, n / FPS)
    cue = c.cue
    (out_dir / "cue.json").write_text(json.dumps(cue.to_json() | {"source": c.source}, indent=1))

    say("painting", 0.12, "Painting the cue into the scene")
    strokes = _plate_strokes(spec)
    if strokes is not None:
        point = strokes.source
    matte_fn = spec.matte or getattr(renderer, "matte", None) or grabcut_matte
    occluder = matte_fn(plate[0], point)
    if occluder is not None:
        Image.fromarray((np.clip(occluder, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out_dir / "matte.png")
    # sizes are designed for a prop about a third of the frame wide; a much smaller or bigger one scales a little
    # (a drawing already says how big it is)
    scale = (1.0 if strokes is not None else
             spec.scale if spec.scale is not None else float(np.clip(auto_scale(occluder, point, eff.behaviour), 0.8, 1.15)))
    start, length = plan_window(cue, eff, None if still else n, multi_hit=spec.multi_hit)
    sp = paint_scene_guide(cue, eff, point, plate[:1] if still else plate, start=start, length=length,
                           direction=spec.direction, scale=scale, size=scene_size(pw, ph), multi_hit=spec.multi_hit,
                           hold=str(spec.extra.get("hold", "pre")), strokes=strokes,
                           gaps=str(spec.extra.get("gaps", "hold")))
    if not sp.events:  # nothing to render: say so before spending a GPU render on it
        raise ValueError("the cue's hit is too close to the end of the clip (or outside it) to paint an effect; "
                         "cue it earlier or use a longer clip")
    write_guide(sp.guide, out_dir / "guide.mp4", fps=FPS)
    write_guide_preview(sp.guide, out_dir / "guide_preview.mp4", fps=FPS)
    (out_dir / "guide.json").write_text(json.dumps(sp.to_json(), indent=1))
    first = plate[min(max(start, 0), n - 1)]
    Image.fromarray(cv2.resize(first, sp.guide.size, interpolation=cv2.INTER_AREA)).save(out_dir / "start_frame.png")
    biggest = max((ev.size for ev in sp.events), default=1.0)
    prompt = scene_prompt(eff, spec.subject, sp.direction, len(sp.events), biggest, spec.describe)

    say("rendering", 0.18, f"Rendering the {eff.fluid} into your scene with LTX-2.5")
    seconds = renderer(out_dir / "guide.mp4", prompt, spec.seed, out_dir / "render.mp4", image=out_dir / "start_frame.png")
    receipt = {"renderer": renderer.name, "seconds": round(seconds, 2), "prompt": prompt, "model": MODEL,
               "start_frame": start, "length": length}
    return _finish_scene(spec, eff, plate, still, point, cue, c.source, c.samples, c.offset_s, sp, occluder, scale,
                         out_dir, say, t0, render_receipt=receipt)


def retime_scene_shot(src: Path, spec: ShotSpec, out_dir: Path, hit_shift: int = 0,
                      progress: Progress | None = None) -> ShotResult:
    """Move a finished scene shot by ``hit_shift`` frames without rendering again (the stop is painted into
    the render, so only the timing of the whole effect moves)."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    eff = load_effect(spec.effect)
    old = json.loads((src / "receipt.json").read_text())
    say("reading", 0.05, "Reading your plate")
    plate, point, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    for name in ("cue.json", "guide.json", "guide.mp4", "guide_preview.mp4", "render.mp4", "start_frame.png", "matte.png"):
        if (src / name).exists() and src != out_dir:
            shutil.copyfile(src / name, out_dir / name)
    c = shot_cue(spec, plate.shape[0] / FPS)
    cue = CueSheet.from_json(json.loads((src / "cue.json").read_text()))
    sp = ScenePlan.from_json(json.loads((src / "guide.json").read_text()), read_video(src / "guide.mp4").frames)
    if sp.strokes is not None:
        point = sp.strokes.source
    occluder = None
    if (src / "matte.png").exists():
        occluder = np.asarray(Image.open(src / "matte.png").convert("L"), dtype=np.float32) / 255.0
    say("compositing", 0.2, "Re-timing: no new render needed")
    prev = int(old.get("alignment", {}).get("retime_frames", 0))
    return _finish_scene(spec, eff, plate, bool(old["plate"]["still"]), point, cue, c.source, c.samples, c.offset_s,
                         sp, occluder, float(old.get("scale", 1.0)), out_dir, say, t0, hit_shift=prev + hit_shift,
                         render_receipt=old["render"], retimed={"from": src.name, "hit_shift_frames": hit_shift,
                                                                "stop_frame": None})


def _finish_scene(spec, eff, plate, still, point, cue, cue_source, samples, audio_offset, sp: ScenePlan, occluder,
                  scale, out_dir, say, t0, hit_shift: int = 0, render_receipt=None, retimed=None) -> ShotResult:
    n, ph, pw = plate.shape[:3]
    say("compositing", 0.72, "Timing the render to your cue")
    render = read_video(out_dir / "render.mp4").frames
    if render.shape[1:3] != (ph, pw):
        render = np.stack([cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA) for f in render])
    if not still:
        say("compositing", 0.76, "Putting what changed onto your footage")
    voice_end = cue.voice_end_frame
    st = time_scene(plate, still, render, sp, eff, int(round(spec.max_plate_s * FPS)), hit_shift, voice_end=voice_end)
    comp, plate_view, n_out = st.frames, st.plate_view, st.frames.shape[0]
    onset, auto, start = st.onset, st.auto_shift, sp.start
    at0 = st.painted
    index_of = lambda t: st.index[t]  # noqa: E731

    audio = place_audio(samples, audio_offset, n_out)
    comp_path = out_dir / "composite.mp4"
    write_video(comp_path, comp, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)

    say("finishing", 0.85, "Adding your face and the cue ribbon" if spec.face else "Adding the cue ribbon")
    faces = None
    if spec.face is not None:
        try:
            faces = _face_track(spec.face, n_out, spec.offset_s, spec.face_offset_s, min(ph, pw) // 3)
        except Exception:  # a broken face clip must not cost the shot
            faces = None
    env = np.zeros(n_out)
    m = min(n_out, len(cue.envelope))
    env[:m] = np.asarray(cue.envelope)[:m]
    final_path, breakdown_path = out_dir / "final.mp4", out_dir / "breakdown.mp4"

    def write_final() -> None:
        write_video(final_path, _finalise(comp, env, spec.label or eff.id.replace("_", " "), faces), fps=FPS,
                    audio=audio, crf=16, preset=FINISH_PRESET)

    def write_breakdown() -> None:
        label = "2 drawing, filled by the cue" if sp.strokes is not None else "2 cue-painted scene guide"
        b = _breakdown(plate_view, sp.guide.frames, render, comp, None, None, sp.guide.keyframes,
                       guide_label=label, index_of=index_of,
                       labels=("3 LTX-2.5, rendered into the scene", "4 the shot, timed to the cue"))
        write_video(breakdown_path, b, fps=FPS, audio=audio, crf=18, preset=FINISH_PRESET)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for fut in [pool.submit(write_final), pool.submit(write_breakdown)]:
            fut.result()

    region = effect_region(sp.guide.blobs, sp.guide.size, (pw, ph))
    stops = eff.voice_gate == "both"
    raw = st.raw
    first = start + at0 - eff.lead_frames + hit_shift - (0 if raw is None else raw - auto)  # where it now shows
    stop = measure_stop(comp, region, voice_end, eff.fluid == "fire", first) if stops else None
    receipt = {
        "cuesync": __version__,
        "mode": "scene",
        "effect": eff.to_json(),
        "inputs": {"plate_sha": _sha(spec.plate), "voice_sha": _sha(spec.voice), "point": list(spec.point),
                   "seed": spec.seed, "offset_s": spec.offset_s, "face": spec.face is not None,
                   "direction": sp.direction, "subject": spec.subject, "cue": cue_source,
                   "drawn": sp.strokes is not None, "describe": spec.describe,
                   "holds": [list(h) for h in spec.holds] if cue_source == "hold" and spec.holds else None},
        "voice": {"onsets_s": [round(t, 3) for t in cue.onsets_s], "hero_onset_s": round(cue.hero_onset_s, 3),
                  "active_intervals": [list(iv) for iv in cue.intervals()], "voice_end_frame": voice_end,
                  "peak_frame": cue.peak_frame, "source": cue_source},
        "guide": {k: v for k, v in sp.to_json().items() if k != "blobs"},
        "render": render_receipt,
        "plate": {"still": bool(still), "frames": n, "shot_frames": n_out,
                  "size": [pw, ph], "point": [round(point[0], 4), round(point[1], 4)]},
        "hits": [{"frame": h, "size": round(ev.size, 3), "strength": round(ev.peak, 3), "merged": k in st.merged}
                 for k, (h, ev) in enumerate(zip(st.hits, sp.events))] if len(sp.events) > 1 else None,
        "dropped_hits": sp.dropped,
        "scale": round(scale, 3),
        "matte": {"source": "custom" if spec.matte is not None else "auto", "found": occluder is not None,
                  "coverage": None if occluder is None else round(float(occluder.mean()), 4)},
        "alignment": {"mode": "scene", "painted_hit": at0, "model_onset": onset, "auto_shift": auto,
                      "clamped": raw is not None and raw != auto, "retime_frames": hit_shift,
                      "offset": start + st.shift, "lead": eff.lead_frames,
                      # the shot frame of the (biggest) hit: where the re-time handle sits
                      "hit_frame": start + at0 - eff.lead_frames + hit_shift,
                      # a rhythm put on its cues hit by hit: [render frame of each hit's onset, its cue frame]
                      "warp": [list(a) for a in st.warp] or None},
        "measured": {
            # how early (+) or late the model put the effect against the cue (measured on the render)
            "model_early_frames": raw,
            # what the re-timing could not take out (the model was further off than MAX_SHIFT): the error left by
            # the onset measured on the render, not a second measurement of the finished shot
            "residual_frames": None if raw is None else raw - auto,
            "stops_with_voice": stops,  # what the effect is meant to do
            # what it did: frames from the end of the cue until the effect is visibly gone (measured on the shot)
            "stop_latency_frames": None if stop is None or voice_end is None else stop - voice_end,
            "stopped_within_plate": None if not stops else stop is not None,
        },
        "transfer": None if st.transfer is None else asdict(st.transfer),
        "warnings": st.warnings + plate_notes(spec.plate, spec.max_plate_s),
        "retimed": retimed,
        "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=breakdown_path, receipt=receipt)
