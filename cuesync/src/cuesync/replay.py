"""Perform a finished scene shot again, with a new take and no new render.

A scene shot's render already holds each source's effect: its fire, its light on the scene, what it leaves behind
(a candle blown out stays out). Measured again the way the shot was made (the same timing and transfer, without
the camera's flash and shake), each hit's change against the plate becomes a response that can be laid onto the
plate at any frame. A new take (another rhythm, another voice) fires those responses on its own hits: hit j plays
the render's hit j at its source (by loudness when the counts match, see ``assign``; the next hit at a source takes over from the last within a quarter of a
second), a little brighter or dimmer as the new hit is louder or quieter than the one it replaces; then the shot
gets its flash, shake, Foley and cue ribbon. (Shrinking a rendered burst for a quiet hit left ghosts of it.) It takes seconds on a CPU. What it can't change is an effect's shape or how long it
lasts: those are the render's. A moving clip needs a fixed camera (each response is laid where it was rendered).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .effects import load_effect
from .media import read_video, write_video
from .pipeline import FINISH_PRESET, FPS, SOUND, Progress, ShotResult, ShotSpec, _finalise, load_plate, place_audio, shot_cue, soundtrack
from .scene import ScenePlan
from .scene_shot import impact, time_scene
from .transfer import gam, lin

BRIGHT = (0.75, 1.1)  # a quieter or louder hit plays its response this much dimmer or brighter
BUILD = 3  # frames before its visible onset a response starts (a fire grows out of nothing)
HANDOVER = 6  # frames over which a source's previous response gives way to the next hit's
SOFT = 0.08  # share of the short side over which two sources' halves of the frame blend


def source_weights(sources: list | None, w: int, h: int) -> np.ndarray:
    """(n, h, w): how much of each pixel's change belongs to each source, by nearness, blending softly between
    them (a source's fire lights its neighbour's side less than its own). One source has all of it."""
    if not sources or len(sources) < 2:
        return np.ones((1, h, w), np.float32)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d = np.stack([np.hypot(xx - x * w, yy - y * h) for x, y in sources])
    e = np.exp(-(d - d.min(0)) / (SOFT * min(w, h)))
    return (e / e.sum(0)).astype(np.float32)


def new_hits(cue, n: int) -> list[tuple[int, float]]:
    """(frame, loudness) of each hit of a take: its distinct hits, or its one onset."""
    env = np.asarray(cue.envelope, np.float64)
    peak = lambda f: float(env[max(0, f): max(0, f) + 8].max()) if 0 <= f < env.size else 0.0  # noqa: E731
    hits = [h.frame for h in cue.hits()] or [cue.frame_of(cue.hero_onset_s)]
    return [(f, peak(f)) for f in hits if 0 <= f < n]


def assign(takes: list[tuple[int, float]], olds: list[tuple[int, float, int]]) -> list[int]:
    """Which of the render's hits each new hit plays. As many hits as the render: by loudness, so the loudest sound
    fires the biggest effect wherever it falls in the take ("BOOM, pf, pf" blows up the big jack-o'-lantern first).
    Otherwise in turn, starting again after the last."""
    if len(takes) == len(olds) > 1:
        new_rank = sorted(range(len(takes)), key=lambda j: (-takes[j][1], j))
        old_rank = sorted(range(len(olds)), key=lambda i: (-olds[i][1], i))
        out = [0] * len(takes)
        for j, i in zip(new_rank, old_rank):
            out[j] = i
        return out
    return [j % len(olds) for j in range(len(takes))]


def replay_scene_shot(src: Path, spec: ShotSpec, out_dir: Path, progress: Progress | None = None) -> ShotResult:
    """The scene shot in ``src`` performed again by the cue in ``spec`` (its plate, effect and point are the shot's
    own); see the module docstring."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    eff = load_effect(spec.effect)
    old = json.loads((src / "receipt.json").read_text())
    say("reading", 0.05, "Reading your plate")
    plate, point, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    n, ph, pw = plate.shape[:3]
    still = bool(old["plate"]["still"])
    sp = ScenePlan.from_json(json.loads((src / "guide.json").read_text()), read_video(src / "guide.mp4").frames)
    render = read_video(src / "render.mp4").frames
    if render.shape[1:3] != (ph, pw):
        render = np.stack([cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA) for f in render])
    say("compositing", 0.2, "Taking each hit's effect from the render (no new render)")
    max_frames = int(round(spec.max_plate_s * FPS))
    st = time_scene(plate, still, render, sp, eff, max_frames, int(old["alignment"].get("retime_frames", 0)), impacts=False)
    comp = st.frames
    shown = [k for k in range(len(sp.events)) if k not in st.merged] or [0]
    olds = [(st.hits[k], sp.events[k].peak if sp.events else 1.0, sp.source_of(k)) for k in shown]
    first_old = min(f for f, _, _ in olds)
    pre = int(np.clip(first_old - BUILD - 1, 0, len(comp) - 1))
    calm = lin(comp[pre]) if still else None

    def scene(t: int) -> np.ndarray:  # the plate with nothing happening, on the shot's timeline
        return calm if still else lin(plate[int(np.clip(t, 0, n - 1))])

    weights = source_weights(sp.sources, pw, ph)

    say("reading", 0.5, "Reading your new take")
    c = shot_cue(spec, max_frames / FPS)
    takes = new_hits(c.cue, max_frames)
    if not takes:
        raise ValueError("the new take has no hit in the shot")
    lead = eff.lead_frames
    plays = []  # (frame it shows, the render's hit it plays, its loudness against that hit's)
    for j, (f, loud) in enumerate(takes):
        i = assign(takes, olds)[j]
        plays.append((f - lead, i, loud / max(olds[i][1], 1e-3)))
    last_show, last_old = max(p[0] for p in plays), max(f for f, _, _ in olds)
    n_out = n if not still else int(np.clip(last_show + (len(comp) - last_old), len(comp), max_frames))

    def change(show: int, i: int, r: float, t: int) -> np.ndarray:  # what render hit i does at t, played from show
        f_old = int(np.clip(olds[i][0] + (t - show), 0, len(comp) - 1))
        d = float(np.clip(r ** 0.5, *BRIGHT)) * (lin(comp[f_old]) - scene(f_old))
        return d

    def frame(t: int) -> np.ndarray:
        acc = scene(t).copy()
        started: dict[int, list[tuple[int, int, float]]] = {}  # per source: the hits already playing there
        for show, i, g in sorted(plays):
            if show - BUILD <= t:
                started.setdefault(olds[i][2], []).append((show, i, g))
        for src_k, hits in started.items():
            d = change(*hits[-1], t)
            fade = 1.0 - (t - hits[-1][0]) / HANDOVER
            if len(hits) > 1 and fade > 0:  # the hit before gives way: the larger change of the two shows, not both
                prev = fade * change(*hits[-2], t)  # (added, a second burst over the first doubled it; cut, it vanished)
                d = np.where(np.abs(prev) > np.abs(d), prev, d)
            acc += weights[min(src_k, len(weights) - 1)][..., None] * d
        return gam(acc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = np.stack(list(pool.map(frame, range(n_out))))
    sc = eff.scene
    shows = [p[0] for p in plays]
    strengths = [float(np.clip(p[2], 0.2, 1.0)) for p in plays]
    frames = impact(frames, shows, strengths, sc.shake if sc else 0.0, sc.flash if sc else 0.0)

    say("finishing", 0.8, "Adding the sound and the cue ribbon")
    env = np.zeros(n_out)
    m = min(n_out, len(c.cue.envelope))
    env[:m] = np.asarray(c.cue.envelope)[:m]
    audio = soundtrack(place_audio(c.samples, c.offset_s, n_out), eff, n_out, env, shows, strengths, None, spec.seed)
    comp_path, final_path = out_dir / "composite.mp4", out_dir / "final.mp4"
    write_video(comp_path, frames, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)
    write_video(final_path, _finalise(frames, env, spec.label or eff.id.replace("_", " "), None), fps=FPS, audio=audio,
                crf=16, preset=FINISH_PRESET)
    (out_dir / "cue.json").write_text(json.dumps(c.cue.to_json() | {"source": c.source}, indent=1))
    receipt = old | {
        "cuesync": __version__,
        "voice": {"onsets_s": [round(t, 3) for t in c.cue.onsets_s], "hero_onset_s": round(c.cue.hero_onset_s, 3),
                  "voice_end_frame": c.cue.voice_end_frame, "source": c.source},
        "hits": [{"frame": s, "plays_hit": i + 1, "source": olds[i][2], "loudness": round(r, 3)} for s, i, r in plays],
        "alignment": old["alignment"] | {"hit_frame": shows[0]},
        "plate": old["plate"] | {"shot_frames": n_out},
        "replayed": {"from": src.name, "render_hits": len(olds), "new_hits": len(plays), "no_new_render": True,
                     "how": "each new hit plays the render's hit in turn, at its source, brighter or dimmer by how loud it is"},
        "sound": SOUND,
        "retimed": None,
        "warnings": [],
        "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=None, receipt=receipt)
