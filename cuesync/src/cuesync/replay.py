"""Perform a finished scene shot again, with a new take and no new render.

A scene shot's render already holds each source's effect: its fire, its light on the scene, what it leaves behind
(a candle blown out stays out). Measured again the way the shot was made (the same timing and transfer, without
the camera's flash and shake), each hit's change against the plate becomes a response that can be laid onto the
plate at any frame. A new take (another rhythm, another voice) fires those responses on its own hits: hit j plays
the render's hit j at its source (by loudness when the counts match, see ``assign``; the next hit at a source takes over from the last within a quarter of a
second), a little brighter or dimmer as the new hit is louder or quieter than the one it replaces; then the shot
gets its flash, shake, Foley and cue ribbon. (Shrinking a rendered burst for a quiet hit left ghosts of it.) It takes seconds on a CPU. What it can't change is an effect's shape or how long it
lasts: those are the render's. A moving clip needs a fixed camera (each response is laid where it was rendered).

Several shots on the same still plate play as one scene, each with its own responses: one render per source
(tap the little jack-o'-lantern and say "pf", tap the big one and say "BOOM"), then any rhythm plays all of them,
with no render at all. Each render then holds one effect only, so nothing has to be shared out between sources by
nearness (a fireball that drifts over the next source in a one-render call and response is split between them).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import __version__
from .effects import load_effect
from .media import read_video, write_video
from .pipeline import (FINISH_PRESET, FPS, SOUND, Progress, ShotResult, ShotSpec, _finalise, foley, load_plate, mix_foley,
                       place_audio, plate_box, shot_cue)
from .scene import ScenePlan, performer_in
from .scene_shot import SHAKE_AFTER, impact, shake, time_scene
from .transfer import gam, lin

BRIGHT = (0.75, 1.1)  # a quieter or louder hit plays its response this much dimmer or brighter
BUILD = 3  # frames before its visible onset a response starts (a fire grows out of nothing)
HANDOVER = 6  # frames over which a source's previous response gives way to the next hit's
SOFT = 0.08  # share of the short side over which two sources' halves of the frame blend
SAME_SPOT = 0.03  # sources of shots performed together this close (share of the frame) are one source: they hand over
MISMATCH = 0.08  # linear light: where one render's plate-before-its-hit differs from the shared one by more...
STRONG = 0.2  # ...and every render's differs from the plate itself by more than this, the plate is the shared one


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


def _measure(src: Path, plate: np.ndarray, still: bool, max_frames: int) -> dict:
    """One finished scene shot measured again: its frames without the camera's flash and shake, the plate before
    its first hit (linear light; a still plate's), its effect, its hits (shown frame, loudness, its own source) and
    where its sources are. A performer's hand is kept as the shot kept it (its matte, see time_scene)."""
    n, ph, pw = plate.shape[:3]
    old = json.loads((src / "receipt.json").read_text())
    eff = load_effect(old["effect"]["id"])
    sp = ScenePlan.from_json(json.loads((src / "guide.json").read_text()), read_video(src / "guide.mp4").frames)
    render = read_video(src / "render.mp4").frames
    if render.shape[1:3] != (ph, pw):
        render = np.stack([cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA) for f in render])
    point = tuple(old["plate"]["point"])
    hand = None
    if performer_in(old["inputs"].get("subject")) and (src / "performer.png").exists():
        hand = np.asarray(Image.open(src / "performer.png").convert("L"), dtype=np.float32) / 255.0
    st = time_scene(plate, still, render, sp, eff, max_frames, int(old["alignment"].get("retime_frames", 0)), impacts=False,
                    performer=point if performer_in(old["inputs"].get("subject")) else None, hand=hand)
    comp = st.frames
    shown = [k for k in range(len(sp.events)) if k not in st.merged] or [0]
    olds = [(st.hits[k], sp.events[k].peak if sp.events else 1.0, sp.source_of(k)) for k in shown]
    pre = int(np.clip(min(f for f, _, _ in olds) - BUILD - 1, 0, len(comp) - 1))
    return {"old": old, "eff": eff, "comp": comp, "calm": lin(comp[pre]) if still else None, "olds": olds,
            "weights": source_weights(sp.sources, pw, ph), "points": [tuple(p) for p in (sp.sources or [point])]}


def _shared_calm(shots: list[dict], plate0: np.ndarray) -> np.ndarray:
    """Shots performed together: the plate before any hit that they share, and each shot's own one made to agree
    with it. Per pixel, the render whose frame before its hit is nearest the plate (a flame one render painted
    where none burns is in no other), or the plate where all of them are far from it; a shot's own frame gives way
    to the shared one wherever they differ, so its change there is everything it shows (added onto a frame that
    lacked its flame, a burst's colours went negative and turned green)."""
    calms = np.stack([sh["calm"] for sh in shots])
    dev = np.abs(calms - plate0).max(-1)
    base = np.take_along_axis(calms, dev.argmin(0)[None, ..., None], 0)[0]
    far = dev.min(0) > STRONG
    base[far] = plate0[far]
    for sh in shots:
        m = (np.abs(sh["calm"] - base).max(-1) > MISMATCH).astype(np.uint8)
        m = cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3)[..., None]
        sh["calm"] = sh["calm"] * (1 - m) + base * m
    return base


def replay_scene_shot(src: Path | Sequence[Path], spec: ShotSpec, out_dir: Path, progress: Progress | None = None) -> ShotResult:
    """The scene shot in ``src`` (or the shots, all on the plate in ``spec``) performed again by the cue in
    ``spec`` (its plate, effect and point are the first shot's own); see the module docstring."""
    srcs = [Path(src)] if isinstance(src, (str, Path)) else [Path(x) for x in src]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    first = json.loads((srcs[0] / "receipt.json").read_text())
    say("reading", 0.05, "Reading your plate")
    plate, point, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    n, ph, pw = plate.shape[:3]
    still = bool(first["plate"]["still"])
    max_frames = int(round(spec.max_plate_s * FPS))
    say("compositing", 0.2, "Taking each hit's effect from the render (no new render)")
    box = plate_box(spec.plate, tuple(first["inputs"]["point"]), spec.plate_size)
    for x in srcs[1:]:
        rc = json.loads((x / "receipt.json").read_text())
        if not (still and rc["plate"]["still"]) or rc["inputs"]["plate_sha"] != first["inputs"]["plate_sha"]:
            raise ValueError("shots performed together must all be on the same still plate")
        if max(abs(u - v) for u, v in zip(plate_box(spec.plate, tuple(rc["inputs"]["point"]), spec.plate_size), box)) > 0.01:
            raise ValueError("shots performed together must have been cropped alike from the clip")
    shots = [_measure(x, plate, still, max_frames) for x in srcs]
    calm = _shared_calm(shots, lin(plate[0])) if len(shots) > 1 else shots[0]["calm"]
    olds = []  # (shown frame, loudness, source across all the shots, the shot it is from)
    weights, spots = [], []  # per source across the shots: its share of each pixel, where it is
    for k, sh in enumerate(shots):
        ids = []
        for j, p in enumerate(sh["points"]):
            same = next((g for g, q in enumerate(spots) if np.hypot(p[0] - q[0], p[1] - q[1]) < SAME_SPOT), None)
            if same is None:
                spots.append(p)
                weights.append(sh["weights"][min(j, len(sh["weights"]) - 1)])
                same = len(spots) - 1
            ids.append(same)
        olds += [(f, peak, ids[min(j, len(ids) - 1)], k) for f, peak, j in sh["olds"]]

    def scene(t: int) -> np.ndarray:  # the plate with nothing happening, on the shot's timeline
        return calm if still else lin(plate[int(np.clip(t, 0, n - 1))])

    say("reading", 0.5, "Reading your new take")
    c = shot_cue(spec, max_frames / FPS)
    takes = new_hits(c.cue, max_frames)
    if not takes:
        raise ValueError("the new take has no hit in the shot")
    order = assign(takes, [o[:3] for o in olds])
    plays = [(f - shots[olds[order[j]][3]]["eff"].lead_frames, order[j], loud / max(olds[order[j]][1], 1e-3))
             for j, (f, loud) in enumerate(takes)]
    # long enough for the last response to play out (its shot's frames after its hit), within the plate's limit
    n_out = n if not still else int(np.clip(max(s_ + len(shots[olds[i][3]]["comp"]) - olds[i][0] for s_, i, _ in plays),
                                            max(len(sh["comp"]) for sh in shots), max_frames))

    def change(show: int, i: int, r: float, t: int) -> np.ndarray:  # what render hit i does at t, played from show
        sh = shots[olds[i][3]]
        f_old = int(np.clip(olds[i][0] + (t - show), 0, len(sh["comp"]) - 1))
        base = sh["calm"] if still else lin(plate[int(np.clip(f_old, 0, n - 1))])
        return float(np.clip(r ** 0.5, *BRIGHT)) * (lin(sh["comp"][f_old]) - base)

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
            acc += weights[src_k][..., None] * d
        return gam(acc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = np.stack(list(pool.map(frame, range(n_out))))
    shows = [p[0] for p in plays]
    strengths = [float(np.clip(p[2], 0.2, 1.0)) for p in plays]
    effs = {sh["eff"].id: sh["eff"] for sh in shots}
    of = [shots[olds[i][3]]["eff"].id for _, i, _ in plays]  # each play's effect
    scenes = {k: e.scene for k, e in effs.items()}
    shake_max = max((sc.shake for sc in scenes.values() if sc), default=0.0)
    for k, sc in scenes.items():  # each effect's flash on its own hits, then one shake for all of them
        g = [j for j, e in enumerate(of) if e == k]
        frames = impact(frames, [shows[j] for j in g], [strengths[j] for j in g], 0.0, sc.flash if sc else 0.0)
    if shake_max > 0:
        frames = shake(frames, [h + SHAKE_AFTER for h in shows],
                       [strengths[j] * (scenes[e].shake if scenes[e] else 0.0) / shake_max for j, e in enumerate(of)], shake_max)

    say("finishing", 0.8, "Adding the sound and the cue ribbon")
    env = np.zeros(n_out)
    m = min(n_out, len(c.cue.envelope))
    env[:m] = np.asarray(c.cue.envelope)[:m]
    owner = np.full(n_out, -1)  # which play each frame of the take belongs to: the latest started (2 frames early)
    for j in np.argsort(shows):
        owner[max(0, shows[j] - 2):] = j
    sounds = []
    for k, e in effs.items():
        mine = [j for j, x in enumerate(of) if x == k]
        if not mine:  # an effect no new hit plays makes no sound
            continue
        env_k = np.where(np.isin(owner, mine), env, 0.0)
        sounds.append(foley(e, n_out, env_k, [shows[j] for j in mine], [strengths[j] for j in mine], None, spec.seed))
    audio = mix_foley(place_audio(c.samples, c.offset_s, n_out), sounds)
    old = shots[0]["old"]
    comp_path, final_path = out_dir / "composite.mp4", out_dir / "final.mp4"
    write_video(comp_path, frames, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)
    label = spec.label or " + ".join(k.replace("_", " ") for k in effs)
    write_video(final_path, _finalise(frames, env, label, None), fps=FPS, audio=audio, crf=16, preset=FINISH_PRESET)
    (out_dir / "cue.json").write_text(json.dumps(c.cue.to_json() | {"source": c.source}, indent=1))
    receipt = old | {
        "cuesync": __version__,
        "voice": {"onsets_s": [round(t, 3) for t in c.cue.onsets_s], "hero_onset_s": round(c.cue.hero_onset_s, 3),
                  "voice_end_frame": c.cue.voice_end_frame, "source": c.source},
        "hits": [{"frame": s_, "plays_hit": i + 1, "source": olds[i][2], "loudness": round(r, 3)} for s_, i, r in plays],
        "alignment": old["alignment"] | {"hit_frame": shows[0]},
        "plate": old["plate"] | {"shot_frames": n_out},
        "replayed": {"from": srcs[0].name, "together": [x.name for x in srcs[1:]], "renders": len(srcs), "render_hits": len(olds),
                     "effects": list(effs),
                     "new_hits": len(plays), "no_new_render": True,
                     "how": "each new hit plays one of the renders' hits at its source (by loudness when the counts match, "
                            "else in turn), brighter or dimmer by how loud it is"},
        "sound": SOUND,
        "auto": None,  # the new take played the renders' effects; it picked none
        "retimed": None,
        "warnings": [],
        "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=None, receipt=receipt)
