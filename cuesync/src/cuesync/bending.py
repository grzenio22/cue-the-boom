"""Element bending: a performer's hand in the shot, an element out of it on the beat.

The hand's track (``hand_track.json``, from ``cuesync.hand_track``: where the emitter is, which way it points, how big
the hand is, its gesture and its strikes, frame by frame) meets the performance here:

- **Which element** (``bending_effect``): what the sound is (``gestures.heard``: a boom, a whoosh, a hiss, a puff) and
  what the hand is doing at the hit. Two fingers or a pointing finger with a long "ffff" or "fwooom" is a flamethrower;
  an open palm hissing is steam (or a blast of air, without steam); a fist's strike with a "BOOM" is a burst, a palm's
  too (a fireball out of the palm); an unvoiced "pf" from a palm is a blast of air; a snap blows out the candle you
  tapped. A combo of booms is a burst per strike. A hand doing nothing recognisable leaves it to the sound alone.
- **When** (``snap_launch``): the element goes off on the arm's lock (the strike's extension) when the voice's hit is
  close to it, since a punch that fires before it lands reads wrong. The window is lopsided because a sound heard before
  the sight of its cause is noticed sooner than one heard after: the motion may come up to 4 frames before the voice's hit
  (the voice then lags by at most 167 ms) but only 1 frame after (it leads by at most 42 ms). Outside it, the voice wins
  and the shot says the two disagree.
- **Where and how** (``make_bending_shot``): the ainvfx-fluid IC-LoRA learned from fires and smoke that stay put, with
  guides on an 8-frame grid over black. A guide that chased a moving hand would be out of what it knows (and would only
  move every third of a second). So the element is rendered where the hand is on the strike, over black, as a static
  source; then it is carried: each frame moved, turned and scaled with the tracked emitter (its base at once, its tip a
  few frames late, the way a flame trails a moving hand), slid in time so its ignition lands on the strike, added onto
  the clip in linear light (smoke laid over), and the hand is lit from where the emitter is now, not where it was.

Every decision lands in the receipt with what it was made from.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .gestures import effect_for_kind

POINTING = ("Victory", "Pointing_Up")  # two fingers, or one: a jet from the fingertips
SNAP_BEFORE, SNAP_AFTER = 4, 1  # frames at 24 fps the strike may come before / after the voice's hit and still take it
COMBO_S = 1.5  # booms closer together than this are one combo: a burst per strike

BENDING_PROMPTS = {  # the element alone, over black, in the LoRA's own wording ("ainvfxfluid, ... isolated on ...")
    "fire_breath": "ainvfxfluid, a narrow jet of bright orange-yellow flame shooting {towards}, a white-hot core near the "
                   "source, licking turbulent edges and embers, isolated on a pure black background",
    "fire_burst": "ainvfxfluid, a short explosive ball of clean bright flame bursting {towards} from a point, a white-hot "
                  "core, quickly dissipating into embers, isolated on a pure black background",
    "flame_plume": "ainvfxfluid, a tall clean flame rising from a point, a bright core and licking edges, isolated on a "
                   "pure black background",
    "smoke_puff": "ainvfxfluid, a fast puff of pale grey smoke pushed {towards} from a point, wispy edges, dissipating, "
                  "isolated on a pure black background",
    "steam_jet": "ainvfxfluid, a pressurized jet of white steam and mist shooting {towards}, billowing and thinning, "
                 "isolated on a pure black background",
}
LAG_FRAMES = {"fire_breath": 4, "fire_burst": 2, "flame_plume": 3, "smoke_puff": 6, "steam_jet": 5}  # the tip's delay
LENGTH = 6.0  # the element's reach in hand sizes (wrist to middle knuckle): a jet of about six hands
IGNITION = 0.15  # an element has ignited once its light is this share of its brightest frame's
LOST_FADE = 4  # frames over which an element fades out once the hand has been lost for more than 8
CANVAS = 16  # the guide's ground: a dark grey, not black, so a keyframe before the hit says "nothing here yet" (over
# pure black the model took the empty keyframes as unconstrained and had steam there from the first frame)
PREGLOW = 4  # frames of the render's build-up shown before its ignition (a spark-up, not an element that pops on);
# earlier ones are dropped (a smoke render can start drifting long before its puff)


def load_track(path: str | Path) -> dict:
    """A hand_track.json (see docs/bending_contracts.md)."""
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(t.get("version", 1)) != 1:
        raise ValueError(f"hand track version {t.get('version')} is not one this build reads")
    return t


def gesture_at(track: dict, frame: int, around: int = 3) -> str | None:
    """The hand's gesture at ``frame``: the most frequent recognised one within +-``around`` frames (a single frame's
    label flickers), or None when nothing is recognised there or the hand is lost."""
    labels = track.get("gesture") or []
    src = track.get("src") or []
    lo, hi = max(0, frame - around), min(len(labels), frame + around + 1)
    seen = [labels[i] for i in range(lo, hi) if labels[i] not in (None, "None") and (i >= len(src) or src[i] != "lost")]
    return max(set(seen), key=seen.count) if seen else None


def bending_effect(kind: str | None, gesture: str | None, hits: int = 1, span_s: float = 0.0, snap: bool = False,
                   lit: bool = False, steam: bool = True) -> tuple[str | None, str]:
    """(effect, why) for a hand in the shot. ``kind``: the take's sound (gestures.heard); ``gesture``: the hand's at
    the hit (gesture_at); ``hits``/``span_s``: how many hits the take has and how long from the first to the last;
    ``snap``: the track saw a finger snap; ``lit``: a flame burns at the tapped spot; ``steam``: steam may be picked.
    None when there is no sound to read (the caller falls back to its own default)."""
    if snap:
        return "blow_out", "a snap: the candle goes out"
    if kind is None:
        return None, "no sound to read"
    combo = kind == "boom" and hits >= 2 and span_s <= COMBO_S
    if gesture in POINTING:
        if kind in ("hiss", "whoosh"):
            return "fire_breath", f"a long {kind} from pointed fingers: a flamethrower"
        if kind == "boom":
            return ("fire_breath", "strikes from the fingertips") if combo else ("fire_burst", "a boom from the fingertips")
    if gesture == "Open_Palm":
        if kind == "hiss":
            return ("steam_jet", "an open palm hissing: steam") if steam else ("smoke_puff", "an open palm hissing: a blast of air")
        if kind == "whoosh":
            return "flame_plume", "an open palm and a swelling voice: a flame rising from it"
        if kind == "boom":
            return "fire_burst", "a palm strike with a boom: a fireball" + (" per strike" if combo else "")
        return "smoke_puff", "an open palm and a puff: a blast of air"
    if gesture == "Closed_Fist" and kind in ("boom", "puff"):
        return "fire_burst", "a fist's strike: a burst" + (" per strike" if combo else "")
    why = "the hand's gesture isn't one we read" if gesture is None else f"{gesture} with a {kind}: by the sound alone"
    return effect_for_kind(kind, True, lit, aimed=True, rhythm=hits >= 2), why


def snap_launch(cue_frame: int, events: list[dict], fps: float = 24.0, kinds: tuple = ("extension", "open")) -> dict:
    """Where the element goes off for a voice hit at ``cue_frame``: the nearest strike event (an arm's lock, a palm
    opening) when it falls from SNAP_BEFORE frames before the hit to SNAP_AFTER after (scaled to ``fps``), else the hit
    itself, flagged as a mismatch. Returns {"cue", "motion", "frame", "used", "mismatch"}."""
    k = fps / 24.0
    strikes = [int(e["frame"]) for e in events if e.get("kind") in kinds]
    m = min(strikes, key=lambda f: abs(f - cue_frame)) if strikes else None
    ok = m is not None and cue_frame - SNAP_BEFORE * k <= m <= cue_frame + SNAP_AFTER * k
    return {"cue": int(cue_frame), "motion": m, "frame": int(m if ok else cue_frame), "used": "motion" if ok else "cue",
            "mismatch": m is not None and not ok}


# ------------------------------------------------------------------ carrying the element on the hand

def emitter_at(track: dict, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(position px, unit direction, hand size px) of the emitter at frame ``k`` (clamped to the track)."""
    k = int(np.clip(k, 0, len(track["xy"]) - 1))
    u = np.asarray(track["dir"][k], np.float64)
    return np.asarray(track["xy"][k], np.float64), u / max(float(np.hypot(*u)), 1e-6), float(track["scale"][k])


def angle_of(u) -> float:
    """A unit direction as degrees clockwise from straight up (the studio's ``direction``)."""
    return math.degrees(math.atan2(float(u[0]), -float(u[1])))


def towards_words(direction: float) -> str:
    a = math.radians(direction)
    ux, uy = math.sin(a), -math.cos(a)
    return "upwards" if uy < -0.7 else "downwards" if uy > 0.7 else "to the left" if ux < 0 else "to the right"


def ignition_frame(element: np.ndarray, floor: int) -> int | None:
    """The element frame where it ignites: the first whose light above its black is IGNITION of the brightest's."""
    lum = np.array([max(0.0, float(f[::4, ::4].astype(np.float32).mean()) - floor) for f in element])
    if lum.max() <= 0:
        return None
    return int(np.argmax(lum >= IGNITION * lum.max()))


def pose_inverse(track: dict, k_hit: int, t: int) -> np.ndarray:
    """2x3: plate frame ``t``'s pixel -> the rendered element's pixel. The element was rendered with the emitter as it
    was at ``k_hit``; at ``t`` it is moved, turned and scaled with the emitter (a similarity transform)."""
    p0, u0, s0 = emitter_at(track, k_hit)
    p1, u1, s1 = emitter_at(track, t)
    turn = math.atan2(u1[1], u1[0]) - math.atan2(u0[1], u0[0])
    k = s1 / max(s0, 1e-6)
    c, s = math.cos(turn) * k, math.sin(turn) * k
    fwd = np.array([[c, -s, p1[0] - (c * p0[0] - s * p0[1])], [s, c, p1[1] - (s * p0[0] + c * p0[1])]], np.float64)
    return cv2.invertAffineTransform(fwd)


def warp_element(frame: np.ndarray, track: dict, k_hit: int, t: int, lag: int) -> np.ndarray:
    """An element frame (float32, over black, at the plate's size) carried onto plate frame ``t``: its base follows the
    emitter now, a point further along it follows where the emitter was up to ``lag`` frames before (a flame's tip
    trails the hand). One remap, each pixel taking the lag level of how far along the element it lies."""
    h, w = frame.shape[:2]
    p0, u0, s0 = emitter_at(track, k_hit)
    reach = max(LENGTH * s0, 1.0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    maps = []
    for k in range(lag + 1):
        A = pose_inverse(track, k_hit, t - k).astype(np.float32)
        maps.append((A[0, 0] * xx + A[0, 1] * yy + A[0, 2], A[1, 0] * xx + A[1, 1] * yy + A[1, 2]))
    mx, my = maps[0]
    along = (mx - p0[0]) * u0[0] + (my - p0[1]) * u0[1]  # how far along the element (in its own frame) this pixel is
    level = np.clip(np.rint(along / reach * lag), 0, lag).astype(np.int32)
    for k in range(1, lag + 1):
        sel = level == k
        mx = np.where(sel, maps[k][0], mx)
        my = np.where(sel, maps[k][1], my)
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def root_holdout(track: dict, t: int, h: int, w: int, mask: np.ndarray | None) -> np.ndarray:
    """Where the hand hides the element's root (float 0..1): within a quarter of a hand size of the emitter, the hand
    itself (its matte; without one, the side behind the fingertips, coming in over 0.3 of a hand size so the base fades
    into the hand instead of being cut off along a line), so the flame leaves the fingers rather than sitting on them."""
    p, u, s = emitter_at(track, t)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = xx - p[0], yy - p[1]
    disc = np.clip(1.0 - (np.hypot(dx, dy) - 0.25 * s) / max(0.25 * s, 1.0), 0, 1)
    if mask is not None:
        return disc * mask
    back = -(dx * u[0] + dy * u[1])  # how far behind the emitter, along its direction
    return disc * np.clip(back / max(0.3 * s, 1.0), 0, 1)


def track_slice(track: dict, a: int, b: int) -> dict:
    """The track for frames a..b-1 only (the per-frame lists cut; events and sizes kept)."""
    per = ("xy", "dir", "scale", "conf", "src", "speed", "landmarks", "gesture")
    return track | {k: list(track[k][a:b]) for k in per if k in track}


def point_light(plate_lin: np.ndarray, element_lin: np.ndarray, track: dict, t: int, fluid: str,
                mask: np.ndarray | None) -> np.ndarray:
    """Until cuesync.fx_light lands: a point light half a hand ahead of the emitter, as bright and as coloured as the
    element is on this frame, strong on the hand (its matte), weak on the room; added, never darkening."""
    if fluid == "smoke":
        return plate_lin
    h, w = plate_lin.shape[:2]
    p, u, s = emitter_at(track, t)
    c = p + 0.5 * s * u
    power = float(element_lin.mean())
    if power <= 1e-5:
        return plate_lin
    colour = element_lin.reshape(-1, 3).mean(0)
    colour = colour / max(float(colour.max()), 1e-6)
    if fluid == "fire":
        colour = 0.5 * colour + 0.5 * np.array([1.0, 0.55, 0.22])
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.hypot(xx - c[0], yy - c[1])
    room = 1.0 / (1.0 + (r / (6.0 * s)) ** 2)
    near = 1.0 / (1.0 + (r / (1.5 * s)) ** 2) * (mask if mask is not None else 0.0)
    gain = min(power * 25.0, 1.5)
    return plate_lin + plate_lin * (gain * (0.6 * room + 1.4 * near))[..., None] * colour.astype(np.float32)


def make_bending_shot(spec, renderer, out_dir: Path, track_path: str | Path, mask_path: str | Path | None = None,
                      progress=None):
    """A bending shot: the plate (``spec.plate``, as cuesync.pipeline loads it), the take (``spec.voice``), the hand's
    track (``track_path``, made on that loaded plate) and optionally its matte (hand_mask.npz); ``spec.effect`` may be
    "auto" (the sound and the hand's gesture pick it). Renders one element over black at the strike and carries it on
    the hand."""
    import dataclasses
    import time

    from . import __version__
    from .composite import estimate_black_level
    from .effects import load_effect
    from .gestures import heard
    from .guide import write_guide
    from .media import read_video, write_video
    from .pipeline import FINISH_PRESET, FPS, SOUND, SR, ShotResult, _finalise, load_plate, place_audio, shot_cue, soundtrack
    from .scene import paint_scene_guide, plan_window, scene_size, suited_effect
    from .transfer import gam, lin

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    say("reading", 0.03, "Reading your plate and your hand's track")
    plate, _, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    n, h, w = plate.shape[:3]
    track = load_track(track_path)
    if (int(track["W"]), int(track["H"])) != (w, h):
        raise ValueError(f"the track was made on a {track['W']}x{track['H']} plate, this one is {w}x{h}")
    mask = np.load(mask_path)["alpha"] if mask_path is not None and Path(mask_path).exists() else None
    c = shot_cue(spec, n / FPS)
    cue = c.cue
    env_all = np.asarray(cue.envelope, np.float64)
    hits = [hh.frame for hh in cue.hits()] or [cue.frame_of(cue.hero_onset_s)]
    hero = max(hits, key=lambda f: float(env_all[f: f + 8].max()) if 0 <= f < env_all.size else 0.0)
    snap = snap_launch(hero, track.get("events", []))
    k_launch = snap["frame"]

    auto = None
    effect = spec.effect
    if effect == "auto":  # the sound and the hand's gesture at the strike
        h_ = heard(cue, c.samples, SR, c.source)
        g = gesture_at(track, k_launch)
        span = (max(hits) - min(hits)) / FPS if len(hits) > 1 else 0.0
        snapped = any(e.get("kind") == "snap" and abs(int(e["frame"]) - hero) <= 4 for e in track.get("events", []))
        effect, why = bending_effect(h_["kind"], g, hits=len(hits), span_s=span, snap=snapped)
        effect = effect or "fire_burst"
        auto = h_ | {"heard": h_["kind"], "hand": g, "picked": effect, "why": why}
    eff = load_effect(effect)

    say("painting", 0.12, "Painting the element where your hand is on the strike")
    p_hit, u_hit, s_hit = emitter_at(track, k_launch)
    aim = angle_of(u_hit)
    at = (float(p_hit[0] / w), float(p_hit[1] / h))
    black = np.full((1, h, w, 3), CANVAS, np.uint8)
    eff = suited_effect(eff, "the open hand", aim, black[0], at)
    eff = dataclasses.replace(eff, scene=dataclasses.replace(eff.scene, clean=True, resting=0.0))
    start, length = plan_window(cue, eff, None)
    scale = float(np.clip(LENGTH * s_hit / (eff.scene.r_max * min(w, h)), 0.5, 1.6))
    sp = paint_scene_guide(cue, eff, at, black, start=start, length=length, direction=aim, scale=scale, size=scene_size(w, h))
    write_guide(sp.guide, out_dir / "guide.mp4", fps=FPS)
    prompt = BENDING_PROMPTS.get(effect, eff.prompt).replace("{towards}", towards_words(aim))

    say("rendering", 0.2, f"Rendering the {eff.fluid} over black with LTX-2.5")
    seconds = renderer(out_dir / "guide.mp4", prompt, spec.seed, out_dir / "element.mp4")
    element = read_video(out_dir / "element.mp4").frames
    if element.shape[1:3] != (h, w):
        element = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in element])
    floor = estimate_black_level(element)
    ign = ignition_frame(element, floor)
    if ign is None:
        raise ValueError("the render shows no element to carry")
    offset = k_launch - ign  # element frame e shows on plate frame e + offset: its ignition on the strike

    say("compositing", 0.7, "Carrying the element on your hand")
    from . import fx_light as fx

    lag = LAG_FRAMES.get(effect, 3)
    src = list(track.get("src") or []) + ["mp"] * n
    floor_lin = float(lin(np.full((1, 1, 3), floor, np.uint8))[0, 0, 0])
    lost, run = np.zeros(n, int), 0
    for t in range(n):
        run = run + 1 if src[t] == "lost" else 0
        lost[t] = run

    def element_at(t: int) -> np.ndarray | None:
        """The element on plate frame t (linear, over black), carried on the hand, its root held out; None when
        it isn't on (before its build-up, after it ends)."""
        e = t - offset
        if not ign - PREGLOW <= e < len(element):
            return None
        E = warp_element(np.clip(lin(element[e]) - floor_lin, 0.0, None), track, k_launch, t, lag)
        if e < ign:  # the build-up, coming up to the ignition
            E *= (e - (ign - PREGLOW) + 1) / (PREGLOW + 1)
        elif ign < PREGLOW and e < PREGLOW:  # a render lit from its first frame: bring it up rather than pop it on
            E *= (e + 1) / (PREGLOW + 1)
        if lost[t] > 8:  # the hand is gone: let the element go rather than freeze it
            E *= max(0.0, 1.0 - (lost[t] - 8) / LOST_FADE)
        m_t = None if mask is None else mask[min(t, len(mask) - 1)].astype(np.float32) / 255.0
        return E * (1.0 - root_holdout(track, t, h, w, m_t))[..., None]

    # the light (cuesync.fx_light) needs the element's light on every frame first: a small copy is enough
    q = 8
    small = np.zeros((n, h // q, w // q, 3), np.float32)
    for t in range(n):
        E = element_at(t)
        if E is not None:
            small[t] = cv2.resize(E, (w // q, h // q), interpolation=cv2.INTER_AREA)
    white = fx.plate_white(np.stack([lin(plate[i]) for i in np.linspace(0, n - 1, 8).astype(int)]))
    noise = fx.shadow_noise(plate)
    out = np.empty_like(plate)
    CHUNK = 12  # frames lit at a time (a whole 10 s 1080p clip in float32 is 6 GB), overlapping a frame each side
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        a0, b1 = max(0, a - 1), min(n, b + 1)
        P = np.stack([lin(plate[i]) for i in range(a0, b1)])
        m = None if mask is None else mask[a0:b1].astype(np.float32) / 255.0
        fx.light_plate(P, small[a0:b1], track_slice(track, a0, b1), m, fluid=eff.fluid, white=white, out=P)
        for i in range(a, b):
            E = element_at(i)
            if E is None:
                continue
            E = fx.grain(E[None], noise, seed=spec.seed, frame0=i)[0]
            j = i - a0
            if eff.fluid == "smoke":  # smoke is laid over; fire is light, added; steam is lit vapour, screened
                al = np.clip(E.max(-1) / 0.35, 0.0, 0.9)[..., None]
                P[j] = P[j] * (1.0 - al) + np.clip(E / np.maximum(al, 1e-3), 0, 1) * al
            elif eff.fluid == "steam":
                P[j] = 1.0 - (1.0 - np.clip(P[j], 0, 1)) * (1.0 - np.clip(E, 0, 1))
            else:
                P[j] = P[j] + E
        if eff.fluid == "fire":  # the hot air above and around the flame shimmers
            alpha = np.stack([cv2.resize(np.clip(fx._lum(small[i]) * 4.0, 0, 1), (w, h)) for i in range(a0, b1)])
            fx.heat_haze(P, alpha, seed=spec.seed, out=P, frame0=a0)
        fx.flash(P, [k_launch - a0], [1.0], out=P)
        out[a:b] = gam(P[a - a0: b - a0])
    plate = element = small = None  # (a 10 s 1080p plate and its render are 2 GB: let them go before the finishing copies)
    if effect == "smoke_puff":  # the air blast: a pressure front out of the palm
        lo, hi = k_launch, min(n, k_launch + fx.AIR_FRAMES + 2)
        W_ = np.stack([lin(out[i]) for i in range(lo, hi)])
        fx.air_wave(W_, tuple(p_hit), tuple(u_hit), start_frame=0, seed=spec.seed, out=W_)
        out[lo:hi] = gam(W_)
    frames = fx.shake(out, [k_launch + 2], [1.0], px=4.0 * h / 1080, seed=spec.seed)

    say("finishing", 0.85, "Adding the sound and the cue ribbon")
    env = np.zeros(n)
    m = min(n, env_all.size)
    env[:m] = env_all[:m]
    audio = soundtrack(place_audio(c.samples, c.offset_s, n), eff, n, env, [k_launch], [1.0], cue.voice_end_frame, spec.seed)
    comp_path, final_path = out_dir / "composite.mp4", out_dir / "final.mp4"
    write_video(comp_path, frames, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)
    write_video(final_path, _finalise(frames, env, spec.label or effect.replace("_", " "), None), fps=FPS, audio=audio,
                crf=16, preset=FINISH_PRESET)
    receipt = {
        "cuesync": __version__, "mode": "bending", "effect": eff.to_json(),
        "inputs": {"point": list(spec.point), "seed": spec.seed, "offset_s": spec.offset_s, "cue": c.source},
        "render": {"seconds": round(seconds, 1), "prompt": prompt, "guide_scale": round(scale, 3)},
        "bending": {"track": str(track_path), "mask": None if mask is None else str(mask_path), "k_hit": int(k_launch),
                    "k_launch": int(k_launch), "ignition": int(ign), "offset": int(offset), "lag_frames": lag,
                    "snap": snap, "emitter": {"xy": [round(float(v), 1) for v in p_hit], "direction": round(aim, 1),
                                              "scale": round(s_hit, 1)}},
        "auto": auto, "sound": SOUND, "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=None, receipt=receipt)
