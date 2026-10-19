"""The performer's hand and forearm as a soft matte, frame by frame (element bending, WP2).

The element comes out of a moving hand, so the compositor needs to know, on every frame, where the hand is:
the fingers hold the element back where it should pass behind them, the plate's own hand pixels are never
replaced by the render, the camera tracker ignores them, and the light lands on them strongest. This module
turns the plate and the hand track (``hand_track.json``, WP1) into that matte.

Primary: the SAM 2.1-small video predictor (facebookresearch/sam2, Apache-2.0), prompted from the track: positive
points on the wrist, the index, middle and pinky knuckles and the emitter's fingertips, negative points on the
background around them, and a box from the landmarks that reaches down a bare forearm but ends at the wrist of a
sleeved arm (a bare forearm is part of the matte; a sleeve on a dark suit would bring the torso with it). It is
prompted again every 16 frames and wherever the tracker changed how it found the hand (a re-detection, a gap it
bridged), and its video memory carries the matte through the frames in between, motion blur included, forward
from the first prompt and back from each hand's first prompt. The frames reach the model one at a time
(``offload_video_to_cpu``: only the frame being segmented sits on the GPU) at the model's 1024 px; its mattes
come back at half resolution and are cleaned up there. The performer's other hands (the second palm) are found
with MediaPipe and segmented too.

Fallback, when SAM 2.1 or a GPU isn't there or loses the hand on a frame: MediaPipe's selfie multiclass
segmenter (Apache-2.0), its body-skin class, kept where it touches the track's hand (``selfie``); with no skin
found, or no MediaPipe, the landmarks' dilated hull (``hull``).

Then, on every frame: holes inside a hand filled, a 3-frame temporal median (with the neighbours moved onto the
hand first, so a fast hand isn't eroded by it), a 2 px feather, and on fast frames a motion blur along the hand's
own optical flow, so the matte's edge smears like the plate's.

    masks = hand_masks(frames, track)          # uint8 [T, H, W], 255 = hand; one track dict or a list (both hands)
    python -m cuesync.hand_mask plate.mp4 --track hand_track.json --out run/hand_mask.npz   # from venv-mask

The SAM and MediaPipe parts need their own environment (torch, sam2, mediapipe: runs/bending/venv-mask); the
rest (prompts, hull, clean-up, the npz) is numpy + OpenCV and runs anywhere.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SAM_MODEL = "facebook/sam2.1-hiera-small"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM_CHECKPOINT = "sam2.1_hiera_small.pt"
SAM_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"  # pinned weights, as in the CueSync node pack
MP_MODELS = "https://storage.googleapis.com/mediapipe-models/"
SELFIE_MODEL = MP_MODELS + "image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
HAND_MODEL = MP_MODELS + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
SKIN_COVER = 0.6  # the skin found must cover this much of the hand's own outline, or the segmenter missed the hand
SELFIE_SKIN, SELFIE_CLOTHES = 2, 4  # selfie multiclass: 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes
OTHERS_EVERY = 2  # frames between looks for the performer's other hands

REPROMPT_EVERY = 16  # frames between fresh prompts
PROMPT_NEAR = 4  # a scheduled prompt may move this many frames to a frame whose landmarks are good
PROMPT_GAP = 3  # ... and comes at least this many frames after the one before
MAX_COND = 4  # prompted frames each frame attends to (the nearest in time)
BOX_ARM = 2.0  # the prompt box reaches this many hand-scales past the wrist, for a bare forearm ...
SLEEVE = 0.5  # ... unless this share of the arm just past the wrist is clothes: then it ends at the wrist
GOOD_SRC = {"mp", "rtm"}  # track sources that carry real landmarks (flow/tapir/interp/hold/lost only move a point)
MIN_CONF = 0.5
KNUCKLES = (0, 5, 9, 17)  # wrist, index, middle and pinky knuckles: always on the hand, whatever the fingers do
TIPS = {"index_tip": (8,), "two_finger": (8, 12), "palm": (), "fist": ()}  # the emitter's fingertips, if any
WORK_SHORT = 540  # SAM's mattes come back, and are cleaned up, at this short side (half of 1080p)
HOLE_FILL = 0.10  # holes inside a hand smaller than this share of its area are filled (a real gap is bigger)
MIN_HELD = 0.6  # SAM's matte must hold this share of a well-tracked hand's landmarks ...
MAX_AREA = 30.0  # ... and cover at most this many hand-scales squared (a hand and the forearm in frame)
FEATHER_PX = 2.0
FAST = 0.012  # hand speed (share of the short side per frame) above which the matte is motion-blurred
SHUTTER = 0.5  # 180 degree shutter: the blur is half the frame-to-frame motion


@dataclass
class Prompt:
    frame: int
    points: np.ndarray  # [N, 2] px in the frame
    labels: np.ndarray  # [N] 1 = hand, 0 = background
    box: np.ndarray  # [4] x0, y0, x1, y1 px


# ---------------------------------------------------------------------------------------------- the track


def load_track(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tracks(track) -> list[dict]:
    return [track] if isinstance(track, dict) else list(track)


def _good(track: dict, t: int) -> bool:
    lm = track["landmarks"][t]
    return lm is not None and track["conf"][t] >= MIN_CONF and track["src"][t] in GOOD_SRC


def _lm(track: dict, t: int) -> np.ndarray:
    return np.asarray(track["landmarks"][t], np.float64)[:, :2]


def _scale(track: dict, t: int, lm: np.ndarray) -> float:
    s = float(track["scale"][t]) if track.get("scale") else 0.0
    return s if s > 1.0 else float(np.linalg.norm(lm[0] - lm[9]))


def prompt_frames(track: dict) -> list[int]:
    """The frames SAM is prompted on: every REPROMPT_EVERY frames and on every change of the tracker's source,
    each moved to the nearest frame (within PROMPT_NEAR) whose landmarks are good; frames with no good
    landmarks near them get no prompt (SAM's memory carries the hand through)."""
    T = len(track["src"])
    good = np.array([_good(track, t) for t in range(T)])
    src = track["src"]
    wanted = set(range(0, T, REPROMPT_EVERY))
    for t in range(1, T):
        if src[t] != src[t - 1]:
            wanted |= {t - 1, t}  # the last frame before the change and the first after it
    if good.any():
        wanted.add(int(np.flatnonzero(good)[0]))  # the hand's first good frame, where SAM starts
    picked = set()
    for k in sorted(wanted):
        near = [t for t in range(max(0, k - PROMPT_NEAR), min(T, k + PROMPT_NEAR + 1)) if good[t]]
        if near:
            picked.add(min(near, key=lambda t: (abs(t - k), t)))
    out = []
    for t in sorted(picked):  # a source flickering frame to frame needs one fresh prompt, not one per frame
        if not out or t - out[-1] >= PROMPT_GAP:
            out.append(t)
    return out


def _palm(lm: np.ndarray) -> np.ndarray:
    """The palm's centre: the wrist and the four knuckles."""
    return lm[[0, 5, 9, 13, 17]].mean(0)


def _negatives(c: np.ndarray, lm: np.ndarray, s: float, others: list[np.ndarray], W: int, H: int) -> list:
    """Up to two points on the background around the hand: 2.4 hand-scales from the palm, never toward the
    forearm (the wrist's side) and never on another tracked hand; the second at least 90 degrees from the first."""
    arm = lm[0] - lm[9]
    arm = arm / (np.linalg.norm(arm) + 1e-9)
    hands = [lm] + others
    cands = []
    for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        u = np.array([math.cos(a), math.sin(a)])
        if float(u @ arm) > math.cos(math.radians(60)):
            continue
        p = c + 2.4 * s * u
        if not (0.02 * W <= p[0] <= 0.98 * W and 0.02 * H <= p[1] <= 0.98 * H):
            continue
        clear = min(float(np.min(np.linalg.norm(h - p, axis=1))) for h in hands)
        if clear < 1.2 * s:
            continue
        cands.append((clear, a, p))
    cands.sort(key=lambda x: -x[0])
    out = []
    for clear, a, p in cands:
        if all(abs((a - b + math.pi) % (2 * math.pi) - math.pi) >= math.pi / 2 for b, _ in out):
            out.append((a, p))
        if len(out) == 2:
            break
    return [p for _, p in out]


def prompts(track: dict, others: list[dict] = (), W: int | None = None, H: int | None = None,
            reach: float = BOX_ARM) -> list[Prompt]:
    """SAM prompts for one hand. ``others`` are the other tracked hands (kept clear of this hand's negatives).
    ``reach``: how many hand-scales past the wrist the box goes; 0 for a sleeved arm (see ``sleeved``), where a
    box that ends at the wrist is what keeps SAM at the cuff. (Without a box SAM can settle on part of the hand:
    the wrist and forearm of the cupped palm.)"""
    W = int(W or track["W"])
    H = int(H or track["H"])
    tips = TIPS.get(track.get("emitter", ""), ())
    out = []
    for t in prompt_frames(track):
        lm = _lm(track, t)
        s = _scale(track, t, lm)
        pos = [lm[i] for i in KNUCKLES + tips]
        oth = [_lm(o, t) for o in others if o["landmarks"][t] is not None]
        neg = _negatives(_palm(lm), lm, s, oth, W, H)
        # The box: the landmarks plus a finger's width, and ``reach`` hand-scales from the wrist away from the
        # fingers and to both sides, so a bare forearm isn't cut off at the wrist whichever way the wrist bends
        # (pushing a palm out bends it ninety degrees: the forearm leaves sideways).
        arm = (lm[0] - lm[9]) / (np.linalg.norm(lm[0] - lm[9]) + 1e-9)
        side = np.array([-arm[1], arm[0]])
        ext = np.vstack([lm, lm[0] + reach * s * arm, lm[0] + reach * s * side, lm[0] - reach * s * side])
        lo, hi = ext.min(0) - 0.35 * s, ext.max(0) + 0.35 * s
        box = np.array([max(lo[0], 0), max(lo[1], 0), min(hi[0], W - 1), min(hi[1], H - 1)], np.float32)
        # A hand coming into frame has landmarks past the edge (the tracker extrapolates): only points on the
        # picture are given, and a hand with fewer than two of them on it is left to SAM's memory.
        pos = [p for p in pos if 0 <= p[0] < W and 0 <= p[1] < H]
        if len(pos) < 2:
            continue
        pts = np.array(pos + neg, np.float32).reshape(-1, 2)
        labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
        out.append(Prompt(t, pts, labels, box))
    return out


def _is_tracked(tracks: list[dict], t: int, c: np.ndarray, s: float, near: int) -> bool:
    """Is a hand seen at palm centre ``c`` on frame t one of the tracked hands? Compared with each track's
    landmarks on the nearest frame that has them (within ``near``), else with its emitter point."""
    for tr in tracks:
        T = len(tr["src"])
        for u in sorted(range(max(0, t - near), min(T, t + near + 1)), key=lambda u: abs(u - t)):
            if tr["landmarks"][u] is not None:
                if np.linalg.norm(_palm(_lm(tr, u)) - c) < 1.5 * s:
                    return True
                break
        else:
            if tr.get("xy") and t < len(tr["xy"]) and np.linalg.norm(np.asarray(tr["xy"][t]) - c) < 2.5 * s:
                return True
    return False


def other_hands(frames: np.ndarray, tracks: list[dict], every: int = OTHERS_EVERY) -> list[dict]:
    """The performer's hands the track doesn't follow (the other palm, a hand resting at the frame's edge), so the
    matte holds every hand in the shot, not just the emitter's. MediaPipe's hand landmarker (Apache-2.0, VIDEO
    mode: it follows a hand from one look to the next, which finds far more hands on a dark plate than looking
    at single frames) looks every ``every`` frames; sightings close together are one hand; a hand seen at least
    twice becomes a minimal track (landmarks where it was seen, none between) that SAM is prompted from."""
    import mediapipe as mp

    T, H, W = frames.shape[:3]
    k = min(1.0, 1280 / max(W, H))
    small = (max(8, round(W * k)), max(8, round(H * k)))
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_mp_model(HAND_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    seen = []
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as det:
        for t in range(0, T, every):
            img = cv2.resize(frames[t], small, interpolation=cv2.INTER_AREA)
            r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)),
                                     int(round(t * 1000 / 24)))
            for hand, side in zip(r.hand_landmarks, r.handedness):
                lm = np.array([[p.x * W, p.y * H, p.z * W] for p in hand])
                s = float(np.linalg.norm(lm[0, :2] - lm[9, :2]))
                c = _palm(lm[:, :2])
                if s > 4 and not _is_tracked(tracks, t, c, s, 8):
                    seen.append((t, lm, s, c, side[0].category_name))
    # Sightings within half a second are one hand when they are close, or further apart (a fast move) when
    # MediaPipe calls them the same side.
    chains: list[list] = []
    for t, lm, s, c, side in seen:
        fit = []
        for ch in chains:
            d = float(np.linalg.norm(ch[-1][3] - c)) / max(s, ch[-1][2])  # in hand-scales
            if 0 < t - ch[-1][0] <= 12 and (d < 2.5 or (ch[-1][4] == side and d < 6)):
                fit.append((d, ch))
        if fit:
            min(fit, key=lambda x: x[0])[1].append((t, lm, s, c, side))
        else:
            chains.append([(t, lm, s, c, side)])
    out = []
    for ch in chains:
        if len(ch) < 2:
            continue  # one sighting is too easily a false detection
        tr = {"version": 1, "fps": 24, "W": W, "H": H, "emitter": "palm", "hand": "other",
              "src": ["lost"] * T, "conf": [0.0] * T, "landmarks": [None] * T, "scale": [0.0] * T,
              "xy": [[float(ch[0][3][0]), float(ch[0][3][1])]] * T}
        for a, b in zip(ch, ch[1:]):  # one source along a run of sightings: no fresh prompt at every look
            if b[0] - a[0] <= 8:
                tr["src"][a[0]:b[0]] = ["mp"] * (b[0] - a[0])
        for t, lm, s, c, _ in ch:
            tr["src"][t], tr["conf"][t], tr["scale"][t] = "mp", 0.9, s
            tr["landmarks"][t] = lm.tolist()
        out.append(tr)
    return out


# ------------------------------------------------------------------------------------------------ SAM 2.1

_PREDICTOR = {}


def sam2_available() -> bool:
    try:
        import sam2  # noqa: F401
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _sam_checkpoint() -> str:
    """SAM 2.1-small's checkpoint (Apache-2.0, 184 MB), fetched once into the Hugging Face cache, pinned."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(SAM_MODEL, SAM_CHECKPOINT, revision=SAM_REVISION)


def fetch_models() -> list[str]:
    """Download every model the matte uses ahead of time (they otherwise come on first use): SAM 2.1-small's
    checkpoint into the Hugging Face cache, MediaPipe's hand landmarker and selfie segmenter into
    ~/.cache/cuesync. Returns their paths."""
    return [_sam_checkpoint(), _mp_model(HAND_MODEL), _mp_model(SELFIE_MODEL)]


def _predictor(device: str):
    """SAM 2.1-small's video predictor, built once per process (weights pinned on the Hugging Face hub)."""
    if device not in _PREDICTOR:
        from sam2.build_sam import build_sam2_video_predictor

        ckpt = _sam_checkpoint()
        # Each frame attends to the MAX_COND prompted frames nearest it in time, not all of them: the cost stays
        # flat however many prompts a long clip gets.
        _PREDICTOR[device] = build_sam2_video_predictor(
            SAM_CONFIG, ckpt, device=device, hydra_overrides_extra=[f"++model.max_cond_frames_in_attn={MAX_COND}"])
    return _PREDICTOR[device]


class _Frames:
    """The plate as SAM's video input, one frame at a time. sam2 normally decodes a folder of JPEGs into one
    float tensor of the whole clip (3 GB for 10 s); here each frame stays uint8 in RAM until SAM asks for it,
    and is resized to the model's square input and normalised on the GPU just then."""

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, frames: np.ndarray, size: int, device: str):
        import torch

        self.frames, self.size, self.device = frames, size, device
        self.mean = torch.tensor(self.MEAN, device=device).view(3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(self.frames[i])).to(self.device).permute(2, 0, 1)[None].float()
        x = F.interpolate(x / 255.0, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        return (x[0] - self.mean) / self.std


def _sam2(frames, tracks: list[dict], sleeves: list[bool], work: tuple[int, int], device: str = "cuda") -> np.ndarray:
    """SAM 2.1 video segmentation of each hand: bool [T, K, h, w] at the ``work`` size (w, h). ``sleeves``: which
    hands are in sleeves (their box ends at the wrist)."""
    import sam2.sam2_video_predictor as svp
    import torch

    T, H, W = frames.shape[:3]
    w, h = work
    k = np.array([w / W, h / H], np.float32)
    pred = _predictor(device)
    out = np.zeros((T, len(tracks), h, w), bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # init_state() with the frames from memory (its loader only knows file paths); the state is built by
        # sam2 itself. Its mattes come back at the work size, and prompts are given in work-size pixels.
        load = svp.load_video_frames
        svp.load_video_frames = lambda **kw: (_Frames(frames, pred.image_size, device), h, w)
        try:
            state = pred.init_state(video_path="<frames in memory>", offload_video_to_cpu=True)
        finally:
            svp.load_video_frames = load
        by_frame: dict[int, list] = {}
        for j, tr in enumerate(tracks):
            others = [o for i, o in enumerate(tracks) if i != j]
            for p in prompts(tr, others, W, H, reach=0.0 if sleeves[j] else BOX_ARM):
                by_frame.setdefault(p.frame, []).append((j, p))
        if not by_frame:
            return out
        for t in sorted(by_frame):  # frame by frame, so each frame's image features are computed once
            for j, p in by_frame[t]:
                pred.add_new_points_or_box(state, frame_idx=t, obj_id=j + 1, points=p.points * k,
                                           labels=p.labels, box=p.box * np.tile(k, 2))
        # Forward from the first prompt, then back from the latest of the hands' first prompts to frame 0, so
        # each hand is followed back in time from where it was first seen (a hand coming into frame, the other
        # palm before it was found) instead of guessed from memory alone; the backward pass has the last word.
        first: dict[int, int] = {}
        for t in sorted(by_frame):
            for j, _ in by_frame[t]:
                first.setdefault(j, t)
        start, back = min(first.values()), max(first.values())
        passes = [(start, False)] + ([(back, True)] if back > 0 else [])
        for start, reverse in passes:
            for t, ids, logits in pred.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                m = (logits[:, 0] > 0).cpu().numpy()
                for i, obj in enumerate(ids):
                    out[t, obj - 1] = m[i]
        pred.reset_state(state)
    del state
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------- fallback: selfie, hull


def _mp_model(url: str) -> str:
    """A MediaPipe model file (Apache-2.0), downloaded once into ~/.cache/cuesync."""
    import urllib.request

    path = Path.home() / ".cache" / "cuesync" / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return str(path)


def _selfie_segmenter():
    import mediapipe as mp

    opts = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_mp_model(SELFIE_MODEL)),
        output_confidence_masks=True, output_category_mask=False)
    return mp.tasks.vision.ImageSegmenter.create_from_options(opts)


def _selfie(seg, rgb: np.ndarray, category: int) -> np.ndarray:
    """Where the selfie segmenter puts ``category`` (bool, the size of ``rgb``)."""
    import mediapipe as mp

    res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    return np.squeeze(res.confidence_masks[category].numpy_view()) > 0.5  # [h, w, 1] in mediapipe 1.x


def sleeved(frames: np.ndarray, track: dict, seg, work: tuple[int, int]) -> bool:
    """Is this hand's forearm in a sleeve? The selfie segmenter's "clothes" share in a patch 0.7 hand-scales
    past the wrist, over the hand's prompt frames (clothes don't change within a shot). A bare forearm belongs
    in the matte; a sleeve stops it at the cuff: on a dark suit SAM can't tell the sleeve from the jacket,
    and the brief wants the hands and forearms, never the torso."""
    W, H = frames.shape[2], frames.shape[1]
    k = np.array([work[0] / W, work[1] / H])
    shares = []
    for t in prompt_frames(track):
        lm = _lm(track, t)
        s = _scale(track, t, lm)
        arm = (lm[0] - lm[9]) / (np.linalg.norm(lm[0] - lm[9]) + 1e-9)
        q, r = (lm[0] + 0.7 * s * arm) * k, max(2.0, 0.25 * s * k[0])
        if not (0 <= q[0] < work[0] and 0 <= q[1] < work[1]):
            continue  # the forearm is past the frame's edge here
        clothes = _selfie(seg, cv2.resize(frames[t], work, interpolation=cv2.INTER_AREA), SELFIE_CLOTHES)
        yy, xx = np.ogrid[:work[1], :work[0]]
        shares.append(float(clothes[(xx - q[0]) ** 2 + (yy - q[1]) ** 2 <= r * r].mean()))
    return bool(shares) and float(np.mean(shares)) >= SLEEVE


def hull_mask(lm: np.ndarray, s: float, shape: tuple[int, int], grow: float = 0.3) -> np.ndarray:
    """The landmarks' convex hull grown by ``grow`` hand-scales (landmarks are joint centres, a finger is
    wider), plus the wrist's stub of forearm, as a bool mask of ``shape`` (h, w). ``lm`` and ``s`` in its px."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    arm = lm[0] - lm[9]
    arm = arm / (np.linalg.norm(arm) + 1e-9)
    pts = np.vstack([lm, lm[0] + 0.8 * s * arm])
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
    r = max(1, int(round(grow * s)))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
FINGER_WIDTH = 0.24  # hand-scales (wrist to the middle knuckle): a finger is about a quarter of the palm's length


def bones_mask(lm: np.ndarray, s: float, shape: tuple[int, int]) -> np.ndarray:
    """The hand drawn from its landmarks (bool, ``shape``): the palm's polygon, each finger a line through its
    joints a finger wide and running on past the tip, and the wrist's stub of forearm. Unlike the hull it leaves
    the air between and above the fingers out (the fire's light on it drew an ellipse round the hand)."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    pts = lm[:, :2].astype(np.float64)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts[[0, 1, 2, 5, 9, 13, 17]]).astype(np.int32)), 1)
    width = max(2, int(round(FINGER_WIDTH * s)))
    for k, f in enumerate(FINGERS):
        chain = pts[list(f)]
        tip = chain[-1] + 0.3 * width * (chain[-1] - chain[-2]) / (np.linalg.norm(chain[-1] - chain[-2]) + 1e-9)
        line = np.round(np.vstack([chain, tip])).astype(np.int32)
        cv2.polylines(m, [line], False, 1, thickness=int(width * (1.2 if k == 0 else 1.0)), lineType=cv2.LINE_8)
    arm = pts[0] - pts[9]
    arm = arm / (np.linalg.norm(arm) + 1e-9)
    cv2.line(m, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[0] + 0.8 * s * arm).astype(int)), 1,
             thickness=max(3, int(round(0.7 * s))))
    return m > 0


def _fallback(small: np.ndarray, lm: np.ndarray | None, s: float, prev: np.ndarray | None, seg) -> tuple:
    """One hand on one work-size frame without SAM: the body-skin regions that touch the hand's hull (the hull
    itself where there is no skin), or, with no landmarks, the skin that overlaps the previous frame's matte."""
    h, w = small.shape[:2]
    if lm is None and prev is None:
        return np.zeros((h, w), bool), "hull"
    near = hull_mask(lm, s, (h, w)) if lm is not None else cv2.dilate(prev.astype(np.uint8), np.ones((9, 9))) > 0
    bones = bones_mask(lm, s, (h, w)) if lm is not None else prev
    if seg is None:
        return bones, "hull"
    skin = _selfie(seg, small, SELFIE_SKIN)
    _, lab = cv2.connectedComponents(skin.astype(np.uint8), connectivity=8)
    keep = np.unique(lab[near & skin])
    keep = keep[keep > 0]
    if not len(keep):
        return bones, "hull"
    m = np.isin(lab, keep)
    if lm is not None:  # in a dark room it finds a few fingers and drops the rest of the hand: the hull, then
        core = hull_mask(lm, s, (h, w), grow=0.0)
        if (m & core).sum() < SKIN_COVER * max(int(core.sum()), 1):
            return bones, "hull"
    if lm is not None:  # the forearm, not the whole arm and shoulder: skin within 5 hand-scales of the wrist
        yy, xx = np.ogrid[:h, :w]
        m &= (xx - lm[0, 0]) ** 2 + (yy - lm[0, 1]) ** 2 <= (5.0 * s) ** 2
        m |= (near & skin) | bones  # (the skin segmenter drops fingers over a dark floor: the drawn hand fills them)
    return m, "selfie"


# ------------------------------------------------------------------------------------------------ clean-up


def fill_holes(m: np.ndarray, max_share: float = HOLE_FILL) -> np.ndarray:
    """Fill the background islands inside a hand's matte smaller than ``max_share`` of its area (a bright
    highlight or a dark crease SAM let go); a bigger enclosed gap (thumb and finger touching) is kept."""
    area = int(m.sum())
    if not area:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    small = [i for i in range(1, n) if i not in edge and stats[i, cv2.CC_STAT_AREA] < max_share * area]
    return m | np.isin(lab, small) if small else m


def holes(alpha: np.ndarray) -> float:
    """The biggest enclosed hole in a matte (alpha >= 128) as a share of its area (0 when there is none)."""
    m = alpha >= 128
    area = int(m.sum())
    if not area:
        return 0.0
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    return max((stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if i not in edge), default=0) / area


def _centroid(m: np.ndarray) -> np.ndarray | None:
    """The centre of the matte's largest piece (a stray speck doesn't move it), or None when it is empty."""
    n, _, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return None if n < 2 else cent[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]


def _shift(m: np.ndarray, d: np.ndarray) -> np.ndarray:
    if not m.any() or not np.any(np.abs(d) >= 0.5):
        return m
    M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    return cv2.warpAffine(m.astype(np.uint8), M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST) > 0


def temporal_median(masks: np.ndarray, anchors: list | None = None) -> np.ndarray:
    """3-frame median (a majority vote) of one hand's bool mattes [T, h, w], each neighbour first moved by the
    hand's own displacement, so a single-frame flicker or drop-out goes away while the edges of a fast-moving
    hand stay where they are. The displacement is the track's (``anchors``: the palm centre per frame in
    matte px, or None), else the matte's centroid: a frame whose matte suddenly lost the forearm moves its
    centroid, not the hand."""
    T = len(masks)
    if T < 3:
        return masks.copy()
    c = [_centroid(m) for m in masks]
    anchors = anchors if anchors is not None else [None] * T
    out = masks.copy()
    for t in range(T):
        votes = masks[t].astype(np.uint8)
        here = c[t]
        if here is None and 0 < t < T - 1 and c[t - 1] is not None and c[t + 1] is not None:
            here = (c[t - 1] + c[t + 1]) / 2  # a dropped frame: the hand is between its neighbours
        for u in (t - 1, t + 1):
            if not 0 <= u < T:
                votes = votes + masks[t]  # the clip's ends: the frame votes for itself
                continue
            if anchors[t] is not None and anchors[u] is not None and masks[u].any():
                d = anchors[t] - anchors[u]  # the track's motion: the same point on the hand in both frames
            elif here is not None and c[u] is not None:
                d = here - c[u]
            else:
                d = np.zeros(2)
            votes = votes + _shift(masks[u], d)
        out[t] = votes >= 2
    return out


def _flow(frames: np.ndarray, t: int, work: tuple[int, int]) -> np.ndarray:
    """Optical flow around frame t in work-size px per frame: DIS from t-1 to t+1, halved."""
    a, b = max(t - 1, 0), min(t + 1, len(frames) - 1)
    g = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), work, interpolation=cv2.INTER_AREA) for i in (a, b)]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    return dis.calc(g[0], g[1], None) / max(b - a, 1)


def _line_kernel(v: np.ndarray) -> np.ndarray:
    """A normalised anti-aliased line of length |v| px along v: a box motion blur."""
    n = int(math.ceil(float(np.hypot(*v)))) | 1
    k = np.zeros((n + 2, n + 2), np.float32)
    c = (n + 1) / 2
    p0, p1 = np.round((c - v / 2) * 16).astype(int), np.round((c + v / 2) * 16).astype(int)
    cv2.line(k, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, 1, cv2.LINE_AA, shift=4)
    return k / max(float(k.sum()), 1e-6)


def finish(masks: np.ndarray, frames, size: tuple[int, int], anchors: list | None = None) -> np.ndarray:
    """Bool mattes per hand [T, K, h, w] at the work size -> the uint8 matte [T, H, W] at ``size`` (W, H):
    holes filled, the temporal median, upsampled, a FEATHER_PX feather, and on fast frames the motion blur
    along the hand's optical flow (from ``frames``, the plate; None skips the blur). ``masks`` is cleaned in
    place."""
    alpha = np.empty((len(masks), size[1], size[0]), np.uint8)
    for t, a in enumerate(finish_frames(masks, frames, size, anchors)):
        alpha[t] = a
    return alpha


def finish_frames(masks: np.ndarray, frames, size: tuple[int, int], anchors: list | None = None):
    """``finish`` one frame at a time (yields uint8 [H, W]), so a caller writing them out never holds the
    whole full-size matte. ``anchors``: per hand, its palm centre per frame in work px (or None), from the
    track, for the temporal median."""
    T, K, h, w = masks.shape
    W, H = size
    for t in range(T):
        for j in range(K):
            masks[t, j] = fill_holes(masks[t, j])
    for j in range(K):
        masks[:, j] = temporal_median(masks[:, j], anchors[j] if anchors else None)
    fast = FAST * min(w, h)
    to_full = np.array([W / w, H / h])
    cent = [[_centroid(masks[t, j]) for t in range(T)] for j in range(K)]
    for t in range(T):
        alpha = np.zeros((H, W), np.uint8)
        flow = None
        for j in range(K):
            m = masks[t, j]
            if not m.any():
                continue
            blur = None
            ca, cb = cent[j][max(t - 1, 0)], cent[j][min(t + 1, T - 1)]
            if frames is not None and ca is not None and cb is not None and np.hypot(*(cb - ca)) / 2 > fast:
                flow = _flow(frames, t, (w, h)) if flow is None else flow
                v = np.median(flow[m], axis=0)  # the hand's own motion, not its outline's
                if np.hypot(*v) > fast:
                    blur = v * to_full * SHUTTER
            pad = 4 + (int(math.ceil(float(np.hypot(*blur)))) if blur is not None else 0)
            up, X0, Y0 = _upsample(m, (W, H), pad)
            a = cv2.GaussianBlur((up >= 0.5).astype(np.float32), (0, 0), FEATHER_PX / 2)
            if blur is not None:
                a = cv2.filter2D(a, -1, _line_kernel(blur), borderType=cv2.BORDER_CONSTANT)
            region = alpha[Y0:Y0 + a.shape[0], X0:X0 + a.shape[1]]
            np.maximum(region, np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8), out=region)
        yield alpha


def _upsample(m: np.ndarray, size: tuple[int, int], pad: int) -> tuple[np.ndarray, int, int]:
    """Bool matte ``m`` (work size) as a float matte at ``size`` (W, H), bilinear like resizing the whole frame
    (pixel centres aligned, edges replicated), but only in the box around it grown by ``pad`` full-size px:
    a hand covers a few percent of a 1080p frame. Returns (crop, x0, y0)."""
    W, H = size
    h, w = m.shape
    sx, sy = w / W, h / H
    ys, xs = np.nonzero(m)
    X0, X1 = max(int(xs.min() / sx) - pad, 0), min(int((xs.max() + 1) / sx) + pad + 1, W)
    Y0, Y1 = max(int(ys.min() / sy) - pad, 0), min(int((ys.max() + 1) / sy) + pad + 1, H)
    x0, x1 = max(int(X0 * sx) - 1, 0), min(int(X1 * sx) + 2, w)
    y0, y1 = max(int(Y0 * sy) - 1, 0), min(int(Y1 * sy) + 2, h)
    M = np.float32([[sx, 0, (X0 + 0.5) * sx - 0.5 - x0], [0, sy, (Y0 + 0.5) * sy - 0.5 - y0]])
    crop = cv2.warpAffine(m[y0:y1, x0:x1].astype(np.float32), M, (X1 - X0, Y1 - Y0),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return crop, X0, Y0


# ------------------------------------------------------------------------------------------------ the plate


class PackedFrames:
    """A plate held as JPEG in memory (about 0.2 MB a 1080p frame instead of 6 MB) and decoded only when a frame
    is used, so the CLI never holds the clip whole: this box's RAM is shared with the renders. Indexing gives
    uint8 RGB [H, W, 3]; ``shape`` is the clip's [T, H, W, 3]. Reads the plate as the pipeline writes it (SDR,
    24 fps; a phone's rotation is undone like ``cuesync.media.read_video`` does)."""

    QUALITY = 95  # a mean error of half a level: nothing SAM, MediaPipe or the flow can see

    def __init__(self, path: str | Path):
        import av

        from .media import _rotation_k

        self.jpegs = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                bgr = frame.to_ndarray(format="bgr24")
                k = _rotation_k(frame)
                bgr = np.ascontiguousarray(np.rot90(bgr, k)) if k else bgr
                self.jpegs.append(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.QUALITY])[1])
        if not self.jpegs:
            raise ValueError(f"no decodable video frames in {path}")
        self.shape = (len(self.jpegs), *bgr.shape)

    def __len__(self) -> int:
        return len(self.jpegs)

    def __getitem__(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imdecode(self.jpegs[int(i)], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------------------------------------ the matte


def _fit(track: dict, T: int, W: int, H: int) -> dict:
    """The track on these frames: padded (as lost) or cut to T frames, and in their pixels if it was measured
    on another size of the same video."""
    tr = dict(track)
    n = len(tr["src"])
    pad = max(0, T - n)
    tr["src"] = (list(tr["src"]) + ["lost"] * pad)[:T]
    tr["conf"] = (list(tr["conf"]) + [0.0] * pad)[:T]
    tr["landmarks"] = (list(tr["landmarks"]) + [None] * pad)[:T]
    tr["scale"] = (list(tr.get("scale") or [0.0] * n) + [0.0] * pad)[:T]
    sx, sy = W / float(tr.get("W") or W), H / float(tr.get("H") or H)
    if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
        tr["landmarks"] = [None if lm is None else [[p[0] * sx, p[1] * sy, *p[2:]] for p in lm]
                           for lm in tr["landmarks"]]
        tr["scale"] = [s * math.sqrt(sx * sy) for s in tr["scale"]]
    tr["W"], tr["H"] = W, H
    return tr


def _work_size(W: int, H: int) -> tuple[int, int]:
    k = min(1.0, WORK_SHORT / min(W, H))
    return max(8, int(round(W * k))), max(8, int(round(H * k)))


def _plausible(m: np.ndarray, lm: np.ndarray, s: float) -> bool:
    """Does SAM's matte hold the tracked hand? Most of its landmarks inside it (grown a little), and not so
    big that it ran into the body or the background."""
    if not m.any():
        return False
    h, w = m.shape
    x, y = np.round(lm[:, 0]).astype(int), np.round(lm[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if inside.sum() < 5:
        return True  # the hand is mostly out of frame: nothing to check it against
    r = max(1, int(round(0.15 * s)))
    grown = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return bool(grown[y[inside], x[inside]].mean() >= MIN_HELD and m.sum() <= MAX_AREA * s * s)


def hand_matte(frames, track, **kw) -> tuple[np.ndarray, list[str]]:
    """The soft hand + forearm matte (uint8 [T, H, W], 255 = hand) and each frame's source (sam2/selfie/hull).

    ``frames``: the plate, uint8 RGB [T, H, W, 3] (or a ``PackedFrames``). ``track``: one hand_track.json dict, or
    a list of them (one per hand, e.g. both palms). ``others``: also find and matte the performer's hands no track
    follows (needs MediaPipe; default on). ``method``: "auto" (SAM 2.1 when torch, sam2 and a GPU are there, else
    the selfie skin, else the hull), or "sam2" / "selfie" / "hull" to force one. ``log``: a callable for progress."""
    T, H, W = frames.shape[:3]
    out, src = matte_frames(frames, track, **kw)
    alpha = np.empty((T, H, W), np.uint8)
    for t, a in enumerate(out):
        alpha[t] = a
    return alpha, src


def matte_frames(frames, track, *, method: str = "auto", others: bool = True, device: str = "cuda", log=None):
    """``hand_matte`` with the matte coming out one frame at a time: (an iterator of uint8 [H, W], src)."""
    say = log or (lambda _msg: None)
    T, H, W = frames.shape[:3]
    tracks = [_fit(t, T, W, H) for t in _tracks(track)]
    w, h = work = _work_size(W, H)
    k = np.array([w / W, h / H])
    t0 = time.perf_counter()
    if others:
        try:
            more = other_hands(frames, tracks)
        except ImportError:
            more = []
        tracks += more
        say(f"{len(more)} more hand(s) found in {time.perf_counter() - t0:.1f} s")
    K = len(tracks)

    sam = method in ("auto", "sam2") and sam2_available()
    if method == "sam2" and not sam:
        raise RuntimeError("SAM 2.1 needs torch with CUDA and the sam2 package (runs/bending/venv-mask)")
    seg, tried = None, method == "hull"

    def segmenter():  # the selfie segmenter, loaded the first time it is needed
        nonlocal seg, tried
        if not tried:
            tried = True
            try:
                seg = _selfie_segmenter()
            except ImportError:
                if method == "selfie":
                    raise
        return seg

    if sam:
        sleeves = [segmenter() is not None and sleeved(frames, tr, seg, work) for tr in tracks]
        masks = _sam2(frames, tracks, sleeves, work, device)
        say(f"sam2 in {time.perf_counter() - t0:.1f} s ({sum(sleeves)} of {K} hands in sleeves)")
    else:
        masks = np.zeros((T, K, h, w), bool)

    # The guard: a frame where SAM lost or overran a hand the tracker sees well is redone by the fallback.
    src = np.full((T, K), "sam2" if sam else "hull", dtype=object)
    for j, tr in enumerate(tracks):
        prev = None
        for t in range(T):
            lm, s = None, 0.0
            if tr["landmarks"][t] is not None and tr["conf"][t] >= MIN_CONF:
                full = _lm(tr, t)
                lm, s = full * k, _scale(tr, t, full) * math.sqrt(k[0] * k[1])
            if sam and (lm is None or not _good(tr, t) or _plausible(masks[t, j], lm, s)):
                prev = masks[t, j] if masks[t, j].any() else prev
                continue
            small = cv2.resize(frames[t], work, interpolation=cv2.INTER_AREA)
            masks[t, j], src[t, j] = _fallback(small, lm, s, prev, segmenter())
            prev = masks[t, j] if masks[t, j].any() else prev
    if seg is not None:
        seg.close()
    redone = int((src != "sam2").sum())
    say(f"guard and fallback in {time.perf_counter() - t0:.1f} s ({redone} hand-frames not SAM's)")
    # Where each hand's track has it, its palm centre: the temporal median moves the neighbouring mattes by it.
    anchors = [[_palm(_lm(tr, t)) * k if tr["landmarks"][t] is not None and tr["conf"][t] >= MIN_CONF else None
                for t in range(T)] for tr in tracks]
    rank = {"sam2": 0, "selfie": 1, "hull": 2}

    def cleaned():
        yield from finish_frames(masks, frames, (W, H), anchors)
        say(f"clean-up in {time.perf_counter() - t0:.1f} s")

    return cleaned(), [max(src[t], key=rank.get) if K else "hull" for t in range(T)]


def hand_masks(frames: np.ndarray, track, out: str | Path | None = None, **kw) -> np.ndarray:
    """The matte (uint8 [T, H, W]); with ``out``, also written there as hand_mask.npz (docs/bending_contracts.md)."""
    alpha, src = hand_matte(frames, track, **kw)
    if out is not None:
        save(out, alpha, src)
    return alpha


def save(path: str | Path, alpha, src: list[str], fps: int = 24) -> Path:
    """hand_mask.npz (docs/bending_contracts.md), the file np.savez_compressed writes. ``alpha``: uint8 [T, H, W],
    or its frames one at a time (``matte_frames``), written as they come so the full-size matte (0.5 GB for 10 s
    of 1080p) never has to sit in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = iter(alpha)
    first = next(frames, None)
    shape = (len(src), *first.shape) if first is not None else (0, 0, 0)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open("alpha.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array_header_1_0(f, {"descr": "|u1", "fortran_order": False, "shape": shape})
            n = 0
            for frame in itertools.chain([first] if first is not None else [], frames):
                f.write(np.ascontiguousarray(frame, np.uint8).tobytes())
                n += 1
        if n != shape[0]:
            raise ValueError(f"{n} matte frames for {shape[0]} sources")
        for name, value in (("src", np.array(src, dtype=str)), ("fps", np.array(fps)), ("version", np.array(1))):
            with zf.open(f"{name}.npy", "w") as f:
                np.lib.format.write_array(f, value, allow_pickle=False)
    return path


def load(path: str | Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as z:
        return z["alpha"], [str(s) for s in z["src"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cuesync.hand_mask", description=__doc__.split("\n\n")[0])
    ap.add_argument("video", help="the plate as the pipeline loads it (24 fps, cropped)")
    ap.add_argument("--track", action="append", required=True, help="hand_track.json (repeat it for a second hand)")
    ap.add_argument("--out", required=True, help="where to write hand_mask.npz")
    ap.add_argument("--method", default="auto", choices=["auto", "sam2", "selfie", "hull"])
    ap.add_argument("--only-tracked", action="store_true",
                    help="matte only the tracked hand(s), not every hand in the shot")
    ap.add_argument("--preview", help="also write the matte over the plate as an mp4, to check it by eye")
    a = ap.parse_args(argv)

    def say(msg: str) -> None:
        print(f"hand_mask: {msg}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    frames = PackedFrames(a.video)
    T, H, W = frames.shape[:3]
    say(f"{T} frames of {W}x{H} read in {time.perf_counter() - t0:.1f} s")
    out, src = matte_frames(frames, [load_track(p) for p in a.track], method=a.method, others=not a.only_tracked,
                            log=say)
    save(a.out, _preview(out, frames, a.preview) if a.preview else out, src)
    say(f"wrote {a.out}, {time.perf_counter() - t0:.1f} s in all; frames by source "
        f"{ {s: src.count(s) for s in ('sam2', 'selfie', 'hull')} }")
    return 0


def _preview(out, frames, path: str):
    """Pass the matte frames through, and write them over the plate at half size as an mp4 (to check by eye)."""
    from .media import write_video

    T, H, W = frames.shape[:3]
    size = (max(2, W // 4 * 2), max(2, H // 4 * 2))
    shots = np.empty((T, size[1], size[0], 3), np.uint8)
    tint = np.float32([40, 255, 90])
    for t, a in enumerate(out):
        g = cv2.resize(a, size, interpolation=cv2.INTER_AREA)[..., None].astype(np.float32) / 255.0
        plate = cv2.resize(frames[t], size, interpolation=cv2.INTER_AREA)
        shots[t] = (plate * (1 - 0.45 * g) + tint * 0.45 * g).astype(np.uint8)
        yield a
    write_video(path, shots, fps=24)


if __name__ == "__main__":
    sys.exit(main())
