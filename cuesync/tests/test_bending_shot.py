"""Element bending, carried on the hand: the warp follows the tracked emitter (base at once, tip late), the element's
ignition lands on the strike, and a whole shot runs with a fake renderer (cuesync.bending)."""

import json
from pathlib import Path

import numpy as np

from cuesync.bending import emitter_at, ignition_frame, make_bending_shot, warp_element
from cuesync.media import read_video, write_video
from cuesync.pipeline import ShotSpec


def track_of(n, w, h, xy, d=(1.0, 0.0), scale=20.0, events=None, gesture="Victory"):
    """A hand_track.json dict: the emitter at xy(t), pointing d, hand size ``scale``."""
    return {"version": 1, "fps": 24, "W": w, "H": h, "emitter": "two_finger", "hand": "right",
            "xy": [list(map(float, xy(t))) for t in range(n)], "dir": [list(d)] * n, "scale": [scale] * n,
            "conf": [1.0] * n, "src": ["mp"] * n, "speed": [0.0] * n, "landmarks": [None] * n,
            "gesture": [gesture] * n, "events": events or []}


def test_the_element_follows_the_emitter_base_first_tip_late():
    w, h = 200, 100
    tr = track_of(30, w, h, lambda t: (50 + (4 * (t - 10) if t > 10 else 0), 50))  # still, then moving right 4 px/frame
    el = np.zeros((h, w, 3), np.float32)
    el[48:53, 48:53] = 1.0  # at the emitter (as rendered on frame 10)
    el[48:53, 48 + 120:53 + 120] = 1.0  # at the far end of the element (6 hand sizes along it)
    out = warp_element(el, tr, emitter_at(tr, 10), 20, lag=4)
    base = np.argwhere(out[:, :120, 0] > 0.5)[:, 1].mean()
    tip = np.argwhere(out[:, 120:, 0] > 0.5)[:, 1].mean() + 120
    assert abs(base - (50 + 40)) < 3  # the base is where the emitter is now (less the half frame its blur trails)
    assert abs(tip - (170 + 4 * (20 - 4 - 10))) < 3  # the tip is where the emitter was four frames ago
    assert np.allclose(warp_element(el, tr, emitter_at(tr, 10), 10, lag=4), el, atol=1e-3)  # as rendered on the strike


def test_ignition_is_where_the_light_comes_up():
    frames = np.zeros((20, 16, 16, 3), np.uint8)
    frames[:] = 6  # the render's black
    for i, v in enumerate((10, 40, 120, 250)):
        frames[8 + i] = v
    assert ignition_frame(frames, floor=8) == 10  # the first frame with 15 % of the brightest frame's light (112 of 242)
    assert ignition_frame(np.full((5, 8, 8, 3), 6, np.uint8), floor=8) is None


class GuideAsElement:
    """Renders 'the element' as its own guide (fire shapes on black on the keyframes), held between them."""

    name, scenes = "fake-element", True

    def __init__(self):
        self.prompts = []

    def __call__(self, guide_path, prompt, seed, out_path, image=None):
        self.prompts.append(prompt)
        g = read_video(guide_path).frames.copy()
        last = g[0]
        for i in range(len(g)):
            last = g[i] if g[i].max() > 40 else last  # (compressed black is not quite 0)
            g[i] = last
        write_video(out_path, g, fps=24)
        return 0.1


def test_a_bending_shot_carries_the_element_and_says_how(tmp_path, make_video, make_wav):
    w, h, n = 256, 128, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (40, 35, 30 + (i % 2)))
    sr = 48000
    x = np.random.default_rng(2).normal(0, 0.001, 3 * sr).astype(np.float32)
    t = np.arange(int(0.25 * sr)) / sr
    x[sr: sr + t.size] += (0.6 * np.sin(2 * np.pi * 120 * t) * np.exp(-6 * t)).astype(np.float32)  # a voiced "BOOM" at 1 s
    voice = make_wav(tmp_path, x, sr)
    tr = track_of(n, w, h, lambda t: (90 + (2 * (t - 26) if t > 26 else 0), 64), events=[{"frame": 23, "kind": "extension",
                                                                                         "peak_speed": 900.0}])
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(tr))
    r = GuideAsElement()
    spec = ShotSpec(plate=plate, voice=voice, effect="auto", point=(0.35, 0.5), mode="scene", plate_size=(w, h), max_plate_s=3.0)
    res = make_bending_shot(spec, r, tmp_path / "shot", track_path)
    rc = res.receipt
    assert rc["mode"] == "bending" and rc["auto"]["hand"] == "Victory" and rc["auto"]["heard"] == "boom"
    assert rc["bending"]["snap"]["used"] == "motion" and rc["bending"]["k_launch"] == 23  # on the arm's lock
    assert "black background" in r.prompts[0] and "hand" not in r.prompts[0]  # the element alone, over black
    out = read_video(res.composite).frames
    p, _, _ = emitter_at(tr, 40)
    x0, y0 = int(p[0]), int(p[1])
    ahead = out[40, y0 - 10: y0 + 10, x0 + 5: x0 + 60].astype(int)
    assert ahead.max() > 150  # the fire, carried to where the emitter is on frame 40, in front of it
    assert out[40, :, : max(0, x0 - 60)].astype(int).max() < 190  # and not left behind where it was rendered (its light and glow only)
    assert Path(res.final).exists() and (tmp_path / "shot" / "receipt.json").exists()


def test_a_bending_shot_is_performed_again_with_no_new_render(tmp_path, make_video, make_wav):
    w, h, n = 256, 128, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (40, 35, 30 + (i % 2)))
    sr = 48000
    t = np.arange(int(0.25 * sr)) / sr
    boom = (0.6 * np.sin(2 * np.pi * 120 * t) * np.exp(-6 * t)).astype(np.float32)

    def take(at_s):
        x = np.random.default_rng(3).normal(0, 0.001, 3 * sr).astype(np.float32)
        for a, g in at_s:
            x[int(a * sr): int(a * sr) + t.size] += g * boom
        return make_wav(tmp_path, x, sr)

    tr = track_of(n, w, h, lambda t: (90.0, 64.0))
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(tr))
    r = GuideAsElement()
    spec = ShotSpec(plate=plate, voice=take([(1.0, 1.0)]), effect="fire_burst", point=(0.35, 0.5), mode="scene",
                    plate_size=(w, h), max_plate_s=3.0)
    first = make_bending_shot(spec, r, tmp_path / "one", track_path)
    again = ShotSpec(plate=plate, voice=take([(0.6, 0.5), (1.3, 1.0), (2.0, 0.5)]), effect="auto", point=(0.35, 0.5),
                     mode="scene", plate_size=(w, h), max_plate_s=3.0)
    res = make_bending_shot(again, r, tmp_path / "again", track_path, element_from=first.out_dir)
    rc = res.receipt["bending"]
    assert len(r.prompts) == 1 and rc["performed_again"]["no_new_render"]  # rendered once, played twice
    assert len(rc["hits"]) == 3 and rc["hits"][1]["gain"] == 1.0 and rc["hits"][0]["gain"] < 1.0  # quieter, smaller
    assert res.receipt["effect"]["id"] == "fire_burst"  # the element's own effect, whatever the new take sounds like


def test_an_air_blast_is_made_not_rendered_and_is_gone_in_half_a_second():
    from cuesync.bending import air_puff

    ref = (np.array([60.0, 64.0]), np.array([1.0, 0.0]), 20.0)
    puff = air_puff(128, 256, ref, seed=1)
    lum = puff.reshape(len(puff), -1, 3).mean((1, 2))
    centre = [float(np.argwhere(f[..., 0] > 0.5 * f[..., 0].max())[:, 1].mean()) for f in puff[:10]]
    assert len(puff) <= 14 and lum[-1] < 0.3 * lum.max()  # short, and thinned out by its end
    assert centre[9] > centre[0] + 30  # it travels along the aim
    assert np.array_equal(puff, air_puff(128, 256, ref, seed=1))  # the same for the same seed


def test_the_aim_arrow_turns_the_direction_and_is_carried_with_the_hand():
    from cuesync.bending import anchored

    tr = track_of(10, 200, 100, lambda t: (50 + 5 * t, 50), d=(1.0, 0.0))  # moving right, pointing right
    a = anchored(tr, 4, (70.0, 50.0), aim=0.0)  # tapped on frame 4, the arrow drawn straight up
    assert np.allclose(a["dir"][4], (0.0, -1.0), atol=1e-6) and np.allclose(a["dir"][9], (0.0, -1.0), atol=1e-6)
    assert np.allclose(a["xy"][9], (50 + 45 + 0, 50), atol=1e-6) and a["anchor"]["turn"] == -90.0
    assert np.allclose(anchored(tr, 4, (70.0, 50.0))["dir"][4], (1.0, 0.0))  # no arrow: the hand's own direction


def test_the_camera_is_followed_when_handheld_and_left_alone_on_a_tripod():
    import cv2

    from cuesync.bending import camera_path

    rng = np.random.default_rng(4)
    room = cv2.GaussianBlur(rng.random((700, 1100)).astype(np.float32), (0, 0), 2.0)
    room = ((room - room.mean()) / room.std() * 40 + 128).clip(0, 255)  # a textured wall
    room = np.repeat(room.astype(np.uint8)[..., None], 3, -1)
    n, shift = 16, [(1.5 * t, 0.8 * np.sin(t)) for t in range(16)]  # a slow pan and a wobble
    frames = np.stack([cv2.warpAffine(room, np.float32([[1, 0, dx - 60], [0, 1, dy - 60]]), (960, 540)) for dx, dy in shift])
    W = camera_path(frames)
    assert W is not None
    for t in range(1, n):
        step = W[t] @ np.linalg.inv(W[t - 1])
        assert abs(step[0, 2] - (shift[t][0] - shift[t - 1][0])) < 0.3 and abs(step[1, 2] - (shift[t][1] - shift[t - 1][1])) < 0.3
    assert camera_path(np.repeat(frames[:1], n, 0)) is None  # a tripod: nothing to follow


def test_thrown_fire_moves_with_the_room_as_the_camera_pans():
    w, h = 300, 120
    tr = track_of(30, w, h, lambda t: (60, 60))  # the hand still in the picture
    el = np.zeros((h, w, 3), np.float32)
    el[58:63, 58:63] = 1.0  # at the emitter
    el[58:63, 58 + 120:63 + 120] = 1.0  # the far end, thrown 4 frames ago
    cam = np.repeat(np.eye(3)[None], 30, 0)
    cam[:, 0, 2] = 3.0 * np.arange(30)  # the camera pans: the room moves 3 px a frame across the picture
    out = warp_element(el, tr, emitter_at(tr, 10), 20, lag=4, cam=cam)
    base = np.argwhere(out[:, :120, 0] > 0.3)[:, 1].mean()
    tip = np.argwhere(out[:, 120:, 0] > 0.3)[:, 1].mean() + 120
    assert abs(base - 60) < 2  # the base stays on the hand
    assert abs(tip - (180 + 3 * 4)) < 3  # what was thrown 4 frames ago has moved with the room since


def test_before_the_strike_the_build_up_stays_on_the_hand_under_a_panning_camera():
    w, h = 300, 120
    tr = track_of(30, w, h, lambda t: (60, 60))
    el = np.zeros((h, w, 3), np.float32)
    el[58:63, 58:63] = 1.0
    cam = np.repeat(np.eye(3)[None], 30, 0)
    cam[:, 0, 2] = 3.0 * np.arange(30)
    for t in (16, 18, 19):  # the spark-up, frames before the strike at 20: nothing thrown yet, all of it on the hand
        out = warp_element(el, tr, emitter_at(tr, 10), t, lag=4, cam=cam, since=20)
        assert abs(np.argwhere(out[..., 0] > 0.3)[:, 1].mean() - 60) < 2


def test_a_hum_charges_the_strike_and_launches_nothing(tmp_path, make_video, make_wav):
    w, h, n = 256, 128, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (30, 28, 26))
    sr = 48000
    x = np.random.default_rng(5).normal(0, 0.0005, 3 * sr).astype(np.float32)
    t = np.arange(sr) / sr  # "mmmmmm" from 0.6 s to 1.6 s: voiced, steady, swelling a little
    hum = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in (1, 2, 3)) * (0.05 + 0.05 * t)
    x[int(0.6 * sr): int(1.6 * sr)] += hum.astype(np.float32)
    tb = np.arange(int(0.3 * sr)) / sr  # ...BOOM at 1.6 s
    x[int(1.6 * sr): int(1.6 * sr) + tb.size] += (0.7 * np.sin(2 * np.pi * 110 * tb) * np.exp(-5 * tb)).astype(np.float32)
    voice = make_wav(tmp_path, x, sr)
    tr = track_of(n, w, h, lambda t: (100.0, 90.0), d=(0.0, -1.0), gesture="Open_Palm")
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(tr))
    spec = ShotSpec(plate=plate, voice=voice, effect="fire_burst", point=(100 / w, 90 / h), mode="scene",
                    plate_size=(w, h), max_plate_s=3.0)
    res = make_bending_shot(spec, GuideAsElement(), tmp_path / "charge", track_path)
    b = res.receipt["bending"]
    assert b["charge"] and abs(b["charge"]["from"] - 14) <= 3 and abs(b["charge"]["release"] - 38) <= 3
    assert [hh["frame"] for hh in b["hits"]] == [b["charge"]["release"]]  # the hum's start launched nothing
    out = read_video(res.composite).frames.astype(int)
    near = lambda f: out[f, 70:100, 85:115].mean()  # noqa: E731  (round the palm)
    assert near(30) > near(8) + 3  # charging: the ember lights the palm before the strike


def test_a_tracker_still_being_installed_is_not_ready(tmp_path, monkeypatch):
    import sys as _sys

    import cuesync.bending as bending

    venv = tmp_path / "cuesync" / ".venv-track"  # the installer's own environment
    exe = venv / ("Scripts/python.exe" if _sys.platform == "win32" else "bin/python")
    exe.parent.mkdir(parents=True)
    exe.write_text("")  # its Python exists, its packages don't yet
    monkeypatch.setattr(bending, "hand_python", lambda kind: str(exe))
    monkeypatch.setattr(bending, "_ready", {})
    monkeypatch.setattr(bending.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not asked")))
    assert not bending.helpers_ready()  # its mark or nothing: not even an import check while it installs
    (venv / bending.READY_MARK).write_text("ok\n")  # the installer checked it imports
    assert bending.helpers_ready()
