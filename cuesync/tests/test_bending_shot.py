"""Element bending, carried on the hand: the warp follows the tracked emitter (base at once, tip late), the element's
ignition lands on the strike, and a whole shot runs with a fake renderer (cuesync.bending)."""

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_a_lone_pf_performed_on_a_booms_render_is_a_pop(tmp_path, make_video, make_wav):
    """However loud a lone unvoiced "pf" is, played on a BOOM's fireball it is a pop, not the fireball."""
    w, h, n = 256, 128, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (40, 35, 30))
    sr = 48000
    t = np.arange(int(0.25 * sr)) / sr
    x = np.random.default_rng(3).normal(0, 0.001, 3 * sr).astype(np.float32)
    x[sr: sr + t.size] += (0.6 * np.sin(2 * np.pi * 120 * t) * np.exp(-6 * t)).astype(np.float32)  # BOOM
    pf = np.random.default_rng(4).normal(0, 0.001, 3 * sr).astype(np.float32)
    pf[sr: sr + int(0.08 * sr)] += np.random.default_rng(5).normal(0, 0.3, int(0.08 * sr)).astype(np.float32)  # pf
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(track_of(n, w, h, lambda t: (90.0, 64.0))))
    spec = ShotSpec(plate=plate, voice=make_wav(tmp_path, x, sr), effect="fire_burst", point=(0.35, 0.5), mode="scene",
                    plate_size=(w, h), max_plate_s=3.0)
    first = make_bending_shot(spec, GuideAsElement(), tmp_path / "one", track_path)
    again = make_bending_shot(ShotSpec(plate=plate, voice=make_wav(tmp_path, pf, sr), effect="auto", point=(0.35, 0.5),
                                       mode="scene", plate_size=(w, h), max_plate_s=3.0),
                              GuideAsElement(), tmp_path / "again", track_path, element_from=first.out_dir)
    [hit] = again.receipt["bending"]["hits"]
    assert hit["size"] <= 0.5 and hit["gain"] <= 0.7


def test_performing_again_puts_a_render_of_another_size_where_the_first_shot_did(tmp_path, make_video, make_wav):
    """A 96 px tall plate renders at 128 (as 1080 renders at 1088): the render is placed by its pose. Its receipt must
    say where the emitter is in the render as saved, or performing it again moves the fire off the hand."""
    w, h, n = 256, 96, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (40, 35, 30))
    sr = 48000
    t = np.arange(int(0.25 * sr)) / sr
    x = np.random.default_rng(3).normal(0, 0.001, 3 * sr).astype(np.float32)
    x[sr: sr + t.size] += (0.6 * np.sin(2 * np.pi * 120 * t) * np.exp(-6 * t)).astype(np.float32)
    voice = make_wav(tmp_path, x, sr)
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(track_of(n, w, h, lambda t: (60.0, 70.0), d=(0.0, -1.0))))
    spec = ShotSpec(plate=plate, voice=voice, effect="fire_burst", point=(60 / w, 70 / h), mode="scene",
                    plate_size=(w, h), max_plate_s=3.0)
    first = make_bending_shot(spec, GuideAsElement(), tmp_path / "one", track_path)
    again = make_bending_shot(spec, GuideAsElement(), tmp_path / "again", track_path, element_from=first.out_dir)
    k = first.receipt["bending"]["hits"][0]["frame"] + 4
    base = read_video(plate).frames[k].astype(float).mean(-1)

    def fire_x(path):
        f = np.maximum(read_video(path).frames[k].astype(float).mean(-1) - base - 20, 0)
        return float((f * np.arange(w)).sum() / f.sum())

    assert abs(fire_x(again.composite) - fire_x(first.composite)) < 2


@pytest.mark.parametrize("effect", ["fire_burst", "flame_plume"])
def test_a_burst_or_a_flame_on_a_portrait_clip_is_rendered_landscape_and_carried_onto_the_hand(tmp_path, make_video,
                                                                                              make_wav, effect):
    """On a portrait canvas the model drew a narrow fountain of sparks for a fireball, and a thin wisp for a palm's
    flame: both render landscape, and still land on the hand."""
    w, h, n = 128, 256, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (40, 35, 30))
    sr = 48000
    t = np.arange(int(0.25 * sr)) / sr
    x = np.random.default_rng(3).normal(0, 0.001, 3 * sr).astype(np.float32)
    x[sr: sr + t.size] += (0.6 * np.sin(2 * np.pi * 120 * t) * np.exp(-6 * t)).astype(np.float32)
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(track_of(n, w, h, lambda t: (40.0, 200.0), d=(0.0, -1.0))))
    r = GuideAsElement()
    spec = ShotSpec(plate=plate, voice=make_wav(tmp_path, x, sr), effect=effect, point=(40 / w, 200 / h),
                    mode="scene", plate_size=(w, h), max_plate_s=3.0)
    res = make_bending_shot(spec, r, tmp_path / "shot", track_path)
    guide = read_video(tmp_path / "shot" / "guide.mp4").frames
    assert guide.shape[2] > guide.shape[1]  # landscape
    comp = read_video(res.composite).frames.astype(float).mean(-1)
    hit = res.receipt["bending"]["hits"][0]["frame"]
    k = max(range(hit, len(comp)), key=lambda i: np.maximum(comp[i] - 60, 0).sum())  # (a flame grows in; a burst is at once)
    fire = np.maximum(comp[k] - 60, 0)
    ys, xs = np.nonzero(fire)
    assert xs.size and abs(float(np.average(xs, weights=fire[ys, xs])) - 40) < 20  # above the hand, where it was tapped
    assert float(np.average(ys, weights=fire[ys, xs])) < 200


def test_a_held_flame_loops_while_the_voice_holds_and_lets_go_when_it_stops():
    from cuesync.bending import held_frames

    ign, rel, end = 10, 40, 52  # 30 frames burning, then 12 letting go
    as_rendered = [held_frames(d, ign, rel, end, hold=30) for d in range(-3, 45)]
    assert [p[0][0] for p in as_rendered if p] == list(range(7, 52)) and all(len(p) == 1 for p in as_rendered if p)
    long = [held_frames(d, ign, rel, end, hold=90) for d in range(110)]
    assert all(ign <= f < rel for p in long[:90] for f, _ in p)  # burning for as long as the voice holds on
    assert all(abs(sum(wt for _, wt in p) - 1.0) < 1e-6 for p in long[:90])  # (the loop's seam cross-faded)
    assert long[97] == [(rel + 7, 1.0)] and long[102] == []  # then its own letting go, and out
    lead = 6  # every wrap is cross-faded, the first one too: where the frame shown jumps, it is a blend
    top = [max(p, key=lambda fw: fw[1]) for p in long[:90]]
    for d in range(1, 90):
        if abs(top[d][0] - top[d - 1][0]) > 1:
            assert len(long[d]) == 2 and max(wt for _, wt in long[d]) <= 0.75, (d, long[d - 1], long[d])
    assert long[30][0][0] == ign + lead and top[29][0] == ign + lead - 1  # (the first wrap lands on its start from just before it)
    assert held_frames(12, ign, rel, end, hold=10)[-1][0] == rel + 2  # stopping early lets go from where it was
    dying = [sum(wt for _, wt in held_frames(d, ign, rel, rel, hold=10)) for d in range(10, 22)]  # no letting go rendered
    assert all(a > b for a, b in zip(dying, dying[1:10])) and dying[0] > 0.85 and dying[-1] == 0  # dies down, not cut


def test_performed_again_a_quiet_take_plays_the_render_smaller_than_the_take_it_was_made_for():
    from cuesync.bending import hit_looks

    ref = 0.9  # the loudest hit of the take the render was made for
    [(g, s)] = hit_looks([ref - 18 / 54], ref)  # a lone "pf" 18 dB under that BOOM
    assert s < 0.6 and g < 1.0
    assert hit_looks([ref - 4 / 54], ref) == [(1.0, 1.0)]  # within the slack: as rendered
    assert hit_looks([ref + 0.05], ref) == [(1.0, 1.0)]  # louder: no bigger than rendered


class GuideAsKeyframes(GuideAsElement):
    """Renders 'the element' as its guide, each keyframe held until the next (8 frames) and then gone: it burns as
    long as the take it was made for."""

    def __call__(self, guide_path, prompt, seed, out_path, image=None):
        self.prompts.append(prompt)
        g = read_video(guide_path).frames.copy()
        last, at = g[0], -99
        for i in range(len(g)):
            if g[i].max() > 40:
                last, at = g[i].copy(), i
            g[i] = last if i - at < 8 else 0
        write_video(out_path, g, fps=24)
        return 0.1


def test_a_flamethrower_performed_again_burns_as_long_as_the_new_fffff(tmp_path, make_video, make_wav):
    import dataclasses

    w, h, n = 256, 128, 96
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (40, 35, 30 + (i % 2)))
    sr = 48000

    def fffff(seconds):  # a hiss from 1 s
        x = np.random.default_rng(4).normal(0, 0.001, 4 * sr).astype(np.float32)
        x[sr: sr + int(seconds * sr)] += np.random.default_rng(5).normal(0, 0.15, int(seconds * sr)).astype(np.float32)
        return make_wav(tmp_path, x, sr)

    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(track_of(n, w, h, lambda t: (60.0, 64.0))))
    spec = ShotSpec(plate=plate, voice=fffff(0.5), effect="fire_breath", point=(60 / w, 64 / h), mode="scene",
                    plate_size=(w, h), max_plate_s=4.0)
    first = make_bending_shot(spec, GuideAsKeyframes(), tmp_path / "one", track_path)
    again = make_bending_shot(dataclasses.replace(spec, voice=fffff(2.0)), GuideAsKeyframes(), tmp_path / "again",
                              track_path, element_from=first.out_dir)
    held = again.receipt["bending"]["held"]
    assert held and held["holds"][0] > 36  # the new fffff holds on for most of its two seconds
    k = 24 + 41  # 1.7 s into the fffff: the first take's flame has long burned out
    base = read_video(plate).frames[k].astype(float)

    def lit(path):
        return float(np.maximum(read_video(path).frames[k].astype(float) - base - 6, 0).sum())

    assert lit(again.composite) > 3 * lit(first.composite) + 1000


def test_a_hit_plays_as_big_as_it_is_loud_in_db():
    """The cue's envelope is in dB (54 dB over 0..1): a "pf" 12 dB under the "BOOM" plays about half its size."""
    from cuesync.bending import hit_looks

    (g0, s0), (g1, s1) = hit_looks([0.9 - 12 / 54, 0.9])
    assert (g1, s1) == (1.0, 1.0)
    assert 0.45 <= s0 <= 0.55 and s0 < g0 < 1.0


def test_a_hand_that_leaves_the_frame_is_held_where_it_went_out():
    from cuesync.bending import left_frame

    tr = track_of(10, 200, 100, lambda t: (100 - 30 * t, 50), d=(-1.0, 0.0), scale=20.0)
    tr["dir"][6:] = [[1.0, 0.0]] * 4  # the trackers' guesses once it is gone: turned round
    held, gone = left_frame(tr, 200, 100)
    assert list(gone) == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]  # out past half a hand (x < -10) from frame 4
    assert held["xy"][9] == held["xy"][3] and held["dir"][9] == [-1.0, 0.0]


def test_fire_over_a_white_wall_keeps_its_colour(tmp_path, make_video, make_wav):
    """Added alone, fire on a white wall went white on white; its soot keeps it orange."""
    w, h, n = 256, 128, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (235, 235, 230))
    sr = 48000
    t = np.arange(int(0.25 * sr)) / sr
    x = np.random.default_rng(3).normal(0, 0.001, 3 * sr).astype(np.float32)
    x[sr: sr + t.size] += (0.6 * np.sin(2 * np.pi * 120 * t) * np.exp(-6 * t)).astype(np.float32)
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(track_of(n, w, h, lambda t: (128.0, 100.0), d=(0.0, -1.0))))
    spec = ShotSpec(plate=plate, voice=make_wav(tmp_path, x, sr), effect="fire_burst", point=(0.5, 100 / h), mode="scene",
                    plate_size=(w, h), max_plate_s=3.0)
    res = make_bending_shot(spec, GuideAsElement(), tmp_path / "shot", track_path)
    k = res.receipt["bending"]["hits"][0]["frame"] + 4
    f = read_video(res.composite).frames[k].astype(int)
    assert (f[..., 0] - f[..., 2]).max() > 80  # orange against the white, somewhere


def test_the_fire_is_smeared_as_far_as_the_hand_moved():
    from cuesync.bending import motion_blur

    E = np.zeros((60, 120, 3), np.float32)
    E[28:32, 58:62] = 1.0
    still = motion_blur(E, np.array([0.5, 0.0]))
    assert still is E  # a still hand: as it is
    moved = motion_blur(E, np.array([20.0, 0.0]))  # 20 px a frame to the right: smeared over half of that, sideways
    lit = np.nonzero(moved[30, :, 0] > 1e-3)[0]
    assert 12 <= lit.max() - lit.min() <= 16 and np.nonzero(moved[:, 60, 0] > 1e-3)[0].size <= 6
    assert abs(float(moved.sum()) - float(E.sum())) < 1e-3 * float(E.sum())  # no light gained or lost


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


def test_a_tapped_fingertip_stays_on_the_fingertip_as_it_curls():
    from cuesync.bending import anchored

    def hand(tip):  # the palm's points still, the index fingertip (8) where given
        return [list(tip) + [0.0] if j == 8 else [50.0, 60.0 + j, 0.0] for j in range(21)]

    tr = track_of(3, 200, 100, lambda t: (50, 60), d=(0.0, -1.0), scale=10.0)  # the tracker's point: the palm
    tr["landmarks"] = [hand((80, 20)), hand((55, 50)), None]  # pointing out, curled into a fist, lost
    a = anchored(tr, 0, (80.0, 20.0))
    assert np.allclose(a["xy"][0], (80, 20), atol=0.1) and a["anchor"]["on"][0] == 8
    assert np.hypot(a["xy"][1][0] - 55, a["xy"][1][1] - 50) < 1.0  # on the curled fingertip, not 5 hands up in the air
    assert a["xy"][2] == a["xy"][1]  # lost: held where it was last seen


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


def test_a_tap_with_no_hand_is_said_and_a_lost_one_is_followed_again_from_a_clearer_frame(tmp_path, monkeypatch):
    import cuesync.bending as bending

    calls = []

    def fake_track(plate, out_dir, tap, frame, emitter, lost=None):
        calls.append((frame, tuple(round(v) for v in tap)))
        n = 40
        src = lost(len(calls)) if lost else ["mp"] * n
        t = track_of(n, 200, 100, lambda i: (50.0 + i, 50.0)) | {"src": src, "conf": [0.5 + 0.01 * i for i in range(n)]}
        (Path(out_dir) / "hand_track.json").write_text(json.dumps(t))
        return Path(out_dir) / "hand_track.json"

    monkeypatch.setattr(bending, "track_hand", lambda *a: fake_track(*a, lost=lambda c: ["lost"] * 40))
    import pytest
    with pytest.raises(RuntimeError, match="couldn't find a hand"):  # a tap on the background
        bending.follow_tap(Path("p.mp4"), tmp_path, (10.0, 10.0), 5, "palm")
    calls.clear()
    # the first follow loses the hand on most of the clip; from its surest frame it holds it
    monkeypatch.setattr(bending, "track_hand", lambda *a: fake_track(*a, lost=lambda c: (["lost"] * 24 + ["mp"] * 16) if c == 1
                                                                      else (["mp"] * 38 + ["lost"] * 2)))
    track = bending.follow_tap(Path("p.mp4"), tmp_path, (55.0, 50.0), 5, "palm")
    assert len(calls) == 2 and calls[1][0] == 39  # followed again from the frame it was seen most surely
    frame, point = bending.retapped(track, 5, (0.275, 0.5))
    assert frame == 39 and abs(point[0] - 89.0 / 200) < 0.01  # the tap carried onto that frame, with the hand


def test_a_small_flame_on_a_big_black_canvas_still_ignites():
    """A palm's flame on a landscape canvas, a hundredth of it, never lifted a frame's mean over its black: the render
    read as empty. It is measured where it is."""
    el = np.zeros((20, 216, 384, 3), np.uint8)
    for i in range(6, 20):
        el[i, 150:150 + 2 * (i - 5), 190:196] = (255, 150, 40)  # a thin flame growing from frame 6
    assert ignition_frame(el, 3) is not None and 6 <= ignition_frame(el, 3) <= 9


def test_a_tap_on_a_frame_the_hand_was_not_seen_on_goes_on_the_part_of_it_nearest():
    """A finger gun's recoil blurred the hand on the tapped frame: measured from where the hand was first seen, the tap
    landed 0.7 hands off the fingertip. It goes on the part of the hand nearest it, on the nearest frame it was seen."""
    from cuesync.bending import anchored

    hand = [[60.0, 40.0, 0.0] if j == 8 else [50.0, 70.0 + j, 0.0] for j in range(21)]
    tr = track_of(3, 100, 100, lambda t: (50, 70), d=(0.0, -1.0), scale=10.0)
    tr["src"], tr["landmarks"] = ["hold", "hold", "mp"], [None, None, hand]
    a = anchored(tr, 0, (60.0, 20.0))  # tapped above the fingertip on frame 0, where the hand was a blur
    assert np.allclose(a["xy"][2], (60, 40), atol=0.01) and a["anchor"]["on"] == [8]


def test_an_element_made_ahead_for_a_close_up_hand_plays_as_big_as_a_render_for_this_hand():
    """The library's fireball was painted for a close-up hand (713 px) and capped at 0.45 of its frame: 0.68 of that
    hand. Hand for hand on a 268 px hand it was a pilot light; a render for that hand reaches 1.8 hands."""
    from cuesync.bending import library_size

    c = library_size("burst", 713.0, 1088, 268.0, 1080)
    assert 2.5 < c < 2.8  # (486 px wanted against 182)
    assert library_size("burst", 268.0, 1080, 268.0, 1080) == pytest.approx(1.0)  # its own hand: as rendered
    assert library_size("jet", 256.0, 1088, 268.0, 1080) == pytest.approx(0.96, abs=0.02)  # a jet was capped alike
