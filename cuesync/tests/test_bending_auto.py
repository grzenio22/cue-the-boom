"""Element bending: which element a hand's gesture and the sound ask for, and when it goes off (cuesync.bending)."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuesync.bending import bending_effect, gesture_at, snap_launch


def test_the_hand_and_the_sound_pick_the_element():
    assert bending_effect("hiss", "Victory")[0] == "fire_breath"  # two fingers, "ffffff": a flamethrower
    assert bending_effect("whoosh", "Pointing_Up")[0] == "fire_breath"
    assert bending_effect("hiss", "Open_Palm")[0] == "flame_plume"  # a held "fffff" over a palm: a flame in it
    assert bending_effect("puff", "Open_Palm")[0] == "fire_burst"  # "pf" from a palm: a pop of fire (the hand throws fire)
    assert bending_effect("puff", None)[0] == "fire_burst" and bending_effect("puff", "Thumb_Up")[0] == "fire_burst"
    assert bending_effect("puff", None, lit=True)[0] == "blow_out"  # (a flame already burning there is blown out)
    assert bending_effect("boom", "Open_Palm")[0] == "fire_burst"  # "BOOM" from a palm: a fireball
    assert bending_effect("boom", "Closed_Fist")[0] == "fire_burst"
    assert bending_effect("boom", "Victory", hits=3, span_s=1.2)[0] == "fire_breath"  # jab, jab, strike from the fingers
    assert bending_effect("puff", None, snap=True)[0] == "blow_out"
    assert bending_effect(None, "Victory")[0] is None
    effect, why = bending_effect("hiss", None)  # no gesture read: the sound alone
    assert effect == "steam_jet" and "gesture" in why


def test_the_strike_takes_the_hit_only_close_to_the_voice():
    events = [{"frame": 40, "kind": "extension", "peak_speed": 2000.0}]
    assert snap_launch(42, events) == {"cue": 42, "motion": 40, "frame": 40, "used": "motion", "mismatch": False}
    assert snap_launch(39, events)["used"] == "motion"  # the arm locks a frame after the voice: still fine
    assert snap_launch(38, events)["used"] == "cue" and snap_launch(38, events)["mismatch"]  # two frames late: no
    assert snap_launch(45, events)["used"] == "cue"  # five frames early: no
    assert snap_launch(45, events, fps=48)["used"] == "motion"  # windows scale with the frame rate
    assert snap_launch(42, [])["used"] == "cue" and not snap_launch(42, [])["mismatch"]


def test_a_flickering_gesture_is_read_over_a_few_frames():
    track = {"gesture": ["None", "Victory", "Open_Palm", "Victory", "Victory", "None", "None"], "src": ["mp"] * 7}
    assert gesture_at(track, 3) == "Victory"
    assert gesture_at({"gesture": ["None"] * 5, "src": ["mp"] * 5}, 2) is None


def test_a_buzz_is_lightning_and_a_long_strike_boom_an_explosion():
    assert bending_effect("buzz", "Pointing_Up")[0] == "lightning_arc"  # "zzzzz" from a finger: lightning
    assert bending_effect("buzz", "Open_Palm")[0] == "lightning_ball"  # over a palm: ball lightning
    assert bending_effect("buzz", None)[0] == "lightning_arc"
    assert bending_effect("boom", "Closed_Fist", longest_s=0.8)[0] == "explosion"  # a rolling "BOOOOM" with a punch
    assert bending_effect("boom", "Open_Palm", longest_s=0.6)[0] == "explosion"
    assert bending_effect("boom", "Closed_Fist", longest_s=0.25)[0] == "fire_burst"  # a short BOOM: a fireball still
    assert bending_effect("boom", "Pointing_Up", longest_s=0.8)[0] == "fire_burst"  # (not from a fingertip)


def test_a_voiced_buzz_is_heard_as_one():
    import numpy as np

    from cuesync.audio import analyze
    from cuesync.gestures import heard

    sr = 48000
    t = np.arange(int(1.2 * sr)) / sr
    rng = np.random.default_rng(5)
    hum = 0.3 * np.sin(2 * np.pi * 140 * t)  # the voice's pitch...
    hiss = rng.normal(0, 1.0, t.size)  # ...with the buzz's hiss on it, high up
    spec = np.fft.rfft(hiss)
    f = np.fft.rfftfreq(t.size, 1 / sr)
    spec[(f < 4500) | (f > 9000)] = 0
    hiss = np.fft.irfft(spec, n=t.size)
    hiss *= 0.25 / np.sqrt((hiss ** 2).mean())
    x = rng.normal(0, 0.001, int(3 * sr)).astype(np.float32)
    x[sr // 2: sr // 2 + t.size] += (hum + hiss).astype(np.float32)
    h = heard(analyze(x, sr), x, sr)
    assert h["kind"] == "buzz", h
    assert h["longest_s"] > 1.0


def test_a_sample_noise_lands_on_the_sample_hands_strike(tmp_path, make_wav):
    """Started with the clip, a sample BOOM went off long before the punch it was for: it is placed with its first
    sound on the hand's marked frame, earlier only as far as it takes to end in the clip (a held noise letting go),
    and never more than SAMPLE_LEAD_S before the hand is there."""
    from cuesync.bending import SAMPLE_LEAD_S, SAMPLE_TAIL_S, sample_noise_offset

    sr = 48000
    rng = np.random.default_rng(0)
    hit = np.zeros(int(1.5 * sr), np.float32)
    hit[int(0.2 * sr): int(0.35 * sr)] = 0.8 * rng.standard_normal(int(0.15 * sr))  # one burst at 0.2 s
    held = 0.5 * rng.standard_normal(int(2.2 * sr)).astype(np.float32)
    held[int(2.0 * sr):] = 0  # held from the start for 2 s
    on_hit = sample_noise_offset(make_wav(tmp_path, hit, sr), 3.5, 48)
    assert abs(on_hit + 0.2 - 48 / 24) < 0.06  # its burst on frame 48
    on_held = sample_noise_offset(make_wav(tmp_path, held, sr), 4.3, 48)
    assert on_held + 2.0 <= 4.3 - SAMPLE_TAIL_S + 0.06 and on_held < 48 / 24  # let go inside the clip
    short = sample_noise_offset(make_wav(tmp_path, held, sr), 3.5, 48)  # (in a clip too short for that, it starts
    assert abs(short - (48 / 24 - SAMPLE_LEAD_S)) < 0.06  # no earlier than just before the hand is there)


def test_a_sample_hands_track_is_moved_onto_the_elements_point():
    """A sample hand followed for its fireball's palm, given lightning: its point moves onto the fingertips (read off
    its landmarks), and where a frame has none it moves as its nearest frame's did."""
    from cuesync.bending import track_for

    lm = [[100.0 + 10 * (i % 5), 400.0 - 15 * i, 0.0] for i in range(21)]  # a hand pointing up the picture
    t = {"version": 1, "W": 1080, "H": 1920, "emitter": "palm", "hand": "right", "xy": [[120.0, 300.0]] * 3,
         "dir": [[0.0, 1.0]] * 3, "landmarks": [lm, None, lm]}
    moved = track_for(t, "two_finger")
    tips = (np.array(lm[8][:2]) + np.array(lm[12][:2])) / 2
    assert moved["emitter"] == "two_finger" and np.linalg.norm(np.array(moved["xy"][0]) - tips) < 40
    assert moved["dir"][0][1] < 0  # along the fingers, up the picture, no longer down out of the palm
    assert np.allclose(moved["xy"][1], moved["xy"][0], atol=1e-6)  # (no landmarks: moved as its neighbour was)
    assert track_for(t, "palm") is t


@pytest.mark.parametrize("name", ["pf-pf-ka-boom.wav", "pf-pf-boom-author.wav", "pf-x4-author.wav",
                                 "zzzzz-tsst.wav", "hoo-hoo-HOO.wav", "ka-boom.wav"])
@pytest.mark.parametrize("frame", [68, 83])
def test_a_sample_noises_loudest_hit_lands_on_the_strike(name, frame):
    from cuesync.audio import analyze
    from cuesync.bending import SAMPLE_TAIL_S, sample_noise_offset
    from cuesync.media import read_audio
    from cuesync.pipeline import FPS, SR

    voice = Path("samples/voices") / name
    offset = sample_noise_offset(voice, 3.5, frame)
    cue = analyze(read_audio(voice, SR), SR, offset_s=offset, fps=FPS)
    assert abs(cue.hero_onset_s - min(frame / FPS, 3.5 - SAMPLE_TAIL_S)) < 1 / FPS
    assert 0 <= cue.hero_onset_s <= 3.5 - SAMPLE_TAIL_S


@pytest.mark.parametrize("sample", [True, False])
def test_a_made_sample_keeps_its_elements_emitter_and_a_custom_tap(tmp_path, monkeypatch, sample):
    import cuesync.bending as bending
    from cuesync.pipeline import ShotSpec

    src = tmp_path / "src"
    src.mkdir()
    lm = [[100.0 + 10 * (i % 5), 400.0 - 15 * i, 0.0] for i in range(21)]
    tr = {"version": 1, "W": 1080, "H": 1920, "emitter": "two_finger", "hand": "right",
          "xy": [[120.0, 300.0]] * 3, "dir": [[0.0, -1.0]] * 3, "scale": [100.0] * 3,
          "landmarks": [lm] * 3}
    (src / "hand_track.json").write_text(json.dumps(tr))
    (src / "receipt.json").write_text(json.dumps({"bending": {"track": "hand_track.json", "mask": None}}))
    got = {}

    def made(spec, renderer, out, track, mask, progress, **kw):
        got["track"] = bending.anchored(bending.load_track(track), kw["tap_frame"],
                                        (spec.point[0] * tr["W"], spec.point[1] * tr["H"]))

    monkeypatch.setattr(bending, "make_bending_shot", made)
    spec = ShotSpec(plate=tmp_path / "p.mp4", voice=None, effect="water_orb", point=(0.2, 0.3),
                    extra={"tap_frame": 1, "sample_emitter": sample})
    bending.bending_again(src, spec, tmp_path / "again")
    point = np.asarray(got["track"]["xy"][1])
    if sample:
        from cuesync.hand_track import emitter_points

        palm, _, _ = emitter_points(np.asarray(lm), "palm", 1.0, frame=(1080, 1920))
        assert np.linalg.norm(point - palm) < 1
    else:
        assert point == pytest.approx([216.0, 576.0])


@pytest.mark.parametrize("effect", ["water_orb", "fire_burst"])
def test_performing_again_uses_the_prepared_hand_without_changing_the_element(tmp_path, monkeypatch, effect):
    import cuesync.bending as bending
    from cuesync.pipeline import ShotSpec

    src, reuse = tmp_path / "src", tmp_path / "prep"
    src.mkdir()
    reuse.mkdir()
    (src / "receipt.json").write_text(json.dumps({"bending": {"track": "hand_track.json", "mask": None}}))
    for folder, x in ((src, 10.0), (reuse, 80.0)):
        (folder / "hand_track.json").write_text(json.dumps({"version": 1, "W": 100, "H": 100, "emitter": "palm",
                                                            "xy": [[x, 50.0]], "dir": [[0.0, -1.0]]}))
        (folder / "hand_mask.npz").write_bytes(b"mask")
    got = {}

    def made(spec, renderer, out, track, mask, progress, **kw):
        got.update(track=bending.load_track(track), mask=mask, **kw)

    monkeypatch.setattr(bending, "make_bending_shot", made)
    spec = ShotSpec(plate=tmp_path / "p.mp4", voice=None, effect=effect, point=(0.8, 0.5),
                    extra={"tap_frame": 0, "reuse": str(reuse)})
    bending.bending_again(src, spec, tmp_path / "again")
    assert got["track"]["xy"][0] == [80.0, 50.0]
    assert Path(got["mask"]) == reuse / "hand_mask.npz"
    if effect == "fire_burst":
        assert got["element_from"] == src


def test_an_explosions_foley_only_fires_on_its_visible_hero(tmp_path, monkeypatch):
    import cuesync.bending as bending
    from cuesync import explosion, media
    from cuesync.pipeline import FPS, SR, ShotSpec

    n = 72
    plate = np.full((n, 32, 32, 3), 30, np.uint8)
    track = {"xy": [[16.0, 16.0]] * n, "dir": [[0.0, -1.0]] * n, "scale": [8.0] * n}
    cue = SimpleNamespace(intervals=lambda: [], voice_end_frame=70)
    c = SimpleNamespace(samples=np.zeros(3 * SR, np.float32), cue=cue, offset_s=0.0, source="voice")
    spec = ShotSpec(plate=tmp_path / "p.mp4", voice=None, effect="explosion", point=(0.5, 0.5))
    out = tmp_path / "shot"
    out.mkdir()
    audio = []
    monkeypatch.setattr(explosion, "render", lambda plate, *a, **k: plate)
    monkeypatch.setattr(media, "write_video", lambda path, frames, **kw: audio.append(kw["audio"]))
    snaps = [{"frame": f} for f in (10, 24, 50)]
    env = np.zeros(n)
    env[10:16], env[24:30], env[50:58] = 0.65, 0.7, 1.0
    bending.made_shot(spec, out, "explosion", plate, track, None, c, env, snaps,
                      [(0.65, 0.6), (0.7, 0.7), (1.0, 1.0)], 50, 12, None, None, None, None, False,
                      "track.json", None, lambda *a: None, time.perf_counter())
    spf = SR // FPS
    assert np.max(np.abs(audio[0][:50 * spf])) < 1e-6
    assert np.sqrt(np.mean(audio[0][50 * spf:54 * spf] ** 2)) > 0.05


def test_held_procedural_sample_fits_before_the_end_of_a_short_take(tmp_path, make_wav):
    from cuesync.bending import SAMPLE_TAIL_S, sample_noise_offset
    sr=48000
    held=np.random.default_rng(8).normal(0,.3,3*sr).astype(np.float32)
    held[int(2.3*sr):]=0
    voice=make_wav(tmp_path,held,sr)
    offset=sample_noise_offset(voice,3.5,78,fit_held=True)
    assert 0<=offset and offset+2.3<=3.5-SAMPLE_TAIL_S+.06
