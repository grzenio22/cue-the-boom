"""Element bending: which element a hand's gesture and the sound ask for, and when it goes off (cuesync.bending)."""

import numpy as np

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
