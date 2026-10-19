"""Element bending: which element a hand's gesture and the sound ask for, and when it goes off (cuesync.bending)."""

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
