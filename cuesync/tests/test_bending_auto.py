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
