"""A foreshortened palm's projected tracker axis is not a user's aim arrow."""

import json

import numpy as np
import pytest

from cuesync.bending import bending_again, make_bending_shot
from cuesync.media import read_video, write_video
from cuesync.pipeline import ShotSpec


def _guide_renderer(guide_path, prompt, seed, out_path):
    """Replace only the external GPU render with the generated guide's lit frames."""
    frames = read_video(guide_path).frames.copy()
    last = frames[0]
    for i in range(len(frames)):
        if frames[i].max() > 40:
            last = frames[i].copy()
        frames[i] = last
    write_video(out_path, frames, fps=24)
    return 0.0


@pytest.mark.parametrize(
    "effect,gesture,emitter,aim,spec_direction,want,mode",
    [
        ("fire_burst", "Open_Palm", "palm", None, None, [0.0, -1.0], "upward_default"),
        ("flame_plume", "Open_Palm", "palm", None, None, [0.0, -1.0], "upward_default"),
        ("flame_plume", "Open_Palm", "palm", 180.0, None, [0.0, 1.0], "explicit"),
        ("fire_burst", "Open_Palm", "palm", None, 90.0, [1.0, 0.0], "explicit"),
        ("fire_breath", "Pointing_Up", "index", None, None, [0.0, 1.0], "tracked"),
        ("fire_burst", "Victory", "two_finger", None, None, [0.0, 1.0], "tracked"),
        ("auto", "Open_Palm", "palm", None, None, [0.0, -1.0], "upward_default"),
    ],
)
def test_direction_respects_intent_on_a_foreshortened_hand(
    tmp_path, make_video, make_wav, effect, gesture, emitter, aim, spec_direction, want, mode
):
    # Wrist-to-knuckle projects downward on this side-on palm. Fire should rise
    # unless an arrow or a pointing gesture intentionally gives it another aim.
    w = h = 128
    n, sr = 48, 48000
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (25, 25, 25))
    voice = np.zeros(2 * sr, np.float32)
    time = np.arange(int(0.2 * sr)) / sr
    voice[sr // 2:sr // 2 + len(time)] = 0.6 * np.sin(2 * np.pi * 120 * time) * np.exp(-6 * time)
    landmarks = [[64.0 + (i % 5 - 2) * 4, 70.0 + i * 0.45, -(i // 4) * 6.0] for i in range(21)]
    track = {
        "version": 1, "fps": 24, "W": w, "H": h, "emitter": emitter, "hand": "right",
        "xy": [[64.0, 64.0]] * n, "dir": [[0.0, 1.0]] * n, "scale": [16.0] * n,
        "conf": [1.0] * n, "src": ["mp"] * n, "speed": [0.0] * n,
        "landmarks": [landmarks] * n, "gesture": [gesture] * n, "events": [],
    }
    track_path = tmp_path / "hand_track.json"
    track_path.write_text(json.dumps(track))
    spec = ShotSpec(
        plate=plate, voice=make_wav(tmp_path, voice, sr), effect=effect,
        point=(0.5, 0.5), direction=spec_direction, plate_size=(w, h), max_plate_s=2.0,
    )
    result = make_bending_shot(spec, _guide_renderer, tmp_path / "shot", track_path, aim=aim)
    receipt = result.receipt["bending"]
    assert np.allclose(receipt["ref"]["dir"], want, atol=1e-5)
    assert receipt["aim_mode"] == mode
    if mode == "upward_default":
        assert receipt["aim"] == 0.0
    if effect == "flame_plume" and aim is None:
        # Reusing the source must not turn a recorded default into a drawn arrow.
        again = bending_again(result.out_dir, spec, tmp_path / "again")
        assert again.receipt["bending"]["aim_mode"] == "upward_default"


def test_lightning_without_an_arrow_follows_a_downward_pointing_finger(tmp_path):
    from test_lightning import take

    # Rotate the known synthetic extended-finger fixture to point down. The
    # rendered bolt must leave below its tip even though palm fire defaults up.
    n, w, h = 24, 128, 128
    plate, track, mask = take(n, lambda t: (64, 90), 20, size=(w, h), wall=(25, 25, 25), texture=0)
    plate, mask = plate[:, ::-1].copy(), mask[:, ::-1].copy()
    track["emitter"], track["gesture"], track["dir"] = "index", ["Pointing_Up"] * n, [[0.0, 1.0]] * n
    for point, landmarks in zip(track["xy"], track["landmarks"]):
        point[1] = h - 1 - point[1]
        for landmark in landmarks:
            landmark[1] = h - 1 - landmark[1]
    plate_path, track_path, mask_path = tmp_path / "plate.mp4", tmp_path / "hand_track.json", tmp_path / "hand_mask.npz"
    write_video(plate_path, plate, fps=24)
    track_path.write_text(json.dumps(track))
    np.savez_compressed(mask_path, alpha=mask)
    point = track["xy"][0]
    spec = ShotSpec(plate=plate_path, voice=None, effect="lightning_arc", point=(point[0] / w, point[1] / h),
                    cue="hold", holds=[(0.0, 0.8)], plate_size=(w, h), max_plate_s=1.0)
    result = make_bending_shot(spec, None, tmp_path / "shot", track_path, mask_path)
    assert result.receipt["bending"]["aim"] is None
    assert result.receipt["bending"]["aim_mode"] == "tracked"
    rendered = read_video(result.composite).frames.astype(np.float32)
    blue_added = np.maximum(rendered[..., 2] - plate[..., 2].astype(np.float32) - 40, 0)
    tip_y = int(point[1])
    assert blue_added[:, tip_y + 8:].sum() > 2 * blue_added[:, :tip_y - 8].sum()
    assert blue_added[:, tip_y + 8:].max() > 100
