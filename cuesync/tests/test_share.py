"""Share exports caption each element with its own cue, including procedural effects."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageDraw

from cuesync import media, share


@pytest.fixture
def export_caption(tmp_path, monkeypatch):
    """Run the real layout with a one-frame plate and capture the text that it draws."""
    monkeypatch.setattr(share, "SIZE", (90, 160))
    monkeypatch.setattr(media, "read_video", lambda _: SimpleNamespace(frames=np.zeros((1, 160, 90, 3), np.uint8)))
    audio = np.array([0.1, -0.1], np.float32)
    monkeypatch.setattr(media, "read_audio", lambda *_: audio)
    written = {}

    def write_video(path, frames, **kwargs):
        written.update(path=path, frames=frames, **kwargs)

    monkeypatch.setattr(media, "write_video", write_video)
    drawn = []
    original_text = ImageDraw.ImageDraw.text

    def draw_text(self, xy, text, *args, **kwargs):
        drawn.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", draw_text)

    def export(effect):
        (tmp_path / "receipt.json").write_text(json.dumps({"effect": effect, "bending": {"hits": [{"frame": 0}]}}))
        out = tmp_path / "share.mp4"
        assert share.share_reel(tmp_path, out) == out
        assert written["frames"].shape == (1, 160, 90, 3)
        assert written["audio"] is audio
        assert written["fps"] == 24
        assert share.TAG in drawn
        return set(drawn) - {share.TAG}

    return export


@pytest.mark.parametrize(("effect", "expected"), [
    ("fire_burst", "BOOM"), ("fireball", "BOOM"), ("explosion", "BOOM"),
    ("fire_breath", "fwoooOOOM"), ("flame_plume", "fwoooOOOM"), ("wick_flare", "fwoooOOOM"),
    ("steam_jet", "hissss"), ("smoke_puff", "pf"), ("blow_out", "pf"),
    ("lightning_arc", "zzzzz"), ("lightning_ball", "zzzzz"),
    ("water_orb", "sploosh"), ("water_whip", "shhhhh"),
    ("air_ball", "hoooooo"), ("air_blast", "HOO!"),
])
def test_share_draws_the_supported_effects_cue(export_caption, effect, expected):
    assert export_caption({"id": effect}) == {expected}


@pytest.mark.parametrize("effect", [{"id": "custom_mist"}, {"id": ""}, {}, None])
def test_unknown_effect_keeps_the_reel_without_a_misleading_caption(export_caption, effect):
    assert export_caption(effect) == set()
