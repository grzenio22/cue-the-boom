"""Instant mode: library elements placed at your point, chosen by your performance, timed by your voice."""

import json
from pathlib import Path

import numpy as np
import pytest

from cuesync.library import LIBRARY_POINT, LIBRARY_SIZE, LibraryRenderer, place, reference_performance
from cuesync.media import write_video
from cuesync.pipeline import ShotSpec, make_shot


def test_place_moves_the_emission_point_and_scales_about_it():
    f = np.zeros((2, 64, 32, 3), np.uint8)
    f[:, 48, 16] = 255  # the "emission point"
    out = place(f, (16, 48), (80, 20), (128, 72))
    ys, xs = np.nonzero(out[0].max(-1))
    assert (ys[0], xs[0]) == (20, 80)
    big = place(f, (16, 48), (64, 36), (128, 72), scale=2.0)
    ys, xs = np.nonzero(big[0].max(-1) > 0)
    assert abs(ys.mean() - 36) <= 1.5 and abs(xs.mean() - 64) <= 1.5


def test_place_off_canvas_is_black():
    f = np.full((1, 16, 16, 3), 200, np.uint8)
    assert not place(f, (8, 8), (-100, -100), (32, 32)).any()


def test_reference_performances_are_padded_and_ordered_by_length():
    lengths = {n: np.count_nonzero(np.abs(reference_performance(n)) > 0.01) for n in ("hit", "short", "medium", "long")}
    assert lengths["hit"] < lengths["short"] < lengths["medium"] < lengths["long"]


def fake_library(root: Path) -> Path:
    """Two lengths of a 'fireball' element: a warm blob at the library's emission point."""
    lw, lh = LIBRARY_SIZE
    entries = []
    for perf, n in (("hit", 49), ("medium", 97)):
        for seed in (1, 2):
            f = np.zeros((n, lh, lw, 3), np.uint8)
            cx, cy = int(LIBRARY_POINT[0] * lw), int(LIBRARY_POINT[1] * lh)
            f[4:n - 10, cy - 120: cy, cx - 60: cx + 60] = (255, 150, 30)
            name = f"fireball_{perf}_s{seed}.mp4"
            write_video(root / name, f, fps=24)
            entries.append({"effect": "fireball", "performance": perf, "seed": seed, "file": name, "length": n,
                            "anchor_frame": 24, "voice_end": None, "peak_level": 0.8, "render_s": 20.0})
    (root / "manifest.json").write_text(json.dumps({"point": list(LIBRARY_POINT), "size": list(LIBRARY_SIZE),
                                                    "fps": 24, "entries": entries}))
    return root


def test_choose_prefers_the_shortest_long_enough_element_and_cycles_seeds(tmp_path):
    lib = LibraryRenderer(fake_library(tmp_path))
    assert lib.choose("fireball", 60, 1)["file"] == "fireball_medium_s1.mp4"
    assert lib.choose("fireball", 40, 2)["file"] == "fireball_hit_s2.mp4"
    assert lib.choose("fireball", 400, 3)["file"] == "fireball_medium_s1.mp4"  # none long enough: the longest
    with pytest.raises(ValueError):
        lib.choose("steam_jet", 40, 1)


def test_an_instant_shot_end_to_end(tmp_path, make_video, make_wav):
    (tmp_path / "lib").mkdir()
    lib = LibraryRenderer(fake_library(tmp_path / "lib"))
    plate = make_video(tmp_path, n=72, fps=24, w=216, h=384, color_fn=lambda i: (50, 50, 50))
    x = np.random.default_rng(1).normal(0, 0.001, 3 * 48000).astype(np.float32)
    x[48000: 48000 + 4800] += (0.6 * np.sin(np.arange(4800) * 0.05)).astype(np.float32)
    voice = make_wav(tmp_path, x, 48000)
    r = make_shot(ShotSpec(plate=plate, voice=voice, effect="fireball", point=(0.3, 0.6)), lib, tmp_path / "shot")
    assert r.final.exists() and r.breakdown.exists()
    assert r.receipt["render"]["library"]["entry"].startswith("fireball_")
    assert "guide" not in r.receipt["render"]["library"]  # frames stay out of the receipt
    assert r.receipt["measured"]["hit_error_frames"] == 0
