"""Scene mode: the effect painted into the plate's own zones at its source, rendered from the plate frame,
re-timed onto the voice; delta transfer for moving plates."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from cuesync.audio import analyze
from cuesync.effects import load_effect
from cuesync.guide import FIRE_INNER, FIRE_OUTER
from cuesync.media import read_video, write_video
from cuesync.pipeline import ShotSpec, make_shot, retime_shot
from cuesync.runner import GraphBindError, bind_fluid_graph
from cuesync.scene import (MARGIN_PX, PREROLL, events_for, paint_scene_guide, paint_zones, plan_window, scene_prompt,
                           scene_size, unit, zone_palette)
from cuesync.transfer import change_energy, effect_region, is_still, transfer, visible_onset

SR = 48000


def voice(bursts, length_s=4.0, seed=0):
    """Tonal bursts [(start_s, dur_s, amplitude)] with sharp attacks over a quiet noise floor."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 10 ** (-62 / 20), int(length_s * SR)).astype(np.float32)
    for t0, dur, amp in bursts:
        n = int(dur * SR)
        t = np.arange(n) / SR
        x[int(t0 * SR): int(t0 * SR) + n] += (amp * np.sin(2 * np.pi * 150 * t) * np.minimum((dur - t) / 0.03, 1)).astype(np.float32)
    return x


def cue_of(bursts, **kw):
    return analyze(voice(bursts, **kw), SR, fps=24)


def two_tone_plate(w=576, h=1024):
    f = np.zeros((h, w, 3), np.uint8)
    f[: h // 2] = (40, 30, 25)  # dark wall
    f[h // 2:] = (170, 110, 60)  # wooden table
    f[h // 2 - 30: h // 2, w // 2 - 12: w // 2 + 12] = (220, 210, 190)  # a pale prop on the table
    return f


def fire_pixels(frame):
    return np.all(frame == FIRE_OUTER, -1) | np.all(frame == FIRE_INNER, -1)


def test_unit_directions():
    assert np.allclose(unit(0), (0, -1)) and np.allclose(unit(-90), (-1, 0)) and np.allclose(unit(90), (1, 0))


def test_scene_size_is_the_plate_rounded_up_to_64():
    assert scene_size(1080, 1920) == (1088, 1920) and scene_size(1920, 1080) == (1920, 1088)


def test_zones_are_few_flat_and_deterministic():
    f = two_tone_plate(128, 256)
    pal = zone_palette(f)
    z1, z2 = paint_zones(f, pal, (128, 256)), paint_zones(f, zone_palette(f), (128, 256))
    assert np.array_equal(z1, z2)
    assert len(np.unique(z1.reshape(-1, 3), axis=0)) <= 6
    assert abs(int(z1[10, 10].mean()) - 32) < 20 and abs(int(z1[240, 10].mean()) - 113) < 25  # wall stays dark, table stays wood


def test_first_hit_lands_on_a_keyframe_after_the_preroll():
    cue = cue_of([(1.5, 1.2, 0.4)])
    eff = load_effect("fire_breath")
    start, length = plan_window(cue, eff, None)
    assert cue.frame_of(cue.onsets_s[0]) - start == PREROLL and (PREROLL - 1) % 8 == 0
    assert length % 8 == 1 and 49 <= length <= 121
    # a moving plate can't start before its first frame
    s2, _ = plan_window(cue_of([(0.2, 1.0, 0.4)]), eff, 200)
    assert s2 == 0


def test_the_scene_is_held_until_the_hit_then_only_keyframes_are_painted():
    """Black latents before the hit let the model start early; after it, single keyframes leave it its motion."""
    cue = cue_of([(1.0, 1.2, 0.4)])
    eff = load_effect("fire_breath")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.45), two_tone_plate()[None], start=start, length=length)
    keys, at = set(sp.guide.keyframes), sp.events[0].at
    shapes_at = {b["e"]: b["shapes"] for b in sp.guide.blobs}
    for e in range(length):
        painted = sp.guide.frames[e].max() > 0
        if e in keys or e < at:
            assert painted, e
        elif painted:  # held between keyframes after the hit: only the scene once the effect is over
            k = max(k for k in keys if k < e)
            assert shapes_at[k] == [], e
    allheld = paint_scene_guide(cue, eff, (0.5, 0.45), two_tone_plate()[None], start=start, length=length, hold="all")
    assert all(allheld.guide.frames[e].max() > 0 for e in range(length))


def test_smoke_is_dim_in_a_dark_room():
    """Smoke is painted relative to the scene (light grey reads as white cotton in a dark room) and fades
    towards what is under it, never towards black."""
    cue = cue_of([(1.0, 0.8, 0.6)], length_s=5.0)
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    seen = []
    for level in (18, 200):
        plate = np.full((1024, 576, 3), level, np.uint8)
        sp = paint_scene_guide(cue, eff, (0.5, 0.6), plate[None], start=start, length=length)
        k, sh = next((b["e"], s) for b in sp.guide.blobs for s in b["shapes"]
                     if s.get("fluid") == "smoke" and not any(x.get("fluid") is None for x in b["shapes"]))
        seen.append((sp.guide.frames[k][int(sh["cy"]), int(sh["cx"])].astype(int), level))
    (dark, dl), (bright, bl) = seen
    assert dark.mean() < 0.6 * bright.mean()
    assert dark.mean() > dl and abs(bright.mean() - bl) < 40  # visible in the dark, never a black blot in the light


def test_fire_breath_comes_out_of_the_source_and_goes_its_way():
    cue = cue_of([(1.0, 1.5, 0.5)])
    eff = load_effect("fire_breath")
    start, length = plan_window(cue, eff, None)
    plate = two_tone_plate()[None]
    src = (0.7, 0.45)
    sp = paint_scene_guide(cue, eff, src, plate, start=start, length=length, direction=-90)
    at = sp.events[0].at
    assert not any(fire_pixels(sp.guide.frames[k]).any() for k in sp.guide.keyframes if k < at)
    during = [k for k in sp.guide.keyframes if at <= k < sp.events[0].end]
    assert during
    ys, xs = np.nonzero(fire_pixels(sp.guide.frames[during[-1]]))
    sx, sy = src[0] * 576, src[1] * 1024
    assert xs.min() < sx - 80 and xs.max() <= sx + 6  # it goes left, nothing behind the mouth
    assert xs.min() >= MARGIN_PX - 8  # and its tip stays inside the frame
    assert ys.mean() < sy + 4  # it curls up, never down


def test_the_jet_turns_with_the_direction():
    cue = cue_of([(1.0, 1.5, 0.5)])
    eff = load_effect("fire_breath")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.3, 0.6), two_tone_plate()[None], start=start, length=length, direction=90)
    k = [k for k in sp.guide.keyframes if sp.events[0].at <= k < sp.events[0].end][-1]
    _, xs = np.nonzero(fire_pixels(sp.guide.frames[k]))
    assert xs.min() >= 0.3 * 576 - 6 and xs.max() > 0.3 * 576 + 60


def test_a_flare_grows_out_of_the_resting_flame_and_settles_back():
    cue = cue_of([(1.0, 1.2, 0.5)], length_s=5.0)
    eff = load_effect("wick_flare")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.5), two_tone_plate()[None], start=start, length=length)
    keys = sp.guide.keyframes
    area = {k: int(fire_pixels(sp.guide.frames[k]).sum()) for k in keys}
    before = [area[k] for k in keys if k < sp.events[0].at]
    during = [area[k] for k in keys if sp.events[0].at <= k < sp.events[0].end]
    assert all(a > 0 for a in before), "the candle is lit before the noise"
    assert max(during) > 3 * max(before), "the flare is much bigger than the resting flame"
    assert 0 < area[keys[-1]] <= 1.5 * max(before), "it settles back to a small flame, not out"


def test_blowing_a_candle_out_leaves_it_out():
    cue = cue_of([(1.0, 0.15, 0.6)])
    eff = load_effect("blow_out")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.5), two_tone_plate()[None], start=start, length=length)
    keys = sp.guide.keyframes
    assert all(fire_pixels(sp.guide.frames[k]).any() for k in keys if k < sp.events[0].at)
    assert not any(fire_pixels(sp.guide.frames[k]).any() for k in keys if k >= sp.events[0].at)


def test_a_rhythm_is_one_event_per_hit_and_the_boom_is_biggest():
    cue = cue_of([(1.0, 0.12, 0.5), (1.6, 0.12, 0.5), (2.2, 0.8, 0.9)])
    eff = load_effect("fireball")
    start, _ = plan_window(cue, eff, None)
    evs = events_for(cue, eff, start)
    assert len(evs) == 3
    assert evs[0].at < evs[1].at < evs[2].at
    assert evs[2].size == max(e.size for e in evs) and evs[0].size < 1.0


def test_scene_prompt_fills_subject_and_direction():
    eff = load_effect("fire_breath")
    p = scene_prompt(eff, "the toy dinosaur", -90)
    assert p.startswith("ainvfxfluid, the toy dinosaur breathes") and "to the left" in p and "{" not in p
    assert "to the right" in scene_prompt(eff, None, 80) and "the object" in scene_prompt(eff, None, 80)


def test_a_hand_gets_no_mouth_and_no_carved_face():
    # a creature breathes out of its mouth, a jack-o'-lantern's carved face flares; a hand only shoots fire
    hand = scene_prompt(load_effect("fire_breath"), "the two pointed fingers", 90)
    assert "mouth" not in hand and "from the two pointed fingers" in hand and "to the right" in hand
    assert "carved" in scene_prompt(load_effect("fire_burst"), "the big jack-o'-lantern", 0)
    assert "carved" not in scene_prompt(load_effect("fire_burst"), "the open palm", 0)
    assert "explodes" not in scene_prompt(load_effect("fireball"), "the open palm", 0)
    sideways, up = scene_prompt(load_effect("steam_jet"), "the palm", 90), scene_prompt(load_effect("steam_jet"), "the mug", 0)
    assert "jet" in sideways and "to the right" in sideways and "wisps" in up


def test_a_resting_flame_only_where_one_burns(tmp_path):
    from cuesync.scene import lit_at, suited_effect

    dark = np.full((90, 160, 3), 30, np.uint8)
    candle = dark.copy()
    candle[40:50, 76:84] = (255, 235, 150)  # a flame's near-white warm core
    assert lit_at(candle, (0.5, 0.5)) and not lit_at(dark, (0.5, 0.5))
    burst = load_effect("fire_burst")
    assert suited_effect(burst, "the candle", None, candle, (0.5, 0.5)).scene.resting > 0
    assert suited_effect(burst, "the open palm", None, dark, (0.5, 0.5)).scene.resting == 0
    assert suited_effect(burst, "the big jack-o'-lantern", None, dark, (0.5, 0.5)).scene.resting == 0  # tapped on its face
    assert suited_effect(burst, "the glowing ghost lamp", None, candle, (0.5, 0.5)).scene.resting == 0  # glows, no flame


def test_every_effect_has_a_scene_section():
    for eid in ("fireball", "flame_plume", "steam_jet", "smoke_puff", "fire_breath", "wick_flare", "blow_out"):
        eff = load_effect(eid)
        assert eff.scene is not None and eff.scene.prompt.startswith("ainvfxfluid, ") and "{subject}" in eff.scene.prompt


def test_bind_switches_the_graph_to_image_to_video():
    g = json.loads(Path("workflows/fluid_element.api.json").read_text())
    b = bind_fluid_graph(g, prompt="ainvfxfluid, fire", seed=3, image=True)
    i2v = [v for v in b.graph.values() if isinstance(v, dict) and v.get("class_type") == "LTXVImgToVideoInplace"][0]
    assert i2v["inputs"]["bypass"] is False and i2v["inputs"]["strength"] == 1.0
    assert i2v["inputs"]["image"] == [b.load_image_id, 0] and b.graph[b.load_image_id]["class_type"] == "LoadImage"
    plain = bind_fluid_graph(g, prompt="ainvfxfluid, fire", seed=3)
    assert plain.load_image_id is None
    del g[[k for k, v in g.items() if v.get("class_type") == "LTXVImgToVideoInplace"][0]]
    with pytest.raises(GraphBindError):
        bind_fluid_graph(g, prompt="ainvfxfluid, fire", seed=3, image=True)


def test_still_detection():
    f = np.repeat(two_tone_plate(128, 256)[None], 10, 0)
    assert is_still(f)
    g = f.copy()
    g[5:] = np.roll(g[5:], 9, axis=2)
    assert not is_still(g)


def test_visible_onset_finds_the_rise():
    e = np.r_[np.full(20, 1.0), np.linspace(1, 30, 10), np.full(20, 30.0)]
    assert visible_onset(e, 15) in range(20, 24)
    assert visible_onset(np.full(40, 1.0), 15) is None


def test_transfer_null_and_light():
    """A render identical to the plate changes nothing; light the render throws on the table reaches the
    plate's table (and the plate keeps its own texture)."""
    rng = np.random.default_rng(1)
    plate1 = two_tone_plate(128, 256)
    plate1 = np.clip(plate1.astype(int) + rng.integers(-12, 12, plate1.shape), 0, 255).astype(np.uint8)
    n = 12
    plate = np.repeat(plate1[None], n, 0)
    plate[:, 0, 0] = np.arange(n)[:, None]  # a moving pixel so the plate isn't a still
    render = np.repeat(two_tone_plate(128, 256)[None], n, 0)
    region = np.zeros(render.shape[1:3], bool)
    region[100:180, 40:90] = True
    out, stats = transfer(plate, render, [None] * 2 + list(range(n - 2)), region, ref_frames=1, start=2)
    assert np.abs(out.astype(int) - plate).mean() < 1.5
    lit = render.copy()
    lit[6:, 128:] = np.clip(lit[6:, 128:].astype(int) * 1.8, 0, 255).astype(np.uint8)  # the table lights up
    out, _ = transfer(plate, lit, list(range(n)), region, ref_frames=3, start=0)
    assert out[9, 200:, :].mean() > plate[9, 200:, :].mean() * 1.4
    assert out[9, 20:60].mean() == pytest.approx(plate[9, 20:60].mean(), abs=3)  # the wall stays
    tex = out[9, 200:240, 10:50].astype(float)
    assert tex.std() > 4  # the plate's own grain is kept under the new light


class FakeSceneRenderer:
    """Renders 'into the scene': the start frame held, a bright disc at the source from ``early`` frames before
    the painted hit (models land early or late)."""

    name, scenes = "fake-scene", True

    def __init__(self, early=6):
        self.early = early
        self.calls = []

    def __call__(self, guide_path, prompt, seed, out_path, image=None):
        from PIL import Image

        self.calls.append({"prompt": prompt, "image": image})
        guide = read_video(guide_path).frames
        start = np.asarray(Image.open(image).convert("RGB"))
        n, h, w = guide.shape[:3]
        plan = json.loads((Path(guide_path).parent / "guide.json").read_text())
        at = plan["events"][0]["at"]
        frames = np.repeat(start[None], n, 0)
        yy, xx = np.mgrid[:h, :w]
        disc = (yy - h * 0.45) ** 2 + (xx - w * 0.5) ** 2 < (0.15 * w) ** 2
        for e in range(max(0, at - self.early), n):
            frames[e][disc] = (255, 200, 80)
        write_video(out_path, frames, fps=24, audio=np.zeros(int(n / 24 * 48000), np.float32))
        return 0.1


def test_scene_shot_end_to_end_lands_the_effect_on_the_voice(tmp_path, make_video, make_wav):
    plate = make_video(tmp_path, 48, 24, 128, 256, lambda i: (90, 70, 50))
    wav = make_wav(tmp_path, voice([(1.2, 1.0, 0.5)], length_s=3.5), SR)
    r = FakeSceneRenderer(early=6)
    spec = ShotSpec(plate=plate, voice=wav, effect="fire_breath", point=(0.5, 0.45), mode="scene",
                    subject="the toy", plate_size=(128, 256), matte=lambda f, p: None)
    res = make_shot(spec, r, tmp_path / "shot")
    rc = res.receipt
    assert rc["mode"] == "scene" and rc["plate"]["still"] is True
    assert r.calls[0]["image"] is not None and "the toy" in r.calls[0]["prompt"]
    assert rc["measured"]["model_early_frames"] in range(5, 8)
    assert rc["measured"]["residual_frames"] in (-1, 0, 1)
    for name in ("final.mp4", "composite.mp4", "breakdown.mp4", "guide.mp4", "render.mp4", "guide_preview.mp4"):
        assert (tmp_path / "shot" / name).exists()
    comp = read_video(tmp_path / "shot" / "composite.mp4").frames
    hit = int(round(1.2 * 24))
    assert comp.shape[0] >= hit + 20  # the still plate is held as long as the render needs
    lit = [int(comp[f, 110:120, 60:68].mean() > 150) for f in range(comp.shape[0])]
    assert abs(lit.index(1) - hit) <= 2
    # re-time: the whole effect moves, nothing is rendered again
    re = retime_shot(tmp_path / "shot", spec, tmp_path / "re", hit_shift=5)
    comp2 = read_video(tmp_path / "re" / "composite.mp4").frames
    lit2 = [int(comp2[f, 110:120, 60:68].mean() > 150) for f in range(comp2.shape[0])]
    assert abs((lit2.index(1) - lit.index(1)) - 5) <= 1
    assert re.receipt["retimed"]["hit_shift_frames"] == 5 and len(r.calls) == 1


def test_effect_region_covers_the_painted_shapes():
    cue = cue_of([(1.0, 1.2, 0.4)])
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), two_tone_plate()[None], start=start, length=length)
    reg = effect_region(sp.guide.blobs, sp.guide.size)
    for k in sp.guide.keyframes:
        assert reg[fire_pixels(sp.guide.frames[k])].all()
    assert 0.02 < reg.mean() < 0.95


def test_short_breaths_stay_apart():
    """'pf, pf, KA-BOOM' as breaths: every pf is followed by a fire-free keyframe before the next one."""
    cue = cue_of([(1.0, 0.12, 0.5), (1.6, 0.12, 0.5), (2.2, 0.8, 0.9)])
    eff = load_effect("fire_breath")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.6, 0.45), two_tone_plate()[None], start=start, length=length, direction=-90)
    evs = sp.events
    assert len(evs) == 3
    for a, b in zip(evs, evs[1:]):
        between = [k for k in sp.guide.keyframes if a.at < k < b.at]
        assert between and any(not fire_pixels(sp.guide.frames[k]).any() for k in between)


def test_a_blast_shakes_the_camera_and_never_shows_an_edge():
    from cuesync.scene_shot import shake

    rng = np.random.default_rng(3)
    frames = np.repeat(rng.integers(40, 220, (1, 64, 48, 3)).astype(np.uint8), 30, 0)
    out = shake(frames.copy(), [10], [1.0], 0.02)
    assert np.array_equal(out[:10], frames[:10])  # still before the hit
    assert np.abs(out[11].astype(int) - frames[11]).mean() > 3  # moving after it
    assert np.array_equal(out[29], frames[29])  # and settled again
    assert out[11:20].min() > 0  # no black border
    assert shake(frames, [10], [1.0], 0.0) is frames


def test_a_rhythm_lines_up_on_its_loudest_hit():
    from cuesync.transfer import steepest_rise

    e = np.r_[np.full(15, 1.0), np.full(15, 6.0), np.full(10, 9.0), np.linspace(9, 40, 4), np.full(20, 40.0)]
    assert steepest_rise(e, 40) in range(40, 44)  # the boom, not the first small pop at 15
    assert steepest_rise(np.full(60, 3.0), 30) is None


def test_the_flash_is_warm_and_short():
    from cuesync.scene_shot import PEAK_AFTER, impact

    frames = np.full((20, 8, 8, 3), 100, np.uint8)
    out = impact(frames.copy(), [5], [1.0], 0.0, 0.5)
    f = 5 + PEAK_AFTER
    assert out[f, 0, 0, 0] > out[f, 0, 0, 2] > 100  # brighter, warmer
    assert out[f + 1].mean() < out[f].mean() and out[f + 2].mean() == 100  # and gone two frames later
    assert np.array_equal(out[: f], frames[: f])


# ---------------------------------------------------------------- review 5 fixes

@pytest.mark.parametrize("phase", range(8))
def test_every_short_hit_gets_fire_on_a_keyframe(phase):
    """A quick 'pf' can fall anywhere between two keyframes; it must still be painted (8 phases of the grid)."""
    from cuesync.scene import is_fire

    t2 = 1.6 + phase / 24
    cue = cue_of([(1.0, 0.12, 0.5), (t2, 0.12, 0.5), (2.6, 0.8, 0.9)], length_s=5.0)
    eff = load_effect("fire_breath")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.6, 0.45), two_tone_plate()[None], start=start, length=length, direction=-90)
    assert len(sp.events) == 3
    for ev in sp.events:
        k = next(k for k in sp.guide.keyframes if k >= ev.at)
        shapes = next(b["shapes"] for b in sp.guide.blobs if b["e"] == k)
        assert any(is_fire(s) for s in shapes), (phase, ev.at, k)


def test_a_quiet_pf_is_a_small_pop_and_says_so():
    from cuesync.scene import SMALL, single_size

    assert single_size(2.9) < 0.5 < single_size(8.0) < single_size(16.5) == 1.0
    eff = load_effect("fireball")
    small = scene_prompt(eff, "the jar", 0, hits=1, size=single_size(2.9))
    assert "small pop" in small and single_size(2.9) < SMALL
    assert "explodes" in scene_prompt(eff, "the jar", 0, hits=1, size=1.0)
    pf, boom = cue_of([(1.0, 0.1, 0.5)]), cue_of([(1.0, 0.9, 0.9)])
    sizes = [events_for(c, eff, 0)[0].size for c in (pf, boom)]
    assert sizes[0] < 0.6 * sizes[1]


def test_blown_out_stays_out_through_more_hits():
    from cuesync.scene import is_fire

    cue = cue_of([(1.0, 0.1, 0.6), (1.7, 0.1, 0.6), (2.4, 0.1, 0.6)])
    eff = load_effect("blow_out")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.5), two_tone_plate()[None], start=start, length=length)
    first = sp.events[0].at
    for b in sp.guide.blobs:
        assert any(is_fire(s) for s in b["shapes"]) == (b["e"] < first), b["e"]


def test_thin_smoke_fades_into_the_scene_and_then_is_gone():
    from cuesync.scene import MIN_DENSITY

    cue = cue_of([(1.0, 0.8, 0.9)], length_s=5.0)
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1024, 576, 3), 170, np.uint8)  # a bright kitchen
    sp = paint_scene_guide(cue, eff, (0.5, 0.7), plate[None], start=start, length=length)
    dens = [max([s["density"] for s in b["shapes"] if s.get("fluid")], default=None) for b in sp.guide.blobs]
    seen = [d for d in dens if d is not None]
    assert seen and min(seen) >= MIN_DENSITY and seen[-1] < seen[0]
    assert dens[-1] is None  # the last keyframe says "nothing here"
    for b in sp.guide.blobs:  # a faded cloud is near the scene's grey, never a dark blot
        for s in b["shapes"]:
            if s.get("fluid") == "smoke" and s["density"] < 0.4 and 0 <= s["cy"] < 1024:
                assert abs(int(sp.guide.frames[b["e"]][int(s["cy"]), int(s["cx"])].mean()) - 170) < 30


def test_hits_beyond_the_render_are_reported_not_dropped_silently():
    cue = cue_of([(1.0, 0.12, 0.6), (1.6, 0.12, 0.6), (7.5, 0.5, 0.9)], length_s=9.0)
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), two_tone_plate()[None], start=start, length=length)
    assert len(sp.events) == 2 and len(sp.dropped) == 1 and sp.dropped[0] > start + length - 9


def test_a_drawing_is_filled_from_where_it_starts_and_the_fire_recedes():
    from cuesync.scene import Strokes

    st = Strokes.parse([{"points": [[0.3, 0.5], [0.5, 0.45], [0.7, 0.5]], "width": 0.06}])
    assert st.source == (0.3, 0.5) and 75 < st.direction < 105  # drawn left to right: it goes right
    cue = cue_of([(1.0, 1.2, 0.6)], length_s=5.0)
    eff = load_effect("flame_plume")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.9, 0.9), two_tone_plate()[None], start=start, length=length, strokes=st)
    assert sp.guide.point == (0.3, 0.5)  # the drawing's source wins over the tapped point
    fire_x = {}
    for b in sp.guide.blobs:
        f = sp.guide.frames[b["e"]]
        ys, xs = np.nonzero(np.all(f == FIRE_OUTER, -1) | np.all(f == FIRE_INNER, -1))
        fire_x[b["e"]] = xs.max() if xs.size else None
    at, end = sp.events[0].at, sp.events[0].end
    during = [fire_x[k] for k in sp.guide.keyframes if at <= k < end and fire_x[k] is not None]
    assert during and max(during) > 0.6 * 576  # it fills the drawing to the right
    assert all(x is None or x <= 0.75 * 576 for x in fire_x.values())  # and never beyond it
    assert fire_x[sp.guide.keyframes[-1]] is None  # and it is gone once the voice is


def test_a_hold_is_a_cue():
    from cuesync.cues import cue_from_holds

    cue = cue_from_holds([(1.0, 2.0), (3.0, 3.25)], 5.0)
    assert cue.intervals() == [(24, 48), (72, 78)]
    assert cue.frame_of(cue.hero_onset_s) == 24 and len(cue.hits()) == 2
    assert cue.envelope[30] == pytest.approx(0.85) and cue.envelope[60] == 0


def test_the_clips_own_sound_is_a_cue(tmp_path):
    from cuesync.cues import CueError, load_cue

    a = np.zeros(int(3 * SR), np.float32)
    a[int(1.5 * SR): int(1.5 * SR) + 2400] = 0.8 * np.sin(np.arange(2400) / 3.0)  # a snap at 1.5 s
    snap = tmp_path / "snap.mp4"
    write_video(snap, np.zeros((72, 64, 48, 3), np.uint8), fps=24, audio=a)
    c = load_cue("clip", plate=snap)
    assert c.source == "clip" and abs(c.cue.frame_of(c.cue.onsets_s[0]) - 36) <= 2
    silent = tmp_path / "silent.mp4"
    write_video(silent, np.zeros((24, 64, 48, 3), np.uint8), fps=24, audio=np.zeros(SR, np.float32))
    with pytest.raises(CueError):
        load_cue("clip", plate=silent)


def test_a_drifting_render_is_rejected_and_a_lit_table_keeps_its_texture():
    from cuesync.transfer import MAX_RESIDUAL

    rng = np.random.default_rng(4)
    base = np.clip(two_tone_plate(128, 256).astype(int) + rng.integers(-20, 20, (256, 128, 3)), 0, 255).astype(np.uint8)
    n = 10
    plate = np.repeat(base[None], n, 0)
    plate[:, :40, :40] = np.clip(plate[:, :40, :40].astype(int) + (np.arange(n) * 3)[:, None, None, None], 0, 255)  # it moves
    region = np.zeros((256, 128), bool)
    region[150:250, 30:100] = True
    # the render re-textures the whole wall (drift): rejected
    drift = plate.copy()
    drift[3:, :120] = np.clip(drift[3:, :120].astype(int) + rng.integers(-40, 40, drift[3:, :120].shape), 0, 255)
    _, st = transfer(plate, drift, list(range(n)), region, ref_frames=2, start=0)
    assert not st.accepted and st.residual > MAX_RESIDUAL
    # the render only lights the table: accepted, and the plate's own grain survives under the light
    lit = plate.copy()
    lit[3:, 128:] = np.clip(lit[3:, 128:].astype(int) * 1.4, 0, 255).astype(np.uint8)
    out, st = transfer(plate, lit, list(range(n)), region, ref_frames=2, start=0)
    assert st.accepted
    assert out[8, 200:240, 10:50].astype(float).std() > 6
    _, full = transfer(plate, lit, list(range(n)), np.ones((256, 128), bool), ref_frames=2, start=0)
    assert not full.accepted and full.residual is None


def test_a_small_moving_thing_is_not_a_still():
    f = np.repeat(two_tone_plate(128, 256)[None], 30, 0)
    assert is_still(f)
    g = f.copy()
    g[2:5, 100:110, 10:20] = 255  # a small thing flickers for three frames
    assert not is_still(g)


def test_the_flash_lands_on_the_burst_not_on_the_models_error():
    from cuesync.scene_shot import PEAK_AFTER, time_scene

    cue = cue_of([(1.5, 0.6, 0.9)], length_s=5.0)
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1, 256, 128, 3), 60, np.uint8)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), plate, start=start, length=length, size=(128, 256))
    at = sp.events[0].at
    render = np.full((length, 256, 128, 3), 60, np.uint8)
    yy, xx = np.mgrid[:256, :128]
    disc = (yy - 150) ** 2 + (xx - 64) ** 2 < 30 ** 2
    for e in range(at - 6, length):  # the model is 6 frames early
        render[e][disc] = (250, 180, 60)
    st = time_scene(plate, True, render, sp, eff, 240)
    hit = start + at - eff.lead_frames
    assert st.hits == [hit]
    corner = [int(st.frames[f, 10:40, 10:40].mean()) for f in range(len(st.frames))]
    assert corner.index(max(corner)) == hit + PEAK_AFTER  # the flash, seen on a flat region away from the burst
    lit = [int(st.frames[f][disc].mean() > 150) for f in range(len(st.frames))]
    assert abs(lit.index(1) - hit) <= 1


def test_the_stop_is_measured_not_assumed():
    from cuesync.scene_shot import measure_stop

    frames = np.full((60, 64, 48, 3), 30, np.uint8)
    frames[10:40, 20:40, 10:30] = (250, 160, 40)  # fire from 10 until 40
    region = np.zeros((64, 48), bool)
    region[16:44, 6:34] = True
    assert measure_stop(frames, region, voice_end=32, fire=True, first=10) == 40
    assert measure_stop(frames, region, voice_end=None, fire=True, first=10) is None


def test_a_drawn_effect_with_no_voice_and_the_performers_words(tmp_path, make_video):
    plate = make_video(tmp_path, 48, 24, 128, 256, lambda i: (90, 70, 50))
    r = FakeSceneRenderer(early=0)
    spec = ShotSpec(plate=plate, voice=None, effect="smoke_puff", point=(0.5, 0.5), mode="scene", cue="hold",
                    holds=[(1.0, 2.0)], strokes=[{"points": [[0.2, 0.6], [0.5, 0.4]], "width": 0.1}],
                    describe="thick smoke pours out of the lamp", plate_size=(128, 256), matte=lambda f, p: None)
    res = make_shot(spec, r, tmp_path / "drawn")
    rc = res.receipt
    assert rc["inputs"]["cue"] == "hold" and rc["inputs"]["drawn"] is True and rc["inputs"]["voice_sha"] is None
    assert r.calls[0]["prompt"].startswith("ainvfxfluid, thick smoke pours out of the lamp,")
    plan = json.loads((tmp_path / "drawn" / "guide.json").read_text())
    assert plan["strokes"] and plan["point"] == [0.2, 0.6]


# ---------------------------------------------------------------- the pivot review's confirmed findings


def _fire_share(frame, mask):
    f = np.all(frame == FIRE_OUTER, -1) | np.all(frame == FIRE_INNER, -1)
    return float((f & mask).sum()) / max(1, int(mask.sum()))


def test_a_drawn_pf_does_not_grow_after_the_sound_stops():
    from cuesync.scene import Strokes

    st = Strokes.parse([{"points": [[0.2, 0.6], [0.8, 0.6]], "width": 0.08}])
    cue = cue_of([(1.0, 0.12, 0.5), (1.6, 0.12, 0.5), (2.2, 0.8, 0.9)], length_s=5.0)
    eff = load_effect("flame_plume")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.2, 0.6), two_tone_plate()[None], start=start, length=length, strokes=st)
    mask = st.mask(576, 1024)
    shares = {b["e"]: _fire_share(sp.guide.frames[b["e"]], mask) for b in sp.guide.blobs}
    first = sp.events[0]
    on = [shares[k] for k in sp.guide.keyframes if first.at <= k < first.end]
    off = [shares[k] for k in sp.guide.keyframes if first.end <= k < sp.events[1].at]
    assert on and max(off, default=0) <= max(on) + 0.02  # it recedes (or stays), it never swells after the "pf"


def test_a_drawn_ring_keeps_its_middle():
    from cuesync.scene import Strokes, draw

    ring = [[0.5 + 0.2 * np.cos(t), 0.5 + 0.12 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 40)]
    st = Strokes.parse([{"points": ring, "width": 0.04}])
    cue = cue_of([(1.0, 1.5, 0.9)], length_s=5.0)
    eff = load_effect("flame_plume")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.5), two_tone_plate()[None], start=start, length=length, strokes=st)
    k = max(sp.guide.keyframes, key=lambda e: _fire_share(sp.guide.frames[e], st.mask(576, 1024)))
    centre = sp.guide.frames[k][490:534, 268:308]
    assert not (np.all(centre == FIRE_OUTER, -1) | np.all(centre == FIRE_INNER, -1)).any()
    reg = effect_region(sp.guide.blobs, sp.guide.size, grow=0.0)
    assert not reg[512, 288]  # the transfer won't replace what the ring surrounds


def test_a_drawn_candle_keeps_its_flame_and_a_drawn_blow_out_still_blows_it_out():
    from cuesync.scene import Strokes, is_fire

    st = Strokes.parse([{"points": [[0.5, 0.5], [0.5, 0.3]], "width": 0.08}])
    cue = cue_of([(1.0, 1.2, 0.6)], length_s=5.0)
    for eid in ("wick_flare", "blow_out"):
        eff = load_effect(eid)
        start, length = plan_window(cue, eff, None)
        sp = paint_scene_guide(cue, eff, (0.5, 0.5), two_tone_plate()[None], start=start, length=length, strokes=st)
        lit = {b["e"]: any(is_fire(s) for s in b["shapes"]) for b in sp.guide.blobs}
        before = [lit[k] for k in sp.guide.keyframes if k < sp.events[0].at]
        assert all(before), eid  # the candle is lit before the cue either way
        if eid == "wick_flare":
            assert lit[sp.guide.keyframes[-1]]  # and still lit after the flare
        else:
            assert not any(lit[k] for k in sp.guide.keyframes if k >= sp.events[0].at)  # and out after "pf"


def test_quiet_gaps_hold_only_when_truly_quiet():
    cue = cue_of([(1.0, 0.12, 0.5), (1.9, 0.12, 0.5), (2.8, 0.6, 0.9)], length_s=5.0)
    eff = load_effect("steam_jet")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), two_tone_plate()[None], start=start, length=length)
    keys = set(sp.guide.keyframes)
    shapes_at = {b["e"]: b["shapes"] for b in sp.guide.blobs}
    for e in range(sp.events[0].at, length):
        if e not in keys and sp.guide.frames[e].max() > 0:  # a held frame after the first hit
            k = max(x for x in keys if x < e)
            assert all(s.get("rest") for s in shapes_at[k]), (e, k)  # live steam is never frozen


def test_close_hits_merge_and_omitted_hits_are_reported():
    from cuesync.scene import omitted_hits

    cue = cue_of([(1.0, 0.1, 0.6), (1.27, 0.1, 0.6), (2.2, 0.8, 0.9)], length_s=5.0)  # 6-7 frames apart
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), two_tone_plate()[None], start=start, length=length)
    keys = sp.guide.keyframes
    firsts = [next(k for k in keys if k >= ev.at) for ev in sp.events]
    assert len(firsts) == len(set(firsts))  # no two events share a keyframe
    many = cue_of([(0.5 + 0.45 * i, 0.1, 0.6) for i in range(6)], length_s=5.0)
    assert len(omitted_hits(many)) == len(many.hits(max_hits=99)) - len(many.hits())


def test_a_late_hit_is_still_painted_and_a_jet_at_the_edge_is_turned_up():
    from cuesync.scene import is_fire

    cue = cue_of([(1.0, 1.2, 0.8)], length_s=5.0)
    eff = load_effect("fire_breath")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.04, 0.6), two_tone_plate()[None], start=start, length=length, direction=-90)
    assert any(is_fire(s) for b in sp.guide.blobs for s in b["shapes"])
    assert any("turned upwards" in n for n in sp.notes)


def test_bad_strokes_are_refused_not_crashed_on():
    from cuesync.scene import Strokes

    for bad in ("x", [1, 2], [{"points": "no"}], [{"points": [[0.1, float("nan")]]}], [{"points": [[0.1]]}]):
        with pytest.raises(ValueError):
            Strokes.parse(bad)
    assert Strokes.parse([]) is None


def test_a_perfect_render_does_not_ghost_what_moves_in_the_plate():
    rng = np.random.default_rng(7)
    base = np.clip(two_tone_plate(128, 256).astype(int) + rng.integers(-10, 10, (256, 128, 3)), 0, 255).astype(np.uint8)
    n = 12
    plate = np.repeat(base[None], n, 0)
    for i in range(n):  # something moves across the wall, far from the effect
        plate[i, 30:50, 5 + 6 * i: 25 + 6 * i] = 240
    region = np.zeros((256, 128), bool)
    region[170:240, 40:90] = True
    render = np.repeat(plate[:1], n, 0)  # the render: the start frame, the effect inside its region only
    render[4:, 180:230, 50:80] = (250, 170, 60)
    out, st = transfer(plate, render, list(range(n)), region, ref_frames=3, start=0)
    assert np.abs(out[10, 20:60].astype(int) - plate[10, 20:60]).mean() < 4  # the moving thing is the plate's
    assert out[10, 200, 65, 0] > 200  # and the effect is there


def test_the_onset_ignores_a_drifting_camera():
    from cuesync.scene_shot import registered

    rng = np.random.default_rng(3)
    tex = rng.integers(40, 200, (256, 128, 3)).astype(np.uint8)
    import cv2
    tex = cv2.GaussianBlur(tex, (0, 0), 1.5)
    frames = []
    for i in range(40):  # a slow drift right (inside the warp bound), a subtle glow only from frame 30
        M = np.float32([[1, 0, 0.15 * i], [0, 1, 0]])
        f = cv2.warpAffine(tex, M, (128, 256), borderMode=cv2.BORDER_REFLECT)
        if i >= 30:
            f[150:200, 40:90] = (0.6 * f[150:200, 40:90] + 0.4 * np.array([250, 170, 60])).astype(np.uint8)
        frames.append(f)
    render = np.stack(frames)
    region = np.zeros((256, 128), bool)
    region[140:210, 30:100] = True
    raw = visible_onset(change_energy(render, region, 10), 10)
    steady = visible_onset(change_energy(registered(render, region, 10), region, 10), 10)
    assert steady in range(29, 32) and raw is not None and raw < steady  # the drift alone fools the raw measure


def test_strokes_outside_the_crop_are_cut_not_squashed(tmp_path, make_video):
    from cuesync.scene_shot import _plate_strokes

    plate = make_video(tmp_path, 4, 24, 256, 256, lambda i: (80, 80, 80))  # square: cropped to 9:16 around x
    spec = ShotSpec(plate=plate, voice=None, effect="flame_plume", point=(0.5, 0.5), mode="scene", cue="hold",
                    strokes=[{"points": [[0.05, 0.5], [0.5, 0.5], [0.95, 0.5]], "width": 0.05}], plate_size=(144, 256))
    st = _plate_strokes(spec)
    xs = [p[0] for s in st.strokes for p in s["points"]]
    assert xs and all(0.0 < x < 1.0 for x in xs)  # the parts outside the crop are gone, none pinned to an edge


def test_a_clip_cue_after_the_end_of_the_shot_is_refused(tmp_path):
    from cuesync.cues import CueError, load_cue

    a = np.zeros(int(6 * SR), np.float32)
    a[int(5.0 * SR): int(5.0 * SR) + 2400] = 0.8 * np.sin(np.arange(2400) / 3.0)  # a snap at 5 s
    clip = tmp_path / "late.mp4"
    write_video(clip, np.zeros((144, 64, 48, 3), np.uint8), fps=24, audio=a)
    with pytest.raises(CueError):
        load_cue("clip", plate=clip, duration_s=3.0)
    with pytest.raises(CueError):
        load_cue("hold", plate=clip, holds=[(4.0, 4.5)], duration_s=3.0)


def test_the_sample_noises_read_as_what_they_are_at_any_offset():
    from cuesync.gestures import gesture
    from cuesync.media import read_audio

    want = {"pf": "hit", "ka-boom": "hit", "fwoooom": "sustained", "pfffff": "breath", "hissss": "hiss",
            "pf-pf-ka-boom": "rhythm"}
    for name, g in want.items():
        x = read_audio(f"samples/voices/{name}.wav", SR)
        for off in (0.0, 1.3):
            assert gesture(analyze(x, SR, offset_s=off, fps=24), x) == g, (name, off)


def test_a_rhythm_lines_up_the_hits_it_can_see_and_never_guesses_a_merged_one():
    from cuesync.transfer import hit_onsets

    flame = np.zeros(70)
    flame[10:20] = np.linspace(0.005, 0.05, 10)  # "pf": shows at 10, dies back
    flame[26:40] = np.linspace(0.01, 0.08, 14)  # "pf": shows at 26 (two frames early)
    flame[40:70] = np.linspace(0.08, 0.3, 30)  # the BOOM grows straight out of it: merged, no onset of its own
    assert hit_onsets(flame, [12, 28, 50], ref_frames=6) == [10, 26, None]


def test_the_onset_is_the_first_frame_you_see_it_not_the_middle_of_its_growth():
    from cuesync.transfer import first_visible

    share = np.concatenate([np.zeros(12), np.linspace(0.004, 0.4, 12), np.full(10, 0.4)])
    assert first_visible(share, ref_frames=8) == 13  # 6 % of the way up, not 20 %


def test_a_candle_blown_out_early_by_the_model_is_still_found():
    from cuesync.scene_shot import EARLIEST, flame_out

    frames = np.full((30, 64, 64, 3), 200, np.uint8)
    frames[:11, 20:40, 28:36] = (255, 190, 90)  # lit until frame 11, out from then on
    region = np.zeros((64, 64), bool)
    region[10:50, 20:44] = True
    assert flame_out(frames, region, ref_frames=max(4, 17 - EARLIEST)) == 11


def test_a_white_hot_flame_over_wax_it_lights_goes_out_when_it_goes_out():
    # a real candle: its wax stays warm-lit in the render, and the flame flares white as it is blown out, so
    # flame colour alone never sees it go out; the flame's own brightness does
    from cuesync.scene_shot import flame_out

    frames = np.full((30, 64, 64, 3), 40, np.uint8)
    frames[:, 40:, 16:48] = (230, 190, 120)  # the wax, warm all along
    for f in range(12):
        frames[f, 20:40, 28:36] = (255, 250, 235) if f >= 8 else (255, 230, 170)  # the flame; a white flare at the end
    region = np.zeros((64, 64), bool)
    region[10:, 10:54] = True
    assert flame_out(frames, region, ref_frames=8) == 12


def test_a_scene_that_re_shapes_at_a_block_boundary_is_not_the_effect_starting():
    from cuesync.transfer import changed_share, first_visible, onset_past_steps

    rng = np.random.default_rng(3)
    base = rng.integers(60, 140, (64, 64, 3)).astype(np.uint8)
    frames = np.repeat(base[None], 30, axis=0)
    frames[9:, :, ::4] += 60  # frame 9: the model re-shapes the whole scene, inside the effect and out
    region = np.zeros((64, 64), bool)
    region[20:44, 20:44] = True
    for f, rows in ((13, 8), (14, 16), (15, 24)):  # the effect grows in from frame 13
        frames[f:, 20:20 + rows, 20:44] = 250
    share = changed_share(frames, region, 8)
    assert first_visible(share, 8) == 9  # taken for the effect without the check
    assert onset_past_steps(frames, region, 8, share) == 13

    lit = np.repeat(base[None], 30, axis=0)
    lit[12:, 20:44, 20:44] = 250  # a fire from frame 12 ...
    lit[12:, :8] += 60  # ... that lights part of the room as well: not a step
    assert onset_past_steps(lit, region, 8, changed_share(lit, region, 8)) == 12


def _fire_px(frame):
    return int((np.all(frame == FIRE_OUTER, -1) | np.all(frame == FIRE_INNER, -1)).sum())


def test_a_flame_goes_within_a_few_frames_of_the_voice():
    from cuesync.scene import DIE_DOWN, Strokes

    cue = cue_of([(1.0, 1.2, 0.8)], length_s=5.0)
    drawn = Strokes.parse([{"points": [[0.5, 0.6], [0.5, 0.35]], "width": 0.08}])
    for eid, point, strokes in (("flame_plume", (0.5, 0.6), None), ("fire_breath", (0.6, 0.6), None),
                                ("flame_plume", (0.5, 0.6), drawn)):
        eff = load_effect(eid)
        start, length = plan_window(cue, eff, None)
        sp = paint_scene_guide(cue, eff, point, two_tone_plate()[None], start=start, length=length, strokes=strokes)
        end = sp.events[0].end
        later = [k for k in sp.guide.keyframes if k >= end + DIE_DOWN]
        assert later and all(_fire_px(sp.guide.frames[k]) == 0 for k in later), (eid, strokes is not None)


def test_a_burst_is_a_ragged_blast_not_a_disc():
    cue = cue_of([(1.0, 0.4, 0.9)], length_s=4.0)
    eff = load_effect("fire_burst")
    start, length = plan_window(cue, eff, None)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), two_tone_plate()[None], start=start, length=length)
    k = sp.events[0].at + 8
    k = min(sp.guide.keyframes, key=lambda x: abs(x - k))
    m = (np.all(sp.guide.frames[k] == FIRE_OUTER, -1) | np.all(sp.guide.frames[k] == FIRE_INNER, -1)).astype(np.uint8)
    import cv2
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(big)
    assert cv2.contourArea(big) / cv2.contourArea(hull) < 0.95  # tongues and notches, not a filled ellipse
    mo = cv2.moments(big)
    cy, ys = mo["m01"] / mo["m00"], big[:, 0, 1]
    assert cy - ys.min() > ys.max() - cy  # it licks upward more than it hangs down


def test_a_take_whose_noise_lands_off_the_clip_is_refused(tmp_path, make_wav):
    from cuesync.cues import CueError, load_cue

    x = np.zeros(3 * SR, np.float32)
    x[SR: SR + 4800] = 0.6 * np.sin(np.arange(4800) * 0.05)  # "pf." 1 s into the recording
    voice = make_wav(tmp_path, x, SR)
    assert load_cue("voice", plate=voice, voice=voice, offset_s=-0.3, duration_s=4.0).cue.hero_onset_s == pytest.approx(0.7, abs=0.05)
    with pytest.raises(CueError, match="before the clip starts"):
        load_cue("voice", plate=voice, voice=voice, offset_s=-11.6, duration_s=4.0)  # a misread clock
    with pytest.raises(CueError, match="after the end"):
        load_cue("voice", plate=voice, voice=voice, offset_s=5.0, duration_s=4.0)


def test_a_hit_the_model_merged_gets_no_flash_of_its_own():
    from cuesync.scene_shot import PEAK_AFTER, time_scene

    cue = cue_of([(1.0, 0.2, 0.9), (1.8, 0.2, 0.9)], length_s=5.0)
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1, 256, 128, 3), 60, np.uint8)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), plate, start=start, length=length, size=(128, 256))
    assert len(sp.events) == 2
    render = np.full((length, 256, 128, 3), 60, np.uint8)
    yy, xx = np.mgrid[:256, :128]
    disc = (yy - 150) ** 2 + (xx - 64) ** 2 < 30 ** 2
    for e in range(sp.events[0].at - 2, length):  # one fire from the first hit on: the second never shows apart
        render[e][disc] = (250, 180, 60)
    st = time_scene(plate, True, render, sp, eff, 240)
    assert st.merged == [1] and len(st.hits) == 2
    corner = [int(st.frames[f, 10:40, 10:40].mean()) for f in range(len(st.frames))]
    assert corner[st.hits[0] + PEAK_AFTER] > corner[0] + 5  # the first hit flashes
    assert abs(corner[st.hits[1] + PEAK_AFTER] - corner[0]) <= 2  # the merged one doesn't


def test_a_rhythm_pauses_between_its_hits_on_the_guide():
    from cuesync.scene import is_fire, is_rest

    cue = cue_of([(1.0, 0.12, 0.6), (1.67, 0.12, 0.6), (2.5, 0.6, 0.95)], length_s=5.0)
    eff = load_effect("fire_burst")
    start, length = plan_window(cue, eff, None)
    plate = two_tone_plate()[None]
    shapes = lambda sp, e: next(b["shapes"] for b in sp.guide.blobs if b["e"] == e)  # noqa: E731
    on = paint_scene_guide(cue, eff, (0.5, 0.6), plate, start=start, length=length, gaps="none")
    a, b = on.events[0].at, on.events[1].at
    gap = max(k for k in on.guide.keyframes if a < k < b)
    assert any(is_fire(s) and not is_rest(s) for s in shapes(on, gap))  # without gaps the first burst burns on
    free = paint_scene_guide(cue, eff, (0.5, 0.6), plate, start=start, length=length, gaps="free")
    assert not any(is_fire(s) and not is_rest(s) for s in shapes(free, gap))  # a pause the model can see
    held = paint_scene_guide(cue, eff, (0.5, 0.6), plate, start=start, length=length, gaps="hold")
    assert all(is_rest(s) for s in shapes(held, gap))
    assert all(np.array_equal(held.guide.frames[f], held.guide.frames[gap]) for f in range(gap + 1, b))  # held to the hit
    assert held.guide.frames[b].max() > 0 or b not in held.guide.keyframes


def test_light_a_render_takes_away_darkens_the_plate_and_its_texture_survives():
    from cuesync.transfer import gam, light_gain, lin

    plate = lin(np.full((4, 4, 3), (150, 140, 130), np.uint8))  # a warm-white surface in the real clip
    ref = lin(np.full((4, 4, 3), (220, 200, 140), np.uint8))  # the render lit it more brightly, and warmer
    after = lin(np.full((4, 4, 3), (60, 60, 120), np.uint8))  # then the candle went out in the render
    out = gam(plate * light_gain(after, ref, 1.0)[0]).astype(int)[0, 0]
    old = gam(np.maximum(plate + after - ref, 0.0)).astype(int)[0, 0]
    assert old[0] == 0  # the old additive transfer: red subtracted below black
    assert out.max() < 150 and out.min() > 20  # darker, and no channel crushed to black
    glow = gam(plate * light_gain(lin(np.full((4, 4, 3), (255, 200, 120), np.uint8)), ref, 1.0)[0]).astype(int)[0, 0]
    assert glow[0] > 150  # light the render adds (a fire's glow) still brightens it
    rng = np.random.default_rng(2)  # under a room going dark and warm (a fire), the clip's texture stays
    tex = lin(np.clip(rng.normal(170, 25, (64, 64, 3)), 0, 255).astype(np.uint8))
    dim = gam(tex * light_gain(np.full_like(tex, 0.06) * np.array([1.4, 0.9, 0.8], np.float32), np.full_like(tex, 0.4), 4.0)[0])
    assert dim[..., 0].mean() > dim[..., 2].mean() + 5 and dim[..., 1].std() > 3


def test_a_smooth_fireball_covers_what_is_behind_it_instead_of_ghosting_it():
    # a flame against a soft, blurred background has almost no structure, and the render's background is not the clip's
    n = 12
    yy, xx = np.mgrid[:256, :128]

    def soft_spot(cy, cx):  # something dark and out of focus: no edges for a structure detector to catch
        return 214 - 170 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 14.0 ** 2))

    plate = np.repeat(np.clip(soft_spot(135, 62), 0, 255).astype(np.uint8)[None, ..., None], n, 0).repeat(3, -1)
    plate[::2, 5:15, 5:15] = 120  # the clip moves a little elsewhere, so the transfer path runs
    render = np.repeat(np.clip(soft_spot(143, 70), 0, 255).astype(np.uint8)[None, ..., None], n, 0).repeat(3, -1)
    region = np.zeros((256, 128), bool)
    region[90:200, 30:100] = True
    alpha = np.clip((44 - np.hypot(yy - 140, xx - 64)) / 8.0, 0.0, 1.0)[..., None]  # a soft-edged, smooth flame
    flame = (render[0] * (1 - alpha) + np.array([250, 160, 60]) * alpha).astype(np.uint8)
    render[4:] = flame
    out, st = transfer(plate, render, list(range(n)), region, ref_frames=3, start=0)
    patch = out[10, 130:140, 57:67].astype(int)  # where the clip's dark spot was (the render had it elsewhere)
    assert np.abs(patch - np.array([250, 160, 60])).mean() < 6  # the flame, not a ghost of what is behind it (14.8 without)


def test_each_hit_of_a_rhythm_goes_on_its_own_cue_within_a_speed_limit():
    from cuesync.scene_shot import from_cue, on_cue, warp_anchors

    # the model put three hits 4 early, 1 early and 3 late: no single shift lands all three
    anchors = warp_anchors([12, 30, 54], [16, 31, 51])
    assert anchors == [(12, 16), (30, 31), (54, 51)]
    assert [on_cue(o, anchors) for o, _ in anchors] == [16, 31, 51]
    assert from_cue(on_cue(41.5, anchors), anchors) == 41.5  # one is the other's inverse
    assert on_cue(5, anchors) == 9 and on_cue(70, anchors) == 67  # the render's own speed outside the hits
    # a merged hit has no anchor; one only reachable at more than a third faster is left where the shift puts it
    assert warp_anchors([12, None, 54], [16, 31, 51]) == [(12, 16), (54, 51)]
    assert warp_anchors([12, 22, 54], [16, 31, 51]) == [(12, 16), (54, 51)]  # 10 render frames over 15: too slow
    # lanterns_call: the big blast came too late to reach after the ghost's pop; landing the blast beats landing the pop
    assert warp_anchors([8, 24, 52], [16, 32, 49], [0.7, 0.75, 1.0]) == [(8, 16), (52, 49)]


def test_a_rhythm_the_model_spaced_unevenly_lands_hit_by_hit():
    from cuesync.scene_shot import time_scene

    cue = cue_of([(1.0, 0.2, 0.9), (1.8, 0.2, 0.9), (2.7, 0.3, 1.0)], length_s=5.0)
    eff = load_effect("fireball")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1, 256, 128, 3), 60, np.uint8)
    sp = paint_scene_guide(cue, eff, (0.5, 0.6), plate, start=start, length=length, size=(128, 256))
    assert len(sp.events) == 3
    wants = [ev.at - eff.lead_frames for ev in sp.events]
    shown = [w + off for w, off in zip(wants, (-4, -1, 3))]  # where the model put each burst
    render = np.full((length, 256, 128, 3), 60, np.uint8)
    yy, xx = np.mgrid[:256, :128]
    disc = (yy - 150) ** 2 + (xx - 64) ** 2 < 30 ** 2
    for o in shown:
        for e in range(o, min(o + 6, length)):  # a burst, then it dies back before the next
            render[e][disc] = (250, 180, 60)
    st = time_scene(plate, True, render, sp, eff, 240)
    assert [o for o, _ in st.warp] == shown
    for o, w, h in zip(shown, wants, st.hits):
        assert h == start + w and st.index[h] == o  # each burst is on the shot frame of its own cue


def _fire_centre_x(frame, w):
    """Mean x of the painted fire in a guide frame, as a share of the width (None: no fire)."""
    fire = np.all(frame == FIRE_OUTER, -1) | np.all(frame == FIRE_INNER, -1)
    if fire.sum() < 20:
        return None
    return float(np.nonzero(fire)[1].mean()) / w


def test_call_and_response_paints_each_hit_at_its_own_source_in_order():
    cue = cue_of([(1.0, 0.2, 0.9), (1.8, 0.2, 0.9), (2.7, 0.3, 1.0)], length_s=5.0)
    eff = load_effect("fire_burst")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1, 256, 128, 3), 50, np.uint8)
    sources = [(0.2, 0.6), (0.5, 0.6), (0.8, 0.6)]
    sp = paint_scene_guide(cue, eff, sources[0], plate, start=start, length=length, size=(128, 256), sources=sources)
    assert len(sp.events) == 3 and sp.sources == sources
    xs = []
    for ev in sp.events:
        k = next(b["e"] for b in sp.guide.blobs if b["e"] >= ev.at + 2)  # a keyframe a little after the hit
        xs.append(_fire_centre_x(sp.guide.frames[k], 128))
    assert xs[0] < 0.35 and 0.35 < xs[1] < 0.65 and xs[2] > 0.65  # left, middle, right, in the rhythm's order
    again = sp.from_json(sp.to_json(), sp.guide.frames)
    assert again.sources == sources and again.source_of(5) == 2
    # only the last hit leaves smoke: an early pop's smoke drifted over the next source and hid its hit
    smoky = [b["e"] for b in sp.guide.blobs if any(s.get("fluid") == "smoke" for s in b["shapes"])]
    assert smoky and min(smoky) > sp.events[-1].at


def test_call_and_response_times_each_hit_at_its_own_source():
    from cuesync.scene_shot import time_scene

    cue = cue_of([(1.0, 0.2, 0.9), (1.8, 0.2, 0.9), (2.7, 0.3, 1.0)], length_s=5.0)
    eff = load_effect("fire_burst")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1, 256, 128, 3), 50, np.uint8)
    sources = [(0.2, 0.6), (0.5, 0.6), (0.8, 0.6)]
    sp = paint_scene_guide(cue, eff, sources[0], plate, start=start, length=length, size=(128, 256), sources=sources)
    wants = [ev.at - eff.lead_frames for ev in sp.events]
    shown = [w + off for w, off in zip(wants, (-4, 2, -1))]  # where the model put each source's fire
    render = np.full((length, 256, 128, 3), 50, np.uint8)
    yy, xx = np.mgrid[:256, :128]
    for (sx, sy), o in zip(sources, shown):
        disc = (yy - sy * 256) ** 2 + (xx - sx * 128) ** 2 < 14 ** 2
        for e in range(o, length):  # each fire stays: there is no dip between them, only another place
            render[e][disc] = (250, 180, 60)
    st = time_scene(plate, True, render, sp, eff, 240)
    assert [o for o, _ in st.warp] == shown
    assert [st.index[h] for h in st.hits] == shown  # each fire on the shot frame of its own cue


def test_a_source_a_neighbours_fire_spread_over_stays_on_its_cue():
    from cuesync.scene_shot import time_scene

    cue = cue_of([(1.0, 0.2, 0.9), (1.8, 0.2, 0.9), (2.7, 0.3, 1.0)], length_s=5.0)
    eff = load_effect("fire_burst")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1, 256, 128, 3), 50, np.uint8)
    sources = [(0.2, 0.6), (0.5, 0.6), (0.8, 0.6)]
    sp = paint_scene_guide(cue, eff, sources[0], plate, start=start, length=length, size=(128, 256), sources=sources)
    wants = [ev.at - eff.lead_frames for ev in sp.events]
    render = np.full((length, 256, 128, 3), 50, np.uint8)
    yy, xx = np.mgrid[:256, :128]
    for (sx, sy), o in zip(sources, (wants[0] - 2, wants[1] + 1, wants[0])):  # the third lit along with the first
        disc = (yy - sy * 256) ** 2 + (xx - sx * 128) ** 2 < 14 ** 2
        for e in range(o, length):
            render[e][disc] = (250, 180, 60)
    st = time_scene(plate, True, render, sp, eff, 240)
    assert len(st.warp) == 2 and all(o != wants[2] - 16 for o, _ in st.warp)
    assert st.hits[2] == start + wants[2]  # not at the edge of its window, where the spread fire already was


def test_call_and_response_blows_candles_out_one_at_a_time():
    cue = cue_of([(1.0, 0.15, 0.9), (1.9, 0.15, 0.9)], length_s=4.0)
    eff = load_effect("blow_out")
    start, length = plan_window(cue, eff, None)
    plate = np.full((1, 256, 128, 3), 50, np.uint8)
    sources = [(0.3, 0.6), (0.7, 0.6)]
    sp = paint_scene_guide(cue, eff, sources[0], plate, start=start, length=length, size=(128, 256), sources=sources)
    between = next(b["e"] for b in sp.guide.blobs if sp.events[0].at + 2 <= b["e"] < sp.events[1].at)
    x = _fire_centre_x(sp.guide.frames[between], 128)
    assert x is not None and x > 0.5  # the first candle is out, the second still burns
    after = sp.guide.blobs[-1]["e"]
    assert _fire_centre_x(sp.guide.frames[after], 128) is None  # then both are out


def test_a_flame_taller_than_where_it_was_painted_stays_solid_over_a_bright_window():
    rng = np.random.default_rng(5)
    n = 12
    base = np.full((256, 128, 3), 70, np.uint8)
    base[:110] = 235  # a bright window above a dark table
    base = np.clip(base.astype(int) + rng.integers(-6, 6, base.shape), 0, 255).astype(np.uint8)
    plate = np.repeat(base[None], n, 0)
    plate[::2, 250:254, 5:15] = 120  # the clip moves a little elsewhere, so the transfer path runs
    region = np.zeros((256, 128), bool)
    region[115:200, 40:90] = True  # where the flame was painted
    render = np.repeat(base[None], n, 0)
    render[4:, 95:200, 50:80] = (250, 150, 40)  # the model's flame grew taller, up over the window
    out, st = transfer(plate, render, list(range(n)), region, ref_frames=3, start=0)
    assert np.abs(out[10, 100:106, 58:72].astype(int) - np.array([250, 150, 40])).mean() < 25  # solid, not see-through
    assert np.abs(out[10, 20:40, 10:30].astype(int) - plate[10, 20:40, 10:30]).mean() < 4  # the rest of the window stays


def test_a_fires_warm_light_carries_over_more_fully_than_a_glow_going_out():
    from cuesync.transfer import light_gain

    ref = np.full((8, 8, 3), 0.3, np.float32)
    warm, _ = light_gain(ref * np.array([2.0, 1.2, 0.8], np.float32), ref, 1.0)
    cool, _ = light_gain(ref * np.array([0.8, 1.2, 2.0], np.float32), ref, 1.0)
    assert warm[0, 0, 0] / warm[0, 0, 2] > cool[0, 0, 2] / cool[0, 0, 0] > 1.0


def test_a_performers_hand_keeps_its_own_pixels_and_gets_the_fires_light():
    from cuesync.transfer import behind

    rng = np.random.default_rng(7)
    n = 12
    base = np.full((256, 128, 3), 60, np.uint8)
    base[100:140, 0:60] = (170, 130, 110)  # an arm coming in from the left, fingertips at x=60
    base = np.clip(base.astype(int) + rng.integers(-8, 8, base.shape), 0, 255).astype(np.uint8)
    plate = np.repeat(base[None], n, 0)
    plate[::2, 250:254, 100:110] = 120  # the clip moves a little elsewhere, so the transfer path runs
    region = np.zeros((256, 128), bool)
    region[95:145, 64:120] = True  # the jet painted to the right of the fingertips
    render = np.repeat(base[None], n, 0)
    render[4:, 105:135, 70:115] = (250, 160, 50)  # the fire
    render[4:, 100:140, 30:60] = (120, 200, 90)  # the model re-drew the hand next to it
    keep = behind((60 / 128, 120 / 256), 90.0, 128, 256)
    assert keep[120, 20] == 1.0 and keep[120, 100] == 0.0 and keep[20, 20] == 0.0  # the arm, not the fire, not the wall
    plain, _ = transfer(plate, render, list(range(n)), region, ref_frames=3, start=0)
    held, _ = transfer(plate, render, list(range(n)), region, ref_frames=3, start=0, keep=keep)
    hand = (slice(108, 132), slice(36, 54))
    assert np.abs(held[10][hand].astype(int) - plate[10][hand]).mean() < np.abs(plain[10][hand].astype(int) - plate[10][hand]).mean()
    assert held[10][hand][..., 1].mean() < 175  # not the model's green hand
    assert np.abs(held[10, 112:128, 80:105].astype(int) - np.array([250, 160, 50])).mean() < 30  # the fire is still there


def test_the_sound_picks_the_effect_by_what_it_is():
    from cuesync.gestures import effect_for_kind

    assert effect_for_kind("boom", True) == "fire_burst"
    assert effect_for_kind("whoosh", True, aimed=True) == "fire_breath"
    assert effect_for_kind("whoosh", True) == "flame_plume"  # a prop nobody aimed flares up
    assert effect_for_kind("whoosh", True, strokes=[{"points": [[0.4, 0.8], [0.42, 0.3]], "width": 0.05}]) == "flame_plume"
    assert effect_for_kind("whoosh", True, strokes=[{"points": [[0.2, 0.5], [0.8, 0.45]], "width": 0.05}]) == "fire_breath"
    assert effect_for_kind("whoosh", True, lit=True) == "wick_flare"
    assert effect_for_kind("hiss", True) == "steam_jet"
    assert effect_for_kind("puff", True, lit=True) == "blow_out" and effect_for_kind("puff", True) == "smoke_puff"
    assert effect_for_kind("boom", True, aimed=True, rhythm=True) == "fire_breath"  # jab, jab, palm-strike: jets
    assert effect_for_kind("boom", False) == "fireball" and effect_for_kind("hiss", False) == "steam_jet"


def test_what_a_take_sounds_like(make_wav, tmp_path):
    from cuesync.audio import analyze
    from cuesync.gestures import heard

    sr = 48000
    t = np.arange(int(0.5 * sr)) / sr
    rng = np.random.default_rng(3)

    def take(sound):
        x = rng.normal(0, 0.001, int(2.5 * sr)).astype(np.float32)
        x[sr: sr + sound.size] += sound.astype(np.float32)
        return x

    boom = take(0.6 * np.sin(2 * np.pi * 110 * t) * np.exp(-3 * t))  # voiced, dying away
    pf = take(rng.normal(0, 0.4, int(0.12 * sr)) * np.exp(-np.arange(int(0.12 * sr)) / (0.03 * sr)))  # short, unvoiced
    hiss = take(np.concatenate([rng.normal(0, 0.2, sr)]))  # long, unvoiced
    for x, want in ((boom, "boom"), (pf, "puff"), (hiss, "hiss")):
        h = heard(analyze(x, sr), x, sr)
        assert h["kind"] == want, (want, h)
        assert isinstance(h["detected_kinds"], list)  # cue_detect's own reading, kept for the record


def test_the_hand_is_segmented_just_behind_the_fingertips():
    from cuesync.transfer import grown, hand_matte

    frame = np.zeros((200, 400, 3), np.uint8)
    seen = []

    def matte_fn(f, at):
        seen.append(at)
        m = np.zeros(f.shape[:2], np.float32)
        m[80:120, 200:300] = 1.0  # a hand to the right of the fingertips
        return m

    m = hand_matte(matte_fn, frame, (0.5, 0.5), -90.0)  # fire to the left: the hand is to the right
    assert m is not None and seen[0][0] > 0.5 and seen[0][1] == 0.5
    assert grown(m)[100, 196] > 0.5 and grown(m)[100, 150] == 0.0  # a little past its edge, not the whole frame
    assert hand_matte(lambda f, at: np.ones(f.shape[:2], np.float32), frame, (0.5, 0.5), -90.0) is None  # the wall


def test_a_dark_background_stays_the_clips_own_around_the_fire():
    rng = np.random.default_rng(11)
    n = 12
    base = np.zeros((256, 128, 3), np.uint8)
    base[..., 2] = 14  # a near-black navy background
    base[150:200, 30:100] = (170, 130, 110)  # a palm
    base = np.clip(base.astype(int) + rng.integers(0, 3, base.shape), 0, 255).astype(np.uint8)
    plate = np.repeat(base[None], n, 0)
    plate[::2, 250:254, 100:110] = 40  # the clip moves a little elsewhere, so the transfer path runs
    region = np.zeros((256, 128), bool)
    region[60:150, 30:100] = True
    render = np.repeat(base[None], n, 0)
    render[4:, ..., 2] = np.where(region, 30, render[4:, ..., 2])  # the model's black is a little bluer around the fire
    render[4:, 100:140, 50:80] = (250, 160, 50)  # the fire
    out, _ = transfer(plate, render, list(range(n)), region, ref_frames=3, start=0)
    assert np.abs(out[10, 65:90, 35:95, 2].astype(int) - plate[10, 65:90, 35:95, 2]).mean() < 4  # the clip's own black
    assert np.abs(out[10, 110:130, 58:72].astype(int) - np.array([250, 160, 50])).mean() < 30  # the fire is still there


def test_words_for_hands_do_not_catch_ordinary_things():
    from cuesync.scene import performer_in

    assert performer_in("the open palm") and performer_in("my two fingers") and performer_in("the fists")
    for thing in ("the alarm clock", "a warm mug of tea", "the chandelier", "the mug handle", "the palm tree", "the armchair"):
        assert not performer_in(thing), thing
    assert "explodes" in scene_prompt(load_effect("fireball"), "the alarm clock", 0)  # not the hand's burst
    assert "explodes" not in scene_prompt(load_effect("fireball"), "my open hands", 0)


def test_made_ahead_sample_shots_still_match_their_effects():
    import json as _json

    from cuesync.precompute import cache_key, sample_pairs

    root = Path(__file__).resolve().parents[2] / "samples"
    index = _json.loads((root / "made_ahead" / "scene" / "index.json").read_text())
    pairs = [p for p in sample_pairs(root, "scene") if p["plate"] != "toy_desk.mp4"]  # made before a rhythm change
    keys = [cache_key("scene", p["plate"], p["voice"], p["effect"], p["point"], 1, p.get("direction"), p.get("subject"))
            for p in pairs]
    assert keys and all(k in index for k in keys)


def test_fire_from_a_hand_is_clean_and_a_pumpkins_still_smokes():
    from cuesync.scene import suited_effect

    frame = np.zeros((256, 128, 3), np.uint8)
    for eid in ("fire_burst", "fire_breath", "fireball"):
        hand = suited_effect(load_effect(eid), "the open palm", 90.0, frame, (0.5, 0.5))
        assert hand.scene.clean and "smoke" not in scene_prompt(hand, "the open palm", 90.0), eid
    pumpkin = suited_effect(load_effect("fire_burst"), "the carved jack-o'-lantern", 0.0, frame, (0.5, 0.5))
    assert not pumpkin.scene.clean and "smoke" in scene_prompt(pumpkin, "the carved jack-o'-lantern", 0.0)
    cue = cue_of([(1.0, 0.6, 0.9)], length_s=4.0)
    base = load_effect("fire_burst")
    guides = []
    for clean in (False, True):  # the same burst, with and without its smoke painted after it
        eff = dataclasses.replace(base, scene=dataclasses.replace(base.scene.suited("the open palm", 0.0, False), clean=clean))
        start, length = plan_window(cue, eff, None)
        guides.append(paint_scene_guide(cue, eff, (0.5, 0.6), frame[None], start=start, length=length, size=(128, 256)).guide.frames)
    assert np.abs(guides[0].astype(int) - guides[1]).max() > 10  # the smoke after the burst is gone from the keyframes
