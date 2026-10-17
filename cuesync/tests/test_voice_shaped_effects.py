"""Element mode: the effect ends when the voice ends, voice-shaped bursts, hit alignment, the compositor."""

import json

import numpy as np
import pytest

from cuesync.align import plan_alignment, element_index_for
from cuesync.audio import CueSheet, active_intervals, analyze
from cuesync.composite import composite, voice_gates
from cuesync.effects import load_effect
from cuesync.experiment import _manifest_matches, element_manifest
from cuesync.guide import CANVAS, MIN_RADIUS_PX, paint_guide, performance_length
from cuesync.score import ElementTiming, measure_composite_end, measure_element

SR = 48000


def tone(seconds, amp, f=300):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def room(seconds, dbfs=-55, seed=2):
    return np.random.default_rng(seed).normal(0, 10 ** (dbfs / 20), int(seconds * SR)).astype(np.float32)


def cue_with(env, hero_frame=24, n=200):
    e = np.zeros(n); e[hero_frame:hero_frame + len(env)] = env
    return CueSheet(24, 0.0, n / 24, [hero_frame / 24], hero_frame / 24, e, e / max(e.max(), 1e-9), int(e.argmax()))


# ---------------------------------------------------------------- audio: activity and peak
def test_voice_end_and_pauses():
    x = room(4.0)
    a, b = tone(0.8, 0.3), tone(0.5, 0.3)
    x[SR: SR + a.size] += a                                        # phrase 1: 1.0-1.8 s
    x[int(2.3 * SR): int(2.3 * SR) + b.size] += b                  # 0.5 s pause, phrase 2: 2.3-2.8 s
    cue = analyze(x, SR)
    iv = cue.intervals()
    assert len(iv) == 2                                            # a long pause ends the gesture
    assert abs(iv[0][0] - 24) <= 1 and abs(iv[0][1] - 43) <= 2 and abs(cue.voice_end_frame - 67) <= 2
    assert CueSheet.from_json(json.loads(json.dumps(cue.to_json()))).intervals() == iv


def test_short_breath_does_not_split_a_gesture():
    assert active_intervals(np.r_[np.ones(10), np.zeros(4), np.ones(10)].astype(bool)) == [(0, 24)]
    assert active_intervals(np.r_[np.ones(10), np.zeros(8), np.ones(10)].astype(bool)) == [(0, 10), (18, 28)]


def test_peak_of_a_saturating_rise_is_at_its_top():
    t = np.arange(int(2.0 * SR)) / SR
    x = room(3.5)
    x[SR: SR + t.size] += ((0.05 + 0.425 * t) * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # 0.05 -> 0.9
    cue = analyze(x, SR)                                           # the envelope clips from ~1.1 s (-6 dBFS) on
    assert cue.envelope[24 + 30: 24 + 46].min() == 1.0            # ...so the clipped curve is flat at the top
    assert abs(cue.peak_frame - (24 + 47)) <= 2                   # the loudest frame, not the first clipped one


# ---------------------------------------------------------------- guide
def test_no_painted_blob_is_smaller_than_one_latent_cell():
    for eff in ("fireball", "flame_plume", "smoke_puff", "steam_jet"):
        for env in (np.ones(5) * 0.2, np.r_[np.linspace(0.2, 1, 30), np.zeros(20)]):
            p = paint_guide(cue_with(env), load_effect(eff), (0.5, 0.75))
            for b in p.blobs:
                for s in b["shapes"]:
                    if s["kind"] != "fire_core":
                        assert min(s["rx"], s["ry"]) >= MIN_RADIUS_PX, (eff, b["e"], s)


def test_element_length_fits_the_performance():
    fire, plume = load_effect("fireball"), load_effect("flame_plume")
    assert performance_length(cue_with(np.ones(3)), plume) == 49        # short: the 49-frame floor
    assert performance_length(cue_with(np.ones(40)), plume) == 73        # 40 + 24 tail + 8 -> 8k+1
    assert performance_length(cue_with(np.ones(150), n=400), plume) == 121
    assert performance_length(cue_with(np.ones(3)), fire) == 97        # 3 + 80 tail (smoke lingers) -> 8k+1


def test_longer_boom_holds_the_fireball_and_makes_more_smoke():
    short = paint_guide(cue_with(np.ones(4) * 0.8), load_effect("fireball"), (0.5, 0.7), length=97)
    long = paint_guide(cue_with(np.ones(36) * 0.8), load_effect("fireball"), (0.5, 0.7), length=97)
    rx = lambda p, e, k: max([s["rx"] for b in p.blobs if b["e"] == e for s in b["shapes"] if s["kind"] == k] or [0])
    assert rx(long, 33, "fire") > rx(short, 33, "fire")               # still burning while the voice lasts
    assert rx(long, 73, "smoke") > rx(short, 73, "smoke")


def test_longer_puff_grows_bigger():
    short = paint_guide(cue_with(np.ones(4) * 0.6), load_effect("smoke_puff"), (0.5, 0.7), length=73)
    long = paint_guide(cue_with(np.ones(40) * 0.6), load_effect("smoke_puff"), (0.5, 0.7), length=73)
    area = lambda p: (p.frames[p.keyframes[-1]].max(-1) > 0).sum()
    assert area(long) > 1.3 * area(short)


def test_canvas_keyframes():
    env = np.r_[np.linspace(0.2, 1.0, 30), np.zeros(40)]
    p = paint_guide(cue_with(env), load_effect("flame_plume"), (0.5, 0.8), canvas=True)
    for e in p.keyframes:
        assert tuple(p.frames[e][0, 0]) == CANVAS                    # every keyframe is on the canvas
    non = [i for i in range(p.frames.shape[0]) if i not in p.keyframes]
    assert p.frames[non].max() == 0                                # in-betweens stay black (unconstrained)
    assert (p.frames[p.keyframes[-1]] == CANVAS).all()             # the end is an empty canvas: "nothing here"


# ---------------------------------------------------------------- score / align
def fireball_like(n=60):
    f = np.zeros((n, 64, 64, 3), np.uint8)
    for i in range(n):
        if 4 <= i < 20:
            f[i, 20:44, 20:44] = (255, 150, 30)                     # fireball, frames 4-19
        if i >= 20:
            f[i, 10:54, 10:54] = (235, 235, 235)                    # big white smoke cloud afterwards
    return f


def test_fire_peak_ignores_the_late_white_smoke():
    t = measure_element(fireball_like(), kind="fire")
    assert 4 <= t.peak < 20 and t.visible_onset <= 5 and 4 <= t.hit_frame <= 6
    assert measure_element(fireball_like()).peak >= 20             # plain luma is fooled, as before


def test_alignment_modes():
    t = ElementTiming(visible_onset=1, peak=12, curve=np.zeros(1), hit=9)
    g = plan_alignment(40, t, mode="guide")
    assert element_index_for(g, t, 40, 97) == 0 and element_index_for(g, t, 39, 97) is None
    h = plan_alignment(40, t, mode="hit", lead=1)
    assert element_index_for(h, t, 39, 97) == 9                    # the hit, one frame before the sound
    assert element_index_for(h, t, 36, 97) is None                 # the seed frames before it are hidden
    assert element_index_for(h, t, 37, 97) == 7


# ---------------------------------------------------------------- compositor
def test_voice_gates():
    gf, gs = voice_gates(80, [(20, 40)], 20, "both", release_fire=5, release_smoke=24)
    assert gf[19] == 0 and gf[20] == 1 and gf[39] == 1
    assert 0 < gf[40] < 1 and gf[45] == 0                          # fire gone 5 frames after the voice
    assert gs[45] > 0 and gs[64] == 0                              # smoke disperses over 24
    gf, gs = voice_gates(80, [(20, 30), (50, 60)], 20, "both", 5, 24)
    assert gf[40] == 0 and gf[50] == 1                             # a new phrase re-attacks
    gf, _ = voice_gates(80, [(20, 25)], 20, "attack", 5, 24)
    assert gf[19] == 0 and gf[79] == 1


def fire_setup(plate_val, n=40):
    plate = np.full((n, 128, 72, 3), plate_val, np.uint8)
    el = np.zeros((n, 128, 72, 3), np.uint8); el[:, 30:80, 20:52] = (255, 150, 30)
    t = ElementTiming(0, 1, np.zeros(1))
    return plate, el, t


def test_the_effect_stops_when_the_voice_stops():
    plate, el, t = fire_setup(90)
    gates = voice_gates(40, [(5, 20)], 5, "both", release_fire=5, release_smoke=24)
    out = composite(plate, el, plan_alignment(5, t, mode="guide"), t, gates=gates)
    assert measure_composite_end(plate, out) == 25                 # 20 + 5-frame release, though the element burns on
    assert np.array_equal(out[25:], plate[25:])


def test_fire_stays_orange_over_a_bright_plate():
    plate, el, t = fire_setup(235)
    out = composite(plate, el, plan_alignment(0, t), t)[3, 55, 36].astype(int)
    assert out[0] - out[2] > 60                                    # visibly orange, not a pale veil


def test_holdout_cuts_below_the_emission_point():
    plate, el, t = fire_setup(90)
    el[:, 80:120, 20:52] = (255, 150, 30)                          # a "nozzle" below the emission point
    cut = composite(plate, el, plan_alignment(0, t), t, emission=(36, 80), holdout_below=0.02)
    uncut = composite(plate, el, plan_alignment(0, t), t)
    below = (slice(None), slice(100, 120), slice(20, 52))
    assert uncut[below][..., 0].mean() > cut[below][..., 0].mean() + 50  # the nozzle is gone...
    assert cut[below].mean() < plate[below].mean() + 30            # ...leaving only the fire's light on the table
    assert cut[3, 50, 36, 0] > plate[3, 50, 36, 0] + 50           # the fire above the cut is untouched


def test_dense_smoke_core_hides_a_bright_plate():
    yy, xx = np.mgrid[:128, :128]
    r = np.hypot(yy - 64, xx - 64)
    cloud = np.where(r < 50, np.clip(0.25 + 0.6 * (r / 50) ** 2, 0, 1), 0)   # dark self-shadowed core, lit rim
    el = (cloud[..., None] * np.array([230, 230, 230]))[None].repeat(3, 0).astype(np.uint8)
    plate = np.zeros((3, 128, 128, 3), np.uint8); plate[:, :, ::8] = 255           # bright stripes behind
    t = ElementTiming(0, 1, np.zeros(1))
    core = composite(plate, el, plan_alignment(0, t), t)[1, 56:72, 56:72].astype(int)
    stripes = core[:, ::8].mean() - core[:, 4::8].mean()
    assert stripes < 40                                            # the stripes barely show through the core


def test_light_pool_is_finite():
    plate = np.full((3, 256, 256, 3), 60, np.uint8)
    el = np.zeros((3, 256, 256, 3), np.uint8); el[:, 120:136, 120:136] = (255, 140, 0)
    t = ElementTiming(0, 1, np.zeros(1))
    out = composite(plate, el, plan_alignment(0, t), t)
    assert np.array_equal(out[:, :30, :30], plate[:, :30, :30])
    assert out[1, 128, 100, 0] > plate[1, 128, 100, 0]            # but the pool does light the plate nearby


# ---------------------------------------------------------------- experiment hygiene
def test_element_reuse_needs_matching_inputs(tmp_path):
    guide = tmp_path / "g.mp4"; guide.write_bytes(b"guide-a")
    el = tmp_path / "element_n1_seed1.mp4"; el.write_bytes(b"x")
    m = element_manifest(guide, "ainvfxfluid, fire", 1, None)
    assert not _manifest_matches(el, m)                            # no manifest: not reusable
    el.with_suffix(".json").write_text(json.dumps(m))
    assert _manifest_matches(el, m)
    guide.write_bytes(b"guide-b")
    assert not _manifest_matches(el, element_manifest(guide, "ainvfxfluid, fire", 1, None))
    assert not _manifest_matches(el, element_manifest(tmp_path / "g.mp4", "ainvfxfluid, smoke", 1, None))


# ---------------------------------------------------------------- round-2 review regressions
from cuesync.composite import first_shown_frame  # noqa: E402
from cuesync.experiment import run  # noqa: E402
from cuesync.media import write_video  # noqa: E402


def test_gate_releases_in_a_pause_when_the_anchor_sits_inside_the_first_phrase():
    gf, _ = voice_gates(100, [(23, 44), (55, 68)], 24, "both", 5, 24)
    assert gf[50] == 0 and gf[55] == 1 and gf[43] == 1
    gf, _ = voice_gates(100, [(11, 13), (35, 60)], 12, "both", 5, 24)   # a lip smack before the noise
    assert gf[20:34].max() == 0


def test_hit_aligned_burst_shows_its_build_up_before_the_sound():
    t = ElementTiming(visible_onset=5, peak=12, curve=np.zeros(1), hit=9)
    a = plan_alignment(40, t, mode="hit", lead=1)
    gates = voice_gates(80, [(40, 44)], 40, "attack", 5, 24, open_from=first_shown_frame(a, t))
    assert gates[0][37] == 1 and gates[0][36] == 0                 # pre-roll (hit-2) shown, nothing earlier
    plate = np.full((80, 64, 64, 3), 50, np.uint8)
    el = np.zeros((40, 64, 64, 3), np.uint8)
    for i in range(5, 40):
        r = min(20, 2 + 2 * (i - 5))
        el[i, 32 - r:32 + r, 32 - r:32 + r] = (255, 150, 30)
    out = composite(plate, el, a, t, gates=gates)
    assert not np.array_equal(out[37], plate[37]) and np.array_equal(out[36], plate[36])


def test_breaths_and_recorder_zeros_do_not_extend_the_performance():
    base = room(4.0, -50)
    base[SR: SR + int(0.5 * SR)] += tone(0.5, 0.3)                   # the performance: 1.0-1.5 s
    clean = analyze(base, SR).voice_end_frame
    breath = base.copy()
    b = np.random.default_rng(5).normal(0, 10 ** (-40 / 20), int(0.4 * SR)).astype(np.float32)
    breath[int(2.1 * SR): int(2.1 * SR) + b.size] += b * np.hanning(b.size).astype(np.float32)
    zeros = base.copy(); zeros[: int(0.3 * SR)] = 0                  # recorder start-up silence
    for x in (breath, zeros):
        assert abs(analyze(x, SR).voice_end_frame - clean) <= 1


def test_steam_jet_length_follows_the_voice():
    quiet = paint_guide(cue_with(np.ones(30) * 0.2), load_effect("steam_jet"), (0.5, 0.8), length=73)
    loud = paint_guide(cue_with(np.ones(30) * 1.0), load_effect("steam_jet"), (0.5, 0.8), length=73)
    ry = lambda p: max(s["ry"] for b in p.blobs if b["e"] == 17 for s in b["shapes"])
    assert ry(loud) > 1.5 * ry(quiet)


@pytest.mark.parametrize("eff", ["flame_plume", "steam_jet"])
@pytest.mark.parametrize("y", [0.1, 0.3, 0.5])
def test_the_wisp_tail_stays_in_frame(eff, y):
    p = paint_guide(cue_with(np.r_[np.ones(30), np.zeros(10)]), load_effect(eff), (0.5, y))
    assert (p.frames[p.keyframes[-1]].max(-1) > 0).sum() > (2 * MIN_RADIUS_PX) ** 2 // 2


def test_bursts_and_puffs_do_not_reach_far_below_the_point():
    for eff in ("fireball", "smoke_puff"):
        e = load_effect(eff)
        p = paint_guide(cue_with(np.ones(20)), e, (0.5, 0.6))
        cut = int(0.6 * 1024 + e.holdout_below * 1024)
        assert p.frames[9][cut + 1:].max() == 0, eff


def flame_with_fringe(n=40):
    el = np.zeros((n, 128, 96, 3), np.uint8)
    el[:, 40:80, 30:66] = (255, 150, 30)                            # orange body
    el[:, 30:40, 30:66] = (82, 20, 5)                               # dim red fringe
    el[:, 90:110, 70:90] = (150, 150, 150)                          # a separate grey smoke patch
    return el


def test_a_flame_goes_out_as_a_whole_no_ghost():
    plate = np.full((40, 128, 96, 3), 120, np.uint8)
    t = ElementTiming(0, 1, np.zeros(1))
    gates = voice_gates(40, [(5, 20)], 5, "both", 5, 10)
    out = composite(plate, flame_with_fringe(), plan_alignment(5, t, mode="guide"), t, gates=gates)
    flame = (slice(None), slice(25, 85), slice(25, 70))
    assert np.array_equal(out[25][flame[1:]], plate[25][flame[1:]])   # the whole flame is gone with the fire gate
    patch = (slice(88, 112), slice(68, 92))
    assert not np.array_equal(out[27][patch], plate[27][patch])        # the grey smoke still disperses
    assert np.array_equal(out[31][patch], plate[31][patch])


@pytest.mark.parametrize("plate_val", [128, 235])
def test_fire_keeps_its_hue_over_any_plate(plate_val):
    plate = np.full((4, 128, 72, 3), plate_val, np.uint8)
    el = np.zeros((4, 128, 72, 3), np.uint8)
    ramp = np.linspace(0.5, 1.0, 50)[:, None, None]
    el[:, 30:80, 20:52] = (ramp * np.array([255, 120, 20])).astype(np.uint8)   # a gradient flame, not one flat patch
    t = ElementTiming(0, 1, np.zeros(1))
    o = composite(plate, el, plan_alignment(0, t), t)[2, 40:75, 30:42].reshape(-1, 3).astype(int)
    assert (o[:, 0] - o[:, 1]).mean() > 45                          # R well above G: orange, not yellow-white


def test_dense_smoke_has_no_halo_and_no_hard_edge():
    yy, xx = np.mgrid[:160, :160]
    r = np.hypot(yy - 80, xx - 80)
    rim = np.clip(1 - np.abs(r - 45) / 12, 0, 1)                    # rim-lit puff: bright rim, dark core
    puff = np.where(r < 57, 0.25 + 0.55 * rim, 0) * np.clip((57 - r) / 6, 0, 1)
    el = (puff[..., None] * np.array([235, 235, 235]))[None].repeat(3, 0).astype(np.uint8)
    plate = np.full((3, 160, 160, 3), 235, np.uint8)
    t = ElementTiming(0, 1, np.zeros(1))
    out = composite(plate, el, plan_alignment(0, t), t)[1].astype(int)
    ring = (r > 60) & (r < 100)
    assert out[ring].max() <= 235 + 2                                  # nothing lifted outside the cloud
    profile = out[80, 80:150, 0]
    assert np.abs(np.diff(profile)).max() <= 25                       # no cut-out edge


def test_one_stray_pixel_does_not_recolour_the_cloud():
    plate = np.zeros((3, 200, 120, 3), np.uint8)
    plate[:, :100] = (110, 160, 230); plate[:, 100:] = (120, 85, 50)  # sky over ground
    el = np.zeros((3, 200, 120, 3), np.uint8)
    yy, xx = np.mgrid[:200, :120]
    el[:, (yy - 150) ** 2 + (xx - 60) ** 2 < 25 ** 2] = (230, 230, 230)
    t = ElementTiming(0, 1, np.zeros(1))
    a = composite(plate, el, plan_alignment(0, t), t)
    el2 = el.copy(); el2[:, 5, 5] = (12, 12, 12)
    b = composite(plate, el2, plan_alignment(0, t), t)
    assert np.abs(a[1, 140:160, 50:70].astype(int) - b[1, 140:160, 50:70]).max() <= 2


def test_stale_elements_are_refused_unless_allowed(tmp_path, make_video, make_wav):
    plate = make_video(tmp_path, n=48, fps=24, w=180, h=320, color_fn=lambda i: (60, 60, 60))
    x = room(2.0, -55); x[SR // 2: SR // 2 + 7200] += tone(0.15, 0.6)
    noise = make_wav(tmp_path, x, SR)
    old = tmp_path / "old"; old.mkdir()
    f = np.zeros((121, 256, 144, 3), np.uint8); f[10:40, 100:160, 50:94] = (255, 140, 0)
    write_video(old / "element_n1_seed1.mp4", f, fps=24)                # rendered from a 121-frame guide
    kw = dict(plates=[plate], noises=[noise], offsets=[0.0], effect="fireball", point=(0.5, 0.5), seeds=[1],
              elements_dir=old, workflow=None, element_size=(128, 256), plate_size=(180, 320))
    with pytest.raises(ValueError, match="another guide"):
        run(out=tmp_path / "a", **kw)
    m = run(out=tmp_path / "b", allow_stale=True, **kw)
    assert m["runs"][0]["stale_element"] and m["runs"][0]["stop_latency_frames"] is None


def test_a_soft_lead_in_belongs_to_the_gesture():
    """A sharp quiet "f" (the onset) swelling smoothly into a loud "WOOOM": the performance starts at the
    onset, not when it gets loud, so a plume's voice gate never closes in between."""
    from cuesync.audio import analyze
    from cuesync.composite import voice_gates
    sr = 48000
    rng = np.random.default_rng(5)
    x = rng.normal(0, 10 ** (-62 / 20), 4 * sr).astype(np.float32)  # room tone
    n1 = int(0.3 * sr)
    x[sr: sr + n1] += (0.03 * rng.standard_normal(n1)).astype(np.float32)  # the "f": abrupt, quiet
    t2 = np.arange(int(1.2 * sr)) / sr
    swell = np.minimum(t2 / 0.4, 1.0) ** 2  # no second onset: it fades in
    x[sr + n1: sr + n1 + t2.size] += (0.8 * swell * np.sin(2 * np.pi * 120 * t2)).astype(np.float32)
    cue = analyze(x, sr, fps=24)
    hero = round(cue.hero_onset_s * 24)
    assert hero <= 25  # the onset is the "f"
    first = cue.intervals()[0]
    assert first[0] <= hero + 1
    gf, _ = voice_gates(100, cue.intervals(), hero, "both", 9, 10)
    assert (gf[hero: first[1]] == 1.0).all()  # never closes during the gesture
