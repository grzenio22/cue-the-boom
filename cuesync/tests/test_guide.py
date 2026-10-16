import numpy as np
import pytest

from cuesync.audio import CueSheet, analyze
from cuesync.effects import load_effect
from cuesync.guide import keyframe_indices, paint_guide, write_guide
from cuesync.media import read_audio, read_video


def cue_with(env, hero_frame=24, n=200):      # env values are already absolute (0..1)
    e = np.zeros(n); e[hero_frame:hero_frame+len(env)] = env
    return CueSheet(24, 0.0, n/24, [hero_frame/24], hero_frame/24, e, e/max(e.max(), 1e-9), int(e.argmax()))


def test_keyframe_indices():
    assert keyframe_indices(121, 8) == [0, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 120]
    assert keyframe_indices(49, 16) == [0, 17, 33, 48]
    with pytest.raises(ValueError):
        keyframe_indices(120, 8)


def test_non_keyframes_black_and_shape():
    p = paint_guide(cue_with(np.ones(10)), load_effect("fireball"), (0.5, 0.7), length=121)
    assert p.frames.shape == (121, 1024, 576, 3)
    non = [i for i in range(121) if i not in p.keyframes]
    assert p.frames[non].max() == 0
    assert p.anchor_frame == 24


def test_frame0_blob_sits_on_the_point_and_is_orange():
    p = paint_guide(cue_with(np.ones(10)), load_effect("fireball"), (0.25, 0.5))
    ys, xs = np.nonzero(p.frames[0].max(-1))
    assert abs(xs.mean() - 144) < 3                                  # centred on x
    assert 512 <= ys.max() <= 512 + 12                               # erupts out of the point: its base sits there
    assert tuple(p.frames[0][int(ys.mean()), 144]) in [(255, 220, 0), (255, 140, 0)]


def test_louder_performance_makes_bigger_blob():          # whisper vs scream, end to end
    def cue(amp):
        x = np.zeros(3*48000, np.float32); n = 7200
        x[48000:48000+n] = amp*np.sin(2*np.pi*700*np.arange(n)/48000)
        return analyze(x, 48000)
    quiet = paint_guide(cue(0.05), load_effect("smoke_puff"), (0.5, 0.6))
    loud = paint_guide(cue(0.9), load_effect("smoke_puff"), (0.5, 0.6))
    area = lambda p: (p.frames[9].max(-1) > 0).sum()
    assert area(loud) > 1.5*area(quiet)


def test_edge_point_does_not_crash():
    paint_guide(cue_with(np.ones(10)), load_effect("flame_plume"), (0.99, 0.01))


def test_written_guide_has_audio_and_exact_length(tmp_path):
    p = paint_guide(cue_with(np.ones(10)), load_effect("fireball"), (0.5, 0.7), length=73)
    write_guide(p, tmp_path/"g.mp4")
    assert read_video(tmp_path/"g.mp4").frames.shape[0] == 73 and len(read_audio(tmp_path/"g.mp4")) > 0


def test_plume_follows_the_voice_then_paints_its_end():
    env = np.r_[np.linspace(0.2, 1.0, 30), np.zeros(40)]          # rising "fwoooOOOM", then silence
    p = paint_guide(cue_with(env), load_effect("flame_plume"), (0.5, 0.8))
    h = lambda e: (p.frames[e].max(-1).max(1) > 0).sum()           # painted height in rows
    orange = lambda e: ((p.frames[e] == (255, 140, 0)).all(-1)).sum()
    assert p.voice_end == 30 and p.frames.shape[0] == 65          # the element fits the performance
    assert h(9) < h(25)                                           # grows with the voice
    last = p.keyframes[-1]
    assert orange(last) == 0 and p.frames[last].max() > 0         # ends as a smoke wisp: never a black last frame
    old = paint_guide(cue_with(env), load_effect("flame_plume"), (0.5, 0.8), tail="fade")
    assert old.frames[old.keyframes[-1]].max() == 0               # the old shrink-to-nothing tail, kept for A/B


def test_steam_jet_and_burst_secondary_smoke_colors():
    p = paint_guide(cue_with(np.ones(40)), load_effect("steam_jet"), (0.5, 0.8))
    assert (p.frames[9] == 255).all(-1).sum() > 0                 # white steam
    f = paint_guide(cue_with(np.ones(10)), load_effect("fireball"), (0.5, 0.7))
    # smoke appears behind the fading fire (fire holds its peak to e=17, then shrinks; by e=41 smoke shows),
    # painted grey and darkening as it thins
    px = f.frames[41].reshape(-1, 3).astype(int)
    grey = (px[:, 0] == px[:, 1]) & (px[:, 1] == px[:, 2]) & (px[:, 0] > 40) & (px[:, 0] < 190)
    assert grey.sum() > 0


def test_size_must_be_multiple_of_64():
    with pytest.raises(ValueError):
        paint_guide(cue_with(np.ones(10)), load_effect("fireball"), (0.5, 0.5), size=(544, 960))
