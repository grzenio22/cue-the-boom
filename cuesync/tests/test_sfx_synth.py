"""The made elements' Foley (cuesync.sfx_synth): each one sounds on its hit and not before, within the API's level
bound, the same bytes for the same seed, and with its own character: an explosion's weight in the lows, lightning's
crackle up high, water's bubbles in the middle."""

import numpy as np
import pytest

from cuesync.sfx_synth import synthesize_shot_sfx

SR, FPS, N = 48000, 24, 72


def band_share(x, lo, hi):
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / SR)
    return float(spec[(f >= lo) & (f < hi)].sum() / max(spec[f >= 20].sum(), 1e-12))


def sound(effect, fluid, behaviour, env=None, hits=(24,), seed=3):
    env = np.zeros(N) if env is None else env
    return synthesize_shot_sfx(effect, fluid, behaviour, N, env, list(hits), [1.0] * len(hits), fps=FPS, sr=SR, seed=seed)


@pytest.mark.parametrize("effect,fluid,behaviour", [("explosion", "fire", "burst"), ("lightning_arc", "electric", "arc"),
                                                     ("water_orb", "water", "orb"), ("air_blast", "air", "blast"),
                                                     ("earth_rock", "earth", "rock"), ("earth_shatter", "earth", "shatter")])
def test_each_made_element_sounds_on_its_hit(effect, fluid, behaviour):
    x = sound(effect, fluid, behaviour)
    assert x.size == N * SR // FPS and x.dtype == np.float32
    hit = 24 * SR // FPS
    assert np.abs(x[:hit]).max() == 0.0  # nothing before its hit
    assert np.abs(x[hit: hit + SR // 10]).max() > 0.05  # and something on it
    assert np.abs(x).max() <= 0.85
    assert np.array_equal(x, sound(effect, fluid, behaviour))  # the same bytes again
    assert not np.array_equal(x, sound(effect, fluid, behaviour, seed=4))


def test_each_sounds_like_itself():
    hit = 24 * SR // FPS
    boom = sound("explosion", "fire", "burst")[hit:]
    zap = sound("lightning_arc", "electric", "arc")[hit:]
    splash = sound("water_orb", "water", "orb")[hit:]
    assert band_share(boom, 20, 200) > 0.4  # an explosion's weight is low
    assert band_share(zap, 3000, 16000) > band_share(boom, 3000, 16000) * 4  # lightning crackles high
    assert band_share(splash, 400, 3000) > band_share(boom, 400, 3000)  # water bubbles in the middle
    assert np.abs(boom[int(1.2 * SR):int(1.6 * SR)]).max() > 0.01  # the explosion's rumble rolls on past a second


def test_a_held_buzz_follows_the_voice():
    env = np.zeros(N)
    env[12:48] = 0.8
    x = sound("lightning_ball", "electric", "ball", env=env, hits=())
    a, b = 12 * SR // FPS, 48 * SR // FPS
    assert np.abs(x[:a]).max() == 0.0 and np.sqrt((x[a:b] ** 2).mean()) > 0.01
    assert np.abs(x[b + SR // 50:]).max() < 0.02  # it stops with the voice
