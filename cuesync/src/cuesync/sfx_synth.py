"""Procedural, cue-shaped fire, smoke, steam, lightning, water and air Foley (NumPy only).

Import ``synthesize_shot_sfx`` directly; no assets, codecs, devices, or package
initialization are needed. Frame values describe the interval starting at that
frame. The envelope shapes the continuous bed. Hit accents decay independently
of the voice, so a short cue can launch a longer fireball or puff.
``stop_frame`` is the first silent frame, not the start of an audible release.
All randomness is local to the call. Gains are fixed: quiet cues stay quiet.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["synthesize_shot_sfx"]

# name: low/mid/air gains, slow/fast turbulence Hz, crackles per second,
#       transient duration/attack/decay seconds, transient gain.
_PROFILES = {
    "fire_burst": (0.155, 0.105, 0.024, 5.5, 31.0, 13.0, 0.95, 0.0025, 0.13, 0.90),
    "fire_breath": (0.105, 0.145, 0.043, 3.5, 23.0, 7.0, 0.58, 0.007, 0.18, 0.38),
    "flame_plume": (0.175, 0.085, 0.017, 1.8, 17.0, 16.0, 0.90, 0.013, 0.26, 0.35),
    "wick_flare": (0.004, 0.017, 0.027, 7.0, 47.0, 31.0, 0.30, 0.0015, 0.055, 0.13),
    "steam_jet": (0.010, 0.080, 0.175, 2.5, 39.0, 0.0, 0.48, 0.004, 0.13, 0.27),
    "smoke_puff": (0.080, 0.038, 0.003, 2.3, 13.0, 0.0, 0.32, 0.015, 0.20, 0.85),
    "blow_out": (0.026, 0.088, 0.032, 4.0, 27.0, 0.0, 0.22, 0.008, 0.12, 0.85),
    "explosion": (0.190, 0.090, 0.020, 2.5, 19.0, 9.0, 2.40, 0.003, 0.60, 1.00),
    "lightning": (0.050, 0.060, 0.090, 9.0, 61.0, 140.0, 0.45, 0.0004, 0.08, 0.80),
    "water": (0.060, 0.110, 0.030, 1.2, 9.0, 0.0, 0.90, 0.004, 0.25, 0.75),
    "air": (0.070, 0.120, 0.050, 0.9, 7.0, 0.0, 0.70, 0.020, 0.22, 0.80),
    "earth": (0.105, 0.065, 0.012, 2.0, 13.0, 22.0, 1.1, 0.004, 0.35, 0.85),
}
_FLUIDS = ("fire", "smoke", "steam", "electric", "water", "air", "earth")
_BEHAVIOURS = ("burst", "jet", "plume", "puff", "flare", "extinguish", "arc", "ball", "orb", "whip", "blast", "rock", "shatter")
_MADE = {"electric": "lightning", "water": "water", "air": "air", "earth": "earth"}
# The next float below 0.85 prevents float32 rounding exceeding the API bound.
_PEAK = float(np.nextafter(np.float32(0.85), np.float32(0.0)))


def _band_noise(rng, size, sr, low, high):
    """Unit-variance colored noise, with bounded-size equal-power FFT blocks.

    Independent overlapping blocks avoid periodic loops and whole-shot FFT
    allocations. The normalization is the filter's theoretical noise gain,
    never a measured peak/RMS normalization of the finished sound.
    """
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    block = 1 << max(10, min(16, int(math.ceil(math.log2(sr * 0.34)))))
    hop = block // 2
    frequencies = np.fft.rfftfreq(block, 1.0 / sr)
    high = max(1.0, min(high, 0.46 * sr))
    low = max(0.5, min(low, high * 0.8))
    # Smooth 12 dB/octave skirts; DC is removed before any amplitude gating.
    response = (frequencies**2 / (frequencies**2 + low**2))
    response /= np.sqrt(1.0 + (frequencies / high)**4)
    response[0] = 0.0
    power = (2.0 * np.sum(response[1:-1]**2) + response[-1]**2) / block
    response /= math.sqrt(max(float(power), 1e-20))
    window = np.sin(math.pi * (np.arange(block) + 0.5) / block)
    result = np.zeros(size, dtype=np.float32)
    for start in range(-hop, size, hop):
        noise = rng.standard_normal(block)
        colored = np.fft.irfft(np.fft.rfft(noise) * response, n=block)
        left, right = max(0, start), min(size, start + block)
        a, b = left - start, right - start
        result[left:right] += (colored[a:b] * window[a:b]).astype(np.float32)
    return result


def _drift(rng, size, sr, rate):
    """Aperiodic, bounded amplitude turbulence; no audible oscillator."""
    count = int(math.ceil(size / sr * rate)) + 2
    values = rng.uniform(0.45, 1.45, count)
    return np.interp(np.arange(size, dtype=np.float64) * (rate / sr),
                     np.arange(count), values).astype(np.float32)


def _frame_envelope(envelope, size, fps, sr):
    """Hold frame levels; smooth attacks forward and releases backward.

    In particular, a transition from/to zero never leaks sound into a silent
    frame. Each ramp uses at most half a frame so attacks and releases cannot
    overwrite each other at high frame rates. Rounded boundaries agree with
    the public duration/stop convention.
    """
    edges = np.rint(np.arange(envelope.size + 1) / fps * sr).astype(np.int64)
    edges[-1] = size
    result = np.repeat(envelope.astype(np.float32), np.diff(edges))
    ramp_size = max(1, round(0.004 * sr))
    previous = 0.0
    for frame, level in enumerate(envelope):
        start, end = int(edges[frame]), int(edges[frame + 1])
        if level > previous:
            width = min(ramp_size, (end - start) // 2)
            if width:
                ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, width))
                result[start:start + width] = previous + (level - previous) * ramp
        elif level < previous and frame:
            width = min(ramp_size, (start - int(edges[frame - 1])) // 2)
            if width:
                ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, width))
                result[start - width:start] = previous + (level - previous) * ramp
        previous = level
    return result


def _crackle(rng, signal, sr, rate, small, activity=None):
    """Sparse, independently shaped ember snaps, not unfiltered impulses."""
    # Exponential inter-arrivals keep the event sequence stable when only the
    # stop time changes. Full snap patches also keep later RNG draws stable.
    start = int(round(rng.exponential(sr / rate)))
    while start < signal.size:
        # Thinning changes event density, not just the loudness of a fixed
        # pattern. The candidate sequence and full patches stay prefix-stable.
        accepted = rng.random() < (1.0 if activity is None else float(activity[start]))
        duration = rng.uniform(0.002, 0.011 if small else 0.019)
        size = max(4, round(duration * sr))
        t = np.arange(size, dtype=np.float64) / sr
        noise = rng.standard_normal(size + 1)
        snap = np.diff(noise) * 0.5
        attack = 1.0 - np.exp(-t / 0.0002)
        decay = np.exp(-t / (duration * 0.20))
        snap *= attack * decay * rng.uniform(0.025, 0.10 if small else 0.18)
        # Taper each snap so its truncation cannot introduce a hard edge.
        snap *= np.minimum(1.0, (size - 1 - np.arange(size)) / max(1, sr * 0.001))
        used = min(size, signal.size - start)
        if accepted:
            signal[start:start + used] += snap[:used].astype(np.float32)
        start += max(1, int(round(rng.exponential(sr / rate))))


def _bubbles(rng, signal, sr, rate, activity=None, big=1.0):
    """Water's own voice: Minnaert bubbles, each a damped sine whose pitch rises as it closes (a bubble's radius sets
    its note, 400 Hz to 2.5 kHz here), at about ``rate`` a second, thinned by ``activity``."""
    start = int(round(rng.exponential(sr / rate)))
    while start < signal.size:
        accepted = rng.random() < (1.0 if activity is None else float(activity[start]))
        f0 = math.exp(rng.uniform(math.log(400.0), math.log(2500.0))) / big
        duration = rng.uniform(0.018, 0.06) * big
        size = max(8, round(duration * sr))
        t = np.arange(size, dtype=np.float64) / sr
        rise = rng.uniform(0.1, 0.35)
        phase = 2.0 * math.pi * f0 * (t + rise * t * t / (2.0 * duration))
        chirp = np.sin(phase) * (1.0 - np.exp(-t / 0.0015)) * np.exp(-t / (duration * 0.3))
        chirp *= rng.uniform(0.02, 0.07)
        used = min(size, signal.size - start)
        if accepted:
            signal[start:start + used] += chirp[:used].astype(np.float32)
        start += max(1, int(round(rng.exponential(sr / rate))))


def _hum(rng, size, sr, f0=110.0):
    """An electric buzz: a few harmonics of ``f0`` (the odd ones strong: a square-ish arc), wandering a little in
    pitch."""
    wobble = 1.0 + 0.012 * (_drift(rng, size, sr, 3.0).astype(np.float64) - 0.95)
    phase = 2.0 * math.pi * f0 * np.cumsum(wobble) / sr
    out = np.zeros(size, dtype=np.float64)
    for k, a in ((1, 1.0), (2, 0.35), (3, 0.6), (5, 0.4), (7, 0.25), (9, 0.15)):
        out += a * np.sin(k * phase + rng.uniform(0, 2 * math.pi))
    return (out / 1.6).astype(np.float32)


def _decay(t, attack, decay, duration):
    """Physical layer envelope with an exact, gently tapered end."""
    shape = (1.0 - np.exp(-t / attack)) * np.exp(-t / decay)
    release = min(0.012, duration * 0.25)
    tail = np.clip((duration - t) / release, 0.0, 1.0)
    shape *= 0.5 - 0.5 * np.cos(math.pi * tail)
    return shape.astype(np.float32)


def _accent(rng, size, sr, name, profile):
    """Local physical layers; voice envelope is deliberately absent here."""
    t = np.arange(size, dtype=np.float64) / sr
    if name == "fire_burst":
        # 1. Dry ignition snap, then the pressure front hits the listener.
        accent = 0.22 * _band_noise(rng, size, sr, 1800.0, 10000.0)
        accent *= _decay(t, 0.00045, 0.016, 0.080)
        # Analytically integrate an exponential 115 -> 28 Hz drop in 220 ms.
        drop = math.log(115.0 / 28.0) / 0.220
        pitch_time = np.minimum(t, 0.220)
        phase = 2.0 * math.pi * (
            115.0 / drop * (1.0 - np.exp(-drop * pitch_time))
            + 28.0 * np.maximum(t - 0.220, 0.0))
        pressure = 0.58 * np.sin(phase) * _decay(t, 0.002, 0.090, 0.220)
        pressure += 0.48 * _band_noise(rng, size, sr, 35.0, 160.0) * _decay(t, 0.006, 0.27, 0.850)
        # Local drive supplies upper bass harmonics on small speakers before
        # the final mix limiter, rather than flattening the entire fireball.
        accent += 0.62 * np.tanh(pressure * 2.1)
        # 2. Combustion blooms around 45 ms, then rolls away over 600 ms.
        accent += 0.42 * _band_noise(rng, size, sr, 160.0, 1400.0) * _decay(t, 0.0145, 0.30, 0.600)
        # 3. Discrete embers survive the voice and thin as the fireball cools.
        embers = np.zeros(size, dtype=np.float32)
        cooling = _decay(t, 0.003, 0.35, 0.900)
        _crackle(rng, embers, sr, 85.0, False, cooling)
        accent += embers * cooling
    elif name == "explosion":
        # A blast heard close: a dry crack, a long deep drop (100 -> 22 Hz over 0.4 s) that the room rolls on for two
        # seconds, the fireball's roar, and debris and sparks pattering down after it.
        accent = 0.30 * _band_noise(rng, size, sr, 1500.0, 12000.0) * _decay(t, 0.0003, 0.022, 0.120)
        drop = math.log(100.0 / 22.0) / 0.40
        pitch_time = np.minimum(t, 0.40)
        phase = 2.0 * math.pi * (100.0 / drop * (1.0 - np.exp(-drop * pitch_time)) + 22.0 * np.maximum(t - 0.40, 0.0))
        pressure = 0.70 * np.sin(phase) * _decay(t, 0.002, 0.20, 0.60)
        pressure += 0.62 * _band_noise(rng, size, sr, 25.0, 140.0) * _decay(t, 0.010, 0.70, 2.30)
        accent += 0.70 * np.tanh(pressure * 2.4)
        accent += 0.45 * _band_noise(rng, size, sr, 140.0, 1600.0) * _decay(t, 0.025, 0.45, 1.20)
        debris = np.zeros(size, dtype=np.float32)
        falling = _decay(t, 0.15, 0.60, 2.30)
        _crackle(rng, debris, sr, 140.0, False, np.clip(falling * 1.6, 0.0, 1.0))
        accent += 1.4 * debris * falling
    elif name == "lightning":
        # A strike: a sharp broadband crack, a sizzle that tears off it, and a low thump (the air it heats).
        accent = 0.65 * _band_noise(rng, size, sr, 2000.0, 15000.0) * _decay(t, 0.0002, 0.012, 0.060)
        sizzle = _band_noise(rng, size, sr, 3500.0, 14000.0) * _decay(t, 0.002, 0.07, profile[6])
        sizzle *= 0.6 + 0.4 * _drift(rng, size, sr, 90.0)
        accent += 0.38 * sizzle
        accent += 0.40 * np.tanh(2.0 * _band_noise(rng, size, sr, 40.0, 220.0) * _decay(t, 0.002, 0.05, 0.25))
        snaps = np.zeros(size, dtype=np.float32)
        _crackle(rng, snaps, sr, 260.0, True, _decay(t, 0.001, 0.08, profile[6]))
        accent += 2.0 * snaps
    elif name == "earth":
        # Weight first, then dry fracture and falling grit; no electric or watery pitch.
        accent = .65 * _band_noise(rng, size, sr, 40, 220) * _decay(t, .004, .18, .8)
        accent += .32 * _band_noise(rng, size, sr, 400, 4500) * _decay(t, .002, .045, .22)
        debris = np.zeros(size, dtype=np.float32)
        falling = _decay(t, .03, .28, 1.1)
        _crackle(rng, debris, sr, 95, False, falling)
        accent += 1.3 * debris * falling
    elif name == "water":
        # A splash: the slap, a burst of spray, then bubbles and drops falling back.
        accent = 0.45 * _band_noise(rng, size, sr, 180.0, 1400.0) * _decay(t, 0.003, 0.05, 0.25)
        accent += 0.40 * _band_noise(rng, size, sr, 1200.0, 8000.0) * _decay(t, 0.008, 0.14, profile[6])
        drops = np.zeros(size, dtype=np.float32)
        _bubbles(rng, drops, sr, 90.0, _decay(t, 0.02, 0.35, profile[6]))
        accent += 2.2 * drops
    elif name == "air":
        # A gust: a soft low push, then the rush of air going past, falling in pitch as it slows.
        accent = 0.55 * _band_noise(rng, size, sr, 50.0, 260.0) * _decay(t, 0.008, 0.09, 0.30)
        falling = np.exp(-t / 0.18).astype(np.float32)
        rush = _band_noise(rng, size, sr, 250.0, 1400.0) * (1.0 - 0.5 * falling)
        rush += 0.8 * _band_noise(rng, size, sr, 1400.0, 5000.0) * falling
        accent += 0.55 * rush * _decay(t, 0.03, 0.22, profile[6])
    elif name == "smoke_puff":
        # A low air thump followed by a longer, dusty, unpitched whoosh.
        accent = 0.55 * _band_noise(rng, size, sr, 55.0, 320.0) * _decay(t, 0.006, 0.065, 0.180)
        accent += 0.43 * _band_noise(rng, size, sr, 300.0, 1600.0) * _decay(t, 0.018, 0.10, 0.320)
    elif name == "blow_out":
        # The breath's band balance falls from the mouth's brighter onset to
        # a soft low gust. No continuous noise follows this 220 ms gesture.
        falling = np.exp(-t / 0.050).astype(np.float32)
        accent = _band_noise(rng, size, sr, 120.0, 650.0) * (0.75 - 0.25 * falling)
        accent += 0.65 * _band_noise(rng, size, sr, 600.0, 1800.0) * falling
        accent *= 0.65 * _decay(t, 0.006, 0.065, 0.220)
        accent += 0.24 * _band_noise(rng, size, sr, 2500.0, 9000.0) * _decay(t, 0.0004, 0.007, 0.035)
    elif name in ("steam_jet", "wick_flare"):
        shape = _decay(t, profile[7], profile[8], profile[6])
        accent = _band_noise(rng, size, sr, 1500.0, 12500.0)
        accent *= shape
        if name == "steam_jet":
            # A brief pressure release has a little lower, breathy body.
            accent += 0.22 * _band_noise(rng, size, sr, 300.0, 1800.0) * shape
    else:
        shape = _decay(t, profile[7], profile[8], profile[6])
        accent = _band_noise(rng, size, sr, 38.0, 430.0)
        accent *= shape * 0.80
        accent += 0.34 * _band_noise(rng, size, sr, 450.0, 4100.0) * shape
        ignition = (1.0 - np.exp(-t / 0.001)) * np.exp(-t / 0.030)
        accent += 0.13 * _band_noise(rng, size, sr, 1700.0, 11000.0) * ignition
        # A restrained, quickly damped pressure thud, buried beneath turbulence.
        phase = 2.0 * math.pi * (42.0 * t + 7.0 * (1.0 - np.exp(-12.0 * t)))
        accent += 0.14 * np.sin(phase) * shape * np.exp(-t / 0.075)
    width = min(size, max(2, round(sr * 0.012)))
    accent[-width:] *= np.linspace(1.0, 0.0, width, dtype=np.float32)
    return accent.astype(np.float32)


def synthesize_shot_sfx(
    effect_id: str,
    fluid: str,
    behaviour: str,
    n_frames: int,
    envelope: np.ndarray,
    hits: list[int],
    strengths: list[float],
    stop_frame: int | None = None,
    fps: int = 24,
    sr: int = 48000,
    seed: int = 0,
) -> np.ndarray:
    """Return mono float32 Foley, length ``round(n_frames / fps * sr)``.

    Known effect IDs select tuned profiles. New IDs fall back to ``fluid`` and
    ``behaviour``. Real envelope/strength values are clipped to [0, 1]; complex
    or nonfinite values, malformed shapes, or invalid time bases raise
    ``ValueError``.
    Envelope levels shape the bed's low/mid/air bands with powers .55/1.25/2.2.
    Hits have independent physical decays, even if their voice envelope is zero.
    Burst/puff beds are subordinate; extinguish has no bed and, only when hits
    is empty, launches one puff at the envelope's first maximum.
    Out-of-shot hits are ignored. A zero-strength hit adds nothing. Inputs are
    never modified. ``stop_frame`` is clamped to the shot and preserves length.
    Changing only the stop preserves audio before its 12 ms release. The
    default sample rate is 48 kHz; other positive integer rates work too.
    """
    for label, value, minimum in (("n_frames", n_frames, 0), ("fps", fps, 1), ("sr", sr, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if stop_frame is not None and not isinstance(stop_frame, (int, np.integer)):
        raise ValueError("stop_frame must be an integer or None")
    if fluid not in _FLUIDS or behaviour not in _BEHAVIOURS:
        raise ValueError("unsupported fluid or behaviour")
    if np.iscomplexobj(envelope) or np.iscomplexobj(strengths):
        raise ValueError("envelope and strengths must be real-valued")
    env = np.asarray(envelope, dtype=np.float64)
    strength = np.asarray(strengths, dtype=np.float64)
    if env.ndim != 1 or env.size != n_frames:
        raise ValueError("envelope must be one-dimensional with length n_frames")
    if strength.ndim != 1 or strength.size != len(hits):
        raise ValueError("hits and strengths must have the same length")
    if not np.isfinite(env).all() or not np.isfinite(strength).all():
        raise ValueError("envelope and strengths must contain only finite values")
    if any(not isinstance(frame, (int, np.integer)) for frame in hits):
        raise ValueError("hit frames must be integers")
    env = np.clip(env, 0.0, 1.0)
    strength = np.clip(strength, 0.0, 1.0)
    size = int(round(n_frames / fps * sr))
    stop = int(n_frames) if stop_frame is None else min(int(n_frames), max(0, int(stop_frame)))
    cutoff = min(size, round(stop / fps * sr))
    result = np.zeros(size, dtype=np.float32)
    valid_hit = any(0 <= frame < n_frames and level > 0 for frame, level in zip(hits, strength))
    if size == 0 or cutoff == 0 or (not np.any(env) and not valid_hit):
        return result

    name = effect_id
    if name not in _PROFILES and fluid in _MADE:
        name = _MADE[fluid]
    if name not in _PROFILES:
        if behaviour == "extinguish":
            name = "blow_out"
        elif fluid == "steam":
            name = "steam_jet"
        elif fluid == "smoke":
            name = "smoke_puff"
        else:
            name = {"burst": "fire_burst", "jet": "fire_breath", "flare": "wick_flare"}.get(
                behaviour, "flame_plume")
    profile = _PROFILES[name]
    bed_gain = {"burst": 0.25, "puff": 0.15, "extinguish": 0.0, "blast": 0.2}.get(behaviour, 1.0)
    event_name = name
    if behaviour == "extinguish":
        event_name = "blow_out"
    elif behaviour == "puff":
        event_name = "smoke_puff"
    elif behaviour == "burst" and fluid == "fire" and name != "explosion":
        event_name = "fire_burst"
    event_profile = _PROFILES[event_name]
    # Signed integer seeds are supported without touching NumPy's global RNG.
    seed_value = int(seed) % (1 << 64)
    streams = np.random.SeedSequence(seed_value).spawn(7)
    active = cutoff
    if name == "steam_jet":
        bands = ((65.0, 650.0), (500.0, 4200.0), (2400.0, 14500.0))
    elif name == "smoke_puff":
        bands = ((45.0, 420.0), (180.0, 1250.0), (800.0, 2400.0))
    elif name == "wick_flare":
        bands = ((100.0, 700.0), (700.0, 3900.0), (2900.0, 11500.0))
    elif name == "lightning":  # the arc's hiss sits high; its body is the hum (below)
        bands = ((60.0, 400.0), (900.0, 5000.0), (4000.0, 16000.0))
    elif name == "water":
        bands = ((60.0, 380.0), (250.0, 1800.0), (1800.0, 7000.0))
    elif name == "air":
        bands = ((70.0, 420.0), (300.0, 2200.0), (2000.0, 7500.0))
    elif name == "flame_plume":
        # Keep the foundation separate from the moving combustion formant;
        # excessive low-band leakage into the mid band masks their motion.
        bands = ((38.0, 240.0), (300.0, 1800.0), (1700.0, 9500.0))
    else:
        bands = ((38.0, 480.0), (240.0, 2600.0), (1700.0, 9500.0))
    if bed_gain and np.any(env):
        e_env = _frame_envelope(env, size, fps, sr)[:active]
        brightness = e_env ** 2.2
        # Opposite low/mid motion gives a moving combustion formant instead
        # of simply turning a stationary noise spectrum up and down.
        balance = 2.0 * (_drift(np.random.default_rng(streams[6]), active, sr, 0.65) - 0.95)
        for index, (gain, (low, high)) in enumerate(zip(profile[:3], bands)):
            band = _band_noise(np.random.default_rng(streams[index]), active, sr, low, high)
            band *= e_env ** (0.55, 1.25, 2.2)[index]
            if index < 2:
                band *= 1.0 + (0.40 if index == 0 else -0.40) * balance
            result[:active] += (gain * bed_gain) * band
        turbulence = _drift(np.random.default_rng(streams[3]), active, sr, profile[3])
        turbulence *= 0.72 + 0.28 * _drift(np.random.default_rng(streams[4]), active, sr, profile[4])
        if name == "steam_jet":
            turbulence = 0.72 + 0.28 * turbulence
        result[:active] *= turbulence
        if profile[5]:
            crackles = np.zeros(active, dtype=np.float32)
            _crackle(np.random.default_rng(streams[5]), crackles, sr, profile[5],
                     name in ("wick_flare", "lightning"), brightness)
            result[:active] += crackles * brightness * bed_gain * (2.5 if name == "lightning" else 1.0)
        if name == "lightning":  # the arc buzzes with the voice, louder as it is louder
            buzz = _hum(np.random.default_rng(streams[6]), active, sr) * e_env ** 1.3
            result[:active] += 0.075 * bed_gain * buzz
        elif name == "water":  # bubbles in the water as it moves, more as the voice is louder
            bubbles = np.zeros(active, dtype=np.float32)
            _bubbles(np.random.default_rng(streams[5]), bubbles, sr, 35.0, np.clip(e_env, 0.0, 1.0))
            result[:active] += bubbles * bed_gain

    event_hits, event_strengths = hits, strength
    if behaviour == "extinguish" and len(hits) == 0 and np.any(env):
        peak = int(np.argmax(env))
        event_hits, event_strengths = [peak], [float(env[peak])]

    occurrences = {}
    for frame, level in zip(event_hits, event_strengths):
        if not 0 <= frame < n_frames or level == 0:
            continue
        start = int(round(int(frame) / fps * sr))
        if start >= active:
            continue
        # Render the bounded full accent before truncating to the shot. This
        # keeps later noise bands unchanged when a short shot is extended.
        length = max(1, round(event_profile[6] * sr))
        # Per-hit RNG isolates accents from other hits and from the bed.
        occurrence = occurrences.get(int(frame), 0)
        occurrences[int(frame)] = occurrence + 1
        hit_rng = np.random.default_rng([seed_value, int(frame), occurrence, 0x534658])
        accent = _accent(hit_rng, length, sr, event_name, event_profile)
        used = min(length, active - start)
        result[start:start + used] += accent[:used] * (float(level) * event_profile[9])

    # Release ends BEFORE the stop boundary, including the natural shot end.
    width = min(active, max(2, round(0.012 * sr)))
    result[active - width:active] *= np.linspace(1.0, 0.0, width, dtype=np.float32)
    result[active:] = 0.0
    # Smooth overload protection preserves the ordering of quiet/strong cues.
    result /= _PEAK
    np.tanh(result, out=result)
    result *= _PEAK
    return result
