"""Mouth-sound cues from mono audio, using only NumPy and the standard library.

Times refer to perceptual attacks, not maxima of the loudness envelope. A
positive ``audio_offset_ms`` means captured audio is late: 44 subtracts 44 ms
from audio times. Use the same correction for hits, envelope and stop_frame.
Frame indices use nearest-frame rounding (half frames round forward).

The six kinds are acoustic heuristics, not speech recognition. A soft breath
and an intentional breathy whoosh can be acoustically indistinguishable. A
short quiet lead-in improves the noise estimate. Strengths preserve recording
level, but cannot reconstruct dynamics removed by microphone AGC.

``stop_frame`` is the first silent frame after the last sustained cue, including
any later transients; it is None for solely impulsive cues or a truncated tail.
It may equal n_frames. The envelope is zero outside accepted cue regions.
"""

from __future__ import annotations

import math
import operator

import numpy as np

__all__ = ["detect_cues", "measure_clap_offset"]

_SR = 24000
_HOP = 120  # 5 ms; timing is refined independently of the FFT window.
_FFT = 512


def _uniform_filter1d(x, size):
    """Centered box average, zero extension, and even-window origin zero."""
    if not len(x):
        return np.empty(0, np.float64)
    left = size // 2
    padded = np.pad(x, (left, size - 1 - left))
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    return (cumulative[size:] - cumulative[:-size]) / size


def _gaussian_filter1d(x, sigma):
    """Order-zero Gaussian, four-sigma truncation and nearest boundaries."""
    if not len(x):
        return np.empty(0, np.float64)
    radius = int(4 * sigma + 0.5)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 / (sigma * sigma) * positions ** 2)
    weights /= weights.sum()
    return np.convolve(np.pad(x, (radius, radius), mode="edge"), weights, mode="valid")


def _resample_poly(x, up, down):
    """Zero-extended polyphase FIR, Kaiser beta 5, centered group delay."""
    divisor = math.gcd(up, down)
    up //= divisor
    down //= divisor
    if up == down == 1:
        return x.copy()
    n_out = (len(x) * up + down - 1) // down
    if not n_out:
        return np.empty(0, np.float64)
    max_rate = max(up, down)
    half_len = 10 * max_rate
    positions = np.arange(-half_len, half_len + 1, dtype=np.float64)
    taps = np.sinc(positions / max_rate) / max_rate
    taps *= np.kaiser(len(taps), 5.0)
    taps *= up / taps.sum()
    # Algebraically remove the padded FIR's group delay. Each output visits
    # only its nonzero upsample phase, without allocating an upsampled signal.
    locations = np.arange(n_out, dtype=np.int64) * down + half_len
    latest = locations // up
    phase = locations % up
    result = np.zeros(n_out, np.float64)
    for lag in range((len(taps) + up - 1) // up):
        tap = phase + lag * up
        sample = latest - lag
        valid = (tap < len(taps)) & (sample >= 0) & (sample < len(x))
        result[valid] += taps[tap[valid]] * x[sample[valid]]
    return result


def _highpass65(x):
    """Zero-state, second-order Butterworth high-pass at 65 Hz / 24 kHz."""
    k = math.tan(math.pi * 65 / _SR)
    b0 = 1 / (1 + math.sqrt(2) * k + k * k)
    b1, b2 = -2 * b0, b0
    a1 = 2 * (k * k - 1) * b0
    a2 = (1 - math.sqrt(2) * k + k * k) * b0
    result = np.empty_like(x)
    z1 = z2 = 0.0
    for i, value in enumerate(x):
        filtered = b0 * value + z1
        z1 = b1 * value - a1 * filtered + z2
        z2 = b2 * value - a2 * filtered
        result[i] = filtered
    return result


def _find_peaks(x, *, height=None, prominence=None, distance=1):
    """Finite 1D peaks with scalar height/distance/prominence lower bounds.

    Plateaus use their midpoint rounded down. Height, distance and prominence
    are applied in that order; equal-height distance ties use argsort priority.
    Only the properties consumed by the detector are returned.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        peaks = np.empty(0, dtype=np.intp)
    else:
        edges = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, n]
        levels = x[edges[:-1]]
        maxima = np.flatnonzero((levels[1:-1] > levels[:-2])
                                & (levels[1:-1] > levels[2:])) + 1
        peaks = (edges[maxima] + edges[maxima + 1] - 1) // 2
    if height is not None:
        peaks = peaks[x[peaks] >= height]
    if distance > 1 and len(peaks):
        spacing = math.ceil(distance)
        keep = np.ones(len(peaks), dtype=bool)
        for j in np.argsort(x[peaks])[::-1]:
            if keep[j]:
                a = np.searchsorted(peaks, peaks[j] - spacing, side="right")
                b = np.searchsorted(peaks, peaks[j] + spacing, side="left")
                keep[a:b] = False
                keep[j] = True
        peaks = peaks[keep]
    props = {}
    if height is not None:
        props["peak_heights"] = x[peaks]
    if prominence is None:
        return peaks, props
    if not len(peaks):
        props["prominences"] = np.empty(0, dtype=np.float64)
        return peaks, props
    # Strictly higher samples bound each contour search. Equal-height peaks
    # are crossed. Monotone stacks avoid quadratic work on periodic signals.
    bounds = []
    for indices, edge in ((range(n), -1), (range(n - 1, -1, -1), n)):
        stack = []
        bound = np.empty(n, dtype=np.intp)
        for i in indices:
            while stack and x[stack[-1]] <= x[i]:
                stack.pop()
            bound[i] = stack[-1] if stack else edge
            stack.append(i)
        bounds.append(bound[peaks])
    # Batch range-minimum queries on both sides of every retained peak.
    size = 1 << (n - 1).bit_length()
    tree = np.full(2 * size, np.inf)
    tree[size:size + n] = x
    level = size
    while level > 1:
        tree[level // 2:level] = np.minimum(tree[level:2 * level:2],
                                          tree[level + 1:2 * level:2])
        level //= 2
    left = np.r_[bounds[0] + 1, peaks] + size
    right = np.r_[peaks + 1, bounds[1]] + size
    minima = np.full(2 * len(peaks), np.inf)
    while np.any(left < right):
        active = left < right
        take = active & (left % 2 == 1)
        minima[take] = np.minimum(minima[take], tree[left[take]])
        take = active & (right % 2 == 1)
        minima[take] = np.minimum(minima[take], tree[right[take] - 1])
        left = (left + 1) // 2
        right //= 2
    count = len(peaks)
    prominences = x[peaks] - np.maximum(minima[:count], minima[count:])
    keep = prominences >= prominence
    props = {key: value[keep] for key, value in props.items()}
    props["prominences"] = prominences[keep]
    return peaks[keep], props


def _integer(value, name, minimum):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _audio(audio, sr):
    sr = _integer(sr, "sr", 1)
    x = np.asarray(audio)
    if x.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono array")
    if np.iscomplexobj(x):
        raise ValueError("audio must be real")
    x = np.array(x, dtype=np.float64, copy=True)
    x[~np.isfinite(x)] = 0.0
    if not x.size:
        return x
    # Remove true DC before the filter, so constant input cannot create an onset.
    x -= np.median(x)
    if sr != _SR:
        divisor = math.gcd(sr, _SR)
        x = _resample_poly(x, _SR // divisor, sr // divisor)
    return _highpass65(x)


def _features(x):
    centres = np.arange(0, len(x), _HOP)
    # Centered 10 ms power windows avoid the latency of 20-40 ms FFT windows.
    rms = np.sqrt(np.maximum(0, _uniform_filter1d(x * x, 240))[centres])
    rms = _gaussian_filter1d(rms, 0.65)
    padded = np.pad(x, (_FFT // 2, _FFT))
    windows = np.lib.stride_tricks.sliding_window_view(padded, _FFT)[::_HOP][:len(centres)]
    power = np.abs(np.fft.rfft(windows * np.hanning(_FFT), axis=1)) ** 2
    freq = np.fft.rfftfreq(_FFT, 1 / _SR)
    power[:, freq < 70] = 0
    return centres / _SR, rms, power, freq


def _runs(mask, gap=0):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > gap + 1) + 1
    return [(int(a[0]), int(a[-1]) + 1) for a in np.split(indices, cuts)]


def _event_features(rms, power, freq, a, b, noise_power):
    curve = rms[a:b]
    peak = float(curve.max())
    use = curve >= 0.3 * peak
    spectrum = np.maximum(power[a:b][use].sum(axis=0) - noise_power * int(use.sum()), 0)
    total = float(spectrum.sum()) + 1e-30
    centroid = float(spectrum @ freq / total)
    low = float(spectrum[freq < 650].sum() / total)
    air = float(spectrum[freq > 3500].sum() / total)
    # Flatness must be measured inside the breath's passband. Including silent
    # high-frequency bins makes colored noise look falsely harmonic.
    band = spectrum[(freq > 150) & (freq < 1800)]
    flatness = float(np.exp(np.mean(np.log(band + total * 1e-12))) / (band.mean() + 1e-30))
    duration = (b - a) * _HOP / _SR
    # Smoothed RMS peaks, rather than individual noise-waveform zero crossings.
    local_peaks, _ = _find_peaks(curve, prominence=0.23 * peak, distance=4)
    occupancy = float(np.mean(curve >= 0.35 * peak))
    attack = int(np.argmax(curve)) * _HOP / _SR
    if len(local_peaks) >= 3 and occupancy < 0.72:
        kind = "crackle"
    elif air > 0.47 and centroid > 3300 and duration > 0.12:
        kind = "hiss"
    elif duration < 0.12:
        kind = "pop"
    elif low > 0.57 and centroid < 1300:
        kind = "boom"
    elif duration < 0.42 and centroid < 2300:
        kind = "puff"
    else:
        kind = "whoosh"
    # Slow, low-mid, unvoiced breaths are not intentional attacks. Impulsive
    # puffs and bright fricatives are retained even when similarly quiet.
    breath = (duration > 0.22 and attack > 0.085 and centroid < 1750
              and low < 0.70 and air < 0.10 and flatness > 0.20)
    return kind, breath, peak, duration


def _events(times, rms, power, freq):
    if len(rms) < 2 or float(rms.max()) < 0.0007:
        return [], 0.0
    interior = rms[3:-3] if len(rms) > 12 else rms
    floor = max(float(np.quantile(interior, 0.15)), 1e-6)
    # A fixed percentile mistakes a long sustained cue for background when it
    # fills >85% of the take. Find a quiet 50 ms interval instead. Exclude short
    # digital startup padding (e.g. AAC delay), but honor real longer silence.
    silent = rms < 1e-7
    if any(b - a >= 16 for a, b in _runs(silent)):
        floor = 1e-6
    elif len(rms) >= 10:
        average = np.convolve(rms * rms, np.ones(10) / 10, mode="valid")
        valid = np.convolve((~silent).astype(int), np.ones(10, int), mode="valid") == 10
        if np.any(valid):
            quiet_floor = max(1e-6, float(np.sqrt(average[valid].min())))
            # Ordinary AGC moves room tone by several dB. Lower the global
            # estimate only for a distinct quiet interval, not every AGC dip.
            if floor > 2 * quiet_floor:
                floor = quiet_floor
    high = max(floor * 2.3, float(rms.max()) * 0.025, 0.0007)
    low = max(floor * 1.55, high * 0.36)
    quiet = rms < floor * 1.35
    noise_power = np.median(power[quiet], axis=0) if np.any(quiet) else np.zeros(power.shape[1])
    result = []
    # Bridge 50 ms unvoiced gaps within crackles and noisy sustained gestures.
    regions = []
    for a, b in _runs(rms > low, gap=10):
        parts = [(a + c, a + d) for c, d in _runs(rms[a:b] > low)
                 if d - c >= 2 and float(rms[a + c:a + d].max()) >= high]
        # Two isolated rapid pops are distinct gestures. Many close grains are
        # left as a crackle rather than turned into an avalanche of accents.
        if (len(parts) == 2 and all(d - c <= 20 for c, d in parts)
                and parts[1][0] - parts[0][0] >= 9
                and parts[1][0] - parts[0][1] >= 3):
            regions.extend(parts)
        else:
            regions.append((a, b))
    for a, b in regions:
        if float(rms[a:b].max()) < high:
            continue
        if b - a < 2:
            continue
        kind, breath, peak, duration = _event_features(rms, power, freq, a, b, noise_power)
        if breath:
            continue
        # AAC padding followed by AGC settling can expose the initial room bed
        # as a short apparent pop. Require contrast against the settled bed at
        # this boundary; a real isolated sample-zero attack still has a tail.
        if times[a] < 0.075 and duration < 0.15 and b + 20 < len(rms):
            following = float(np.median(rms[b:b + 20]))
            if peak < 2.5 * following:
                continue
        # Early energy crossing is stable for impulses and deliberately precedes
        # a rising whoosh's RMS maximum. Short attacks compensate for RMS
        # centering; swells backtrack a little instead.
        threshold = max(floor * 1.85, peak * 0.18, 0.0007)
        crossing = np.flatnonzero(rms[a:b] >= threshold)
        if not len(crossing):
            continue
        correction = 0.003 if duration < 0.12 else -0.003
        onset = max(0.0, float(times[a + int(crossing[0])]) + correction)
        # Absolute recording level: no per-take peak normalization.
        strength = float(np.clip((20 * np.log10(max(peak, 1e-8)) + 60) / 54, 0, 1))
        result.append({"onset": onset, "end": float(times[min(b, len(times) - 1)]),
                       "strength": strength, "kind": kind,
                       "sustained": duration >= 0.20 and kind != "pop",
                       "a": a, "b": b, "peak": peak})
    return result, floor


def detect_cues(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    fps: int = 24,
    audio_offset_ms: float = 0.0,
    max_hits: int = 8,
) -> dict:
    """Return envelope, chronologically sorted hits/strengths/kinds and stop_frame.

    Nonfinite audio samples are treated as silence; input is never modified.
    Events outside [0, n_frames) are discarded, never clamped to the edges.
    At most max_hits distinct frame positions are returned, strongest first
    for selection and then restored to time order. max_hits=0 suppresses hits
    while retaining the continuous envelope. Invalid dimensions/rates/counts
    raise ValueError. A silent or empty input returns an all-zero envelope.
    """
    sr = _integer(sr, "sr", 1)
    n_frames = _integer(n_frames, "n_frames", 0)
    max_hits = _integer(max_hits, "max_hits", 0)
    fps = _integer(fps, "fps", 1)
    offset = float(audio_offset_ms) / 1000
    if not np.isfinite(offset):
        raise ValueError("audio_offset_ms must be finite")
    envelope = np.zeros(n_frames, np.float32)
    empty = {"envelope": envelope, "hits": [], "strengths": [], "kinds": [], "stop_frame": None}
    x = _audio(audio, sr)
    if not n_frames or len(x) < 2 or float(np.max(np.abs(x))) < 1e-8:
        return empty
    times, rms, power, freq = _features(x)
    events, floor = _events(times, rms, power, freq)
    if not events:
        return empty
    # Envelope powers are integrated over video intervals, not point-sampled;
    # a 3 ms pop between frame centers must remain audible to the synthesizer.
    active = np.zeros(len(rms), bool)
    for event in events:
        active[event["a"]:event["b"]] = True
    levels = np.sqrt(np.maximum(rms * rms - floor * floor, 0))
    levels = np.clip((20 * np.log10(np.maximum(levels, 1e-8)) + 60) / 54, 0, 1)
    levels[~active] = 0
    frames = np.floor((times - offset) * fps).astype(np.int64)
    inside = (frames >= 0) & (frames < n_frames)
    np.maximum.at(envelope, frames[inside], levels[inside].astype(np.float32))
    by_frame = {}
    for event in events:
        time = event["onset"] - offset
        if time < 0 or time >= n_frames / fps:
            continue
        frame = min(n_frames - 1, int(math.floor(time * fps + 0.5)))
        # The synth gates accents by envelope at each hit's interval.
        envelope[frame] = max(envelope[frame], event["strength"] * 0.18)
        if frame not in by_frame or event["strength"] > by_frame[frame]["strength"]:
            by_frame[frame] = event
    kept = sorted(sorted(by_frame, key=lambda f: (-by_frame[f]["strength"], f))[:max_hits])
    stop = None
    if any(e["sustained"] for e in events):
        ending = max(e["end"] for e in events) - offset
        # A cue sounding at the recording edge has no observed end.
        if max(e["b"] for e in events) < len(rms) - 2 and 0 < ending <= n_frames / fps:
            stop = min(n_frames, max(0, int(math.ceil(ending * fps))))
            envelope[stop:] = 0
    return {"envelope": envelope, "hits": kept,
            "strengths": [by_frame[f]["strength"] for f in kept],
            "kinds": [by_frame[f]["kind"] for f in kept], "stop_frame": stop}


def measure_clap_offset(audio, sr, video_flash_frame, fps) -> float:
    """Audio clap attack minus flash time, in ms (positive = captured audio late).

    Use a calibration take with one isolated clap synchronized to a visible
    flash/contact. Among broadband transients choose the strongest attack;
    this is not intended to identify a clap inside a sound-effects montage.
    Raises ValueError when no isolated attack can be measured.
    """
    fps = _integer(fps, "fps", 1)
    frame = float(video_flash_frame)
    if not np.isfinite(frame) or frame < 0:
        raise ValueError("video_flash_frame must be finite and nonnegative")
    x = _audio(audio, sr)
    if len(x) < 2:
        raise ValueError("no clap attack found")
    # One ms RMS permits calibration independent of 5 ms event features.
    width = max(1, _SR // 1000)
    rms = np.sqrt(np.maximum(_uniform_filter1d(x * x, width), 0))
    floor = max(float(np.quantile(rms, 0.20)), 1e-7)
    peaks, props = _find_peaks(np.pad(rms, (1, 1)), height=max(0.003, floor * 5),
                               prominence=max(0.003, floor * 4), distance=int(0.005 * _SR))
    peaks -= 1
    if not len(peaks):
        raise ValueError("no clap attack found")
    candidates = []
    for peak, prominence in zip(peaks, props["prominences"]):
        lo = max(0, int(peak) - int(0.06 * _SR))
        threshold = max(floor * 2.5, rms[peak] * 0.15)
        below = np.flatnonzero(rms[lo:peak] < threshold)
        onset = lo + int(below[-1]) + 1 if len(below) else lo
        # A clap is broadband. A loud room hum's gated attack or oscillation
        # must not outrank a quieter real clap during calibration.
        segment = x[max(0, int(peak) - 240):min(len(x), int(peak) + 480)]
        spectrum = np.abs(np.fft.rfft(segment, n=1024)) ** 2
        frequencies = np.fft.rfftfreq(1024, 1 / _SR)
        spectrum[frequencies < 70] = 0
        total = float(spectrum.sum()) + 1e-30
        broadband = (spectrum[frequencies > 1500].sum() > 0.10 * total
                     and spectrum.max() < 0.45 * total)
        if broadband and (peak - onset) / _SR <= 0.035:
            candidates.append((float(prominence), onset))
    if not candidates:
        raise ValueError("no isolated clap attack found")
    onset = max(candidates)[1]
    return float(1000 * (onset / _SR - frame / fps))
