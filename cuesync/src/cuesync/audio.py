"""Vocal-performance analysis: band-limited onsets and an absolute-loudness envelope per video frame.

numpy only. Spectral features use bins below 11 kHz (some phone browsers low-pass the mic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

N_FFT = 1024
HOP = 480  # 10 ms at 48 kHz
F_MIN = 45.0  # ignore DC and rumble
F_MAX = 11000.0
SILENCE_OVER_FLOOR_DB = 15.0  # a take must peak this far above its room tone, or it's "near-silent"
ONSET_OVER_FLOOR = 4.0  # x room tone (+12 dB) within 50 ms of an onset
RISE_FACTOR = 2.0  # +6 dB over the preceding ~50 ms
ENV_GATE = 2.0  # frames within 6 dB of the room tone count as silence (envelope 0)
DB_FLOOR, DB_CEIL = -60.0, -6.0  # envelope 0 .. 1


class NoOnsetError(ValueError):
    """The recording is silent or contains no detectable onset."""


@dataclass
class CueSheet:
    fps: int
    offset_s: float  # plate time of audio sample 0
    duration_s: float  # audio duration
    onsets_s: list[float]  # plate time, ascending
    hero_onset_s: float  # plate time
    envelope: np.ndarray  # per plate frame, absolute: -60 dBFS -> 0 ... -6 dBFS -> 1
    envelope_rel: np.ndarray  # envelope / max(envelope), shape only
    peak_frame: int  # argmax(envelope)
    extra: dict = field(default_factory=dict)

    def frame_of(self, t_s: float) -> int:
        return int(round(t_s * self.fps))

    def to_json(self) -> dict:
        r = lambda v: round(float(v), 4)  # noqa: E731
        return {
            "fps": self.fps,
            "offset_s": r(self.offset_s),
            "duration_s": r(self.duration_s),
            "onsets_s": [r(t) for t in self.onsets_s],
            "hero_onset_s": r(self.hero_onset_s),
            "envelope": [r(v) for v in self.envelope],
            "envelope_rel": [r(v) for v in self.envelope_rel],
            "peak_frame": int(self.peak_frame),
        }

    @classmethod
    def from_json(cls, d: dict) -> "CueSheet":
        return cls(
            fps=int(d["fps"]),
            offset_s=float(d["offset_s"]),
            duration_s=float(d["duration_s"]),
            onsets_s=[float(t) for t in d["onsets_s"]],
            hero_onset_s=float(d["hero_onset_s"]),
            envelope=np.asarray(d["envelope"], dtype=np.float64),
            envelope_rel=np.asarray(d["envelope_rel"], dtype=np.float64),
            peak_frame=int(d["peak_frame"]),
        )


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _taper_edges(x: np.ndarray, sr: int, fade_s: float = 0.005) -> np.ndarray:
    """Raised-cosine fade at both ends. The zero padding would otherwise add an artificial digital
    step whose broadband leakage looks like an onset (a real mic never starts with a hard step).
    5 ms is shorter than one 10 ms hop, so a genuine onset at sample 0 is still detected."""
    n = min(int(round(fade_s * sr)), x.size // 2)
    if n < 2:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    y = x.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def _spectral_frames(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (kept-bin magnitudes [frames, bins], hop centre times in audio seconds, window)."""
    x = _taper_edges(x.astype(np.float64), sr)
    pad = np.concatenate([np.zeros(N_FFT, np.float64), x, np.zeros(N_FFT, np.float64)])
    n_frames = 1 + (len(pad) - N_FFT) // HOP
    window = _hann(N_FFT)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(pad[idx] * window, axis=1)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    keep = (freqs >= F_MIN) & (freqs < F_MAX)
    mag = np.abs(spec[:, keep])
    # frame j covers pad[j*HOP : j*HOP+N_FFT]; its centre in original-signal samples is j*HOP + N_FFT/2 - N_FFT
    centres = (np.arange(n_frames) * HOP + N_FFT / 2 - N_FFT) / sr
    return mag, centres, window


def _moving_median(x: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.pad(x, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=1)


def analyze(samples: np.ndarray, sr: int = 48000, *, offset_s: float = 0.0, fps: int = 24) -> CueSheet:
    """Analyze a mono performance recorded at ``sr``. ``offset_s`` is the plate time of sample 0."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise NoOnsetError("empty recording")
    mag, centres, window = _spectral_frames(x, sr)
    band_rms = np.sqrt(2.0 * np.sum(mag**2, axis=1) / (N_FFT * np.sum(window**2)))
    duration_s = x.size / sr
    inside = (centres >= 0) & (centres <= duration_s)
    power_in = band_rms[inside] ** 2
    k = max(1, min(10, power_in.size))  # quietest 100 ms of the take = its room tone
    noise_floor = max(float(np.sqrt(np.convolve(power_in, np.ones(k) / k, mode="valid").min())), 1e-6)
    peak_rms = float(band_rms.max())
    if peak_rms < 1e-3:
        raise NoOnsetError("recording is silent below 11 kHz")
    if peak_rms < 10 ** (SILENCE_OVER_FLOOR_DB / 20) * noise_floor and peak_rms * math.sqrt(2.0) < 10 ** (-30 / 20):
        # quiet AND flat: room tone or hum with nobody performing. (A loud steady sound that fills the
        # whole take is still accepted, so a performance that starts at sample 0 isn't rejected.)
        raise NoOnsetError(f"near-silent recording: peak only {20 * math.log10(peak_rms / noise_floor):.1f} dB over the room tone")

    # --- onsets: half-wave-rectified log-magnitude flux
    logmag = np.log1p(100.0 * mag)
    flux = np.zeros(len(logmag))
    flux[1:] = np.maximum(logmag[1:] - logmag[:-1], 0.0).sum(axis=1)
    hops_per_s = sr / HOP
    med = _moving_median(flux, int(round(0.5 * hops_per_s)) | 1)
    mad = float(np.median(np.abs(flux - np.median(flux))))
    thresh = med + 1.5 * mad
    floor = 0.05 * flux.max()  # robustness against numerical ripple when the background is digital silence
    min_gap = int(round(0.080 * hops_per_s))
    max_back = int(round(0.050 * hops_per_s))
    rms_gate = 0.02 * peak_rms
    attack = 6  # hops (~50 ms) to look into the attack, so a sample-0 onset still counts

    def candidates(relative_to_floor: bool) -> list[int]:
        hops: list[int] = []
        for j in range(1, len(flux)):
            lo, hi = max(0, j - 3), min(len(flux), j + 4)
            if flux[j] <= 0 or flux[j] < flux[lo:hi].max() or flux[j] <= thresh[j] or flux[j] < floor:
                continue
            level = band_rms[j: j + attack].max()
            if level <= rms_gate:
                continue
            if relative_to_floor:
                if level < ONSET_OVER_FLOOR * noise_floor:  # +12 dB over the room tone
                    continue
                before = band_rms[max(0, j - attack): max(0, j - 1)]
                # a real hit is a level RISE (+6 dB), not a flux ripple inside a sustained sound
                if before.size and band_rms[j: j + 4].mean() < RISE_FACTOR * max(float(before.mean()), noise_floor):
                    continue
            if hops and j - hops[-1] < min_gap:
                continue
            hops.append(j)
        return hops

    onset_hops = candidates(relative_to_floor=True)
    if not onset_hops:  # e.g. a steady sound filling the whole take: its "room tone" is the performance itself
        onset_hops = candidates(relative_to_floor=False)[:1]
    if not onset_hops:
        raise NoOnsetError("no onset passed the threshold")

    def backtrack(j: int) -> int:
        """Walk back (max 50 ms) to the last hop whose RMS is below 30% of the onset's peak RMS."""
        peak_rms = band_rms[j: j + int(round(0.15 * hops_per_s)) + 1].max()
        k = j
        while k > 0 and j - k < max_back and band_rms[k] >= 0.3 * peak_rms:
            k -= 1
        return k

    onset_times = [max(float(centres[backtrack(j)]), 0.0) for j in onset_hops]
    after = int(round(0.150 * hops_per_s))
    nxt = onset_hops[1:] + [len(band_rms)]
    # an onset may not claim the peak of the NEXT hit ("vrrrm... KA-BOOM": the BOOM is the hero)
    strengths = [band_rms[j: max(j + 1, min(j + after + 1, n))].max() for j, n in zip(onset_hops, nxt)]
    hero_idx = int(np.argmax(strengths))

    # --- envelope per plate frame (absolute dBFS scale)
    n_plate = max(1, int(math.ceil((offset_s + duration_s) * fps)))
    plate_t = centres + offset_s
    env = np.zeros(n_plate)
    frame_ids = np.floor(plate_t * fps).astype(int)
    valid = (frame_ids >= 0) & (frame_ids < n_plate) & (centres >= 0) & (centres <= duration_s)
    gate = ENV_GATE * noise_floor if onset_hops and noise_floor * ONSET_OVER_FLOOR <= peak_rms else 0.0
    for k in np.unique(frame_ids[valid]):
        rms = np.sqrt(np.mean(band_rms[valid & (frame_ids == k)] ** 2))
        if rms < gate:  # within 6 dB of the room tone: nobody is performing, so plumes can fade out
            continue
        level = 20.0 * np.log10(max(rms * math.sqrt(2.0), 1e-6))
        env[k] = min(max((level - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0), 1.0)
    env_max = env.max()
    env_rel = env / env_max if env_max > 0 else env.copy()

    onsets_plate = [t + offset_s for t in onset_times]
    return CueSheet(
        fps=fps,
        offset_s=float(offset_s),
        duration_s=float(duration_s),
        onsets_s=onsets_plate,
        hero_onset_s=onsets_plate[hero_idx],
        envelope=env,
        envelope_rel=env_rel,
        peak_frame=int(np.argmax(env)),
    )
