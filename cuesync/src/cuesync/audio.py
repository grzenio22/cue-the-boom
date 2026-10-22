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
ACTIVE_OVER_FLOOR = 10 ** (10 / 20)  # a frame is "performing" when it is 10 dB over the room tone
MERGE_GAP_FRAMES = 6  # pauses shorter than 250 ms are breaths inside one gesture; longer ones end it
MIN_ACTIVE_FRAMES = 2
GESTURE_RANGE_DB = 20.0  # a frame belongs to the performance within this range of the main gesture's peak
PHRASE_RANGE_DB = 12.0  # a later phrase must come this close to the main gesture's peak...
PHRASE_MAX_GAP_FRAMES = 24  # ...start within a second of the previous one, and carry its own onset


class NoOnsetError(ValueError):
    """The recording is silent or contains no detectable onset."""


@dataclass(frozen=True)
class Hit:
    frame: int  # plate frame of the hit (the loudest onset of its group)
    strength: float  # absolute envelope peak right after it (0..1, -60..-6 dBFS)
    start: int  # first onset of its group
    end: int  # where the next hit starts (or the end of the take)


@dataclass
class CueSheet:
    fps: int
    offset_s: float  # plate time of audio sample 0
    duration_s: float  # audio duration
    onsets_s: list[float]  # plate time, ascending
    hero_onset_s: float  # plate time
    envelope: np.ndarray  # per plate frame, absolute: -60 dBFS -> 0 ... -6 dBFS -> 1
    envelope_rel: np.ndarray  # envelope / max(envelope), shape only
    peak_frame: int  # loudest plate frame (unclipped level, so a rising sound peaks at its top)
    extra: dict = field(default_factory=dict)
    # [start, end) plate frames where the performer is actually making the noise. None = derive from the envelope.
    active_intervals: list[tuple[int, int]] | None = None

    def frame_of(self, t_s: float) -> int:
        return int(round(t_s * self.fps))

    def intervals(self) -> list[tuple[int, int]]:
        if self.active_intervals is not None:
            return list(self.active_intervals)
        return active_intervals(np.asarray(self.envelope) > 0)

    @property
    def voice_end_frame(self) -> int | None:
        """First plate frame after the performance (end of the last active interval)."""
        iv = self.intervals()
        return iv[-1][1] if iv else None

    def hits(self, min_gap: int = 6, max_hits: int = 4, range_db: float = 14.0) -> list["Hit"]:
        """The distinct hits of a rhythmic performance ("pf, pf, KA-BOOM"): onsets closer than ``min_gap``
        frames are one hit (anchored on its loudest onset), and only hits within ``range_db`` of the loudest
        count (a breath or a click is not a hit). At most ``max_hits``, the loudest kept, in time order."""
        env = np.asarray(self.envelope, dtype=np.float64)
        frames = sorted({self.frame_of(t) for t in self.onsets_s if 0 <= self.frame_of(t) < env.size})
        if not frames:
            return []
        groups: list[list[int]] = [[frames[0]]]
        for f in frames[1:]:
            (groups[-1].append(f) if f - groups[-1][-1] < min_gap else groups.append([f]))
        level = lambda f: float(env[f: f + 6].max()) if f < env.size else 0.0  # noqa: E731
        hits = []
        for k, g in enumerate(groups):
            end = groups[k + 1][0] if k + 1 < len(groups) else env.size
            peak = g[0] + int(np.argmax(env[g[0]: max(end, g[0] + 1)]))  # "KA-BOOM": the hit is the BOOM
            anchor = max((f for f in g if f <= peak), default=g[0])
            hits.append(Hit(frame=anchor, strength=max(level(f) for f in g), start=g[0], end=end))
        loudest = max(h.strength for h in hits)
        keep = [h for h in hits if h.strength >= loudest - range_db / 54.0]  # the envelope spans 54 dB
        # the loudest are kept; at equal loudness the longer one (the "KA-BOOM", not a "pf")
        keep = sorted(sorted(keep, key=lambda h: (-h.strength, -(h.end - h.start)))[:max_hits], key=lambda h: h.frame)
        return keep

    def for_hit(self, hit: "Hit") -> "CueSheet":
        """This performance as if it were only ``hit``: its onset is the hero, the envelope is silent outside
        it, and it is active from its start for as long as it sounds."""
        env = np.asarray(self.envelope, dtype=np.float64).copy()
        lo, hi = max(0, hit.start - 1), min(env.size, hit.end)
        env[:lo], env[hi:] = 0.0, 0.0
        sounding = [(a, b) for a, b in active_intervals(env > 0, merge_gap=3, min_len=1) if b > hit.start - 2]
        active = [(max(sounding[0][0], hit.start), sounding[0][1])] if sounding else [(hit.start, hit.start + 2)]
        peak = float(env.max()) or 1.0
        return CueSheet(fps=self.fps, offset_s=self.offset_s, duration_s=self.duration_s,
                        onsets_s=[hit.frame / self.fps], hero_onset_s=hit.frame / self.fps, envelope=env,
                        envelope_rel=env / peak, peak_frame=int(np.argmax(env)), extra=dict(self.extra),
                        active_intervals=active)

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
            "active_intervals": [[int(a), int(b)] for a, b in self.intervals()],
            "voice_end_frame": self.voice_end_frame,
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
            active_intervals=[(int(a), int(b)) for a, b in d["active_intervals"]] if "active_intervals" in d else None,
        )


def active_intervals(active: np.ndarray, merge_gap: int = MERGE_GAP_FRAMES,
                     min_len: int = MIN_ACTIVE_FRAMES) -> list[tuple[int, int]]:
    """[start, end) runs of True, with short gaps bridged and blips dropped."""
    runs: list[list[int]] = []
    for i, on in enumerate(np.asarray(active, dtype=bool)):
        if not on:
            continue
        if runs and i - runs[-1][1] < merge_gap:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return [(a, b) for a, b in runs if b - a >= min_len]


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
    peak_rms = float(band_rms.max())
    # Room tone = the quietest 100 ms of REAL audio: recorder start-up zeros or a dropout must not pass for
    # an impossibly quiet room (every breath would then count as performing). A take whose whole background
    # is digital silence (synthetic tests) has no real room tone; it keeps the old estimate.
    zeros = np.cumsum(np.concatenate([[0], (np.concatenate([np.zeros(N_FFT), x, np.zeros(N_FFT)]) == 0)]))
    starts = np.arange(len(band_rms)) * HOP
    zero_frac = (zeros[starts + N_FFT] - zeros[starts]) / N_FFT
    real = inside & (zero_frac < 0.25)
    if (real & (band_rms < 0.5 * peak_rms)).sum() < 10:
        real = inside
    power_in = band_rms[real] ** 2
    k = max(1, min(10, power_in.size))  # quietest 100 ms of the take = its room tone
    noise_floor = max(float(np.sqrt(np.convolve(power_in, np.ones(k) / k, mode="valid").min())), 1e-6)
    if peak_rms < 1e-3:
        raise NoOnsetError("your take is silent: no sound reached the microphone (is it muted, or another input picked?)")
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
                # A real hit is a level RISE (+6 dB), not a flux ripple inside a sustained sound. Use the
                # full attack window: stopping early made tiny AAC changes erase a still-rising phone cue.
                if before.size and band_rms[j: j + attack].mean() < RISE_FACTOR * max(float(before.mean()), noise_floor):
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
    frame_rms = np.zeros(n_plate)
    frame_ids = np.floor(plate_t * fps).astype(int)
    valid = (frame_ids >= 0) & (frame_ids < n_plate) & (centres >= 0) & (centres <= duration_s)
    has_room_tone = bool(onset_hops) and noise_floor * ONSET_OVER_FLOOR <= peak_rms
    gate = ENV_GATE * noise_floor if has_room_tone else 0.0
    for k in np.unique(frame_ids[valid]):
        rms = np.sqrt(np.mean(band_rms[valid & (frame_ids == k)] ** 2))
        frame_rms[k] = rms
        if rms < gate:  # within 6 dB of the room tone: nobody is performing, so plumes can fade out
            continue
        level = 20.0 * np.log10(max(rms * math.sqrt(2.0), 1e-6))
        env[k] = min(max((level - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0), 1.0)
    onsets_plate = [t + offset_s for t in onset_times]
    hero_frame = int(round(onsets_plate[hero_idx] * fps))
    intervals = _performance_intervals(frame_rms, env, noise_floor, peak_rms, has_room_tone, hero_frame,
                                       [int(round(t * fps)) for t in onsets_plate])
    env_max = env.max()
    env_rel = env / env_max if env_max > 0 else env.copy()

    return CueSheet(
        fps=fps,
        offset_s=float(offset_s),
        duration_s=float(duration_s),
        onsets_s=onsets_plate,
        hero_onset_s=onsets_plate[hero_idx],
        envelope=env,
        envelope_rel=env_rel,
        peak_frame=int(np.argmax(frame_rms)),  # unclipped: env saturates at -6 dBFS and would report the first loud frame
        active_intervals=intervals,
    )


def _performance_intervals(frame_rms: np.ndarray, env: np.ndarray, noise_floor: float, peak_rms: float,
                           has_room_tone: bool, hero_frame: int, onset_frames: list[int]) -> list[tuple[int, int]]:
    """When the performer is actually making the noise, from the main gesture (around the hero onset) on.

    Levels are judged against the performance, not just the room: a frame counts when it is 10 dB over the
    room tone AND within 20 dB of the main gesture's peak, so an inhale, an exhale or the phone being put
    down doesn't extend the performance. A later phrase counts only if it comes within 12 dB of the main
    gesture, starts within a second of the previous phrase and carries its own onset (a breath fades in).
    """
    if not has_room_tone:  # a sound that fills the whole take: fall back to the gated envelope
        return active_intervals(env > 0)
    lo, hi = max(0, hero_frame - 12), min(len(frame_rms), hero_frame + 13)
    gesture_peak = float(frame_rms[lo:hi].max()) if hi > lo else peak_rms
    level = max(ACTIVE_OVER_FLOOR * noise_floor, gesture_peak * 10 ** (-GESTURE_RANGE_DB / 20), 0.01 * peak_rms)
    runs = active_intervals(frame_rms >= level)
    main = next((iv for iv in runs if iv[1] > hero_frame - 2), None)
    if main is None:
        return runs[-1:] if runs else []
    # A soft lead-in ("fff" before "WOOOM") belongs to the gesture: extend its start back through frames
    # still clearly above the room, but never before the onset (a breath before it stays out).
    later = runs[runs.index(main) + 1:]
    start, floor_level = main[0], ACTIVE_OVER_FLOOR * noise_floor
    while start - 1 >= max(0, hero_frame - 2) and frame_rms[start - 1] >= floor_level:
        start -= 1
    kept = [(start, main[1])]
    for a, b in later:
        loud = frame_rms[a:b].max() >= gesture_peak * 10 ** (-PHRASE_RANGE_DB / 20)
        near = a - kept[-1][1] <= PHRASE_MAX_GAP_FRAMES
        struck = any(a - 1 <= f < b for f in onset_frames)
        if not (loud and near and struck):
            break
        kept.append((a, b))
    return kept
