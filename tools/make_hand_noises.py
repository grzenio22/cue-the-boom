"""The sample noises for the elements made from a hand (lightning, water, air, the explosion), synthesised like the
other sample noises so anyone can cue them with no microphone: python tools/make_hand_noises.py [-o samples/voices]

- zzzzz-tsst.wav: a voiced buzz (a pitch with a hiss on it, the hiss strong above 4 kHz), then a sharp "tsst"
- boooom.wav: a long, rolling "BOOOOM", unbroken for most of a second, falling away from its strike
- shhhhh-sploosh.wav: a held "shhhhh", then a burst "PSSHH" on which the water is thrown
- hoooo.wav: a breathy voiced "hoooo" that swells and lets go
- hoo-hoo-HOO.wav: three breathy hits, the last the loudest

Each is 48 kHz mono 16-bit, from a fixed seed: the same bytes every time.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SR = 48000


def band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``x`` with only lo..hi Hz kept (a soft-edged band in the frequency domain)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / SR)
    edge = 0.15 * (hi - lo)
    gain = np.clip((f - lo + edge) / edge, 0, 1) * np.clip((hi + edge - f) / edge, 0, 1)
    return np.fft.irfft(spec * gain, n=x.size)


def voiced(pitch: np.ndarray, formants: list[tuple[float, float]], tilt: float = 1.2) -> np.ndarray:
    """A glottal source on ``pitch`` (Hz per sample), its harmonics falling off as 1/k**tilt, shaped by formants
    (centre Hz, bandwidth Hz) as resonance peaks over the harmonics."""
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    out = np.zeros_like(pitch)
    for k in range(1, 60):
        fk = k * pitch
        g = sum(1.0 / (1.0 + ((fk - c) / bw) ** 2) for c, bw in formants) + 0.05
        out += np.where(fk < SR / 2 - 500, g / k ** tilt, 0.0) * np.sin(k * phase)
    return out


def envelope(n: int, points: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear gain over ``n`` samples through (seconds, gain) points."""
    t = np.arange(n) / SR
    xs, ys = zip(*points)
    return np.interp(t, xs, ys)


def norm(x: np.ndarray, peak: float = 0.85) -> np.ndarray:
    return x * (peak / max(1e-9, float(np.abs(x).max())))


def zzzzz_tsst(rng: np.random.Generator) -> np.ndarray:
    n = int(2.4 * SR)
    t = np.arange(n) / SR
    pitch = 135 + 6 * np.sin(2 * np.pi * 5.2 * t) + 10 * np.clip(t - 1.0, 0, 1)  # a little vibrato, rising as it pushes
    buzz = voiced(pitch, [(300, 120), (1500, 300)])
    hiss = band(rng.standard_normal(n), 4200, 10000)
    hiss *= 0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(pitch) / SR) ** 2  # a voiced fricative: the hiss pulses with the pitch
    z = 0.45 * norm(buzz) + 0.9 * norm(hiss)
    z *= envelope(n, [(0, 0), (0.06, 0.8), (0.7, 0.9), (1.3, 1.0), (1.6, 0.0), (2.4, 0.0)])
    tsst = norm(band(rng.standard_normal(n), 3000, 11000), 1.0) * envelope(n, [(0, 0), (1.72, 0), (1.73, 1.1), (1.9, 0.4),
                                                                             (2.05, 0.0), (2.4, 0)])
    return norm(z + tsst)


def boooom(rng: np.random.Generator) -> np.ndarray:
    n = int(2.4 * SR)
    t = np.arange(n) / SR
    pitch = 105 - 25 * np.clip(t / 1.4, 0, 1)
    body = voiced(pitch, [(320, 90), (800, 160)], tilt=1.5)
    body *= envelope(n, [(0, 0), (0.25, 0), (0.27, 1.0), (0.45, 0.8), (0.8, 0.45), (1.1, 0.2), (1.35, 0.0), (2.4, 0)])
    b = band(rng.standard_normal(n), 200, 3000) * envelope(n, [(0, 0), (0.25, 0), (0.255, 1.2), (0.29, 0.0), (2.4, 0)])
    breath = band(rng.standard_normal(n), 150, 900) * envelope(n, [(0, 0), (0.27, 0), (0.4, 0.12), (1.5, 0.05),
                                                                   (1.8, 0), (2.4, 0)])
    return norm(norm(body) + 0.35 * norm(b) + breath)


def shhhhh_sploosh(rng: np.random.Generator) -> np.ndarray:
    n = int(2.8 * SR)
    shh = band(rng.standard_normal(n), 1800, 6500) * envelope(n, [(0, 0), (0.15, 0.55), (1.0, 0.6), (1.5, 0.65),
                                                                   (1.6, 0.0), (2.8, 0)])
    burst = band(rng.standard_normal(n), 700, 9000) * envelope(n, [(0, 0), (1.78, 0), (1.79, 1.5), (1.9, 0.9),
                                                                    (2.2, 0.25), (2.45, 0.0), (2.8, 0)])
    return norm(shh + burst)


def hoooo(rng: np.random.Generator) -> np.ndarray:
    n = int(2.6 * SR)
    t = np.arange(n) / SR
    pitch = 160 + 4 * np.sin(2 * np.pi * 4.5 * t) - 10 * np.clip(t - 1.2, 0, 1)
    voice = voiced(pitch, [(330, 110), (780, 150)], tilt=1.8)
    breath = band(rng.standard_normal(n), 200, 1800)
    x = 0.8 * norm(voice) + 0.5 * norm(breath)
    return norm(x * envelope(n, [(0, 0), (0.3, 0.5), (1.0, 0.85), (1.6, 1.0), (1.9, 0.0), (2.6, 0)]))


def hoo_hoo_hoo(rng: np.random.Generator) -> np.ndarray:
    n = int(2.8 * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for start, gain in ((0.35, 0.55), (1.0, 0.7), (1.65, 1.0)):
        pitch = 175 - 30 * np.clip((t - start) / 0.35, 0, 1)
        voice = voiced(pitch, [(340, 120), (820, 160)], tilt=1.7)
        breath = band(rng.standard_normal(n), 250, 2500)
        e = envelope(n, [(0, 0), (start, 0), (start + 0.02, 1.0), (start + 0.12, 0.7), (start + 0.32, 0.0), (2.8, 0)])
        out += gain * (0.7 * norm(voice) + 0.6 * norm(breath)) * e
    return norm(out)


NOISES = {"zzzzz-tsst.wav": zzzzz_tsst, "boooom.wav": boooom, "shhhhh-sploosh.wav": shhhhh_sploosh,
          "hoooo.wav": hoooo, "hoo-hoo-HOO.wav": hoo_hoo_hoo}


def write(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="samples/voices")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, make) in enumerate(NOISES.items()):
        write(out / name, make(np.random.default_rng(100 + i)))
        print(out / name)


if __name__ == "__main__":
    main()
