import { describe, expect, it } from "vitest";
import { BUZZ_HF, buzzOf, channel, ride, rng, streamStep } from "./lightning";

describe("channel", () => {
  it("runs from where it is struck to where it strikes, jagged, the same for the same seed", () => {
    const c = channel([0, 0], [100, 0], rng(3));
    expect(c.pts[0]).toEqual([0, 0]);
    expect(c.pts[c.pts.length - 1]).toEqual([100, 0]);
    expect(c.pts.length).toBe(65); // six levels of midpoints
    expect(Math.max(...c.pts.map(([, y]) => Math.abs(y)))).toBeGreaterThan(2); // it kinks
    expect(channel([0, 0], [100, 0], rng(3))).toEqual(c);
    expect(channel([0, 0], [100, 0], rng(4))).not.toEqual(c);
  });

  it("rides the hand at its root and holds where it struck", () => {
    const c = channel([0, 0], [100, 0], rng(5));
    const r = ride(c, [10, 20]);
    expect(r.pts[0]).toEqual([10, 20]);
    expect(r.pts[r.pts.length - 1]).toEqual([100, 0]);
    r.forks.forEach((f) => expect(f.pts[0]).toEqual(r.pts[f.at]));
  });
});

describe("streamStep", () => {
  it("strikes anew when its time is up or on a hit, and rides the hand between", () => {
    let s = streamStep({ ch: null, left: 0, aim: 0, seed: 1 }, [0, 0], 0, 100, 2, false, 1);
    const first = s.ch!;
    expect(s.left).toBeGreaterThanOrEqual(1);
    s = { ...s, left: 2 };
    const held = streamStep(s, [5, 0], 0, 100, 2, false, 2);
    expect(held.ch!.pts[held.ch!.pts.length - 1]).toEqual(first.pts[first.pts.length - 1]); // far end held
    expect(streamStep(s, [5, 0], 0, 100, 2, true, 2).ch).not.toEqual(held.ch); // a hit strikes it anew
  });
});

describe("buzzOf", () => {
  const sr = 48000, n = 2048, bins = 1024;
  const spectrumOf = (x: Float32Array) => { // a crude dB spectrum, as the analyser gives
    const out = new Float32Array(bins);
    for (let k = 0; k < bins; k++) {
      let re = 0, im = 0;
      for (let i = 0; i < n; i += 2) { re += x[i] * Math.cos((2 * Math.PI * k * i) / n); im -= x[i] * Math.sin((2 * Math.PI * k * i) / n); }
      out[k] = 10 * Math.log10(re * re + im * im + 1e-12);
    }
    return out;
  };
  const noise = (seed: number, hi: boolean) => { // white, or only high (a first difference, twice)
    const r = rng(seed), x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = r() - 0.5;
    if (hi) for (let pass = 0; pass < 2; pass++) for (let i = n - 1; i > 0; i--) x[i] -= x[i - 1];
    return x;
  };

  it("hears a pitch with a high hiss on it as a buzz, a hiss alone as no pitch", () => {
    const hiss = noise(1, true);
    const buzz = new Float32Array(n);
    for (let i = 0; i < n; i++) buzz[i] = 0.3 * Math.sin((2 * Math.PI * 140 * i) / sr) + 0.5 * hiss[i];
    const b = buzzOf(spectrumOf(buzz), buzz, sr);
    expect(b.voiced).toBe(true);
    expect(b.hf).toBeGreaterThan(BUZZ_HF);
    expect(buzzOf(spectrumOf(hiss), hiss, sr).voiced).toBe(false); // an "fffff": no pitch
    const tone = new Float32Array(n).map((_, i) => 0.3 * Math.sin((2 * Math.PI * 140 * i) / sr));
    expect(buzzOf(spectrumOf(tone), tone, sr).hf).toBeLessThan(BUZZ_HF); // a hum: a pitch, no hiss
  });
});
