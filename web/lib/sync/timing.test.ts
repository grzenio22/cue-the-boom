import { describe, expect, it } from "vitest";
import { fitMediaClock } from "./mediaClock";
import { buildTimingJson, estimateOffsetS, plausibleOffset } from "./timing";

it("timing json places sample 0 on the plate timeline", () => {
  const clock = fitMediaClock([{ perfMs: 5000, mediaTime: 1 }, { perfMs: 6000, mediaTime: 2 }]);
  const t = buildTimingJson({ clock, firstFrame: 48000, sampleRate: 48000, ctxToPerfOffsetMs: 3500,
    inputLatencyS: 0.02, onsetSInNoise: 0.8, faceOffsetS: 0.0312, emissionPoint: [0.51234, 0.7] });
  expect(t).toEqual({ version: 1, plate_fps: 24, sample_rate: 48000, noise_offset_s: 0.5,
    input_latency_s: 0.02, onset_s_in_noise: 0.8, face_offset_s: 0.0312, emission_point: [0.5123, 0.7] });
});
it("mic starting before the plate gives a negative offset", () => {
  const clock = fitMediaClock([{ perfMs: 5000, mediaTime: 0 }, { perfMs: 6000, mediaTime: 1 }]);
  const t = buildTimingJson({ clock, firstFrame: 0, sampleRate: 48000, ctxToPerfOffsetMs: 4800,
    inputLatencyS: 0.02, onsetSInNoise: null, faceOffsetS: null, emissionPoint: null });
  expect(t.noise_offset_s).toBe(-0.2);
});
it("finds a 250 ms delay between two recordings of the same noise", () => {
  const sr = 48000, ref = new Float32Array(sr * 3);
  for (let i = sr; i < sr + 6000; i++) ref[i] = 0.6 * Math.sin(i * 0.09) * Math.exp(-(i - sr) / 2000);
  const other = new Float32Array(sr * 3); other.set(ref.subarray(0, sr * 3 - 12000), 12000);
  expect(Math.abs(estimateOffsetS(ref, other, sr)! - 0.25)).toBeLessThan(0.001);
  expect(estimateOffsetS(ref, new Float32Array(sr * 3), sr)).toBeNull();
});
it("finds a negative lag when the face recording started later (its sounds come earlier in it)", () => {
  const sr = 48000, other = new Float32Array(sr * 3);
  for (let i = sr; i < sr + 6000; i++) other[i] = 0.6 * Math.sin(i * 0.09) * Math.exp(-(i - sr) / 2000);
  const ref = new Float32Array(sr * 3); ref.set(other.subarray(0, sr * 3 - 4800), 4800); // ref is 100 ms later
  expect(Math.abs(estimateOffsetS(ref, other, sr)! + 0.1)).toBeLessThan(0.001);
});

describe("plausibleOffset", () => {
  it("keeps a fitted offset that makes sense", () => expect(plausibleOffset(-0.31, -0.25)).toEqual({ offsetS: -0.31, fitted: true }));
  it("falls back to the stopwatch when a clock was misread", () =>
    expect(plausibleOffset(-11.6, -0.4)).toEqual({ offsetS: -0.4, fitted: false }));
});
