import { frameRms } from "../audio/onset";
import { type MediaClock, perfMsOfSample } from "./mediaClock";

/** Dub Mode timing.json, schema version 1. cuesync uses offset = noise_offset_s - input_latency_s. */
export interface TimingJson {
  version: 1;
  plate_fps: 24;
  sample_rate: number;
  noise_offset_s: number;
  input_latency_s: number;
  onset_s_in_noise: number | null;
  face_offset_s: number | null;
  emission_point: [number, number] | null;
}

const r4 = (v: number) => {
  const r = Math.round(v * 10000) / 10000;
  return Object.is(r, -0) ? 0 : r;
};

export function buildTimingJson(args: {
  clock: MediaClock;
  firstFrame: number;
  sampleRate: number;
  ctxToPerfOffsetMs: number;
  inputLatencyS: number;
  onsetSInNoise: number | null;
  faceOffsetS: number | null;
  emissionPoint: [number, number] | null;
}): TimingJson {
  const perf0 = perfMsOfSample(args.firstFrame, 0, args.sampleRate, args.ctxToPerfOffsetMs);
  return {
    version: 1,
    plate_fps: 24,
    sample_rate: args.sampleRate,
    noise_offset_s: r4(args.clock.mediaTimeAt(perf0)),
    input_latency_s: r4(args.inputLatencyS),
    onset_s_in_noise: args.onsetSInNoise === null ? null : r4(args.onsetSInNoise),
    face_offset_s: args.faceOffsetS === null ? null : r4(args.faceOffsetS),
    emission_point: args.emissionPoint === null ? null : [r4(args.emissionPoint[0]), r4(args.emissionPoint[1])],
  };
}

function normCorr(a: Float32Array, b: Float32Array, lag: number, from = 0, to = a.length): number {
  let dot = 0, ea = 0, eb = 0;
  const start = Math.max(from, -lag), end = Math.min(to, b.length - lag);
  for (let i = start; i < end; i++) {
    const x = a[i], y = b[i + lag];
    dot += x * y; ea += x * x; eb += y * y;
  }
  return ea > 0 && eb > 0 ? dot / Math.sqrt(ea * eb) : 0;
}

/**
 * Lag of `other` relative to `ref` in seconds: how much later a sound comes in `other` than in `ref` (positive =
 * `other` started recording earlier), or null when
 * either is silent or the normalised correlation peak is < 0.3. Coarse: 10 ms RMS envelopes within
 * +/-maxLagS; fine: sample-accurate correlation within +/-10 ms around the coarse lag (over a 2 s
 * window around the loudest part of `ref`, to keep it fast).
 */
export function estimateOffsetS(ref: Float32Array, other: Float32Array, sampleRate: number, maxLagS = 1.0): number | null {
  const frame = Math.max(1, Math.round(sampleRate * 0.01));
  const ea = frameRms(ref, frame), eb = frameRms(other, frame);
  const peak = (x: Float32Array) => x.reduce((m, v) => (v > m ? v : m), 0);
  if (peak(ea) < 1e-3 || peak(eb) < 1e-3) return null;
  const maxLag = Math.round(maxLagS / 0.01);
  let bestLag = 0, best = -Infinity;
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    const c = normCorr(ea, eb, lag);
    if (c > best) { best = c; bestLag = lag; }
  }
  if (best < 0.3) return null;
  let loud = 0;
  for (let i = 0; i < ea.length; i++) if (ea[i] > ea[loud]) loud = i;
  const from = Math.max(0, loud * frame - sampleRate), to = Math.min(ref.length, loud * frame + sampleRate);
  const coarse = bestLag * frame;
  let fineLag = coarse, fine = -Infinity;
  for (let lag = coarse - frame; lag <= coarse + frame; lag++) {
    const c = normCorr(ref, other, lag, from, to);
    if (c > fine) { fine = c; fineLag = lag; }
  }
  if (fine < 0.3) return null;
  return fineLag / sampleRate;
}

/** The recording starts just before the clip plays, so where its first sample falls on the clip is within a
 * second or so of 0. A fitted offset far outside that means a clock was misread: use the stopwatch estimate. */
export function plausibleOffset(fittedS: number, stopwatchS: number): { offsetS: number; fitted: boolean } {
  return fittedS >= -2 && fittedS <= 1 ? { offsetS: fittedS, fitted: true } : { offsetS: stopwatchS, fitted: false };
}
