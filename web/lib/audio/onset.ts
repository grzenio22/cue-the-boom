/** UI preview of the hit time (seconds into the recording). The server-side cuesync analysis is authoritative. */
export function frameRms(pcm: Float32Array, frameSize: number): Float32Array {
  const n = Math.floor(pcm.length / frameSize);
  const out = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let sum = 0;
    const start = f * frameSize;
    for (let i = start; i < start + frameSize; i++) sum += pcm[i] * pcm[i];
    out[f] = Math.sqrt(sum / frameSize);
  }
  return out;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function detectOnsetPCM(pcm: Float32Array, sampleRate: number): number | null {
  const frame = Math.max(1, Math.round(sampleRate * 0.01));
  const rms = frameRms(pcm, frame);
  let max = 0;
  for (let i = 0; i < rms.length; i++) if (rms[i] > max) max = rms[i];
  if (max < 1e-3) return null;
  const history = 20; // 200 ms of 10 ms frames
  for (let f = 0; f < rms.length; f++) {
    const prev = Array.from(rms.subarray(Math.max(0, f - history), f));
    const base = Math.max(median(prev), 1e-4);
    if (rms[f] >= 0.1 * max && rms[f] >= 3 * base) return (f * frame) / sampleRate;
  }
  return null;
}
