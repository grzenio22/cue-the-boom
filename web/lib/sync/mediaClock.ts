/** Map performance.now() time to the plate's media time, from rVFC (or rAF fallback) samples. */

export interface ClockSample {
  perfMs: number;
  mediaTime: number;
}

export interface MediaClock {
  mediaTimeAt(perfMs: number): number;
  residualMs: number;
}

export function fitMediaClock(samples: ClockSample[]): MediaClock {
  if (samples.length < 2) throw new Error("need at least 2 clock samples");
  const n = samples.length;
  let mx = 0, my = 0;
  for (const s of samples) { mx += s.perfMs; my += s.mediaTime; }
  mx /= n; my /= n;
  let sxx = 0, sxy = 0;
  for (const s of samples) { sxx += (s.perfMs - mx) ** 2; sxy += (s.perfMs - mx) * (s.mediaTime - my); }
  if (sxx === 0) throw new Error("clock samples span no time");
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let sse = 0;
  for (const s of samples) sse += (intercept + slope * s.perfMs - s.mediaTime) ** 2;
  return {
    mediaTimeAt: (perfMs: number) => intercept + slope * perfMs,
    residualMs: Math.sqrt(sse / n) * 1000,
  };
}

/** The plate's clock near one moment (the noise): media time = performance time + c, with c the median over the
 * frames presented within ``halfWindowMs`` of it (playback runs at rate 1). A line fitted to the whole take is
 * thrown off by a stall when playback starts, and extrapolating it back to the noise put takes up to 0.9 s off. */
export function fitMediaClockNear(samples: ClockSample[], perfMs: number, halfWindowMs = 400): MediaClock {
  if (samples.length === 0) throw new Error("need clock samples");
  let near = samples.filter((s) => Math.abs(s.perfMs - perfMs) <= halfWindowMs);
  if (near.length < 3) near = [...samples].sort((a, b) => Math.abs(a.perfMs - perfMs) - Math.abs(b.perfMs - perfMs)).slice(0, 8);
  const c = median(near.map((s) => s.mediaTime - s.perfMs / 1000));
  return {
    mediaTimeAt: (p: number) => p / 1000 + c,
    residualMs: median(near.map((s) => Math.abs(s.mediaTime - s.perfMs / 1000 - c))) * 1000,
  };
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function perfMsOfSample(firstFrame: number, sampleIndex: number, sampleRate: number, ctxToPerfOffsetMs: number): number {
  return ((firstFrame + sampleIndex) / sampleRate) * 1000 + ctxToPerfOffsetMs;
}

type RvfcVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, md: { mediaTime: number; expectedDisplayTime?: number }) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

/** Sample (display time, media time) per presented frame. Returns a stop function. */
export function sampleClock(video: HTMLVideoElement, onSample: (s: ClockSample) => void): () => void {
  const v = video as RvfcVideo;
  if (typeof v.requestVideoFrameCallback === "function") {
    let stopped = false;
    let id = 0;
    const cb = (now: number, md: { mediaTime: number; expectedDisplayTime?: number }) => {
      if (stopped) return;
      onSample({ perfMs: md.expectedDisplayTime ?? now, mediaTime: md.mediaTime });
      id = v.requestVideoFrameCallback!(cb);
    };
    id = v.requestVideoFrameCallback(cb);
    return () => { stopped = true; v.cancelVideoFrameCallback?.(id); };
  }
  console.warn("clock_source: raf");
  let raf = 0;
  let stopped = false;
  const loop = () => {
    if (stopped) return;
    onSample({ perfMs: performance.now(), mediaTime: video.currentTime });
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  return () => { stopped = true; cancelAnimationFrame(raf); };
}
