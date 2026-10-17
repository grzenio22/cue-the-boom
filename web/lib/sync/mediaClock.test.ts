import { expect, it, vi } from "vitest";
import { fitMediaClock, fitMediaClockNear, sampleClock } from "./mediaClock";

it("fits a 1x playback clock exactly", () => {
  const s = [0, 1, 2, 3].map((k) => ({ perfMs: 1000 + k * 41.6667, mediaTime: 0.25 + k / 24 }));
  const c = fitMediaClock(s);
  expect(c.mediaTimeAt(1000)).toBeCloseTo(0.25, 4); expect(c.residualMs).toBeLessThan(0.01);
});
it("rejects fewer than 2 samples", () => {
  expect(() => fitMediaClock([{ perfMs: 0, mediaTime: 0 }])).toThrow();
});
it("falls back to rAF when requestVideoFrameCallback is missing", () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const raf = vi.spyOn(globalThis, "requestAnimationFrame").mockImplementation(() => 1);
  const video = { currentTime: 0.5 } as unknown as HTMLVideoElement;
  const stop = sampleClock(video, () => {});
  expect(warn).toHaveBeenCalledWith("clock_source: raf");
  stop(); warn.mockRestore(); raf.mockRestore();
});
it("uses rVFC media time and expected display time when available", () => {
  const got: { perfMs: number; mediaTime: number }[] = [];
  type Cb = (now: number, md: { mediaTime: number; expectedDisplayTime?: number }) => void;
  let cb: Cb = () => {};
  const cancel = vi.fn();
  const video = { requestVideoFrameCallback: (f: Cb) => { cb = f; return 7; }, cancelVideoFrameCallback: cancel } as unknown as HTMLVideoElement;
  const stop = sampleClock(video, (s) => got.push(s));
  cb(100, { mediaTime: 0.125, expectedDisplayTime: 116 });
  expect(got[0]).toEqual({ perfMs: 116, mediaTime: 0.125 });
  stop(); expect(cancel).toHaveBeenCalled();
});

it("reads the plate's clock near the noise, whatever a stall did when playback started", () => {
  const s = [];
  for (let p = 0; p <= 500; p += 40) s.push({ perfMs: 1000 + p, mediaTime: 0 }); // stalled at the first frame
  for (let p = 540; p <= 3000; p += 40) s.push({ perfMs: 1000 + p, mediaTime: (p - 540) / 1000 }); // then plays
  const near = fitMediaClockNear(s, 1700); // a noise just after playback got going
  expect(near.mediaTimeAt(1700)).toBeCloseTo(0.16, 2);
  expect(Math.abs(fitMediaClock(s).mediaTimeAt(1700) - 0.16)).toBeGreaterThan(0.05); // the old whole-take line
});
