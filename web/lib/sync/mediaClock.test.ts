import { expect, it, vi } from "vitest";
import { fitMediaClock, sampleClock } from "./mediaClock";

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
