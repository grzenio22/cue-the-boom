import { expect, it } from "vitest";
import { applyAndVerify, AUDIO_CONSTRAINTS, videoConstraints } from "./constraints";

type Track = Parameters<typeof applyAndVerify>[0];

const fake = (kind: string, settings: MediaTrackSettings, reject = false) => ({
  kind, getSettings: () => settings,
  applyConstraints: () => (reject ? Promise.reject(new Error("nope")) : Promise.resolve()),
});
it("audio ok when all processing off", async () => {
  const r = await applyAndVerify(fake("audio", { echoCancellation: false, noiseSuppression: false, autoGainControl: false }) as unknown as Track, AUDIO_CONSTRAINTS);
  expect(r.ok).toBe(true);
});
it("flags iOS-style forced echo cancellation", async () => {
  const r = await applyAndVerify(fake("audio", { echoCancellation: true, noiseSuppression: false, autoGainControl: false }) as unknown as Track, AUDIO_CONSTRAINTS);
  expect(r.ok).toBe(false); expect(r.mismatches).toContain("echoCancellation");
});
it("flags low-light frame rate drop and applyConstraints rejection", async () => {
  const r = await applyAndVerify(fake("video", { frameRate: 15 }, true) as unknown as Track, videoConstraints("environment"));
  expect(r.mismatches.some((m) => m.startsWith("applyConstraints rejected"))).toBe(true);
  expect(r.mismatches).toContain("frameRate");
});
it("exact audio constraints, never a hard sample rate", () => {
  expect(AUDIO_CONSTRAINTS).toEqual({ echoCancellation: false, noiseSuppression: false, autoGainControl: false,
    channelCount: 1, sampleRate: { ideal: 48000 } });
});
