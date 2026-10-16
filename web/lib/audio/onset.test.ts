import { expect, it } from "vitest";
import { detectOnsetPCM } from "./onset";

it("onset at 0.5 s within 10 ms", () => {
  const sr = 48000, x = new Float32Array(sr * 2);
  for (let i = sr / 2; i < sr / 2 + 4800; i++) x[i] = 0.7 * Math.sin(i * 0.12);
  expect(Math.abs(detectOnsetPCM(x, sr)! - 0.5)).toBeLessThanOrEqual(0.01);
});
it("silence gives null", () => {
  expect(detectOnsetPCM(new Float32Array(48000), 48000)).toBeNull();
});
it("ignores steady background hiss before the hit", () => {
  const sr = 48000, x = new Float32Array(sr * 2);
  let seed = 1;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
  for (let i = 0; i < x.length; i++) x[i] = 0.004 * rnd();
  for (let i = sr; i < sr + 4800; i++) x[i] += 0.6 * Math.sin(i * 0.1);
  expect(Math.abs(detectOnsetPCM(x, sr)! - 1.0)).toBeLessThanOrEqual(0.01);
});
