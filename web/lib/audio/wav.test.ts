import { expect, it } from "vitest";
import { encodeWav } from "./wav";

it("wav header and length", () => {
  const u = encodeWav(new Float32Array([0, 1, -1, 0.5]), 48000);
  expect(new TextDecoder().decode(u.slice(0, 4))).toBe("RIFF");
  expect(u.length).toBe(44 + 4 * 2);
  expect(new DataView(u.buffer).getUint32(24, true)).toBe(48000);
  expect(new DataView(u.buffer).getInt16(44 + 2, true)).toBe(32767);
});
