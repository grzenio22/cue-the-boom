import { expect, it } from "vitest";
import { pickMime } from "./pickMime";

it("prefers mp4, falls back to webm, then null", () => {
  expect(pickMime((t) => t === "video/webm")).toBe("video/webm");
  expect(pickMime(() => false)).toBeNull();
  expect(pickMime(() => true)).toBe("video/mp4;codecs=avc1.42E01E,mp4a.40.2");
});
