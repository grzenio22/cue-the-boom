import { expect, it } from "vitest";
import { contentRect, normalizedPoint } from "./geometry";

it("portrait video letterboxed in a square box", () => {
  expect(contentRect({ width: 400, height: 400 }, 1080, 1920)).toEqual({ x: 87.5, y: 0, width: 225, height: 400 });
  expect(normalizedPoint(200, 300, { width: 400, height: 400 }, 1080, 1920)).toEqual([0.5, 0.75]);
  expect(normalizedPoint(10, 10, { width: 400, height: 400 }, 1080, 1920)).toBeNull();
});
