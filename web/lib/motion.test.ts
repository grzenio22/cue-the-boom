import { afterEach, describe, expect, it, vi } from "vitest";
import { peek, peekHandlers } from "./motion";

/** A <video> as far as peek touches it. */
function clip(paused = true) {
  const v = { paused, muted: false, loop: false, play: vi.fn(() => Promise.resolve()), pause: vi.fn() };
  return v as unknown as HTMLVideoElement & { play: ReturnType<typeof vi.fn>; pause: ReturnType<typeof vi.fn> };
}
const motion = (reduce: boolean) => vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: reduce })));

afterEach(() => vi.unstubAllGlobals());

describe("peek", () => {
  it("plays a tile's clip muted and looping while pointed at, and stops it where it is when the pointer leaves", () => {
    motion(false);
    const v = clip();
    peek(v, true);
    expect(v.play).toHaveBeenCalledOnce();
    expect(v.muted).toBe(true);
    expect(v.loop).toBe(true);
    (v as { paused: boolean }).paused = false;
    peek(v, false);
    expect(v.pause).toHaveBeenCalledOnce();
  });

  it("leaves the clip still when the viewer asked for less motion", () => {
    motion(true);
    const v = clip();
    peek(v, true);
    expect(v.play).not.toHaveBeenCalled();
    expect(v.muted).toBe(false);
  });

  it("does nothing without a clip, and survives a browser whose play() returns nothing", () => {
    motion(false);
    expect(() => peek(null, true)).not.toThrow();
    const v = clip();
    v.play.mockReturnValue(undefined);
    expect(() => peek(v, true)).not.toThrow();
  });

  it("peeks for a mouse, not for touch (a tap opens the tile)", () => {
    motion(false);
    const v = clip();
    const tile = { querySelector: () => v } as unknown as Element;
    peekHandlers.onPointerEnter({ pointerType: "touch", currentTarget: tile });
    expect(v.play).not.toHaveBeenCalled();
    peekHandlers.onPointerEnter({ pointerType: "mouse", currentTarget: tile });
    expect(v.play).toHaveBeenCalledOnce();
  });
});
