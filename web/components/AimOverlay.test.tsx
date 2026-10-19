import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AimOverlay, { aimAngle, type AimValue, carried, type HandTrack, lostRuns, lostShare, trackedAim, turnAngle } from "./AimOverlay";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom lays nothing out: the box is `size` px, and the ResizeObserver reports it at once
let size = { w: 360, h: 640 };
vi.stubGlobal("ResizeObserver", class {
  constructor(private cb: (e: { contentRect: { width: number; height: number } }[]) => void) {}
  observe() { this.cb([{ contentRect: { width: size.w, height: size.h } }]); }
  disconnect() {}
});
HTMLElement.prototype.getBoundingClientRect = () => ({ left: 0, top: 0, x: 0, y: 0, width: size.w, height: size.h,
  right: size.w, bottom: size.h, toJSON: () => ({}) });
Element.prototype.setPointerCapture = () => {};

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  size = { w: 360, h: 640 };
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); });

/** A portrait 1080 x 1920 track: the hand in the middle on the hit frame (2), pointing right. */
const portrait = (src = ["mp", "mp", "mp", "mp", "mp"]): HandTrack => ({
  W: 1080, H: 1920, src,
  xy: [[300, 1200], [400, 1100], [540, 960], [600, 900], [650, 860]],
  dir: [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
});

function show(props: Partial<Parameters<typeof AimOverlay>[0]> = {}) {
  const onChange = vi.fn<(v: AimValue) => void>();
  act(() => root.render(<AimOverlay src="clip.mp4" frame={2} track={portrait()} value={null} onChange={onChange} {...props} />));
  return { onChange, box: host.querySelector<HTMLElement>("[role=slider]")! };
}

const pointer = (el: Element, type: string, x: number, y: number, init: PointerEventInit = {}) =>
  act(() => { el.dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y, pointerId: 1, ...init })); });
const key = (el: Element, k: string, init: KeyboardEventInit = {}) =>
  act(() => { el.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, ...init })); });
const last = (fn: { mock: { calls: unknown[][] } }) => fn.mock.calls[fn.mock.calls.length - 1][0];

describe("angles", () => {
  it("snap a drag to the nearest of 8 directions, clockwise from up", () => {
    expect(aimAngle(0, -10)).toBe(0);
    expect(aimAngle(10, 1)).toBe(90);
    expect(aimAngle(-10, 0)).toBe(-90);
    expect(aimAngle(6, 7)).toBe(135);
    expect(aimAngle(0, 10)).toBe(180);
  });
  it("keep any angle, to the degree, when free", () => expect(aimAngle(6, 7, true)).toBe(139));
  it("turn on to the next direction from any angle", () => {
    expect(turnAngle(90, 1)).toBe(135);
    expect(turnAngle(100, -1)).toBe(90);
    expect(turnAngle(100, 1)).toBe(135);
    expect(turnAngle(180, 1)).toBe(-135);
    expect(turnAngle(-135, -1)).toBe(180);
  });
  it("read the track's own aim on the hit frame", () => expect(trackedAim(portrait(), 2)).toEqual({ frame: 2, x: 0.5, y: 0.5, angle: 90 }));
});

// in a 360 x 640 box the portrait clip fills it: the seed at (180, 320), the arrow 108 px long, its head at (288, 320)
describe("AimOverlay", () => {
  it("starts on the tracked emitter, along the tracked direction", () => {
    const { box } = show();
    expect(box.getAttribute("aria-valuenow")).toBe("90");
    expect(box.getAttribute("aria-valuetext")).toBe("shoots right, from 50% across and 50% down");
  });

  it("snaps a dragged head to the 8 directions", () => {
    const { box, onChange } = show();
    pointer(box, "pointerdown", 288, 320);
    expect(onChange).not.toHaveBeenCalled(); // taking hold of the head changes nothing
    pointer(box, "pointermove", 180 + 60, 320 + 70);
    pointer(box, "pointerup", 180 + 60, 320 + 70);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(last(onChange)).toEqual({ frame: 2, x: 0.5, y: 0.5, angle: 135 });
  });

  it("keeps any angle with Alt held", () => {
    const { box, onChange } = show();
    pointer(box, "pointerdown", 288, 320);
    pointer(box, "pointermove", 180 + 60, 320 + 70, { altKey: true });
    expect(last(onChange)).toEqual({ frame: 2, x: 0.5, y: 0.5, angle: 139 });
    show({ value: last(onChange) as AimValue });
    expect(box.getAttribute("aria-valuetext")).toBe("shoots about down and right (139°), from 50% across and 50% down");
  });

  it("aims at a tap away from the arrow, in one tap", () => {
    const { box, onChange } = show();
    pointer(box, "pointerdown", 180, 100);
    pointer(box, "pointerup", 180, 100);
    expect(onChange.mock.calls).toEqual([[{ frame: 2, x: 0.5, y: 0.5, angle: 0 }]]);
  });

  it("moves the seed when the ring is dragged, keeping the angle", () => {
    const { box, onChange } = show();
    pointer(box, "pointerdown", 185, 325); // a finger lands a little off the ring's middle
    pointer(box, "pointermove", 185 + 36, 325 - 64);
    expect(last(onChange)).toEqual({ frame: 2, x: 0.6, y: 0.4, angle: 90 });
  });

  it("draws and drags the performer's own aim over the track's", () => {
    const { box, onChange } = show({ value: { frame: 2, x: 0.25, y: 0.75, angle: -90 } });
    expect(box.getAttribute("aria-valuetext")).toBe("shoots left, from 25% across and 75% down");
    pointer(box, "pointerdown", 90 - 108, 480); // its head
    pointer(box, "pointermove", 90, 480 - 100);
    expect(last(onChange)).toEqual({ frame: 2, x: 0.25, y: 0.75, angle: 0 });
  });

  it("turns 45 degrees an arrow key, moves the seed with Shift, and goes back to the track with Home", () => {
    const { box, onChange } = show({ value: { frame: 2, x: 0.3, y: 0.3, angle: 100 } });
    key(box, "ArrowRight");
    expect(last(onChange)).toEqual({ frame: 2, x: 0.3, y: 0.3, angle: 135 });
    key(box, "ArrowLeft");
    expect(last(onChange)).toEqual({ frame: 2, x: 0.3, y: 0.3, angle: 90 });
    key(box, "ArrowUp", { shiftKey: true });
    expect(last(onChange)).toEqual({ frame: 2, x: 0.3, y: 0.29, angle: 100 });
    key(box, "Home");
    expect(last(onChange)).toEqual({ frame: 2, x: 0.5, y: 0.5, angle: 90 });
  });

  it("works on a landscape clip", () => {
    size = { w: 640, h: 360 };
    const track: HandTrack = { W: 1920, H: 1080, src: ["mp"], xy: [[1440, 270]], dir: [[0, -1]] };
    const { box, onChange } = show({ frame: 0, track });
    expect(box.style.aspectRatio).toBe("1920 / 1080");
    pointer(box, "pointerdown", 480, 90);
    pointer(box, "pointermove", 480 + 64, 90 + 36);
    expect(last(onChange)).toEqual({ frame: 0, x: 0.85, y: 0.35, angle: 0 });
  });

  it("maps through the letterbox of a portrait clip in a wide box (object-fit: contain)", () => {
    size = { w: 640, h: 360 }; // the clip shows 202.5 px wide, from x = 218.75
    const { box, onChange } = show();
    pointer(box, "pointerdown", 320, 180);
    pointer(box, "pointermove", 320 + 20.25, 180);
    expect(last(onChange)).toEqual({ frame: 2, x: 0.6, y: 0.5, angle: 90 });
  });

  it("with nothing tracked, a press is the seed and a drag from it the way", () => {
    const { box, onChange } = show({ track: undefined });
    const video = host.querySelector("video")!;
    Object.defineProperty(video, "videoWidth", { value: 1080 });
    Object.defineProperty(video, "videoHeight", { value: 1920 });
    act(() => { video.dispatchEvent(new Event("loadeddata")); });
    pointer(box, "pointerdown", 90, 160);
    pointer(box, "pointermove", 90 - 50, 160);
    expect(onChange.mock.calls).toEqual([[{ frame: 2, x: 0.25, y: 0.25, angle: 0 }], [{ frame: 2, x: 0.25, y: 0.25, angle: -90 }]]);
  });

  it("draws lost frames in red, with the count under the clip", () => {
    const src = ["mp", "lost", "lost", "interp", "mp"];
    expect(lostRuns(src)).toEqual([[1, 2]]);
    show({ track: portrait(src), frame: 4 });
    const red = host.querySelectorAll('polyline[stroke="#ff5d5d"]');
    expect(red).toHaveLength(1);
    expect(red[0].getAttribute("points")!.split(" ")).toHaveLength(4); // 0 -> 1, held at 2, then the jump to 3
    expect(host.querySelectorAll('polyline[stroke-dasharray="3 4"]')).toHaveLength(1); // 3 -> 4, filled in
    expect(host.querySelectorAll('circle[stroke="#ff5d5d"]')).toHaveLength(1);
    expect(host.querySelectorAll('rect[fill="#ff5d5d"]')).toHaveLength(1);
    expect(host.textContent).toContain("Lost the hand on 2 frames (red)");
  });

  it("says so when the hand is lost on the hit frame itself", () => {
    show({ track: portrait(["mp", "mp", "lost", "mp", "mp"]) });
    expect(host.textContent).toContain("Lost the hand on the hit frame: drag the ring onto it.");
  });

  it("draws no red for a clean track", () => {
    show();
    expect(host.querySelectorAll('[stroke="#ff5d5d"], [fill="#ff5d5d"]')).toHaveLength(0);
    expect(host.textContent).not.toContain("Lost");
  });
});

describe("carried", () => {
  it("keeps the tap where it is on the tapped frame and moves, turns and scales it with the hand", () => {
    const t: HandTrack = { W: 100, H: 100, xy: [[50, 50], [60, 50]], dir: [[0, -1], [1, 0]], scale: [10, 20], src: ["mp", "mp"] };
    const c = carried(t, 0, [0.5, 0.3]); // 2 hand sizes along the hand from its point on frame 0 (up)
    expect(c.xy[0][0]).toBeCloseTo(50);
    expect(c.xy[0][1]).toBeCloseTo(30);
    expect(c.xy[1][0]).toBeCloseTo(100); // frame 1: the hand points right, twice the size: 2 x 20 px along it
    expect(c.xy[1][1]).toBeCloseTo(50);
  });
  it("turns the direction by the aim on every frame, as the shot does", () => {
    const t: HandTrack = { W: 100, H: 100, xy: [[50, 50], [50, 50]], dir: [[0, 1], [1, 0]], src: ["mp", "mp"] };
    const c = carried(t, 0, [0.5, 0.5], 0); // the hand points down on frame 0; aimed up: half a turn
    expect(c.dir[0][0]).toBeCloseTo(0);
    expect(c.dir[0][1]).toBeCloseTo(-1);
    expect(c.dir[1][0]).toBeCloseTo(-1); // frame 1: the hand points right, so the aim points left
    expect(c.dir[1][1]).toBeCloseTo(0);
  });
  it("says how much of the clip the hand was lost on", () => {
    expect(lostShare({ W: 1, H: 1, xy: [], dir: [], src: ["mp", "lost", "lost", "hold"] })).toBe(0.5);
  });
});
