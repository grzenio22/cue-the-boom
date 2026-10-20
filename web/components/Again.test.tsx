import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Again from "./Again";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
let reducedMotion: boolean;
let running: Set<HTMLMediaElement>;
let frame: FrameRequestCallback | undefined;
const shot = (again: number) => ({ id: `take-${again}`, again, files: { "final.mp4": `/take-${again}.mp4` } });
const gallery = () => [shot(2), { id: "other", files: { "final.mp4": "/other.mp4" } }, shot(0), shot(1)];

beforeEach(() => {
  reducedMotion = true;
  running = new Set();
  frame = undefined;
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => gallery() }));
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: reducedMotion, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
  vi.stubGlobal("requestAnimationFrame", vi.fn((callback: FrameRequestCallback) => { frame = callback; return 1; }));
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(function (this: HTMLMediaElement) {
    running.add(this);
    return Promise.resolve();
  });
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(function (this: HTMLMediaElement) { running.delete(this); });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const videos = () => [...host.querySelectorAll("video")];
const button = (text: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === text)!;
const tap = async (element: HTMLElement) => { await act(async () => { element.click(); }); };
const emit = async (element: HTMLElement, event: string) => { await act(async () => { element.dispatchEvent(new Event(event)); }); };

async function show() {
  await act(async () => { root.render(<Again />); });
  for (const video of videos()) {
    Object.defineProperty(video, "readyState", { value: 4, configurable: true, writable: true });
    await emit(video, "canplay");
  }
}

describe("Again", () => {
  it("orders the three numeric performances and explains the single render", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => [
      ...gallery(), { ...shot(0), again: "0", id: "string-index" }, shot(4),
    ] } as Response);
    await show();
    expect(videos().map((v) => v.getAttribute("src"))).toEqual(["/take-0.mp4", "/take-1.mp4", "/take-2.mp4"]);
    expect([...host.querySelectorAll("figcaption")].map((c) => c.textContent)).toEqual(["BOOM", "pf, pf, BOOM", "pf pf pf PF"]);
    expect(host.textContent).toContain("1 generation, 3 performances, 0 new renders");
    expect(host.textContent).toContain("The same LTX-2.5 render on the author's hand, timed three times by three takes of their voice: one generation, three performances, no new render, no GPU.");
    expect(host.querySelector("a")?.getAttribute("href")).toBe("#studio");
  });

  it("respects reduced motion and lets Play start all three muted", async () => {
    await show();
    expect(running.size).toBe(0);
    await tap(button("Play"));
    expect(running.size).toBe(3);
    expect(videos().every((v) => v.muted)).toBe(true);
  });

  it("starts the muted group only after all three clips can play", async () => {
    reducedMotion = false;
    await act(async () => { root.render(<Again />); });
    for (const video of videos()) Object.defineProperty(video, "readyState", { value: 4, configurable: true });
    await emit(videos()[0], "canplay");
    await emit(videos()[1], "canplay");
    expect(running.size).toBe(0);
    await emit(videos()[2], "canplay");
    expect(running.size).toBe(3);
    expect(videos().every((v) => v.muted)).toBe(true);
  });

  it("pauses the group and aligns its clocks when resumed", async () => {
    await show();
    await tap(button("Play"));
    await tap(button("Pause"));
    expect(running.size).toBe(0);
    videos()[0].currentTime = 2;
    videos()[1].currentTime = 0.4;
    await tap(button("Play"));
    expect(videos().map((v) => v.currentTime)).toEqual([2, 2, 2]);
    videos()[1].currentTime = 0.6;
    act(() => frame?.(100));
    expect(videos()[1].currentTime).toBe(2);
  });

  it("stops the whole group if playback is refused and allows a retry", async () => {
    await show();
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(new Error("blocked"));
    await tap(button("Play"));
    expect(running.size).toBe(0);
    expect(host.querySelector('[role="status"]')?.textContent).toContain("Press Play");
    await tap(button("Play"));
    expect(running.size).toBe(3);
  });

  it("ignores an older playback failure after switching to listening", async () => {
    await show();
    let rejectOld: (reason: Error) => void = () => {};
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(() => new Promise<void>((_resolve, reject) => { rejectOld = reject; }));
    await tap(button("Play"));
    await tap(button("Hear them one by one"));
    await act(async () => { rejectOld(new Error("old playback cancelled")); });
    expect([...running]).toEqual([videos()[0]]);
    expect(host.querySelector('[role="status"]')?.textContent).toBe("Hearing performance 1 of 3.");
  });

  it("keeps the newest group playback after an earlier paused attempt rejects", async () => {
    await show();
    let rejectOld: (reason: Error) => void = () => {};
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(() => new Promise<void>((_resolve, reject) => { rejectOld = reject; }));
    await tap(button("Play"));
    await tap(button("Pause"));
    await tap(button("Play"));
    await act(async () => { rejectOld(new Error("cancelled while paused")); });
    expect(running.size).toBe(3);
    expect(host.querySelector('[role="status"]')?.textContent).toBe("Playing together, muted.");
  });

  it("lets a gesture load media before canplay without later interrupting listening", async () => {
    reducedMotion = false;
    await act(async () => { root.render(<Again />); });
    expect(button("Hear them one by one").disabled).toBe(false);
    await tap(button("Hear them one by one"));
    for (const video of videos()) {
      Object.defineProperty(video, "readyState", { value: 4, configurable: true });
      await emit(video, "canplay");
    }
    expect([...running]).toEqual([videos()[0]]);
    expect(videos()[0].muted).toBe(false);
  });

  it("hears each performance in order, with only one audible at a time", async () => {
    await show();
    await tap(button("Hear them one by one"));
    for (let index = 0; index < 3; index++) {
      expect(videos().map((v) => v.muted)).toEqual([0, 1, 2].map((i) => i !== index));
      expect([...running]).toEqual([videos()[index]]);
      await emit(videos()[index], "ended");
    }
    expect(running.size).toBe(0);
    expect(videos().every((v) => v.muted)).toBe(true);
    expect(button("Hear them one by one")).toBeDefined();
  });

  it("retains the next audible performance when the browser asks for another gesture", async () => {
    await show();
    await tap(button("Hear them one by one"));
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(new Error("needs gesture"));
    await emit(videos()[0], "ended");
    expect(running.size).toBe(0);
    expect(host.querySelector('[role="status"]')?.textContent).toContain("Press Play");
    await tap(button("Play"));
    expect([...running]).toEqual([videos()[1]]);
    expect(videos()[1].muted).toBe(false);
  });

  it("waits for a buffering peer and respects Pause while waiting", async () => {
    await show();
    await tap(button("Play"));
    Object.defineProperty(videos()[1], "readyState", { value: 2, configurable: true, writable: true });
    await emit(videos()[1], "waiting");
    expect(running.size).toBe(0);
    Object.defineProperty(videos()[1], "readyState", { value: 4, configurable: true });
    await emit(videos()[1], "canplay");
    expect(running.size).toBe(3);
    await emit(videos()[1], "waiting");
    await tap(button("Pause"));
    await emit(videos()[1], "canplay");
    expect(running.size).toBe(0);
  });

  it("restarts all three together when a preview reaches its end", async () => {
    await show();
    await tap(button("Play"));
    videos().forEach((video) => { video.currentTime = 5; });
    await emit(videos()[1], "ended");
    expect(videos().map((video) => video.currentTime)).toEqual([0, 0, 0]);
    expect(running.size).toBe(3);
  });

  it("pauses and mutes saved video elements on unmount", async () => {
    await show();
    await tap(button("Hear them one by one"));
    const saved = videos();
    act(() => root.unmount());
    expect(running.size).toBe(0);
    expect(saved.every((video) => video.muted)).toBe(true);
  });

  it("shows an unavailable state for an incomplete gallery", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => [shot(0), shot(1)] } as Response);
    await show();
    expect(videos()).toHaveLength(0);
    expect(host.textContent).toContain("The three performances are unavailable right now.");
  });

  it("keeps the studio link available when the gallery request fails", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error("offline"));
    await show();
    expect(host.textContent).toContain("The three performances are unavailable right now.");
    expect(host.querySelector("a")?.getAttribute("href")).toBe("#studio");
  });
});
