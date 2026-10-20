import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Again from "./Again";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
let reducedMotion: boolean;
let running: Set<HTMLMediaElement>;
const shot = (again: number) => ({ id: `take-${again}`, again, files: { "final.mp4": `/take-${again}.mp4` } });
const gallery = () => [shot(2), { id: "other", files: { "final.mp4": "/other.mp4" } }, shot(0), shot(1)];

beforeEach(() => {
  reducedMotion = false;
  running = new Set();
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => gallery() }));
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: reducedMotion })));
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
const show = async () => { await act(async () => { root.render(<Again />); }); };

describe("Again", () => {
  it("shows the three performances in order and starts them together, muted", async () => {
    await show();
    expect(videos().map((v) => v.getAttribute("src"))).toEqual(["/take-0.mp4", "/take-1.mp4", "/take-2.mp4"]);
    expect([...host.querySelectorAll("figcaption")].map((c) => c.textContent)).toEqual(["BOOM", "pf, pf, BOOM", "pf pf pf PF"]);
    expect(videos().every((v) => running.has(v) && v.muted)).toBe(true);
  });

  it("waits for a press when the viewer asked for less motion", async () => {
    reducedMotion = true;
    await show();
    expect(running.size).toBe(0);
    await act(async () => { button("Play together").click(); });
    expect(running.size).toBe(3);
  });

  it("plays each with its sound in turn, then all three together again", async () => {
    await show();
    await act(async () => { button("Hear them one by one").click(); });
    expect(videos().map((v) => running.has(v))).toEqual([true, false, false]);
    expect(videos()[0].muted).toBe(false);
    for (const i of [0, 1]) {
      await act(async () => { videos()[i].dispatchEvent(new Event("ended")); });
      expect(running.has(videos()[i + 1]) && !videos()[i + 1].muted).toBe(true);
    }
    await act(async () => { videos()[2].dispatchEvent(new Event("ended")); });
    expect(videos().every((v) => running.has(v) && v.muted)).toBe(true);
  });

  it("says so when the performances are missing", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => [shot(0)] } as Response);
    await show();
    expect(host.textContent).toContain("not available");
  });
});
