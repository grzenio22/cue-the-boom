import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { JobStatus } from "@/lib/shot";
import Studio from "./Studio";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
let submitted: FormData[];
const scrollIntoView = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollIntoView");
const job: JobStatus = {
  id: "candle-shot", status: "done", stage: "done", progress: 1, message: "Done", error: null,
  mode: "scene", cached: true, files: { "final.mp4": "/candle-shot.mp4" },
  receipt: {
    mode: "scene", plate: { still: true, point: [0.5, 0.5] }, inputs: { cue: "clip" },
    render: { seconds: 1, renderer: "precomputed", prompt: "candle", model: "LTX-2.5" },
    measured: { stop_latency_frames: null, stops_with_voice: false },
    voice: { hero_onset_s: 0.5, voice_end_frame: null, active_intervals: [[0.5, 0.6]] },
    guide: { length: 48, tail: "fade", keyframes: [12], anchor_frame: 12 }, seconds_total: 1,
  },
};

beforeEach(() => {
  submitted = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    let data: unknown;
    switch (url) {
      case "/api/health":
        data = { ok: true, renderer: null, modes: ["instant"], made_ahead: ["scene"], queued: 0 };
        break;
      case "/api/effects":
        data = [{ id: "fireball", fluid: "fire", behaviour: "burst", stops_with_voice: false, scene: true }];
        break;
      case "/api/samples":
        data = {
          plates: [{ name: "candle_snap.mp4", url: "/candle.mp4", title: "Candle", point: [0.5, 0.5], sound: true,
            scene: { "pop.wav": "fireball" } }],
          voices: [{ name: "pop.wav", url: "/pop.wav", title: "Pop", effect: "fireball" }],
        };
        break;
      case "/api/jobs":
        submitted.push(init!.body as FormData);
        data = job;
        break;
      default:
        throw new Error(`Unexpected request: ${url}`);
    }
    return { ok: true, status: 200, json: async () => data } as Response;
  }));
  vi.stubGlobal("EventSource", class extends EventTarget {
    constructor() {
      super();
      queueMicrotask(() => this.dispatchEvent(new MessageEvent("end", { data: JSON.stringify(job) })));
    }
    close() {}
  });
  vi.spyOn(HTMLMediaElement.prototype, "load").mockImplementation(() => {});
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  if (scrollIntoView) Object.defineProperty(HTMLElement.prototype, "scrollIntoView", scrollIntoView);
  else Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
});

const button = (text: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === text)!;
const click = async (text: string) => { await act(async () => { button(text).click(); }); };

describe("Studio replay cues", () => {
  it("limits Perform it again to replay cues after making an offline sample from its own sound", async () => {
    await act(async () => { root.render(<Studio />); });
    await click("Candle");
    expect(button("The clip's own sound").disabled).toBe(false);
    await click("The clip's own sound");
    await click("Lock it to my clip's sound");
    await click("Make the shot");
    expect(submitted).toHaveLength(1);
    expect(submitted[0].get("cue")).toBe("clip");
    expect(submitted[0].get("plate_sample")).toBe("candle_snap.mp4");
    expect(submitted[0].get("mode")).toBe("scene");

    await click("Perform it again");
    expect(button("Say it").getAttribute("aria-checked")).toBe("true");
    expect(button("The clip's own sound").disabled).toBe(true);
    expect(button("Use my microphone").disabled).toBe(false);
    expect(button("Press and hold").disabled).toBe(false);
  });
});
