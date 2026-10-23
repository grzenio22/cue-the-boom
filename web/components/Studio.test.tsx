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
      case "/api/gallery":
        data = [];
        break;
      case "/api/samples":
        data = {
          plates: [{ name: "candle_snap.mp4", url: "/candle.mp4", title: "Candle", point: [0.5, 0.5], sound: true,
            scene: { "pop.wav": "fireball" } },
            { name: "hand_palm.mp4", url: "/palm.mp4", title: "Prepared palm", point: [0.509, 0.654], hand: { frame: 54 } }],
          voices: [{ name: "pop.wav", url: "/pop.wav", title: "Pop", effect: "fireball" },
            { name: "pf-pf-boom-author.wav", url: "/rhythm.wav", title: "pf, pf, BOOM", effect: "fire_burst" }],
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
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  if (scrollIntoView) Object.defineProperty(HTMLElement.prototype, "scrollIntoView", scrollIntoView);
  else Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
});

const button = (text: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === text)!;
const click = async (text: string) => { await act(async () => { button(text).click(); }); };

describe("Studio prepared palm shortcut", () => {
  it("offers creation first and selects the prepared palm without requesting microphone permission or submitting", async () => {
    const getUserMedia = vi.fn();
    const original = Object.getOwnPropertyDescriptor(navigator, "mediaDevices");
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
    try {
      await act(async () => { root.render(<Studio />); });
      expect(button("Perform this fire with your voice")).toBeDefined();
      expect(button("Candle")).toBeUndefined();
      await click("Perform this fire with your voice");
      expect(host.textContent).toContain("Record your rhythm");
      expect(host.textContent).not.toContain("Where does it come from?");
      expect(host.querySelector("video")?.getAttribute("src")).toBe("/palm.mp4");
      expect(button("Use my microphone").disabled).toBe(false);
      expect(getUserMedia).not.toHaveBeenCalled();
      expect(submitted).toHaveLength(0);
      await click("Different shot");
      await click("Use your own clip or explore effects");
      expect(button("Candle")).toBeDefined();
      expect(host.textContent).toContain("Use your own clip");
    } finally {
      if (original) Object.defineProperty(navigator, "mediaDevices", original);
      else Reflect.deleteProperty(navigator, "mediaDevices");
    }
  });

  it("composes the selected sample explicitly and performs again through the existing replay route", async () => {
    vi.useFakeTimers();
    const apiFetch = fetch;
    const replayed: FormData[] = [];
    const palmJob: JobStatus = { ...job, id: "fresh-palm", cached: false, mode: "bending", files: { "final.mp4": "/fresh-palm.mp4" },
      receipt: { ...job.receipt!, mode: "bending", bending: { mask: null, snap: { used: "cue", mismatch: false },
        hits: [{ frame: 24, gain: 1, size: 1 }], performed_again: { from: "prepared-palm", no_new_render: true } } } };
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (url === "/api/jobs") { submitted.push(init!.body as FormData); return Response.json(palmJob); }
      if (url === "/api/jobs/fresh-palm/replay") { replayed.push(init!.body as FormData); return Response.json({ ...palmJob, id: "second-palm" }); }
      return apiFetch(url, init);
    });
    await act(async () => { root.render(<Studio />); });
    await click("Perform this fire with your voice");
    expect(host.querySelector("details")?.open).toBe(false);
    expect(host.textContent).toContain("These are recorded examples, not your microphone");
    await click("pf, pf, BOOM");
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(submitted).toHaveLength(1);
    const form = submitted[0];
    expect(form.get("plate_sample")).toBe("hand_palm.mp4");
    expect(form.get("mode")).toBe("bending");
    expect(form.get("performance_mode")).toBe("reuse");
    expect(form.get("effect")).toBe("fire_burst");
    expect(form.get("voice_sample")).toBe("pf-pf-boom-author.wav");
    expect(form.get("frame")).toBe("54");
    expect(host.querySelector('a[download="cue-the-boom.mp4"]')?.textContent).toBe("Download");
    await click("Perform again");
    expect(host.textContent).toContain("Record your rhythm");
    await click("pf, pf, BOOM");
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(submitted).toHaveLength(1);
    expect(replayed).toHaveLength(1);
    expect(replayed[0].get("voice_sample")).toBe("pf-pf-boom-author.wav");
  });
});

describe("All-element quick flow", () => {
  it("previews the chosen prepared hand instead of an older shot with the same effect", async () => {
    const apiFetch=fetch;
    vi.stubGlobal("fetch",async (url:string,init?:RequestInit)=> {
      if(url==="/api/gallery")return Response.json([
        {...job,effect:"lightning_arc",files:{"final.mp4":"/old-point.mp4"},receipt:{...job.receipt,inputs:{point:[.83,.31]}}},
        {...job,effect:"lightning_arc",files:{"final.mp4":"/new-palm.mp4"},receipt:{...job.receipt,inputs:{point:[.509,.654]}}},
      ]);
      return apiFetch(url,init);
    });
    await act(async()=>{root.render(<Studio/>);});
    await click("Lightning");
    expect(host.querySelector(".quick-preview video")?.getAttribute("src")).toBe("/new-palm.mp4");
  });

  it.each([
    ["Air", "Vortex", "air_ball", "hoooo.wav", "hand_open.mp4"],
    ["Air", "Blast", "air_blast", "hoo-hoo-HOO.wav", "hand_open.mp4"],
    ["Water", "Orb", "water_orb", "shhhhh-sploosh.wav", "hand_open.mp4"],
    ["Water", "Whip", "water_whip", "shhhhh-sploosh.wav", "hand_open.mp4"],
    ["Lightning", "Arcs", "lightning_arc", "zzzzz-tsst.wav", "hand_palm.mp4"],
    ["Lightning", "Ball", "lightning_ball", "zzzzz-tsst.wav", "hand_palm.mp4"],
    ["Earth", "Floating stone", "earth_rock", "hoooo.wav", "hand_open.mp4"],
    ["Earth", "Shatter", "earth_shatter", "pf-pf-boom-author.wav", "hand_open.mp4"],
  ])("makes %s / %s directly without fire reuse", async (element, variant, effect, voice, plate) => {
    vi.useFakeTimers();
    const apiFetch = fetch;
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (url === "/api/samples") return Response.json({
        plates: ["hand_open.mp4", "hand_palm.mp4"].map(name => ({ name, url: `/${name}`, point: [0.5,0.6], hand: { frame: 30 } })),
        voices: [{ name: voice, title: "Element cue", url: "/cue.wav", effect, hand: true }],
      });
      return apiFetch(url, init);
    });
    await act(async () => { root.render(<Studio />); });
    await click(element);
    await click(variant);
    await click(`Perform this ${element.toLowerCase()} with your voice`);
    expect(host.textContent).not.toContain("make fire");
    expect(host.querySelector("video")?.getAttribute("src")).toBe(`/${plate}`);
    await click("Element cue");
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(submitted).toHaveLength(1);
    expect(submitted[0].get("effect")).toBe(effect);
    expect(submitted[0].get("mode")).toBe("bending");
    expect(submitted[0].get("plate_sample")).toBe(plate);
    expect(submitted[0].get("performance_mode")).toBeNull();
  });
});

describe("Studio replay cues", () => {
  it("limits Perform it again to replay cues after making an offline sample from its own sound", async () => {
    await act(async () => { root.render(<Studio />); });
    await click("Use your own clip or explore effects");
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

describe("Studio interrupted render recovery", () => {
  it.each(["queued", "running"])("keeps a %s job when Move it unmounts and then reopens the shot panel", async (lastStatus) => {
    vi.useFakeTimers();
    vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
    class ProgressStream extends EventTarget {
      static latest: ProgressStream;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      constructor() { super(); ProgressStream.latest = this; }
      close() {}
    }
    vi.stubGlobal("EventSource", ProgressStream);
    const apiFetch = fetch;
    let online = true;
    let finished = false;
    const queued: JobStatus = { ...job, status: "queued", stage: "queued", progress: 0, message: "Waiting", files: {}, receipt: null };
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (url === "/api/jobs") {
        submitted.push(init!.body as FormData);
        return Response.json(queued);
      }
      if (url === "/api/jobs/candle-shot") {
        if (!online) throw new TypeError("Network offline");
        return Response.json(finished ? job : { ...queued, status: "running", stage: "rendering", message: "Rendering", progress: 0.5 });
      }
      return apiFetch(url, init);
    });
    await act(async () => { root.render(<Studio />); });
    await click("Use your own clip or explore effects");
    await click("Candle");
    await click("The clip's own sound");
    await click("Lock it to my clip's sound");
    await click("Make the shot");
    if (lastStatus === "running") await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
    online = false;
    await act(async () => {
      ProgressStream.latest.onerror?.();
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(button("Move it").disabled).toBe(false);
    await click("Move it");
    expect(button("Check existing shot")).toBeUndefined();
    await click("That's the spot");

    expect(button("Check existing shot")).toBeDefined();
    expect(button("Make the shot")).toBeUndefined();
    expect(host.textContent).toContain(`Last known status: ${lastStatus}`);
    online = true;
    finished = true;
    await click("Check existing shot");
    await act(async () => { ProgressStream.latest.onerror?.(); });
    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/candle-shot.mp4");
    expect(submitted).toHaveLength(1);
    expect(button("Perform it again")).toBeDefined();
  });
});
