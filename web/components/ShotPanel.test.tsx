import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ShotPanel from "./ShotPanel";
import type { EffectInfo, JobStatus } from "@/lib/shot";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Props = ComponentProps<typeof ShotPanel>;
let host: HTMLDivElement, root: Root;
let sent: FormData[];

const effectInfo = (id: string, element = true, scene = false): EffectInfo => ({
  id, element, scene, fluid: "fire", behaviour: "burst", stops_with_voice: false,
});
const effects = [
  ...["fire_breath", "fire_burst", "flame_plume", "steam_jet"].map((id) => effectInfo(id)),
  ...["explosion", "lightning_arc", "lightning_ball", "water_orb", "water_whip", "air_ball", "air_blast"]
    .map((id) => effectInfo(id, false)),
  effectInfo("smoke_puff"), effectInfo("wick_flare", false, true),
];
const plate: Props["plate"] = {
  source: { sample: "hand_open.mp4" }, url: "/hand_open.mp4", title: "Your hand: open palm, lit room",
  point: [0.532, 0.707], direction: null, subject: null, sample: true, sound: false, hand: { frame: 30 },
};
const performance: Props["performance"] = {
  cue: "voice", voice: { sample: "shhhhh-sploosh.wav" }, timing: null, face: null, faceExt: "webm",
  suggested: "water_orb", onsetS: null, warnings: [], label: "shhhhh", previewUrl: "",
};

beforeEach(() => {
  sent = [];
  // Keep submitShot and its FormData real; stop at the network boundary before a worker is needed.
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (url !== "/api/jobs" || !(init?.body instanceof FormData)) throw new Error(`Unexpected request: ${url}`);
    sent.push(init.body);
    return { ok: false, status: 503, text: async () => "Test submission captured; no worker started" };
  }));
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const show = async (overrides: Partial<Props> = {}) => {
  await act(async () => {
    root.render(<ShotPanel plate={plate} point={[0.532, 0.707]} direction={null} performance={performance}
      effects={effects} modes={["instant"]} strokes={[]} hand={{ frame: 30, prep: "prepared-hand" }}
      onDone={() => undefined} {...overrides} />);
  });
};
const autoPerformance = { ...performance, suggested: "fireball" };
const radios = () => [...host.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Effect"] [role="radio"]')];
const radio = (label: string) => {
  const found = radios().find((b) => b.textContent?.startsWith(label));
  expect(found, `The effect "${label}" should be visible`).toBeDefined();
  return found!;
};
const button = (label: string) => {
  const found = [...host.querySelectorAll<HTMLButtonElement>("button")].find((b) => b.textContent === label);
  expect(found, `The button "${label}" should be visible`).toBeDefined();
  return found!;
};
const tap = async (el: HTMLButtonElement) => { await act(async () => { el.click(); }); };
const offlineCopy = () => [...host.querySelectorAll("p")].find((p) => p.textContent?.startsWith("The GPU is offline:"))?.textContent;
const finishedHand = (effect: string): JobStatus => ({
  id: "finished-hand", effect, status: "done", stage: "done", progress: 1, message: "Done", error: null, files: {},
  receipt: {
    mode: "bending",
    bending: { mask: null, snap: { used: "cue", mismatch: false }, hits: [{ frame: 30, gain: 1, size: 1 }], performed_again: null },
    render: { seconds: 0, renderer: "", prompt: "", model: "" },
    measured: { stop_latency_frames: null, stops_with_voice: false },
    voice: { hero_onset_s: 1, voice_end_frame: null, active_intervals: [] },
    guide: { length: 120, tail: "", keyframes: [], anchor_frame: 30 },
    seconds_total: 76,
  },
});

describe("ShotPanel offline sample hand", () => {
  it("lets an automatic cue reveal the available made effects without offering rendered hand effects", async () => {
    await show({ performance: autoPerformance });
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("true");
    await tap(button("Choose it yourself"));

    for (const label of ["Explosion", "Lightning", "Ball lightning", "Water orb", "Water whip", "Air ball", "Air blast"]) {
      expect(radio(label).disabled).toBe(false);
    }
    expect(radios()).toHaveLength(8);
    expect(radios().some((b) => /^(Flamethrower|Fireball|Flame in the palm|Steam|Smoke puff|Flare-up)/.test(b.textContent ?? ""))).toBe(false);
  });

  it("offers only made effects advertised by this server", async () => {
    await show({ performance: autoPerformance, effects: [effectInfo("water_orb", false), effectInfo("fire_burst")] });
    await tap(button("Choose it yourself"));
    expect(radio("Water orb").disabled).toBe(false);
    expect(radios()).toHaveLength(2);
    expect(radios().some((b) => b.textContent?.startsWith("Air ball"))).toBe(false);
  });

  it("shows the suggested water selection and submits it as a hand shot", async () => {
    await show();
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("true");
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("false");
    await tap(button("Make the shot"));

    expect(sent).toHaveLength(1);
    expect(sent[0].get("effect")).toBe("water_orb");
    expect(sent[0].get("mode")).toBe("bending");
    expect(sent[0].get("plate_sample")).toBe("hand_open.mp4");
    expect(sent[0].get("frame")).toBe("30");
    expect(sent[0].get("reuse")).toBe("prepared-hand");
  });

  it("can submit the cue's choice and then reselect and submit water on the same take", async () => {
    await show();
    await tap(radio("Your cue decides"));
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("false");
    await tap(button("Make the shot"));
    await tap(radio("Water orb"));
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("true");
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("false");
    await tap(button("Make the shot"));

    expect(sent.map((form) => form.get("effect"))).toEqual(["auto", "water_orb"]);
    expect(sent.map((form) => form.get("mode"))).toEqual(["bending", "bending"]);
  });

  it("submits a manually chosen air effect in place of the suggested water", async () => {
    await show();
    await tap(radio("Air ball"));
    expect(radio("Air ball").getAttribute("aria-checked")).toBe("true");
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("false");
    expect(offlineCopy()).toContain("Air ball");
    expect(offlineCopy()).not.toContain("Water orb");
    await tap(button("Make the shot"));
    expect(sent.map((form) => form.get("effect"))).toEqual(["air_ball"]);
  });

  it.each([
    ["lightning_arc", "Lightning"], ["lightning_ball", "Ball lightning"],
    ["water_orb", "Water orb"], ["water_whip", "Water whip"], ["air_ball", "Air ball"], ["air_blast", "Air blast"],
  ])("describes selected %s as made locally with no new render", async (id, label) => {
    await show({ performance: { ...performance, suggested: id } });
    expect(offlineCopy()).toContain(label);
    expect(offlineCopy()).toContain("made on this computer");
    expect(offlineCopy()).toContain(". No new render");
    expect(offlineCopy()).not.toContain("fireball");
  });

  it("gives an offline explosion a longer estimate and explains its pre-made fireball", async () => {
    await show({ performance: { ...performance, suggested: "explosion" } });
    expect(offlineCopy()).toContain("fireball made ahead of time");
    expect(offlineCopy()).toContain("its blast is made on this computer from your hand and your voice");
    expect(offlineCopy()).toContain("No new render; takes a few minutes on a laptop CPU");
    expect(offlineCopy()).not.toContain("processing time varies with your clip and computer");
  });

  it("shows the same local explosion estimate when hand effects are available without a scene GPU", async () => {
    await show({ performance: { ...performance, suggested: "explosion" }, modes: ["bending", "instant"] });
    expect(host.textContent).toContain("its blast is made on this computer from your hand and your voice");
    expect(host.textContent).toContain("No new render; takes a few minutes on a laptop CPU");
    expect(host.textContent).not.toContain("Processing time varies with your clip and computer");
    expect(host.textContent).not.toContain("Rendered once over black on the GPU");
  });

  it("explains variable processing time for other offline hand effects", async () => {
    await show();
    expect(offlineCopy()).toContain("No new render; processing time varies with your clip and computer");
    await tap(radio("Explosion"));
    expect(offlineCopy()).toContain("takes a few minutes on a laptop CPU");
    await tap(radio("Water orb"));
    expect(offlineCopy()).toContain("No new render; processing time varies with your clip and computer");
  });

  it.each([
    ["explosion", "takes a few minutes on a laptop CPU"],
    ["water_orb", "processing time varies with your clip and computer"],
  ])("gives a completed %s the matching replay estimate", async (effect, estimate) => {
    await show({ initialJob: finishedHand(effect), onReplay: vi.fn() });
    const replayCopy = button("Perform it again").parentElement?.textContent;
    expect(replayCopy).toContain(`a new rhythm on the same hand take: no new render, ${estimate}`);
  });

  it("explains both local effects and the fireball made ahead when the cue decides", async () => {
    await show();
    await tap(radio("Your cue decides"));
    expect(offlineCopy()).toContain("Your cue chooses");
    expect(offlineCopy()).toContain("made on this computer");
    expect(offlineCopy()).toContain("fireball, made ahead of time");
    expect(offlineCopy()).toContain("No new render");
  });

  it("accepts a new cue's suggestion while keeping manual choices across unrelated rerenders", async () => {
    await show();
    await tap(radio("Air ball"));
    await show({ modes: ["bending", "instant"] });
    expect(radio("Air ball").getAttribute("aria-checked")).toBe("true");
    await show({ performance: { ...performance, suggested: "lightning_arc" } });
    expect(radio("Lightning").getAttribute("aria-checked")).toBe("true");
    await show({ performance: autoPerformance });
    expect(radio("Lightning").getAttribute("aria-checked")).toBe("true");
  });
});

describe("ShotPanel monitoring recovery", () => {
  const queued: JobStatus = {
    id: "submitted-shot", effect: "water_orb", status: "queued", stage: "queued", progress: 0,
    message: "Waiting for the GPU", error: null, files: {}, receipt: null,
  };
  class ProgressStream extends EventTarget {
    static latest: ProgressStream;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: (() => void) | null = null;
    closed = false;
    constructor() { super(); ProgressStream.latest = this; }
    close() { this.closed = true; }
  }

  it("checks the existing shot after monitoring fails instead of submitting another render", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("EventSource", ProgressStream);
    let online = false;
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (url === "/api/jobs" && init?.body instanceof FormData) {
        sent.push(init.body);
        return Response.json(queued);
      }
      if (url === "/api/jobs/submitted-shot") {
        if (!online) throw new TypeError("Network offline");
        return Response.json({ ...queued, status: "done", stage: "done", progress: 1, files: { "final.mp4": "/shot.mp4" } });
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    const completed: { id: string; effect: string }[] = [];
    await show({ onDone: (job, effect) => completed.push({ id: job.id, effect }) });
    await tap(button("Make the shot"));
    await act(async () => {
      ProgressStream.latest.onerror?.();
      await vi.advanceTimersByTimeAsync(6000);
    });

    expect(host.textContent).not.toContain("The shot failed");
    expect(host.textContent).toContain("submitted-shot");
    expect(button("Check existing shot").disabled).toBe(false);
    await tap(radio("Air ball"));
    online = true;
    await tap(button("Check existing shot"));
    await act(async () => { ProgressStream.latest.onerror?.(); });

    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/shot.mp4");
    expect(sent).toHaveLength(1);
    expect(completed).toEqual([{ id: "submitted-shot", effect: "water_orb" }]);
  });

  it("stops observing a submitted shot when the panel unmounts", async () => {
    vi.stubGlobal("EventSource", ProgressStream);
    vi.stubGlobal("fetch", async () => Response.json(queued));
    await show();
    await tap(button("Make the shot"));
    await act(async () => { root.render(null); });
    expect(ProgressStream.latest.closed).toBe(true);
  });

  it("lets the user make a replacement after the server confirms the job no longer exists", async () => {
    vi.stubGlobal("EventSource", ProgressStream);
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (url === "/api/jobs" && init?.body instanceof FormData) {
        sent.push(init.body);
        return Response.json(sent.length === 1 ? queued : {
          ...queued, id: "replacement-shot", status: "done", stage: "done", files: { "final.mp4": "/replacement.mp4" },
        });
      }
      if (url === "/api/jobs/submitted-shot") return new Response("Not found", { status: 404 });
      throw new Error(`Unexpected request: ${url}`);
    });
    await show();
    await tap(button("Make the shot"));
    await act(async () => { ProgressStream.latest.onerror?.(); });
    expect(host.textContent).toContain("could not find this shot");
    expect(button("Make the shot").disabled).toBe(false);
    await tap(button("Make the shot"));
    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/replacement.mp4");
    expect(sent).toHaveLength(2);
  });

  it("does not let a late guide-preview response overwrite a completed shot", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("EventSource", ProgressStream);
    let finishPreview!: (response: Response) => void;
    vi.stubGlobal("fetch", async (url: string) => {
      if (url === "/api/jobs") return Response.json(queued);
      return new Promise<Response>((resolve) => { finishPreview = resolve; });
    });
    await show();
    await tap(button("Make the shot"));
    await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
    await act(async () => {
      ProgressStream.latest.dispatchEvent(new MessageEvent("end", { data: JSON.stringify({
        ...queued, status: "done", stage: "done", files: { "final.mp4": "/shot.mp4" },
      }) }));
    });
    await act(async () => { finishPreview(Response.json({ ...queued, status: "running" })); });
    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/shot.mp4");
    expect(button("Make it again").disabled).toBe(false);
  });
});

