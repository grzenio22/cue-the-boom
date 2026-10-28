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
  suggested: "water_orb", experimental: true, onsetS: null, warnings: [], label: "shhhhh", previewUrl: "",
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
const autoPerformance = { ...performance, suggested: "fireball", experimental: false };
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

describe("ShotPanel prepared fire shortcut", () => {
  const quickPlate: Props["plate"] = { ...plate, source: { sample: "hand_palm.mp4" }, hand: { frame: 54 } };

  it("automatically submits the recorded voice once, prevents duplicate taps, and retries a failed request", async () => {
    vi.useFakeTimers();
    let finishRequest!: (response: Response) => void;
    const recorded = { ...autoPerformance, voice: { file: new Blob(["new microphone PCM"], { type: "audio/wav" }), filename: "mic.wav" } };
    const done = { ...finishedHand("fire_burst"), files: { "final.mp4": "/fresh-fire.mp4" } };
    const onReplay = vi.fn();
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      expect(url).toBe("/api/jobs");
      sent.push(init!.body as FormData);
      if (sent.length === 1) return new Promise<Response>((resolve) => { finishRequest = resolve; });
      return Response.json(done);
    });
    await show({ quick: true, plate: quickPlate, performance: recorded, modes: ["bending", "scene", "instant"], onReplay });
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(sent).toHaveLength(1);
    expect(sent[0].get("voice")).toBeInstanceOf(File);
    expect(sent[0].get("voice_sample")).toBeNull();
    expect(sent[0].get("performance_mode")).toBe("reuse");
    expect(sent[0].get("effect")).toBe("fire_burst");
    expect(sent[0].get("mode")).toBe("bending");
    expect(radios()).toHaveLength(0);
    expect(button("Making your fire…").disabled).toBe(true);
    await tap(button("Making your fire…"));
    expect(sent).toHaveLength(1);
    await act(async () => { finishRequest(new Response("Temporary service failure", { status: 503 })); });
    expect(host.textContent).toContain("Temporary service failure");
    expect(button("Try again").disabled).toBe(false);
    await tap(button("Try again"));
    expect(sent).toHaveLength(2);
    expect((sent[1].get("voice") as File).name).toBe("mic.wav");
    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/fresh-fire.mp4");
    expect(host.querySelector('a[download="cue-the-boom.mp4"]')?.textContent).toBe("Download");
    await tap(button("Perform again"));
    expect(onReplay).toHaveBeenCalledWith("finished-hand", []);
  });

  it("does not resubmit a pending quick take after the panel remounts", async () => {
    vi.useFakeTimers();
    const pending: JobStatus = { ...finishedHand("fire_burst"), status: "running", stage: "compositing", progress: 0.4, files: {}, receipt: null };
    await show({ quick: true, plate: quickPlate, initialJob: pending, initialEffect: "fire_burst" });
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(sent).toHaveLength(0);
    expect(button("Check existing shot").disabled).toBe(false);
    expect(host.textContent).toContain("Last known status: running");
  });
});

describe("ShotPanel stone material finish", () => {
  const stone = (): JobStatus => ({ ...finishedHand("earth_rock"), can_finish: true,
    files: { "final.mp4": "/original-stone.mp4" } });

  it("starts a GPU pass only after a click and performs the original again", async () => {
    const source = stone();
    const onReplay = vi.fn();
    const done: JobStatus = { ...source, id: "refined-stone", can_finish: false,
      files: { "final.mp4": "/refined-stone.mp4" }, receipt: { ...source.receipt!,
        material_finish: { parent_id: source.id, recipe: "tracked-stone-v1", new_gpu_render: true, native_size: [576, 1024] },
        render: { ...source.receipt!.render, seconds: 24 } } };
    let resolve!: (r: Response) => void;
    const request = vi.fn((_url: string, _init?: RequestInit) => new Promise<Response>((r) => { resolve = r; }));
    vi.stubGlobal("fetch", request);
    await show({ quick: true, quickEffect: "earth_rock", initialJob: source, modes: ["bending", "finish"], onReplay });
    expect(request).not.toHaveBeenCalled();
    await tap(button("Refine stone · GPU"));
    await tap(button("Starting material pass…"));
    expect(request).toHaveBeenCalledTimes(1);
    expect(request.mock.calls[0][0]).toBe(`/api/jobs/${source.id}/finish`);
    await act(async () => { resolve(Response.json(done)); });
    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/refined-stone.mp4");
    expect(host.textContent).toContain("a fresh 24 s GPU render");
    await tap(button("Perform again"));
    expect(onReplay).toHaveBeenCalledWith(source.id, []);
    await tap(button("Back to the original performance"));
    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/original-stone.mp4");
  });

  it("keeps the original result on rejection and hides the option without a ready GPU", async () => {
    const source = stone();
    await show({ quick: true, initialJob: source, modes: ["bending"] });
    expect(host.textContent).not.toContain("Refine stone · GPU");
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ detail: "GPU unavailable" }, { status: 409 })));
    await show({ quick: true, initialJob: source, modes: ["bending", "finish"] });
    await tap(button("Refine stone · GPU"));
    expect(host.querySelector('[role="alert"]')?.textContent).toBe("GPU unavailable");
    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/original-stone.mp4");
  });
});

describe("ShotPanel offline sample hand", () => {
  it("does not preselect experimental suggestions without an explicit choice", async () => {
    await show({ performance: { ...performance, suggested: "air_ball", experimental: false } });
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("true");
    await tap(button("Make the shot"));
    expect(sent[0].get("effect")).toBe("auto");
  });

  it("lets an automatic cue reveal the available made effects without offering rendered hand effects", async () => {
    await show({ performance: autoPerformance });
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("true");
    await tap(button("Choose it yourself"));

    for (const label of ["Explosion", "Lightning", "Ball lightning", "Water orb", "Water whip", "Air ball", "Air blast"]) {
      expect(radio(label).disabled).toBe(false);
    }
    expect(radios()).toHaveLength(8);
    expect(radios().some((b) => /^(Flamethrower|Fireball|Flame in the palm|Steam|Smoke puff|Flare-up)/.test(b.textContent ?? ""))).toBe(false);
    const experimental = radio("Air ball").closest("details")!;
    expect(experimental.querySelector("summary")?.textContent).toBe("Experimental");
    expect(experimental.open).toBe(false);
    for (const label of ["Air ball", "Air blast"]) {
      expect(radio(label).closest("details")).toBe(experimental);
      expect(radio(label).textContent).toContain("Experimental");
    }
    for (const label of ["Explosion", "Lightning", "Ball lightning", "Water orb", "Water whip"]) {
      expect(radio(label).textContent).not.toContain("Experimental");
      expect(radio(label).closest("details")).toBeNull();
    }
  });

  it("keeps the live hand picker focused on fire, water, explosion and lightning", async () => {
    await show({ performance: autoPerformance, modes: ["bending", "instant"] });
    await tap(button("Choose it yourself"));
    const offered = radios().filter((b) => !b.closest("details"));
    expect(offered).toHaveLength(9); // the automatic choice and eight core forms
    expect(radios().find((b) => b.textContent?.startsWith("Steam"))).toBeUndefined();
    expect(radio("Fireball").closest("details")).toBeNull();
    expect(radio("Lightning").closest("details")).toBeNull();
    expect(radio("Water orb").closest("details")).toBeNull();
    expect(radio("Air ball").closest("details")?.open).toBe(false);
  });

  it("keeps detected-effect badge overrides within the core hand effects", async () => {
    const completed = finishedHand("fire_burst");
    completed.receipt!.auto = { heard: "boom", hand: "palm", picked: "fire_burst", why: "A hit from the palm" };
    await show({ performance: autoPerformance, initialJob: completed, modes: ["bending", "instant"] });
    await tap(host.querySelector<HTMLButtonElement>('[aria-label="All effects"]')!);
    const badgeGroup = [...host.querySelectorAll('[role="radiogroup"][aria-label="Effect"]')].at(-1)!;
    expect([...badgeGroup.querySelectorAll('[role="radio"]')].map((b) => b.textContent)).toEqual([
      "Flamethrower", "Fireball", "Flame in the palm", "Water whip", "Water jet", "Water orb", "Water ribbon", "Water surge",
      "Tidal shield", "Air vortex", "Explosion", "Lightning", "Ball lightning",
    ]);
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
    const disclosure = radio("Air ball").closest("details")!;
    await act(async () => { disclosure.querySelector("summary")!.click(); });
    expect(disclosure.open).toBe(true);
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

  it.each([false, true])("checks the existing shot after monitoring fails instead of submitting another render (quick=%s)", async (quick) => {
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
    await show({ quick, onDone: (job, effect) => completed.push({ id: job.id, effect }) });
    if (quick) await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    else await tap(button("Make the shot"));
    await act(async () => {
      ProgressStream.latest.onerror?.();
      await vi.advanceTimersByTimeAsync(6000);
    });

    expect(host.textContent).not.toContain("The shot failed");
    expect(host.textContent).toContain("submitted-shot");
    expect(button("Check existing shot").disabled).toBe(false);
    if (!quick) await tap(radio("Air ball"));
    online = true;
    await tap(button("Check existing shot"));
    await act(async () => { ProgressStream.latest.onerror?.(); });

    expect(host.querySelector("video.result")?.getAttribute("src")).toBe("/shot.mp4");
    expect(sent).toHaveLength(1);
    expect(completed).toEqual([{ id: "submitted-shot", effect: quick ? "fire_burst" : "water_orb" }]);
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

