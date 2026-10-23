import { afterEach, describe, expect, it, vi } from "vitest";
import { dragDirection, failure, followJob, hitAnchor, madeAhead, pollUntilDone, type JobStatus, type Receipt, replayable, sameStrokes, sceneEffect, submitShot, suggestEffect } from "./shot";

const SR = 48000;

describe("submitShot", () => {
  it.each([undefined, "reuse"] as const)("forwards explicit CPU reuse only when selected: %s", async (performanceMode) => {
    const request = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: "new-shot" })));
    vi.stubGlobal("fetch", request);
    try {
      await submitShot({ plate: { sample: "hand_palm.mp4" }, voice: { file: new Blob(["pcm"]), filename: "mic.wav" },
        face: null, faceExt: "webm", timing: null, effect: "fire_burst", point: [0.509, 0.654], seed: 1,
        mode: "bending", performanceMode });
      const [url, init] = request.mock.calls[0];
      expect(url).toBe("/api/jobs");
      expect(init.body.get("performance_mode")).toBe(performanceMode ?? null);
      expect(init.body.get("voice").name).toBe("mic.wav");
      expect(init.body.get("plate_sample")).toBe("hand_palm.mp4");
    } finally { vi.unstubAllGlobals(); }
  });
});

/** 0.5 s of silence, then `dur` s of either a voiced tone (low zero-crossing rate) or noise (high). */
function noise(dur: number, kind: "voiced" | "noisy"): Float32Array {
  const pcm = new Float32Array(Math.round((dur + 1) * SR));
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
  for (let i = 0; i < dur * SR; i++) {
    const t = i / SR;
    pcm[Math.round(0.5 * SR) + i] = 0.5 * (kind === "voiced" ? Math.sin(2 * Math.PI * 140 * t) : rnd());
  }
  return pcm;
}

describe("suggestEffect", () => {
  it("hears a short hit as a fireball", () => expect(suggestEffect(noise(0.15, "noisy"), SR, 0.5)).toBe("fireball"));
  it("hears a breathy pfffff as a smoke puff", () => expect(suggestEffect(noise(0.5, "noisy"), SR, 0.5)).toBe("smoke_puff"));
  it("hears a long hiss as steam", () => expect(suggestEffect(noise(1.2, "noisy"), SR, 0.5)).toBe("steam_jet"));
  it("hears a long voiced fwoooom as a flame plume", () => expect(suggestEffect(noise(1.2, "voiced"), SR, 0.5)).toBe("flame_plume"));
  it("falls back to a fireball without an onset", () => expect(suggestEffect(new Float32Array(10), SR, null)).toBe("fireball"));
});

describe("dragDirection", () => {
  it("is clockwise from straight up", () => {
    expect(dragDirection(0, -10)).toBe(0);
    expect(dragDirection(10, 0)).toBe(90);
    expect(dragDirection(-10, 0)).toBe(-90);
    expect(Math.abs(dragDirection(0, 10))).toBe(180);
  });
  it("rounds to 5 degrees", () => expect(dragDirection(10, -9)).toBe(50));
});

describe("hitAnchor", () => {
  const guide = { length: 96, tail: "fade", keyframes: [0, 1, 9], anchor_frame: 7 };
  it("puts a scene shot's handle on its hit, not on the render's first frame", () =>
    expect(hitAnchor({ mode: "scene", guide, alignment: { hit_frame: 24 } } as Receipt)).toBe(24));
  it("keeps an element shot's guide anchor", () => expect(hitAnchor({ guide } as Receipt)).toBe(7));
  it("has nothing to anchor without a receipt", () => expect(hitAnchor(null)).toBe(0));
});

describe("madeAhead", () => {
  const candle = { sample: true, point: [0.53, 0.52] as [number, number], scene: { "pf.wav": "blow_out", hit: "blow_out" }, sound: true };
  const top = [{ points: [[0.36, 0.53], [0.5, 0.4]] as [number, number][], width: 0.09 }];
  const pumpkin = { ...candle, drawn: [{ title: "drawn", voice: "fwoooom.wav", effect: "flame_plume", strokes: top }] };
  it("plays a listed sample noise at the marked spot", () =>
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.53, 0.52], [])).toBe(true));
  it("plays the clip's own sound", () => expect(madeAhead(candle, "clip", null, [0.53, 0.52], [])).toBe(true));
  it("not once the spot moves, for a noise it doesn't list, a drawing, or your own clip", () => {
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.4, 0.52], [])).toBe(false);
    expect(madeAhead(candle, "voice", { sample: "hissss.wav" }, [0.53, 0.52], [])).toBe(false);
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.53, 0.52], top)).toBe(false);
    expect(madeAhead({ ...candle, sample: false }, "clip", null, [0.53, 0.52], [])).toBe(false);
  });
  it("not once it is aimed another way: the server keys an aimed shot by its direction", () => {
    const dino = { ...candle, direction: -90 };
    expect(madeAhead(dino, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], -90)).toBe(true);
    expect(madeAhead(dino, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], null)).toBe(true);
    expect(madeAhead(dino, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], 45)).toBe(false);
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], 45)).toBe(false);
  });
  it("plays the sample's own drawing with the noise it was made with", () => {
    expect(madeAhead(pumpkin, "voice", { sample: "fwoooom.wav" }, [0.36, 0.53], top)).toBe(true);
    const nudged = [{ points: [[0.362, 0.531], [0.5, 0.4]] as [number, number][], width: 0.09 }];
    expect(sameStrokes(top, nudged)).toBe(true);
    expect(madeAhead(pumpkin, "voice", { sample: "ka-boom.wav" }, [0.36, 0.53], top)).toBe(false);
    expect(madeAhead(pumpkin, "voice", { sample: "fwoooom.wav" }, [0.36, 0.53], [...top, ...top])).toBe(false);
  });
  it("says what a noise does to this plate", () => {
    expect(sceneEffect(candle.scene, { name: "pf.wav", url: "" })).toBe("blow_out");
    expect(sceneEffect(candle.scene, { name: "ka-boom.wav", url: "", gesture: "hit" })).toBe("blow_out");
    expect(sceneEffect(undefined, { name: "pf.wav", url: "" })).toBeUndefined();
  });
});

describe("failure", () => {
  it("shows the server's own words, not its JSON", async () => {
    expect((await failure(new Response(JSON.stringify({ detail: "the GPU is offline" }), { status: 503 }))).message).toBe("the GPU is offline");
    expect((await failure(new Response("Internal Server Error", { status: 500 }))).message).toBe("Internal Server Error");
    expect((await failure(new Response("", { status: 502 }))).message).toBe("request failed (502)");
  });
});

describe("replayable", () => {
  const plate = { sample: true, point: [0.48, 0.39] as [number, number], direction: null, scene: { "ka-boom.wav": "fire_burst" } };
  const mine = { file: new Blob(), filename: "voice.webm" };
  it("plays your own take on a sample's scene shot made ahead, at its spot", () => {
    expect(replayable(plate, "voice", mine, [0.48, 0.39], [])).toBe(true);
    expect(replayable(plate, "hold", null, [0.48, 0.39], [])).toBe(true);
  });
  it("not a sample noise (that one was made ahead as it is), another spot, a drawing, or nothing made ahead", () => {
    expect(replayable(plate, "voice", { sample: "ka-boom.wav" }, [0.48, 0.39], [])).toBe(false);
    expect(replayable(plate, "voice", mine, [0.3, 0.39], [])).toBe(false);
    expect(replayable(plate, "voice", mine, [0.48, 0.39], [{ points: [[0.4, 0.4]], width: 0.05 }])).toBe(false);
    expect(replayable({ ...plate, scene: {} }, "voice", mine, [0.48, 0.39], [])).toBe(false);
  });
});

describe("job monitoring", () => {
  const status = (state: JobStatus["status"]): JobStatus => ({
    id: "slow-shot", status: state, stage: state, progress: state === "done" ? 1 : 0.25,
    message: state, error: null, files: state === "done" ? { "final.mp4": "/slow-shot/final.mp4" } : {}, receipt: null,
  });
  class ProgressStream extends EventTarget {
    static latest: ProgressStream;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: (() => void) | null = null;
    closed = false;
    constructor() { super(); ProgressStream.latest = this; }
    close() { this.closed = true; }
  }
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

  it("finishes after a 21-minute queue and 15-minute render without a client lifetime limit", async () => {
    vi.useFakeTimers();
    const started = Date.now();
    vi.stubGlobal("fetch", async () => {
      const elapsed = Date.now() - started;
      return Response.json(status(elapsed < 21 * 60_000 ? "queued" : elapsed < 36 * 60_000 ? "running" : "done"));
    });
    const finished = pollUntilDone("slow-shot").then((job) => job, (error: Error) => error);
    await vi.advanceTimersByTimeAsync(36 * 60_000);
    expect(await finished).toMatchObject({ id: "slow-shot", status: "done", files: { "final.mp4": "/slow-shot/final.mp4" } });
  });

  it("recovers from a disconnected stream, a network failure, and a temporary gateway error", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("EventSource", ProgressStream);
    let attempts = 0;
    vi.stubGlobal("fetch", async (url: string) => {
      expect(url).toBe("/api/jobs/slow-shot");
      attempts++;
      if (attempts === 1) throw new TypeError("Failed to fetch");
      if (attempts === 2) return new Response("Bad gateway", { status: 502 });
      return Response.json(status(attempts === 3 ? "running" : "done"));
    });
    const messages: string[] = [];
    const finished = followJob("slow-shot", (event) => messages.push(event.message)).then((job) => job, (error: Error) => error);
    ProgressStream.latest.onerror?.();
    await vi.runAllTimersAsync();
    expect(await finished).toMatchObject({ status: "done" });
    expect(messages).toContain("running");
    expect(ProgressStream.latest.closed).toBe(true);
  });

  it("falls back to status requests if the stream ends before the job is terminal", async () => {
    vi.stubGlobal("EventSource", ProgressStream);
    vi.stubGlobal("fetch", async () => Response.json(status("done")));
    const finished = followJob("slow-shot", () => undefined);
    ProgressStream.latest.dispatchEvent(new MessageEvent("end", { data: JSON.stringify(status("running")) }));
    await expect(finished).resolves.toMatchObject({ status: "done" });
  });

  it("closes the stream when its owner stops monitoring", async () => {
    vi.stubGlobal("EventSource", ProgressStream);
    const controller = new AbortController();
    const finished = followJob("slow-shot", () => undefined, { signal: controller.signal }).catch((error: Error) => error);
    controller.abort();
    expect(ProgressStream.latest.closed).toBe(true);
    expect(await finished).toMatchObject({ name: "AbortError" });
  });

  it("cancels the polling wait without making another status request", async () => {
    vi.useFakeTimers();
    const requests: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      requests.push(url);
      return Response.json(status("running"));
    });
    const controller = new AbortController();
    const finished = pollUntilDone("slow-shot", 1500, { signal: controller.signal }).catch((error: Error) => error);
    await vi.advanceTimersByTimeAsync(100);
    controller.abort();
    await vi.runAllTimersAsync();
    expect(await finished).toMatchObject({ name: "AbortError" });
    expect(requests).toEqual(["/api/jobs/slow-shot"]);
  });
});
