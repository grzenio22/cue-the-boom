import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Live from "./Live";

const vision = vi.hoisted(() => ({ files: vi.fn(), create: vi.fn() }));
vi.mock("@mediapipe/tasks-vision", () => ({
  FilesetResolver: { forVisionTasks: vision.files },
  HandLandmarker: { createFromOptions: vision.create },
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

function capture() {
  const camera = { stop: vi.fn() }, microphone = { stop: vi.fn() };
  const stream = { getTracks: () => [camera, microphone] } as unknown as MediaStream;
  return { stream, camera, microphone };
}

const tracker = () => ({ close: vi.fn(), detectForVideo: vi.fn(() => ({ landmarks: [] })) });

class FakeAudioContext {
  state = "running";
  sampleRate = 48000;
  close = vi.fn(async () => { this.state = "closed"; });
  createAnalyser = () => ({ fftSize: 1024, frequencyBinCount: 512,
    getFloatTimeDomainData: (data: Float32Array) => data.fill(0.1),
    getFloatFrequencyData: (data: Float32Array) => data.fill(-100) });
  createMediaStreamSource = () => ({ connect: () => undefined });
}

let host: HTMLDivElement, root: Root | null;
let mediaDevices: PropertyDescriptor | undefined;
let getUserMedia: ReturnType<typeof vi.fn<() => Promise<MediaStream>>>;
let audio: FakeAudioContext[];

beforeEach(() => {
  mediaDevices = Object.getOwnPropertyDescriptor(navigator, "mediaDevices");
  getUserMedia = vi.fn<() => Promise<MediaStream>>();
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
  vision.files.mockReset().mockResolvedValue({});
  vision.create.mockReset();
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
  vi.stubGlobal("requestAnimationFrame", vi.fn(() => 1));
  audio = [];
  vi.stubGlobal("AudioContext", class extends FakeAudioContext {
    constructor() { super(); audio.push(this); }
  });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined);
  vi.spyOn(HTMLMediaElement.prototype, "load").mockImplementation(() => undefined);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({} as CanvasRenderingContext2D);
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

const unmount = () => { act(() => root?.unmount()); root = null; };
afterEach(() => {
  unmount();
  host.remove();
  if (mediaDevices) Object.defineProperty(navigator, "mediaDevices", mediaDevices);
  else Reflect.deleteProperty(navigator, "mediaDevices");
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const button = (text: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === text)!;
const click = async (text: string) => { await act(async () => { button(text).click(); }); };
const show = async () => { await act(async () => { root!.render(<Live onTake={() => undefined} />); }); };
const waitForTracker = async (count = 1) => {
  await act(async () => { await vi.waitFor(() => expect(vision.create).toHaveBeenCalledTimes(count)); });
};
const leave = async (how: "Close" | "unmount") => { if (how === "Close") await click("Close"); else unmount(); };

describe("Live resource cleanup", () => {
  it.each(["Close", "unmount"] as const)("stops a stream that arrives after %s", async (how) => {
    const pending = deferred<MediaStream>(), late = capture();
    getUserMedia.mockReturnValueOnce(pending.promise);
    await show();
    await click("Try it live");
    await leave(how);

    await act(async () => { pending.resolve(late.stream); });

    expect(late.camera.stop).toHaveBeenCalledTimes(1);
    expect(late.microphone.stop).toHaveBeenCalledTimes(1);
    expect(fetch).not.toHaveBeenCalled();
    expect(vision.create).not.toHaveBeenCalled();
    expect(audio).toHaveLength(0);
  });

  it.each(["Close", "unmount"] as const)("closes a tracker that arrives after %s without stopping its stream twice", async (how) => {
    const source = capture(), pending = deferred<ReturnType<typeof tracker>>(), late = tracker();
    getUserMedia.mockResolvedValueOnce(source.stream);
    vision.create.mockReturnValueOnce(pending.promise);
    await show();
    await click("Try it live");
    await waitForTracker();
    await leave(how);
    expect(source.camera.stop).toHaveBeenCalledTimes(1);
    expect(source.microphone.stop).toHaveBeenCalledTimes(1);

    await act(async () => { pending.resolve(late); });

    expect(late.close).toHaveBeenCalledTimes(1);
    expect(source.camera.stop).toHaveBeenCalledTimes(1);
    expect(source.microphone.stop).toHaveBeenCalledTimes(1);
    expect(audio).toHaveLength(0);
  });

  it("releases a late stream without stopping the reopened preview", async () => {
    const oldStream = deferred<MediaStream>(), old = capture(), current = capture();
    const currentTracker = deferred<ReturnType<typeof tracker>>(), hands = tracker();
    getUserMedia.mockReturnValueOnce(oldStream.promise).mockResolvedValueOnce(current.stream);
    vision.create.mockReturnValueOnce(currentTracker.promise);
    await show();
    await click("Try it live");
    await click("Close");
    await click("Try it live");
    await waitForTracker();
    await act(async () => { currentTracker.resolve(hands); });
    expect(button("Record this").disabled).toBe(false);

    await act(async () => { oldStream.resolve(old.stream); });

    expect(old.camera.stop).toHaveBeenCalledTimes(1);
    expect(old.microphone.stop).toHaveBeenCalledTimes(1);
    expect(current.camera.stop).not.toHaveBeenCalled();
    expect(current.microphone.stop).not.toHaveBeenCalled();
    expect(hands.close).not.toHaveBeenCalled();
    expect(button("Record this").disabled).toBe(false);
    await click("Close");
    expect(current.camera.stop).toHaveBeenCalledTimes(1);
    expect(current.microphone.stop).toHaveBeenCalledTimes(1);
    expect(hands.close).toHaveBeenCalledTimes(1);
    expect(audio[0].close).toHaveBeenCalledTimes(1);
  });

  it("releases a late tracker without closing the reopened preview's tracker", async () => {
    const old = capture(), current = capture(), late = tracker(), hands = tracker();
    const oldTracker = deferred<ReturnType<typeof tracker>>(), currentTracker = deferred<ReturnType<typeof tracker>>();
    getUserMedia.mockResolvedValueOnce(old.stream).mockResolvedValueOnce(current.stream);
    vision.create.mockReturnValueOnce(oldTracker.promise).mockReturnValueOnce(currentTracker.promise);
    await show();
    await click("Try it live");
    await waitForTracker();
    await click("Close");
    await click("Try it live");
    await waitForTracker(2);
    await act(async () => { currentTracker.resolve(hands); });
    expect(button("Record this").disabled).toBe(false);

    await act(async () => { oldTracker.resolve(late); });

    expect(late.close).toHaveBeenCalledTimes(1);
    expect(old.camera.stop).toHaveBeenCalledTimes(1);
    expect(old.microphone.stop).toHaveBeenCalledTimes(1);
    expect(current.camera.stop).not.toHaveBeenCalled();
    expect(current.microphone.stop).not.toHaveBeenCalled();
    expect(hands.close).not.toHaveBeenCalled();
    expect(button("Record this").disabled).toBe(false);
    await click("Close");
    expect(current.camera.stop).toHaveBeenCalledTimes(1);
    expect(current.microphone.stop).toHaveBeenCalledTimes(1);
    expect(hands.close).toHaveBeenCalledTimes(1);
    expect(audio[0].close).toHaveBeenCalledTimes(1);
  });
});

describe("Live library source profile", () => {
  it.each([
    ["a legacy entry", undefined, 1.1],
    ["a dimmed source", { scale: 0.5, exposure: 0.25 }, 0.58577],
    ["a brighter source", { scale: 0.5, exposure: 4 }, 2.06565],
    ["an invalid exposure", { scale: 0.5, exposure: NaN }, 1.1],
  ])("draws %s with preview brightness and the already calibrated fit", async (_, sourceProfile, expectedBrightness) => {
    getUserMedia.mockResolvedValueOnce(capture().stream);
    const landmarks = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 }));
    landmarks[0].y = 0.75; // a 120 px open hand; the element's reference uses the same scale
    vision.create.mockResolvedValueOnce({ close: vi.fn(), detectForVideo: () => ({ landmarks: [landmarks] }) });
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => [{
      effect: "fire_burst", behaviour: "burst", url: "/library/fire.mp4", size: [640, 480], fps: 24,
      ignition: 2, release: 36, end: 48, floor: 0,
      ref: { xy: [320, 240], dir: [0, -1], scale: 120 }, fit: { palm: 0.5, tip: 0.75 },
      ...(sourceProfile ? { source_profile: sourceProfile } : {}),
    }] } as Response);
    vi.spyOn(HTMLVideoElement.prototype, "videoWidth", "get").mockReturnValue(640);
    vi.spyOn(HTMLVideoElement.prototype, "videoHeight", "get").mockReturnValue(480);
    const filters: string[] = [];
    const ctx = {
      filter: "none", save: vi.fn(), restore: vi.fn(), translate: vi.fn(), scale: vi.fn(), rotate: vi.fn(),
      beginPath: vi.fn(), arc: vi.fn(), stroke: vi.fn(), fillRect: vi.fn(), fillText: vi.fn(),
      createRadialGradient: () => ({ addColorStop: vi.fn() }),
      drawImage: (image: HTMLVideoElement) => {
        if (image.getAttribute("src") === "/library/fire.mp4") filters.push(ctx.filter);
      },
    };
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    await show();
    await click("Try it live");
    await waitForTracker();
    await act(async () => {
      vi.mocked(requestAnimationFrame).mock.calls[0][0](performance.now());
    });

    expect(filters).toHaveLength(1);
    expect(filters[0]).toContain("contrast(1.3)");
    expect(Number(filters[0].match(/brightness\(([^)]+)\)/)?.[1])).toBeCloseTo(expectedBrightness, 4);
    expect(ctx.scale).toHaveBeenLastCalledWith(0.5, 0.5); // fit includes source scale: never multiply it twice
  });
});
