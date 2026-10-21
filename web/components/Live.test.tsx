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
  close = vi.fn(async () => { this.state = "closed"; });
  createAnalyser = () => ({ fftSize: 1024, frequencyBinCount: 512 });
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
