import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Performer, { type Performance } from "./Performer";
import { type PcmRecorder, startPcmRecorder } from "@/lib/audio/pcmRecorder";

vi.mock("@/lib/audio/pcmRecorder", () => ({ startPcmRecorder: vi.fn() }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root | null;
const voice = { name: "pop.wav", url: "/pop.wav", title: "Pop", effect: "fireball" };

beforeEach(() => {
  vi.mocked(startPcmRecorder).mockReset();
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined);
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root?.unmount());
  host.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const button = (text: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === text)!;
const clip = () => button("The clip's own sound");
const click = (element: HTMLElement) => act(() => element.click());

function show(props: Partial<ComponentProps<typeof Performer>> = {}) {
  const onDone = vi.fn<(performance: Performance) => void>();
  act(() => root!.render(<Performer plateUrl="/plate.mp4" point={[0.5, 0.5]} voices={[voice]} takes={1}
    clipSound live={false} onDone={onDone} {...props} />));
  return onDone;
}

describe("Performer replay cues", () => {
  it.each([true, false])("starts a replay with voice and prevents the clip cue (filmed=%s)", (filmed) => {
    const onDone = show({ replay: true, filmed });

    expect(button("Say it").getAttribute("aria-checked")).toBe("true");
    expect(button("Use my microphone").disabled).toBe(false);
    expect(clip().disabled).toBe(true);
    expect(clip().title).toMatch(/voice|hold/i);
    click(clip());
    expect(button("Say it").getAttribute("aria-checked")).toBe("true");
    expect(host.textContent).not.toContain("Lock it to my clip's sound");
    expect(onDone).not.toHaveBeenCalled();
  });

  it("uses a sample voice for a replay with the GPU offline", () => {
    const onDone = show({ replay: true, filmed: true, ready: [voice.name] });
    expect(button("Say it").getAttribute("aria-checked")).toBe("true");

    const sample = [...host.querySelectorAll("button")].find((b) => b.textContent?.startsWith("Pop"))!;
    expect(sample.disabled).toBe(false);
    click(sample);
    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ cue: "voice", voice: { sample: "pop.wav" } }));
  });

  it("records a hold cue for a replay with the GPU offline", async () => {
    const onDone = show({ replay: true, filmed: true });
    expect(button("Say it").getAttribute("aria-checked")).toBe("true");
    expect(clip().disabled).toBe(true);
    click(button("Press and hold"));
    const plate = host.querySelector("video")!;
    vi.spyOn(plate, "play").mockResolvedValue(undefined);
    vi.spyOn(plate, "paused", "get").mockReturnValue(false);
    await act(async () => { button("Play the clip").click(); });
    plate.currentTime = 0.5;
    act(() => { button("Hold").dispatchEvent(new Event("pointerdown", { bubbles: true })); });
    plate.currentTime = 1;
    act(() => { button("Holding…").dispatchEvent(new Event("pointerup", { bubbles: true })); });
    act(() => { plate.dispatchEvent(new Event("ended")); });
    click(button("Use 1 hold"));

    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ cue: "hold", voice: null, holds: [[0.5, 1]] }));
  });
});

function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function capture(constraints = Promise.resolve(), withCamera = false) {
  const track = (kind: "audio" | "video") => ({
    kind,
    readyState: "live",
    stop() { this.readyState = "ended"; },
    applyConstraints: () => constraints,
    getSettings: () => kind === "audio"
      ? { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: 48000, channelCount: 1 }
      : { frameRate: 24, width: 1080, height: 1920, facingMode: "user" },
  });
  const microphone = track("audio"), camera = track("video");
  const tracks = withCamera ? [microphone, camera] : [microphone];
  const stream = {
    getTracks: () => tracks,
    getAudioTracks: () => [microphone],
    getVideoTracks: () => withCamera ? [camera] : [],
  } as unknown as MediaStream;
  return { stream, microphone, camera };
}

class MeterAudioContext {
  state = "running";
  close = async () => { this.state = "closed"; };
  createAnalyser() { return { fftSize: 1024, getFloatTimeDomainData: (buffer: Float32Array) => buffer.fill(0.1) }; }
  createMediaStreamSource = () => ({ connect: () => undefined });
}

describe("Performer microphone lifecycle", () => {
  let mediaDevices: PropertyDescriptor | undefined;
  let getUserMedia: ReturnType<typeof vi.fn<() => Promise<MediaStream>>>;
  let contexts: MeterAudioContext[];
  let frames: Map<number, FrameRequestCallback>;

  beforeEach(() => {
    mediaDevices = Object.getOwnPropertyDescriptor(navigator, "mediaDevices");
    getUserMedia = vi.fn<() => Promise<MediaStream>>();
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
    contexts = [];
    frames = new Map();
    let nextFrame = 0;
    vi.stubGlobal("AudioContext", class extends MeterAudioContext {
      constructor() { super(); contexts.push(this); }
    });
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      frames.set(++nextFrame, callback);
      return nextFrame;
    });
    vi.stubGlobal("cancelAnimationFrame", (frame: number) => frames.delete(frame));
  });

  afterEach(() => {
    if (mediaDevices) Object.defineProperty(navigator, "mediaDevices", mediaDevices);
    else Reflect.deleteProperty(navigator, "mediaDevices");
  });

  const arm = async () => { await act(async () => { button("Use my microphone").click(); }); };
  const leave = (how: "unmount" | "Press and hold" | "The clip's own sound") => {
    if (how === "unmount") { act(() => root!.unmount()); root = null; }
    else click(button(how));
  };

  it("requires explicit microphone intent in the shortcut and emits the newly recorded WAV once", async () => {
    const source = capture();
    getUserMedia.mockResolvedValueOnce(source.stream);
    const onDone = show({ quick: true, takes: 0 });
    expect(getUserMedia).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
    expect(host.querySelector('input[type="checkbox"]')).toBeNull();
    await arm();
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    expect(startPcmRecorder).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
    const recorder = pcmRecorder();
    vi.mocked(startPcmRecorder).mockResolvedValue(recorder);
    const plate = host.querySelector("video")!;
    vi.spyOn(plate, "play").mockResolvedValue(undefined);
    await act(async () => { button("Record & make fire").click(); button("Record & make fire").click(); });
    expect(startPcmRecorder).toHaveBeenCalledTimes(1);
    await act(async () => { plate.dispatchEvent(new Event("ended")); plate.dispatchEvent(new Event("ended")); });
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onDone.mock.calls[0][0].voice).toEqual({ file: expect.any(Blob), filename: "mic.wav" });
    expect(recorder.stop).toHaveBeenCalledTimes(1);
    URL.revokeObjectURL(onDone.mock.calls[0][0].previewUrl);
  });

  // Missing invalidation must not leave a device live when its permission prompt resolves after departure.
  it.each(["unmount", "Press and hold", "The clip's own sound"] as const)("stops a stream acquired after %s", async (how) => {
    const pending = deferred<MediaStream>(), late = capture(Promise.resolve(), true);
    getUserMedia.mockReturnValueOnce(pending.promise);
    show();
    click(host.querySelector("input")!);
    await arm();
    leave(how);

    await act(async () => { pending.resolve(late.stream); });

    expect(late.microphone.readyState).toBe("ended");
    expect(late.camera.readyState).toBe("ended");
    expect(contexts).toHaveLength(0);
    expect(frames.size).toBe(0);
    if (how !== "unmount") {
      click(button("Say it"));
      expect(button("Use my microphone")).toBeDefined();
    }
  });

  // The verifier is real: settling its browser applyConstraints promise must not restart an abandoned meter.
  it.each(["unmount", "Press and hold", "The clip's own sound"] as const)("abandons constraint verification after %s", async (how) => {
    const pending = deferred<void>(), source = capture(pending.promise);
    getUserMedia.mockResolvedValueOnce(source.stream);
    show();
    await arm();
    leave(how);

    expect(source.microphone.readyState).toBe("ended");
    await act(async () => { pending.resolve(); });

    expect(contexts).toHaveLength(0);
    expect(frames.size).toBe(0);
    if (how !== "unmount") {
      click(button("Say it"));
      expect(button("Use my microphone")).toBeDefined();
    }
  });

  it("keeps the newest microphone when an older permission request resolves later", async () => {
    const pending = deferred<MediaStream>(), old = capture(), current = capture();
    getUserMedia.mockReturnValueOnce(pending.promise).mockResolvedValueOnce(current.stream);
    show();
    await arm();
    await arm();
    expect(button("Perform again")).toBeDefined();

    await act(async () => { pending.resolve(old.stream); });

    expect(old.microphone.readyState).toBe("ended");
    expect(current.microphone.readyState).toBe("live");
    expect(contexts).toHaveLength(1);
    expect(frames.size).toBe(1);
    leave("unmount");
    expect(current.microphone.readyState).toBe("ended");
    expect(contexts[0].state).toBe("closed");
    expect(frames.size).toBe(0);
  });

  it("does not revive an older meter when its constraints finish after a new microphone is armed", async () => {
    const pending = deferred<void>(), old = capture(pending.promise), current = capture();
    getUserMedia.mockResolvedValueOnce(old.stream).mockResolvedValueOnce(current.stream);
    show();
    await arm();
    await arm();

    await act(async () => { pending.resolve(); });

    expect(old.microphone.readyState).toBe("ended");
    expect(current.microphone.readyState).toBe("live");
    expect(contexts).toHaveLength(1);
    expect(frames.size).toBe(1);
    expect(button("Perform again")).toBeDefined();
  });

  it("releases an armed microphone when changing cues and ignores an already queued meter tick", async () => {
    const source = capture();
    getUserMedia.mockResolvedValueOnce(source.stream);
    show();
    await arm();
    const queuedTick = [...frames.values()][0];

    click(button("Press and hold"));
    act(() => queuedTick(16));

    expect(source.microphone.readyState).toBe("ended");
    expect(contexts[0].state).toBe("closed");
    expect(frames.size).toBe(0);
    click(button("Say it"));
    expect(button("Use my microphone")).toBeDefined();
    expect(host.querySelector('[aria-label="microphone level"]')).toBeNull();
  });

  it("ignores a rejected permission request after another microphone is armed", async () => {
    const pending = deferred<MediaStream>(), current = capture();
    getUserMedia.mockReturnValueOnce(pending.promise).mockResolvedValueOnce(current.stream);
    show();
    await arm();
    await arm();

    await act(async () => { pending.reject(new DOMException("Denied", "NotAllowedError")); });

    expect(host.querySelector(".bad")).toBeNull();
    expect(current.microphone.readyState).toBe("live");
    expect(button("Perform again")).toBeDefined();
  });

  it("abandons camera constraint verification when the cue changes", async () => {
    const pending = deferred<void>(), source = capture(Promise.resolve(), true);
    source.camera.applyConstraints = () => pending.promise;
    getUserMedia.mockResolvedValueOnce(source.stream);
    show();
    click(host.querySelector("input")!);
    await arm();

    click(button("Press and hold"));
    await act(async () => { pending.resolve(); });

    expect(source.microphone.readyState).toBe("ended");
    expect(source.camera.readyState).toBe("ended");
    expect(contexts).toHaveLength(0);
    click(button("Say it"));
    expect(button("Use my microphone")).toBeDefined();
  });

  it("abandons a microphone request when the face option changes", async () => {
    const pending = deferred<MediaStream>(), source = capture();
    getUserMedia.mockReturnValueOnce(pending.promise);
    show();
    await arm();

    click(host.querySelector("input")!);
    await act(async () => { pending.resolve(source.stream); });

    expect(source.microphone.readyState).toBe("ended");
    expect(contexts).toHaveLength(0);
    expect(button("Use my microphone")).toBeDefined();
  });

  it("releases the acquired stream and audio context if the meter cannot start", async () => {
    const source = capture();
    getUserMedia.mockResolvedValueOnce(source.stream);
    vi.spyOn(MeterAudioContext.prototype, "createAnalyser").mockImplementation(() => { throw new Error("Audio unavailable"); });
    show();
    await arm();

    expect(source.microphone.readyState).toBe("ended");
    expect(contexts[0].state).toBe("closed");
    expect(frames.size).toBe(0);
    expect(host.querySelector(".bad")?.textContent).toContain("Audio unavailable");
    expect(button("Use my microphone")).toBeDefined();
  });

  function pcmRecorder() {
    return {
      ctx: {} as AudioContext,
      ctxToPerfOffsetMs: () => 0,
      stop: vi.fn().mockResolvedValue({ pcm: new Float32Array(480), sampleRate: 48000, firstFrame: 0 }),
    } satisfies PcmRecorder;
  }

  async function armed() {
    const source = capture();
    getUserMedia.mockResolvedValueOnce(source.stream);
    const onDone = show();
    await arm();
    const plate = host.querySelector("video")!;
    vi.spyOn(plate, "play").mockResolvedValue(undefined);
    return { source, onDone, plate };
  }

  it("starts one PCM recorder even when Perform is tapped twice before its worklet loads", async () => {
    const pending = deferred<PcmRecorder>();
    vi.mocked(startPcmRecorder).mockReturnValue(pending.promise);
    await armed();
    await act(async () => { button("Perform again").click(); button("Perform again").click(); });
    expect(startPcmRecorder).toHaveBeenCalledTimes(1);
    expect(button("Perform again").disabled).toBe(true);
    leave("unmount");
    expect(vi.mocked(startPcmRecorder).mock.calls[0][1]?.aborted).toBe(true);
    const recorder = pcmRecorder();
    await act(async () => pending.resolve(recorder));
    expect(recorder.stop).toHaveBeenCalledTimes(1);
  });

  it("shows a recoverable start error and permits another attempt", async () => {
    vi.mocked(startPcmRecorder).mockRejectedValueOnce(new Error("Worklet unavailable"));
    const { onDone, plate } = await armed();
    await act(async () => button("Perform again").click());
    expect(host.querySelector(".bad")?.textContent).toMatch(/recording|microphone/i);
    expect(button("Perform again").disabled).toBe(false);
    const recorder = pcmRecorder();
    vi.mocked(startPcmRecorder).mockResolvedValueOnce(recorder);
    await act(async () => button("Perform again").click());
    expect(plate.play).toHaveBeenCalledTimes(1);
    expect(onDone).not.toHaveBeenCalled();
  });

  it("stops active PCM and frame sampling on unmount", async () => {
    const recorder = pcmRecorder();
    vi.mocked(startPcmRecorder).mockResolvedValueOnce(recorder);
    const { source, onDone, plate } = await armed();
    await act(async () => button("Perform again").click());
    expect(frames.size).toBe(2);
    leave("unmount");
    await act(async () => undefined);
    expect(recorder.stop).toHaveBeenCalledTimes(1);
    expect(source.microphone.readyState).toBe("ended");
    expect(frames.size).toBe(0);
    expect(plate.pause).toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
  });

  it("abandons a finishing take after unmount", async () => {
    const recorder = pcmRecorder(), pending = deferred<Awaited<ReturnType<PcmRecorder["stop"]>>>();
    recorder.stop.mockReturnValueOnce(pending.promise);
    vi.mocked(startPcmRecorder).mockResolvedValueOnce(recorder);
    const { onDone, plate } = await armed();
    await act(async () => button("Perform again").click());
    await act(async () => plate.dispatchEvent(new Event("ended")));
    leave("unmount");
    await act(async () => pending.resolve({ pcm: new Float32Array(480), sampleRate: 48000, firstFrame: 0 }));
    expect(onDone).not.toHaveBeenCalled();
    expect(recorder.stop).toHaveBeenCalledTimes(1);
  });

  it("finishes one microphone take when playback ends and can perform again", async () => {
    const recorder = pcmRecorder();
    vi.mocked(startPcmRecorder).mockResolvedValue(recorder);
    const { onDone, plate } = await armed();
    await act(async () => button("Perform again").click());
    await act(async () => { plate.dispatchEvent(new Event("ended")); plate.dispatchEvent(new Event("ended")); });
    expect(recorder.stop).toHaveBeenCalledTimes(1);
    expect(onDone).toHaveBeenCalledTimes(1);
    const take = onDone.mock.calls[0][0];
    expect(take.voice && "file" in take.voice && take.voice.file.type).toBe("audio/wav");
    expect(button("Perform again").disabled).toBe(false);
    expect(frames.size).toBe(1); // only the armed microphone meter remains
    URL.revokeObjectURL(take.previewUrl);
  });

  it("cleans up PCM if playback cannot start, without submitting a partial take", async () => {
    const recorder = pcmRecorder();
    vi.mocked(startPcmRecorder).mockResolvedValueOnce(recorder);
    const { onDone, plate } = await armed();
    vi.mocked(plate.play).mockRejectedValueOnce(new DOMException("Playback blocked", "NotAllowedError"));
    await act(async () => button("Perform again").click());
    expect(recorder.stop).toHaveBeenCalledTimes(1);
    expect(frames.size).toBe(1);
    expect(onDone).not.toHaveBeenCalled();
    expect(host.querySelector(".bad")?.textContent).toMatch(/could not start/i);
    expect(button("Perform again").disabled).toBe(false);
  });

  it("stops the face recorder along with PCM when an active take is abandoned", async () => {
    const source = capture(Promise.resolve(), true), recorder = pcmRecorder();
    const faceRecorder = { state: "inactive", start: vi.fn(() => { faceRecorder.state = "recording"; }),
      stop: vi.fn(() => { faceRecorder.state = "inactive"; }) };
    vi.stubGlobal("MediaRecorder", class {
      static isTypeSupported() { return true; }
      constructor() { return faceRecorder; }
    });
    getUserMedia.mockResolvedValueOnce(source.stream);
    vi.mocked(startPcmRecorder).mockResolvedValueOnce(recorder);
    const onDone = show();
    click(host.querySelector("input")!);
    await arm();
    vi.spyOn(host.querySelector("video")!, "play").mockResolvedValue(undefined);
    await act(async () => button("Perform again").click());
    expect(faceRecorder.start).toHaveBeenCalledTimes(1);
    leave("unmount");
    await act(async () => undefined);
    expect(faceRecorder.stop).toHaveBeenCalledTimes(1);
    expect(recorder.stop).toHaveBeenCalledTimes(1);
    expect(source.camera.readyState).toBe("ended");
    expect(onDone).not.toHaveBeenCalled();
  });
});

describe("Performer touch holds", () => {
  it.each(["ended", "timeupdate", "pointercancel"])("finalizes an active hold on %s within the shot boundary", async (event) => {
    const onDone = show();
    click(button("Press and hold"));
    const plate = host.querySelector("video")!;
    Object.defineProperty(plate, "duration", { value: 12 });
    vi.spyOn(plate, "play").mockResolvedValue(undefined);
    vi.spyOn(plate, "paused", "get").mockReturnValue(false);
    await act(async () => button("Play the clip").click());
    plate.currentTime = 9.96;
    act(() => button("Hold").dispatchEvent(new Event("pointerdown", { bubbles: true })));
    plate.currentTime = event === "pointercancel" ? 9.97 : 10.2;
    act(() => (event === "pointercancel" ? button("Holding…") : plate).dispatchEvent(new Event(event, { bubbles: true })));
    if (event === "pointercancel") act(() => plate.dispatchEvent(new Event("ended")));
    expect(button("Hold").getAttribute("aria-pressed")).toBe("false");
    click(button("Use 1 hold"));
    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ cue: "hold", holds: [[9.9, 10]] }));
  });

  it("clamps a hold to a clip shorter than ten seconds", async () => {
    const onDone = show();
    click(button("Press and hold"));
    const plate = host.querySelector("video")!;
    Object.defineProperty(plate, "duration", { value: 3 });
    vi.spyOn(plate, "play").mockResolvedValue(undefined);
    vi.spyOn(plate, "paused", "get").mockReturnValue(false);
    await act(async () => button("Play the clip").click());
    plate.currentTime = 2.98;
    act(() => button("Hold").dispatchEvent(new Event("pointerdown", { bubbles: true })));
    plate.currentTime = 3;
    act(() => plate.dispatchEvent(new Event("ended")));
    act(() => button("Hold").dispatchEvent(new Event("pointerup", { bubbles: true })));
    click(button("Use 1 hold"));
    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ holds: [[2.9, 3]] }));
  });
});
