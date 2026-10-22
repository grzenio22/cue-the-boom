import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { startPcmRecorder } from "./pcmRecorder";

describe("PCM recorder resources", () => {
  const stream = {} as MediaStream;
  let ctx: {
    state: string; sampleRate: number; destination: object;
    close: ReturnType<typeof vi.fn>; resume: ReturnType<typeof vi.fn>;
    audioWorklet: { addModule: ReturnType<typeof vi.fn> };
    createMediaStreamSource: ReturnType<typeof vi.fn>; createGain: ReturnType<typeof vi.fn>;
  };
  let source: { connect: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> };
  let worklet: { connect: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn>; port: { onmessage: ((event: MessageEvent) => void) | null } };
  let gain: { gain: { value: number }; connect: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    source = { connect: vi.fn(), disconnect: vi.fn() };
    worklet = { connect: vi.fn(), disconnect: vi.fn(), port: { onmessage: null } };
    gain = { gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() };
    ctx = { state: "suspended", sampleRate: 44100, destination: {}, close: vi.fn().mockResolvedValue(undefined),
      resume: vi.fn().mockResolvedValue(undefined), audioWorklet: { addModule: vi.fn().mockResolvedValue(undefined) },
      createMediaStreamSource: vi.fn(() => source), createGain: vi.fn(() => gain) };
    vi.stubGlobal("AudioContext", class { constructor() { return ctx; } });
    vi.stubGlobal("AudioWorkletNode", class { constructor() { return worklet; } });
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

  it.each(["module", "resume", "graph"])("closes the context if %s setup fails", async (stage) => {
    const failure = new Error(`${stage} unavailable`);
    if (stage === "module") ctx.audioWorklet.addModule.mockRejectedValueOnce(failure);
    if (stage === "resume") ctx.resume.mockRejectedValueOnce(failure);
    if (stage === "graph") gain.connect.mockImplementationOnce(() => { throw failure; });
    await expect(startPcmRecorder(stream)).rejects.toBe(failure);
    expect(ctx.close).toHaveBeenCalledTimes(1);
    if (stage === "graph") {
      expect(source.disconnect).toHaveBeenCalled();
      expect(worklet.disconnect).toHaveBeenCalled();
      expect(gain.disconnect).toHaveBeenCalled();
      expect(worklet.port.onmessage).toBeNull();
    }
  });

  it("resumes in the user gesture before the worklet download finishes", async () => {
    let loaded!: () => void;
    ctx.audioWorklet.addModule.mockReturnValueOnce(new Promise<void>((resolve) => { loaded = resolve; }));
    const pending = startPcmRecorder(stream);
    expect(ctx.resume).toHaveBeenCalledTimes(1);
    loaded();
    const recorder = await pending;
    await recorder.stop();
  });

  it("closes a cancelled startup immediately while its worklet download is still pending", async () => {
    let loaded!: () => void;
    ctx.audioWorklet.addModule.mockReturnValueOnce(new Promise<void>((resolve) => { loaded = resolve; }));
    const controller = new AbortController();
    const pending = startPcmRecorder(stream, controller.signal);
    const rejected = expect(pending).rejects.toMatchObject({ name: "AbortError" });
    controller.abort();
    expect(ctx.close).toHaveBeenCalledTimes(1);
    await rejected;
    loaded();
    await Promise.resolve();
    expect(ctx.createMediaStreamSource).not.toHaveBeenCalled();
    expect(ctx.close).toHaveBeenCalledTimes(1);
  });

  it("stops idempotently and preserves the final PCM at the device's sample rate", async () => {
    vi.useFakeTimers();
    const recorder = await startPcmRecorder(stream);
    worklet.port.onmessage?.({ data: { frame: 100, samples: new Float32Array([0.1, 0.2]) } } as MessageEvent);
    const first = recorder.stop(), second = recorder.stop();
    await vi.runAllTimersAsync();
    expect(await first).toEqual({ pcm: new Float32Array([0.1, 0.2]), firstFrame: 100, sampleRate: 44100 });
    expect(await second).toEqual(await first);
    expect(ctx.close).toHaveBeenCalledTimes(1);
    expect(source.disconnect).toHaveBeenCalledTimes(1);
    expect(worklet.disconnect).toHaveBeenCalledTimes(1);
    expect(gain.disconnect).toHaveBeenCalledTimes(1);
  });
});
