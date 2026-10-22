/** Mic PCM capture via AudioWorklet. The PCM is the timing anchor for the whole Dub Mode take. */

export interface PcmRecording {
  pcm: Float32Array;
  sampleRate: number;
  firstFrame: number; // context sample frame of pcm[0]
}

export interface PcmRecorder {
  ctx: AudioContext;
  /** performance.now() ms of context time 0 *at processing time*: the earliest any block arrived on this thread,
   * less its own context time (a block can't arrive before it was processed; the fastest one arrived within a
   * millisecond or two). The output timestamp, the fallback, depends on the output device and can be far off. */
  ctxToPerfOffsetMs: () => number;
  stop(): Promise<PcmRecording>;
}

export async function startPcmRecorder(stream: MediaStream, signal?: AbortSignal): Promise<PcmRecorder> {
  const abortError = () => new DOMException("Recording cancelled", "AbortError");
  if (signal?.aborted) throw abortError();
  const ctx = new AudioContext(); // never force sampleRate: iOS can throw with a 44.1 kHz mic
  let closed: Promise<void> | null = null;
  const close = () => closed ??= ctx.close();
  let onAbort: (() => void) | undefined;
  const cancelled = new Promise<never>((_, reject) => {
    onAbort = () => { void close().catch(() => undefined); reject(abortError()); };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
  let source: MediaStreamAudioSourceNode | undefined;
  let node: AudioWorkletNode | undefined;
  let mute: GainNode | undefined;
  const disconnect = () => {
    if (node) node.port.onmessage = null;
    for (const part of [source, node, mute]) {
      try { part?.disconnect(); } catch { /* a partially connected graph is already detached */ }
    }
  };
  try {
    // Resume while still handling the tap, before downloading the worklet (required by iOS).
    await Promise.race([cancelled, Promise.all([
      ctx.state === "suspended" ? ctx.resume() : Promise.resolve(),
      ctx.audioWorklet.addModule("/worklets/pcm-recorder.js"),
    ])]);
    source = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, "pcm-recorder", { numberOfInputs: 1, numberOfOutputs: 1 });
    mute = ctx.createGain();
    mute.gain.value = 0; // keep the graph pulled without monitoring the mic
    const chunks: Float32Array[] = [];
    let firstFrame = -1;
    let offsetMs = Infinity;
    node.port.onmessage = (ev: MessageEvent<{ frame: number; samples: Float32Array }>) => {
      if (firstFrame < 0) firstFrame = ev.data.frame;
      chunks.push(ev.data.samples);
      offsetMs = Math.min(offsetMs, performance.now() - ((ev.data.frame + ev.data.samples.length) / ctx.sampleRate) * 1000);
    };
    source.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);
    const activeSource = source, activeNode = node, activeMute = mute;
    let stopped: Promise<PcmRecording> | null = null;

    return {
      ctx,
      ctxToPerfOffsetMs: () => {
        if (Number.isFinite(offsetMs)) return offsetMs;
        const ts = ctx.getOutputTimestamp();
        // both are 0 until the context has rendered its first output: then the timestamp means nothing
        if (!(ts.performanceTime && ts.contextTime)) return performance.now() - ctx.currentTime * 1000;
        const outputSide = ((ctx.baseLatency || 0) + ((ctx as AudioContext & { outputLatency?: number }).outputLatency || 0)) * 1000;
        return ts.performanceTime - ts.contextTime * 1000 - outputSide;
      },
      stop() {
        if (stopped) return stopped;
        stopped = (async () => {
          activeSource.disconnect();
          try {
            await new Promise((r) => setTimeout(r, 50)); // let the last quanta arrive
            const total = chunks.reduce((n, c) => n + c.length, 0);
            const pcm = new Float32Array(total);
            let off = 0;
            for (const c of chunks) { pcm.set(c, off); off += c.length; }
            return { pcm, sampleRate: ctx.sampleRate, firstFrame: Math.max(firstFrame, 0) };
          } finally {
            activeNode.port.onmessage = null;
            activeNode.disconnect();
            activeMute.disconnect();
            await close();
          }
        })();
        return stopped;
      },
    };
  } catch (error) {
    disconnect();
    await close().catch(() => undefined);
    throw error;
  } finally {
    if (onAbort) signal?.removeEventListener("abort", onAbort);
  }
}
