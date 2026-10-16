/** Mic PCM capture via AudioWorklet. The PCM is the timing anchor for the whole Dub Mode take. */

export interface PcmRecording {
  pcm: Float32Array;
  sampleRate: number;
  firstFrame: number; // context sample frame of pcm[0]
}

export interface PcmRecorder {
  ctx: AudioContext;
  /** performance.now() ms of context time 0 *at processing time* (output-timestamp mapping minus output-side latency). */
  ctxToPerfOffsetMs: () => number;
  stop(): Promise<PcmRecording>;
}

export async function startPcmRecorder(stream: MediaStream): Promise<PcmRecorder> {
  const ctx = new AudioContext(); // never force sampleRate: iOS can throw with a 44.1 kHz mic
  await ctx.audioWorklet.addModule("/worklets/pcm-recorder.js");
  if (ctx.state === "suspended") await ctx.resume();
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-recorder", { numberOfInputs: 1, numberOfOutputs: 1 });
  const mute = ctx.createGain();
  mute.gain.value = 0; // keep the graph pulled without monitoring the mic
  const chunks: Float32Array[] = [];
  let firstFrame = -1;
  node.port.onmessage = (ev: MessageEvent<{ frame: number; samples: Float32Array }>) => {
    if (firstFrame < 0) firstFrame = ev.data.frame;
    chunks.push(ev.data.samples);
  };
  source.connect(node);
  node.connect(mute);
  mute.connect(ctx.destination);

  return {
    ctx,
    ctxToPerfOffsetMs: () => {
      const ts = ctx.getOutputTimestamp();
      const outputSide = ((ctx.baseLatency || 0) + ((ctx as AudioContext & { outputLatency?: number }).outputLatency || 0)) * 1000;
      return (ts.performanceTime ?? performance.now()) - (ts.contextTime ?? ctx.currentTime) * 1000 - outputSide;
    },
    async stop() {
      source.disconnect();
      await new Promise((r) => setTimeout(r, 50)); // let the last quanta arrive
      node.port.onmessage = null;
      node.disconnect();
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const pcm = new Float32Array(total);
      let off = 0;
      for (const c of chunks) { pcm.set(c, off); off += c.length; }
      const sampleRate = ctx.sampleRate;
      await ctx.close();
      return { pcm, sampleRate, firstFrame: Math.max(firstFrame, 0) };
    },
  };
}
