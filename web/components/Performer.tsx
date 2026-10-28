"use client";

import { type CSSProperties, useCallback, useEffect, useRef, useState } from "react";
import { detectOnsetPCM } from "@/lib/audio/onset";
import { type PcmRecorder, startPcmRecorder } from "@/lib/audio/pcmRecorder";
import { encodeWav } from "@/lib/audio/wav";
import { applyAndVerify, AUDIO_CONSTRAINTS, videoConstraints } from "@/lib/capture/constraints";
import { pickMime } from "@/lib/capture/pickMime";
import { type ClockSample, fitMediaClock, fitMediaClockNear, perfMsOfSample, sampleClock } from "@/lib/sync/mediaClock";
import { type CueKind, quickPerformance, EFFECT_LABELS, isExperimentalEffect, type SampleVoice, sceneEffect, type SceneMap, type Source, suggestEffect } from "@/lib/shot";
import { buildTimingJson, estimateOffsetS, plausibleOffset, type TimingJson } from "@/lib/sync/timing";

export interface Performance {
  cue: CueKind;
  voice: Source | null; // a voice cue's recording or sample; none for the clip's own sound or a hold
  holds?: [number, number][]; // a hold cue: press and release times on the plate (s)
  timing: TimingJson | null;
  face: Blob | null;
  faceExt: string;
  suggested: string;
  experimental?: boolean; // explicitly chosen from the experimental sample voices
  onsetS: number | null;
  warnings: string[];
  label: string; // "your take" or the sample's name
  previewUrl: string;
}

type Step = "idle" | "armed" | "starting" | "performing" | "finishing";

const MAX_SHOT_S = 10;

/**
 * Step 3: cue it. Say it (perform the noise over the playing plate: mic, optional face camera, or a sample
 * noise), use a sound already in the clip (a snap, a click, a slam), or press and hold while it plays.
 */
export default function Performer({ plateUrl, point, voices, takes, clipSound, scene, ready, live = true, drawn = false, hand = false, filmed = false, replay = false, quick = false, quickEffect = "fire_burst", onDone }: {
  plateUrl: string;
  point: [number, number];
  voices: SampleVoice[];
  takes: number;
  clipSound: boolean; // the plate is the user's own clip (it may carry a sound to lock to)
  filmed?: boolean; // filmed in the studio with its noise in it: its own sound is the cue to start with
  replay?: boolean; // performing a finished shot again accepts a new voice or hold cue
  quick?: boolean; // compose the chosen element as soon as the clip ends
  quickEffect?: string;
  scene?: SceneMap; // a sample plate's map: what each noise does to it
  ready?: string[]; // sample noises whose scene shot was made ahead of time (the GPU is offline)
  live?: boolean; // the GPU renders into the scene right now
  drawn?: boolean; // the source is a drawing: it renders only into the scene (no instant shot)
  hand?: boolean; // the source is your hand: the hand and the noise pick the effect together
  onDone: (p: Performance) => void;
}) {
  const choice = quickPerformance(quickEffect);
  const elementName = choice.element.toLowerCase();
  // with the GPU offline, a drawing plays only as the shot made ahead of time with its own noise
  const onlyReady = drawn && !live;
  const why = "With the GPU offline, a drawing plays only with the noise it was made ahead of time with";
  const [kind, setKind] = useState<CueKind>(!replay && filmed && clipSound ? "clip" : "voice");
  const [holds, setHolds] = useState<[number, number][]>([]);
  const [holding, setHolding] = useState(false);
  const [holdPlaying, setHoldPlaying] = useState(false); // the hold pass plays the clip without the microphone
  const holdStart = useRef<number | null>(null);
  const [step, setStep] = useState<Step>("idle");
  const [withFace, setWithFace] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [level, setLevel] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const faceRef = useRef<HTMLVideoElement>(null);
  const plateRef = useRef<HTMLVideoElement>(null);
  const [plateWide, setPlateWide] = useState(false); // a landscape clip plays in a landscape box, not a tall one
  const pcmRef = useRef<PcmRecorder | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const faceChunks = useRef<Blob[]>([]);
  const mimeRef = useRef<string | null>(null);
  const clockSamples = useRef<ClockSample[]>([]);
  const stopClock = useRef<(() => void) | null>(null);
  const ctxOffsetMs = useRef(0);
  const stopwatch = useRef<{ pcm: number; play: number | null }>({ pcm: 0, play: null }); // performance.now() stamps
  const inputLatencyS = useRef(0.02);
  const finishing = useRef(false);
  const meter = useRef<{ ctx: AudioContext; raf: number | null } | null>(null);
  const captureGeneration = useRef(0);
  const takeGeneration = useRef(0);
  const takeAbort = useRef<AbortController | null>(null);
  const takeBusy = useRef(false);
  const activePlate = useRef<HTMLVideoElement | null>(null);
  const busy = step === "starting" || step === "performing" || step === "finishing";

  const stopPlayback = useCallback(() => {
    stopClock.current?.();
    stopClock.current = null;
    const plate = activePlate.current;
    activePlate.current = null;
    if (plate) {
      plate.onended = null;
      plate.ontimeupdate = null;
      plate.pause();
    }
  }, []);

  const discardTake = useCallback(() => {
    ++takeGeneration.current;
    takeAbort.current?.abort();
    takeAbort.current = null;
    takeBusy.current = false;
    finishing.current = false;
    stopPlayback();
    const pcm = pcmRef.current;
    pcmRef.current = null;
    if (pcm) void pcm.stop().catch(() => undefined);
    const rec = recorderRef.current;
    recorderRef.current = null;
    if (rec && rec.state !== "inactive") rec.stop();
    holdStart.current = null;
  }, [stopPlayback]);

  const release = useCallback(() => {
    ++captureGeneration.current; // also abandon permission and constraint requests still in flight
    discardTake();
    if (meter.current) {
      if (meter.current.raf !== null) cancelAnimationFrame(meter.current.raf);
      void meter.current.ctx.close().catch(() => undefined);
      meter.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, [discardTake]);

  const changeKind = (next: CueKind) => {
    if (next === kind) return;
    release();
    setStep("idle");
    setLevel(0);
    setNotes([]);
    setError(null);
    setHolding(false);
    setHoldPlaying(false);
    setKind(next);
  };

  useEffect(() => release, [release]);

  useEffect(() => {
    if (step !== "idle" && withFace && faceRef.current && streamRef.current) faceRef.current.srcObject = streamRef.current;
  }, [step, withFace]);

  const startMeter = (stream: MediaStream, generation: number) => {
    const ctx = new AudioContext();
    const activeMeter = { ctx, raf: null as number | null };
    meter.current = activeMeter; // release the context even if graph setup fails
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const tick = () => {
      if (captureGeneration.current !== generation) return;
      analyser.getFloatTimeDomainData(buf);
      let s = 0;
      for (const v of buf) s += v * v;
      const db = 20 * Math.log10(Math.sqrt(s / buf.length) + 1e-9);
      setLevel(Math.min(1, Math.max(0, (db + 60) / 54)));
      activeMeter.raf = requestAnimationFrame(tick);
    };
    activeMeter.raf = requestAnimationFrame(tick);
  };

  const arm = async () => {
    setError(null);
    release(); // never orphan an earlier stream or meter
    const generation = captureGeneration.current;
    const gone = () => captureGeneration.current !== generation;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS, video: withFace ? videoConstraints("user") : false });
      if (gone()) { stream.getTracks().forEach((t) => t.stop()); return; }
      streamRef.current = stream;
      const found: string[] = [];
      const [audio] = stream.getAudioTracks();
      const [video] = stream.getVideoTracks();
      if (audio) {
        const r = await applyAndVerify(audio, AUDIO_CONSTRAINTS);
        if (gone()) return;
        found.push(...r.mismatches.map((m) => `mic: ${m}`));
        const lat = (r.settings as MediaTrackSettings & { latency?: number }).latency;
        inputLatencyS.current = typeof lat === "number" ? lat : 0.02;
      }
      if (video) {
        const r = await applyAndVerify(video, videoConstraints("user"));
        if (gone()) return;
        found.push(...r.mismatches.map((m) => `camera: ${m}`));
      }
      setNotes(found);
      startMeter(stream, generation);
      setStep("armed");
    } catch (err) {
      if (gone()) return;
      release();
      const name = err instanceof DOMException ? err.name : "";
      setError(
        name === "NotAllowedError" || name === "SecurityError"
          ? `${withFace ? "Camera/microphone" : "Microphone"} permission was denied. Allow it in your browser's site settings, or use a sample noise below.`
          : name === "NotFoundError"
            ? "No microphone was found. Use a sample noise below."
            : `Could not start the microphone (${name || String(err)}).`,
      );
    }
  };

  const finish = useCallback(async () => {
    if (finishing.current || !takeBusy.current) return;
    finishing.current = true;
    const generation = takeGeneration.current;
    const gone = () => takeGeneration.current !== generation;
    setStep("finishing");
    stopPlayback();
    const warnings: string[] = [];
    const chunks = faceChunks.current;
    const mime = mimeRef.current;
    const rec = recorderRef.current;
    recorderRef.current = null;
    const activePcm = pcmRef.current;
    pcmRef.current = null;
    if (activePcm) ctxOffsetMs.current = activePcm.ctxToPerfOffsetMs(); // now the context is surely running
    try {
      const faceStopped = rec && rec.state !== "inactive"
        ? new Promise<void>((resolve) => { rec.onstop = () => resolve(); rec.stop(); })
        : Promise.resolve();
      const [pcm] = await Promise.all([
        activePcm ? activePcm.stop() : Promise.resolve({ pcm: new Float32Array(0), sampleRate: 48000, firstFrame: 0 }),
        faceStopped,
      ]);
      if (gone()) return;
      const faceBlob = rec ? new Blob(chunks, { type: mime ?? undefined }) : null;
      const onsetS = detectOnsetPCM(pcm.pcm, pcm.sampleRate);

      let faceOffsetS: number | null = null;
      if (withFace && faceBlob && faceBlob.size > 0) {
        try {
          const decoded = await new OfflineAudioContext(1, 1, pcm.sampleRate).decodeAudioData(await faceBlob.arrayBuffer());
          faceOffsetS = estimateOffsetS(pcm.pcm, decoded.getChannelData(0), pcm.sampleRate);
          if (faceOffsetS === null) warnings.push("Couldn't sync your face video to the mic; the shot goes without the inset.");
        } catch {
          warnings.push("Couldn't read your face video; the shot goes without the inset.");
        }
        if (gone()) return;
      }
      let timing: TimingJson | null = null;
      if (clockSamples.current.length >= 2) {
        timing = buildTimingJson({
          // the plate's clock where the noise is (the whole-take fit is the fallback when no onset was heard)
          clock: onsetS === null ? fitMediaClock(clockSamples.current)
            : fitMediaClockNear(clockSamples.current, perfMsOfSample(pcm.firstFrame, onsetS * pcm.sampleRate, pcm.sampleRate, ctxOffsetMs.current)),
          firstFrame: pcm.firstFrame, sampleRate: pcm.sampleRate,
          ctxToPerfOffsetMs: ctxOffsetMs.current, inputLatencyS: inputLatencyS.current, onsetSInNoise: onsetS,
          faceOffsetS, emissionPoint: point,
        });
        const sw = stopwatch.current;
        const guard = plausibleOffset(timing.noise_offset_s, sw.play === null ? 0 : -(sw.play - sw.pcm) / 1000);
        if (!guard.fitted) {
          timing = { ...timing, noise_offset_s: guard.offsetS };
          warnings.push("The clip's clock couldn't be read exactly; your noise is placed by a stopwatch (within about 50 ms).");
        }
      } else {
        warnings.push("The clip's timing wasn't captured; the noise is placed from the start of the clip.");
      }
      const voice = new Blob([encodeWav(pcm.pcm, pcm.sampleRate) as BlobPart], { type: "audio/wav" });
      const face = faceOffsetS !== null && faceBlob && mime ? faceBlob : null;
      setStep("armed");
      finishing.current = false;
      takeBusy.current = false;
      onDone({
        cue: "voice", voice: { file: voice, filename: "mic.wav" }, timing, face, faceExt: mime?.includes("mp4") ? "mp4" : "webm",
        suggested: suggestEffect(pcm.pcm, pcm.sampleRate, onsetS), onsetS, warnings, label: `take ${takes + 1}`,
        previewUrl: URL.createObjectURL(voice),
      });
    } catch {
      if (gone()) return;
      discardTake();
      setStep("armed");
      setError("The recording couldn't be saved. Try performing again, or use a sample noise.");
    }
  }, [point, withFace, takes, onDone, discardTake, stopPlayback]);

  const perform = async () => {
    const stream = streamRef.current;
    const plate = plateRef.current;
    if (!stream || !plate || takeBusy.current) return;
    takeBusy.current = true; // guard a second tap before React updates the disabled button
    const generation = ++takeGeneration.current;
    const controller = new AbortController();
    takeAbort.current = controller;
    const gone = () => takeGeneration.current !== generation;
    setStep("starting");
    setError(null);
    clockSamples.current = [];
    faceChunks.current = [];
    try {
      const pcm = await startPcmRecorder(stream, controller.signal); // anchor the PCM before the plate starts
      if (gone()) { await pcm.stop(); return; }
      takeAbort.current = null;
      pcmRef.current = pcm;
      activePlate.current = plate;
      stopwatch.current = { pcm: performance.now(), play: null };
      ctxOffsetMs.current = pcm.ctxToPerfOffsetMs();
      mimeRef.current = withFace && typeof MediaRecorder !== "undefined" ? pickMime((t) => MediaRecorder.isTypeSupported(t)) : null;
      if (mimeRef.current) {
        const rec = new MediaRecorder(stream, { mimeType: mimeRef.current });
        recorderRef.current = rec;
        const chunks = faceChunks.current;
        rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
        rec.start();
      } else {
        recorderRef.current = null;
      }
      plate.muted = true;
      plate.currentTime = 0;
      stopClock.current = sampleClock(plate, (s) => clockSamples.current.push(s));
      plate.onended = () => { void finish(); };
      plate.ontimeupdate = () => { // the shot uses the first 10 s of the clip: stop there
        if (plate.currentTime >= MAX_SHOT_S) { plate.pause(); void finish(); }
      };
      setStep("performing");
      await plate.play();
      if (!gone()) stopwatch.current.play = performance.now();
    } catch {
      if (gone()) return;
      discardTake();
      setStep("armed");
      setError("Could not start the recording. Try Perform again, or use a sample noise.");
    }
  };

  const pickSample = (v: SampleVoice) => {
    onDone({ cue: "voice", voice: { sample: v.name }, timing: null, face: null, faceExt: "webm", suggested: v.effect ?? "fireball",
      experimental: isExperimentalEffect(v.effect ?? ""), onsetS: null, warnings: [], label: v.title ?? v.name, previewUrl: v.url });
  };

  const useClipSound = () => {
    onDone({ cue: "clip", voice: null, timing: null, face: null, faceExt: "webm", suggested: "auto", onsetS: null,
      warnings: [], label: "the clip's own sound", previewUrl: plateUrl });
  };

  // press and hold while the clip plays: each press is a hit, each hold how long it lasts
  const lift = () => {
    const plate = plateRef.current;
    const start = holdStart.current;
    holdStart.current = null;
    setHolding(false);
    if (!plate || start === null) return;
    const end = Math.min(MAX_SHOT_S, Number.isFinite(plate.duration) ? plate.duration : MAX_SHOT_S);
    // Keep even a tap at the final frame usable without sending a cue past the clip.
    const t0 = Math.max(0, Math.min(start, end - 0.1));
    const t1 = Math.min(end, Math.max(plate.currentTime, t0 + 0.1));
    if (t1 <= t0) return;
    setHolds((h) => [...h, [Math.round(t0 * 100) / 100, Math.round(t1 * 100) / 100]]);
  };
  const playForHolds = async () => {
    const plate = plateRef.current;
    if (!plate || activePlate.current) return;
    const generation = ++takeGeneration.current;
    activePlate.current = plate;
    setHolds([]);
    setError(null);
    holdStart.current = null;
    setHolding(false);
    plate.muted = true;
    plate.currentTime = 0;
    const endHolds = () => { lift(); stopPlayback(); setHoldPlaying(false); };
    plate.onended = endHolds;
    plate.ontimeupdate = () => { if (plate.currentTime >= MAX_SHOT_S) endHolds(); };
    setHoldPlaying(true);
    try { await plate.play(); } catch {
      if (takeGeneration.current !== generation) return;
      endHolds();
      setError("Could not play the clip. Try playing it again.");
    }
  };
  const press = () => {
    const plate = plateRef.current;
    if (!plate || plate.paused || holdStart.current !== null) return;
    holdStart.current = plate.currentTime;
    setHolding(true);
  };
  const useHolds = () => {
    onDone({ cue: "hold", voice: null, holds, timing: null, face: null, faceExt: "webm",
      suggested: holds.some(([a, b]) => b - a >= 0.4) ? "flame_plume" : "fireball", onsetS: null, warnings: [],
      label: `${holds.length} hold${holds.length === 1 ? "" : "s"}`, previewUrl: plateUrl });
  };

  return (
    <section className={quick ? "quick-performer" : undefined} data-step={step}>
      <h2>{quick ? "Record your rhythm" : "3. Cue it"}</h2>
      {quick && <p className="muted">Allow your microphone, then press Record &amp; make {elementName}. Say <strong>{choice.cue}</strong> as the clip plays. We make your shot when it ends.</p>}
      {!quick && <div className="row" role="radiogroup" aria-label="How to cue it">
        <button role="radio" aria-checked={kind === "voice"} className={kind === "voice" ? "" : "secondary"}
          disabled={busy || holdPlaying} onClick={() => changeKind("voice")}>Say it</button>
        <button role="radio" aria-checked={kind === "clip"} className={kind === "clip" ? "" : "secondary"}
          disabled={busy || holdPlaying || replay || !clipSound || onlyReady} onClick={() => changeKind("clip")}
          title={replay ? "Perform it again with your voice or press-and-hold" : onlyReady ? why : clipSound ? undefined : "This sample has no sound of its own to lock to; your own clip may"}>The clip&apos;s own sound</button>
        <button role="radio" aria-checked={kind === "hold"} className={kind === "hold" ? "" : "secondary"}
          disabled={busy || holdPlaying || onlyReady} onClick={() => changeKind("hold")}
          title={onlyReady ? why : undefined}>Press and hold</button>
      </div>}
      <div className="perform">
        <video ref={plateRef} src={plateUrl} playsInline muted preload="auto" className={plateWide ? "plate wide" : "plate"}
          onLoadedMetadata={(e) => setPlateWide(e.currentTarget.videoWidth > e.currentTarget.videoHeight)} />
        {withFace && step !== "idle" && kind === "voice" && <video ref={faceRef} autoPlay playsInline muted className="face" />}
      </div>
      {kind === "clip" && (
        <>
          <p className="muted">The effect locks to the sound that&apos;s already in your clip: a finger snap, a lighter&apos;s click, a door slam, a beat. The shot keeps that sound.</p>
          <div className="row"><button onClick={useClipSound}>Lock it to my clip&apos;s sound</button></div>
        </>
      )}
      {kind === "hold" && (
        <>
          <p className="muted">No sound needed: play the clip and hold the button while the effect should be there. Press briefly for a pop, hold for a flame.</p>
          <div className="row">
            <button className="secondary" disabled={holdPlaying} onClick={() => void playForHolds()}>{holds.length ? "Play again" : "Play the clip"}</button>
            <button aria-pressed={holding} disabled={!holdPlaying} onPointerDown={press} onPointerUp={lift} onPointerCancel={lift}
              onPointerLeave={() => { if (holding) lift(); }} style={{ minWidth: 160, touchAction: "none" }}>
              {holding ? "Holding…" : "Hold"}
            </button>
            <button disabled={holdPlaying || holds.length === 0} onClick={useHolds}>Use {holds.length || "my"} hold{holds.length === 1 ? "" : "s"}</button>
          </div>
          {holds.length > 0 && <p className="muted">{holds.map(([a, b]) => `${a.toFixed(2)}–${b.toFixed(2)} s`).join(" · ")}</p>}
        </>
      )}
      {kind === "voice" && step === "idle" && (
        <>
          <div className="row">
            <button disabled={onlyReady} title={onlyReady ? why : undefined} onClick={() => void arm()}>Use my microphone</button>
            {!quick && <label className="muted"><input type="checkbox" checked={withFace} onChange={(e) => { release(); setWithFace(e.target.checked); }} /> add my face to the shot</label>}
          </div>
          <p className="muted">{onlyReady ? `${why}: pick it below.` : withFace ? "Your face video is uploaded with your noise and shown as an inset in the final shot." : "Only your noise is uploaded."}</p>
        </>
      )}
      {kind === "voice" && step !== "idle" && (
        <>
          {/* the level drives the bar and the record button's glow (globals.css: .meter, .record-cta) */}
          <div className="meter" aria-label="microphone level" style={{ "--level": Math.round(level * 100) / 100 } as CSSProperties}><span /></div>
          {notes.map((n) => <span key={n} className="chip warn">{n}</span>)}
          <div className="row">
            <button className="record-cta" style={{ "--level": Math.round(level * 100) / 100 } as CSSProperties} disabled={busy}
              onClick={() => void perform()}>{quick ? `Record & make ${elementName}` : takes > 0 ? "Perform again" : "Perform"}</button>
            <span className="muted rec-status">{step === "starting" ? "Starting the microphone…" : step === "finishing" ? "Saving your recording…" : step === "performing" ? "Recording: make the noise when it should hit!" : "The clip plays muted. Make your noise at the moment it should happen."}</span>
          </div>
        </>
      )}
      {error && <p className="chip bad">{error}</p>}
      {kind === "voice" && voices.length > 0 && (() => {
        // what each noise makes right now: with the GPU offline, the real render only where it was made ahead of time
        const isReady = (v: SampleVoice) => !!ready?.includes(v.name);
        const makes = (v: SampleVoice): string | undefined => {
          if (hand) return undefined; // what it makes depends on the hand too
          const inScene = sceneEffect(scene, v) ?? v.effect;
          if (live || isReady(v)) return inScene && EFFECT_LABELS[inScene]?.name;
          return v.effect && `${EFFECT_LABELS[v.effect]?.name ?? v.effect}, instant`;
        };
        const sample = (v: SampleVoice) => {
          const off = onlyReady && !isReady(v); // a drawing has no instant shot: this one needs the GPU
          return (
            <button key={v.name} className="secondary" disabled={busy || off} title={off ? why : undefined}
              onClick={() => pickSample(v)}>
              {v.title ?? v.name}
              {isExperimentalEffect(v.effect ?? "") && <span className="muted"> (Experimental)</span>}
              <span className="muted" style={{ marginLeft: 6 }}>{off ? "needs the GPU" : makes(v)}</span>
            </button>
          );
        };
        const offered = voices.filter((v) => quick ? choice.voices.includes(v.name) : hand || !v.hand);
        const experimental = offered.filter((v) => isExperimentalEffect(v.effect ?? "") || isExperimentalEffect(sceneEffect(scene, v) ?? ""));
        const standard = quick ? offered : offered.filter((v) => !experimental.includes(v));
        const real = live ? [] : standard.filter(isReady);
        const samples = (
          <>
            {real.length > 0 && (
              <div className="row">
                <span className="muted">plays the real LTX-2.5 render, made ahead of time:</span>
                {real.map(sample)}
              </div>
            )}
            <div className="row">
              <span className="muted">{hand ? "or use a sample noise:" : onlyReady ? "the others need the GPU:" : real.length ? "or, as an instant shot (the GPU is offline):" : live ? "or use a sample noise:" : "or use a sample noise (an instant shot: the GPU is offline):"}</span>
              {standard.filter((v) => !real.includes(v)).map(sample)}
            </div>
            {!quick && experimental.length > 0 && <details>
              <summary>Experimental</summary>
              <p className="muted">Water, air and earth sample voices. Choose one explicitly to try its effect.</p>
              <div className="row">{experimental.map(sample)}</div>
            </details>}
          </>
        );
        return quick ? <details className="quick-fallback" open={!!error}><summary>Try a sample voice instead</summary>
          <p className="muted">These are recorded examples, not your microphone. Choose one to try its performance.</p>{samples}</details> : samples;
      })()}
    </section>
  );
}
