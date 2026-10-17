"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { detectOnsetPCM } from "@/lib/audio/onset";
import { type PcmRecorder, startPcmRecorder } from "@/lib/audio/pcmRecorder";
import { encodeWav } from "@/lib/audio/wav";
import { applyAndVerify, AUDIO_CONSTRAINTS, videoConstraints } from "@/lib/capture/constraints";
import { pickMime } from "@/lib/capture/pickMime";
import { type ClockSample, fitMediaClock, sampleClock } from "@/lib/sync/mediaClock";
import { type CueKind, EFFECT_LABELS, type SampleVoice, sceneEffect, type SceneMap, type Source, suggestEffect } from "@/lib/shot";
import { buildTimingJson, estimateOffsetS, type TimingJson } from "@/lib/sync/timing";

export interface Performance {
  cue: CueKind;
  voice: Source | null; // a voice cue's recording or sample; none for the clip's own sound or a hold
  holds?: [number, number][]; // a hold cue: press and release times on the plate (s)
  timing: TimingJson | null;
  face: Blob | null;
  faceExt: string;
  suggested: string;
  onsetS: number | null;
  warnings: string[];
  label: string; // "your take" or the sample's name
  previewUrl: string;
}

type Step = "idle" | "armed" | "performing";

const MAX_SHOT_S = 10;

/**
 * Step 3: cue it. Say it (perform the noise over the playing plate: mic, optional face camera, or a sample
 * noise), use a sound already in the clip (a snap, a click, a slam), or press and hold while it plays.
 */
export default function Performer({ plateUrl, point, voices, takes, clipSound, scene, ready, onDone }: {
  plateUrl: string;
  point: [number, number];
  voices: SampleVoice[];
  takes: number;
  clipSound: boolean; // the plate is the user's own clip (it may carry a sound to lock to)
  scene?: SceneMap; // a sample plate's map: what each noise does to it
  ready?: string[]; // sample noises whose scene shot was made ahead of time (the GPU is offline)
  onDone: (p: Performance) => void;
}) {
  const [kind, setKind] = useState<CueKind>("voice");
  const [holds, setHolds] = useState<[number, number][]>([]);
  const [holding, setHolding] = useState(false);
  const [holdPlaying, setHoldPlaying] = useState(false); // the hold pass plays the clip; the mic stays as it was
  const holdStart = useRef<number | null>(null);
  const [step, setStep] = useState<Step>("idle");
  const [withFace, setWithFace] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [level, setLevel] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const faceRef = useRef<HTMLVideoElement>(null);
  const plateRef = useRef<HTMLVideoElement>(null);
  const pcmRef = useRef<PcmRecorder | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const faceChunks = useRef<Blob[]>([]);
  const mimeRef = useRef<string | null>(null);
  const clockSamples = useRef<ClockSample[]>([]);
  const stopClock = useRef<(() => void) | null>(null);
  const ctxOffsetMs = useRef(0);
  const inputLatencyS = useRef(0.02);
  const finishing = useRef(false);
  const meter = useRef<{ ctx: AudioContext; raf: number } | null>(null);

  const release = useCallback(() => {
    if (meter.current) {
      cancelAnimationFrame(meter.current.raf);
      void meter.current.ctx.close();
      meter.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => release, [release]);

  useEffect(() => {
    if (step !== "idle" && withFace && faceRef.current && streamRef.current) faceRef.current.srcObject = streamRef.current;
  }, [step, withFace]);

  const startMeter = (stream: MediaStream) => {
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      let s = 0;
      for (const v of buf) s += v * v;
      const db = 20 * Math.log10(Math.sqrt(s / buf.length) + 1e-9);
      setLevel(Math.min(1, Math.max(0, (db + 60) / 54)));
      if (meter.current) meter.current.raf = requestAnimationFrame(tick);
    };
    meter.current = { ctx, raf: requestAnimationFrame(tick) };
  };

  const arm = async () => {
    setError(null);
    release(); // never orphan an earlier stream or meter
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS, video: withFace ? videoConstraints("user") : false });
      streamRef.current = stream;
      const found: string[] = [];
      const [audio] = stream.getAudioTracks();
      const [video] = stream.getVideoTracks();
      if (audio) {
        const r = await applyAndVerify(audio, AUDIO_CONSTRAINTS);
        found.push(...r.mismatches.map((m) => `mic: ${m}`));
        const lat = (r.settings as MediaTrackSettings & { latency?: number }).latency;
        inputLatencyS.current = typeof lat === "number" ? lat : 0.02;
      }
      if (video) {
        const r = await applyAndVerify(video, videoConstraints("user"));
        found.push(...r.mismatches.map((m) => `camera: ${m}`));
      }
      setNotes(found);
      startMeter(stream);
      setStep("armed");
    } catch (err) {
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
    if (finishing.current) return;
    finishing.current = true;
    stopClock.current?.();
    const warnings: string[] = [];

    let faceBlob: Blob | null = null;
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      await new Promise<void>((resolve) => { rec.onstop = () => resolve(); rec.stop(); });
      faceBlob = new Blob(faceChunks.current, { type: mimeRef.current ?? undefined });
    }
    const pcm = pcmRef.current ? await pcmRef.current.stop() : { pcm: new Float32Array(0), sampleRate: 48000, firstFrame: 0 };
    pcmRef.current = null;
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
    }
    let timing: TimingJson | null = null;
    if (clockSamples.current.length >= 2) {
      timing = buildTimingJson({
        clock: fitMediaClock(clockSamples.current), firstFrame: pcm.firstFrame, sampleRate: pcm.sampleRate,
        ctxToPerfOffsetMs: ctxOffsetMs.current, inputLatencyS: inputLatencyS.current, onsetSInNoise: onsetS,
        faceOffsetS, emissionPoint: point,
      });
    } else {
      warnings.push("The clip's timing wasn't captured; the noise is placed from the start of the clip.");
    }
    const voice = new Blob([encodeWav(pcm.pcm, pcm.sampleRate) as BlobPart], { type: "audio/wav" });
    const face = faceOffsetS !== null && faceBlob && mimeRef.current ? faceBlob : null;
    setStep("armed");
    finishing.current = false;
    onDone({
      cue: "voice", voice: { file: voice, filename: "mic.wav" }, timing, face, faceExt: mimeRef.current?.includes("mp4") ? "mp4" : "webm",
      suggested: suggestEffect(pcm.pcm, pcm.sampleRate, onsetS), onsetS, warnings, label: `take ${takes + 1}`,
      previewUrl: URL.createObjectURL(voice),
    });
  }, [point, withFace, takes, onDone]);

  const perform = async () => {
    const stream = streamRef.current;
    const plate = plateRef.current;
    if (!stream || !plate) return;
    clockSamples.current = [];
    faceChunks.current = [];
    const pcm = await startPcmRecorder(stream); // the anchors start BEFORE the plate, so they cover all of it
    pcmRef.current = pcm;
    ctxOffsetMs.current = pcm.ctxToPerfOffsetMs();
    mimeRef.current = withFace && typeof MediaRecorder !== "undefined" ? pickMime((t) => MediaRecorder.isTypeSupported(t)) : null;
    if (mimeRef.current) {
      const rec = new MediaRecorder(stream, { mimeType: mimeRef.current });
      rec.ondataavailable = (e) => { if (e.data.size) faceChunks.current.push(e.data); };
      rec.start();
      recorderRef.current = rec;
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
    try {
      await plate.play();
    } catch {
      void finish();
    }
  };

  const pickSample = (v: SampleVoice) => {
    onDone({ cue: "voice", voice: { sample: v.name }, timing: null, face: null, faceExt: "webm", suggested: v.effect ?? "fireball",
      onsetS: null, warnings: [], label: v.title ?? v.name, previewUrl: v.url });
  };

  const useClipSound = () => {
    onDone({ cue: "clip", voice: null, timing: null, face: null, faceExt: "webm", suggested: "auto", onsetS: null,
      warnings: [], label: "the clip's own sound", previewUrl: plateUrl });
  };

  // press and hold while the clip plays: each press is a hit, each hold how long it lasts
  const playForHolds = async () => {
    const plate = plateRef.current;
    if (!plate) return;
    setHolds([]);
    plate.muted = true;
    plate.currentTime = 0;
    plate.onended = () => setHoldPlaying(false);
    plate.ontimeupdate = () => { if (plate.currentTime >= MAX_SHOT_S) { plate.pause(); setHoldPlaying(false); } };
    setHoldPlaying(true);
    try { await plate.play(); } catch { setHoldPlaying(false); }
  };
  const press = () => {
    const plate = plateRef.current;
    if (!plate || plate.paused) return;
    holdStart.current = plate.currentTime;
    setHolding(true);
  };
  const lift = () => {
    const plate = plateRef.current;
    const t0 = holdStart.current;
    holdStart.current = null;
    setHolding(false);
    if (!plate || t0 === null) return;
    const t1 = Math.max(plate.currentTime, t0 + 0.1);
    setHolds((h) => [...h, [Math.round(t0 * 100) / 100, Math.round(t1 * 100) / 100]]);
  };
  const useHolds = () => {
    onDone({ cue: "hold", voice: null, holds, timing: null, face: null, faceExt: "webm",
      suggested: holds.some(([a, b]) => b - a >= 0.4) ? "flame_plume" : "fireball", onsetS: null, warnings: [],
      label: `${holds.length} hold${holds.length === 1 ? "" : "s"}`, previewUrl: plateUrl });
  };

  return (
    <section>
      <h2>3. Cue it</h2>
      <div className="row" role="radiogroup" aria-label="How to cue it">
        <button role="radio" aria-checked={kind === "voice"} className={kind === "voice" ? "" : "secondary"}
          disabled={step === "performing" || holdPlaying} onClick={() => setKind("voice")}>Say it</button>
        <button role="radio" aria-checked={kind === "clip"} className={kind === "clip" ? "" : "secondary"}
          disabled={step === "performing" || holdPlaying || !clipSound} onClick={() => setKind("clip")}>The clip&apos;s own sound</button>
        <button role="radio" aria-checked={kind === "hold"} className={kind === "hold" ? "" : "secondary"}
          disabled={step === "performing" || holdPlaying} onClick={() => setKind("hold")}>Press and hold</button>
      </div>
      <div className="perform">
        <video ref={plateRef} src={plateUrl} playsInline muted preload="auto" className="plate" />
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
            <button aria-pressed={holding} disabled={!holdPlaying} onPointerDown={press} onPointerUp={lift}
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
            <button onClick={() => void arm()}>Use my microphone</button>
            <label className="muted"><input type="checkbox" checked={withFace} onChange={(e) => setWithFace(e.target.checked)} /> add my face to the shot</label>
          </div>
          <p className="muted">{withFace ? "Your face video is uploaded with your noise and shown as an inset in the final shot." : "Only your noise is uploaded."}</p>
        </>
      )}
      {kind === "voice" && step !== "idle" && (
        <>
          <div className="meter" aria-label="microphone level"><span style={{ width: `${Math.round(level * 100)}%` }} /></div>
          {notes.map((n) => <span key={n} className="chip warn">{n}</span>)}
          <div className="row">
            <button disabled={step === "performing"} onClick={() => void perform()}>{takes > 0 ? "Perform again" : "Perform"}</button>
            <span className="muted">{step === "performing" ? "Recording: make the noise when it should hit!" : "The clip plays muted. Make your noise at the moment it should happen."}</span>
          </div>
        </>
      )}
      {error && <p className="chip bad">{error}</p>}
      {kind === "voice" && voices.length > 0 && (
        <div className="row">
          <span className="muted">or use a sample noise:</span>
          {voices.map((v) => (
            <button key={v.name} className="secondary" disabled={step === "performing"} onClick={() => pickSample(v)}>
              {v.title ?? v.name}
              {(sceneEffect(scene, v) ?? v.effect) && <span className="muted" style={{ marginLeft: 6 }}>{EFFECT_LABELS[(sceneEffect(scene, v) ?? v.effect)!]?.name}</span>}
              {ready?.includes(v.name) && <span className="muted" style={{ marginLeft: 6 }}>· ready</span>}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
