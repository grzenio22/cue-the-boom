"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import ShotPanel from "@/components/ShotPanel";
import TapPoint from "@/components/TapPoint";
import { detectOnsetPCM } from "@/lib/audio/onset";
import { type PcmRecorder, startPcmRecorder } from "@/lib/audio/pcmRecorder";
import { encodeWav } from "@/lib/audio/wav";
import { applyAndVerify, AUDIO_CONSTRAINTS, videoConstraints } from "@/lib/capture/constraints";
import { pickMime } from "@/lib/capture/pickMime";
import { type ClockSample, fitMediaClock, sampleClock } from "@/lib/sync/mediaClock";
import { suggestEffect } from "@/lib/shot";
import { buildTimingJson, estimateOffsetS, type TimingJson } from "@/lib/sync/timing";

type Step = "choose" | "tap" | "armed" | "performing" | "result";

interface Result {
  onsetS: number | null;
  faceOffsetS: number | null;
  timing: TimingJson | null;
  micUrl: string;
  faceUrl: string | null;
  faceExt: string;
  timingUrl: string | null;
  warnings: string[];
  voice: Blob;
  face: Blob | null;
  suggested: string;
}

const MIN_PLATE_S = 1;
const MAX_PLATE_S = 15;

function probeDuration(url: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => resolve(v.duration);
    v.onerror = () => reject(new Error("could not read this video"));
    v.src = url;
  });
}

export default function DubPage() {
  const [step, setStep] = useState<Step>("choose");
  const [plateUrl, setPlateUrl] = useState<string | null>(null);
  const [plateError, setPlateError] = useState<string | null>(null);
  const [point, setPoint] = useState<[number, number] | null>(null);
  const [armError, setArmError] = useState<string | null>(null);
  const [chips, setChips] = useState<string[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [plateFile, setPlateFile] = useState<File | null>(null);

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

  useEffect(() => () => { streamRef.current?.getTracks().forEach((t) => t.stop()); }, []);

  useEffect(() => {
    if ((step === "armed" || step === "performing") && faceRef.current && streamRef.current) {
      faceRef.current.srcObject = streamRef.current;
    }
  }, [step]);

  const onPlate = async (file: File | undefined) => {
    setPlateError(null);
    if (!file) return;
    const url = URL.createObjectURL(file);
    try {
      const d = await probeDuration(url);
      if (!Number.isFinite(d) || d < MIN_PLATE_S || d > MAX_PLATE_S) {
        URL.revokeObjectURL(url);
        setPlateError(`Plate must be ${MIN_PLATE_S}–${MAX_PLATE_S} s long (this one is ${Number.isFinite(d) ? d.toFixed(1) : "?"} s).`);
        return;
      }
      setPlateUrl(url);
      setPlateFile(file);
      setPoint(null);
      setStep("tap");
    } catch {
      URL.revokeObjectURL(url);
      setPlateError("Could not read that video. Try an MP4 or MOV from your phone.");
    }
  };

  const arm = async () => {
    setArmError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS, video: videoConstraints("user") });
      streamRef.current = stream;
      const notes: string[] = [];
      const [audio] = stream.getAudioTracks();
      const [video] = stream.getVideoTracks();
      if (audio) {
        const r = await applyAndVerify(audio, AUDIO_CONSTRAINTS);
        notes.push(...r.mismatches.map((m) => `mic: ${m}`));
        const lat = (r.settings as MediaTrackSettings & { latency?: number }).latency;
        inputLatencyS.current = typeof lat === "number" ? lat : 0.02;
      }
      if (video) {
        const r = await applyAndVerify(video, videoConstraints("user"));
        notes.push(...r.mismatches.map((m) => `camera: ${m}`));
      }
      setChips(notes);
      setStep("armed");
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      setArmError(
        name === "NotAllowedError" || name === "SecurityError"
          ? "Camera/microphone permission was denied. Allow access in your browser's site settings, then retry."
          : name === "NotFoundError"
            ? "No camera or microphone was found."
            : `Could not start the camera/microphone (${name || String(err)}).`,
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
    if (!mimeRef.current) {
      warnings.push("Face video unavailable: this browser has no supported recording format. mic.wav and timing.json are still saved.");
    } else if (faceBlob && faceBlob.size > 0) {
      try {
        // decode at the PCM's own rate so both signals share one sample clock
        const decoded = await new OfflineAudioContext(1, 1, pcm.sampleRate).decodeAudioData(await faceBlob.arrayBuffer());
        faceOffsetS = estimateOffsetS(pcm.pcm, decoded.getChannelData(0), pcm.sampleRate);
        if (faceOffsetS === null) warnings.push("Face sync unavailable: couldn't match the face audio to the mic.");
      } catch {
        warnings.push("Face sync unavailable: couldn't decode the face recording's audio.");
      }
    }

    let timing: TimingJson | null = null;
    if (clockSamples.current.length >= 2) {
      timing = buildTimingJson({
        clock: fitMediaClock(clockSamples.current),
        firstFrame: pcm.firstFrame,
        sampleRate: pcm.sampleRate,
        ctxToPerfOffsetMs: ctxOffsetMs.current,
        inputLatencyS: inputLatencyS.current,
        onsetSInNoise: onsetS,
        faceOffsetS,
        emissionPoint: point,
      });
    } else {
      warnings.push("Not enough plate frames were timed, so timing.json couldn't be built. Try again.");
    }

    const voice = new Blob([encodeWav(pcm.pcm, pcm.sampleRate) as BlobPart], { type: "audio/wav" });
    const micUrl = URL.createObjectURL(voice);
    const face = faceBlob && mimeRef.current && faceBlob.size > 0 ? faceBlob : null;
    const faceUrl = face ? URL.createObjectURL(face) : null;
    const timingUrl = timing ? URL.createObjectURL(new Blob([JSON.stringify(timing, null, 2)], { type: "application/json" })) : null;
    const suggested = suggestEffect(pcm.pcm, pcm.sampleRate, onsetS);
    setResult({ onsetS, faceOffsetS, timing, micUrl, faceUrl, faceExt: mimeRef.current?.includes("mp4") ? "mp4" : "webm",
      timingUrl, warnings, voice, face, suggested });
    setStep("result");
    finishing.current = false;
  }, [point]);

  const perform = async () => {
    const stream = streamRef.current;
    const plate = plateRef.current;
    if (!stream || !plate) return;
    setResult(null);
    clockSamples.current = [];
    faceChunks.current = [];

    // Start the anchors (mic PCM + face recorder) BEFORE the plate, so the noise timeline covers it.
    const pcm = await startPcmRecorder(stream);
    pcmRef.current = pcm;
    ctxOffsetMs.current = pcm.ctxToPerfOffsetMs();

    mimeRef.current = typeof MediaRecorder !== "undefined" ? pickMime((t) => MediaRecorder.isTypeSupported(t)) : null;
    if (mimeRef.current) {
      const rec = new MediaRecorder(stream, { mimeType: mimeRef.current }); // video + mic track, for face sync
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
    setStep("performing");
    try {
      await plate.play();
    } catch {
      void finish();
    }
  };

  const redo = () => {
    if (result) [result.micUrl, result.faceUrl, result.timingUrl].forEach((u) => u && URL.revokeObjectURL(u));
    setResult(null);
    setStep("armed");
  };

  return (
    <main>
      <p className="muted"><Link href="/">Cue the Boom</Link> · Dub Mode</p>
      <h1>Perform over your clip</h1>

      {step === "choose" && (
        <section>
          <p>1. Choose a plate: a 1–15 s clip of your object (pumpkin, mug, toy), filmed with the phone held still.</p>
          <label className="btn">
            Choose a plate
            <input type="file" accept="video/*" hidden onChange={(e) => void onPlate(e.target.files?.[0])} />
          </label>
          {plateError && <p className="chip bad">{plateError}</p>}
        </section>
      )}

      {step === "tap" && plateUrl && (
        <section>
          <p>2. Tap where the effect should come from.</p>
          <TapPoint src={plateUrl} value={point} onChange={setPoint} />
          <p className="muted">Your face is recorded only on this device.</p>
          <div className="row">
            <button disabled={!point} onClick={() => void arm()}>Arm camera &amp; mic</button>
            <button className="secondary" onClick={() => setStep("choose")}>Different plate</button>
          </div>
          {armError && (
            <div className="row"><span className="chip bad">{armError}</span><button className="secondary" onClick={() => void arm()}>Retry</button></div>
          )}
        </section>
      )}

      {(step === "armed" || step === "performing" || step === "result") && plateUrl && (
        <section>
          <div className="row">
            {chips.length === 0 ? <span className="chip">capture settings OK</span> : chips.map((c) => <span key={c} className="chip warn">{c}</span>)}
          </div>
          <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
            <video ref={plateRef} src={plateUrl} playsInline muted style={{ width: 240, aspectRatio: "9 / 16", background: "#000", objectFit: "contain" }} />
            <video ref={faceRef} autoPlay playsInline muted style={{ width: 120, aspectRatio: "3 / 4", background: "#000", objectFit: "cover", transform: "scaleX(-1)", borderRadius: 12 }} />
          </div>
          {step === "armed" && (
            <div className="row">
              <button onClick={() => void perform()}>Perform</button>
              <span className="muted">The clip plays muted. Make your noise when it should hit.</span>
            </div>
          )}
          {step === "performing" && <p className="chip warn">Recording… make the noise!</p>}
        </section>
      )}

      {step === "result" && result && (
        <section>
          <div className="row">
            {result.onsetS === null
              ? <span className="chip bad">No noise detected — try again</span>
              : <span className="chip">Hit at {result.onsetS.toFixed(2)} s</span>}
            {result.faceOffsetS !== null && <span className="chip">face synced ({Math.round(result.faceOffsetS * 1000)} ms)</span>}
            {result.timing && <span className="chip">plate offset {result.timing.noise_offset_s.toFixed(3)} s</span>}
          </div>
          {result.warnings.map((w) => <p key={w} className="chip warn">{w}</p>)}
          <div className="row">
            <button className="secondary" onClick={redo}>Redo noise</button>
            <details>
              <summary className="muted">Raw recordings</summary>
              <div className="row">
                <a className="btn secondary" href={result.micUrl} download="mic.wav">mic.wav</a>
                {result.faceUrl && <a className="btn secondary" href={result.faceUrl} download={`face.${result.faceExt}`}>face.{result.faceExt}</a>}
                {result.timingUrl && <a className="btn secondary" href={result.timingUrl} download="timing.json">timing.json</a>}
              </div>
            </details>
          </div>
        </section>
      )}

      {step === "result" && result && result.onsetS !== null && plateFile && point && (
        <ShotPanel key={result.micUrl} input={{ plate: plateFile, voice: result.voice, face: result.face, faceExt: result.faceExt,
          timing: result.timing, point, suggested: result.suggested }} />
      )}
    </main>
  );
}
