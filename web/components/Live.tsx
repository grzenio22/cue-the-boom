"use client";

import { useEffect, useRef, useState } from "react";
import { AUDIO_CONSTRAINTS } from "@/lib/capture/constraints";
import { pickMime } from "@/lib/capture/pickMime";
import { type LiveElement, liveHand, onsetStep, type OnsetState, playLook } from "@/lib/live";

// MediaPipe's hand landmarker in the browser (Apache-2.0), its runtime and model from Google's own hosting
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MAX_REC_S = 8;

interface Play { el: LiveElement; video: HTMLVideoElement; gain: number; size: number; held: boolean; letGo: number | null }
type Landmarks = { x: number; y: number }[];

/** Try it live: your hand on the camera, your noise on the microphone, the fire rendered ahead of time (LTX-2.5)
 * thrown from your palm as you make it, as big as it is loud; a held "fffff" holds a flame (pointing: a jet). A
 * rough preview: Record this, and the shot is made from your take with the hand followed, matted and lit. */
export default function Live({ onTake }: { onTake: (file: File, url: string) => void }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recording, setRecording] = useState<number | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const run = useRef<{ stop: () => void; record: () => void; stopRecording: () => void } | null>(null);

  useEffect(() => () => run.current?.stop(), []);
  useEffect(() => {
    if (recording === null) return;
    if (recording >= MAX_REC_S) { run.current?.stopRecording(); return; }
    const t = setTimeout(() => setRecording((r) => (r === null ? null : r + 0.1)), 100);
    return () => clearTimeout(t);
  }, [recording]);

  const start = async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) { setError("No camera here: the page needs localhost or HTTPS."); return; }
    setOpen(true);
    setStatus("Starting the camera…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: AUDIO_CONSTRAINTS });
      const video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();
      setStatus("Loading the hand tracker…");
      const elements = ((await (await fetch("/api/library")).json()) as LiveElement[]);
      const byEffect = Object.fromEntries(elements.map((e) => [e.effect, e])) as Record<string, LiveElement>;
      const vision = await import("@mediapipe/tasks-vision");
      const files = await vision.FilesetResolver.forVisionTasks(WASM);
      const hands = await vision.HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" }, runningMode: "VIDEO", numHands: 1 });
      const audio = new AudioContext();
      const analyser = audio.createAnalyser();
      analyser.fftSize = 1024;
      audio.createMediaStreamSource(stream).connect(analyser);
      const pcm = new Float32Array(analyser.fftSize);
      const pool = Object.values(byEffect).flatMap((el) => [0, 1, 2].map(() => {
        const v = document.createElement("video");
        v.src = el.url; v.muted = true; v.playsInline = true; v.preload = "auto";
        return { el, v, busy: false };
      }));
      const plays: Play[] = [];
      let onset: OnsetState = { floor: -60, top: -20, prev: -90, since: null, lastHit: -1, peak: -90 };
      let lm: Landmarks | null = null;
      let alive = true;
      let rec: MediaRecorder | null = null;
      const ctx = canvas.current!.getContext("2d")!;
      const fire = (effect: string, gain: number, size: number, held: boolean) => {
        const slot = pool.find((p) => p.el.effect === effect && !p.busy);
        if (!slot) return null;
        slot.busy = true;
        slot.v.currentTime = Math.max(0, (slot.el.ignition - (held ? 0 : 1)) / slot.el.fps);
        slot.v.playbackRate = held ? 1 : 1 / Math.max(0.5, size); // a smaller blast is over sooner
        void slot.v.play();
        const p: Play = { el: slot.el, video: slot.v, gain, size, held, letGo: null };
        plays.push(p);
        return p;
      };
      let heldPlay: Play | null = null;
      setStatus(null);

      const frame = () => {
        if (!alive) return;
        const W = (canvas.current!.width = video.videoWidth), H = (canvas.current!.height = video.videoHeight);
        ctx.save();
        ctx.translate(W, 0);
        ctx.scale(-1, 1); // a mirror, as you see yourself
        ctx.drawImage(video, 0, 0, W, H);
        ctx.restore();
        const res = hands.detectForVideo(video, performance.now());
        lm = res.landmarks?.[0]?.map((p) => ({ x: (1 - p.x) * W, y: p.y * H })) ?? null;
        analyser.getFloatTimeDomainData(pcm);
        let sum = 0;
        for (const v of pcm) sum += v * v;
        const db = 10 * Math.log10(sum / pcm.length + 1e-12);
        const step = onsetStep(onset, db, performance.now() / 1000);
        onset = step.state;
        const hand = lm ? liveHand(lm) : null;
        if (step.hit && hand) {
          const look = playLook(step.hit.db, onset.top);
          fire("fire_burst", look.gain, look.size, false);
        }
        if (step.holding && hand && !heldPlay) {
          heldPlay = fire(hand.pointing ? "fire_breath" : "flame_plume", 1, 1, true);
        }
        if (!step.holding && heldPlay && heldPlay.letGo === null) heldPlay.letGo = performance.now();
        if (heldPlay && heldPlay.letGo === null) { // held: its steady burn, looped
          const t = heldPlay.video.currentTime * heldPlay.el.fps;
          if (t >= heldPlay.el.release - 2) heldPlay.video.currentTime = (heldPlay.el.ignition + 6) / heldPlay.el.fps;
        }
        let glow = 0;
        for (let i = plays.length - 1; i >= 0; i--) {
          const p = plays[i];
          const t = p.video.currentTime * p.el.fps;
          const fading = p.letGo === null ? 1 : Math.max(0, 1 - (performance.now() - p.letGo) / 400);
          if (t >= p.el.end - 1 || p.video.ended || fading <= 0) {
            p.video.pause();
            pool.find((s) => s.v === p.video)!.busy = false;
            plays.splice(i, 1);
            if (p === heldPlay) heldPlay = null;
            continue;
          }
          if (!hand) continue;
          const jet = p.el.behaviour === "jet";
          const at = jet ? hand.tip : hand.palm;
          const k = (hand.scale / p.el.ref.scale) * p.size;
          const turn = jet ? Math.atan2(hand.dir[1], hand.dir[0]) - Math.atan2(p.el.ref.dir[1], p.el.ref.dir[0]) : 0;
          ctx.save();
          ctx.globalCompositeOperation = "lighter";
          ctx.globalAlpha = Math.min(1, p.gain * fading);
          ctx.filter = "contrast(1.3) brightness(1.1)"; // (the render's black, a few levels up, goes to black: added, it greyed the picture)
          ctx.translate(at[0], at[1]);
          ctx.rotate(turn);
          ctx.scale(k, k);
          ctx.drawImage(p.video, -p.el.ref.xy[0], -p.el.ref.xy[1], p.el.size[0], p.el.size[1]);
          ctx.restore();
          glow = Math.max(glow, p.gain * fading * Math.min(1, t > p.el.ignition ? 1 : 0.4));
        }
        if (hand && glow > 0) { // the fire's light on the hand
          const g = ctx.createRadialGradient(hand.palm[0], hand.palm[1], 0, hand.palm[0], hand.palm[1], hand.scale * 3);
          g.addColorStop(0, `rgba(255, 150, 60, ${0.35 * glow})`);
          g.addColorStop(1, "rgba(255, 120, 40, 0)");
          ctx.save();
          ctx.globalCompositeOperation = "screen";
          ctx.fillStyle = g;
          ctx.fillRect(0, 0, W, H);
          ctx.restore();
        }
        requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
      run.current = {
        stop: () => {
          alive = false;
          if (rec?.state === "recording") rec.stop();
          stream.getTracks().forEach((t) => t.stop());
          void audio.close();
          hands.close();
          pool.forEach((p) => { p.v.pause(); p.v.removeAttribute("src"); });
        },
        record: () => { // the raw take (camera and microphone), for the real shot
          const mime = pickMime((t) => MediaRecorder.isTypeSupported(t));
          const r = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
          const chunks: Blob[] = [];
          r.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
          r.onstop = () => {
            setRecording(null);
            const type = r.mimeType || mime || "video/webm";
            const file = new File(chunks, `my live take.${type.includes("mp4") ? "mp4" : "webm"}`, { type });
            run.current?.stop();
            setOpen(false);
            onTake(file, URL.createObjectURL(file));
          };
          rec = r;
          r.start(250);
          setRecording(0);
        },
        stopRecording: () => { if (rec?.state === "recording") rec.stop(); },
      };
    } catch (e) {
      setError((e as Error).message);
      setStatus(null);
    }
  };
  const close = () => { run.current?.stop(); run.current = null; setOpen(false); setRecording(null); };

  if (!open) {
    return (
      <>
        <button onClick={() => void start()}>Try it live</button>
        {error && <span className="chip bad">{error}</span>}
      </>
    );
  }
  return (
    <div className="film live">
      <canvas ref={canvas} aria-label="your camera, with the fire on your hand" />
      <p className="muted">{status ?? "Raise your hand and make the noise: BOOM throws a fireball, a long fffff holds a flame (point to make it a jet). A rough live preview of fire rendered ahead with LTX-2.5; Record this and the shot is made from your take."}</p>
      <div className="row">
        {recording === null ? (
          <>
            <button disabled={!!status} onClick={() => run.current?.record()}>Record this</button>
            <button className="secondary" onClick={close}>Close</button>
          </>
        ) : (
          <>
            <button onClick={() => run.current?.stopRecording()}>Stop</button>
            <span className="chip">recording {recording.toFixed(1)} s</span>
          </>
        )}
      </div>
      {error && <p className="chip bad">{error}</p>}
    </div>
  );
}
