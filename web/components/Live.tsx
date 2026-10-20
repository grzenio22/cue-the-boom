"use client";

import { useEffect, useRef, useState } from "react";
import { AUDIO_CONSTRAINTS } from "@/lib/capture/constraints";
import { pickMime } from "@/lib/capture/pickMime";
import {
  arcChannels, arcStep, type Arcs, BALL_RGB, type Ball, ballStep, BUZZ_HF, buzzOf, dense, dipStep, drawBall, drawChannels,
  drawDip, drawLight, drawSpark, HAND_RGB, type Light, lightStep, newArcs, type P2, ROOM_RGB,
} from "@/lib/lightning";
import { type LiveElement, liveHand, onsetStep, type OnsetState, playLook, RELEASE_MS, thrownAt } from "@/lib/live";

// MediaPipe's hand landmarker in the browser (Apache-2.0), its runtime and model from Google's own hosting
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MAX_REC_S = 8;
const HOLD_HAND_FRAMES = 5;
const BUZZ_ON = 3, BUZZ_MAX = 8; // frames of buzz that start the lightning (it lasts as many after the buzz stops)

interface Play {
  el: LiveElement; video: HTMLVideoElement; gain: number; size: number; held: boolean; letGo: number | null;
  fit: number; born: number; left: { at: [number, number]; scale: number; t: number } | null; // (thrown: where it left the hand)
}
type Landmarks = { x: number; y: number }[];
/** What one start of the preview holds, so closing (or failing, or leaving the page) at any moment of it, loading
 * included, lets go of the camera, the microphone, the tracker and the videos. */
interface Session {
  over: boolean; stream?: MediaStream; audio?: AudioContext; hands?: { close: () => void }; videos: HTMLVideoElement[];
  rec?: MediaRecorder | null; keep?: boolean; // keep: a recording stopped to make the shot (not thrown away)
  torn?: boolean; // let go already (Record this, then the page's own close: an AudioContext closed twice threw)
}

function teardown(s: Session) {
  s.over = true;
  if (s.torn) return;
  s.torn = true;
  if (s.rec?.state === "recording") s.rec.stop();
  s.stream?.getTracks().forEach((t) => t.stop());
  if (s.audio && s.audio.state !== "closed") void s.audio.close();
  s.hands?.close();
  s.videos.forEach((v) => { v.pause(); v.removeAttribute("src"); v.load(); });
}

/** Try it live: your hand on the camera, your noise on the microphone, the fire rendered ahead of time (LTX-2.5)
 * thrown from your palm as you make it, as big as it is loud; a held "fffff" holds a flame (pointing: a jet). A
 * rough preview: Record this, and the shot is made from your take with the hand followed, matted and lit. */
export default function Live({ onTake }: { onTake: (file: File, url: string) => void }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recording, setRecording] = useState<number | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const session = useRef<Session | null>(null);
  const run = useRef<{ record: () => void; stopRecording: () => void } | null>(null);

  useEffect(() => () => { if (session.current) teardown(session.current); }, []);
  useEffect(() => {
    if (recording === null) return;
    if (recording >= MAX_REC_S) { run.current?.stopRecording(); return; }
    const t = setTimeout(() => setRecording((r) => (r === null ? null : r + 0.1)), 100);
    return () => clearTimeout(t);
  }, [recording]);

  const start = async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) { setError("No camera here: the page needs localhost or HTTPS."); return; }
    if (session.current) teardown(session.current);
    const ses: Session = { over: false, videos: [] };
    session.current = ses;
    run.current = null;
    setOpen(true);
    setStatus("Starting the camera…");
    const gone = () => ses.over; // closed (or started again) while this was loading
    let step = "The camera or the microphone didn't start";
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: AUDIO_CONSTRAINTS });
      ses.stream = stream;
      if (gone()) { teardown(ses); return; }
      const video = document.createElement("video");
      ses.videos.push(video);
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();
      if (gone()) { teardown(ses); return; }
      setStatus("Loading the hand tracker…");
      step = "The hand tracker didn't load (it comes from the internet the first time)";
      const elements = ((await (await fetch("/api/library")).json()) as LiveElement[]);
      if (gone()) { teardown(ses); return; }
      const byEffect = Object.fromEntries(elements.map((e) => [e.effect, e])) as Record<string, LiveElement>;
      const vision = await import("@mediapipe/tasks-vision");
      const files = await vision.FilesetResolver.forVisionTasks(WASM);
      const hands = await vision.HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" }, runningMode: "VIDEO", numHands: 1 });
      ses.hands = hands;
      if (gone()) { teardown(ses); return; }
      step = "The microphone didn't start";
      const audio = new AudioContext();
      ses.audio = audio;
      const analyser = audio.createAnalyser();
      analyser.fftSize = 1024;
      audio.createMediaStreamSource(stream).connect(analyser);
      const pcm = new Float32Array(analyser.fftSize);
      const spectrum = new Float32Array(analyser.frequencyBinCount);
      let buzzRun = 0; // how long a buzz has gone on (it starts the lightning at BUZZ_ON, and it lets go as slowly)
      let arcs: Arcs | null = null; // the lightning: arcs from the pointed fingertip, or a ball over the open palm
      let ball: Ball | null = null;
      let zapLight: Light | null = null; // the light it throws (it holds while the strokes strike, dies with them)
      let zapAt: { at: P2; room: string; hand: string; gain: number } | null = null; // where that light comes from,
      // its colours on the room and the hand, and how strong it is on the hand
      let zapDip = 0; // stops the camera exposes down for the ball
      let frameNo = 0;
      const pool = Object.values(byEffect).flatMap((el) => [0, 1, 2].map(() => {
        const v = document.createElement("video");
        v.src = el.url; v.muted = true; v.playsInline = true; v.preload = "auto";
        ses.videos.push(v);
        return { el, v, busy: false };
      }));
      const plays: Play[] = [];
      let onset: OnsetState = { floor: -60, top: -20, prev: -90, since: null, lastHit: -1, peak: -90 };
      let lm: Landmarks | null = null;
      let lastHand: ReturnType<typeof liveHand> | null = null;
      let lostFor = 1e9;
      const ctx = canvas.current?.getContext("2d");
      if (!ctx || gone()) { teardown(ses); return; }
      const fire = (effect: string, gain: number, size: number, held: boolean, tip = false) => {
        const slot = pool.find((p) => p.el.effect === effect && !p.busy);
        if (!slot) return null;
        slot.busy = true;
        slot.v.currentTime = Math.max(0, (slot.el.ignition - (held ? 0 : 1)) / slot.el.fps);
        slot.v.playbackRate = held ? 1 : 1 / Math.max(0.5, size); // a smaller blast is over sooner
        void slot.v.play();
        const fit = (tip ? slot.el.fit?.tip : slot.el.fit?.palm) ?? 1; // as big as the shot will play it
        const p: Play = { el: slot.el, video: slot.v, gain, size, held, letGo: null, fit, born: performance.now(), left: null };
        plays.push(p);
        return p;
      };
      let heldPlay: Play | null = null;
      setStatus(null);

      const frame = () => {
        if (ses.over || !canvas.current) return;
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
        analyser.getFloatFrequencyData(spectrum);
        const bz = buzzOf(spectrum, pcm, audio.sampleRate); // a voiced "zzzzz": lightning, not fire
        buzzRun = bz.voiced && bz.hf >= BUZZ_HF && db > onset.floor + 12 ? Math.min(BUZZ_MAX, buzzRun + 1) : Math.max(0, buzzRun - 1);
        const zapping = buzzRun >= BUZZ_ON || (buzzRun > 0 && (arcs !== null || ball !== null));
        frameNo++;
        // the hand, or for a few frames the one last seen (a hit on a frame the tracker missed it was dropped)
        if (lm) { lastHand = liveHand(lm); lostFor = 0; } else lostFor++;
        const hand = lostFor <= HOLD_HAND_FRAMES ? lastHand : null;
        const buzzy = bz.voiced && bz.hf >= BUZZ_HF; // (the buzz's own onset is no BOOM: no fireball on it)
        if (step.hit && hand && !zapping && !buzzy) {
          const look = playLook(step.hit.db, onset.top);
          fire("fire_burst", look.gain, look.size, false, hand.pointing);
        }
        if (zapping && heldPlay && heldPlay.letGo === null) heldPlay.letGo = performance.now(); // the buzz takes over
        if (step.holding && hand && !heldPlay && !zapping) {
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
          const now = performance.now();
          if (p.el.behaviour === "burst" && !p.left && hand && now - p.born >= RELEASE_MS) { // a fireball leaves the hand
            p.left = { at: hand.pointing ? hand.tip : hand.palm, scale: hand.scale, t: now };
          }
          if (!hand && !p.left) continue;
          const jet = p.el.behaviour === "jet";
          const at = p.left ? thrownAt(p.left.at, p.left.scale, now - p.left.t) : jet ? hand!.tip : hand!.palm;
          const k = ((p.left ? p.left.scale : hand!.scale) / p.el.ref.scale) * p.size * p.fit;
          const turn = jet ? Math.atan2(hand!.dir[1], hand!.dir[0]) - Math.atan2(p.el.ref.dir[1], p.el.ref.dir[0]) : 0;
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
        // lightning: arcs from the pointed fingertip (welded to it, along the finger), or a ball of light over the
        // open palm; the light they throw goes under them, held while they strike and dying within a few frames, and
        // the camera exposes down for the ball. It follows the hand the tracker sees (held over a frame it missed,
        // but no longer: held while the take moved on, a bolt hung in the air where the hand had been)
        const zapHand = lostFor <= 1 ? hand : null;
        const level = Math.min(1, Math.max(0, (db - onset.floor) / 40));
        const px = H / 1080;
        const pts: P2[] | null = lm ? lm.map((p): P2 => [p.x, p.y]) : null;
        let drawZap = () => {};
        let power = 0;
        if (zapping && zapHand && zapHand.pointing) {
          ball = null;
          const tip: P2 = pts ? pts[8] : zapHand.tip; // (welded to the fingertip, along its last two joints)
          const aim = pts ? Math.atan2(pts[8][1] - pts[6][1], pts[8][0] - pts[6][0])
            : Math.atan2(zapHand.dir[1], zapHand.dir[0]);
          const now = arcStep(arcs ?? newArcs(), tip, aim, zapHand.scale, level, !!step.hit, frameNo, px);
          arcs = now;
          const lines = arcChannels(now, level);
          power = Math.max(0, ...lines.map(([, g]) => g));
          const end = now.main.ch!.pts[now.main.ch!.pts.length - 1];
          const at: P2 = [(2 * tip[0] + end[0]) / 3, (2 * tip[1] + end[1]) / 3]; // (the bolt's light comes from it)
          zapAt = { at, room: ROOM_RGB, hand: HAND_RGB, gain: 1 };
          drawZap = () => {
            drawChannels(ctx, lines, px);
            drawSpark(ctx, tip, 0.12 * zapHand.scale, (0.35 + 0.65 * level) * Math.max(0.4, power));
          };
        } else if (zapping && zapHand) {
          arcs = null;
          const now = ballStep(ball, level, !!step.hit, frameNo);
          ball = now;
          if (now) {
            power = (0.4 + 0.6 * now.level) * (now.born ? dense(now.level, now.charge) : 0.1); // (a crackle: dim)
            zapAt = { at: zapHand.palm, room: BALL_RGB, hand: BALL_RGB, gain: 0.45 }; // (under it: a pool, not a wash)
            drawZap = () => drawBall(ctx, now, zapHand.palm, zapHand.scale, px);
          }
        } else {
          arcs = null;
          ball = null;
        }
        zapLight = lightStep(zapLight, power);
        zapDip = dipStep(zapDip, ball?.born ? power : 0);
        drawDip(ctx, zapDip, W, H);
        if (zapAt && zapLight.held > 0) {
          drawLight(ctx, zapAt.at, zapHand?.scale ?? H / 6, zapLight.held, zapHand ? pts : null,
            zapAt.room, zapAt.hand, W, H, zapAt.gain);
        }
        drawZap();
        // that it sees you and hears you, before anything is thrown: a faint ring on the palm, the microphone's level
        if (hand && lm) {
          ctx.save();
          ctx.strokeStyle = "rgba(255, 176, 32, 0.45)";
          ctx.lineWidth = Math.max(2, hand.scale * 0.04);
          ctx.beginPath();
          ctx.arc(hand.palm[0], hand.palm[1], hand.scale * 0.55, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.restore();
        } else if (!hand) {
          ctx.save();
          ctx.font = `${Math.round(H / 24)}px system-ui, sans-serif`;
          ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
          ctx.textAlign = "center";
          ctx.fillText("Show me your hand", W / 2, H * 0.12);
          ctx.restore();
        }
        const lvl = Math.min(1, Math.max(0, (db - onset.floor) / 40)); // over the room's own level
        ctx.save();
        ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
        ctx.fillRect(W * 0.04, H * 0.93, W * 0.25, H * 0.025);
        ctx.fillStyle = lvl > 15 / 40 ? "rgba(255, 140, 40, 0.95)" : "rgba(255, 255, 255, 0.7)";
        ctx.fillRect(W * 0.04, H * 0.93, W * 0.25 * lvl, H * 0.025);
        ctx.restore();
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
        record: () => { // the raw take (camera and microphone), for the real shot
          const mime = pickMime((t) => MediaRecorder.isTypeSupported(t));
          const r = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
          const chunks: Blob[] = [];
          r.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
          r.onstop = () => {
            setRecording(null);
            if (!ses.keep) return; // closed while recording: thrown away
            const type = r.mimeType || mime || "video/webm";
            const file = new File(chunks, `my live take.${type.includes("mp4") ? "mp4" : "webm"}`, { type });
            teardown(ses);
            setOpen(false);
            onTake(file, URL.createObjectURL(file));
          };
          ses.rec = r;
          r.start(250);
          setRecording(0);
        },
        stopRecording: () => { if (ses.rec?.state === "recording") { ses.keep = true; ses.rec.stop(); } },
      };
    } catch (e) {
      teardown(ses); // (a camera refused, the tracker or its model not loading: nothing is left running)
      if (session.current === ses) {
        setError(`${step}: ${(e as Error).message || "it stopped"}.`);
        setStatus(null);
      }
    }
  };
  const close = () => {
    if (session.current) teardown(session.current);
    session.current = null;
    run.current = null;
    setOpen(false);
    setRecording(null);
    setStatus(null);
  };

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
      <p className="muted">{status ?? "Raise your hand and make the noise: BOOM throws a fireball, a long fffff holds a flame (point to make it a jet), a buzzing zzzzz is lightning. A rough live preview of fire rendered ahead with LTX-2.5; Record this and the shot is made from your take."}</p>
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
      {error && (
        <div className="row">
          <span className="chip bad">{error}</span>
          <button className="secondary" onClick={() => void start()}>Try again</button>
        </div>
      )}
    </div>
  );
}
