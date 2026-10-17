"use client";

import { useState } from "react";
import type { Performance } from "@/components/Performer";
import Retime from "@/components/Retime";
import type { PickedPlate } from "@/components/PlatePicker";
import { AUTO_EFFECT, EFFECT_LABELS, type EffectInfo, followJob, type JobEvent, type JobStatus, type Mode, type Stroke, submitShot } from "@/lib/shot";

type View = "final.mp4" | "breakdown.mp4" | "composite.mp4";
const VIEWS: { file: View; label: string }[] = [
  { file: "final.mp4", label: "Shot" },
  { file: "breakdown.mp4", label: "How it was made" },
  { file: "composite.mp4", label: "Clean" },
];

const pollUntilDone = async (id: string): Promise<JobStatus> => {
  for (let i = 0; i < 400; i++) { // ~10 minutes
    const r = await fetch(`/api/jobs/${id}`);
    if (r.status === 404) throw new Error("the server lost this job (it may have restarted); please make it again");
    const s: JobStatus = await r.json();
    if (s.status === "done" || s.status === "error") return s;
    await new Promise((res) => setTimeout(res, 1500));
  }
  throw new Error("this is taking too long; please try again");
};

const ms = (frames: number) => Math.round((frames / 24) * 1000);

/** The receipt as plain claims: what rendered it, where the hit landed, when it stopped. */
function Chips({ job, effect }: { job: JobStatus; effect: EffectInfo | undefined }) {
  const r = job.receipt;
  if (!r) return null;
  const lib = r.render.library;
  const hit = r.measured.hit_error_frames;
  const stop = r.measured.stop_latency_frames;
  const scene = r.mode === "scene";
  const early = r.measured.model_early_frames ?? null;
  const cue = r.inputs?.cue ?? "voice";
  return (
    <div className="row">
      {job.cached && <span className="chip warn">a sample made earlier: perform your own noise to make one live</span>}
      {r.retimed && <span className="chip">re-timed by hand: hit {r.retimed.hit_shift_frames >= 0 ? "+" : ""}{ms(r.retimed.hit_shift_frames)} ms{r.retimed.stop_frame !== null ? `, stop at ${(r.retimed.stop_frame / 24).toFixed(2)} s` : ""}, same element</span>}
      <span className="chip">{scene ? `rendered into your scene in ${r.render.seconds.toFixed(0)} s on open weights (LTX-2.5, from your plate's frame)`
        : lib ? `instant: a pre-rendered LTX-2.5 element, timed by your voice` : `rendered from your voice in ${r.render.seconds.toFixed(0)} s on open weights`}</span>
      {r.hits && r.hits.length > 1 && <span className="chip">{r.hits.length} hits, each its own {scene ? "event" : "burst"}, sized by how you made it</span>}
      {job.effect && job.gesture && <span className="chip">your {cue === "hold" ? "timing" : "noise"} read as a {job.gesture}: {EFFECT_LABELS[job.effect]?.name ?? job.effect}</span>}
      {cue === "clip" && <span className="chip">locked to the sound in your clip</span>}
      {r.inputs?.drawn && <span className="chip">where you drew it</span>}
      {scene && early !== null && early !== 0 && <span className="chip">the model put it {ms(Math.abs(early))} ms {early > 0 ? "early" : "late"}; moved onto your cue</span>}
      {(r.warnings ?? []).map((w) => <span key={w} className="chip warn">{w}</span>)}
      {hit !== null && <span className="chip">{hit === 0 ? "the hit lands on your exact frame" : `hit ${hit > 0 ? "+" : ""}${ms(hit)} ms from your noise`}</span>}
      {(effect?.stops_with_voice ?? r.measured.stops_with_voice) && stop !== null && (
        <span className="chip">{stop <= 0 ? "stops when you stop" : `stops ${ms(stop)} ms after you do`}</span>
      )}
      <span className="chip">{job.cached ? `it took ${r.seconds_total.toFixed(0)} s when it was made` : `${r.seconds_total.toFixed(0)} s in total`}</span>
    </div>
  );
}

/** Step 4: pick the effect and how to make it, then the shot. */
export default function ShotPanel({ plate, point, direction, performance, effects, modes, strokes, onDone }: {
  plate: PickedPlate;
  point: [number, number];
  direction: number | null;
  performance: Performance;
  effects: EffectInfo[];
  modes: Mode[];
  strokes: Stroke[];
  onDone: (job: JobStatus, effect: string) => void;
}) {
  const firstMode: Mode = modes.includes("scene") ? "scene" : modes.includes("instant") ? "instant" : "render";
  const [mode, setModeState] = useState<Mode>(firstMode);
  // the cue decides (its gesture, read through the plate's own map), unless the performer picks
  const [effect, setEffect] = useState(AUTO_EFFECT);
  const [subject, setSubject] = useState(plate.subject ?? "");
  const [describe, setDescribe] = useState("");
  const drawn = strokes.length > 0;
  const choices = effects.filter((e) => (mode === "scene" ? e.scene : e.element !== false));
  const setMode = (m: Mode) => {
    setModeState(m);
    if (effect !== AUTO_EFFECT && !effects.some((e) => e.id === effect && (m === "scene" ? e.scene : e.element !== false))) {
      setEffect(AUTO_EFFECT);
    }
  };
  const [seed] = useState(1); // one seed per session: retakes differ only by the performance
  const [busy, setBusy] = useState(false);
  const [event, setEvent] = useState<JobEvent | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [view, setView] = useState<View>("final.mp4");
  const [guideUrl, setGuideUrl] = useState<string | null>(null);

  const make = async () => {
    setBusy(true);
    setJob(null);
    setEvent({ stage: "uploading", progress: 0.01, message: "Uploading", t: 0 });
    try {
      const queued = await submitShot({ plate: plate.source, voice: performance.voice, face: performance.face, faceExt: performance.faceExt,
        timing: performance.timing, effect, point, seed, mode, direction, subject: subject.trim() || null,
        cue: performance.cue, holds: performance.holds ?? null, strokes: drawn ? strokes : null,
        describe: describe.trim() || null });
      setGuideUrl(null);
      const peek = setInterval(() => { // the guide exists a second in: show it while the element is made
        fetch(`/api/jobs/${queued.id}`).then((r) => r.json()).then((s: JobStatus) => {
          if (s.files["guide_preview.mp4"]) { setGuideUrl(s.files["guide_preview.mp4"]); clearInterval(peek); }
        }).catch(() => undefined);
      }, 1500);
      let done: JobStatus;
      try {
        done = await followJob(queued.id, setEvent);
      } finally {
        clearInterval(peek);
      }
      if (done.status !== "done" && done.status !== "error") done = await pollUntilDone(queued.id);
      setJob(done);
      setView("final.mp4");
      if (done.status === "done") onDone(done, effect);
    } catch (e) {
      setJob({ id: "", status: "error", stage: "error", progress: 0, message: "", error: (e as Error).message, files: {}, receipt: null });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section>
      <h2>4. Make the shot</h2>
      {performance.warnings.map((w) => <p key={w} className="chip warn">{w}</p>)}
      <div className="row" role="radiogroup" aria-label="Effect">
        <button role="radio" aria-checked={effect === AUTO_EFFECT} disabled={busy}
          className={effect === AUTO_EFFECT ? "" : "secondary"} onClick={() => setEffect(AUTO_EFFECT)}>
          Your cue decides <span className="muted" style={{ marginLeft: 6 }}>{performance.cue === "hold" ? "short pops, long flames" : "pf · KA-BOOM · fwoooom · hissss"}</span>
        </button>
        {choices.map((e) => (
          <button key={e.id} role="radio" aria-checked={effect === e.id} disabled={busy}
            className={effect === e.id ? "" : "secondary"} onClick={() => setEffect(e.id)}>
            {EFFECT_LABELS[e.id]?.name ?? e.id}
            <span className="muted" style={{ marginLeft: 6 }}>{EFFECT_LABELS[e.id]?.cue}</span>

          </button>
        ))}
      </div>
      {mode === "scene" && (
        <label className="row">
          <span className="muted">Say what happens (optional)</span>
          <input value={describe} maxLength={200} disabled={busy} style={{ flex: "1 1 260px" }}
            placeholder={drawn ? "thick smoke pours out of the lamp" : "a fireball bursts out of the jack-o'-lantern"}
            onChange={(e) => setDescribe(e.target.value)} />
        </label>
      )}
      <div className="row" role="radiogroup" aria-label="How to make it">
        <button role="radio" aria-checked={mode === "scene"} disabled={busy || !modes.includes("scene")}
          className={mode === "scene" ? "" : "secondary"} onClick={() => setMode("scene")}>
          Into the scene <span className="muted">LTX-2.5 renders it into your shot and the scene reacts, ~2 min</span>
        </button>
        <button role="radio" aria-checked={mode === "instant"} disabled={busy || !modes.includes("instant") || drawn}
          className={mode === "instant" ? "" : "secondary"} onClick={() => setMode("instant")}>
          Instant <span className="muted">a pre-rendered element, no GPU, ~15 s</span>
        </button>
      </div>
      <details>
        <summary className="muted">Advanced</summary>
        <div className="row">
          <button role="radio" aria-checked={mode === "render"} disabled={busy || !modes.includes("render") || drawn}
            className={mode === "render" ? "" : "secondary"} onClick={() => setMode("render")}>
            Element <span className="muted">rendered on black and composited; your footage is never regenerated, ~45 s</span>
          </button>
        </div>
      </details>
      {drawn && mode !== "scene" && (
        <p className="chip warn">{modes.includes("scene")
          ? "A drawing needs \"Into the scene\"."
          : "A drawing needs \"Into the scene\", and the GPU is offline right now: tap the source instead for an instant shot."}</p>
      )}
      {!modes.includes("render") && <p className="muted">The GPU is offline right now: instant mode places a pre-rendered element with your timing.</p>}
      {mode === "scene" && !plate.sample && (
        <label className="row">
          <span className="muted">What is it?</span>
          <input value={subject} maxLength={120} disabled={busy} placeholder="the toy dinosaur on the desk"
            onChange={(e) => setSubject(e.target.value)} />
        </label>
      )}
      <div className="row">
        <button disabled={busy || effects.length === 0 || (drawn && mode !== "scene")} onClick={() => void make()}>{job?.status === "done" ? "Make it again" : "Make the shot"}</button>
      </div>

      {busy && event && (
        <div aria-live="polite">
          <progress value={event.progress} max={1} style={{ width: "100%" }} />
          <p className="muted">{event.message}…</p>
          {guideUrl && (
            <div className="row">
              <video src={guideUrl} autoPlay loop muted playsInline className="guide" />
              <span className="muted">{mode === "scene"
                ? (drawn ? "Your drawing, filled by your cue: the model renders the effect where you drew it, when you cued it, as big as you made it."
                  : "Your cue, painted into your scene: the model renders the effect where it comes from, when you cued it, as big as you made it.")
                : "Your noise, painted into the guide the model follows: the blobs appear on your hit and grow with your voice."}</span>
            </div>
          )}
        </div>
      )}
      {job?.status === "error" && <p className="chip bad">The shot failed: {job.error}</p>}
      {job?.status === "done" && (
        <div>
          <div className="row" role="tablist">
            {VIEWS.filter((v) => job.files[v.file]).map((v) => (
              <button key={v.file} role="tab" aria-selected={view === v.file} className={view === v.file ? "" : "secondary"}
                onClick={() => setView(v.file)}>{v.label}</button>
            ))}
          </div>
          <video key={view} src={job.files[view]} controls autoPlay playsInline className="result" />
          <Chips job={job} effect={effects.find((e) => e.id === effect)} />
          <Retime key={job.id} job={job} stopsWithVoice={job.receipt?.mode !== "scene" && (effects.find((e) => e.id === effect)?.stops_with_voice ?? false)}
            onRetimed={(next) => { setJob(next); setView("final.mp4"); onDone(next, effect); }} />
          <div className="row">
            {Object.entries(job.files).map(([name, url]) => <a key={name} className="btn secondary" href={url} download={name}>{name}</a>)}
          </div>
        </div>
      )}
    </section>
  );
}
