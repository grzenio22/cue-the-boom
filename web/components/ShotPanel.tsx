"use client";

import { useState } from "react";
import type { Performance } from "@/components/Performer";
import { EFFECT_LABELS, type EffectInfo, followJob, type JobEvent, type JobStatus, type Mode, type Source, submitShot } from "@/lib/shot";

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
  return (
    <div className="row">
      <span className="chip">{lib ? `instant: a pre-rendered LTX-2.5 element, timed by your voice` : `rendered from your voice in ${r.render.seconds.toFixed(0)} s on open weights`}</span>
      {r.hits && r.hits.length > 1 && <span className="chip">{r.hits.length} hits, each its own burst, sized by how you made it</span>}
      {hit !== null && <span className="chip">{hit === 0 ? "the hit lands on your exact frame" : `hit ${hit > 0 ? "+" : ""}${ms(hit)} ms from your noise`}</span>}
      {(effect?.stops_with_voice ?? r.measured.stops_with_voice) && stop !== null && (
        <span className="chip">{stop <= 0 ? "stops when you stop" : `stops ${ms(stop)} ms after you do`}</span>
      )}
      <span className="chip">{r.seconds_total.toFixed(0)} s in total</span>
    </div>
  );
}

/** Step 4: pick the effect and how to make it, then the shot. */
export default function ShotPanel({ plate, point, performance, effects, modes, onDone }: {
  plate: Source;
  point: [number, number];
  performance: Performance;
  effects: EffectInfo[];
  modes: Mode[];
  onDone: (job: JobStatus, effect: string) => void;
}) {
  const [effect, setEffect] = useState(performance.suggested);
  const [mode, setMode] = useState<Mode>(modes.includes("render") ? "render" : "instant");
  const [seed] = useState(1); // one seed per session: retakes differ only by the performance
  const [busy, setBusy] = useState(false);
  const [event, setEvent] = useState<JobEvent | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [view, setView] = useState<View>("final.mp4");

  const make = async () => {
    setBusy(true);
    setJob(null);
    setEvent({ stage: "uploading", progress: 0.01, message: "Uploading", t: 0 });
    try {
      const queued = await submitShot({ plate, voice: performance.voice, face: performance.face, faceExt: performance.faceExt,
        timing: performance.timing, effect, point, seed, mode });
      let done = await followJob(queued.id, setEvent);
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
        {effects.map((e) => (
          <button key={e.id} role="radio" aria-checked={effect === e.id} disabled={busy}
            className={effect === e.id ? "" : "secondary"} onClick={() => setEffect(e.id)}>
            {EFFECT_LABELS[e.id]?.name ?? e.id}
            <span className="muted" style={{ marginLeft: 6 }}>{EFFECT_LABELS[e.id]?.cue}</span>
            {e.id === performance.suggested && <span className="chip" style={{ marginLeft: 6 }}>your noise</span>}
          </button>
        ))}
      </div>
      <div className="row" role="radiogroup" aria-label="How to make it">
        <button role="radio" aria-checked={mode === "render"} disabled={busy || !modes.includes("render")}
          className={mode === "render" ? "" : "secondary"} onClick={() => setMode("render")}>
          Render it from my voice <span className="muted">LTX-2.5 on a GPU, ~45 s</span>
        </button>
        <button role="radio" aria-checked={mode === "instant"} disabled={busy || !modes.includes("instant")}
          className={mode === "instant" ? "" : "secondary"} onClick={() => setMode("instant")}>
          Instant <span className="muted">pre-rendered, ~15 s</span>
        </button>
      </div>
      {!modes.includes("render") && <p className="muted">The GPU is offline right now: instant mode places a pre-rendered element with your timing.</p>}
      <div className="row">
        <button disabled={busy || effects.length === 0} onClick={() => void make()}>{job?.status === "done" ? "Make it again" : "Make the shot"}</button>
      </div>

      {busy && event && (
        <div aria-live="polite">
          <progress value={event.progress} max={1} style={{ width: "100%" }} />
          <p className="muted">{event.message}…</p>
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
          <div className="row">
            {Object.entries(job.files).map(([name, url]) => <a key={name} className="btn secondary" href={url} download={name}>{name}</a>)}
          </div>
        </div>
      )}
    </section>
  );
}
