"use client";

import { useEffect, useState } from "react";
import { EFFECT_LABELS, type EffectInfo, fetchEffects, followJob, type JobEvent, type JobStatus, submitShot } from "@/lib/shot";

type View = "final.mp4" | "breakdown.mp4" | "composite.mp4";
const VIEWS: { file: View; label: string }[] = [
  { file: "final.mp4", label: "Shot" },
  { file: "breakdown.mp4", label: "How it was made" },
  { file: "composite.mp4", label: "Clean" },
];

export interface ShotInput {
  plate: File;
  voice: Blob;
  face: Blob | null;
  faceExt: string;
  timing: object | null;
  point: [number, number];
  suggested: string;
}

const pollUntilDone = async (id: string): Promise<JobStatus> => {
  for (;;) {
    const s: JobStatus = await (await fetch(`/api/jobs/${id}`)).json();
    if (s.status === "done" || s.status === "error") return s;
    await new Promise((r) => setTimeout(r, 1500));
  }
};

export default function ShotPanel({ input }: { input: ShotInput }) {
  const [effects, setEffects] = useState<EffectInfo[] | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [effect, setEffect] = useState(input.suggested);
  const [seed, setSeed] = useState(1);
  const [busy, setBusy] = useState(false);
  const [event, setEvent] = useState<JobEvent | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [view, setView] = useState<View>("final.mp4");

  useEffect(() => {
    fetchEffects().then(setEffects, (e: Error) => setApiError(e.message));
  }, []);

  const make = async (nextSeed = seed) => {
    setBusy(true);
    setJob(null);
    setEvent({ stage: "uploading", progress: 0.01, message: "Uploading your plate and performance", t: 0 });
    try {
      const queued = await submitShot({ ...input, effect, seed: nextSeed });
      let done = await followJob(queued.id, setEvent);
      if (done.status !== "done" && done.status !== "error") done = await pollUntilDone(queued.id);
      setJob(done);
      setView("final.mp4");
    } catch (e) {
      setJob({ id: "", status: "error", stage: "error", progress: 0, message: "", error: (e as Error).message, files: {}, receipt: null });
    } finally {
      setBusy(false);
    }
  };

  if (apiError) {
    return (
      <section>
        <p className="chip bad">The render server isn&apos;t reachable ({apiError}).</p>
        <p className="muted">Start it with <code>cuesync serve</code> (see the README), then reload this page.</p>
      </section>
    );
  }

  const r = job?.receipt;
  return (
    <section>
      <h2>3. Make the shot</h2>
      <div className="row" role="radiogroup" aria-label="Effect">
        {(effects ?? []).map((e) => (
          <button key={e.id} role="radio" aria-checked={effect === e.id} disabled={busy}
            className={effect === e.id ? "" : "secondary"} onClick={() => setEffect(e.id)}>
            {EFFECT_LABELS[e.id]?.name ?? e.id}
            <span className="muted" style={{ marginLeft: 6 }}>{EFFECT_LABELS[e.id]?.cue}</span>
            {e.id === input.suggested && <span className="chip" style={{ marginLeft: 6 }}>your noise</span>}
          </button>
        ))}
      </div>
      <div className="row">
        <button disabled={busy || !effects} onClick={() => void make()}>{job?.status === "done" ? "Make it again" : "Make the shot"}</button>
        {job?.status === "done" && (
          <button className="secondary" disabled={busy} onClick={() => { const s = seed + 1; setSeed(s); void make(s); }}>
            Another take (seed {seed + 1})
          </button>
        )}
      </div>

      {busy && event && (
        <div aria-live="polite">
          <progress value={event.progress} max={1} style={{ width: "100%" }} />
          <p className="muted">{event.message}…</p>
        </div>
      )}

      {job?.status === "error" && <p className="chip bad">Render failed: {job.error}</p>}

      {job?.status === "done" && (
        <div>
          <div className="row" role="tablist">
            {VIEWS.filter((v) => job.files[v.file]).map((v) => (
              <button key={v.file} role="tab" aria-selected={view === v.file} className={view === v.file ? "" : "secondary"}
                onClick={() => setView(v.file)}>{v.label}</button>
            ))}
          </div>
          <video key={view} src={job.files[view]} controls autoPlay playsInline
            style={{ width: "100%", maxWidth: 420, aspectRatio: view === "final.mp4" ? "1080 / 1984" : "9 / 16", background: "#000", borderRadius: 12 }} />
          {r && (
            <div className="row">
              <span className="chip">rendered in {r.render.seconds.toFixed(0)} s on open weights</span>
              {r.measured.onset_error_frames !== null && (
                <span className="chip">{r.measured.onset_error_frames === 0 ? "lands on your exact frame" : `onset ${r.measured.onset_error_frames > 0 ? "+" : ""}${r.measured.onset_error_frames} frames`}</span>
              )}
              {r.measured.stop_latency_frames !== null && (
                <span className="chip">stops {Math.round((r.measured.stop_latency_frames / 24) * 1000)} ms after you do</span>
              )}
            </div>
          )}
          <div className="row">
            {Object.entries(job.files).map(([name, url]) => <a key={name} className="btn secondary" href={url} download={name}>{name}</a>)}
          </div>
        </div>
      )}
    </section>
  );
}
