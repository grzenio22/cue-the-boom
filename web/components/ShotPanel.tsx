"use client";

import { useEffect, useState } from "react";
import type { Performance } from "@/components/Performer";
import Retime from "@/components/Retime";
import type { PickedPlate } from "@/components/PlatePicker";
import { AUTO_EFFECT, EFFECT_LABELS, type EffectInfo, fetchObjects, followJob, type JobEvent, type JobStatus, madeAhead, type Mode, type ObjectKind, pollUntilDone, type Stroke, submitShot } from "@/lib/shot";

type View = "final.mp4" | "breakdown.mp4" | "composite.mp4";
const VIEWS: { file: View; label: string }[] = [
  { file: "final.mp4", label: "Shot" },
  { file: "breakdown.mp4", label: "How it was made" },
  { file: "composite.mp4", label: "Clean" },
];

const ms = (frames: number) => Math.round((frames / 24) * 1000);

/** The receipt as plain claims: what rendered it, where the hit landed, when it stopped. */
function Chips({ job, effect, live }: { job: JobStatus; effect: EffectInfo | undefined; live: boolean }) {
  const r = job.receipt;
  if (!r) return null;
  const lib = r.render.library;
  const hit = r.measured.residual_frames ?? r.measured.hit_error_frames ?? null;
  const stop = r.measured.stop_latency_frames;
  const scene = r.mode === "scene";
  const early = r.measured.model_early_frames ?? null;
  const cue = r.inputs?.cue ?? "voice";
  return (
    <div className="row">
      {job.cached && <span className="chip warn">{live ? "a sample made earlier: perform your own noise to make one live" : "a sample made earlier"}</span>}
      {r.retimed && <span className="chip">re-timed by hand: hit {r.retimed.hit_shift_frames >= 0 ? "+" : ""}{ms(r.retimed.hit_shift_frames)} ms{r.retimed.stop_frame !== null ? `, stop at ${(r.retimed.stop_frame / 24).toFixed(2)} s` : ""}, same element</span>}
      <span className="chip">{scene ? `rendered into your scene in ${r.render.seconds.toFixed(0)} s on open weights (LTX-2.5, from your plate's frame)`
        : lib ? `instant: a pre-rendered LTX-2.5 element, timed by your voice` : `rendered from your voice in ${r.render.seconds.toFixed(0)} s on open weights`}</span>
      {r.hits && r.hits.length > 1 && (() => {
        const merged = r.hits.filter((h) => h.merged).length;
        return <span className="chip">{merged
          ? `${r.hits.length} hits; the model kept ${r.hits.length - merged} apart and merged ${merged} into a neighbour`
          : `${r.hits.length} hits, each its own ${scene ? "event" : "burst"}, sized by how you made it`}</span>;
      })()}
      {job.effect && job.gesture && <span className="chip">{job.cached ? "the" : "your"} {cue === "hold" ? "timing" : "noise"} read as a {job.gesture}: {EFFECT_LABELS[job.effect]?.name ?? job.effect}</span>}
      {cue === "clip" && <span className="chip">locked to the sound in your clip</span>}
      {r.inputs?.drawn && <span className="chip">where you drew it</span>}
      {r.inputs?.sources && r.inputs.sources.length > 1 && <span className="chip">call and response: each hit at its own source, in order</span>}
      {scene && early !== null && early !== 0 && <span className="chip">the model put it {ms(Math.abs(early))} ms {early > 0 ? "early" : "late"}; moved onto your cue</span>}
      {(r.warnings ?? []).map((w) => <span key={w} className="chip warn">{w}</span>)}
      {hit !== null && <span className="chip">{hit === 0 ? (scene ? "moved onto your cue's frame (as measured on the render)" : "the hit is on your cue's frame")
        : `the model was too far off to move all the way: ${hit > 0 ? "+" : ""}${ms(hit)} ms from your cue`}</span>}
      {(effect?.stops_with_voice ?? r.measured.stops_with_voice) && stop !== null && (
        <span className="chip">{stop <= 0 ? "stops when you stop" : `stops ${ms(stop)} ms after you do`}</span>
      )}
      {!job.cached && <span className="chip">{r.seconds_total.toFixed(0)} s in total</span>}
    </div>
  );
}

/** Step 4: pick the effect and how to make it, then the shot. */
export default function ShotPanel({ plate, point, direction, performance, effects, modes, madeAheadScene, strokes, sources = [], initialJob, onBusy, onDone }: {
  plate: PickedPlate;
  point: [number, number];
  direction: number | null;
  performance: Performance;
  effects: EffectInfo[];
  modes: Mode[];
  madeAheadScene?: boolean; // the GPU is offline, but sample scene shots were made ahead of time
  strokes: Stroke[];
  sources?: [number, number][]; // call and response: one source per hit, in order (the first is the point)
  initialJob?: JobStatus | null; // this take's shot, if it was already made (coming back to it)
  onBusy?: (busy: boolean) => void;
  onDone: (job: JobStatus, effect: string) => void;
}) {
  const drawn = strokes.length > 0;
  const several = sources.length >= 2;
  const sceneOnly = drawn || several; // a drawing, or a source per hit, is something only a render into the scene can do
  const sceneLive = modes.includes("scene");
  // with the GPU offline, a sample shot that was made into the scene ahead of time still plays
  const ahead = !sceneLive && !several && !!madeAheadScene && madeAhead(plate, performance.cue, performance.voice, point, strokes, direction);
  const sceneOk = sceneLive || ahead;
  const firstMode: Mode = sceneOk ? "scene" : modes.includes("instant") ? "instant" : "render";
  const [mode, setModeState] = useState<Mode>(firstMode);
  // the cue decides (its gesture, read through the plate's own map), unless the performer picks
  const [effect, setEffect] = useState(AUTO_EFFECT);
  const [subject, setSubject] = useState(plate.subject ?? "");
  const [kinds, setKinds] = useState<ObjectKind[]>([]);
  const [kind, setKind] = useState<string | null>(null); // your own clip: what it is
  useEffect(() => { if (!plate.sample) fetchObjects().then(setKinds, () => setKinds([])); }, [plate.sample]);
  const [describe, setDescribe] = useState("");
  const choices = effects.filter((e) => (mode === "scene" ? e.scene : e.element !== false));
  const setMode = (m: Mode) => {
    setModeState(m);
    // an effect the new route can't make goes back to the cue's own; offline, the scene shot made ahead is the cue's
    if ((m === "scene" && ahead) || (effect !== AUTO_EFFECT
      && !effects.some((e) => e.id === effect && (m === "scene" ? e.scene : e.element !== false)))) {
      setEffect(AUTO_EFFECT);
    }
  };
  const [seed] = useState(1); // one seed per session: retakes differ only by the performance
  const [busy, setBusyState] = useState(false);
  const setBusy = (b: boolean) => { setBusyState(b); onBusy?.(b); };
  const [event, setEvent] = useState<JobEvent | null>(null);
  const [job, setJob] = useState<JobStatus | null>(initialJob ?? null);
  const [view, setView] = useState<View>("final.mp4");
  const [guideUrl, setGuideUrl] = useState<string | null>(null);

  const make = async () => {
    setBusy(true);
    setJob(null);
    setEvent({ stage: "uploading", progress: 0.01, message: "Uploading", t: 0 });
    try {
      const queued = await submitShot({ plate: plate.source, voice: performance.voice, face: performance.face, faceExt: performance.faceExt,
        timing: performance.timing, effect, point, seed, mode, direction, subject: subject.trim() || null, object: kind,
        cue: performance.cue, holds: performance.holds ?? null, strokes: drawn ? strokes : null,
        describe: describe.trim() || null, sources: several ? sources : null });
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
        <span className="muted">Effect</span>
        <button role="radio" aria-checked={effect === AUTO_EFFECT} disabled={busy}
          className={effect === AUTO_EFFECT ? "" : "secondary"} onClick={() => setEffect(AUTO_EFFECT)}>
          Your cue decides <span className="muted" style={{ marginLeft: 6 }}>{performance.cue === "hold" ? "short pops, long flames" : "pf · KA-BOOM · fwoooom · hissss"}</span>
        </button>
        {choices.map((e) => (
          <button key={e.id} role="radio" aria-checked={effect === e.id} disabled={busy || (ahead && mode === "scene")}
            title={ahead && mode === "scene" ? "needs the GPU, which is offline: this sample plays the effect it was made with" : undefined}
            className={effect === e.id ? "" : "secondary"} onClick={() => setEffect(e.id)}>
            {EFFECT_LABELS[e.id]?.name ?? e.id}
            <span className="muted" style={{ marginLeft: 6 }}>{EFFECT_LABELS[e.id]?.cue}</span>

          </button>
        ))}
      </div>
      {mode === "scene" && !ahead && (
        <label className="row">
          <span className="muted">Say what happens (optional)</span>
          <input value={describe} maxLength={200} disabled={busy} style={{ flex: "1 1 260px" }}
            placeholder={drawn ? "flames spread along the top of the pumpkin" : "a burst of flame blasts out of the jack-o'-lantern"}
            onChange={(e) => setDescribe(e.target.value)} />
        </label>
      )}
      <div className="row" role="radiogroup" aria-label="How to make it">
        <span className="muted">How</span>
        <button role="radio" aria-checked={mode === "scene"} disabled={busy || !sceneOk}
          className={mode === "scene" ? "" : "secondary"} onClick={() => setMode("scene")}>
          Into the scene <span className="muted">{ahead ? "made ahead of time for this sample: it plays now"
            : "LTX-2.5 renders it into your shot and the scene reacts, ~2 min"}</span>
        </button>
        <button role="radio" aria-checked={mode === "instant"} disabled={busy || !modes.includes("instant") || sceneOnly}
          className={mode === "instant" ? "" : "secondary"} onClick={() => setMode("instant")}>
          Instant <span className="muted">a pre-rendered element, no GPU, 20-30 s</span>
        </button>
      </div>
      <details>
        <summary className="muted">Advanced</summary>
        <div className="row">
          <button role="radio" aria-checked={mode === "render"} disabled={busy || !modes.includes("render") || sceneOnly}
            className={mode === "render" ? "" : "secondary"} onClick={() => setMode("render")}>
            Element <span className="muted">rendered on black and composited; your footage is never regenerated, ~45 s</span>
          </button>
        </div>
      </details>
      {sceneOnly && mode !== "scene" && (
        <p className="chip warn">{modes.includes("scene")
          ? `${drawn ? "A drawing" : "A source for each hit"} needs "Into the scene".`
          : `${drawn ? "A drawing" : "A source for each hit"} needs "Into the scene", and the GPU is offline right now: tap one source instead for an instant shot.`}</p>
      )}
      {!sceneLive && (
        <p className="muted">{ahead ? (mode === "scene"
          ? "The GPU is offline right now, but this sample was rendered into the scene ahead of time, so it plays now. The other effects need the GPU."
          : "The GPU is offline right now: instant mode places a pre-rendered element with your timing; Into the scene plays the one made ahead of time.")
          : madeAheadScene ? "The GPU is offline right now: instant mode places a pre-rendered element with your timing. Into the scene still plays for a sample clip at its marked spot with a noise marked ready, or with the clip's own snap."
            : "The GPU is offline right now: instant mode places a pre-rendered element with your timing."}</p>
      )}
      {mode === "scene" && !plate.sample && kinds.length > 0 && (
        <div className="row" role="radiogroup" aria-label="What is it">
          <span className="muted">What is it?</span>
          {kinds.map((k) => (
            <button key={k.id} role="radio" aria-checked={kind === k.id} disabled={busy}
              className={kind === k.id ? "" : "secondary"}
              onClick={() => { setKind(kind === k.id ? null : k.id); if (k.subject && !subject.trim()) setSubject(k.subject); }}>
              {k.title}
            </button>
          ))}
          {effect === AUTO_EFFECT && kind && <span className="muted">your cue does to it what it would to a {kinds.find((k) => k.id === kind)?.title.replace(/^an? /, "")}</span>}
        </div>
      )}
      {mode === "scene" && !plate.sample && (
        <label className="row">
          <span className="muted">{kinds.length ? "In your words (optional)" : "What is it?"}</span>
          <input value={subject} maxLength={120} disabled={busy} placeholder="the toy dinosaur on the desk"
            onChange={(e) => setSubject(e.target.value)} />
        </label>
      )}
      <div className="row">
        <button disabled={busy || effects.length === 0 || (sceneOnly && mode !== "scene")} onClick={() => void make()}>{job?.status === "done" ? "Make it again" : "Make the shot"}</button>
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
                  : several ? "Your rhythm, painted into your scene: each hit at its own source, in order, as big as you made it."
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
          <Chips job={job} effect={effects.find((e) => e.id === (job.effect ?? effect))} live={sceneLive} />
          <Retime key={job.id} job={job} stopsWithVoice={job.receipt?.mode !== "scene" && (effects.find((e) => e.id === (job.effect ?? effect))?.stops_with_voice ?? false)}
            onRetimed={(next) => { setJob(next); setView("final.mp4"); onDone(next, effect); }} />
          <div className="row">
            {job.files["final.mp4"] && <a className="btn secondary" href={job.files["final.mp4"]} download="cue-the-boom.mp4">Download the shot</a>}
            <details>
              <summary className="muted">Every file (the render, the guide, the receipt)</summary>
              <div className="row">
                {Object.entries(job.files).map(([name, url]) => <a key={name} className="btn secondary" href={url} download={name}>{name}</a>)}
              </div>
            </details>
          </div>
        </div>
      )}
    </section>
  );
}
