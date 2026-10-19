"use client";

import { useEffect, useState } from "react";
import type { Performance } from "@/components/Performer";
import EffectBadge from "@/components/EffectBadge";
import Retime from "@/components/Retime";
import type { PickedPlate } from "@/components/PlatePicker";
import { AUTO_EFFECT, EFFECT_LABELS, type EffectInfo, HAND_EFFECTS, HAND_LABELS, fetchObjects, followJob, type JobEvent, type JobStatus, madeAhead, type Mode, type ObjectKind, pollUntilDone, replayable, replayShot, type Stroke, submitShot } from "@/lib/shot";

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
      {r.replayed && <span className="chip">{(r.replayed.renders ?? 1) > 1
        ? `${r.replayed.renders} shots performed together, each at its own spot (${r.replayed.new_hits} hit${r.replayed.new_hits === 1 ? "" : "s"}): no new render`
        : `performed again on the same render (${r.replayed.new_hits} hit${r.replayed.new_hits === 1 ? "" : "s"} from its ${r.replayed.render_hits}): no new render`}</span>}
      {r.retimed && <span className="chip">re-timed by hand: hit {r.retimed.hit_shift_frames >= 0 ? "+" : ""}{ms(r.retimed.hit_shift_frames)} ms{r.retimed.stop_frame !== null ? `, stop at ${(r.retimed.stop_frame / 24).toFixed(2)} s` : ""}, same element</span>}
      <span className="chip">{scene ? (r.replayed ? `rendered into the scene ahead of time (${r.render.seconds.toFixed(0)} s on open weights, LTX-2.5)`
        : `rendered into your scene in ${r.render.seconds.toFixed(0)} s on open weights (LTX-2.5, from your plate's frame)`)
        : lib ? `instant: a pre-rendered LTX-2.5 element, timed by your voice` : `rendered from your voice in ${r.render.seconds.toFixed(0)} s on open weights`}</span>
      {r.hits && r.hits.length > 1 && (() => {
        const merged = r.hits.filter((h) => h.merged).length;
        return <span className="chip">{merged
          ? `${r.hits.length} hits; the model kept ${r.hits.length - merged} apart and merged ${merged} into a neighbour`
          : r.replayed ? `${r.hits.length} hits, each on your cue, brighter or dimmer by how loud it was`
          : `${r.hits.length} hits, each its own ${scene ? "event" : "burst"}, sized by how you made it`}</span>;
      })()}
      {job.effect && job.heard && !r.replayed && <span className="chip">heard a &ldquo;{job.heard}&rdquo;: {EFFECT_LABELS[job.effect]?.name ?? job.effect}</span>}
      {job.effect && job.gesture && !job.heard && <span className="chip">{job.cached ? "the" : "your"} {cue === "hold" ? "timing" : "noise"} read as a {job.gesture}: {EFFECT_LABELS[job.effect]?.name ?? job.effect}</span>}
      {cue === "clip" && <span className="chip">locked to the sound in your clip</span>}
      {r.inputs?.drawn && <span className="chip">where you drew it</span>}
      {r.inputs?.sources && r.inputs.sources.length > 1 && <span className="chip">call and response: each hit at its own source, in order</span>}
      {scene && !r.replayed && early !== null && early !== 0 && <span className="chip">the model put it {ms(Math.abs(early))} ms {early > 0 ? "early" : "late"}; moved onto your cue</span>}
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

/** A shot from your hand, as plain claims: rendered once over black, carried on your hand, fired on each strike. */
function HandChips({ job }: { job: JobStatus }) {
  const r = job.receipt;
  const b = r?.bending;
  if (!r || !b) return null;
  return (
    <div className="row">
      <span className="chip">{b.instant ? "instant: an LTX-2.5 element rendered ahead of time, carried on your hand (no GPU)"
        : b.performed_again ? "performed again on the same render: no new render"
        : r.render.seconds > 0 ? `rendered over black in ${r.render.seconds.toFixed(0)} s on open weights (LTX-2.5)` : "made, not rendered: an air blast"}</span>
      <span className="chip">carried on your hand frame by frame, and lighting it from where it is</span>
      {b.hits.length > 1 && <span className="chip">{b.hits.length} hits, each on its own strike, as big as it was loud</span>}
      {b.snap.used === "motion" && <span className="chip">launched on your hand&apos;s strike</span>}
      {b.charge && <span className="chip">charged up by your hum for {b.charge.seconds.toFixed(1)} s, released on the strike</span>}
      {r.auto?.why && <span className="chip">{r.auto.why}</span>}
      <span className="chip">{r.seconds_total.toFixed(0)} s in total</span>
    </div>
  );
}

/** Step 4: pick the effect and how to make it, then the shot. */
export default function ShotPanel({ plate, point, direction, performance, effects, modes, madeAheadScene, strokes, sources = [], hand = null, initialJob, replayOf = null, replayWith = [], siblings = [], onReplay, onBusy, onDone }: {
  plate: PickedPlate;
  point: [number, number];
  direction: number | null;
  performance: Performance;
  effects: EffectInfo[];
  modes: Mode[];
  madeAheadScene?: boolean; // the GPU is offline, but sample scene shots were made ahead of time
  strokes: Stroke[];
  sources?: [number, number][]; // call and response: one source per hit, in order (the first is the point)
  hand?: { frame: number } | null; // your hand is the source (element bending), tapped on this frame
  initialJob?: JobStatus | null; // this take's shot, if it was already made (coming back to it)
  replayOf?: string | null; // play this take on that finished shot's render (no new render)
  replayWith?: string[]; // and those other shots on the plate with it, as one scene
  siblings?: string[]; // the other finished shots on this still plate that could be performed with this one
  onReplay?: (jobId: string, together: string[]) => void; // the performer wants to perform this shot (and those) again
  onBusy?: (busy: boolean) => void;
  onDone: (job: JobStatus, effect: string) => void;
}) {
  const drawn = strokes.length > 0;
  const several = sources.length >= 2;
  const sceneOnly = drawn || several; // a drawing, or a source per hit, is something only a render into the scene can do
  const sceneLive = modes.includes("scene");
  // with the GPU offline, a sample shot that was made into the scene ahead of time still plays
  const ahead = !sceneLive && !several && !!madeAheadScene && madeAhead(plate, performance.cue, performance.voice, point, strokes, direction);
  // ... and your own take plays a sample's scene shot made ahead: each of your hits plays one of the render's
  const replay = !sceneLive && !several && !ahead && !!madeAheadScene && replayable(plate, performance.cue, performance.voice, point, strokes, direction);
  const sceneOk = sceneLive || ahead || replay;
  const firstMode: Mode = hand ? "bending" : sceneOk ? "scene" : modes.includes("instant") ? "instant" : "render";
  const [mode, setModeState] = useState<Mode>(firstMode);
  // the cue decides (its gesture, read through the plate's own map), unless the performer picks
  const [effect, setEffect] = useState(AUTO_EFFECT);
  const [pickOwn, setPickOwn] = useState(false); // a hand shot: the cue decides unless you open the list
  const [subject, setSubject] = useState(plate.subject ?? "");
  const [kinds, setKinds] = useState<ObjectKind[]>([]);
  const [kind, setKind] = useState<string | null>(null); // your own clip: what it is
  useEffect(() => { if (!plate.sample) fetchObjects().then(setKinds, () => setKinds([])); }, [plate.sample]);
  const [describe, setDescribe] = useState("");
  const handOffline = !!hand && !modes.includes("bending"); // a sample hand's element made ahead: the cue plays it
  const choices = handOffline ? [] : hand ? effects.filter((e) => HAND_EFFECTS.includes(e.id))
    : effects.filter((e) => (mode === "scene" ? e.scene : e.element !== false));
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

  const make = async (pick?: string, reuse?: string) => {
    const eff = pick ?? effect;
    setBusy(true);
    setJob(null);
    setEvent({ stage: "uploading", progress: 0.01, message: "Uploading", t: 0 });
    try {
      const queued = replayOf
        ? await replayShot(replayOf, { voice: performance.voice, cue: performance.cue, holds: performance.holds ?? null, timing: performance.timing,
          together: replayWith })
        : await submitShot({ plate: plate.source, voice: performance.voice, face: performance.face, faceExt: performance.faceExt,
          timing: performance.timing, effect: eff, point, seed, mode, direction, frame: hand?.frame ?? null, reuse: reuse ?? null, subject: subject.trim() || null, object: kind,
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
      if (done.status === "done") onDone(done, eff);
    } catch (e) {
      setJob({ id: "", status: "error", stage: "error", progress: 0, message: "", error: (e as Error).message, files: {}, receipt: null });
    } finally {
      setBusy(false);
    }
  };

  // a take for performing a shot again plays at once: nothing to choose, nothing to render
  useEffect(() => {
    if (!replayOf || initialJob) return;
    const start = setTimeout(() => void make(), 0); // once, after mounting (strict mode mounts twice in development)
    return () => clearTimeout(start);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const bent = job?.receipt?.mode === "bending";
  const again = job?.status === "done" && ((job.receipt?.mode === "scene" && job.receipt.plate?.still) || bent) ? (replayOf ?? job.id) : null;
  const auto = job?.receipt?.auto;

  return (
    <section>
      <h2>4. Make the shot</h2>
      {performance.warnings.map((w) => <p key={w} className="chip warn">{w}</p>)}
      {!replayOf && <div className="row" role="radiogroup" aria-label="Effect">
        <span className="muted">Effect</span>
        <button role="radio" aria-checked={effect === AUTO_EFFECT} disabled={busy}
          className={effect === AUTO_EFFECT ? "" : "secondary"} onClick={() => setEffect(AUTO_EFFECT)}>
          Your cue decides <span className="muted" style={{ marginLeft: 6 }}>{performance.cue === "hold" ? "short pops, long flames" : "pf · KA-BOOM · fwoooom · hissss"}</span>
        </button>
        {hand && !pickOwn && choices.length > 0 && (
          <button className="secondary" disabled={busy} onClick={() => setPickOwn(true)}>Choose it yourself</button>
        )}
        {(!hand || pickOwn) && choices.map((e) => (
          <button key={e.id} role="radio" aria-checked={effect === e.id} disabled={busy || (ahead && mode === "scene")}
            title={ahead && mode === "scene" ? "needs the GPU, which is offline: this sample plays the effect it was made with" : undefined}
            className={effect === e.id ? "" : "secondary"} onClick={() => setEffect(e.id)}>
            {(hand && HAND_LABELS[e.id]) || EFFECT_LABELS[e.id]?.name || e.id}
            <span className="muted" style={{ marginLeft: 6 }}>{EFFECT_LABELS[e.id]?.cue}</span>

          </button>
        ))}
      </div>}
      {handOffline && !replayOf && (
        <p className="muted">The GPU is offline: your take plays this hand&apos;s fireball, made ahead of time, on each of your hits.
          No new render; ready in about a minute.</p>
      )}
      {hand && !handOffline && !replayOf && (
        <p className="muted">Your cue decides with your hand: a long fffff from two fingers is a flamethrower, KA-BOOM from a palm a
          fireball, a hiss from a palm steam, a pf a blast of air. {sceneLive
            ? "Rendered once over black on the GPU (~1 min), then carried on your hand."
            : "No GPU right now: an instant shot, an element rendered ahead of time with LTX-2.5, carried on your hand (a minute or two on a laptop CPU)."}</p>
      )}
      {!hand && mode === "scene" && !ahead && (
        <label className="row">
          <span className="muted">Say what happens (optional)</span>
          <input value={describe} maxLength={200} disabled={busy} style={{ flex: "1 1 260px" }}
            placeholder={drawn ? "flames spread along the top of the pumpkin" : "a burst of flame blasts out of the jack-o'-lantern"}
            onChange={(e) => setDescribe(e.target.value)} />
        </label>
      )}
      {!hand && <>
      <div className="row" role="radiogroup" aria-label="How to make it">
        <span className="muted">How</span>
        <button role="radio" aria-checked={mode === "scene"} disabled={busy || !sceneOk}
          className={mode === "scene" ? "" : "secondary"} onClick={() => setMode("scene")}>
          Into the scene <span className="muted">{ahead ? "made ahead of time for this sample: it plays now"
            : replay ? "made ahead of time for this sample: your take plays its render, no GPU, ~20 s"
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
      </>}
      {sceneOnly && mode !== "scene" && (
        <p className="chip warn">{modes.includes("scene")
          ? `${drawn ? "A drawing" : "A source for each hit"} needs "Into the scene".`
          : `${drawn ? "A drawing" : "A source for each hit"} needs "Into the scene", and the GPU is offline right now: tap one source instead for an instant shot.`}</p>
      )}
      {!hand && !sceneLive && (
        <p className="muted">{ahead ? (mode === "scene"
          ? "The GPU is offline right now, but this sample was rendered into the scene ahead of time, so it plays now. The other effects need the GPU."
          : "The GPU is offline right now: instant mode places a pre-rendered element with your timing; Into the scene plays the one made ahead of time.")
          : madeAheadScene ? "The GPU is offline right now: instant mode places a pre-rendered element with your timing. Into the scene still plays for a sample clip at its marked spot: with a noise marked ready, the clip's own snap, or your own take on the render made ahead."
            : "The GPU is offline right now: instant mode places a pre-rendered element with your timing."}</p>
      )}
      {!hand && mode === "scene" && !plate.sample && kinds.length > 0 && (
        <div className="row" role="radiogroup" aria-label="What is it">
          <span className="muted">What is it?</span>
          {kinds.map((k) => (
            <button key={k.id} role="radio" aria-checked={kind === k.id} disabled={busy}
              className={kind === k.id ? "" : "secondary"}
              onClick={() => { setKind(kind === k.id ? null : k.id); if (k.subject && !subject.trim()) setSubject(k.subject); }}>
              {k.title}
            </button>
          ))}
          {effect === AUTO_EFFECT && kind && (Object.keys(kinds.find((k) => k.id === kind)?.scene ?? {}).length
            ? <span className="muted">your cue does to it what it would to a {kinds.find((k) => k.id === kind)?.title.replace(/^an? /, "")}</span>
            : <span className="muted">the sound picks it: a boom bursts, a whoosh is a jet of fire where you aim it (a plume if you don&apos;t), a hiss is steam, a puff is a blast of air</span>)}
        </div>
      )}
      {!hand && mode === "scene" && !plate.sample && (
        <label className="row">
          <span className="muted">{kinds.length ? "In your words (optional)" : "What is it?"}</span>
          <input value={subject} maxLength={120} disabled={busy} placeholder="the toy dinosaur on the desk"
            onChange={(e) => setSubject(e.target.value)} />
        </label>
      )}
      <div className="row">
        <button disabled={busy || effects.length === 0 || (!hand && sceneOnly && mode !== "scene")} onClick={() => void make()}>{job?.status === "done" ? "Make it again" : "Make the shot"}</button>
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
          <video key={view} src={job.files[view]} poster={job.files["poster.jpg"]} controls autoPlay playsInline className="result" />
          {bent ? <HandChips job={job} /> : <Chips job={job} effect={effects.find((e) => e.id === (job.effect ?? effect))} live={sceneLive} />}
          {bent && auto && !job.receipt?.bending?.performed_again && (
            <EffectBadge choices={HAND_EFFECTS} names={HAND_LABELS} onPick={(id) => { setEffect(id); void make(id, job.id); }}
              detected={{ effect: auto.picked, kind: auto.heard, gesture: auto.hand, confidence: auto.hand ? 0.8 : 0.4,
                mismatch: !!job.receipt?.bending?.snap.mismatch }} />
          )}
          {again && onReplay && (
            <div className="row">
              <button onClick={() => onReplay(again, replayWith)}>Perform it again</button>
              <span className="muted">{bent ? "a new rhythm on the same hand take: no new render, ready in about a minute"
                : "a new rhythm or another voice on the same render: no new render, ready in about 20 s"}</span>
            </div>
          )}
          {again && onReplay && !bent && siblings.length > 0 && (
            <div className="row">
              <button onClick={() => onReplay(again, siblings)}>Perform all {siblings.length + 1} together</button>
              <span className="muted">your shots on this plate as one scene, each effect at its own spot: one rhythm plays them all, no new render</span>
            </div>
          )}
          {!job.receipt?.replayed && !bent && <Retime key={job.id} job={job} stopsWithVoice={job.receipt?.mode !== "scene" && (effects.find((e) => e.id === (job.effect ?? effect))?.stops_with_voice ?? false)}
            onRetimed={(next) => { setJob(next); setView("final.mp4"); onDone(next, effect); }} />}
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
