"use client";

import { useEffect, useState } from "react";
import AimOverlay, { type AimValue } from "@/components/AimOverlay";
import DrawPad from "@/components/DrawPad";
import Performer, { type Performance } from "@/components/Performer";
import PlatePicker, { type PickedPlate } from "@/components/PlatePicker";
import ShotPanel from "@/components/ShotPanel";
import TapPoint, { MAX_SOURCES } from "@/components/TapPoint";
import { atSampleSpot, drawnReady, type EffectInfo, fetchEffects, fetchHealth, fetchSamples, type Health, type JobStatus, type Samples, type Stroke } from "@/lib/shot";

interface Take { id: number; performance: Performance; job: JobStatus | null; effect: string | null; replayOf?: string | null; replayWith?: string[] }

/** The plate's first frame, small, with where the effect comes from marked on it. */
function Spot({ src, at }: { src: string; at: [number, number] }) {
  const [aspect, setAspect] = useState(9 / 16);
  return (
    <span className="spot" style={{ aspectRatio: aspect }}>
      <video src={`${src}#t=0.1`} muted playsInline preload="metadata"
        onLoadedMetadata={(e) => { const v = e.currentTarget; if (v.videoWidth) setAspect(v.videoWidth / v.videoHeight); }} />
      <i style={{ left: `${at[0] * 100}%`, top: `${at[1] * 100}%` }} />
    </span>
  );
}

/** How many frames (24 fps, at most 10 s: what a shot keeps) a clip has, once its metadata is in. */
function useFrames(src: string | undefined): number {
  const [n, setN] = useState(240);
  useEffect(() => {
    if (!src) return;
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => { if (Number.isFinite(v.duration)) setN(Math.max(1, Math.min(240, Math.floor(v.duration * 24)))); };
    v.src = src;
    return () => { v.removeAttribute("src"); v.load(); };
  }, [src]);
  return n;
}

export default function Studio() {
  const [health, setHealth] = useState<Health | null>(null);
  const [effects, setEffects] = useState<EffectInfo[]>([]);
  const [samples, setSamples] = useState<Samples>({ plates: [], voices: [] });
  const [apiError, setApiError] = useState<string | null>(null);
  const [plate, setPlate] = useState<PickedPlate | null>(null);
  const [point, setPoint] = useState<[number, number] | null>(null);
  const [pointSet, setPointSet] = useState(false);
  const [direction, setDirection] = useState<number | null>(null);
  const [more, setMore] = useState<[number, number][]>([]); // call and response: the later hits' sources
  const [adding, setAdding] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [takes, setTakes] = useState<Take[]>([]);
  const [performing, setPerforming] = useState(true);
  const [rendering, setRendering] = useState(false); // a shot is being made: stepping back would lose it
  const [replayOf, setReplayOf] = useState<string | null>(null); // the next take plays this shot's render again
  const [replayWith, setReplayWith] = useState<string[]>([]); // and these other shots' renders with it
  const [hand, setHand] = useState(false); // your hand is the source: the element comes off it (element bending)
  const [aimFrame, setAimFrame] = useState(0); // the frame your hand is tapped on
  const [aim, setAim] = useState<AimValue | null>(null);
  const frames = useFrames(plate?.url);

  useEffect(() => {
    Promise.all([fetchHealth(), fetchEffects(), fetchSamples()]).then(([h, e, s]) => {
      setHealth(h);
      setEffects(e);
      setSamples(s);
    }, (err: Error) => setApiError(err.message));
  }, []);

  const pick = (p: PickedPlate) => {
    setPlate(p);
    setPoint(p.point);
    setDirection(p.direction);
    setMore([]);
    setAdding(false);
    setPointSet(p.point !== null); // a sample comes with its spot; it can still be moved
    setDrawing(false);
    setStrokes([]);
    setTakes([]);
    setPerforming(true);
    setReplayOf(null); // a replay of the last plate's shots has nothing to play here
    setReplayWith([]);
    setHand(!!p.hand); // a sample hand clip starts as a hand shot, at its own spot
    setAim(null);
    setAimFrame(p.hand?.frame ?? 0);
  };
  const handLive = health?.modes.includes("bending") ?? false;
  const [settingUp, setSettingUp] = useState(false);
  // a set-up under way (started here or before a reload): keep asking until it is done, ready or not
  const installing = !!health?.hand?.installing;
  useEffect(() => {
    if (!installing || settingUp) return;
    const t = setInterval(() => { fetchHealth().then(setHealth, () => undefined); }, 3000);
    return () => clearInterval(t);
  }, [installing, settingUp]);
  const setUpHands = async () => { // the server sets the tracker up; the hand appears when health says it is ready
    setSettingUp(true);
    try {
      await fetch("/api/hands/install", { method: "POST" });
      for (let i = 0; i < 600; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        const h = await fetchHealth();
        setHealth(h);
        if (h.hand?.tracker || h.hand?.install_error || (!h.hand?.installing && i > 1)) break; // done, ready or not
      }
    } finally {
      setSettingUp(false);
    }
  };
  // with the GPU offline, a sample hand clip whose element was made ahead: your take plays it on the hand
  const handAhead = !handLive && (health?.made_ahead?.includes("bending") ?? false) && !!plate?.hand;
  const toHand = (on: boolean) => {
    setHand(on); setDrawing(false); setMore([]); setAdding(false); setAim(null);
    if (on && handAhead && plate) { setPoint(plate.point); setAimFrame(plate.hand?.frame ?? 0); } else if (on) setPoint(null);
  };

  const current = takes.at(-1);
  const sceneLive = health?.modes.includes("scene") ?? false;
  const madeAheadScene = !sceneLive && (health?.made_ahead?.includes("scene") ?? false);
  // with the GPU offline, the sample noises whose scene shot was made ahead of time (at the sample's own spot)
  const ready = madeAheadScene && plate?.sample && more.length === 0 && atSampleSpot(plate.point, point, plate.direction, direction)
    ? Object.keys(plate.scene ?? {}).filter((k) => /\.(wav|mp3|m4a|ogg|webm)$/.test(k)) : [];
  // with the GPU offline, a sample's own drawing still plays: it was rendered ahead of time
  const replay = madeAheadScene && !!plate?.drawn?.length;
  const startDrawing = () => {
    setDrawing(true);
    if (!sceneLive && replay && plate?.drawn) setStrokes(plate.drawn[0].strokes);
  };
  // a drawing says where it comes from (its first point); in tap mode the tapped point wins
  const where: [number, number] | null = drawing && strokes.length ? strokes[0].points[0] : point;
  const sources: [number, number][] = !drawing && point && more.length ? [point, ...more] : [];
  const cue = current?.performance;
  // the shots rendered into this still plate so far (their own renders, not shots performed again), one per spot: they
  // play together, each at its own spot (a retake at the same spot is the same source, not another one)
  const spotOf = (t?: Take) => t?.job?.receipt?.plate?.point ?? null;
  const near = (a: [number, number] | null, b: [number, number] | null) => !!a && !!b && Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.03;
  const rendered: { id: string; at: [number, number] | null }[] = [];
  for (const t of takes) {
    if (t.job?.status !== "done" || t.job.receipt?.mode !== "scene" || !t.job.receipt.plate?.still || t.job.receipt.replayed) continue;
    const at = spotOf(t);
    if (!rendered.some((r) => near(r.at, at))) rendered.push({ id: t.job.id, at });
  }
  const here = takes.find((t) => t.job?.id === (current?.replayOf ?? current?.job?.id)) ?? current;
  const siblings = rendered.filter((r) => r.id !== (current?.replayOf ?? current?.job?.id) && !near(r.at, spotOf(here))).map((r) => r.id);

  if (apiError && !health) {
    return (
      <section>
        <p className="chip bad">The render server isn&apos;t reachable ({apiError}).</p>
        <p className="muted">Start it with <code>cuesync serve</code> (see the README), then reload this page.</p>
      </section>
    );
  }

  return (
    <>
      {!plate && <PlatePicker samples={samples.plates} onPick={pick} />}

      {plate && (
        <section>
          <h2>2. Where does it come from?</h2>
          <div className="row">
            <span className="muted">{plate.title}</span>
            <button className="secondary" disabled={rendering} onClick={() => setPlate(null)}>Different shot</button>
          </div>
          {!pointSet && !handLive && !handAhead && health?.hand && !health.hand.tracker && (
            <div className="row">
              <span className="muted"><strong>Your hand</strong> (fire off your fingers or palm) needs the hand tracker, set up once
                (about a minute). The sample <em>Your hand: cupped palm</em> plays without it.</span>
              {health.hand.can_install ? (
                <button className="secondary" disabled={!!health.hand.installing || settingUp} onClick={() => void setUpHands()}>
                  {health.hand.installing || settingUp ? "Setting up the hand tracker…" : "Set up the hand tracker"}
                </button>
              ) : <code>cuesync hands --install</code>}
              {health.hand.install_error && <span className="chip bad">{health.hand.install_error}</span>}
            </div>
          )}
          {!pointSet && (handLive || handAhead) && (
            <div className="row" role="radiogroup" aria-label="What it comes from">
              <button role="radio" aria-checked={!hand} className={hand ? "secondary" : ""} onClick={() => toHand(false)}>A thing in the shot</button>
              <button role="radio" aria-checked={hand} className={hand ? "" : "secondary"} onClick={() => toHand(true)}>
                Your hand <span className="muted" style={{ marginLeft: 6 }}>fire, steam or air off your fingers or palm</span>
              </button>
            </div>
          )}
          {!pointSet && hand && handAhead ? (
            <>
              <p className="muted">The GPU is offline right now: this hand&apos;s fireball was made ahead of time, at the spot marked on its
                palm. Perform your own rhythm and it fires on each beat, carried on the hand, with no new render.</p>
              <div className="row">
                {plate.point && <Spot src={plate.url} at={plate.point} />}
                <button onClick={() => setPointSet(true)}>Perform it</button>
              </div>
            </>
          ) : !pointSet && hand ? (
            <>
              <p className="muted">Find a frame where your hand is clear, tap where it comes from (your fingertips, your palm) and drag to aim it.
                It is carried on your hand from there, wherever the hand goes.</p>
              <label className="row">
                <span className="muted">Frame</span>
                <input type="range" min={0} max={frames - 1} value={Math.min(aimFrame, frames - 1)} style={{ flex: "1 1 260px" }}
                  onChange={(e) => { setAimFrame(Number(e.target.value)); setAim(null); setPoint(null); }} />
                <span className="muted">{(aimFrame / 24).toFixed(1)} s</span>
              </label>
              <AimOverlay src={plate.url} frame={Math.min(aimFrame, frames - 1)} value={aim}
                onChange={(v) => { setAim(v); setPoint([v.x, v.y]); setDirection(v.angle); }} />
              <div className="row"><button disabled={!point} onClick={() => setPointSet(true)}>That&apos;s my hand</button></div>
            </>
          ) : !pointSet ? (
            <>
              <div className="row" role="radiogroup" aria-label="Tap or draw">
                <button role="radio" aria-checked={!drawing} className={drawing ? "secondary" : ""} onClick={() => setDrawing(false)}>Tap the source</button>
                <button role="radio" aria-checked={drawing} className={drawing ? "" : "secondary"} disabled={!sceneLive && !replay}
                  onClick={startDrawing}>Draw it</button>
              </div>
              {!sceneLive && health && (
                <p className="muted">{replay
                  ? "The GPU is offline right now: Draw it plays the drawing made ahead of time for this sample (cue it with fwoooOOOM). A drawing of your own needs the GPU."
                  : "Drawing renders live on the GPU, which is offline right now: tap the source instead."}</p>
              )}
              {drawing ? (
                <>
                  <p className="muted">{sceneLive
                    ? "Draw where it goes: flames along the top of the pumpkin, smoke pouring out of a lamp. Start where it comes from; your cue fills the drawing."
                    : "The drawing made ahead of time: flames along the top of the pumpkin. The circle is where they start; the cue fills the rest."}</p>
                  <DrawPad src={plate.url} strokes={strokes} onChange={setStrokes} locked={!sceneLive} />
                </>
              ) : (
                <>
                  <p className="muted">Tap where it comes from: a toy&apos;s mouth, a candle&apos;s wick, the coffee in a mug. Drag from there to aim it (a breath, a jet).</p>
                  <TapPoint src={plate.url} value={point} onChange={setPoint} direction={direction} onDirection={setDirection}
                    more={more} adding={adding} onMore={(m) => { setMore(m); if (m.length >= MAX_SOURCES - 1) setAdding(false); }} />
                  {sceneLive && (
                    <div className="row">
                      <button className={adding ? "" : "secondary"} aria-pressed={adding} disabled={!point || more.length >= MAX_SOURCES - 1}
                        onClick={() => setAdding((a) => !a)}>
                        {adding ? `Tap the thing for hit ${more.length + 2}` : more.length ? "Add another" : "One thing per hit"}
                      </button>
                      {more.length > 0 && <button className="secondary" onClick={() => { setMore([]); setAdding(false); }}>Back to one source</button>}
                      <span className="muted">{more.length
                        ? `${more.length + 1} sources: your rhythm fires them in order, one per hit (pf, pf, KA-BOOM).`
                        : "Call and response: tap a different thing for each hit of a rhythm."}</span>
                    </div>
                  )}
                  {madeAheadScene && plate.sample && plate.point && !atSampleSpot(plate.point, point, plate.direction, direction) && (
                    <div className="row">
                      <span className="muted">Off the sample&apos;s marked spot, Into the scene needs the GPU (offline right now): this makes an instant shot.</span>
                      <button className="secondary" onClick={() => { setPoint(plate.point); setDirection(plate.direction); }}>Back to the sample&apos;s spot</button>
                    </div>
                  )}
                </>
              )}
              <div className="row"><button disabled={drawing ? strokes.length === 0 : !point} onClick={() => setPointSet(true)}>{drawing ? "That's where it goes" : more.length ? "Those are the sources" : "That's the spot"}</button></div>
            </>
          ) : (
            <div className="row">
              {where && <Spot src={plate.url} at={where} />}
              <span className="chip">{hand ? `your hand at ${(aimFrame / 24).toFixed(1)} s${direction != null ? `, aimed ${direction}°` : ""}`
                : drawing && strokes.length ? `drawn (${strokes.length} stroke${strokes.length === 1 ? "" : "s"})`
                : sources.length ? `${sources.length} sources, one per hit` : direction != null ? `source set, aimed ${direction}°` : "source set"}</span>
              <button className="secondary" disabled={rendering} onClick={() => setPointSet(false)}>{drawing ? "Change the drawing" : plate.drawn?.length ? "Move it, or draw it" : "Move it"}</button>
              {madeAheadScene && plate.sample && plate.point && !drawing && !atSampleSpot(plate.point, point, plate.direction, direction) && (
                <button className="secondary" disabled={rendering} onClick={() => { setPoint(plate.point); setDirection(plate.direction); }}
                  title="The sample's scene shots were made at its marked spot; anywhere else needs the GPU">Back to the sample&apos;s spot</button>
              )}
            </div>
          )}
        </section>
      )}

      {plate && where && pointSet && performing && (
        <Performer plateUrl={plate.url} point={where} voices={samples.voices} takes={takes.length} clipSound={plate.sound}
          scene={plate.sample ? plate.scene : undefined} ready={drawing ? drawnReady(plate, strokes) : ready} live={sceneLive}
          drawn={drawing && strokes.length > 0} hand={hand}
          onDone={(p) => {
            setTakes((t) => [...t, { id: t.length + 1, performance: p, job: null, effect: null, replayOf, replayWith }]);
            setReplayOf(null);
            setReplayWith([]);
            setPerforming(false);
          }} />
      )}

      {plate && where && pointSet && !performing && cue && (
        <section>
          <h2>3. Cue it</h2>
          <div className="row">
            <span className="chip">{cue.cue === "clip" ? "the clip's own sound" : cue.cue === "hold" ? cue.label : cue.voice && "sample" in cue.voice ? `sample noise: ${cue.label}` : `your ${cue.label}`}</span>
            {cue.cue === "voice" && cue.previewUrl && <audio src={cue.previewUrl} controls preload="metadata" className="cue-audio" />}
            <button className="secondary" disabled={rendering} onClick={() => { setReplayOf(null); setPerforming(true); }}>Cue it differently</button>
          </div>
          <p className="muted">Same shot, same seed: a shorter, louder or different noise makes a different shot.</p>
        </section>
      )}

      {plate && where && pointSet && !performing && current && (
        <ShotPanel key={current.id} plate={plate} point={where} direction={direction} performance={current.performance}
          strokes={drawing ? strokes : []} sources={sources} hand={hand ? { frame: aim?.frame ?? aimFrame } : null}
          initialJob={current.job} onBusy={setRendering}
          replayOf={current.replayOf ?? null} replayWith={current.replayWith ?? []}
          siblings={siblings}
          onReplay={(id, together) => { setReplayOf(id); setReplayWith(together); setPerforming(true); }}
          effects={effects} modes={health?.modes ?? []} madeAheadScene={madeAheadScene}
          onDone={(job, effect) => {
            const id = current.id; // this take, even if the user has moved on to another before it finishes
            setTakes((t) => t.map((x) => (x.id === id ? { ...x, job, effect } : x)));
          }} />
      )}
    </>
  );
}
