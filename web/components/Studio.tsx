"use client";

import { useEffect, useState } from "react";
import DrawPad from "@/components/DrawPad";
import Performer, { type Performance } from "@/components/Performer";
import PlatePicker, { type PickedPlate } from "@/components/PlatePicker";
import ShotPanel from "@/components/ShotPanel";
import TapPoint, { MAX_SOURCES } from "@/components/TapPoint";
import { atSampleSpot, drawnReady, type EffectInfo, fetchEffects, fetchHealth, fetchSamples, type Health, type JobStatus, type Samples, type Stroke } from "@/lib/shot";

interface Take { id: number; performance: Performance; job: JobStatus | null; effect: string | null }

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
          {!pointSet ? (
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
              <span className="chip">{drawing && strokes.length ? `drawn (${strokes.length} stroke${strokes.length === 1 ? "" : "s"})`
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
          drawn={drawing && strokes.length > 0}
          onDone={(p) => { setTakes((t) => [...t, { id: t.length + 1, performance: p, job: null, effect: null }]); setPerforming(false); }} />
      )}

      {plate && where && pointSet && !performing && cue && (
        <section>
          <h2>3. Cue it</h2>
          <div className="row">
            <span className="chip">{cue.cue === "clip" ? "the clip's own sound" : cue.cue === "hold" ? cue.label : cue.voice && "sample" in cue.voice ? `sample noise: ${cue.label}` : `your ${cue.label}`}</span>
            {cue.cue === "voice" && cue.previewUrl && <audio src={cue.previewUrl} controls preload="none" className="cue-audio" />}
            <button className="secondary" disabled={rendering} onClick={() => setPerforming(true)}>Cue it differently</button>
          </div>
          <p className="muted">Same shot, same seed: a shorter, louder or different noise makes a different shot.</p>
        </section>
      )}

      {plate && where && pointSet && !performing && current && (
        <ShotPanel key={current.id} plate={plate} point={where} direction={direction} performance={current.performance}
          strokes={drawing ? strokes : []} sources={sources} initialJob={current.job} onBusy={setRendering}
          effects={effects} modes={health?.modes ?? []} madeAheadScene={madeAheadScene}
          onDone={(job, effect) => {
            const id = current.id; // this take, even if the user has moved on to another before it finishes
            setTakes((t) => t.map((x) => (x.id === id ? { ...x, job, effect } : x)));
          }} />
      )}
    </>
  );
}
