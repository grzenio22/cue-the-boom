"use client";

import { useEffect, useState } from "react";
import DrawPad from "@/components/DrawPad";
import Performer, { type Performance } from "@/components/Performer";
import PlatePicker, { type PickedPlate } from "@/components/PlatePicker";
import ShotPanel from "@/components/ShotPanel";
import TapPoint from "@/components/TapPoint";
import { type EffectInfo, fetchEffects, fetchHealth, fetchSamples, type Health, type JobStatus, type Samples, type Stroke } from "@/lib/shot";

interface Take { id: number; performance: Performance; job: JobStatus | null; effect: string | null }

export default function Studio() {
  const [health, setHealth] = useState<Health | null>(null);
  const [effects, setEffects] = useState<EffectInfo[]>([]);
  const [samples, setSamples] = useState<Samples>({ plates: [], voices: [] });
  const [apiError, setApiError] = useState<string | null>(null);
  const [plate, setPlate] = useState<PickedPlate | null>(null);
  const [point, setPoint] = useState<[number, number] | null>(null);
  const [pointSet, setPointSet] = useState(false);
  const [direction, setDirection] = useState<number | null>(null);
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
    setPointSet(p.point !== null); // a sample comes with its spot; it can still be moved
    setDrawing(false);
    setStrokes([]);
    setTakes([]);
    setPerforming(true);
  };

  const current = takes.at(-1);
  // a drawing says where it comes from (its first point); in tap mode the tapped point wins
  const where: [number, number] | null = drawing && strokes.length ? strokes[0].points[0] : point;

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
                <button role="radio" aria-checked={drawing} className={drawing ? "" : "secondary"} onClick={() => setDrawing(true)}>Draw it</button>
              </div>
              {drawing ? (
                <>
                  <p className="muted">Draw where it goes: flames along the top of the pumpkin, smoke pouring out of a lamp. Start where it comes from; your cue fills the drawing.</p>
                  <DrawPad src={plate.url} strokes={strokes} onChange={setStrokes} />
                </>
              ) : (
                <>
                  <p className="muted">Tap where it comes from: a toy&apos;s mouth, a candle&apos;s wick, the coffee in a mug. Drag from there to aim it (a breath, a jet).</p>
                  <TapPoint src={plate.url} value={point} onChange={setPoint} direction={direction} onDirection={setDirection} />
                </>
              )}
              <div className="row"><button disabled={drawing ? strokes.length === 0 : !point} onClick={() => setPointSet(true)}>{drawing ? "That's where it goes" : "That's the spot"}</button></div>
            </>
          ) : (
            <div className="row">
              <span className="chip">{drawing && strokes.length ? `drawn (${strokes.length} stroke${strokes.length === 1 ? "" : "s"})` : direction != null ? `source set, aimed ${direction}°` : "source set"}</span>
              <button className="secondary" disabled={rendering} onClick={() => setPointSet(false)}>{drawing ? "Change the drawing" : "Move it"}</button>
            </div>
          )}
        </section>
      )}

      {plate && where && pointSet && performing && (
        <Performer plateUrl={plate.url} point={where} voices={samples.voices} takes={takes.length} clipSound={plate.sound}
          onDone={(p) => { setTakes((t) => [...t, { id: t.length + 1, performance: p, job: null, effect: null }]); setPerforming(false); }} />
      )}

      {plate && where && pointSet && !performing && current && (
        <ShotPanel key={current.id} plate={plate} point={where} direction={direction} performance={current.performance}
          strokes={drawing ? strokes : []} initialJob={current.job} onBusy={setRendering}
          effects={effects} modes={health?.modes ?? []}
          onDone={(job, effect) => {
            const id = current.id; // this take, even if the user has moved on to another before it finishes
            setTakes((t) => t.map((x) => (x.id === id ? { ...x, job, effect } : x)));
          }} />
      )}

      {plate && where && pointSet && !performing && (
        <section>
          <div className="row">
            <button className="secondary" disabled={rendering} onClick={() => setPerforming(true)}>Cue it differently</button>
            <span className="muted">Same shot, same seed: a shorter, louder or different noise makes a different shot.</span>
          </div>
        </section>
      )}
    </>
  );
}
