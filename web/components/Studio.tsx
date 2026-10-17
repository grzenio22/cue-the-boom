"use client";

import { useEffect, useState } from "react";
import Performer, { type Performance } from "@/components/Performer";
import PlatePicker, { type PickedPlate } from "@/components/PlatePicker";
import ShotPanel from "@/components/ShotPanel";
import TapPoint from "@/components/TapPoint";
import { compareTakes, type EffectInfo, fetchEffects, fetchHealth, fetchSamples, type Health, type JobStatus, type Samples } from "@/lib/shot";

interface Take { id: number; performance: Performance; job: JobStatus | null; effect: string | null }

export default function Studio() {
  const [health, setHealth] = useState<Health | null>(null);
  const [effects, setEffects] = useState<EffectInfo[]>([]);
  const [samples, setSamples] = useState<Samples>({ plates: [], voices: [] });
  const [apiError, setApiError] = useState<string | null>(null);
  const [plate, setPlate] = useState<PickedPlate | null>(null);
  const [point, setPoint] = useState<[number, number] | null>(null);
  const [pointSet, setPointSet] = useState(false);
  const [takes, setTakes] = useState<Take[]>([]);
  const [performing, setPerforming] = useState(true);
  const [compare, setCompare] = useState<JobStatus | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

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
    setPointSet(p.point !== null); // a sample comes with its spot; it can still be moved
    setTakes([]);
    setCompare(null);
    setPerforming(true);
  };

  const current = takes.at(-1);
  const done = takes.filter((t) => t.job?.status === "done");

  const runCompare = async () => {
    const [a, b] = done.slice(-2);
    if (!a?.job || !b?.job) return;
    setComparing(true);
    setCompareError(null);
    try {
      setCompare(await compareTakes(a.job.id, b.job.id));
    } catch (e) {
      setCompareError((e as Error).message);
    } finally {
      setComparing(false);
    }
  };

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
            <button className="secondary" onClick={() => setPlate(null)}>Different shot</button>
          </div>
          {!pointSet ? (
            <>
              <p className="muted">Tap the spot the effect should erupt from: the top of the pumpkin, the mug&apos;s rim.</p>
              <TapPoint src={plate.url} value={point} onChange={setPoint} />
              <div className="row"><button disabled={!point} onClick={() => setPointSet(true)}>That&apos;s the spot</button></div>
            </>
          ) : (
            <div className="row"><span className="chip">emission point set</span><button className="secondary" onClick={() => setPointSet(false)}>Move it</button></div>
          )}
        </section>
      )}

      {plate && point && pointSet && performing && (
        <Performer plateUrl={plate.url} point={point} voices={samples.voices} takes={takes.length}
          onDone={(p) => { setTakes((t) => [...t, { id: t.length + 1, performance: p, job: null, effect: null }]); setPerforming(false); }} />
      )}

      {plate && point && pointSet && !performing && current && (
        <ShotPanel key={current.id} plate={plate.source} point={point} performance={current.performance}
          effects={effects} modes={health?.modes ?? []}
          onDone={(job, effect) => {
            const id = current.id; // this take, even if the user has moved on to another before it finishes
            setTakes((t) => t.map((x) => (x.id === id ? { ...x, job, effect } : x)));
          }} />
      )}

      {plate && point && pointSet && !performing && (
        <section>
          <div className="row">
            <button className="secondary" onClick={() => { setPerforming(true); setCompare(null); }}>Perform it differently</button>
            {done.length >= 2 && (
              <button disabled={comparing} onClick={() => void runCompare()}>{comparing ? "Lining them up…" : "Compare my last two takes"}</button>
            )}
          </div>
          {done.length === 1 && <p className="muted">Same shot, same seed: perform it again (shorter, louder, a different noise) and compare the takes.</p>}
          {compareError && <p className="chip bad">Couldn&apos;t line the takes up: {compareError}</p>}
          {compare?.files["compare.mp4"] && (
            <video src={compare.files["compare.mp4"]} controls autoPlay playsInline className="result wide" />
          )}
        </section>
      )}
    </>
  );
}
