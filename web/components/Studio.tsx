"use client";

import { useEffect, useState } from "react";
import Performer, { type Performance } from "@/components/Performer";
import PlatePicker, { type PickedPlate } from "@/components/PlatePicker";
import ShotPanel from "@/components/ShotPanel";
import TapPoint from "@/components/TapPoint";
import { compareTakes, type EffectInfo, fetchEffects, fetchHealth, fetchSamples, type Health, type JobStatus, type Samples } from "@/lib/shot";

interface Take { performance: Performance; job: JobStatus | null; effect: string | null }

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
    setPointSet(false);
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
    try {
      setCompare(await compareTakes(a.job.id, b.job.id));
    } catch (e) {
      setApiError((e as Error).message);
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
          onDone={(p) => { setTakes((t) => [...t, { performance: p, job: null, effect: null }]); setPerforming(false); }} />
      )}

      {plate && point && pointSet && !performing && current && (
        <ShotPanel key={takes.length} plate={plate.source} point={point} performance={current.performance}
          effects={effects} modes={health?.modes ?? []}
          onDone={(job, effect) => setTakes((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, job, effect } : x)))} />
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
          {compare?.files["compare.mp4"] && (
            <video src={compare.files["compare.mp4"]} controls autoPlay playsInline className="result wide" />
          )}
        </section>
      )}
    </>
  );
}
