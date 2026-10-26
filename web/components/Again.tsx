"use client";

import { useEffect, useRef, useState } from "react";
import type { JobStatus } from "@/lib/shot";

const CUES = ["BOOM", "pf, pf, BOOM", "pf pf pf PF"];
const DRIFT_S = 0.15; // a performance this far off the first one's clock is put back in step

function playTogether(players: HTMLVideoElement[]) {
  players.forEach((v) => { v.muted = true; v.loop = true; v.currentTime = 0; void v.play().catch(() => undefined); });
}

/** Three performances of the same render side by side: all three muted and looping in step, or heard one by one.
 * (Every pause, seek and wait was once coordinated between them, and a seek's own "waiting" restarted the wait: the
 * three sat on their first frame. Now each plays on its own and is only nudged back in step.) */
export default function Again() {
  const [shots, setShots] = useState<JobStatus[] | null>(null);
  const [strangers, setStrangers] = useState<JobStatus[]>([]); // the same render in other people's voices
  const [playing, setPlaying] = useState(false);
  const [listening, setListening] = useState<number | null>(null);
  const players = useRef<(HTMLVideoElement | null)[]>([]);

  useEffect(() => {
    let active = true;
    fetch("/api/gallery")
      .then((r) => (r.ok ? r.json() : []))
      .then((payload: JobStatus[]) => {
        if (!active || !Array.isArray(payload)) return;
        const ordered = [0, 1, 2].map((i) => payload.find((s) => s && s.again === i && s.files?.["final.mp4"]));
        setShots(ordered.every(Boolean) ? (ordered as JobStatus[]) : []);
        setStrangers(payload.filter((s) => s && (s.again ?? 0) >= 3 && s.files?.["final.mp4"])
          .sort((a, b) => (a.again ?? 0) - (b.again ?? 0)));
        setPlaying(ordered.every(Boolean) && !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
      })
      .catch(() => { if (active) setShots([]); });
    return () => { active = false; };
  }, []);

  const media = () => players.current.filter((v): v is HTMLVideoElement => v !== null);

  const together = () => { // all three, muted, from the start, in step
    setListening(null);
    playTogether(media());
    setPlaying(true);
  };

  useEffect(() => { // start them together once they are on the page, unless the viewer asked for less motion
    if (!shots || shots.length !== 3) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    playTogether(players.current.filter((v): v is HTMLVideoElement => v !== null));
  }, [shots]);

  useEffect(() => { // keep the three on one clock while they play together
    if (!playing || listening !== null) return;
    const id = setInterval(() => {
      const [lead, ...rest] = media();
      if (!lead) return;
      rest.forEach((v) => { if (Math.abs(v.currentTime - lead.currentTime) > DRIFT_S) v.currentTime = lead.currentTime; });
    }, 400);
    return () => clearInterval(id);
  }, [playing, listening]);

  const pause = () => { media().forEach((v) => v.pause()); setListening(null); setPlaying(false); };

  const hear = (i: number) => { // one performance with its sound, the others still
    setListening(i);
    media().forEach((v, j) => {
      v.loop = false;
      v.muted = j !== i;
      if (j === i) { v.currentTime = 0; void v.play().catch(() => undefined); } else v.pause();
    });
    setPlaying(true);
  };

  const ended = (i: number) => {
    if (listening !== i) return;
    if (i < 2) hear(i + 1);
    else together();
  };

  return (
    <section className="again" aria-labelledby="again-heading">
      <h2 id="again-heading">One render. Three performances.</h2>
      <p>The same LTX-2.5 render on the author&apos;s hand, timed by three performances using their recorded voice: one generation, three performances, no new render, no GPU. The two rhythms are edited from their recorded <em>pf</em> and <em>BOOM</em> sounds.</p>
      <p className="again-fact"><strong>1 generation, {3 + strangers.length} performances, 0 new renders</strong></p>
      {shots === null ? <p className="muted" role="status">Loading the performances…</p>
        : shots.length !== 3 ? <p className="muted" role="status">The three performances are not available here.</p>
        : <>
          <div className="again-row">
            {shots.map((shot, i) => (
              <figure key={shot.id} className={listening === i ? "again-current" : undefined}>
                <video ref={(v) => { players.current[i] = v; }} src={shot.files["final.mp4"]} poster={shot.files["poster.jpg"]}
                  muted playsInline loop preload="auto" aria-label={`Performance ${i + 1}: ${CUES[i]}`}
                  onEnded={() => ended(i)} />
                <figcaption>{CUES[i]}</figcaption>
              </figure>
            ))}
          </div>
          <div className="row">
            <button type="button" onClick={() => (playing ? pause() : together())}>{playing ? "Pause" : "Play together"}</button>
            <button type="button" className="secondary" onClick={() => (listening === null ? hear(0) : together())}>
              {listening === null ? "Hear them one by one" : "Back to all three"}
            </button>
          </div>
          <p className="muted again-status" role="status">
            {listening !== null ? `Hearing performance ${listening + 1} of 3: ${CUES[listening]}` : playing ? "Playing together, muted." : "Paused."}
          </p>
          {strangers.length > 0 && <>
            <h3 id="strangers-heading">The same render, in {strangers.length} strangers&apos; voices</h3>
            <p>Each is one sound cut from another person&apos;s own recording (CC0, on Freesound), placed on the hand&apos;s strike. A soft <em>powf</em> throws a small fireball, a loud <em>boom</em> a big one, a <em>ka-boom</em> two. Still one generation.</p>
            <div className="again-row again-strangers" aria-labelledby="strangers-heading">
              {strangers.map((shot) => (
                <figure key={shot.id}>
                  <video src={shot.files["final.mp4"]} poster={shot.files["poster.jpg"]} controls playsInline preload="none"
                    aria-label={shot.title ?? "A stranger's performance"} />
                  <figcaption>{shot.title}</figcaption>
                </figure>
              ))}
            </div>
          </>}
        </>}
      <p className="muted"><a href="#studio">Make your own performance</a> with the prepared palm and your microphone.</p>
    </section>
  );
}
