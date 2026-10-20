"use client";

import { useEffect, useState } from "react";
import { effectStart, type JobStatus } from "@/lib/shot";

type Shot = JobStatus & { title: string };

/** The first thing on the page: the best shots made with it, playing (muted until tapped), each with its cue's
 * ribbon under the picture, so the sound-to-effect link reads even with the sound off. Each loops from a beat before
 * its first hit, its poster the effect at its fullest (from the clip's start they were three bare hands). */
export default function Hook() {
  const [shots, setShots] = useState<Shot[]>([]);
  const [heard, setHeard] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/gallery").then((r) => (r.ok ? r.json() : [])).then((s: Shot[]) => setShots(s.slice(0, 3)), () => setShots([]));
  }, []);

  if (shots.length === 0) return null;
  return (
    <section className="hook" aria-label="Shots made with it">
      <div className="hook-row">
        {shots.map((s) => (
          <figure key={s.id}>
            <video src={s.files["final.mp4"]} poster={s.files["poster.jpg"]} autoPlay muted={heard !== s.id} playsInline
              preload="auto" onLoadedMetadata={(e) => { e.currentTarget.currentTime = effectStart(s.receipt); }}
              onEnded={(e) => { e.currentTarget.currentTime = effectStart(s.receipt); void e.currentTarget.play(); }}
              aria-label={`${s.title}: tap to ${heard === s.id ? "mute" : "hear the cue"}`}
              onClick={() => setHeard(heard === s.id ? null : s.id)} />
            <figcaption>{s.title}</figcaption>
          </figure>
        ))}
      </div>
      <p className="muted">Real clips, each cued by a sound (the ribbon under it). Tap one to hear it.</p>
    </section>
  );
}
