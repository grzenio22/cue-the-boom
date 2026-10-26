"use client";

import { useEffect, useRef, useState } from "react";
import { handMade, type JobStatus } from "@/lib/shot";

type Shot = JobStatus & { title: string };

/** What became of a moving clip's own pixels: kept (only what changed added), or re-drawn by the model. */
const pixels = (s: JobStatus) => !s.receipt?.transfer ? ""
  : s.receipt.transfer.accepted ? "; the clip's own pixels, plus what changed" : "; re-drawn by LTX-2.5 (the render moved the scene)";

/** Sample shots made ahead of time: something to watch before trying it. */
export default function Gallery() {
  const [shots, setShots] = useState<Shot[]>([]);
  const [open, setOpen] = useState<{ shot: Shot; file: string } | null>(null);
  const player = useRef<HTMLDivElement>(null);
  const openId = open?.shot.id;
  // the player opens below the grid: on a phone that is off-screen, so bring it into view
  useEffect(() => { if (openId) player.current?.scrollIntoView({ behavior: "smooth", block: "center" }); }, [openId]);

  useEffect(() => {
    fetch("/api/gallery").then((r) => (r.ok ? r.json() : []))
      // a render's further performances play together above, under "One render": the grid shows it once
      .then((all: Shot[]) => setShots(all.filter((s) => !s.again)), () => setShots([]));
  }, []);

  if (shots.length === 0) return null;
  return (
    <section>
      <h2>Made with it</h2>
      <div className="grid">
        {shots.map((s) => (
          <button key={s.id} className="tile" onClick={() => setOpen({ shot: s, file: "final.mp4" })}>
            <video src={s.files["final.mp4"]} poster={s.files["poster.jpg"]} muted playsInline preload="none" />
            <span>{s.title}</span>
            <span className="muted note">{s.mode === "scene"
              ? `rendered into the scene (LTX-2.5)${s.receipt?.inputs?.drawn ? ", drawn" : ""}${s.receipt?.inputs?.cue === "clip" ? ", locked to the clip's sound" : s.receipt?.inputs?.cue === "hold" ? ", press and hold" : ""}${pixels(s)}`
              : s.mode === "bending" ? handMade(s.receipt)
              : s.mode === "render" ? "element rendered from the noise" : "instant (pre-rendered element)"}</span>
          </button>
        ))}
      </div>
      {open && (
        <div ref={player}>
          <div className="row" role="tablist">
            {(["final.mp4", "breakdown.mp4"] as const).filter((f) => open.shot.files[f]).map((f) => (
              <button key={f} role="tab" aria-selected={open.file === f} className={open.file === f ? "" : "secondary"}
                onClick={() => setOpen({ ...open, file: f })}>{f === "final.mp4" ? "Shot" : "How it was made"}</button>
            ))}
            <button className="secondary" onClick={() => setOpen(null)}>Close</button>
          </div>
          <video key={open.shot.id + open.file} src={open.shot.files[open.file]} controls autoPlay playsInline className="result" />
          <p className="muted">{open.shot.title}</p>
          {(open.shot.receipt?.warnings ?? []).map((w) => <span key={w} className="chip warn">{w}</span>)}
        </div>
      )}
    </section>
  );
}
