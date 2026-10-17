"use client";

import { useEffect, useState } from "react";
import type { JobStatus } from "@/lib/shot";

type Shot = JobStatus & { title: string };

/** Sample shots made ahead of time: something to watch before trying it. */
export default function Gallery() {
  const [shots, setShots] = useState<Shot[]>([]);
  const [open, setOpen] = useState<{ shot: Shot; file: string } | null>(null);

  useEffect(() => {
    fetch("/api/gallery").then((r) => (r.ok ? r.json() : [])).then(setShots, () => setShots([]));
  }, []);

  if (shots.length === 0) return null;
  return (
    <section>
      <h2>Made with it</h2>
      <div className="grid">
        {shots.map((s) => (
          <button key={s.id} className="tile" onClick={() => setOpen({ shot: s, file: "final.mp4" })}>
            <video src={s.files["final.mp4"]} muted playsInline preload="metadata" />
            <span>{s.title}</span>
            <span className="muted note">{s.mode === "render" ? "rendered from the noise" : "instant (pre-rendered element)"}</span>
          </button>
        ))}
      </div>
      {open && (
        <div>
          <div className="row" role="tablist">
            {(["final.mp4", "breakdown.mp4"] as const).filter((f) => open.shot.files[f]).map((f) => (
              <button key={f} role="tab" aria-selected={open.file === f} className={open.file === f ? "" : "secondary"}
                onClick={() => setOpen({ ...open, file: f })}>{f === "final.mp4" ? "Shot" : "How it was made"}</button>
            ))}
            <button className="secondary" onClick={() => setOpen(null)}>Close</button>
          </div>
          <video key={open.shot.id + open.file} src={open.shot.files[open.file]} controls autoPlay playsInline className="result" />
          <p className="muted">{open.shot.title}</p>
        </div>
      )}
    </section>
  );
}
