"use client";

import { useState } from "react";
import type { SamplePlate, Source } from "@/lib/shot";

export interface PickedPlate {
  source: Source;
  url: string; // something a <video> can play
  title: string;
  point: [number, number] | null; // a sample's suggested emission point
}

const MIN_PLATE_S = 1;
const MAX_PLATE_S = 15;

function probeDuration(url: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => resolve(v.duration);
    v.onerror = () => reject(new Error("could not read this video"));
    v.src = url;
  });
}

/** Step 1: a sample plate (no camera needed) or your own clip. */
export default function PlatePicker({ samples, onPick }: { samples: SamplePlate[]; onPick: (p: PickedPlate) => void }) {
  const [error, setError] = useState<string | null>(null);

  const onFile = async (file: File | undefined) => {
    setError(null);
    if (!file) return;
    const url = URL.createObjectURL(file);
    try {
      const d = await probeDuration(url);
      if (!Number.isFinite(d) || d < MIN_PLATE_S || d > MAX_PLATE_S) {
        URL.revokeObjectURL(url);
        setError(`Use a clip of ${MIN_PLATE_S}-${MAX_PLATE_S} s (this one is ${Number.isFinite(d) ? d.toFixed(1) : "?"} s). The shot uses up to 10 s.`);
        return;
      }
      onPick({ source: { file, filename: file.name || "plate.mp4" }, url, title: file.name, point: null });
    } catch {
      URL.revokeObjectURL(url);
      setError("Could not read that video. Try an MP4 or MOV from your phone.");
    }
  };

  return (
    <section>
      <h2>1. Pick a shot</h2>
      {samples.length > 0 && (
        <div className="grid">
          {samples.map((s) => (
            <button key={s.name} className="tile" onClick={() => onPick({ source: { sample: s.name }, url: s.url, title: s.title ?? s.name, point: s.point ?? null })}>
              <video src={s.url} muted playsInline preload="metadata" />
              <span>{s.title ?? s.name}</span>
            </button>
          ))}
        </div>
      )}
      <div className="row">
        <label className="btn secondary">
          Use your own clip
          <input type="file" accept="video/*" hidden onChange={(e) => void onFile(e.target.files?.[0])} />
        </label>
        <span className="muted">An object (a pumpkin, a mug, a toy) filmed with the phone held still, 1-15 s.</span>
      </div>
      {error && <p className="chip bad">{error}</p>}
    </section>
  );
}
