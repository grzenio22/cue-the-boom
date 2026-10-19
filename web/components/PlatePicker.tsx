"use client";

import { useState } from "react";
import FilmIt from "@/components/FilmIt";
import type { DrawnSample, SamplePlate, SceneMap, Source } from "@/lib/shot";

export interface PickedPlate {
  source: Source;
  url: string; // something a <video> can play
  title: string;
  point: [number, number] | null; // a sample's suggested emission point
  direction: number | null; // which way its source faces (a mouth), degrees clockwise from up
  subject: string | null; // what the shot shows, for the scene render's prompt
  scene?: SceneMap; // what each sample noise does to it
  sample: boolean;
  sound: boolean; // may carry a sound to lock to (your own clip, or a sample with one)
  drawn?: DrawnSample[]; // a sample's own drawings, rendered ahead of time
  hand?: { frame: number } | null; // a sample hand clip: tapped on this frame, its element made ahead of time
  filmed?: boolean; // filmed here, its noise in it: cued by its own sound
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

  const onFile = async (file: File | undefined, filmed = false, made?: string) => {
    setError(null);
    if (!file) return;
    const url = made ?? URL.createObjectURL(file);
    try {
      const d = await probeDuration(url);
      if (!Number.isFinite(d) || d < MIN_PLATE_S || d > MAX_PLATE_S) {
        URL.revokeObjectURL(url);
        setError(`Use a clip of ${MIN_PLATE_S}-${MAX_PLATE_S} s (this one is ${Number.isFinite(d) ? d.toFixed(1) : "?"} s). The shot uses up to 10 s.`);
        return;
      }
      onPick({ source: { file, filename: file.name || "plate.mp4" }, url, title: filmed ? "your take" : file.name, point: null,
        direction: null, subject: null, sample: false, sound: true, filmed });
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
            <button key={s.name} className="tile" onClick={() => onPick({ source: { sample: s.name }, url: s.url,
              title: s.title ?? s.name, point: s.point ?? null, direction: s.direction ?? null, subject: s.subject ?? null,
              scene: s.scene, sample: true, sound: !!s.sound, drawn: s.drawn, hand: s.hand ?? null })}>
              <video src={s.url} muted playsInline preload="metadata" />
              <span>{s.title ?? s.name}</span>
              {s.note && <span className="muted note">{s.note}</span>}
            </button>
          ))}
        </div>
      )}
      <div className="row">
        <label className="btn secondary">
          Use your own clip
          <input type="file" accept="video/*" hidden onChange={(e) => void onFile(e.target.files?.[0])} />
        </label>
        <FilmIt onTake={(file, url) => void onFile(file, true, url)} />
        <span className="muted">Your hand, or a thing (a pumpkin, a mug, a toy), filmed with the phone held still, 1-15 s (the
          shot uses the first 10 s).</span>
      </div>
      {error && <p className="chip bad">{error}</p>}
    </section>
  );
}
