/** Client for the render API (cuesync serve, proxied at /api). */

export interface EffectInfo {
  id: string;
  fluid: "fire" | "smoke" | "steam";
  behaviour: "burst" | "plume" | "puff" | "jet";
  stops_with_voice: boolean;
}

export interface JobEvent {
  stage: string;
  progress: number;
  message: string;
  t: number;
}

export interface Receipt {
  render: { seconds: number; renderer: string; prompt: string; model: string };
  measured: { onset_error_frames: number | null; stop_latency_frames: number | null };
  voice: { hero_onset_s: number; voice_end_frame: number | null; active_intervals: number[][] };
  guide: { length: number; tail: string; keyframes: number[] };
  seconds_total: number;
}

export interface JobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string;
  error: string | null;
  files: Record<string, string>;
  receipt: Receipt | null;
}

export const EFFECT_LABELS: Record<string, { name: string; cue: string }> = {
  fireball: { name: "Fireball", cue: "pf. · KA-BOOM" },
  flame_plume: { name: "Flame plume", cue: "fwoooOOOM" },
  smoke_puff: { name: "Smoke puff", cue: "pffff" },
  steam_jet: { name: "Steam", cue: "hissss" },
};

/** First guess from the performance itself; the performer can always override it. */
export function suggestEffect(pcm: Float32Array, sampleRate: number, onsetS: number | null): string {
  if (onsetS === null || pcm.length === 0) return "fireball";
  const frame = Math.round(sampleRate * 0.01);
  const start = Math.floor(onsetS * sampleRate);
  let peak = 0;
  const rms: number[] = [];
  for (let i = start; i + frame <= pcm.length; i += frame) {
    let s = 0;
    for (let j = i; j < i + frame; j++) s += pcm[j] * pcm[j];
    const r = Math.sqrt(s / frame);
    rms.push(r);
    if (r > peak) peak = r;
  }
  let active = 0, quiet = 0;
  for (const r of rms) {
    if (r > 0.15 * peak) { active += 1; quiet = 0; } else if (++quiet > 25) break;
  }
  const durationS = active * 0.01;
  // zero-crossing rate over the active part: a hiss crosses zero far more often than a voiced "fwoom"
  let zc = 0;
  const end = Math.min(pcm.length, start + Math.round(durationS * sampleRate));
  for (let i = start + 1; i < end; i++) if ((pcm[i - 1] < 0) !== (pcm[i] < 0)) zc += 1;
  const zcr = zc / Math.max(1, end - start) * sampleRate;
  if (durationS >= 0.45 && zcr > 3000) return "steam_jet";
  if (durationS >= 0.45) return "flame_plume";
  return "fireball";
}

export async function fetchEffects(): Promise<EffectInfo[]> {
  const r = await fetch("/api/effects");
  if (!r.ok) throw new Error(`render API unavailable (${r.status})`);
  return r.json();
}

export async function submitShot(args: {
  plate: File;
  voice: Blob;
  face: Blob | null;
  faceExt: string;
  timing: object | null;
  effect: string;
  point: [number, number];
  seed: number;
}): Promise<JobStatus> {
  const form = new FormData();
  form.append("plate", args.plate, args.plate.name || "plate.mp4");
  form.append("voice", args.voice, "mic.wav");
  if (args.face) form.append("face", args.face, `face.${args.faceExt}`);
  if (args.timing) form.append("timing", new Blob([JSON.stringify(args.timing)], { type: "application/json" }), "timing.json");
  form.append("effect", args.effect);
  form.append("x", String(args.point[0]));
  form.append("y", String(args.point[1]));
  form.append("seed", String(args.seed));
  const r = await fetch("/api/jobs", { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.text()) || `upload failed (${r.status})`);
  return r.json();
}

/** Follow a job until it ends; returns the final status. */
export function followJob(id: string, onEvent: (e: JobEvent) => void): Promise<JobStatus> {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${id}/events`);
    es.onmessage = (m) => onEvent(JSON.parse(m.data) as JobEvent);
    es.addEventListener("end", (m) => {
      es.close();
      resolve(JSON.parse((m as MessageEvent).data) as JobStatus);
    });
    es.onerror = () => {
      es.close();
      fetch(`/api/jobs/${id}`).then((r) => r.json()).then(resolve, reject);
    };
  });
}
