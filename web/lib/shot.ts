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
  render: { seconds: number; renderer: string; prompt: string; model: string; library?: { entry: string; performance: string; scale: number } };
  measured: { onset_error_frames: number | null; hit_error_frames: number | null; stop_latency_frames: number | null; stops_with_voice: boolean };
  voice: { hero_onset_s: number; voice_end_frame: number | null; active_intervals: number[][] };
  guide: { length: number; tail: string; keyframes: number[] };
  matte?: { source: string; found: boolean };
  seconds_total: number;
}

export type Mode = "instant" | "render";

export interface JobStatus {
  id: string;
  mode?: Mode | "compare";
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string;
  error: string | null;
  files: Record<string, string>;
  receipt: Receipt | null;
}

export interface Health {
  ok: boolean;
  renderer: string | null;
  modes: Mode[];
  queued: number;
}

export interface SamplePlate { name: string; url: string; title?: string; point?: [number, number]; note?: string }
export interface SampleVoice { name: string; url: string; title?: string; effect?: string }
export interface Samples { plates: SamplePlate[]; voices: SampleVoice[] }

/** A plate or a voice: something the browser has (a file or a recording), or a sample the server has. */
export type Source = { file: Blob; filename: string } | { sample: string };

export const EFFECT_LABELS: Record<string, { name: string; cue: string }> = {
  fireball: { name: "Fireball", cue: "pf. · KA-BOOM" },
  flame_plume: { name: "Flame plume", cue: "fwoooOOOM" },
  smoke_puff: { name: "Smoke puff", cue: "pfffff" },
  steam_jet: { name: "Steam", cue: "hissss" },
};

/** Duration of the performance (s, above 15 % of its peak) and its zero-crossing rate (crossings/s). */
export function describeNoise(pcm: Float32Array, sampleRate: number, onsetS: number): { durationS: number; zcr: number } {
  const frame = Math.max(1, Math.round(sampleRate * 0.01));
  const start = Math.max(0, Math.floor(onsetS * sampleRate));
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
  let zc = 0;
  const end = Math.min(pcm.length, start + Math.round(durationS * sampleRate));
  for (let i = start + 1; i < end; i++) if ((pcm[i - 1] < 0) !== (pcm[i] < 0)) zc += 1;
  return { durationS, zcr: (zc / Math.max(1, end - start)) * sampleRate };
}

/**
 * First guess from the performance itself (the performer can always override it): a short hit is a
 * fireball, a breathy "pfffff" a smoke puff, a long hiss steam, a long voiced "fwoooom" a flame plume.
 * A hiss or a breath crosses zero far more often than a voiced sound.
 */
export function suggestEffect(pcm: Float32Array, sampleRate: number, onsetS: number | null): string {
  if (onsetS === null || pcm.length === 0) return "fireball";
  const { durationS, zcr } = describeNoise(pcm, sampleRate, onsetS);
  const noisy = zcr > 2500;
  if (durationS < 0.3) return "fireball";
  if (noisy) return durationS >= 0.8 ? "steam_jet" : "smoke_puff";
  return durationS >= 0.45 ? "flame_plume" : "fireball";
}

const json = async <T,>(r: Response): Promise<T> => {
  if (!r.ok) throw new Error((await r.text()) || `request failed (${r.status})`);
  return r.json() as Promise<T>;
};

export const fetchEffects = () => fetch("/api/effects").then((r) => json<EffectInfo[]>(r));
export const fetchHealth = () => fetch("/api/health").then((r) => json<Health>(r));
export const fetchSamples = () => fetch("/api/samples").then((r) => json<Samples>(r));

export async function submitShot(args: {
  plate: Source;
  voice: Source;
  face: Blob | null;
  faceExt: string;
  timing: object | null;
  effect: string;
  point: [number, number];
  seed: number;
  mode: Mode;
}): Promise<JobStatus> {
  const form = new FormData();
  if ("sample" in args.plate) form.append("plate_sample", args.plate.sample);
  else form.append("plate", args.plate.file, args.plate.filename);
  if ("sample" in args.voice) form.append("voice_sample", args.voice.sample);
  else form.append("voice", args.voice.file, args.voice.filename);
  if (args.face) form.append("face", args.face, `face.${args.faceExt}`);
  if (args.timing) form.append("timing", new Blob([JSON.stringify(args.timing)], { type: "application/json" }), "timing.json");
  form.append("effect", args.effect);
  form.append("x", String(args.point[0]));
  form.append("y", String(args.point[1]));
  form.append("seed", String(args.seed));
  form.append("mode", args.mode);
  return json<JobStatus>(await fetch("/api/jobs", { method: "POST", body: form }));
}

export async function compareTakes(a: string, b: string): Promise<JobStatus> {
  const form = new FormData();
  form.append("a", a);
  form.append("b", b);
  return json<JobStatus>(await fetch("/api/compare", { method: "POST", body: form }));
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
