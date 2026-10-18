/** Client for the render API (cuesync serve, proxied at /api). */

export interface EffectInfo {
  id: string;
  fluid: "fire" | "smoke" | "steam";
  behaviour: "burst" | "plume" | "puff" | "jet" | "extinguish";
  stops_with_voice: boolean;
  scene?: boolean; // can be rendered into the scene
  direction?: number | null; // its default direction in the scene, degrees clockwise from up
  element?: boolean; // can be rendered as an element (false: scene mode only)
}

export interface JobEvent {
  stage: string;
  progress: number;
  message: string;
  t: number;
}

/** The shot frame of the hit: a scene shot's render starts well before it, so its receipt says where. */
export function hitAnchor(r: Receipt | null | undefined): number {
  if (!r) return 0;
  return r.mode === "scene" ? (r.alignment?.hit_frame ?? r.guide.anchor_frame) : r.guide.anchor_frame;
}

export interface Receipt {
  mode?: "scene";
  alignment?: { model_onset?: number | null; auto_shift?: number; hit_frame?: number; retime_frames?: number };
  render: { seconds: number; renderer: string; prompt: string; model: string; library?: { entry: string; performance: string; scale: number } };
  warnings?: string[];
  inputs?: { cue?: CueKind; drawn?: boolean; describe?: string | null; sources?: [number, number][] | null };
  // hit_error_frames: an element shot's hit, measured on the shot; residual_frames: what a scene shot's re-time
  // could not take out (older scene receipts call it hit_error_frames)
  measured: { onset_error_frames?: number | null; hit_error_frames?: number | null; residual_frames?: number | null; stop_latency_frames: number | null; stops_with_voice: boolean; model_early_frames?: number | null };
  voice: { hero_onset_s: number; voice_end_frame: number | null; active_intervals: number[][] };
  guide: { length: number; tail: string; keyframes: number[]; anchor_frame: number };
  matte?: { source: string; found: boolean };
  plate?: { still: boolean; point?: [number, number] }; // a still plate becomes the render; a moving one keeps its pixels if the check passes
  transfer?: { accepted: boolean; reason?: string | null } | null;
  hits?: { frame: number; strength: number; size: number; merged?: boolean }[] | null;
  retimed?: { hit_shift_frames: number; stop_frame: number | null } | null;
  replayed?: { from: string; render_hits: number; new_hits: number; no_new_render: boolean; renders?: number } | null;
  seconds_total: number;
}

export type Mode = "instant" | "render" | "scene";

/** A drawn stroke: points in normalised video coordinates, brush width as a share of the frame width. */
export interface Stroke { points: [number, number][]; width: number }

/** What times the shot: a performance, a sound already in the clip, or press-and-hold. */
export type CueKind = "voice" | "clip" | "hold";

export interface JobStatus {
  id: string;
  effect?: string | null; // the effect actually used (the noise may have decided it)
  gesture?: string | null; // how the cue was read: hit, rhythm, sustained, breath, hiss
  heard?: string | null; // effect "auto" on a plate with no map: what the sound is (boom, whoosh, hiss, pop, puff, crackle)
  mode?: Mode | "compare";
  cached?: boolean;
  title?: string;
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
  made_ahead?: Mode[]; // modes with sample shots made ahead of time (they play with the GPU offline)
  queued: number;
}

/** What each sample noise does to a sample plate in scene mode (an effect id, or {effect, seed}). */
export type SceneMap = Record<string, string | { effect: string }>;
/** A sample's own drawing, rendered into the scene ahead of time with one of the sample noises. */
export interface DrawnSample { title: string; voice: string; effect: string; strokes: Stroke[]; describe?: string }
export interface SamplePlate {
  name: string; url: string; title?: string; point?: [number, number]; note?: string;
  subject?: string; direction?: number; scene?: SceneMap;
  sound?: boolean; // the sample carries a sound of its own to lock to (a snap)
  drawn?: DrawnSample[];
}
/** A kind of thing your own clip can show: what each gesture does to it in scene mode. */
export interface ObjectKind { id: string; title: string; subject: string | null; scene: SceneMap }

export const fetchObjects = async (): Promise<ObjectKind[]> => {
  const r = await fetch("/api/objects");
  return r.ok ? r.json() : [];
};

export interface SampleVoice { name: string; url: string; title?: string; effect?: string; gesture?: string }

/** What a sample noise does to a sample plate in scene mode: its own entry in the plate's map, else its gesture's. */
export function sceneEffect(scene: SceneMap | undefined, v: SampleVoice): string | undefined {
  const e = scene?.[v.name] ?? (v.gesture ? scene?.[v.gesture] : undefined);
  return typeof e === "string" ? e : e?.effect;
}

/** Is drawing `b` the drawing `a`, to within half a percent of the frame (what the server accepts as the same)? */
export function sameStrokes(a: Stroke[], b: Stroke[], tol = 0.005): boolean {
  return a.length === b.length && a.every((s, i) => Math.abs(s.width - b[i].width) < tol && s.points.length === b[i].points.length
    && s.points.every(([x, y], j) => Math.abs(x - b[i].points[j][0]) < tol && Math.abs(y - b[i].points[j][1]) < tol));
}

/** The sample noises a sample's own drawing was rendered with ahead of time (none for any other drawing). */
export function drawnReady(plate: { sample: boolean; drawn?: DrawnSample[] }, strokes: Stroke[]): string[] {
  if (!plate.sample || strokes.length === 0) return [];
  return (plate.drawn ?? []).filter((d) => sameStrokes(d.strokes, strokes)).map((d) => d.voice);
}

/** Was this exact sample shot made into the scene ahead of time (so it plays with the GPU offline)? A sample clip
 * at its marked spot, cued by a sample noise its map lists by name, or by the clip's own sound; or the sample's own
 * drawing with the noise it was rendered with. */
/** Is this the sample's own marked spot, aimed its own way (where its scene shots were made ahead of time)? The
 * server keys a shot by its point and, once aimed, its direction: an aim of your own is another shot. */
export function atSampleSpot(sample: [number, number] | null | undefined, point: [number, number] | null,
  sampleDir: number | null = null, dir: number | null = null): boolean {
  const sameAim = dir === null || (sampleDir !== null && Math.round(dir) === Math.round(sampleDir));
  return sameAim && !!sample && !!point && Math.abs(sample[0] - point[0]) < 0.005 && Math.abs(sample[1] - point[1]) < 0.005;
}

export function madeAhead(plate: { sample: boolean; point: [number, number] | null; direction?: number | null; scene?: SceneMap; sound: boolean; drawn?: DrawnSample[] },
  cue: CueKind, voice: Source | null, point: [number, number], strokes: Stroke[], direction: number | null = null): boolean {
  if (strokes.length) return cue === "voice" && voice !== null && "sample" in voice && drawnReady(plate, strokes).includes(voice.sample);
  if (!plate.sample || !atSampleSpot(plate.point, point, plate.direction ?? null, direction)) return false;
  if (cue === "clip") return plate.sound;
  return cue === "voice" && voice !== null && "sample" in voice && !!plate.scene && voice.sample in plate.scene;
}

/** With the GPU offline, can your own take (your voice, or press-and-hold) play a sample's scene shot made ahead of
 * time? A sample clip at its marked spot, with a scene shot made for it: each of your hits plays the render's. */
export function replayable(plate: { sample: boolean; point: [number, number] | null; direction?: number | null; scene?: SceneMap },
  cue: CueKind, voice: Source | null, point: [number, number], strokes: Stroke[], direction: number | null = null): boolean {
  if (strokes.length || !plate.sample || !plate.scene || Object.keys(plate.scene).length === 0) return false;
  if (!atSampleSpot(plate.point, point, plate.direction ?? null, direction)) return false;
  return cue === "hold" || (cue === "voice" && voice !== null && !("sample" in voice));
}
export interface Samples { plates: SamplePlate[]; voices: SampleVoice[] }

/** A plate or a voice: something the browser has (a file or a recording), or a sample the server has. */
export type Source = { file: Blob; filename: string } | { sample: string };

export const EFFECT_LABELS: Record<string, { name: string; cue: string }> = {
  fireball: { name: "Fireball", cue: "pf. · KA-BOOM" },
  flame_plume: { name: "Flame plume", cue: "fwoooOOOM" },
  smoke_puff: { name: "Smoke puff", cue: "pfffff" },
  steam_jet: { name: "Steam", cue: "hissss" },
  fire_breath: { name: "Fire breath", cue: "fwoooOOOM" },
  wick_flare: { name: "Flare-up", cue: "fwoooOOOM" },
  blow_out: { name: "Blow it out", cue: "pf. · pfffff" },
  fire_burst: { name: "Fire burst", cue: "pf. · KA-BOOM" },
};

export const AUTO_EFFECT = "auto";

/** Degrees clockwise from straight up for a drag (dx right, dy down), rounded to 5. */
export function dragDirection(dx: number, dy: number): number {
  return Math.round((Math.atan2(dx, -dy) * 180) / Math.PI / 5) * 5;
}

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

/** The server's own words for a failed request: its JSON `detail`, else the text, else the status. */
export async function failure(r: Response): Promise<Error> {
  const text = await r.text().catch(() => "");
  try {
    const detail = (JSON.parse(text) as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail) return new Error(detail);
  } catch { /* not JSON */ }
  return new Error(text || `request failed (${r.status})`);
}

const json = async <T,>(r: Response): Promise<T> => {
  if (!r.ok) throw await failure(r);
  return r.json() as Promise<T>;
};

export const fetchEffects = () => fetch("/api/effects").then((r) => json<EffectInfo[]>(r));
export const fetchHealth = () => fetch("/api/health").then((r) => json<Health>(r));
export const fetchSamples = () => fetch("/api/samples").then((r) => json<Samples>(r));

export async function submitShot(args: {
  plate: Source;
  voice: Source | null;
  face: Blob | null;
  faceExt: string;
  timing: object | null;
  effect: string;
  point: [number, number];
  seed: number;
  mode: Mode;
  direction?: number | null;
  subject?: string | null;
  object?: string | null; // your own clip: what it is (a candle, a jack-o'-lantern, ...)
  cue?: CueKind;
  holds?: [number, number][] | null;
  strokes?: Stroke[] | null;
  describe?: string | null;
  sources?: [number, number][] | null; // call and response: one source per hit, in order (the first is the point)
}): Promise<JobStatus> {
  const form = new FormData();
  if ("sample" in args.plate) form.append("plate_sample", args.plate.sample);
  else form.append("plate", args.plate.file, args.plate.filename);
  if (args.voice && "sample" in args.voice) form.append("voice_sample", args.voice.sample);
  else if (args.voice) form.append("voice", args.voice.file, args.voice.filename);
  form.append("cue", args.cue ?? "voice");
  if (args.holds && args.holds.length) form.append("holds", JSON.stringify(args.holds));
  if (args.mode === "scene" && args.strokes && args.strokes.length) form.append("strokes", JSON.stringify(args.strokes));
  if (args.mode === "scene" && args.describe) form.append("describe", args.describe);
  if (args.mode === "scene" && args.sources && args.sources.length >= 2) form.append("sources", JSON.stringify(args.sources));
  if (args.face) form.append("face", args.face, `face.${args.faceExt}`);
  if (args.timing) form.append("timing", new Blob([JSON.stringify(args.timing)], { type: "application/json" }), "timing.json");
  form.append("effect", args.effect);
  form.append("x", String(args.point[0]));
  form.append("y", String(args.point[1]));
  form.append("seed", String(args.seed));
  form.append("mode", args.mode);
  if (args.mode === "scene" && args.direction != null) form.append("direction", String(args.direction));
  if (args.mode === "scene" && args.subject) form.append("subject", args.subject);
  if (args.object && !("sample" in args.plate)) form.append("object", args.object);
  return json<JobStatus>(await fetch("/api/jobs", { method: "POST", body: form }));
}

/** Perform a finished scene shot again with a new take: each new hit plays one of the render's, no new render.
 * ``together``: other finished shots on the same plate, played as one scene (one render per source). */
export async function replayShot(jobId: string, args: { voice: Source | null; cue?: CueKind; holds?: [number, number][] | null;
  timing: object | null; together?: string[] }): Promise<JobStatus> {
  const form = new FormData();
  if (args.together && args.together.length) form.append("together", JSON.stringify(args.together));
  if (args.voice && "sample" in args.voice) form.append("voice_sample", args.voice.sample);
  else if (args.voice) form.append("voice", args.voice.file, args.voice.filename);
  form.append("cue", args.cue ?? "voice");
  if (args.holds && args.holds.length) form.append("holds", JSON.stringify(args.holds));
  if (args.timing) form.append("timing", new Blob([JSON.stringify(args.timing)], { type: "application/json" }), "timing.json");
  return json<JobStatus>(await fetch(`/api/jobs/${jobId}/replay`, { method: "POST", body: form }));
}

export async function compareTakes(a: string, b: string): Promise<JobStatus> {
  const form = new FormData();
  form.append("a", a);
  form.append("b", b);
  return json<JobStatus>(await fetch("/api/compare", { method: "POST", body: form }));
}

/** Ask for a job's status until it ends (about ten minutes at most). */
export async function pollUntilDone(id: string, wait = 1500): Promise<JobStatus> {
  for (let i = 0; i < 400; i++) {
    const r = await fetch(`/api/jobs/${id}`);
    if (r.status === 404) throw new Error("the server lost this job (it may have restarted); please make it again");
    const s: JobStatus = await r.json();
    if (s.status === "done" || s.status === "error") return s;
    await new Promise((res) => setTimeout(res, wait));
  }
  throw new Error("this is taking too long; please try again");
}

/** Follow a job until it ends; returns the final status. If the progress stream drops, it keeps asking. */
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
      pollUntilDone(id).then(resolve, reject);
    };
  });
}
