/** The live preview's arithmetic: the hand from its landmarks, the noise from the microphone's level, a hit's look. */

/** A hand element rendered ahead of time (GET /api/library): its pose in its render, its frames. */
export interface LiveElement {
  effect: string; behaviour: string; url: string; size: [number, number]; fps: number;
  ignition: number; release: number; end: number; floor: number;
  ref: { xy: [number, number]; dir: [number, number]; scale: number };
}

type P = { x: number; y: number };

/** The hand from MediaPipe's 21 landmarks (px): its palm, the index fingertip a little past the tip, the way the
 * finger points, its size (wrist to the middle knuckle, as the shot's tracker measures it), and whether it points. */
export function liveHand(lm: P[]) {
  const d = (a: P, b: P) => Math.hypot(a.x - b.x, a.y - b.y);
  const palmIdx = [0, 5, 9, 13, 17];
  const palm: [number, number] = [palmIdx.reduce((s, i) => s + lm[i].x, 0) / 5, palmIdx.reduce((s, i) => s + lm[i].y, 0) / 5];
  const fx = lm[8].x - lm[5].x, fy = lm[8].y - lm[5].y, fl = Math.hypot(fx, fy) || 1;
  const tip: [number, number] = [lm[8].x + 0.3 * (lm[8].x - lm[7].x), lm[8].y + 0.3 * (lm[8].y - lm[7].y)];
  const pointing = d(lm[8], lm[0]) > 1.25 * Math.max(d(lm[12], lm[0]), d(lm[16], lm[0]));
  return { palm, tip, dir: [fx / fl, fy / fl] as [number, number], scale: Math.max(d(lm[0], lm[9]), 1), pointing };
}

/** The microphone's state between frames: the room's level (floor), the loudest hit so far (top), the level a
 * frame ago, when the current hit's sound began (since), the last hit's time, and since when a sound has stayed well
 * over the room (loudSince: a held sound, however gently it began). dB are dBFS of the frame's RMS. */
export interface OnsetState {
  floor: number; top: number; prev: number; since: number | null; lastHit: number; peak: number; loudSince?: number | null;
}

export const HIT_OVER_FLOOR = 15; // a hit is this many dB over the room...
export const HIT_RISE = 6; // ...rising this much from a frame before...
export const HIT_GAP_S = 0.2; // ...and this long after the last one
export const HOLD_S = 0.35; // a sound still going this long after it began is held (a flame, not a burst)

/** One frame of the microphone: a hit (a fast rise well over the room), and whether a sound is being held. */
export function onsetStep(s: OnsetState, db: number, t: number): { state: OnsetState; hit: { db: number } | null; holding: boolean } {
  const floor = db < s.floor ? db : s.floor + (db - s.floor) * 0.002; // quick down, slow up: the quiet between noises
  let { since, lastHit, top, peak } = s;
  let hit: { db: number } | null = null;
  if (db > floor + HIT_OVER_FLOOR && db > -50 && db - s.prev > HIT_RISE && t - lastHit > HIT_GAP_S) {
    hit = { db };
    lastHit = t;
    since = since ?? t;
    peak = db;
    top = Math.max(top, db);
  }
  if (since !== null && db < floor + 8) since = null; // the sound has stopped
  // held: a sound staying well over the room, measured on its own (a gently started "ffff" has no sharp onset)
  let loudSince = s.loudSince ?? null;
  if (db > floor + 12 && db > -55) loudSince = loudSince ?? t;
  else if (db < floor + 8) loudSince = null;
  top = Math.max(top - 0.002, -40); // the loudest so far, forgotten slowly (about 7 dB a minute at 60 frames a second)
  if (since !== null) {
    peak = Math.max(peak, db);
    top = Math.max(top, peak);
  }
  return { state: { floor, top, prev: db, since, lastHit, peak, loudSince }, hit,
    holding: loudSince !== null && t - loudSince >= HOLD_S && db > floor + 10 };
}

/** A hit's look against the loudest so far, as the shot plays them (cuesync.bending.hit_looks): 12 dB under it is
 * half the size. */
export function playLook(db: number, top: number): { gain: number; size: number } {
  const under = Math.max(0, top - db);
  return { gain: Math.min(1, Math.max(0.35, 10 ** (-under / 60))), size: Math.min(1, Math.max(0.45, 10 ** (-under / 40))) };
}
