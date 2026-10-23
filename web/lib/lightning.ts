/** Live lightning uses the camera's edges as strike surfaces and a tracked fingertip or explicit palm source.
 * Uneven cores, faint branches and surface-shaped contact light keep the discharge attached to the picture.
 * The palm holds a translucent irregular core threaded with short filaments at several depths; tracked fingers
 * hide the rear filaments. Its nearby light follows the palm and the inner faces of the fingers. */

export type P2 = [number, number];
type P3 = [number, number, number];

/** A small seeded random generator (mulberry32): the same seed, the same channels. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- the arcs (cuesync.lightning's constants, by the same names)
const ARC_REACH = [0.15, 0.6]; // the canvas's diagonals: a main channel's reach where it strikes nothing, quietest
// voice to loudest (in hand sizes, a near hand's bolt ran off the frame where the shot's struck the tub)
const ARC_WIDTH = [2.2, 4.8]; // px at 1080 for a hand HAND_PX across: its width, quietest to loudest
const HAND_PX = 220, WIDTH_K = [0.6, 1.2]; // (widths scale with the hand, within these: a root of 3-5 px)
const ARC_LEAD = 0.2; // of its reach: a main channel's first stretch runs along the finger's aim
const ARC_JITTER = 0.07, ARC_OFF = 0.21; // radians: a new main channel's own turn off its aim, and the most it may be
const FORK_INTEN = [0.3, 0.5], FORK_WIDTH = 0.38, FORK_TURN = [0.52, 1.05]; // a fork against what it leaves
const FORK_LENGTH = [40, 125], MAIN_TAPER = [0.22, 0.7]; // px at 1080; main width at its far end and taper exponent
const REDRAW = 4; // a branch is struck again this many times to keep clear of the others, then dropped
const LIFE = [1, 2, 3], LIFE_P = [0.25, 0.45, 0.3]; // frames a path is kept, and how often each
const RESTROKE = [0.45, 0.5]; // a re-stroke flashes with this chance plus this times the level, else it is dark
export const TAPER = [0.95, 1.2]; // A branch reaches five percent current at its tip, so it disappears into the air.
export const BRANCH_W = [1, 0.08, 0.08]; // Root width, tip width share and subpixel floor preserve fine dim streamers.
const CHUNKS = 12; // Equal-distance stretches retain jagged geometry while resolving a few hot runs and quiet necks.

// ---- where it strikes (cuesync.lightning's TARGET_*, EDGE_*, IMPACT, POOL, SPLASH, BRUSH)
export const TARGET_CONE = [25, 40, 70]; // degrees either side of the finger's aim: a burst's first strike an edge
// within the first, else in the second; struck again, anywhere within the third, never where it struck last
export const TARGET_AT = [0.22, 0.4]; // the canvas's diagonals: how far out it strikes, quietest voice to loudest,
// preferring what is nearest that
const TARGET_BAND = [0.5, 1.5]; // shares of that: how far out what it strikes may be
export const TARGET_BORDER = 0.12; // Normal-cone receivers stay this share of the short side inside the picture.
const WIDE_RECEIVER = [120, 0.05]; // A visible side surface is preferable to a featureless patch, even well off the finger's aim.
export const TARGET_TOP = 3, TARGET_APART = 150; // one of the best this many, px at 1080 apart and from where it
// struck last: a burst's strikes go back and forth between two places at least
export const PRONG = [0.25, 90, 240, 0.72, 0.85, 0.8]; // (chance; px at 1080 apart, its range; share along, its
// range; against its channel): one main channel in four that strikes an edge ends in two prongs, landing on two points
const EDGE_Q = 8; // the picture is read at 1 / EDGE_Q of the canvas
const EDGE_MIN = 40; // the gradient (Sobel, code values, at that size, softened) a line must reach to be struck: a step
// of about 15 levels (a blank wall's grain and the camera's noise stay under it)
const EDGE_NEAR = 3; // ...and at least this many more edge px round it (in its 5 x 5): a line, not a speck
export const IMPACT_MIN = 0.15; // a stroke at least this bright strikes with a contact flash: all but one gone dark
// (only bright ones had one, and a dim re-stroke ended in a hook in the dark)
const POOL = [115, 1 / 1.8, 0.35]; // (px at 1080 along the surface, its width across against that, alpha): its light
// pooled along the surface it strikes (at 110 px and 0.45 it barely lit the floor; at 0.6 two side by side, added,
// made a white strip)
export const SPLASH = [3, 6, 30, 80, 0.19, 0.28, 0.315, 2]; // (count, its range; length px at 1080, its range;
// intensity against its stroke, its range; width against its root's; frames): forks splashing out where it strikes
export const BRUSH = [2, 3, 0.06, 0.14, 0.3, 0.5, 0.26, 0.7]; // (count, its range; length in its own, its range;
// intensity, its range; radians turned, its range): a channel that strikes nothing ends in the air in a brush of dim
// forks fanning on from its end

/** A channel: its points, width (px), brightness against its bolt's main channel (1), forks, the parent's point it
 * forks from (``at``), and (a main channel) ``strikes``: which way the surface its far end strikes runs, or none.
 * ``endSoft`` lets a foreground filament fall away in depth; ``recessSoft`` buries one short stretch without losing
 * its glow. ``depth`` is the material crossed by a volume filament, independently of its optical defocus. */
export interface Channel { pts: P2[]; width: number; inten: number; forks: Channel[]; at: number; strikes?: P2; diffuse?: boolean;
  rootSoft?: number; rootScale?: number; softness?: number; endSoft?: number; recessSoft?: number; source?: PalmSource; filament?: boolean;
  depth?: number; endFade?: number }

/** An observed open palm's compact source and the fingers that may pass in front of it. */
export interface PalmSource { at: P2; radius: number; scale: number; landmarks: P2[]; aim: number; level?: number }

/** Midpoint displacement from ``a`` to ``b``: each midpoint pushed across its segment by up to ``rough`` of its length
 * (gently at the coarse levels so it runs on, sharply at the fine ones so it kinks), ``depth`` levels. */
function displace(a: P2, b: P2, rand: () => number, rough: number, depth: number): P2[] {
  let pts: P2[] = [a, b];
  for (let level = 0; level < depth; level++) {
    const out: P2[] = [pts[0]];
    const k = rough * (0.55 + 0.45 * level / Math.max(1, depth - 1));
    for (let i = 0; i + 1 < pts.length; i++) {
      const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
      const dx = x1 - x0, dy = y1 - y0;
      const push = (rand() - 0.5) * 2 * k;
      out.push([(x0 + x1) / 2 - dy * push, (y0 + y1) / 2 + dx * push], pts[i + 1]);
    }
    pts = out;
  }
  return pts;
}

function toSegment(p: P2, a: P2, b: P2): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

/** Whether polylines ``a`` and ``b`` cross or come within ``pad`` of each other, not counting ``a``'s first ``skip``
 * segments (where a branch leaves what it branches from). */
export function crosses(a: P2[], b: P2[], pad = 0, skip = 0): boolean {
  for (let i = Math.max(0, skip); i + 1 < a.length; i++) {
    const p = a[i], q = a[i + 1];
    const ax0 = Math.min(p[0], q[0]) - pad, ax1 = Math.max(p[0], q[0]) + pad;
    const ay0 = Math.min(p[1], q[1]) - pad, ay1 = Math.max(p[1], q[1]) + pad;
    for (let j = 0; j + 1 < b.length; j++) {
      const r = b[j], s = b[j + 1];
      if (Math.max(r[0], s[0]) < ax0 || Math.min(r[0], s[0]) > ax1) continue;
      if (Math.max(r[1], s[1]) < ay0 || Math.min(r[1], s[1]) > ay1) continue;
      const d1 = (s[0] - r[0]) * (p[1] - r[1]) - (s[1] - r[1]) * (p[0] - r[0]);
      const d2 = (s[0] - r[0]) * (q[1] - r[1]) - (s[1] - r[1]) * (q[0] - r[0]);
      const d3 = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
      const d4 = (q[0] - p[0]) * (s[1] - p[1]) - (q[1] - p[1]) * (s[0] - p[0]);
      if (d1 * d2 <= 0 && d3 * d4 <= 0) return true;
      if (pad > 0 && Math.min(toSegment(p, r, s), toSegment(q, r, s), toSegment(r, p, q), toSegment(s, p, q)) < pad) {
        return true;
      }
    }
  }
  return false;
}

/** A channel's course as the eye follows it: ``n`` points at equal lengths along it, smoothed twice over three, so
 * its finest kinks drop out. */
function course(pts: P2[], n: number): P2[] {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  }
  const out: P2[] = [];
  for (let k = 0, j = 0; k <= n; k++) {
    const at = (cum[cum.length - 1] * k) / n;
    while (j < cum.length - 2 && cum[j + 1] < at) j++;
    const u = cum[j + 1] > cum[j] ? Math.min(1, Math.max(0, (at - cum[j]) / (cum[j + 1] - cum[j]))) : 0;
    out.push([pts[j][0] + u * (pts[j + 1][0] - pts[j][0]), pts[j][1] + u * (pts[j + 1][1] - pts[j][1])]);
  }
  for (let pass = 0; pass < 2; pass++) {
    const prev = out.map((p): P2 => [p[0], p[1]]);
    for (let i = 1; i < n; i++) {
      out[i] = [(prev[i - 1][0] + prev[i][0] + prev[i + 1][0]) / 3, (prev[i - 1][1] + prev[i][1] + prev[i + 1][1]) / 3];
    }
  }
  return out;
}

/** How a channel is struck beyond its course: ``at``, the parent's point it forks from; ``inten``, its brightness
 * against the bolt's main channel; ``lead``, a point its first stretch runs to before it turns to its end;
 * ``keepOff``, channels its forks keep clear of; ``clear``, how far (in its own units; its width by default). */
export interface Strike { at?: number; inten?: number; lead?: P2; keepOff?: P2[][]; clear?: number; px?: number }

/** A channel from ``a`` to ``b``: jagged (six levels of midpoints by default), with two or three short weaker forks
 * when there is room. Each stays off its parent, siblings and ``keepOff`` channels, retried REDRAW times then dropped. */
export function channel(a: P2, b: P2, rand: () => number, width = 2, rough = 0.28, depth = 6, forkRate = 0.12,
  strike: Strike = {}): Channel {
  const { at = 0, inten = 1, lead, keepOff = [], clear = width, px = 1 } = strike;
  const pts = lead
    ? [...displace(a, lead, rand, rough, depth - 2).slice(0, -1), ...displace(lead, b, rand, rough, depth)]
    : displace(a, b, rand, rough, depth);
  const forks: Channel[] = [];
  if (width > 0.8 && depth > 3 && forkRate > 0) {
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const candidates = Array.from({ length: Math.max(0, Math.floor(pts.length * 0.35)) }, (_, i) =>
      Math.floor(pts.length * 0.15) + 2 * i).filter((i) => i < pts.length - 2);
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    const wanted = forkRate > 0 && inten >= 0.85 ? 2 + Number(rand() < 0.25) : 0;
    for (const i of candidates) {
      if (forks.length >= (wanted || 2)) break;
      const along = i / pts.length;
      if (!wanted && rand() >= forkRate * (0.3 + 1.4 * along) * 0.25) continue;
      const [sx, sy] = pts[i], [qx, qy] = pts[Math.min(i + 3, pts.length - 1)];
      const side = rand() < 0.5 ? -1 : 1;
      const ang = Math.atan2(qy - sy, qx - sx) + side * (FORK_TURN[0] + (FORK_TURN[1] - FORK_TURN[0]) * rand());
      const l = Math.min(len * (1 - along) * 0.66, (FORK_LENGTH[0] + (FORK_LENGTH[1] - FORK_LENGTH[0]) * rand()) * px);
      for (let tries = 0; tries <= REDRAW; tries++) {
        const dim = FORK_INTEN[0] + (FORK_INTEN[1] - FORK_INTEN[0]) * rand();
        const f = channel(pts[i], [sx + Math.cos(ang) * l, sy + Math.sin(ang) * l], rand, width * FORK_WIDTH, rough,
          depth - 2, 0, { at: i, inten: inten * dim, clear: clear * FORK_WIDTH });
        if (!crosses(f.pts, pts, clear, 2) && !keepOff.some((o) => crosses(f.pts, o, clear)) &&
          !forks.some((o) => crosses(f.pts, o.pts, clear))) { forks.push(f); break; }
      }
    }
  }
  return { pts, width, inten, forks, at };
}

/** ``c`` moved so its root is at ``root`` and its far end stays: each point by the root's move times how near the
 * root it is (the channel rides the hand at its root, holds where it struck at its end). */
export function ride(c: Channel, root: P2): Channel {
  const [dx, dy] = [root[0] - c.pts[0][0], root[1] - c.pts[0][1]];
  const n = c.pts.length - 1;
  const pts = c.pts.map(([x, y], i): P2 => [x + dx * (1 - i / n), y + dy * (1 - i / n)]);
  return { ...c, pts, forks: c.forks.map((f) => ride(f, pts[f.at])) };
}

// ---- the picture it strikes into
/** The camera's picture, small: its luminance (0-255) ``w`` x ``h``, each of its px ``k`` canvas px. */
export interface Plate { w: number; h: number; k: number; lum: Float32Array }

/** The picture the lightning was last drawn over (drawChannels reads it), where a new bolt looks for what to strike:
 * a frame old, as a strike is struck before it is drawn. */
let seen: Plate | null = null;
const small = new WeakMap<object, Ctx2D>();
const receiverContacts = new WeakMap<object, boolean>();

/** A resolved receiver in an already bright room needs local bounce, while a dark room keeps its broad response. */
export function receiverFocus(plate: Plate | null, contact: boolean): number {
  if (!plate || !contact || !plate.lum.length) return 0;
  let sum = 0, count = 0;
  const step = Math.max(1, Math.floor(plate.lum.length / 4096));
  for (let i = 0; i < plate.lum.length; i += step) { sum += plate.lum[i]; count++; }
  return Math.max(0, Math.min(1, (sum / count - 50) / 60));
}

/** Reads the picture under ``ctx`` as drawn so far, at 1 / EDGE_Q, into ``seen`` (none where it cannot be read). */
function look(ctx: CanvasRenderingContext2D) {
  const main = ctx.canvas as HTMLCanvasElement | undefined;
  if (!main?.width || typeof ctx.getImageData !== "function") return;
  const w = Math.max(1, Math.round(main.width / EDGE_Q)), h = Math.max(1, Math.round(main.height / EDGE_Q));
  let s = small.get(main);
  try {
    if (!s || s.canvas.width !== w || s.canvas.height !== h) {
      const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h)
        : Object.assign(document.createElement("canvas"), { width: w, height: h });
      const made = c.getContext("2d", { willReadFrequently: true }) as Ctx2D | null;
      if (!made) return;
      small.set(main, (s = made));
    }
    s.drawImage(main, 0, 0, w, h);
    const px = s.getImageData(0, 0, w, h).data;
    const lum = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) lum[i] = 0.2126 * px[4 * i] + 0.7152 * px[4 * i + 1] + 0.0722 * px[4 * i + 2];
    seen = { w, h, k: main.width / w, lum };
  } catch {
    seen = null; // (a picture it may not read)
  }
}

/** The edges of ``plate`` a bolt may strike: rows [x, y, strength, ax, ay] in canvas px, (ax, ay) the way the line
 * runs there: the picture softened (3 x 3, twice), its Sobel gradient over EDGE_MIN, thinned to the line's ridge
 * across it and kept where EDGE_NEAR more of them are round it (a line, not a speck). */
export function surfaces(plate: Plate): number[][] {
  const { w, h, k } = plate;
  let a = plate.lum;
  for (let pass = 0; pass < 2; pass++) {
    const b = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && xx < w && yy >= 0 && yy < h) { sum += a[yy * w + xx]; n++; }
        }
        b[y * w + x] = sum / n;
      }
    }
    a = b;
  }
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h), mag = new Float32Array(w * h);
  const at = (x: number, y: number) => a[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      gx[i] = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)
        - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
      gy[i] = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)
        - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
      mag[i] = Math.hypot(gx[i], gy[i]);
    }
  }
  const ridge = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (mag[i] < EDGE_MIN) continue;
      const sx = Math.round(gx[i] / mag[i]), sy = Math.round(gy[i] / mag[i]); // (across the line)
      if (mag[i] >= mag[(y + sy) * w + x + sx] && mag[i] >= mag[(y - sy) * w + x - sx]) ridge[i] = 1;
    }
  }
  const rows: number[][] = [];
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const i = y * w + x;
      if (!ridge[i]) continue;
      let near = -1;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) near += ridge[(y + dy) * w + x + dx];
      if (near < EDGE_NEAR) continue;
      rows.push([(x + 0.5) * k, (y + 0.5) * k, mag[i], -gy[i] / mag[i], gx[i] / mag[i]]);
    }
  }
  return rows;
}

/** Where a struck surface is: the point and the way the surface runs there. */
export interface Hit { at: P2; along: P2; diffuse?: boolean }

/** A wide, readable, low-contrast patch supports a diffuse wall receiver when no strong edge faces the finger. */
function flatPatch(plate: Plate, x: number, y: number): boolean {
  const { w, h, lum } = plate;
  if (x < 4 || x >= w - 4 || y < 4 || y >= h - 4) return false;
  let lo = Infinity, hi = -Infinity, sum = 0;
  for (let dy = -4; dy <= 4; dy += 2) for (let dx = -4; dx <= 4; dx += 2) {
    const v = lum[(y + dy) * w + x + dx];
    lo = Math.min(lo, v); hi = Math.max(hi, v); sum += v;
  }
  return hi - lo < 15 && sum / 25 > 18 && sum / 25 < 245;
}

/** A supported flat background patch in the pointing cone, safely inside the picture. Its response is broad and
 * diffuse because the camera provides no edge direction or visible contact detail to sharpen it against. */
function wallTarget(plate: Plate, root: P2, aim: number, want: number, rand: () => number): Hit | null {
  const { w, h, k } = plate, border = 0.16 * Math.min(w, h) * k;
  let best = -Infinity, hit: Hit | null = null;
  for (let y = 4; y < h - 4; y += 3) for (let x = 4; x < w - 4; x += 3) {
    const at: P2 = [(x + 0.5) * k, (y + 0.5) * k];
    if (at[0] < border || at[0] > w * k - 1 - border || at[1] < border || at[1] > h * k - 1 - border) continue;
    const d = Math.hypot(at[0] - root[0], at[1] - root[1]);
    if (d < 0.45 * want || d > 1.5 * want) continue;
    const off = Math.abs(((Math.atan2(at[1] - root[1], at[0] - root[0]) - aim + 3 * Math.PI) % (2 * Math.PI)) - Math.PI);
    if (off > TARGET_CONE[2] * Math.PI / 180 || !flatPatch(plate, x, y)) continue;
    const score = -Math.abs(d - want) / want - 1.2 * off + 0.08 * rand();
    if (score > best) {
      best = score;
      hit = { at, along: [-Math.sin(aim), Math.cos(aim)], diffuse: true };
    }
  }
  return hit;
}

/** Where a new main channel from ``root`` aimed ``aim`` (radians) strikes, preferring ``want`` px out, and the way
 * the surface runs there, or null; with ``prong``, a second point it lands on too (``two``). Of the ``edges``
 * (surfaces) TARGET_BAND of ``want`` out, TARGET_BORDER inside the ``size`` (w, h) frame and within TARGET_CONE of
 * the aim (a burst's first strike within its first cone, else its second; struck again, after one that struck at
 * ``last``, within its third and TARGET_APART from there): the strongest, nearest ``want`` out and nearest its
 * ``heading``, with a little chance; then one of the best TARGET_TOP, TARGET_APART from each other. The second point:
 * the best of those PRONG's px from the first. ``px``: a pixel at 1080 lines; ``wide``: within the third cone, as
 * if struck again (where the first two have nothing). ``roomWide`` permits a final search of real side surfaces,
 * out to WIDE_RECEIVER's angle and with its smaller frame margin, before inferring a diffuse wall receiver. */
export function pickTarget(edges: number[][], root: P2, aim: number, heading: number, want: number, size: P2,
  rand: () => number, px = 1, last: P2 | null = null, prong = false, wide = false, roomWide = false): (Hit & { two?: Hit }) | null {
  const [w, h] = size;
  const border = (roomWide ? WIDE_RECEIVER[1] : TARGET_BORDER) * Math.min(w, h), apart = TARGET_APART * px;
  const offOf = (d: number) => ((d + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
  const cones = roomWide ? [WIDE_RECEIVER[0]] : last || wide ? TARGET_CONE.slice(2) : TARGET_CONE.slice(0, 2);
  const found: [number, number[], number][] = []; // (cone, row, off)
  for (const row of edges) {
    const [x, y] = row;
    if (x < border || x > w - 1 - border || y < border || y > h - 1 - border) continue;
    if (last && Math.hypot(x - last[0], y - last[1]) < apart) continue;
    const dist = Math.hypot(x - root[0], y - root[1]);
    if (dist < TARGET_BAND[0] * want || dist > TARGET_BAND[1] * want) continue;
    const off = Math.abs(offOf(Math.atan2(y - root[1], x - root[0]) - aim)) * 180 / Math.PI;
    const cone = cones.findIndex((c) => off <= c);
    if (cone >= 0) found.push([cone, row, offOf(Math.atan2(y - root[1], x - root[0]) - heading)]);
  }
  const first = Math.min(...found.map(([c]) => c));
  const scored = found.filter(([c]) => c === first).map(([c, row, off]): [number, number[]] => [
    Math.min(1, row[2] / (2 * EDGE_MIN)) - Math.abs(Math.hypot(row[0] - root[0], row[1] - root[1]) - want) / want -
    0.5 * Math.abs(off) / (cones[c] * Math.PI / 180) + 0.3 * rand(), row]);
  scored.sort((p, q) => q[0] - p[0]);
  const best: number[][] = [];
  for (const [, row] of scored) {
    if (!best.every((p) => Math.hypot(row[0] - p[0], row[1] - p[1]) >= apart)) continue;
    best.push(row);
    if (best.length === TARGET_TOP) break;
  }
  if (!best.length) return null;
  const pick = best[Math.min(best.length - 1, Math.floor(rand() * best.length))];
  const hit: Hit & { two?: Hit } = { at: [pick[0], pick[1]], along: [pick[3], pick[4]] };
  if (prong) {
    const two = scored.find(([, row]) => {
      const gap = Math.hypot(row[0] - pick[0], row[1] - pick[1]);
      return gap >= PRONG[1] * px && gap <= PRONG[2] * px;
    });
    if (two) hit.two = { at: [two[1][0], two[1][1]], along: [two[1][3], two[1][4]] };
  }
  return hit;
}

/** The forks splashing out where ``main`` strikes a surface running ``along`` (SPLASH): two of every three along it,
 * either way, the third back off it the way the channel came in, at least half as bright as their stroke. */
function splash(main: Channel, along: P2, rand: () => number, px: number): Channel[] {
  const [n0, n1, l0, l1, i0, i1, wide] = SPLASH;
  const p = main.pts, end = p[p.length - 1], from = p[Math.max(0, p.length - 1 - Math.max(2, p.length >> 3))];
  const come = Math.atan2(end[1] - from[1], end[0] - from[0]);
  const out: Channel[] = [];
  let way = rand() < 0.5 ? 0 : Math.PI;
  const count = n0 + Math.floor(rand() * (n1 - n0 + 1));
  for (let i = 0; i < count; i++) {
    let a: number;
    if (i % 3 === 2) a = come + Math.PI + (rand() - 0.5) * 1.75;
    else {
      way += Math.PI;
      a = Math.atan2(along[1], along[0]) + way + (rand() - 0.5) * 1.05;
    }
    const l = (l0 + (l1 - l0) * rand()) * px;
    out.push({ ...channel(end, [end[0] + Math.cos(a) * l, end[1] + Math.sin(a) * l], rand, main.width * wide, 0.36, 4,
      0, { inten: i0 + (i1 - i0) * rand() }), at: p.length - 1 });
  }
  return out;
}

/** The brush of dim forks a main channel that strikes nothing ends in (BRUSH), as forks of it. */
function brush(main: Channel, rand: () => number): Channel[] {
  const [n0, n1, l0, l1, i0, i1, t0, t1] = BRUSH;
  const p = main.pts, len = Math.hypot(p[p.length - 1][0] - p[0][0], p[p.length - 1][1] - p[0][1]);
  const out: Channel[] = [];
  const count = n0 + Math.floor(rand() * (n1 - n0 + 1));
  for (let i = 0; i < count; i++) {
    const at = Math.min(p.length - 2, Math.floor((0.82 + 0.14 * rand()) * (p.length - 1)));
    const q = p[Math.min(p.length - 1, at + 3)];
    const a = Math.atan2(q[1] - p[at][1], q[0] - p[at][0]) + (rand() < 0.5 ? -1 : 1) * (t0 + (t1 - t0) * rand());
    const l = len * (l0 + (l1 - l0) * rand());
    out.push(channel(p[at], [p[at][0] + Math.cos(a) * l, p[at][1] + Math.sin(a) * l], rand, main.width * 0.33, 0.36, 4,
      0, { at, inten: i0 + (i1 - i0) * rand() }));
  }
  return out;
}

/** A stream of lightning: its channel, struck anew every 1-3 frames near its last aim (which sways slowly), how
 * bright its stroke is on this frame (1 struck, 0.55-0.9 re-struck, under 0.1 dark), the forks splashing where it
 * strikes a surface (``splash``, shown SPLASH's frames), how many frames since it was struck (``age``) and where it
 * last struck a surface (``hitAt``). */
export interface Stream { ch: Channel | null; left: number; aim: number; seed: number; flash?: number;
  splash?: Channel[]; age?: number; hitAt?: P2 }

function life(rand: () => number): number {
  const r = rand();
  return r < LIFE_P[0] ? LIFE[0] : r < LIFE_P[0] + LIFE_P[1] ? LIFE[1] : LIFE[2];
}

/** One frame of a main channel from ``root`` (welded to it): ridden to the hand and re-struck, or struck anew towards
 * ``aim`` (radians) when its time is up (or on ``strike``: a hit strikes every stream at once), its first ARC_LEAD
 * along the aim (0.07 for a real side surface more than 90 degrees off it, avoiding a root hook), at an edge of the
 * ``plate`` in its way about ``want``
 * px out (pickTarget: a contact flash and a splash there, somewhere new each time, one in four in two prongs) or, with
 * none, ``reach`` px long, ending in the air in a brush. ``level``: the voice, which sets how often a re-stroke
 * flashes; ``px``: a pixel at 1080 lines. */
export function streamStep(s: Stream, root: P2, aim: number, reach: number, width: number, strike: boolean,
  frame: number, level = 1, plate: Plate | null = null, px = 1, want = 0.7 * reach, requireSurface = false): Stream {
  const rand = rng(s.seed * 7919 + frame);
  const edges = plate ? surfaces(plate) : [];
  const oldEnd = s.ch?.pts[s.ch.pts.length - 1];
  const contact = !requireSurface || oldEnd && (s.ch?.diffuse && plate
    ? flatPatch(plate, Math.round(oldEnd[0] / plate.k - 0.5), Math.round(oldEnd[1] / plate.k - 0.5))
    : edges.some(([x, y]) => Math.hypot(x - oldEnd[0], y - oldEnd[1]) <= 24 * px));
  if (s.ch && s.ch.inten > 0 && s.left > 0 && !strike && contact) {
    const on = rand() < RESTROKE[0] + RESTROKE[1] * level;
    return { ...s, ch: ride(s.ch, root), left: s.left - 1, flash: on ? 0.55 + 0.35 * rand() : 0.1 * rand(),
      age: (s.age ?? 0) + 1 };
  }
  const off = s.ch ? s.aim - aim + (rand() - 0.5) * 2 * ARC_JITTER : (rand() - 0.5) * 2 * ARC_JITTER; // (near its last)
  const a = aim + Math.max(-ARC_OFF, Math.min(ARC_OFF, off * 0.6));
  const l = reach * (0.8 + 0.25 * rand());
  const size: P2 | null = plate ? [plate.w * plate.k, plate.h * plate.k] : null;
  const wanted = want * (0.9 + 0.2 * rand()), prong = rand() < PRONG[0];
  const hit: (Hit & { two?: Hit }) | null = size ? pickTarget(edges, root, aim, a, wanted, size, rand, px, s.hitAt ?? null, prong) ??
    pickTarget(edges, root, aim, a, wanted, size, rand, px, null, prong, true) ??
    (requireSurface ? pickTarget(edges, root, aim, a, wanted, size, rand, px, null, false, true, true) : null) ??
    (requireSurface && plate ? wallTarget(plate, root, aim, wanted, rand) : null) : null;
  if (requireSurface && !hit) {
    return { ch: { pts: [root, root], width, inten: 0, forks: [], at: 0 }, left: 0, aim, seed: s.seed, flash: 0,
      splash: [], age: 0 };
  }
  const far: P2 = hit ? hit.at : inside([root[0] + Math.cos(a) * l, root[1] + Math.sin(a) * l], root, size);
  const span = Math.hypot(far[0] - root[0], far[1] - root[1]);
  const turn = Math.abs(((Math.atan2(far[1] - root[1], far[0] - root[0]) - aim + 3 * Math.PI) % (2 * Math.PI)) - Math.PI);
  const leadShare = hit && !hit.diffuse && turn > Math.PI / 2 ? 0.07 : ARC_LEAD;
  const lead: P2 = [root[0] + Math.cos(aim) * leadShare * span, root[1] + Math.sin(aim) * leadShare * span];
  const ch = channel(root, far, rand, width, 0.26, 6, 0.1 + 0.1 * level, { lead, px });
  if (hit) { ch.strikes = hit.along; ch.diffuse = hit.diffuse; }
  else ch.forks.push(...brush(ch, rand));
  if (hit?.two) { // (a second prong, from far along it to the second point)
    const at = Math.floor((PRONG[3] + (PRONG[4] - PRONG[3]) * rand()) * (ch.pts.length - 1));
    for (let tries = 0; tries <= REDRAW; tries++) {
      const p = channel(ch.pts[at], hit.two.at, rand, width * PRONG[5] * (1 - 0.6 * at / ch.pts.length), 0.26, 5, 0,
        { at, inten: PRONG[5] });
      if (!crosses(p.pts, ch.pts, width, 2) && !ch.forks.some((f) => crosses(p.pts, f.pts, 1))) {
        p.strikes = hit.two.along;
        ch.forks.push(p);
        break;
      }
    }
  }
  return { ch, left: life(rand) - 1, aim: a, seed: s.seed, flash: 1, splash: hit && !hit.diffuse ? splash(ch, hit.along, rand, px) : [],
    age: 0, hitAt: hit ? hit.at : s.hitAt };
}

/** ``p``, or where the line from ``root`` to it comes TARGET_BORDER from the edge of the ``size`` (w, h) frame (a
 * channel that strikes nothing ends in the air inside the picture: running off it, it went nowhere). */
function inside(p: P2, root: P2, size: P2 | null): P2 {
  if (!size) return p;
  const m = TARGET_BORDER * Math.min(size[0], size[1]);
  let u = 1;
  for (const k of [0, 1]) {
    const d = p[k] - root[k], lo = Math.min(m, root[k]), hi = Math.max(size[k] - 1 - m, root[k]);
    if (p[k] < lo && d < 0) u = Math.min(u, (lo - root[k]) / d);
    if (p[k] > hi && d > 0) u = Math.min(u, (hi - root[k]) / d);
  }
  return [root[0] + (p[0] - root[0]) * u, root[1] + (p[1] - root[1]) * u];
}

/** One tracked source's main stream. The side list stays empty until another emitter can be tracked explicitly. */
export interface Arcs { main: Stream; sides: Stream[] }

/** No arcs yet: the first frame strikes them. */
export function newArcs(seed = 1): Arcs {
  return { main: { ch: null, left: 0, aim: 0, seed }, sides: [] };
}

function widthK(scale: number, px: number): number {
  return Math.max(WIDTH_K[0], Math.min(WIDTH_K[1], scale / (HAND_PX * px)));
}

/** Explicitly recognizes a palm cue: four extended fingers and a cue well away from every fingertip. The source
 * remains centered over the palm, lifted towards the camera; a 2D track uses the upward projected lift. This is
 * opt-in support: the current Live caller still selects fingertip arcs or an open-palm ball. */
export function palmSource(landmarks: P2[], emitter: P2, scale: number, px: number, lift: P2 = [0, -140 * px]): PalmSource | null {
  if (landmarks.length !== 21 || scale <= 0 || !landmarks.every((p) => p.every(Number.isFinite))) return null;
  const indices = [0, 5, 9, 13, 17];
  const palm: P2 = [0, 1].map((axis) => indices.reduce((sum, i) => sum + landmarks[i][axis], 0) / 5) as P2;
  const distance = (a: P2, b: P2) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  if ([8, 12, 16, 20].some((tip) => distance(landmarks[tip], palm) < 1.4 * distance(landmarks[tip - 2], palm)) ||
    [4, 8, 12, 16, 20].some((tip) => distance(emitter, landmarks[tip]) <= 0.6 * scale)) return null;
  return { at: [palm[0] + 1.6 * lift[0], palm[1] + 1.6 * lift[1]], radius: 0.035 * scale,
    scale, landmarks, aim: -Math.PI / 2 };
}

/** First tracked skin between the hovering root and palm. The short discharge and its receiving pool share it. */
export function sourceContact(source: PalmSource): P2 {
  const indices = [0, 5, 9, 13, 17];
  const palm = [0, 1].map((axis) => indices.reduce((sum, i) => sum + source.landmarks[i][axis], 0) / 5) as P2;
  for (let i = 0; i < 96; i++) {
    const u = i / 95;
    const p: P2 = [source.at[0] + u * (palm[0] - source.at[0]), source.at[1] + u * (palm[1] - source.at[1])];
    if (onHand(p, source.landmarks, source.scale)) return p;
  }
  return palm;
}

const tipLevels = new WeakMap<P2, number>(); // The current tracked tip carries the voice envelope past individual stroke flashes.
const tipOptics = new WeakMap<P2, { softness: number; motion: P2 }>();

/** One tracked source's discharge, normally a fingertip aimed along its distal joints. An explicit palm source
 * supplies its own center and upward aim. A full channel requires a camera surface; voice level invents no roots. */
export function arcStep(arcs: Arcs, tip: P2, aim: number, scale: number, level: number, strike: boolean,
  frame: number, px = 1, plate: Plate | null = seen, source: PalmSource | null = null): Arcs {
  tip = source?.at ?? tip;
  tipLevels.set(tip, Math.max(0, Math.min(1, level)));
  aim = source?.aim ?? aim;
  const kw = widthK(scale, px) * px;
  const width = kw * (ARC_WIDTH[0] + (ARC_WIDTH[1] - ARC_WIDTH[0]) * level);
  const diag = plate ? Math.hypot(plate.w * plate.k, plate.h * plate.k) : 1.6 * 1080 * px;
  const reach = diag * (ARC_REACH[0] + (ARC_REACH[1] - ARC_REACH[0]) * level);
  const want = diag * (TARGET_AT[0] + (TARGET_AT[1] - TARGET_AT[0]) * level) * (source ? 0.65 : 1);
  const main = streamStep(arcs.main, tip, aim, reach, width, strike, frame, level, plate, px, want, true);
  if (main.ch) {
    const old = arcs.main.ch?.pts[0] ?? tip;
    const move = Math.hypot(tip[0] - old[0], tip[1] - old[1]);
    const trackedSoft = (0.7 + Math.min(2.5, 0.12 * move / Math.max(px, 1e-6))) * px;
    main.ch.rootSoft = source ? Math.max(trackedSoft, Math.min(6 * px, 0.012 * scale))
      : Math.min(8 * px, Math.max(trackedSoft, 0.018 * scale));
    main.ch.rootScale = scale;
    main.ch.source = source ? { ...source, level } : undefined;
    if (!source) {
      const smear = Math.min(1, 8 * px / Math.max(move, 1e-6));
      tipOptics.set(tip, { softness: main.ch.rootSoft, motion: [(tip[0] - old[0]) * smear, (tip[1] - old[1]) * smear] });
    }
  }
  return { main, sides: [] };
}

/** Every channel of the arcs as drawn this frame, with how bright: [channel, gain] (the main channel's splash on its
 * first SPLASH frames). */
export function arcChannels(arcs: Arcs, level: number): [Channel, number][] {
  const g = 0.55 + 0.45 * level;
  const out: [Channel, number][] = [];
  const m = arcs.main;
  if (m.ch && m.ch.inten > 0 && (m.flash ?? 1) > 0.02) out.push([m.ch, (m.flash ?? 1) * g]);
  if (m.ch?.source && m.ch.inten > 0) {
    const source = m.ch.source, rand = rng(m.seed + (m.age ?? 0) * 104729);
    const contact = sourceContact(source), lv = source.level ?? level;
    const kw = m.ch.width / (ARC_WIDTH[0] + (ARC_WIDTH[1] - ARC_WIDTH[0]) * lv);
    if (Math.hypot(contact[0] - source.at[0], contact[1] - source.at[1]) > 2 * source.scale / HAND_PX) {
      out.push([channel(source.at, contact, rand, 0.85 * kw, 0.03, 4, 0, { inten: 0.9 * lv }), (m.flash ?? 1) * g]);
    }
    for (let i = 0; i < 2; i++) {
      const angle = -Math.PI * (0.15 + 0.7 * rand()), reach = source.scale * (0.04 + 0.08 * rand());
      out.push([channel(source.at, [source.at[0] + reach * Math.cos(angle), source.at[1] + reach * Math.sin(angle)],
        rand, 0.3 * m.ch.width, 0.2, 4, 0, { inten: 0.35 }), (m.flash ?? 1) * g]);
    }
  }
  if (m.ch && (m.age ?? 0) < SPLASH[7]) for (const s of m.splash ?? []) out.push([s, (m.flash ?? 1) * g]);
  for (const s of arcs.sides) if (s.ch && (s.flash ?? 1) > 0.02) out.push([s.ch, (s.flash ?? 1) * g]);
  return out;
}

/** The same path reuses its hot stretches and current dip, including where the foreground recedes optically. */
function currentProfile(c: Channel, total: number) {
  const start = c.pts[0], end = c.pts[c.pts.length - 1];
  const rand = rng((4096 * Math.round((end[0] - start[0]) / 4) ^ 8192 * Math.round((end[1] - start[1]) / 4) ^
    131071 * Math.round(total / 4)) >>> 0);
  const hot = [0.12 + 0.14 * rand(), 0.4 + 0.13 * rand(), 0.72 + 0.16 * rand()]
    .map((at) => ({ at, span: 0.035 + 0.03 * rand(), power: 0.72 + 0.28 * rand() }));
  return { hot, dipAt: 0.36 + 0.08 * rand() };
}

/** The stretches a channel is drawn in: [points, width px, alpha], thinning along it and, a branch (dimmer than the
 * bolt's main channel), fading (TAPER) and thinning more (BRANCH_W) along it. ``top`` is this frame's strongest
 * emitted intensity, so a bright foreground path retains its localized current dip while the discharge fades. */
export function stretches(c: Channel, gain: number, top = 1): [P2[], number, number][] {
  const branch = !c.filament && c.inten < 0.6 ? Math.min(1, (0.6 - c.inten) / 0.2) : 0;
  const hotFilament = c.filament && (c.depth ?? Number((c.softness ?? 0) > 0)) === 0 &&
    (!(c.softness ?? 0) || !!c.recessSoft) && c.inten * gain >= 0.65 * top;
  const out: [P2[], number, number][] = [];
  const n = c.pts.length - 1;
  if (n < 1) return out;
  const cumulative = [0];
  for (let i = 1; i <= n; i++) cumulative.push(cumulative[i - 1] +
    Math.hypot(c.pts[i][0] - c.pts[i - 1][0], c.pts[i][1] - c.pts[i - 1][1]));
  const total = cumulative[n];
  if (total <= 1e-9) return out;
  const { hot, dipAt } = currentProfile(c, total);
  const chunks = c.filament || branch > 0 ? CHUNKS : 24;
  const currents = Array.from({ length: chunks }, (_, k) => Math.max(...hot.map(({ at, span, power }) =>
    power * Math.exp(-0.5 * (((k + 0.5) / chunks - at) / span) ** 2))));
  const peak = Math.max(...currents, 1e-9);
  const atDistance = (distance: number): P2 => {
    let j = 0;
    while (j < n - 1 && cumulative[j + 1] < distance) j++;
    const u = Math.max(0, Math.min(1, (distance - cumulative[j]) / (cumulative[j + 1] - cumulative[j] || 1)));
    return [c.pts[j][0] + u * (c.pts[j + 1][0] - c.pts[j][0]),
      c.pts[j][1] + u * (c.pts[j + 1][1] - c.pts[j][1])];
  };
  const neck = Math.max(1, Math.min(chunks - 2, Math.round(hot[1].at * chunks - 0.5)));
  for (let k = 0; k < chunks; k++) {
    const start = total * k / chunks, stop = total * (k + 1) / chunks;
    const pts = [atDistance(start)];
    for (let i = 1; i < n; i++) if (cumulative[i] > start && cumulative[i] < stop) pts.push(c.pts[i]);
    pts.push(atDistance(stop));
    const u = k / (chunks - 1), current = 0.015 + 0.985 * currents[k] / peak;
    const taper = 1 - (1 - MAIN_TAPER[0]) * ((k + 0.5) / chunks) ** MAIN_TAPER[1];
    const thin = (c.filament ? 0.65 + 0.55 * current : 0.1 + 1.4 * current ** 0.75) * taper * (1 - branch) +
      (BRANCH_W[0] + (BRANCH_W[1] - BRANCH_W[0]) * u) * branch;
    const interruption = !c.filament && branch < 1 ? Math.exp(-0.5 * ((k - neck) / (0.013 * chunks)) ** 4) : 0;
    const dip = hotFilament ? 1 - 0.80 * Math.exp(-0.5 * (((k + 0.5) / chunks - dipAt) / 0.055) ** 4) : 1;
    const filamentAlpha = (0.55 + 0.6 * current) * dip;
    out.push([pts, Math.max(branch ? BRANCH_W[2] : 0, c.width * thin * (1 - 0.78 * interruption)),
      c.inten * gain * (1 - TAPER[0] * branch * u ** TAPER[1]) * (1 + 0.2 * branch * (1 - u)) *
      (branch + (1 - branch) * (c.filament ? filamentAlpha : (0.025 + 1.45 * current ** 1.5) * (1 - 0.96 * interruption)))]);
  }
  return out;
}

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
const SOFT_Q = 4; // the glows and the hand's light are drawn on a canvas this many times smaller, blurred, laid over
/** How channels are drawn: their glows ((blur in px of the small canvas at 1080 lines, alpha, rgb), the rim round
 * their core, and how much bluer a dim branch's core goes (red, green taken off). */
export interface Look { glows: [number, number, string][]; rim: string; branch: [number, number] }
export const ARC_LOOK: Look = { // the arcs: the air round a channel glows blue, violet further out (a paler glow over a
  glows: [[0.8, 0.55, "110, 150, 255"], [2.5, 0.3, "70, 110, 255"], [7, 0.16, "95, 85, 255"]], // lit wall went white)
  rim: "120, 140, 255", branch: [90, 70],
};
export const BALL_LOOK: Look = { // Cyan-blue near the core; broad bloom stays faint enough to leave the volume open.
  glows: [[0.8, 0.5, "120, 205, 255"], [2.5, 0.28, "80, 170, 255"], [7, 0.0252, "60, 140, 255"]],
  rim: "150, 215, 255", branch: [40, 20],
};
const soft = new WeakMap<object, Map<string, Ctx2D>>();

/** A canvas SOFT_Q times smaller than ``ctx``'s, cleared, drawn on in ``ctx``'s pixels, named so each use keeps its
 * own; null where there is none to make (a test's context). */
function softLayer(ctx: CanvasRenderingContext2D, name: string): Ctx2D | null {
  const main = ctx.canvas as HTMLCanvasElement | undefined;
  if (!main?.width) return null;
  let byName = soft.get(main);
  if (!byName) soft.set(main, (byName = new Map()));
  const w = Math.ceil(main.width / SOFT_Q), h = Math.ceil(main.height / SOFT_Q);
  let s = byName.get(name);
  if (!s || s.canvas.width !== w || s.canvas.height !== h) {
    const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h)
      : Object.assign(document.createElement("canvas"), { width: w, height: h });
    const made = c.getContext("2d") as Ctx2D | null;
    if (!made) return null;
    byName.set(name, (s = made));
  }
  s.setTransform(1, 0, 0, 1, 0, 0);
  s.globalCompositeOperation = "source-over";
  s.globalAlpha = 1;
  s.filter = "none";
  s.clearRect(0, 0, w, h);
  s.setTransform(1 / SOFT_Q, 0, 0, 1 / SOFT_Q, 0, 0);
  return s;
}

function stroke(ctx: Ctx2D, pts: P2[]) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

/** Contact light follows the struck edge, with a narrow bright streak embedded in the broader blue spill. */
function drawImpact(ctx: CanvasRenderingContext2D, at: P2, along: P2, amp: number, px: number, diffuse = false) {
  const [len, across, alpha] = diffuse ? [190, 0.8, 0.22] : POOL;
  const cap = alpha * (1 - 0.45 * receiverFocus(seen, !diffuse));
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.translate(at[0], at[1]);
  ctx.rotate(Math.atan2(along[1], along[0]));
  ctx.scale(1, across);
  const r = len * px;
  const pool = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
  for (let i = 0; i <= 16; i++) {
    const u = i / 16, d = u * len, fade = (1 - u * u) ** 2;
    const response = fade * Math.exp(-0.5 * (d / (0.6 * len)) ** 2) / (1 + (d / 34) ** 2);
    pool.addColorStop(u, `rgba(100, 170, 255, ${cap * amp * response})`);
  }
  ctx.fillStyle = pool;
  ctx.fillRect(-r, -r, 2 * r, 2 * r);
  ctx.restore();
  if (diffuse) return;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.translate(at[0], at[1]);
  ctx.rotate(Math.atan2(along[1], along[0]));
  ctx.scale(1, 4 / 16);
  const spill = ctx.createRadialGradient(0, 0, 0, 0, 0, 24 * px);
  spill.addColorStop(0, `rgba(95, 175, 255, ${Math.min(1, 0.18 * amp)})`);
  spill.addColorStop(0.33, `rgba(80, 150, 255, ${0.11 * amp})`);
  spill.addColorStop(1, "rgba(60, 130, 255, 0)");
  ctx.fillStyle = spill;
  ctx.fillRect(-24 * px, -24 * px, 48 * px, 48 * px);
  ctx.restore();
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.translate(at[0], at[1]);
  ctx.rotate(Math.atan2(along[1], along[0]));
  ctx.scale(1, 0.8 / 18);
  const hot = ctx.createRadialGradient(0, 0, 0, 0, 0, 54 * px);
  for (let i = 0; i <= 12; i++) {
    const u = i / 12;
    hot.addColorStop(u, `rgba(225, 243, 255, ${i === 12 ? 0 : 1 - Math.exp(-0.5 * amp * Math.exp(-0.5 * (3 * u) ** 2))})`);
  }
  ctx.fillStyle = hot;
  ctx.fillRect(-54 * px, -54 * px, 108 * px, 108 * px);
  ctx.restore();
}

/** Draws channels, added: the blue air glowing round them (drawn small, blurred at three sizes, so a Gaussian-like
 * falloff in Canvas 2D at a sixteenth of the pixels; without a canvas for it, wide faint strokes), a blue rim and the
 * core, white for a main channel (far over white: a phone clips it white with a blue rim) and bluer the
 * dimmer a branch, each channel in its stretches (thinning, a branch fading along it), and a contact flash where a
 * bright one strikes a surface (IMPACT_MIN). First (``read``) it reads the picture it is drawn over, for the next
 * strike to look for what to hit in. ``px``: a pixel at 1080 lines; ``most``: the widest glow, in px (a small ball's
 * arcs glow no wider than it is); ``look_``: its colours (ARC_LOOK, BALL_LOOK). */
export function drawChannels(ctx: CanvasRenderingContext2D, channels: Channel[] | [Channel, number][], px: number,
  glow = 1, most = Infinity, read = true, look_: Look = ARC_LOOK) {
  if (read) look(ctx);
  const list = (channels as (Channel | [Channel, number])[])
    .map((c): [Channel, number] => (Array.isArray(c) ? c : [c, 1]));
  if (!list.length) return;
  if (read) {
    const resolved = (c: Channel): boolean => !!(c.strikes && !c.diffuse) || c.forks.some(resolved);
    receiverContacts.set(ctx, list.some(([c]) => resolved(c)));
  }
  const source = list.find(([c]) => c.source)?.[0].source;
  const sourceSoft = source ? Math.min(3 * px, Math.max(0.6 * px, 0.012 * source.scale)) : undefined;
  const top = Math.max(1e-9, ...list.map(([c, gain]) => c.inten * gain));
  const flat: [P2[], number, number, number, number][] = []; // points, width, alpha, branch share, optical softness
  const walk = (c: Channel, g: number) => {
    const branch = c.filament ? 0 : Math.min(1, Math.max(0, (0.6 - c.inten) / 0.4));
    const parts = stretches(c, g, top);
    const recessAt = c.recessSoft && c.pts.length > 1 ? currentProfile(c, c.pts.slice(1).reduce((sum, p, i) =>
      sum + Math.hypot(p[0] - c.pts[i][0], p[1] - c.pts[i][1]), 0)).dipAt : 0;
    for (let i = 0; i < parts.length; i++) {
      const [pts, w, a] = parts[i];
      let rootSoft = i < 2 ? c.rootSoft ?? 0.6 * px : 0.25 * px;
      if (c.rootScale && c.rootSoft !== undefined) {
        const first = pts[0], last = pts[pts.length - 1], root = c.pts[0];
        const distance = Math.hypot((first[0] + last[0]) / 2 - root[0], (first[1] + last[1]) / 2 - root[1]);
        const fade = Math.max(0, Math.min(1, (0.8 * c.rootScale - distance) / (0.65 * c.rootScale)));
        rootSoft = 0.25 * px + (c.rootSoft - 0.25 * px) * fade * fade * (3 - 2 * fade);
      }
      let softness = c.softness ?? sourceSoft ?? rootSoft, alpha = a;
      if (!source && c.rootScale && !c.filament) {
        const first = pts[0], last = pts[pts.length - 1], root = c.pts[0];
        const distance = Math.hypot((first[0] + last[0]) / 2 - root[0], (first[1] + last[1]) / 2 - root[1]);
        alpha *= 1 - 0.15 * Math.exp(-0.5 * (distance / (0.045 * c.rootScale)) ** 2);
      }
      if ((c.endSoft || c.recessSoft) && c.filament && (c.depth ?? 0) === 0 && c.inten * g >= 0.65 * top) {
        const start = c.pts[0], end = c.pts[c.pts.length - 1], dx = end[0] - start[0], dy = end[1] - start[1];
        const first = pts[0], last = pts[pts.length - 1];
        const along = Math.max(0, Math.min(1, (((first[0] + last[0]) / 2 - start[0]) * dx +
          ((first[1] + last[1]) / 2 - start[1]) * dy) / (dx * dx + dy * dy || 1)));
        const fade = Math.max(0, Math.min(1, (Math.abs(along - 0.5) - 0.20) / 0.25));
        const ends = fade * fade * (3 - 2 * fade);
        if (c.endSoft) {
          softness = Math.hypot(softness, c.endSoft * Math.sqrt(ends));
          alpha *= 1 - (c.endFade ?? 0.35) * ends;
        }
        if (c.recessSoft) {
          const recess = Math.exp(-0.5 * ((along - recessAt) / 0.085) ** 4);
          softness = Math.hypot(softness, c.recessSoft * Math.sqrt(recess));
          alpha *= 1 - 0.75 * recess;
        }
      }
      flat.push([pts, w, alpha, branch, softness]);
    }
    c.forks.forEach((f) => walk(f, g));
  };
  list.forEach(([c, g]) => walk(c, g));
  const lit = flat.filter(([pts, , a]) => pts.length > 1 && a > 0.001);
  ctx.save();
  if (source) drawSourceReflection(ctx, source);
  if (source) clipFingers(ctx, source.landmarks, source.scale);
  ctx.globalCompositeOperation = "lighter";
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const air = softLayer(ctx, "glow");
  const blur = air && softLayer(ctx, "glow-soft");
  if (air && blur) {
    air.lineJoin = "round";
    air.lineCap = "round";
    blur.setTransform(1, 0, 0, 1, 0, 0);
    blur.globalCompositeOperation = "lighter";
    for (const [sigma, amp, rgb] of look_.glows) {
      if (sigma * px * SOFT_Q > most) continue;
      air.setTransform(1, 0, 0, 1, 0, 0);
      air.clearRect(0, 0, air.canvas.width, air.canvas.height);
      air.setTransform(1 / SOFT_Q, 0, 0, 1 / SOFT_Q, 0, 0);
      for (const [pts, w, a, , softness] of lit) {
        air.filter = `blur(${softness / SOFT_Q}px)`;
        air.strokeStyle = `rgba(${rgb}, ${Math.min(1, amp * a * glow * Math.min(1, w / (2 * px)))})`;
        air.lineWidth = Math.max(w, 2 * sigma * px * SOFT_Q); // (as wide as its blur: its peak stays near its alpha)
        stroke(air, pts);
      }
      blur.filter = `blur(${sigma * px}px)`;
      blur.drawImage(air.canvas, 0, 0);
    }
    ctx.drawImage(blur.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  } else {
    for (const [pts, w, a] of lit) {
      ctx.strokeStyle = `rgba(60, 90, 255, ${Math.min(1, 0.1 * a * glow * (look_ === BALL_LOOK ? 0.18 : 1))})`;
      ctx.lineWidth = Math.max(0.35 * px, 8 * w);
      stroke(ctx, pts);
    }
  }
  for (const [pts, w, a, , softness] of lit) { // the rim shares the filament's focus, including its far stretches
    ctx.filter = `blur(${softness}px)`;
    ctx.strokeStyle = `rgba(${look_.rim}, ${Math.min(1, 0.3 * a * Math.min(1, w / px))})`;
    ctx.lineWidth = Math.max(0.22 * px, 2.5 * w);
    stroke(ctx, pts);
  }
  for (const [pts, w, a, branch, softness] of lit) { // the core
    const [r, g] = [238 - look_.branch[0] * branch, 243 - look_.branch[1] * branch];
    ctx.strokeStyle = `rgba(${r}, ${g}, 255, ${Math.min(1, a * (1.2 - 0.4 * branch))})`;
    ctx.lineWidth = Math.max(0.08 * px, w);
    ctx.filter = `blur(${softness}px)`;
    stroke(ctx, pts);
  }
  ctx.filter = "none";
  if (source) drawSpark(ctx, source.at, source.radius, 0.65 * Math.max(0, ...list.map(([, gain]) => gain)), false,
    { softness: sourceSoft });
  ctx.restore();
  const impacts = (c: Channel, g: number) => { // (a channel's, and a prong's of it)
    if (c.strikes && c.inten * g >= IMPACT_MIN) drawImpact(ctx, c.pts[c.pts.length - 1], c.strikes, c.inten * g, px, c.diffuse);
    c.forks.forEach((f) => impacts(f, g));
  };
  for (const [c, g] of list) impacts(c, g);
}

/** The fingertip's compact spark leaves its reflected skin light dominant; a hovering source keeps its air corona. */
export function drawSpark(ctx: CanvasRenderingContext2D, at: P2, r: number, amp: number, onSkin = true,
  optics: { softness?: number; motion?: P2 } = onSkin ? tipOptics.get(at) ?? {} : {}) {
  if (amp <= 0) return;
  if (onSkin) { r *= 0.5 * 0.005 / 0.035; amp *= 0.85; }
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.filter = `blur(${Math.max(0, optics.softness ?? 0)}px)`;
  const motion = optics.motion ?? [0, 0], moving = Math.hypot(...motion) > 0.1;
  const taps = moving ? 5 : 1;
  for (let i = 0; i < taps; i++) {
    const t = moving ? i / (taps - 1) : 0;
    const x = at[0] - t * motion[0], y = at[1] - t * motion[1], weight = moving ? (i === 0 || i === taps - 1 ? 0.125 : 0.25) : 1;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(240, 248, 255, ${Math.min(1, amp) * weight})`);
    g.addColorStop(0.3, `rgba(150, 200, 255, ${0.6 * Math.min(1, amp) * weight})`);
    g.addColorStop(1, "rgba(60, 110, 255, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
  ctx.restore();
}

// ---- the light it throws
export const LIGHT_HOLD = 4, LIGHT_TAIL = 0.1; // frames the light holds the strokes' peak while they strike, and the
// share of it it keeps for one frame once they stop (letting go by 0.55 a frame, the room stayed lit with no bolt in
// it)

/** The light's state: the last LIGHT_HOLD frames' power while the strokes strike, and the light held from them. */
export interface Light { recent: number[]; held: number }

/** One frame of the light the lightning throws, for this frame's ``power`` (0-1: how much is striking): while it
 * strikes, the strokes' peak over LIGHT_HOLD frames; once they stop, LIGHT_TAIL of it for one frame, then none. */
export function lightStep(l: Light | null, power: number): Light {
  if (power <= 0) return { recent: [], held: l?.recent.length ? LIGHT_TAIL * l.held : 0 };
  const recent = [...(l?.recent ?? []), power].slice(-LIGHT_HOLD);
  return { recent, held: Math.max(...recent) };
}

export const DIP = [0.12, 4, 24]; // stops, frames rising, frames falling: the camera exposes down for the ball (as a
// phone's does for a bright light in the picture), an eighth of a stop for a light a twentieth of the frame across (at
// a third the whole room went darker), settling for a phrase rather than breathing with the voice

/** Stops the camera is exposed down this frame, following ``target`` (0-1, how bright the ball is) from ``prev``:
 * DIP's stops at most, rising over its first frames, coming back over its second. */
export function dipStep(prev: number, target: number): number {
  const want = DIP[0] * Math.max(0, Math.min(1, target));
  return prev + (want - prev) * (1 - Math.exp(-1 / (want > prev ? DIP[1] : DIP[2])));
}

/** Exposes the picture down ``stops`` (multiplied, as the camera would). */
export function drawDip(ctx: CanvasRenderingContext2D, stops: number, w: number, h: number) {
  if (stops < 0.005) return;
  const v = Math.round(255 * 2 ** (-stops / 2.2)); // (as shown: the linear light's 2^-stops through the tone curve)
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}

const PALM = [0, 1, 2, 5, 9, 13, 17]; // the hand's landmarks round its palm, and along each finger
const FINGERS = [[2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];

export const ROOM_RGB = "202, 224, 255"; // Neutral blue-white reflected light preserves the room's colour under a strike.
export const HAND_RGB = "175, 195, 255"; // Cool white keeps illuminated skin from turning pink under a violet tint.
export const BALL_ROOM_RGB = "0, 0, 0"; // the ball's light through drawLight's broad wash: none (screened, black
export const BALL_HAND_RGB = "0, 0, 0"; // leaves the picture as it was). Its light is its own, close round it and gone
// by three radii (drawBallLight; washed over the frame, it shifted the far room blue: a colour grade)

/** The light on the room and the hand, screened over the picture: round ``at`` (the bolts or the ball) on the room,
 * weaker out to the frame's edge; on the hand (its 21 landmarks ``lm``, px: the palm and each finger as thick as a
 * finger, drawn small and soft) round the same place, strong close in, times ``handGain``. ``strength``: the held
 * light (0-1). */
export function drawLight(ctx: CanvasRenderingContext2D, at: P2, scale: number, strength: number, lm: P2[] | null,
  room: string, onHand: string, w: number, h: number, handGain = 1) {
  if (strength <= 0 || (room === "0, 0, 0" && onHand === "0, 0, 0")) return; // (screened, black changes nothing)
  const keyFocus = receiverFocus(seen, true);
  const focus = receiverContacts.get(ctx) ? keyFocus : 0;
  const roomGain = (1 - 0.85 * focus) * (1 - 0.65 * focus);
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  const g = ctx.createRadialGradient(at[0], at[1], 0, at[0], at[1], 0.8 * Math.hypot(w, h) * (1 - 0.65 * focus));
  g.addColorStop(0, `rgba(${room}, ${0.4 * strength * roomGain})`);
  g.addColorStop(0.3, `rgba(${room}, ${0.18 * strength * roomGain})`);
  g.addColorStop(1, `rgba(${room}, ${0.06 * strength * roomGain})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  const palm = lm ? [0, 1].map((axis) => [0, 5, 9, 13, 17].reduce((sum, i) => sum + lm[i][axis], 0) / 5) as P2 : null;
  const source = lm && palm ? palmSource(lm, palm, scale, Math.min(w, h) / 1080) : null;
  const mask = lm ? handLayer(ctx, "hand", lm, scale) : null;
  if (mask) {
    if (source) sourceShadows(mask, source);
    const k = mask.createRadialGradient(at[0], at[1], 0, at[0], at[1], 2.5 * scale);
    const gain = handGain * (source ? 0.50 * 0.16 : 1 - 0.65 * keyFocus);
    const cap = 0.32 - 0.12 * keyFocus;
    const response = (value: number) => source ? value : cap * Math.tanh(value / cap);
    k.addColorStop(0, `rgba(${onHand}, ${response(0.75 * gain * strength)})`);
    k.addColorStop(1, `rgba(${onHand}, ${response(0.15 * gain * strength)})`);
    mask.fillStyle = k;
    mask.fillRect(0, 0, w, h);
    ctx.drawImage(mask.canvas, 0, 0, w, h);
  }
  if (lm && !source) drawTipReflection(ctx, lm, scale, tipLevels.get(lm[8]) ?? strength);
  ctx.restore();
}

const tipCanvas = new WeakMap<object, Ctx2D>();

/** A broad low-energy reflection follows the finger's curved sides and fades rapidly away from the contact.
 * Screening this local field leaves pores and creases in the camera image visible beneath the blue-white return. */
function drawTipReflection(ctx: CanvasRenderingContext2D, lm: P2[], scale: number, strength: number,
  tipIndex = 8, whitening = 0.12, cap = 0.24) {
  const main = ctx.canvas as HTMLCanvasElement | undefined;
  if (!main?.width || typeof ctx.drawImage !== "function") return;
  const tip = lm[tipIndex], joint = lm[tipIndex - 2], length = Math.hypot(joint[0] - tip[0], joint[1] - tip[1]);
  if (length < 0.08 * scale) return;
  let texture = tipCanvas.get(main);
  if (!texture) {
    const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(64, 64)
      : Object.assign(document.createElement("canvas"), { width: 64, height: 64 });
    const made = canvas.getContext("2d") as Ctx2D | null;
    if (!made || typeof made.createImageData !== "function") return;
    tipCanvas.set(main, (texture = made));
  }
  const pixels = texture.createImageData(64, 64);
  for (let j = 0; j < 64; j++) for (let i = 0; i < 64; i++) {
    const along = -0.025 + 0.385 * (j + 0.5) / 64, across = 0.6 * ((i + 0.5) / 64 - 0.5);
    const curvature = Math.exp(-0.5 * (across / 0.15) ** 2);
    const fade = Math.max(0, Math.min(1, (0.34 - along) / 0.08));
    const contact = Math.max(0, Math.min(1, (along + 0.025) / 0.035));
    const distance = Math.max(0, along);
    const diffuse = curvature * Math.exp(-distance / 0.22) / (1 + (distance / 0.11) ** 2);
    let reflected = (1 - Math.exp(-0.63 * strength * diffuse)) * fade * contact;
    if (cap < 1) reflected = cap * Math.tanh(reflected / cap);
    const white = whitening * (2.1 * strength * diffuse) / (1 + 2.1 * strength * diffuse);
    pixels.data.set([185 + Math.round(70 * white), 218 + Math.round(37 * white), 255, Math.round(255 * reflected)],
      4 * (j * 64 + i));
  }
  texture.putImageData(pixels, 0, 0);
  const mask = handLayer(ctx, "tip-reflection", lm, scale);
  if (!mask) return;
  mask.save();
  mask.translate(...tip);
  mask.rotate(Math.atan2(joint[1] - tip[1], joint[0] - tip[0]) - Math.PI / 2);
  mask.drawImage(texture.canvas, -0.3 * scale, -0.025 * scale, 0.6 * scale, 0.385 * scale);
  mask.restore();
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.25;
  ctx.drawImage(mask.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "screen";
  ctx.drawImage(mask.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

/** A soft layer (softLayer, named ``name``) holding the hand as white (its 21 landmarks ``lm``, px: the palm and each
 * finger as thick as a finger, its outline as soft as a lit hand's), set so what is filled over it next lands on the
 * hand only; null where there is none to make. */
function handLayer(ctx: CanvasRenderingContext2D, name: string, lm: P2[], scale: number): Ctx2D | null {
  const mask = softLayer(ctx, name);
  if (!mask) return null;
  mask.fillStyle = mask.strokeStyle = "#fff";
  mask.lineJoin = "round";
  mask.lineCap = "round";
  mask.lineWidth = 0.2 * scale;
  mask.filter = `blur(${(0.05 * scale) / SOFT_Q}px)`;
  mask.beginPath();
  PALM.forEach((i, k) => (k ? mask.lineTo(...lm[i]) : mask.moveTo(...lm[i])));
  mask.closePath();
  mask.fill();
  mask.stroke();
  for (const f of FINGERS) stroke(mask, f.map((i) => lm[i]));
  mask.filter = "none";
  mask.globalCompositeOperation = "source-in";
  return mask;
}

/** Remove new source light behind each foreground finger. Swept capsules follow rays away from the source;
 * this leaves the camera's existing shadows intact instead of painting dark shapes into the picture. */
function sourceShadows(mask: Ctx2D, source: PalmSource) {
  mask.globalCompositeOperation = "destination-out";
  mask.lineCap = "round";
  mask.filter = `blur(${0.012 * source.scale / SOFT_Q}px)`;
  for (const finger of FINGERS) for (let j = 0; j < finger.length - 1; j++) {
    const a = source.landmarks[finger[j]], b = source.landmarks[finger[j + 1]];
    const midpoint: P2 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    if (Math.hypot(midpoint[0] - source.at[0], midpoint[1] - source.at[1]) > 0.9 * source.scale) continue;
    mask.lineWidth = (j === finger.length - 2 ? 0.1 : 0.13) * source.scale;
    for (let step = 0; step < 4; step++) {
      mask.strokeStyle = `rgba(0, 0, 0, ${0.8 - 0.12 * step})`;
      stroke(mask, [a, b].map((p): P2 => {
        const d = Math.max(1, Math.hypot(p[0] - source.at[0], p[1] - source.at[1]));
        const distance = (0.03 + 0.075 * step) * source.scale;
        return [p[0] + distance * (p[0] - source.at[0]) / d, p[1] + distance * (p[1] - source.at[1]) / d];
      }));
    }
  }
  mask.filter = "none";
  mask.globalCompositeOperation = "source-in";
}

/** The hovering source makes a tight palm hotspot, a weak shoulder and blue-white source-facing finger edges. */
function drawSourceReflection(ctx: CanvasRenderingContext2D, source: PalmSource) {
  const level = Math.max(0, Math.min(1, source.level ?? 1));
  if (level <= 0) return;
  const indices = [0, 5, 9, 13, 17];
  const palm = [0, 1].map((axis) => indices.reduce((sum, i) => sum + source.landmarks[i][axis], 0) / 5) as P2;
  const contact = sourceContact(source), distance = Math.max(1, Math.hypot(palm[0] - source.at[0], palm[1] - source.at[1]));
  const gap = Math.hypot(contact[0] - source.at[0], contact[1] - source.at[1]);
  const contactGain = 1.1 - 0.65 * Math.min(1, gap / Math.max(1, 0.1 * source.scale));
  const at: P2 = [contact[0] + 0.06 * source.scale * (palm[0] - source.at[0]) / distance,
    contact[1] + 0.06 * source.scale * (palm[1] - source.at[1]) / distance];
  ctx.globalCompositeOperation = "screen";
  for (const [name, sigma, peak] of [["palm-source-reflection", 0.11, 1 - Math.exp(-contactGain * level)],
    ["palm-source-shoulder", 0.198, 0.035 * level]] as const) {
    const mask = handLayer(ctx, name, source.landmarks, source.scale);
    if (!mask) continue;
    sourceShadows(mask, source);
    const radius = 3.5 * sigma * source.scale;
    mask.save();
    mask.translate(...at);
    if (name === "palm-source-reflection") {
      mask.rotate(Math.atan2(palm[1] - source.at[1], palm[0] - source.at[0]));
      mask.scale(1.3, 0.85);
    }
    const pool = mask.createRadialGradient(0, 0, 0, 0, 0, radius);
    for (let i = 0; i <= 14; i++) {
      const u = i / 14, alpha = i === 14 ? 0 : peak * Math.exp(-0.5 * (3.5 * u) ** 2);
      pool.addColorStop(u, `rgba(185, 225, 255, ${alpha})`);
    }
    mask.fillStyle = pool;
    mask.fillRect(-ctx.canvas.width, -ctx.canvas.height, 2 * ctx.canvas.width, 2 * ctx.canvas.height);
    mask.restore();
    ctx.drawImage(mask.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }
  const nearestTip = TIPS.reduce((best, tip) => Math.hypot(source.landmarks[tip][0] - source.at[0],
    source.landmarks[tip][1] - source.at[1]) < Math.hypot(source.landmarks[best][0] - source.at[0],
    source.landmarks[best][1] - source.at[1]) ? tip : best, TIPS[0]);
  const tipGap = Math.hypot(source.landmarks[nearestTip][0] - source.at[0], source.landmarks[nearestTip][1] - source.at[1]);
  if (tipGap < 0.2 * source.scale) drawTipReflection(ctx, source.landmarks, source.scale, 0.55 * level, nearestTip, 0.12, 0.2);
  const coverage = handLayer(ctx, "palm-source-edge-mask", source.landmarks, source.scale);
  const facing = softLayer(ctx, "palm-source-edges");
  if (!facing || !coverage) return;
  facing.lineCap = "round";
  facing.lineWidth = 0.045 * source.scale;
  facing.filter = `blur(${0.008 * source.scale / SOFT_Q}px)`;
  const falloff = facing.createRadialGradient(source.at[0], source.at[1], 0, source.at[0], source.at[1], 0.8 * source.scale);
  falloff.addColorStop(0, `rgba(165, 215, 255, ${0.34 * (1 - Math.exp(-5.6 * level))})`);
  falloff.addColorStop(0.45, `rgba(165, 215, 255, ${0.34 * (1 - Math.exp(-5.6 * level * (1 - 0.45 ** 2) ** 2))})`);
  falloff.addColorStop(1, "rgba(165, 215, 255, 0)");
  facing.strokeStyle = falloff;
  const edges: P2[][] = [];
  for (const finger of FINGERS) for (let j = 0; j < finger.length - 1; j++) {
    const a = source.landmarks[finger[j]], b = source.landmarks[finger[j + 1]];
    if (Math.hypot((a[0] + b[0]) / 2 - source.at[0], (a[1] + b[1]) / 2 - source.at[1]) > 0.8 * source.scale) continue;
    const length = Math.max(1, Math.hypot(b[0] - a[0], b[1] - a[1]));
    const normal: P2 = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
    const facingDistance = normal[0] * (source.at[0] - (a[0] + b[0]) / 2) +
      normal[1] * (source.at[1] - (a[1] + b[1]) / 2);
    if (Math.abs(facingDistance) < 0.02 * source.scale) continue;
    const side = Math.sign(facingDistance);
    const points = [a, b].map((p): P2 => [p[0] + side * 0.072 * source.scale * normal[0],
      p[1] + side * 0.072 * source.scale * normal[1]]);
    edges.push(points);
    for (let step = 0; step < 16; step++) {
      const u = (step + 0.5) / 16, taper = Math.min(1, u / 0.15) * Math.min(1, (1 - u) / 0.15);
      facing.lineWidth = 0.045 * source.scale * taper;
      stroke(facing, [step / 16, (step + 1) / 16].map((at): P2 => [
        points[0][0] + at * (points[1][0] - points[0][0]), points[0][1] + at * (points[1][1] - points[0][1]),
      ]));
    }
  }
  facing.filter = "none";
  facing.globalCompositeOperation = "destination-in";
  facing.drawImage(coverage.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(facing.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  const specular = softLayer(ctx, "palm-source-specular");
  if (!specular) return;
  specular.lineCap = "round";
  specular.lineWidth = 0.012 * source.scale;
  specular.filter = `blur(${0.004 * source.scale / SOFT_Q}px)`;
  specular.strokeStyle = `rgba(4, 12, 19, ${level})`; // A weak blue specular return over the diffuse facing edge.
  for (const points of edges) for (let step = 0; step < 16; step++) {
    const u = (step + 0.5) / 16, taper = Math.min(1, u / 0.15) * Math.min(1, (1 - u) / 0.15);
    specular.lineWidth = 0.012 * source.scale * taper;
    stroke(specular, [step / 16, (step + 1) / 16].map((at): P2 => [
      points[0][0] + at * (points[1][0] - points[0][0]), points[0][1] + at * (points[1][1] - points[0][1]),
    ]));
  }
  specular.filter = "none";
  specular.globalCompositeOperation = "destination-in";
  specular.drawImage(coverage.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.globalCompositeOperation = "lighter";
  ctx.drawImage(specular.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

// ---- the ball over the palm
const BALL_R = [0.12, 0.5]; // hand sizes: its radius, from a point to one held long and loud
export const BALL_RISE = 0.2; // its growth (0-1) rises at most this much a frame: from a crackle to full over ~5 frames
const BALL_GROW = 6, BALL_SHRINK = 3; // frames: its charge follows the voice up this slowly, down this fast
const BALL_ON = [0.03, 0.015]; // it shows once the voice or its charge passes the first, goes once both are under
const BALL_FORMED = 0.08; // its charge at which its body is dense: under half of it, only its shell's crackle
const FIL_REACH = 0.7; // radii: a young ball's shell crackles this much further out than its body
const BALL_PULSE = 0.12; // a hit swells it by this share, gone over 3 frames
const BALL_TURN = 0.02; // radians a frame: it turns slowly (the shot's 0.05 at 24 frames a second, on a 60 Hz screen)
const BALL_LIFT = 150; // px at 1080: it hovers this far up the picture off the palm (on it, it was a sticker on skin)
export const SHELL = [6, 10]; // Sparse short filaments leave dark gaps through the volume instead of outlining a globe.
const SHELL_OUTLINE = 0.12, SHELL_SPAN = [0.38, 1.08]; // Rare limb tangencies; short open spans cannot close a rim.
const SHELL_INNER = 0.65; // Most filaments thread the volume rather than trace nested spheres inside it.
export const SHELL_SKIN = 0.05; // radii: a shell arc is held this close to its sphere, however jagged (thrown past it,
// its arcs lay on the palm like glowing veins)
export const GROUND = [0.1, 0.35, 0.35, 0.2]; // how often a side of it (the thumb's, the little finger's) throws 0, 1,
// 2 or 3 arcs to its fingertips when it strikes, at its full size, as many fewer as it is smaller, none once it has
// shrunk under GROUND_GONE of the largest it grew to: a discharge here and there (two or three a side every frame, both
// sides at once, read as a pair of wings; at 6-10, each with its dot, as a spider on white legs)
export const GROUND_GONE = 0.3;
export const GROUND_GAP = [1, 3]; // frames from a side's strike to its next: the two strike on their own, never on the
// same frame but on a hit
const GROUND_LIFE = [[1, 2, 3], [0.3, 0.4, 0.3]]; // frames an arc is kept (re-struck or dark between), how often each
const TIPS = [4, 8, 12, 16, 20]; // the landmarks they go to: the fingertips (to the joints and bases too, they ran down
// the skin to a dot mid-finger)
export const GROUND_BOW = [0.15, 0.35]; // shares of its chord: each bows through the air, its middle this far off the
// straight line, either way (bowed 30-110 px whatever its length, the short ones rose as arches over the fingertips)
const GROUND_LEAVE = 0.26; // radians: it leaves the ball's skin at most this far round from the point facing its tip
export const GROUND_TURN = 120; // degrees: its course turns at most this much in all (swing), or it is drawn again
export const LEADER = [10, 40, 10, 25, 0.75]; // (px at 1080, its range; degrees, its range; share): an arc is a run of
// straight steps this long, each this far off its course, to the other side from the last this share of the time, so
// it kinks 20-50 degrees between them (by midpoint displacement it was an evenly wiggling ribbon)
export const GROUND_FREE = 1.2, GROUND_ON = [0.3, 0.25]; // radii; share, hand sizes: out to the first from its middle
// an arc may pass over the hand; past it, but for where it lands, at most the second's share of it over the hand and
// none of that for longer than its hand sizes (it may cross in front of a finger, never run along the skin), or it
// is drawn again, at most GROUND_TRIES times
const GROUND_TRIES = 8;
export const LAND_IN = 12, LAND_NEAR = 25; // px at 1080: it comes down onto the finger from the air past its tip,
// landing this far inside the finger, at most LAND_NEAR from the fingertip's landmark
export const GROUND_PAD = [0.25, 1.1, 1.6, 0.08, 0.2]; // (chance a frame; radii, its range; bow, its range): now and
// then a short arc, for a frame, to a nearby finger pad or knuckle within this radial range.
const PADS = [5, 6, 7, 10, 11, 17];
const LAND = [2.5, 8]; // px at 1080: where it lands, a white-hot point this wide in a blue glow this wide, the skin
// under the glow taken towards cyan first (added over red skin, a blue glow read as a pink halo)
const LAND_RGB = "111, 168, 255"; // #6fa8ff
const SHELL_GONE = [0.45, 0.3]; // shares of the largest it grew to: shrinking, its shell fades out between these, and
// with its body once it has been a ball (its crackle crawled over the fingers with no source)
export const BALL_LIGHT = [0.75 * 1.3 / 1.9, 0.45, 40, 1, 3, 0.9]; // its light round it (cuesync.lightning's BALL_LIGHT): (alpha on
// the hand by its edge, on the room; px at 1080 its middle is over the palm, radii the wall is behind it; radii it is
// gone by; radii of its dense middle, which hides what it lights there): from its middle as from a ball of its
// radius, R^2 / (d^2 + h^2) at most 1, a cool white on the hand and a blue-white on the room (spread over the frame, a
// deep blue, its light tinted the hand lavender and graded the room)
const BALL_LIGHT_RGB = ["225, 240, 255", "150, 190, 255"]; // (on the hand, on the room)
export const FILL = [0.0528, 2.5, 1.5]; // (alpha a radius out, radii, radii): and on the hand, a softer light as 1 / r
// of the distance from its middle, coming up from its edge to the last radii and gone smoothly from the second to its
// reach: a light among the fingers, not a spot on the palm (cuesync.lightning's FILL)
export const POOL_PALM = [0.096, 0.06, 3]; // (alpha at the palm's point, a radius from it, radii it is gone by): its light
// pooled on the palm under it, added (about 25 levels a radius out; screened alone, the palm under it hardly lit)
const HALO: [number, number, string] = [0.025, 0.62, "20, 90, 255"]; // Gain, radius scale and colour: a small faint haze
// preserves the core's depth instead of surrounding every filament with a broad uniform blue cloud.
const PLASMA_TURNS = 0.65; // Turns per second of the broad ridges seen through the translucent core.
const PLASMA_LAYERS = [[1, 1, 0.2, 1, 0.3], [-0.75, 0.6, -0.5, 1, 0.2]]; // (rate, share, axis x, y, z): its two
// layers of filaments, each turning about its own axis through its middle; seen on the sphere, front and (fainter)
// back, so they wrap it (drawn as a flat tile turned in the picture, they read as camouflage on a disc)
export const PLASMA_PX = 112; // the plasma is worked out this many px across, drawn up to the ball's size softened
// PLASMA_SOFT of its radius (as fibrous as the shot's), and faded out over its limb from PLASMA_LIMB radii
const PLASMA_SOFT = 0.02, PLASMA_LIMB = 0.85;

/** An arc of the ball's shell: its points on the unit sphere (3D, z towards the camera), turned with the ball, and
 * how bright (an inner one dimmer); struck and re-struck as a channel is. */
export interface Shell { pts: P3[]; inten: number; left: number; flash: number }

/** An arc jumping from the ball to a finger: the side of it it goes to (-1 the picture's left, 1 its right), which of
 * that side's available fingertips (``k``-th), or a finger's ``pad``; how far it bows (a share of its chord,
 * to its left as it runs, or its right under 0) and how far round from the point facing its fingertip it leaves the
 * ball's skin (radians), struck and re-struck as a channel is; ``seed``: its steps (and where it looks for another way
 * when this one lies on the hand); ``shape``: its points from (0, 0) to (1, 0) once drawn, kept while it lives. */
export interface Jump { side: number; k: number; pad: boolean; bow: number; leave: number; left: number; flash: number;
  seed: number; to?: number; shape?: P2[] }

/** The ball's state between frames: ``top``, its largest radius (hand sizes) and ``most``, how grown it has been at
 * most, while it shows; ``fade``: how much of its shell shows as it dies; ``due``: the frame each side of it (the
 * picture's left, its right) may strike its arcs again. */
export interface Ball { charge: number; grown: number; born: boolean; pulse: number; flare: number; level: number;
  shell: Shell[]; jumps: Jump[]; turn: number; frame: number; top: number; most: number; fade: number; due: P2 }

/** A normal deviate from ``rand`` (Box-Muller). */
function gauss(rand: () => number): number {
  return Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-12))) * Math.cos(2 * Math.PI * rand());
}

function unit3(v: P3): P3 {
  const l = Math.hypot(...v) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross3(a: P3, b: P3): P3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** A jagged current on one unit sphere, with z towards the camera. Its plane bows the projected path around the
 * same volume as its companions; small tangent kinks preserve the sphere instead of making a flat drawing. */
export function sphereCurrent(angle: number, tilt: number, start: number, end: number, count: number, seed: number): P3[] {
  const a: P3 = [Math.cos(angle), Math.sin(angle), 0];
  const b: P3 = [-Math.sin(angle) * Math.sin(tilt), Math.cos(angle) * Math.sin(tilt), Math.cos(tilt)];
  const normal = cross3(a, b), rand = rng(seed), points: P3[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / Math.max(1, count - 1), envelope = Math.sin(Math.PI * t);
    const phi = start + (end - start) * t + 0.025 * (rand() - 0.5) * envelope;
    const kink = 0.06 * (rand() - 0.5) * envelope;
    points.push(unit3(a.map((v, k) => v * Math.sin(phi) + b[k] * Math.cos(phi) + normal[k] * kink) as P3));
  }
  return points;
}

/** The two resolved foreground paths also supply the escaping roots, so a free discharge cannot detach from them. */
function ballCurrents(b: Ball): P3[][] {
  const seconds = b.frame / 60, seed = 15485863 + Math.floor(b.frame / 2);
  return [sphereCurrent(0.08 + 0.035 * Math.sin(seconds * 0.7), 0.42, -1.28, 1.28, 45, seed),
    sphereCurrent(-0.88 + 0.035 * Math.cos(seconds * 0.9), -0.25, -1.48, 1.48, 45, seed + 32452843)];
}

/** Short discharges emerge from inside the luminous boundary and stop in the air. Nearby finger surfaces steer
 * them; the first foreground finger truncates each path with an air gap, or removes it before it can escape. */
export function ballStreamers(b: Ball, c: P2, radius: number, lm: P2[] | null, scale: number): P2[][] {
  if (radius <= 0) return [];
  const rand = rng(49979687 + Math.floor(b.frame / 2));
  const apart = (a: number, b: number) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
  const candidates: { angle: number; distance: number; finger: number }[] = [];
  if (lm?.length === 21 && scale > 0) for (let f = 0; f < FINGERS.length; f++) {
    const finger = FINGERS[f];
    for (let j = 1; j < finger.length; j++) for (let k = 0; k <= 6; k++) {
      const u = k / 6, a = lm[finger[j - 1]], z = lm[finger[j]];
      const x = a[0] + u * (z[0] - a[0]) - c[0], y = a[1] + u * (z[1] - a[1]) - c[1];
      const distance = (Math.hypot(x, y) - 0.075 * scale) / radius;
      if (distance >= 1.21 && distance <= 1.6) candidates.push({ angle: Math.atan2(y, x), distance, finger: f });
    }
  }
  candidates.sort((a, b) => a.distance - b.distance);
  const directions: { angle: number; length: number; finger: number }[] = [];
  for (const candidate of candidates) {
    if (directions.some((p) => p.finger === candidate.finger || apart(p.angle, candidate.angle) < 0.75)) continue;
    directions.push({ angle: candidate.angle, length: Math.min(0.34, candidate.distance - 1.05), finger: candidate.finger });
    if (directions.length === 2) break;
  }
  for (const fallback of [-0.94, 0.25, 2.3, 3.0]) {
    if (directions.length === 3) break;
    const angle = fallback + 0.12 * (rand() - 0.5);
    if (directions.some((p) => apart(p.angle, angle) < 0.75)) continue;
    directions.push({ angle, length: 0.16 + 0.18 * rand(), finger: -1 });
  }
  const clear = (p: P2) => !lm || lm.length !== 21 || FINGERS.every((finger) => finger.slice(1).every((joint, i) =>
    toSegment(p, lm[finger[i]], lm[joint]) > 0.11 * scale + 0.03 * radius));
  const roots = ballCurrents(b).flatMap((points) => [points[3], points[points.length - 4]])
    .map(([x, y]) => ({ at: [c[0] + radius * x, c[1] + radius * y] as P2, angle: Math.atan2(y, x) }));
  const used = new Set<number>();
  return directions.flatMap(({ angle, length }) => {
    const root = roots.map((p, i) => ({ i, distance: apart(p.angle, angle) })).sort((a, b) => a.distance - b.distance)[0];
    if (root.distance > 0.55 || used.has(root.i)) return [];
    const start = roots[root.i].at;
    const end: P2 = [c[0] + radius * (1 + length) * Math.cos(angle), c[1] + radius * (1 + length) * Math.sin(angle)];
    const shape = displace(start, end, rand, 0.28, 3);
    if (!clear(start)) return [];
    const retained: P2[] = [start];
    const emit = () => { used.add(root.i); return [retained]; };
    for (let i = 1; i < shape.length; i++) {
      const a = shape[i - 1], z = shape[i];
      const steps = Math.max(1, Math.ceil(Math.hypot(z[0] - a[0], z[1] - a[1]) / (0.015 * radius)));
      for (let j = 1; j <= steps; j++) {
        const point = (u: number): P2 => [a[0] + u * (z[0] - a[0]), a[1] + u * (z[1] - a[1])];
        if (clear(point(j / steps))) continue;
        let lo = (j - 1) / steps, hi = j / steps;
        for (let search = 0; search < 12; search++) {
          const mid = (lo + hi) / 2;
          if (clear(point(mid))) lo = mid; else hi = mid;
        }
        retained.push(point(lo));
        return Math.hypot(retained.at(-1)![0] - c[0], retained.at(-1)![1] - c[1]) > 1.02 * radius ? emit() : [];
      }
      retained.push(z);
    }
    return emit();
  });
}

/** How much of a body a ball has (0-1): none until its charge (and the voice) pass half BALL_FORMED. */
export function dense(level: number, charge: number): number {
  return Math.max(0, Math.min(1, (2 * (0.3 * level + 0.7 * charge)) / BALL_FORMED - 1));
}

/** Midpoint displacement perpendicular to each segment in 3D, down to a .025-radius segment. */
function displace3(a: P3, b: P3, rand: () => number): P3[] {
  const span = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const levels = Math.max(0, Math.min(10, Math.ceil(Math.log2(Math.max(span / 0.025, 1)))));
  let pts: P3[] = [a, b];
  for (let level = 0; level < levels; level++) {
    const out: P3[] = [pts[0]], rough = 0.28 * (0.45 + 0.55 * Math.min(1, level / 3));
    for (let i = 0; i + 1 < pts.length; i++) {
      const p = pts[i], q = pts[i + 1], d: P3 = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
      const length = Math.hypot(...d), direction = unit3(d), noise: P3 = [gauss(rand), gauss(rand), gauss(rand)];
      const along = noise[0] * direction[0] + noise[1] * direction[1] + noise[2] * direction[2];
      const middle = [0, 1, 2].map((axis) => (p[axis] + q[axis]) / 2 +
        (noise[axis] - along * direction[axis]) * rough * length) as P3;
      out.push(middle, q);
    }
    pts = out;
  }
  return pts.map((p): P3 => { const k = Math.min(1, 0.98 / (Math.hypot(...p) || 1)); return [p[0] * k, p[1] * k, p[2] * k]; });
}

/** Reject projected loops and tight knots by measuring their course at thirteen equal distances. */
function shellKnotted(pts: P3[]): boolean {
  const cumulative = [0];
  for (let i = 1; i < pts.length; i++) cumulative.push(cumulative[i - 1] +
    Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = cumulative[cumulative.length - 1];
  if (total < 1e-9) return true;
  const sampled: P2[] = [];
  for (let i = 0, segment = 0; i <= 12; i++) {
    const at = total * i / 12;
    while (segment < pts.length - 2 && cumulative[segment + 1] < at) segment++;
    const length = cumulative[segment + 1] - cumulative[segment], u = length ? (at - cumulative[segment]) / length : 0;
    sampled.push([pts[segment][0] + u * (pts[segment + 1][0] - pts[segment][0]),
      pts[segment][1] + u * (pts[segment + 1][1] - pts[segment][1])]);
  }
  const vectors = sampled.slice(1).map((p, i): P2 => [p[0] - sampled[i][0], p[1] - sampled[i][1]]);
  const courseLength = vectors.reduce((sum, p) => sum + Math.hypot(...p), 0);
  if (Math.hypot(sampled[12][0] - sampled[0][0], sampled[12][1] - sampled[0][1]) < 0.45 * courseLength) return true;
  const angles = vectors.map(([x, y]) => Math.atan2(y, x));
  const turns = angles.slice(1).map((angle, i) => ((angle - angles[i] + 3 * Math.PI) % (2 * Math.PI)) - Math.PI);
  return turns.some((_, i) => i + 5 <= turns.length &&
    Math.abs(turns.slice(i, i + 5).reduce((sum, turn) => sum + turn, 0)) > 200 * Math.PI / 180);
}

/** Most filaments cross the charge at changing radii; the rest remain short open paths near its skin. */
function shellArc(rand: () => number): P3[] {
  const inner = rand() < SHELL_INNER, outline = !inner && rand() < SHELL_OUTLINE;
  let pts: P3[] = [];
  for (let attempt = 0; attempt < 6; attempt++) {
    if (inner) {
      const startDirection = unit3([gauss(rand), gauss(rand), gauss(rand)]), radius = 0.65 + 0.3 * rand();
      const a = startDirection.map((v) => v * radius) as P3;
      const direction = unit3([gauss(rand), gauss(rand), gauss(rand)]), reach = 0.25 + 0.3 * rand();
      const b = direction.map((v, axis) => -0.45 * a[axis] + v * reach) as P3;
      pts = displace3(a, b, rand);
    } else {
      const nrm = outline ? unit3([0.25 * gauss(rand), 0.25 * gauss(rand), 1]) : unit3([gauss(rand), gauss(rand), gauss(rand)]);
      const e1 = unit3(cross3(nrm, [gauss(rand), gauss(rand), gauss(rand)])), e2 = cross3(nrm, e1);
      const start = 2 * Math.PI * rand(), span = SHELL_SPAN[0] + (SHELL_SPAN[1] - SHELL_SPAN[0]) * rand();
      pts = [];
      let wobble = 0, velocity = 0;
      for (let i = 0; i < 33; i++) {
        const phi = start + span * i / 32;
        velocity = 0.6 * velocity + 0.07 * gauss(rand);
        wobble = Math.max(-0.28, Math.min(0.28, wobble + velocity));
        const p = [0, 1, 2].map((axis) => Math.cos(phi) * e1[axis] + Math.sin(phi) * e2[axis] + wobble * nrm[axis]) as P3;
        const length = Math.hypot(...p) || 1, held = Math.max(1 - SHELL_SKIN, Math.min(1 + SHELL_SKIN, length)) / length;
        pts.push([p[0] * held, p[1] * held, p[2] * held]);
      }
    }
    if (!shellKnotted(pts)) break;
  }
  return pts;
}

/** One frame of the ball for a voice ``level`` (0-1), or null when it is gone: its charge follows the voice, its body
 * forms at its least size and grows at most BALL_RISE a frame towards the square root of its charge, a hit swells it
 * and flares its middle; its shell (SHELL arcs) is struck and re-struck as channels are, every frame or two, turning
 * slowly with it; dying, its shell fades out with it (SHELL_GONE). Once it has a body its arcs to the fingers strike a
 * side at a time (GROUND_GAP; both on a hit, at least one each), GROUND's draw of them, as many fewer as it is smaller,
 * each kept GROUND_LIFE's frames, re-struck or dark; and now and then a short one to a finger's pad (GROUND_PAD). */
export function ballStep(b: Ball | null, level: number, hit: boolean, frame: number): Ball | null {
  const prev = b?.charge ?? 0;
  const charge = prev + (level - prev) * (1 - Math.exp(-1 / (level > prev ? BALL_GROW : BALL_SHRINK)));
  if (Math.max(level, charge) < BALL_ON[b ? 1 : 0]) return null;
  const born = (b?.born ?? false) || dense(level, charge) > 0;
  const grown = b?.born ? Math.min(Math.sqrt(charge), b.grown + BALL_RISE) : 0;
  const pulse = hit ? BALL_PULSE : (b?.pulse ?? 0) * 0.6;
  const size = sized(grown, pulse);
  const again = b && size > ballRadius(b, 1) && size < SHELL_GONE[0] * b.top; // (grown again from all but nothing)
  const top = Math.max(again ? 0 : b?.top ?? 0, size), most = Math.max(again ? 0 : b?.most ?? 0, grown);
  const body = born ? dense(level, charge) : 0;
  const fade = Math.max(0, Math.min(1, (size / top - SHELL_GONE[1]) / (SHELL_GONE[0] - SHELL_GONE[1]))) *
    (most >= 0.5 ? body : 1);
  const restroke = (rand: () => number) => (rand() < RESTROKE[0] + RESTROKE[1] * level ? 0.55 + 0.35 * rand()
    : 0.1 * rand());
  const shell: Shell[] = [];
  for (let i = 0; i < Math.round(SHELL[0] + (SHELL[1] - SHELL[0]) * charge); i++) {
    const old = b?.shell[i], rand = rng((i + 1) * 104729 + frame);
    if (old && old.left > 0 && !hit) { shell.push({ ...old, left: old.left - 1, flash: restroke(rand) }); continue; }
    const pts = shellArc(rand);
    shell.push({ pts, inten: 1, left: life(rand) - 1, flash: 1 });
  }
  const rand = rng(frame * 15485863 + 7);
  const share = body > 0 && size >= GROUND_GONE * top ? Math.min(1, size / BALL_R[1]) : 0;
  const due: P2 = share > 0 && b?.due ? [...b.due] : [frame, frame];
  let jumps: Jump[] = share > 0 && b ? b.jumps.filter((j) => j.left > 0 && !j.pad)
    .map((j) => ({ ...j, left: j.left - 1, flash: restroke(rand) })) : [];
  if (share > 0) {
    const ready = [0, 1].filter((i) => due[i] <= frame);
    const now = hit ? [0, 1] : ready.length === 2 ? [due[0] < due[1] || (due[0] === due[1] && rand() < 0.5) ? 0 : 1]
      : ready;
    for (const i of now) {
      due[i] = frame + GROUND_GAP[0] + Math.floor(rand() * (GROUND_GAP[1] - GROUND_GAP[0] + 1));
      const side = 2 * i - 1;
      jumps = jumps.filter((j) => j.side !== side);
      let pick = rand(), want = 0;
      while (want < GROUND.length - 1 && pick >= GROUND[want]) pick -= GROUND[want++];
      want = Math.max(hit ? 1 : 0, Math.round(want * share));
      for (let k = 0; k < want; k++) jumps.push(newJump(rand, side, k, false, lifeOf(rand)));
    }
    if (rand() < GROUND_PAD[0]) jumps.push(newJump(rand, 0, 0, true, 1));
  }
  return { charge, grown, born, pulse, flare: hit ? 1 : 0, level, shell, jumps, turn: (b?.turn ?? 0) + BALL_TURN, frame,
    top, most, fade, due };
}

/** GROUND_LIFE's draw of how many frames an arc is kept. */
function lifeOf(rand: () => number): number {
  const [n, p] = GROUND_LIFE;
  let r = rand(), i = 0;
  while (i < n.length - 1 && r >= p[i]) r -= p[i++];
  return n[i];
}

/** A new arc to the fingers: to the ``k``-th fingertip on ``side``, or a finger's ``pad``, kept ``left`` frames. */
function newJump(rand: () => number, side: number, k: number, pad: boolean, left: number): Jump {
  const [b0, b1] = pad ? GROUND_PAD.slice(3) : GROUND_BOW;
  return { side, k, pad, bow: (rand() < 0.5 ? -1 : 1) * (b0 + (b1 - b0) * rand()),
    leave: (2 * rand() - 1) * GROUND_LEAVE, left: left - 1, flash: 1, seed: Math.floor(rand() * 2 ** 31) };
}

/** A ball's radius in hand sizes, ``grown`` (0-1) and swollen by ``pulse``. */
function sized(grown: number, pulse: number): number {
  return (BALL_R[0] + (BALL_R[1] - BALL_R[0]) * grown * grown * (3 - 2 * grown)) * (1 + pulse);
}

/** The ball's radius in px for a hand ``scale`` px across. */
export function ballRadius(b: Ball, scale: number): number {
  return scale * sized(b.grown, b.pulse);
}

/** The charge stays close to its supporting palm and moves into the tracked thumb opening so its lower side clears
 * the fingers. The live renderer's projected lift is screen-up; its perpendicular thumb direction is horizontal.
 * A nearly aligned thumb contributes less shift so tracking jitter cannot flip the charge across the palm. */
export function ballAt(palm: P2, px: number, radius = 0, landmarks: P2[] | null = null): P2 {
  const projected = landmarks?.length === 21 && Number.isFinite(landmarks[4][0]) ? landmarks[4][0] - palm[0] : 0;
  const toward = radius > 0 ? projected / Math.max(Math.abs(projected), 0.35 * radius) : 0;
  return [palm[0] + 0.5 * radius * toward, palm[1] - BALL_LIFT * px - 0.55 * radius];
}

/** Translucent material from the irregular core outwards, [radii, r, g, b, alpha]. The edge fades without a bright
 * rim and the camera remains visible through its middle; filaments carry the hottest light. */
export const BODY_STOPS: [number, number, number, number, number][] = [
  [0, 135, 202, 255, 0.1543], [0.15, 142, 207, 255, 0.1693], [0.3, 125, 195, 255, 0.1543],
  [0.45, 108, 182, 255, 0.1307], [0.6, 95, 164, 255, 0.0917], [0.75, 80, 150, 255, 0.0433],
  [0.9, 70, 137, 255, 0.0121], [1.15, 70, 137, 255, 0],
];

const plasmaCanvas = new WeakMap<object, Ctx2D>();
const hazeCanvas = new WeakMap<object, Ctx2D>();
const ballHands = new WeakMap<object, P2[]>(); // drawBallLight supplies the current hand before drawBall runs.
let plasmaField: Float32Array | null = null;
const FIELD_W = 128, FIELD_H = 64; // the ridged noise over longitude and latitude
const PLASMA_EXTENT = 1.15; // Radii: leave room beyond the irregular volume for its boundary to fade before the tile ends.

/** Ridged noise over the sphere's longitude (wrapping round it) and latitude, made once: two octaves of smooth noise,
 * their zero crossings bright (filaments), scaled to about -1..1. */
function plasmaNoise(): Float32Array {
  if (plasmaField) return plasmaField;
  const rand = rng(17);
  const field = new Float32Array(FIELD_W * FIELD_H);
  for (const [cells, weight] of [[6, 0.6], [13, 0.4]]) {
    const rows = Math.max(2, cells >> 1);
    const g = Array.from({ length: cells * (rows + 1) }, () => rand() * 2 - 1);
    const s = (q: number) => q * q * (3 - 2 * q);
    const at = (i: number, j: number) => g[(Math.min(j, rows) * cells) + (i % cells)];
    for (let y = 0; y < FIELD_H; y++) for (let x = 0; x < FIELD_W; x++) {
      const fx = (x / FIELD_W) * cells, fy = (y / (FIELD_H - 1)) * rows, x0 = Math.floor(fx), y0 = Math.floor(fy);
      const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * s(fx - x0);
      const bot = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * s(fx - x0);
      field[y * FIELD_W + x] += weight * (top + (bot - top) * s(fy - y0));
    }
  }
  for (let i = 0; i < field.length; i++) field[i] = 1.6 * (1 - Math.min(1, Math.abs(field[i]) * 2)) ** 2 - 0.5; // (soft
  // ridges: half as sharp again, drawn up three times, they were crumpled foil)
  return (plasmaField = field);
}

/** PLASMA_LAYERS' turns at ``seconds``: each one's share, axis (unit) and the cosine and sine it is turned by. */
export function plasmaTurns(seconds: number): { share: number; k: P3; c: number; sn: number }[] {
  return PLASMA_LAYERS.map(([rate, share, ax, ay, az]) => {
    const l = Math.hypot(ax, ay, az), a = 2 * Math.PI * PLASMA_TURNS * rate * seconds;
    return { share, k: [ax / l, ay / l, az / l] as P3, c: Math.cos(a), sn: Math.sin(a) };
  });
}

/** The ball's plasma where the picture sees its point (x, y) (its radii from its middle, x right, y down), -1..1, for
 * ``turns`` (plasmaTurns): each layer's filaments read on the sphere through its normal there (u its longitude, v its
 * latitude), turned about the layer's own axis, front and (0.6 as much, seen through it) back, faded out over its
 * limb; 0 off it. */
export function plasmaAt(x: number, y: number, turns: { share: number; k: P3; c: number; sn: number }[]): number {
  const r2 = x * x + y * y;
  if (r2 >= 1) return 0;
  const field = plasmaNoise(), z = Math.sqrt(1 - r2);
  const sample = (p: P3) => {
    const u = (Math.atan2(p[0], p[2]) / (2 * Math.PI) + 0.5) * FIELD_W;
    const v = (Math.acos(Math.max(-1, Math.min(1, p[1]))) / Math.PI) * (FIELD_H - 1);
    const x0 = Math.floor(u), y0 = Math.min(FIELD_H - 2, Math.floor(v)), fu = u - x0, fv = v - y0;
    const at = (i: number, j: number) => field[j * FIELD_W + (((i % FIELD_W) + FIELD_W) % FIELD_W)];
    const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fu;
    return top + (at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fu - top) * fv;
  };
  let v = 0;
  for (const { share, k, c, sn } of turns) for (const [side, w] of [[1, 1], [-1, 0.6]]) {
    const p: P3 = [x, y, side * z]; // (turned about k: Rodrigues)
    const kd = k[0] * p[0] + k[1] * p[1] + k[2] * p[2], kx = cross3(k, p);
    v += share * w * sample([p[0] * c + kx[0] * sn + k[0] * kd * (1 - c), p[1] * c + kx[1] * sn + k[1] * kd * (1 - c),
      p[2] * c + kx[2] * sn + k[2] * kd * (1 - c)]);
  }
  return Math.max(-1, Math.min(1, v / 2)) * Math.min(1, (1 - Math.sqrt(r2)) / (1 - PLASMA_LIMB));
}

/** Low-pass a field in units of the ball's radius, so density and filament support share one spatial scale. */
function smoothBulk(field: Float32Array, n: number, share = 0.09): Float32Array {
  const sigma = share * n / (2 * PLASMA_EXTENT), radius = Math.ceil(3 * sigma);
  const kernel = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-0.5 * ((i - radius) / sigma) ** 2));
  const sum = kernel.reduce((a, b) => a + b);
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const across = new Float32Array(n * n), out = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let value = 0;
    for (let k = -radius; k <= radius; k++) value += field[y * n + Math.max(0, Math.min(n - 1, x + k))] * kernel[k + radius];
    across[y * n + x] = value;
  }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let value = 0;
    for (let k = -radius; k <= radius; k++) value += across[Math.max(0, Math.min(n - 1, y + k)) * n + x] * kernel[k + radius];
    out[y * n + x] = value;
  }
  return out;
}

/** Projected front channels support nearby emission. Bilinear raster samples stay independent of vertex density;
 * a .13-radius blur and normalized .7-power falloff keep the interior open between disconnected discharges. */
export function filamentSupport(lines: [Channel, number][], center: P2, R: number): Float32Array {
  const n = PLASMA_PX, field = new Float32Array(n * n);
  if (R <= 0 || !lines.length) return field;
  const pixel = (p: P2): P2 => [(0.5 + (p[0] - center[0]) / (2 * PLASMA_EXTENT * R)) * n - 0.5,
    (0.5 + (p[1] - center[1]) / (2 * PLASMA_EXTENT * R)) * n - 0.5];
  for (const [line, gain] of lines) for (let j = 1; j < line.pts.length; j++) {
    const a = pixel(line.pts[j - 1]), b = pixel(line.pts[j]);
    const steps = Math.max(1, Math.ceil(2 * Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]))));
    for (let i = 0; i <= steps; i++) {
      const x = a[0] + i * (b[0] - a[0]) / steps, y = a[1] + i * (b[1] - a[1]) / steps;
      const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        if (ix + dx < 0 || iy + dy < 0 || ix + dx >= n || iy + dy >= n) continue;
        const at = (iy + dy) * n + ix + dx;
        field[at] = Math.max(field[at], line.inten * gain * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy));
      }
    }
  }
  const blurred = smoothBulk(field, n, 0.13);
  let peak = 0.01;
  for (const value of blurred) peak = Math.max(peak, value);
  for (let i = 0; i < blurred.length; i++) blurred[i] = Math.min(1, blurred[i] / peak) ** 0.7;
  return blurred;
}

/** A small reusable texture alternates between scattering coverage and luminous material within one draw. */
function volumeTexture(ctx: CanvasRenderingContext2D): Ctx2D | null {
  const main = ctx.canvas as HTMLCanvasElement | undefined;
  if (!main?.width || typeof ctx.drawImage !== "function") return null;
  let texture = plasmaCanvas.get(main);
  if (!texture) {
    const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(PLASMA_PX, PLASMA_PX)
      : Object.assign(document.createElement("canvas"), { width: PLASMA_PX, height: PLASMA_PX });
    const made = canvas.getContext("2d") as Ctx2D | null;
    if (!made) return null;
    plasmaCanvas.set(main, (texture = made));
  }
  return texture;
}

/** Chord-integrated emission and three wandering hot regions make a luminous interior for the front arcs to cross.
 * Smoothed plasma modulates its bulk density; the irregular boundary remains faint and has no enclosing rim. */
function plasmaLayer(ctx: CanvasRenderingContext2D, seconds: number, amount: number,
  hand?: { landmarks: P2[]; scale: number; center: P2; radius: number }, support?: Float32Array): Ctx2D | null {
  const t = volumeTexture(ctx);
  if (!t) return null;
  const n = PLASMA_PX, img = t.createImageData(n, n), turns = plasmaTurns(seconds), phase = 1.7 * seconds;
  const swirl = new Float32Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = (((i + 0.5) / n) * 2 - 1) * PLASMA_EXTENT, y = (((j + 0.5) / n) * 2 - 1) * PLASMA_EXTENT;
    const phi = Math.atan2(y, x), contour = 1 + 0.06 * Math.sin(2 * phi + 0.3 * phase) +
      0.025 * Math.sin(5 * phi - 0.2 * phase);
    const r = Math.hypot(x, y) / contour;
    swirl[j * n + i] = plasmaAt(0.78 * x + 0.08 * Math.sin(phase), 0.8 * y + 0.06 * Math.cos(phase), turns) *
      Math.max(0, Math.min(1, (1.12 - r) / 0.3));
  }
  const bulk = smoothBulk(swirl, n);
  const hotRegions = [[-0.2, -0.08, 0.3, 0.23], [0.18, 0.14, 0.3, 0.23], [0.04, 0.28, 0.25, 0.19]]
    .map(([x, y, sx, sy], i) => [x + 0.07 * Math.sin(1.3 * phase + 2.1 * i),
      y + 0.07 * Math.cos(1.1 * phase + 1.7 * i), sx, sy]);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = (((i + 0.5) / n) * 2 - 1) * PLASMA_EXTENT, y = (((j + 0.5) / n) * 2 - 1) * PLASMA_EXTENT;
    const phi = Math.atan2(y, x), contour = 1 + 0.06 * Math.sin(2 * phi + 0.3 * phase) +
      0.025 * Math.sin(5 * phi - 0.2 * phase);
    const r = Math.hypot(x, y) / contour;
    const k = BODY_STOPS.findIndex(([at]) => at >= r);
    if (k < 0) continue;
    const a = BODY_STOPS[Math.max(0, k - 1)], b = BODY_STOPS[k];
    const u = b[0] > a[0] ? (r - a[0]) / (b[0] - a[0]) : 0;
    const rgb = [1, 2, 3].map((v) => a[v] + (b[v] - a[v]) * u);
    const s = bulk[j * n + i], chord = Math.sqrt(Math.max(1 - (r / 1.04) ** 2, 0) + 0.025) *
      Math.exp(-(Math.max(r - 0.82, 0) ** 2) / 0.08);
    const edge = Math.max(0, Math.min(1, (r - 0.55) / 0.3));
    const open = 1 - 0.4 * Math.exp(-(((phi - 2.1) / 0.65) ** 2)) * edge;
    const cloud = 0.5 + 0.5 * Math.sin(3.1 * x + 1.6 * y + 0.4 * phase) * Math.sin(2.7 * y - 1.3 * x - 0.3 * phase);
    const density = 0.38 + 0.27 * cloud + 0.54 * Math.max(s + 0.1, 0);
    const near = support?.[j * n + i] ?? 1;
    const body = 0.85 * chord * density * (1 - 0.25 * edge) * open * Math.max(1 - (r / 1.04) ** 2, 0) ** 0.65 *
      (0.4 + 0.6 * near);
    let hotShape = 0;
    for (const [ox, oy, sx, sy] of hotRegions) hotShape += 0.5 * Math.exp(-0.5 * (((x - ox) / sx) ** 2 + ((y - oy) / sy) ** 2));
    const hot = 0.78 * hotShape * chord * (0.5 + 0.8 * Math.max(s, -0.45)) * open * (0.5 + 0.5 * near);
    const energy = body + hot;
    const cover = hand ? fingerCoverage([hand.center[0] + x * hand.radius, hand.center[1] + y * hand.radius],
      hand.landmarks, hand.scale) : 0;
    const alpha = (1 - Math.exp(-0.45 * energy)) * amount * (1 - 0.96 * cover);
    const o = 4 * (j * n + i);
    img.data.set([...rgb.map((v, axis) => Math.round((body * v + hot * [182, 220, 255][axis]) / Math.max(energy, 1e-9))),
      Math.round(255 * Math.min(1, alpha))], o);
  }
  t.putImageData(img, 0, 0);
  return t;
}

/** Low cloud contrast breaks the faint haze's radial symmetry without surrounding the ball with a blue cap. */
function cloudHaze(ctx: CanvasRenderingContext2D, seconds: number, amount: number): Ctx2D | null {
  const main = ctx.canvas as HTMLCanvasElement | undefined;
  if (!main?.width || typeof ctx.drawImage !== "function") return null;
  const n = 64;
  let texture = hazeCanvas.get(main);
  if (!texture) {
    const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(n, n)
      : Object.assign(document.createElement("canvas"), { width: n, height: n });
    const made = canvas.getContext("2d") as Ctx2D | null;
    if (!made) return null;
    hazeCanvas.set(main, (texture = made));
  }
  const [gain, sigma, rgb] = HALO, extent = 3.5 * sigma, phase = 1.7 * seconds;
  const color = rgb.split(",").map(Number), pixels = texture.createImageData(n, n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = (2 * (i + 0.5) / n - 1) * extent, y = (2 * (j + 0.5) / n - 1) * extent;
    const cloud = 0.5 + 0.5 * Math.sin(3.1 * x + 1.6 * y + 0.4 * phase) * Math.sin(2.7 * y - 1.3 * x - 0.3 * phase);
    const alpha = 0.3 * gain * amount * Math.exp(-0.5 * (x * x + y * y) / (sigma * sigma)) * (0.55 + 0.45 * cloud);
    pixels.data.set([...color, Math.round(255 * alpha)], 4 * (j * n + i));
  }
  texture.putImageData(pixels, 0, 0);
  return texture;
}

/** Distributed inner emission sits over the attenuated rear channels and under the crisp front channels. */
function drawBallBulk(ctx: CanvasRenderingContext2D, b: Ball, c: P2, R: number, px: number, body: number,
  lm: P2[] | undefined, scale: number, support: Float32Array) {
  ctx.save();
  if (body > 0) {
    const scatter = softLayer(ctx, "ball-scatter");
    if (scatter && typeof ctx.drawImage === "function") {
      scatter.filter = `blur(${0.20 * R / SOFT_Q}px)`;
      scatter.drawImage(ctx.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
      scatter.filter = "none";
      scatter.globalCompositeOperation = "destination-in";
      const mask = volumeTexture(ctx);
      if (mask) {
        const n = PLASMA_PX, pixels = mask.createImageData(n, n);
        for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
          const x = (2 * (i + 0.5) / n - 1) * PLASMA_EXTENT, y = (2 * (j + 0.5) / n - 1) * PLASMA_EXTENT;
          const tau = 2.5 * Math.exp(-0.5 * (x * x + y * y) / 0.37 ** 2) * (0.25 + 0.75 * support[j * n + i]);
          pixels.data.set([255, 255, 255, Math.round(255 * body * (1 - Math.exp(-tau)))], 4 * (j * n + i));
        }
        mask.putImageData(pixels, 0, 0);
        scatter.drawImage(mask.canvas, c[0] - PLASMA_EXTENT * R, c[1] - PLASMA_EXTENT * R,
          2 * PLASMA_EXTENT * R, 2 * PLASMA_EXTENT * R);
      }
      ctx.save();
      if (lm) clipFingers(ctx, lm, scale);
      ctx.globalCompositeOperation = "source-over";
      ctx.drawImage(scatter.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.restore();
    }
    const plasma = plasmaLayer(ctx, b.frame / 60, body * (0.9 + 0.1 * b.flare),
      lm ? { landmarks: lm, scale, center: c, radius: R } : undefined, support);
    ctx.globalCompositeOperation = "screen";
    ctx.filter = `blur(${Math.max(0.55 * px, PLASMA_SOFT * R)}px)`;
    if (plasma) ctx.drawImage(plasma.canvas, c[0] - PLASMA_EXTENT * R, c[1] - PLASMA_EXTENT * R,
      2 * PLASMA_EXTENT * R, 2 * PLASMA_EXTENT * R);
    else {
      const reach = BODY_STOPS[BODY_STOPS.length - 1][0] * R;
      const g = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach);
      for (const [at, r, gg, bb, a] of BODY_STOPS) g.addColorStop(at * R / reach, `rgba(${r}, ${gg}, ${bb}, ${a * body})`);
      ctx.fillStyle = g;
      ctx.beginPath();
      for (let i = 0; i <= 40; i++) {
        const phi = i * 2 * Math.PI / 40;
        const r = reach * (0.84 + 0.09 * Math.sin(3 * phi + b.turn) + 0.065 * Math.sin(5 * phi - b.turn));
        const p: P2 = [c[0] + r * Math.cos(phi), c[1] + r * Math.sin(phi)];
        if (i) ctx.lineTo(...p); else ctx.moveTo(...p);
      }
      ctx.fill();
    }
    ctx.filter = "none";
    const [k, sigma, rgb] = HALO, far = 3.5 * sigma * R;
    const texture = cloudHaze(ctx, b.frame / 60, body);
    if (texture) ctx.drawImage(texture.canvas, c[0] - far, c[1] - far, 2 * far, 2 * far);
    else {
      const haze = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], far);
      haze.addColorStop(0, `rgba(${rgb}, ${0.3 * k * body})`);
      haze.addColorStop(0.4, `rgba(${rgb}, ${0.12 * k * body})`);
      haze.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = haze;
      ctx.fillRect(c[0] - far, c[1] - far, 2 * far, 2 * far);
    }
  }
  ctx.restore();
}

/** Rear filaments lose light through the core's projected thickness and yield to foreground fingers; front
 * channels cross the distributed emission with sharper, brighter cores. */
export function drawBall(ctx: CanvasRenderingContext2D, b: Ball, palm: P2, scale: number, px: number) {
  const lm = ballHands.get(ctx);
  const R = ballRadius(b, scale), c = ballAt(palm, px, R, lm);
  const body = b.born ? dense(b.level, b.charge) : 0;
  ctx.save();
  const reach = 1 + FIL_REACH * (1 - b.most) ** 2;
  const cs = Math.cos(b.turn), sn = Math.sin(b.turn), ct = Math.cos(0.35), st = Math.sin(0.35);
  const rear: [Channel, number][] = [], front: [Channel, number][] = [];
  const width = Math.max(0.8 * px, widthK(scale, px) * px * (1.05 + 0.8 * b.level));
  const projected = b.shell.map((s) => s.pts.map(([x, y, z]): P3 => {
    const x1 = x * cs + z * sn, z1 = -x * sn + z * cs;
    return [x1, y * ct - z1 * st, y * st + z1 * ct];
  }));
  const leaders = new Set(projected.map((pts, k) => ({ k, score: pts.reduce((sum, [x, y, z]) =>
    sum + Math.max(0, z) * Math.max(0, 1 - Math.hypot(x, y)), 0) / Math.max(1, pts.length) * b.shell[k].flash }))
    .sort((a, b) => b.score - a.score).slice(0, 2).map(({ k }) => k));
  for (let k = 0; k < b.shell.length; k++) {
    const s = b.shell[k];
    if (s.flash <= 0.02) continue;
    const shellGain = s.pts.reduce((sum, p) => sum + Math.hypot(...p), 0) / Math.max(1, s.pts.length) > 0.9 ? 0.35 : 1;
    const seen3 = projected[k];
    let run: P2[] = [], depth = 0, thickness = 0, far = false;
    const append = (p: P2, z: number) => {
      run.push(p); depth += z;
      thickness += Math.sqrt(Math.max(0, 1 - ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) / (R * R || 1)));
    };
    const flush = () => {
      if (run.length > 1) {
        const z = depth / run.length;
        const opticalDepth = far ? 1 : 0.35;
        const focusGain = (far ? 0.45 : 1.35 + 0.6 * Math.max(0, z)) * Math.exp(-4 * opticalDepth * body * thickness / run.length);
        const gain = focusGain * shellGain * (0.65 + 0.35 * b.level) * s.flash * s.inten * b.fade * (leaders.has(k) ? 0.24 : 0.14);
        (far ? rear : front).push([{ pts: run, width: width * (far ? 0.5 : 1), inten: 0.85, forks: [], at: 0, filament: true,
          depth: opticalDepth, softness: (far ? 1.8 + 1.4 * Math.max(0, -z) : 1.3) * px }, gain]);
      }
      run = []; depth = 0; thickness = 0;
    };
    for (let i = 0; i < seen3.length; i++) {
      const [x, y, z] = seen3[i], p: P2 = [c[0] + R * reach * x, c[1] + R * reach * y];
      const nextFar = z < -0.1;
      let boundary: P2 | null = null;
      if (run.length && nextFar !== far) {
        const [prevX, prevY, prevZ] = seen3[i - 1], u = (-0.1 - prevZ) / (z - prevZ);
        boundary = [c[0] + R * reach * (prevX + u * (x - prevX)), c[1] + R * reach * (prevY + u * (y - prevY))];
        if (!far || !lm || !fingerCover(boundary, lm, scale)) append(boundary, -0.1);
        flush();
      }
      far = nextFar;
      const hidden = far && lm && fingerCover(p, lm, scale);
      const gap = far && (Math.floor(i * 32 / Math.max(1, seen3.length - 1)) + 3 * k) % 13 >= 10;
      if (hidden || gap) { flush(); continue; }
      if (boundary && (!far || !lm || !fingerCover(boundary, lm, scale))) append(boundary, -0.1);
      append(p, z);
    }
    flush();
  }
  if (body > 0) {
    // Two foreground currents share one sphere; dim broken rear stretches yield to its material and the fingers.
    const seconds = b.frame / 60, rand = rng(15485863 + Math.floor(b.frame / 2));
    const currents = ballCurrents(b);
    const crossing = currents[0].map(([x, y]): P2 => [c[0] + R * x, c[1] + R * y]);
    front.push([{ pts: crossing, width: 0.95 * width, inten: 1.1 * (0.4 + 0.6 * b.level), forks: [], at: 0,
      filament: true, depth: 0, softness: Math.max(0.35 * px, 0.003 * R), endSoft: Math.max(0.3 * px, 0.02 * R),
      endFade: 0.68, recessSoft: Math.max(0.5 * px, 0.055 * R) }, b.fade * body]);
    const visibleFilament = (points: P2[], width: number, inten: number, softness: number, depth: number,
      behind = depth >= 0.5, endSoft?: number, endFade?: number) => {
      let run: P2[] = [];
      const emit = () => {
        if (run.length > 1) {
          const thickness = run.reduce((sum, [x, y]) => sum + Math.sqrt(Math.max(0,
            1 - ((x - c[0]) ** 2 + (y - c[1]) ** 2) / (R * R || 1))), 0) / run.length;
          (behind ? rear : front).push([{ pts: run, width, inten, forks: [], at: 0, filament: true, depth,
            softness, endSoft, endFade }, b.fade * body * Math.exp(-4 * depth * body * thickness)]);
        }
        run = [];
      };
      for (const p of points) {
        if (lm && fingerCover(p, lm, scale)) emit();
        else run.push(p);
      }
      emit();
    };
    const interior = (anchors: P2[], noise: number): P2[] => {
      const knots = anchors.map(([x, y]): P2 => [x + noise * gauss(rand), y + noise * gauss(rand)]);
      const points: P2[] = [];
      for (let i = 1; i < knots.length; i++) {
        const a = knots[i - 1], z = knots[i];
        const depth = Math.max(1, Math.ceil(Math.log2(Math.hypot(z[0] - a[0], z[1] - a[1]) / 0.025)));
        points.push(...displace(a, z, rand, 0.18, depth).slice(0, -1));
      }
      points.push(knots[knots.length - 1]);
      return points.map(([x, y]): P2 => [c[0] + R * x, c[1] + R * y]);
    };
    const far = interior([[-0.63, -0.38], [-0.38, -0.18], [-0.18, -0.04], [0.06, 0.15], [0.28, 0.40], [0.50, 0.53]], 0.035);
    visibleFilament(far, 0.85 * width, 0.45 * (0.4 + 0.6 * b.level), 2.5 * px, 0.85, true);
    const near = currents[1].map(([x, y]): P2 => [c[0] + R * x, c[1] + R * y]);
    visibleFilament(near, 0.78 * width, 0.88 * (0.4 + 0.6 * b.level), 0.65 * px, 0, false,
      Math.max(0.3 * px, 0.02 * R), 0.75);
    const knots: P2[] = [[-0.55, 0.38], [-0.44, 0.58], [-0.22, 0.48], [-0.15, 0.65], [0.10, 0.57]]
      .map(([x, y]) => [x + 0.025 * gauss(rand), y + 0.025 * gauss(rand)]);
    const lower: P2[] = [];
    for (let i = 1; i < knots.length; i++) {
      const a = knots[i - 1], z = knots[i], depth = Math.max(1, Math.ceil(Math.log2(Math.hypot(z[0] - a[0], z[1] - a[1]) / 0.025)));
      lower.push(...displace(a, z, rand, 0.18, depth).slice(0, -1));
    }
    lower.push(knots[knots.length - 1]);
    visibleFilament(lower.map(([x, y]): P2 => [c[0] + R * x, c[1] + R * y]), 0.55 * width,
      0.36 * (0.4 + 0.6 * b.level), 1.4 * px, 0.20);
    for (const [start, end] of [[-2.6, -2.34], [-1.1, -0.82], [0.44, 0.72], [1.65, 1.97]]) {
      const points: P2[] = [];
      for (let i = 0; i <= 18; i++) {
        const theta = start + (end - start) * i / 18 + 0.08 * Math.sin(seconds * 0.7);
        const radius = 0.94 + 0.035 * gauss(rand);
        points.push([c[0] + R * radius * Math.cos(theta), c[1] + R * radius * Math.sin(theta)]);
      }
      visibleFilament(points, 0.62 * width, (end < 0 ? 0.12 : 0.15) * (0.4 + 0.6 * b.level), 4 * px, 1);
    }
    // A low channel intensity selects the existing full branch taper; its gain restores a connected bright root.
    for (const points of ballStreamers(b, c, R, lm ?? null, scale)) front.push([{ pts: points,
      width: 0.80 * width, inten: 0.35 * (0.4 + 0.6 * b.level), forks: [], at: 0,
      softness: Math.max(0.35 * px, 0.003 * R) }, 2.4 * b.fade * body]);
  }
  drawChannels(ctx, rear, px, 0.18, 0.15 * R, false, BALL_LOOK);
  drawBallBulk(ctx, b, c, R, px, body, lm, scale, filamentSupport(front, c, R));
  drawChannels(ctx, front, px, 0.3, 0.15 * R, false, BALL_LOOK);
  ctx.restore();
}

/** Foreground finger coverage, with a soft silhouette and gradual MCP-to-PIP fade. The palm remains a receiver;
 * this wider tapered contour avoids cutting narrow capsule-shaped holes into the faint volume. */
export function fingerCoverage(p: P2, lm: P2[], scale: number): number {
  if (lm.length !== 21 || scale <= 0) return 0;
  const smooth = (v: number) => { const u = Math.max(0, Math.min(1, v)); return u * u * (3 - 2 * u); };
  let coverage = 0;
  for (const finger of FINGERS) for (let j = 0; j < finger.length - 1; j++) {
    const a = lm[finger[j]], b = lm[finger[j + 1]], dx = b[0] - a[0], dy = b[1] - a[1];
    const pad = 0.11 * scale;
    if (p[0] < Math.min(a[0], b[0]) - pad || p[0] > Math.max(a[0], b[0]) + pad ||
      p[1] < Math.min(a[1], b[1]) - pad || p[1] > Math.max(a[1], b[1]) + pad) continue;
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
    const radius = scale * (j === finger.length - 2 ? 0.11 - 0.03 * t : 0.11);
    const distance = Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
    coverage = Math.max(coverage, smooth((radius - distance) / (0.035 * scale)) * (j ? 1 : smooth(t)));
  }
  return coverage;
}

function fingerCover(p: P2, lm: P2[], scale: number): boolean {
  return fingerCoverage(p, lm, scale) >= 0.35;
}

/** Clip each tapered finger segment out separately so overlapping fingers keep their union as foreground.
 * The compact palm source and its channels use this contour; the volume uses soft coverage instead. */
function clipFingers(ctx: CanvasRenderingContext2D, lm: P2[], scale: number) {
  for (const finger of FINGERS) for (let j = 0; j < finger.length - 1; j++) {
    const a = lm[finger[j]], b = lm[finger[j + 1]];
    const start = a, r0 = j ? 0.11 * scale : 0, r1 = (j === finger.length - 2 ? 0.08 : 0.11) * scale;
    const angle = Math.atan2(b[1] - start[1], b[0] - start[0]);
    const nx = -Math.sin(angle), ny = Math.cos(angle);
    ctx.beginPath();
    ctx.rect(-100000, -100000, 200000, 200000);
    ctx.moveTo(start[0] + nx * r0, start[1] + ny * r0);
    ctx.lineTo(b[0] + nx * r1, b[1] + ny * r1);
    ctx.arc(b[0], b[1], r1, angle + Math.PI / 2, angle - Math.PI / 2, true);
    ctx.lineTo(start[0] - nx * r0, start[1] - ny * r0);
    if (r0) ctx.arc(start[0], start[1], r0, angle - Math.PI / 2, angle + Math.PI / 2, true);
    ctx.closePath();
    ctx.clip("evenodd");
  }
}

/** Whether a point ``p`` lies on the hand (its 21 landmarks ``lm``, px: the palm and each finger as thick as a finger,
 * as handLayer draws it, for a hand ``scale`` px across). */
export function onHand(p: P2, lm: P2[], scale: number): boolean {
  const r = 0.1 * scale;
  let inside = false;
  for (let i = 0, j = PALM.length - 1; i < PALM.length; j = i++) {
    const [xi, yi] = lm[PALM[i]], [xj, yj] = lm[PALM[j]];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
    if (toSegment(p, lm[PALM[i]], lm[PALM[j]]) < r) return true;
  }
  if (inside) return true;
  return FINGERS.some((f) => f.slice(1).some((k, i) => toSegment(p, lm[f[i]], lm[k]) < r));
}

/** A jumping arc's points from (0, 0) to (1, 0) for a chord ``span`` px long (``px``: a pixel at 1080 lines): its
 * course a quadratic bow whose middle is ``bow`` of the chord off it (to +y), walked in straight steps LEADER's px
 * long, each along the course where the walk is (turned back towards it as far as it has strayed, at most LEADER's
 * most) and LEADER's degrees off that, to the other side from the last LEADER's share of the time, so it kinks 20-50
 * degrees between them; the last straight onto its end (cuesync.lightning's _leader). */
export function leader(span: number, bow: number, rand: () => number, px: number): P2[] {
  const [s0, s1, d0, d1, alternate] = LEADER;
  const n = 64, ctl: P2 = [0.5 * span, 2 * bow * span], end: P2 = [span, 0];
  const course: P2[] = [], tangent: P2[] = [];
  for (let i = 0; i <= n; i++) {
    const u = i / n;
    course.push([2 * u * (1 - u) * ctl[0] + u * u * end[0], 2 * u * (1 - u) * ctl[1]]);
    const tx = 2 * (1 - u) * ctl[0] + 2 * u * (end[0] - ctl[0]), ty = 2 * (1 - u) * ctl[1] - 2 * u * ctl[1];
    const tl = Math.hypot(tx, ty) || 1;
    tangent.push([tx / tl, ty / tl]);
  }
  const pts: P2[] = [[0, 0]];
  const length = course.slice(1).reduce((sum, p, i) => sum + Math.hypot(p[0] - course[i][0], p[1] - course[i][1]), 0);
  // Four shortest-step traversals leave room for kinks while bounding a walk that misses its endpoint.
  const walkLimit = Math.max(16, 4 * Math.ceil(length / Math.max(s0 * px, 1e-6)));
  let at = 0, sign = rand() < 0.5 ? -1 : 1;
  for (let guard = 0; guard < walkLimit; guard++) {
    const p = pts[pts.length - 1], step = (s0 + (s1 - s0) * rand()) * px;
    if (Math.hypot(end[0] - p[0], end[1] - p[1]) <= 1.3 * step) break;
    let best = at;
    for (let i = at; i <= n; i++) {
      if (Math.hypot(course[i][0] - p[0], course[i][1] - p[1]) < Math.hypot(course[best][0] - p[0], course[best][1] - p[1])) best = i;
    }
    at = best;
    if (at === n && (p[0] - end[0]) * tangent[n][0] + (p[1] - end[1]) * tangent[n][1] >= 0) break;
    const [tx, ty] = tangent[at];
    const off = tx * (p[1] - course[at][1]) - ty * (p[0] - course[at][0]);
    const back = Math.max(-d1, Math.min(d1, (-Math.atan2(off, 2 * step) * 180) / Math.PI));
    if (rand() < alternate) sign = -sign;
    const a = Math.atan2(ty, tx) + ((back + sign * (d0 + (d1 - d0) * rand())) * Math.PI) / 180;
    pts.push([p[0] + Math.cos(a) * step, p[1] + Math.sin(a) * step]);
  }
  pts.push(end);
  return pts.map(([x, y]): P2 => [x / span, y / span]);
}

/** Degrees: how far a path's course swings in all (cut into six pieces of equal length, so its kinks are not
 * counted: its turns summed as it goes, the most less the least); over 180 it has turned back on itself. */
export function swing(pts: P2[]): number {
  const course_ = course(pts, 6);
  let a = 0, lo = 0, hi = 0, last = Math.atan2(course_[1][1] - course_[0][1], course_[1][0] - course_[0][0]);
  for (let i = 2; i < course_.length; i++) {
    const h = Math.atan2(course_[i][1] - course_[i - 1][1], course_[i][0] - course_[i - 1][0]);
    a += ((h - last + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
    last = h;
    lo = Math.min(lo, a);
    hi = Math.max(hi, a);
  }
  return ((hi - lo) * 180) / Math.PI;
}

/** The point of the hand (``hand``: whether a point is on it) nearest ``p`` within ``near`` px: ``p`` itself when it
 * is on it, null when none is (looked for on rings round it). */
function toHand(p: P2, near: number, hand: (q: P2) => boolean): P2 | null {
  if (hand(p)) return p;
  for (let r = near / 6; r <= near + 1e-9; r += near / 6) {
    for (let i = 0; i < 16; i++) {
      const q: P2 = [p[0] + r * Math.cos((i * Math.PI) / 8), p[1] + r * Math.sin((i * Math.PI) / 8)];
      if (hand(q)) return q;
    }
  }
  return null;
}

/** An arc's points from the ball (middle ``c``, radius ``R``) to a landmark ``tip``: ``shape`` (from (0, 0) to (1,
 * 0): leader) carried from its skin, ``leave`` radians round from the point facing the tip, through the air, coming
 * down onto the finger past its tip LAND_IN inside it, or, coming in across the finger, onto the landmark with its last
 * LAND_NEAR over it; or null where it cannot: the hand nowhere within LAND_NEAR of the landmark (it does not end in the
 * air between two fingers), its end further than that from the landmark, back over the ball, or, with ``air``, past
 * GROUND_FREE of the ball more of it over the hand than GROUND_ON lets it (``scale``: the hand's size). ``hand``:
 * whether a point is on the hand, or null (no hand: it lands on the landmark). */
export function jumpPath(shape: P2[], c: P2, R: number, tip: P2, leave: number, px: number,
  hand: ((p: P2) => boolean) | null, scale = 0, air = true): P2[] | null {
  const phi = Math.atan2(tip[1] - c[1], tip[0] - c[0]) + leave;
  const a: P2 = [c[0] + R * Math.cos(phi), c[1] + R * Math.sin(phi)];
  if (Math.hypot(tip[0] - a[0], tip[1] - a[1]) < 1e-6) return null;
  let land = tip, onto = LAND_IN + 3;
  if (hand) {
    const on = toHand(tip, LAND_NEAR * px, hand);
    if (!on) return null;
    const [lx, ly] = [shape[shape.length - 1][0] - shape[shape.length - 2][0], shape[shape.length - 1][1] - shape[shape.length - 2][1]];
    const t = Math.atan2(on[1] - a[1], on[0] - a[0]) + Math.atan2(ly, lx);
    const come: P2 = [Math.cos(t), Math.sin(t)]; // (its way in at its end)
    land = on;
    onto = LAND_NEAR;
    for (let e = 0; e < (LAND_NEAR + LAND_IN) * px; e += px) { // (back along it, to where the finger ends)
      if (!hand([on[0] - come[0] * e, on[1] - come[1] * e])) {
        const back = Math.max(0, e - LAND_IN * px);
        land = [on[0] - come[0] * back, on[1] - come[1] * back];
        onto = LAND_IN + 3;
        break;
      }
    }
    if (Math.hypot(land[0] - tip[0], land[1] - tip[1]) > LAND_NEAR * px + 1e-9) return null;
  }
  const dx = land[0] - a[0], dy = land[1] - a[1];
  const pts = shape.map(([u, v]): P2 => [a[0] + u * dx - v * dy, a[1] + u * dy + v * dx]);
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = cum[cum.length - 1];
  let over = 0, free = 0, run = 0, most = 0;
  for (let i = 1; i < pts.length; i++) {
    const r = Math.hypot(pts[i][0] - c[0], pts[i][1] - c[1]);
    if (cum[i] > 0.15 * total && r < 0.95 * R) return null; // (back over the ball)
    if (!hand || !air || r <= GROUND_FREE * R || cum[i] >= total - onto * px) { run = 0; continue; }
    const l = cum[i] - cum[i - 1];
    free += l;
    if (hand(pts[i])) { over += l; run += l; most = Math.max(most, run); } else run = 0;
  }
  return over > GROUND_ON[0] * Math.max(free, 1e-6) || most > GROUND_ON[1] * scale ? null : pts;
}

/** The ball's light round it (BALL_LIGHT), screened: a cool white on the hand (its 21 landmarks ``lm``, px, or null)
 * and a blue-white on the room, from its middle as from a ball of its radius, R^2 / (d^2 + h^2) at most 1, gone by
 * three radii, little of it on what its dense middle hides, as strong as its body is dense; on the hand also a softer
 * light among the fingers round it, falling off as 1 / r from its edge out (FILL), small surface-clipped patches
 * beside the escaping streamers, and added, a pool of it on the palm
 * under it (POOL_PALM). And, with the hand, the arcs it throws to the fingers (GROUND): a side's to that side's
 * fingertips clear of it, at most one per target, or a nearby pad or knuckle, each in
 * straight kinked steps (leader) bowing through the air off the hand and coming down onto the finger (jumpPath; one
 * that cannot looks for another way, GROUND_TRIES times, or is not drawn), white-hot in a cyan-blue glow, with a
 * white-hot point in a blue glow where each lands (drawLand). ``palm``: the palm it hovers over (ballAt); ``px``: a
 * pixel at 1080 lines. */
export function drawBallLight(ctx: CanvasRenderingContext2D, b: Ball, palm: P2, scale: number, lm: P2[] | null,
  w: number, h: number, px = h / 1080) {
  if (lm && lm.length === 21) ballHands.set(ctx, lm);
  else ballHands.delete(ctx);
  const R = ballRadius(b, scale);
  const c = ballAt(palm, px, R, lm);
  const body = b.born ? dense(b.level, b.charge) : 0;
  if (body > 0) {
    const [onHandA, onRoomA, hover, behind, reach, shade] = BALL_LIGHT;
    const lit = (0.5 + 0.5 * b.level) * body;
    const falloff = (fill: CanvasGradient, rgb: string, alpha: number, hh: number, among = 0) => {
      for (let i = 0; i <= 12; i++) { // (from its middle out to its reach)
        const d = (reach * i) / 12, gone = Math.max(0, 1 - (d / reach) ** 2) ** 2;
        const hide = 1 - Math.exp(-((d / shade) ** 8));
        const u = Math.max(0, Math.min(1, (d - 1) / (FILL[2] - 1))), v = Math.max(0, Math.min(1, (reach - d) / (reach - FILL[1])));
        const fill_ = among * Math.min(1, 1 / Math.max(d, 1e-6)) * u * u * (3 - 2 * u) * v * v * (3 - 2 * v);
        fill.addColorStop(i / 12, `rgba(${rgb}, ${Math.min(1, lit * (alpha * Math.min(1, 1 / (d * d + hh * hh)) * gone * hide + fill_))})`);
      }
    };
    ctx.save();
    const cool = lm ? handLayer(ctx, "ball-cool", lm, scale) : null;
    if (cool) {
      const tint = cool.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach * R);
      tint.addColorStop(0, `rgba(125, 195, 255, ${0.28 * lit})`);
      tint.addColorStop(0.55, `rgba(125, 195, 255, ${0.18 * lit})`);
      tint.addColorStop(1, "rgba(125, 195, 255, 0)");
      cool.fillStyle = tint;
      cool.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
      ctx.globalCompositeOperation = "multiply";
      ctx.drawImage(cool.canvas, 0, 0, w, h);
    }
    ctx.globalCompositeOperation = "screen";
    const room = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach * R);
    falloff(room, BALL_LIGHT_RGB[1], onRoomA, behind);
    ctx.fillStyle = room;
    ctx.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
    const mask = lm ? handLayer(ctx, "ball-hand", lm, scale) : null;
    if (mask) {
      const hand = mask.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach * R);
      falloff(hand, BALL_LIGHT_RGB[0], onHandA, (hover * px) / Math.max(R, 1e-6), FILL[0]);
      mask.fillStyle = hand;
      mask.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
      ctx.drawImage(mask.canvas, 0, 0, w, h);
    }
    const inner = lm ? handLayer(ctx, "ball-inner", lm, scale) : null;
    if (inner && lm) {
      const facing = inner.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach * R);
      facing.addColorStop(0, `rgba(195, 230, 255, ${0.7 * lit})`);
      facing.addColorStop(0.5, `rgba(160, 211, 255, ${0.35 * lit})`);
      facing.addColorStop(1, "rgba(140, 200, 255, 0)");
      inner.strokeStyle = facing;
      inner.lineWidth = 0.075 * scale;
      inner.filter = `blur(${Math.max(0.4, 0.018 * scale / SOFT_Q)}px)`;
      inner.beginPath();
      for (const finger of FINGERS) for (let i = 0; i < finger.length; i++) {
        const p = lm[finger[i]], d = Math.max(1, Math.hypot(c[0] - p[0], c[1] - p[1]));
        const q: P2 = [p[0] + 0.055 * scale * (c[0] - p[0]) / d, p[1] + 0.055 * scale * (c[1] - p[1]) / d];
        if (i) inner.lineTo(...q); else inner.moveTo(...q);
      }
      inner.stroke();
      ctx.drawImage(inner.canvas, 0, 0, w, h);
    }
    if (lm) for (const [k, points] of ballStreamers(b, c, R, lm, scale).entries()) {
      const local = handLayer(ctx, `ball-streamer-${k}`, lm, scale);
      if (!local) continue;
      const end = points[points.length - 1], radius = 0.34 * R;
      const glow = local.createRadialGradient(end[0], end[1], 0, end[0], end[1], radius);
      for (let i = 0; i <= 6; i++) {
        const u = i / 6, alpha = i === 6 ? 0 : 0.28 * lit * b.fade * Math.exp(-0.5 * (0.34 * u / 0.14) ** 2);
        glow.addColorStop(u, `rgba(120, 195, 255, ${alpha})`);
      }
      local.fillStyle = glow;
      local.fillRect(end[0] - radius, end[1] - radius, 2 * radius, 2 * radius);
      ctx.drawImage(local.canvas, 0, 0, w, h);
    }
    const pool = lm ? handLayer(ctx, "ball-pool", lm, scale) : null;
    if (pool) { // (and pooled on the palm under it, added: +25 levels a radius from the palm's point)
      const [a0, a1] = POOL_PALM, r = POOL_PALM[2] * R;
      const g = pool.createRadialGradient(palm[0], palm[1], 0, palm[0], palm[1], r);
      for (let i = 0; i <= 6; i++) { // (from a0 at its point to a1 a radius out, then easing out to none)
        const d = (i / 6) * POOL_PALM[2]; // (radii)
        const a = d <= 1 ? a0 + (a1 - a0) * d : a1 * ((POOL_PALM[2] - d) / (POOL_PALM[2] - 1)) ** 2;
        g.addColorStop(i / 6, `rgba(${BALL_LIGHT_RGB[0]}, ${lit * a})`);
      }
      pool.fillStyle = g;
      pool.fillRect(palm[0] - r, palm[1] - r, 2 * r, 2 * r);
      ctx.globalCompositeOperation = "lighter";
      ctx.drawImage(pool.canvas, 0, 0, w, h);
    }
    ctx.restore();
  }
  if (!lm || !b.born) return;
  const hand = (p: P2) => onHand(p, lm, scale);
  const free = TIPS.filter((t) => Math.hypot(lm[t][0] - c[0], lm[t][1] - c[1]) >= 1.15 * R);
  const pads = PADS.filter((t) => {
    const d = Math.hypot(lm[t][0] - c[0], lm[t][1] - c[1]);
    return d >= GROUND_PAD[1] * R && d <= GROUND_PAD[2] * R;
  });
  const lines: [Channel, number][] = [];
  const lands: [P2, number, P2][] = [];
  const used = new Set<number>();
  for (const j of b.jumps) {
    const mine = (j.pad ? pads : free.filter((t) => (lm[t][0] >= c[0] ? 1 : -1) === j.side)).filter((to) => !used.has(to));
    if (!mine.length) continue;
    let pts = j.shape && j.to !== undefined && mine.includes(j.to)
      ? jumpPath(j.shape, c, R, lm[j.to], j.leave, px, hand, scale, !j.pad) : null;
    if (!pts) { // One path per available finger prevents paired grounding arcs from closing a separate loop.
      const rand = rng(j.seed);
      for (let k = 0; k < GROUND_TRIES && !pts; k++) {
        const to = mine[(j.k + k) % mine.length];
        const [b0, b1] = j.pad ? GROUND_PAD.slice(3) : GROUND_BOW;
        const bow = k ? (rand() < 0.5 ? -1 : 1) * (b0 + (b1 - b0) * rand()) : j.bow;
        const leave = k ? (2 * rand() - 1) * GROUND_LEAVE : j.leave;
        const phi = Math.atan2(lm[to][1] - c[1], lm[to][0] - c[0]) + leave;
        const span = Math.hypot(lm[to][0] - c[0] - R * Math.cos(phi), lm[to][1] - c[1] - R * Math.sin(phi));
        if (span < 1) continue;
        const shape = leader(span, bow, rand, px);
        if (swing(shape) > GROUND_TURN) continue;
        pts = jumpPath(shape, c, R, lm[to], leave, px, hand, scale, !j.pad);
        if (pts) Object.assign(j, { to, shape, bow, leave }); // (kept while it lives)
      }
    }
    if (!pts) continue;
    used.add(j.to!);
    if (j.flash <= 0.05) continue;
    const span = Math.hypot(pts[pts.length - 1][0] - pts[0][0], pts[pts.length - 1][1] - pts[0][1]);
    const proximity = Math.min(1, 0.55 * R / Math.max(span, 1));
    lines.push([{ pts, width: proximity * Math.max(1.2 * px, widthK(scale, px) * px * 1.8 * (0.8 + 0.3 * b.level)), inten: 1,
      forks: [], at: 0 }, 0.55 * 0.85 * j.flash * proximity ** 3]);
    const to = j.to!, joint = lm[to === 5 || to === 17 ? to + 1 : Math.max(0, to - 1)];
    lands.push([pts[pts.length - 1], j.flash * proximity ** 4, [lm[to][0] - joint[0], lm[to][1] - joint[1]]]);
  }
  drawChannels(ctx, lines, px, 0.5, 0.1 * scale, false, BALL_LOOK);
  for (const [at, amp, along] of lands) drawLand(ctx, at, Math.min(1, amp), px, along);
}

/** Where an arc lands on a finger: a white-hot point LAND's px wide in a #6fa8ff glow LAND's wide, the skin under the
 * glow first taken towards cyan (multiplied), so added blue reads blue over red skin, not as a pink halo. */
export function drawLand(ctx: CanvasRenderingContext2D, at: P2, amp: number, px: number, along: P2 = [1, 0]) {
  const r = LAND[1] * px;
  ctx.save();
  ctx.translate(at[0], at[1]);
  ctx.rotate(Math.atan2(along[1], along[0]));
  ctx.scale(1.35, 0.55);
  at = [0, 0];
  ctx.globalCompositeOperation = "multiply";
  const m = ctx.createRadialGradient(at[0], at[1], 0, at[0], at[1], r);
  m.addColorStop(0, `rgba(150, 215, 255, ${0.8 * amp})`);
  m.addColorStop(1, "rgba(150, 215, 255, 0)");
  ctx.fillStyle = m;
  ctx.fillRect(at[0] - r, at[1] - r, 2 * r, 2 * r);
  ctx.globalCompositeOperation = "lighter";
  const g = ctx.createRadialGradient(at[0], at[1], 0, at[0], at[1], r);
  g.addColorStop(0, `rgba(205, 230, 255, ${0.75 * amp})`);
  g.addColorStop(LAND[0] / LAND[1] / 2, `rgba(175, 215, 255, ${0.6 * amp})`);
  g.addColorStop(0.45, `rgba(${LAND_RGB}, ${0.45 * amp})`);
  g.addColorStop(1, `rgba(${LAND_RGB}, 0)`);
  ctx.fillStyle = g;
  ctx.fillRect(at[0] - r, at[1] - r, 2 * r, 2 * r);
  ctx.restore();
}

/** What a frame of the microphone sounds like for lightning: the share of its energy at 4-11 kHz (of 150 Hz-11 kHz,
 * from the analyser's dB spectrum) and whether it has a pitch (70-400 Hz, looked for under ~900 Hz: the hiss on a
 * "zzzzz" swamps it full band). A buzz has both (cuesync.gestures BUZZ_HF). */
export function buzzOf(spectrumDb: Float32Array, pcm: Float32Array, sampleRate: number): { hf: number; voiced: boolean } {
  const binHz = sampleRate / 2 / spectrumDb.length;
  let total = 0, high = 0;
  for (let i = 0; i < spectrumDb.length; i++) {
    const f = i * binHz;
    if (f < 150 || f >= 11000) continue;
    const p = 10 ** (spectrumDb[i] / 10);
    total += p;
    if (f >= 4000) high += p;
  }
  const win = Math.max(1, Math.round(sampleRate / 1800)); // a moving average: under ~900 Hz
  const lp = new Float32Array(pcm.length);
  let acc = 0;
  for (let i = 0; i < pcm.length; i++) {
    acc += pcm[i] - (i >= win ? pcm[i - win] : 0);
    lp[i] = acc / win;
  }
  let mean = 0;
  for (const v of lp) mean += v;
  mean /= lp.length || 1;
  let e = 0;
  for (let i = 0; i < lp.length; i++) { lp[i] -= mean; e += lp[i] * lp[i]; }
  let best = 0;
  if (e > 1e-9) {
    for (let lag = Math.floor(sampleRate / 400); lag <= Math.min(Math.ceil(sampleRate / 70), lp.length - 1); lag++) {
      let c = 0;
      for (let i = 0; i + lag < lp.length; i++) c += lp[i] * lp[i + lag];
      best = Math.max(best, c / e);
    }
  }
  return { hf: total > 0 ? high / total : 0, voiced: best > 0.45 };
}

export const BUZZ_HF = 0.35;
