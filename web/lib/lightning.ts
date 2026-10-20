/** Live lightning for Try it live: arcs struck from a fingertip, or a ball of light over an open palm, drawn on the
 * camera's canvas every frame next to the hand tracker, so everything here is a few hundred points and a few
 * gradients a frame. As the shot's own (cuesync.lightning), as far as Canvas 2D goes:
 * - a finger strikes one dominant main channel, its first stretch along the finger, its root welded to the fingertip;
 *   beside it at most two dim, thin streamers at most half as long, keeping well round from it, and dim forks towards
 *   its far end, every branch fading and thinning along its length and none crossing the main channel or another
 *   branch, nor a streamer running beside it (each is struck again until it keeps clear, or dropped);
 * - a channel is struck, re-struck down the same path for 1-3 frames with its root riding the finger and its far end
 *   where it struck, then struck anew near its last aim;
 * - the light it throws, a cool blue on the room and lavender on the hand, holds while the strokes strike and dies
 *   within a few frames of the last (it never strobes with them);
 * - over an open palm, a ball that is a light, its body the shot's own by radius (white-hot in its middle, cyan round
 *   that, a saturated blue to a cyan ring at its limb, its outline feathered), the lens's narrow band of blue just over
 *   its limb, its light on the skin close round it violet and on the wall a cool blue, and filaments from its middle
 *   out to its skin at angles in 3D, growing from a crackle over about 5 frames, the camera exposing down for it. */

export type P2 = [number, number];

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
const ARC_REACH = [1.4, 5.0]; // hand sizes: a main channel's reach, quietest voice to loudest
const ARC_WIDTH = [1.3, 4.0]; // px at 1080 for a hand HAND_PX across: its width, quietest to loudest
const HAND_PX = 220, WIDTH_K = [0.6, 1.8]; // (widths scale with the hand, within these)
const ARC_LEAD = 0.2; // of its reach: a main channel's first stretch runs along the finger's aim
const ARC_JITTER = 0.07, ARC_OFF = 0.21; // radians: a new main channel's own turn off its aim, and the most it may be
const SIDE_ON = [0.3, 0.85]; // the voice's level over which a first, and a second, streamer strikes
const SIDE_INTEN = 0.35, SIDE_WIDTH = 0.4, SIDE_REACH = [0.3, 0.5], SIDE_GAP = [0.45, 0.63]; // a streamer against
// its main channel: dimmer, thinner, at most half as long, leaving 26-36 degrees off the main channel's way out
export const SIDE_APART = [0.35, 0.2]; // radians, share: and seen from the fingertip, its course past this share of its
// length keeps at least 20 degrees round from the main channel as far out (leaving 18 degrees off its heading, two ran
// beside it as long 1 px hairlines)
const FORK_INTEN = [0.15, 0.35], FORK_WIDTH = 0.45, FORK_TURN = [0.17, 0.6]; // a fork against what it leaves
const REDRAW = 4; // a branch is struck again this many times to keep clear of the others, then dropped
const ALONG = [0.12, 0.25]; // hand sizes, share: a streamer may lie within the first of its main channel for at most
// the second share of its length past where they leave (leaving 20 degrees off, one ran beside it like a second wire)
const LIFE = [1, 2, 3], LIFE_P = [0.25, 0.45, 0.3]; // frames a path is kept, and how often each
const RESTROKE = [0.45, 0.5]; // a re-stroke flashes with this chance plus this times the level, else it is dark
export const TAPER = [0.9, 1.2]; // a branch fades along its length to 1 - 0.9 of itself, as the 1.2th power of how far
// (to a third, a 1 px streamer still read as a hairline to its end)
export const BRANCH_W = [1.5, 0.35, 0.5]; // and thins from this many times its width to this, never under 0.5 px: a
// wedge, not a wire (cuesync.lightning's BRANCH_W)
const CHUNKS = 6; // stretches a channel is drawn in, thinning and (a branch) fading along it

export interface Channel { pts: P2[]; width: number; inten: number; forks: Channel[]; at: number } // at: the parent's
// point it forks from; inten: against the bolt's main channel (1)

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

/** The share of ``a``'s length that lies within ``d`` of ``b``, past ``d`` from where ``a`` leaves. */
export function beside(a: P2[], b: P2[], d: number): number {
  let near = 0, total = 0;
  for (let i = 1; i < a.length; i++) {
    const l = Math.hypot(a[i][0] - a[i - 1][0], a[i][1] - a[i - 1][1]);
    total += l;
    if (Math.hypot(a[i][0] - a[0][0], a[i][1] - a[0][1]) <= d) continue;
    for (let j = 0; j + 1 < b.length; j++) if (toSegment(a[i], b[j], b[j + 1]) < d) { near += l; break; }
  }
  return total > 0 ? near / total : 0;
}

/** Radians: which way ``main`` (a channel's points) lies from ``root`` where it first gets ``far`` out (its end, if it
 * never does). */
export function wayOut(main: P2[], root: P2, far: number): number {
  const p = main.find(([x, y]) => Math.hypot(x - root[0], y - root[1]) >= far) ?? main[main.length - 1];
  return Math.atan2(p[1] - root[1], p[0] - root[0]);
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

/** Whether a streamer's points ``pts`` from ``root`` keep SIDE_APART round from ``main`` as seen from ``root``: every
 * point of its course past SIDE_APART's share of its length against the main channel's course as far out. */
export function apart(main: P2[], root: P2, pts: P2[]): boolean {
  const m = course(main, 48);
  const out: number[] = [];
  let most = 0;
  for (const [x, y] of m) out.push((most = Math.max(most, Math.hypot(x - root[0], y - root[1]))));
  const s = course(pts, 16);
  for (let i = Math.ceil(16 * SIDE_APART[1]); i < s.length; i++) {
    const r = Math.hypot(s[i][0] - root[0], s[i][1] - root[1]);
    let j = out.findIndex((v) => v >= r);
    if (j < 0) j = m.length - 1;
    const off = Math.atan2(s[i][1] - root[1], s[i][0] - root[0]) - Math.atan2(m[j][1] - root[1], m[j][0] - root[0]);
    if (Math.abs(((off + 3 * Math.PI) % (2 * Math.PI)) - Math.PI) < SIDE_APART[0]) return false;
  }
  return true;
}

/** How a channel is struck beyond its course: ``at``, the parent's point it forks from; ``inten``, its brightness
 * against the bolt's main channel; ``lead``, a point its first stretch runs to before it turns to its end;
 * ``keepOff``, channels its forks keep clear of; ``clear``, how far (in its own units; its width by default). */
export interface Strike { at?: number; inten?: number; lead?: P2; keepOff?: P2[][]; clear?: number }

/** A channel from ``a`` to ``b``: jagged (six levels of midpoints by default), with dim, thin forks now and then,
 * more towards its far end, each kept off the channel, its other forks and the ``keepOff`` channels (struck again
 * REDRAW times, then dropped). */
export function channel(a: P2, b: P2, rand: () => number, width = 2, rough = 0.28, depth = 6, forkRate = 0.12,
  strike: Strike = {}): Channel {
  const { at = 0, inten = 1, lead, keepOff = [], clear = width } = strike;
  const pts = lead
    ? [...displace(a, lead, rand, rough, depth - 2).slice(0, -1), ...displace(lead, b, rand, rough, depth)]
    : displace(a, b, rand, rough, depth);
  const forks: Channel[] = [];
  if (width > 0.8 && depth > 3) {
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let i = Math.floor(pts.length * 0.15); i < pts.length - 2; i += 2) {
      const along = i / pts.length;
      if (rand() >= forkRate * (0.3 + 1.4 * along) * 0.25) continue;
      const [px, py] = pts[i], [qx, qy] = pts[Math.min(i + 3, pts.length - 1)];
      const side = rand() < 0.5 ? -1 : 1;
      const ang = Math.atan2(qy - py, qx - px) + side * (FORK_TURN[0] + (FORK_TURN[1] - FORK_TURN[0]) * rand());
      const l = len * (1 - along) * (0.25 + 0.4 * rand());
      for (let tries = 0; tries <= REDRAW; tries++) {
        const dim = FORK_INTEN[0] + (FORK_INTEN[1] - FORK_INTEN[0]) * rand();
        const f = channel(pts[i], [px + Math.cos(ang) * l, py + Math.sin(ang) * l], rand, width * FORK_WIDTH, rough,
          depth - 2, forkRate * 0.5, { at: i, inten: inten * dim, clear: clear * FORK_WIDTH });
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

/** A stream of lightning: its channel, struck anew every 1-3 frames near its last aim (which sways slowly), and how
 * bright its stroke is on this frame (1 struck, 0.55-0.9 re-struck, under 0.1 dark). */
export interface Stream { ch: Channel | null; left: number; aim: number; seed: number; flash?: number }

function life(rand: () => number): number {
  const r = rand();
  return r < LIFE_P[0] ? LIFE[0] : r < LIFE_P[0] + LIFE_P[1] ? LIFE[1] : LIFE[2];
}

/** One frame of a main channel from ``root`` (welded to it): ridden to the hand and re-struck, or struck anew towards
 * ``aim`` (radians) ``reach`` px long when its time is up (or on ``strike``: a hit strikes every stream at once), its
 * first ARC_LEAD along the aim and its heading never more than ARC_OFF off it. ``level``: the voice, which sets how
 * often a re-stroke flashes. */
export function streamStep(s: Stream, root: P2, aim: number, reach: number, width: number, strike: boolean,
  frame: number, level = 1): Stream {
  const rand = rng(s.seed * 7919 + frame);
  if (s.ch && s.left > 0 && !strike) {
    const on = rand() < RESTROKE[0] + RESTROKE[1] * level;
    return { ...s, ch: ride(s.ch, root), left: s.left - 1, flash: on ? 0.55 + 0.35 * rand() : 0.1 * rand() };
  }
  const off = s.ch ? s.aim - aim + (rand() - 0.5) * 2 * ARC_JITTER : (rand() - 0.5) * 2 * ARC_JITTER; // (near its last)
  const a = aim + Math.max(-ARC_OFF, Math.min(ARC_OFF, off * 0.6));
  const l = reach * (0.8 + 0.25 * rand());
  const lead: P2 = [root[0] + Math.cos(aim) * ARC_LEAD * l, root[1] + Math.sin(aim) * ARC_LEAD * l];
  const far: P2 = [root[0] + Math.cos(a) * l, root[1] + Math.sin(a) * l];
  const ch = channel(root, far, rand, width, 0.26, 6, 0.1 + 0.1 * level, { lead });
  return { ch, left: life(rand) - 1, aim: a, seed: s.seed, flash: 1 };
}

/** The arcs from one fingertip: its main channel and the streamers beside it. */
export interface Arcs { main: Stream; sides: Stream[] }

/** No arcs yet: the first frame strikes them. */
export function newArcs(seed = 1): Arcs {
  return { main: { ch: null, left: 0, aim: 0, seed }, sides: [] };
}

function widthK(scale: number, px: number): number {
  return Math.max(WIDTH_K[0], Math.min(WIDTH_K[1], scale / (HAND_PX * px)));
}

/** One frame of the arcs from ``tip`` (the fingertip, px) along ``aim`` (radians) for a hand ``scale`` px across at a
 * voice ``level`` (0-1); ``px``: a pixel at 1080 lines. The main channel is welded to the tip; its streamers (one over
 * SIDE_ON's first level, two over its second) leave its root SIDE_GAP off the main channel's way out on either side,
 * at most SIDE_REACH of its reach, struck anew with it or when their own time is up, and struck again (REDRAW times,
 * then dropped) until they cross neither it nor each other, run beside it no further than ALONG lets them and keep
 * SIDE_APART round from it. */
export function arcStep(arcs: Arcs, tip: P2, aim: number, scale: number, level: number, strike: boolean,
  frame: number, px = 1): Arcs {
  const kw = widthK(scale, px) * px;
  const width = kw * (ARC_WIDTH[0] + (ARC_WIDTH[1] - ARC_WIDTH[0]) * level);
  const reach = scale * (ARC_REACH[0] + (ARC_REACH[1] - ARC_REACH[0]) * level);
  const main = streamStep(arcs.main, tip, aim, reach, width, strike, frame, level);
  const anew = main.flash === 1;
  const ch = main.ch!;
  const end = ch.pts[ch.pts.length - 1];
  const mainReach = Math.hypot(end[0] - tip[0], end[1] - tip[1]);
  const count = level > SIDE_ON[1] ? 2 : level > SIDE_ON[0] ? 1 : 0;
  const sides: Stream[] = [];
  for (let j = 0; j < count; j++) {
    const old = arcs.sides[j];
    const seed = main.seed * 31 + j + 1;
    if (old?.ch && old.left > 0 && !anew && !strike) {
      const ridden = ride(old.ch, tip);
      if (!crosses(ridden.pts, ch.pts, kw, 3) && !sides.some((o) => crosses(ridden.pts, o.ch!.pts, kw, 3))) {
        const flash = Math.min(main.flash ?? 1, 0.55 + 0.35 * rng(seed + frame)()); // (no brighter than its channel)
        sides.push({ ...old, ch: ridden, left: old.left - 1, flash });
      }
      continue;
    }
    const rand = rng(seed * 7919 + frame);
    const side = j % 2 ? -1 : 1;
    for (let tries = 0; tries <= REDRAW; tries++) {
      const l = mainReach * (SIDE_REACH[0] + (SIDE_REACH[1] - SIDE_REACH[0]) * rand());
      const a = wayOut(ch.pts, tip, 0.6 * l) + side * (SIDE_GAP[0] + (SIDE_GAP[1] - SIDE_GAP[0]) * rand());
      const s = channel(tip, [tip[0] + Math.cos(a) * l, tip[1] + Math.sin(a) * l], rand, width * SIDE_WIDTH, 0.26, 5,
        0.1 + 0.1 * level, { inten: SIDE_INTEN, keepOff: [ch.pts, ...sides.map((o) => o.ch!.pts)] });
      if (!crosses(s.pts, ch.pts, kw, 3) && !sides.some((o) => crosses(s.pts, o.ch!.pts, kw, 3)) &&
        beside(s.pts, ch.pts, ALONG[0] * scale) <= ALONG[1] && apart(ch.pts, tip, s.pts)) {
        sides.push({ ch: s, left: Math.min(2, life(rand)) - 1, aim: a, seed, flash: main.flash ?? 1 });
        break;
      }
    }
  }
  return { main, sides };
}

/** Every channel of the arcs as drawn this frame, with how bright: [channel, gain]. */
export function arcChannels(arcs: Arcs, level: number): [Channel, number][] {
  const g = 0.55 + 0.45 * level;
  const out: [Channel, number][] = [];
  if (arcs.main.ch) out.push([arcs.main.ch, (arcs.main.flash ?? 1) * g]);
  for (const s of arcs.sides) if (s.ch && (s.flash ?? 1) > 0.02) out.push([s.ch, (s.flash ?? 1) * g]);
  return out;
}

/** The stretches a channel is drawn in: [points, width px, alpha], thinning along it and, a branch (dimmer than the
 * bolt's main channel), fading (TAPER) and thinning more (BRANCH_W) along it. */
export function stretches(c: Channel, gain: number): [P2[], number, number][] {
  const branch = c.inten < 0.6 ? Math.min(1, (0.6 - c.inten) / 0.2) : 0;
  const out: [P2[], number, number][] = [];
  const n = c.pts.length - 1;
  for (let k = 0; k < CHUNKS; k++) {
    const a = Math.floor((k * n) / CHUNKS), b = Math.floor(((k + 1) * n) / CHUNKS);
    if (b <= a) continue;
    const u = (k + 0.5) / CHUNKS;
    const thin = (1 - 0.6 * k / CHUNKS) * (1 - branch) + (BRANCH_W[0] + (BRANCH_W[1] - BRANCH_W[0]) * u) * branch;
    out.push([c.pts.slice(a, b + 1), Math.max(branch ? BRANCH_W[2] : 0, c.width * thin),
      c.inten * gain * (1 - TAPER[0] * branch * u ** TAPER[1])]);
  }
  return out;
}

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
const SOFT_Q = 4; // the glows and the hand's light are drawn on a canvas this many times smaller, blurred, laid over
const GLOWS: [number, number, string][] = [ // (blur in px of that canvas at 1080 lines, alpha, rgb): the air round a
  [0.8, 0.55, "110, 150, 255"], [2.5, 0.3, "70, 110, 255"], [7, 0.16, "95, 85, 255"], // channel glows blue, violet
]; // further out (a paler glow over a lit wall went white)
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

/** Draws channels, added: the blue air glowing round them (drawn small, blurred at three sizes, so a Gaussian-like
 * falloff in Canvas 2D at a sixteenth of the pixels; without a canvas for it, wide faint strokes), a violet-blue rim
 * and the core, white for a main channel (far over white: a phone clips it white with a blue rim) and bluer the
 * dimmer a branch, each channel in its stretches (thinning, a branch fading along it). ``px``: a pixel at 1080 lines;
 * ``most``: the widest glow, in px (a small ball's filaments glow no wider than it is). */
export function drawChannels(ctx: CanvasRenderingContext2D, channels: Channel[] | [Channel, number][], px: number,
  glow = 1, most = Infinity) {
  const list = (channels as (Channel | [Channel, number])[])
    .map((c): [Channel, number] => (Array.isArray(c) ? c : [c, 1]));
  const flat: [P2[], number, number, number][] = []; // (points, width, alpha, how much of a branch it is)
  const walk = (c: Channel, g: number) => {
    const branch = Math.min(1, Math.max(0, (0.6 - c.inten) / 0.4));
    for (const [pts, w, a] of stretches(c, g)) flat.push([pts, w, a, branch]);
    c.forks.forEach((f) => walk(f, g));
  };
  list.forEach(([c, g]) => walk(c, g));
  const lit = flat.filter(([pts, , a]) => pts.length > 1 && a > 0.01);
  ctx.save();
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
    for (const [sigma, amp, rgb] of GLOWS) {
      if (sigma * px * SOFT_Q > most) continue;
      air.setTransform(1, 0, 0, 1, 0, 0);
      air.clearRect(0, 0, air.canvas.width, air.canvas.height);
      air.setTransform(1 / SOFT_Q, 0, 0, 1 / SOFT_Q, 0, 0);
      for (const [pts, w, a] of lit) {
        air.strokeStyle = `rgba(${rgb}, ${Math.min(1, amp * a * glow)})`;
        air.lineWidth = Math.max(w, 2 * sigma * px * SOFT_Q); // (as wide as its blur: its peak stays near its alpha)
        stroke(air, pts);
      }
      blur.filter = `blur(${sigma * px}px)`;
      blur.drawImage(air.canvas, 0, 0);
    }
    ctx.drawImage(blur.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  } else {
    for (const [pts, w, a] of lit) {
      ctx.strokeStyle = `rgba(60, 90, 255, ${Math.min(1, 0.1 * a * glow)})`;
      ctx.lineWidth = Math.max(6 * px, 8 * w);
      stroke(ctx, pts);
    }
  }
  for (const [pts, w, a] of lit) { // the rim
    ctx.strokeStyle = `rgba(120, 140, 255, ${Math.min(1, 0.3 * a)})`;
    ctx.lineWidth = Math.max(2.5 * px, 2.5 * w);
    stroke(ctx, pts);
  }
  for (const [pts, w, a, branch] of lit) { // the core
    const [r, g] = [238 - 90 * branch, 243 - 70 * branch];
    ctx.strokeStyle = `rgba(${r}, ${g}, 255, ${Math.min(1, a * (1.2 - 0.4 * branch))})`;
    ctx.lineWidth = Math.max(0.9 * px, w);
    stroke(ctx, pts);
  }
  ctx.restore();
}

/** The burning fingertip: a white spark with a cyan-blue halo, added. */
export function drawSpark(ctx: CanvasRenderingContext2D, at: P2, r: number, amp: number) {
  if (amp <= 0) return;
  const g = ctx.createRadialGradient(at[0], at[1], 0, at[0], at[1], r);
  g.addColorStop(0, `rgba(240, 248, 255, ${Math.min(1, amp)})`);
  g.addColorStop(0.3, `rgba(150, 200, 255, ${0.6 * Math.min(1, amp)})`);
  g.addColorStop(1, "rgba(60, 110, 255, 0)");
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.fillStyle = g;
  ctx.fillRect(at[0] - r, at[1] - r, 2 * r, 2 * r);
  ctx.restore();
}

// ---- the light it throws
export const LIGHT_HOLD = 4, LIGHT_LETGO = 0.55; // frames the light holds the strokes' peak, then lets go by this a
// frame (following each stroke, the room strobed with the channels; letting go slowly, it glowed with no source)
const LIGHT_GONE = 0.02;

/** The light's state: the last LIGHT_HOLD frames' power and the light held from them. */
export interface Light { recent: number[]; held: number }

/** One frame of the light the lightning throws, for this frame's ``power`` (0-1: how much is striking): it holds the
 * strokes' peak over LIGHT_HOLD frames and lets go by LIGHT_LETGO a frame once they stop, gone under LIGHT_GONE. */
export function lightStep(l: Light | null, power: number): Light {
  const recent = [...(l?.recent ?? []), power].slice(-LIGHT_HOLD);
  const now = Math.max(...recent);
  let held = Math.max(now, (l?.held ?? 0) * LIGHT_LETGO);
  if (now <= 0 && held <= LIGHT_GONE) held = 0;
  return { recent, held };
}

export const DIP = [0.5, 4, 24]; // stops, frames rising, frames falling: the camera exposes down for the ball (as a
// phone's does for a bright light in the picture), so its body is the brightest thing over the palm, settling for a
// phrase rather than breathing with the voice (with the palm as bright as ever, the lit skin round it outshone it: a
// glass marble)

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

export const ROOM_RGB = "120, 150, 255"; // the arcs' light on the room: a cool blue (in the hand's violet, a tinted
export const HAND_RGB = "185, 160, 255"; // wall went mauve); on the hand lavender, blue-violet on skin
export const BALL_ROOM_RGB = "110, 116, 128"; // the ball's, on the room and the hand alike: the cool near-white of its
export const BALL_HAND_RGB = BALL_ROOM_RGB; // plasma (cuesync.lightning's PLASMA_RGB) at half strength, which brightens
// skin and cools it a little (in a deep blue, skin already at the top of red took only the blue, and was dyed fuchsia;
// at full strength, screened, the palm round it went white)

/** The light on the room and the hand, screened over the picture: round ``at`` (the bolts or the ball) on the room,
 * weaker out to the frame's edge; on the hand (its 21 landmarks ``lm``, px: the palm and each finger as thick as a
 * finger, drawn small and soft) round the same place, strong close in, times ``handGain``. ``strength``: the held
 * light (0-1). */
export function drawLight(ctx: CanvasRenderingContext2D, at: P2, scale: number, strength: number, lm: P2[] | null,
  room: string, onHand: string, w: number, h: number, handGain = 1) {
  if (strength <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  const g = ctx.createRadialGradient(at[0], at[1], 0, at[0], at[1], 0.8 * Math.hypot(w, h));
  g.addColorStop(0, `rgba(${room}, ${0.4 * strength})`);
  g.addColorStop(0.3, `rgba(${room}, ${0.18 * strength})`);
  g.addColorStop(1, `rgba(${room}, ${0.06 * strength})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  const mask = lm ? handLayer(ctx, "hand", lm, scale) : null;
  if (mask) {
    const k = mask.createRadialGradient(at[0], at[1], 0, at[0], at[1], 2.5 * scale);
    k.addColorStop(0, `rgba(${onHand}, ${0.75 * handGain * strength})`);
    k.addColorStop(1, `rgba(${onHand}, ${0.15 * handGain * strength})`);
    mask.fillStyle = k;
    mask.fillRect(0, 0, w, h);
    ctx.drawImage(mask.canvas, 0, 0, w, h);
  }
  ctx.restore();
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

// ---- the ball over the palm
const BALL_R = [0.1, 0.42]; // hand sizes: its radius, from a point to one held long and loud
export const BALL_RISE = 0.2; // its growth (0-1) rises at most this much a frame: from a crackle to full over ~5 frames
const BALL_GROW = 10, BALL_SHRINK = 3; // frames: its charge follows the voice up this slowly, down this fast
const BALL_ON = [0.03, 0.015]; // it shows once the voice or its charge passes the first, goes once both are under
const BALL_FORMED = 0.08; // its charge at which its body is dense: under half of it, only its filaments' crackle
const FIL_REACH = 0.7; // radii: a young ball's filaments crackle this much further out than its body
const BALL_PULSE = 0.12; // a hit swells it by this share, gone over 3 frames
const BALL_TURN = 0.02; // radians a frame: it turns slowly (the shot's 0.05 at 24 frames a second, on a 60 Hz screen)
export const BALL_FILAMENTS = [5, 9]; // filaments from its middle to its skin, small to full grown, each leaving its
// middle at a random angle in 3D, its depth scaled by BALL_DEPTH, so they reach its skin at different lengths as seen
// (all reaching the same radius, a dozen made evenly spaced sun-ray spikes)
const BALL_DEPTH = 0.45, BALL_FLAT = 0.5; // one that reaches less than this many radii across the picture is struck
// again (pointing at the camera, it is a short scribble in the middle)
const NEAR_LIGHT = [0.6, 0.73, 3.5]; // its light on what is close round it: this strong at its edge, falling off
// with distance d (radii) as (1 + h^2) / (d^2 + h^2), h how high it hovers over the palm (radii), the softened inverse
// square of cuesync.lightning's, gone by the last: the skin next to it brighter, the fingertips far from it barely
const HALO: [number, number, string] = [0.25, 0.7, "170, 166, 255"]; // the faint blue-violet haze of its light round
// it, added over skin and wall alike: at most this, falling off as a Gaussian of this sigma (radii; cuesync.lightning's
// HALO)

/** The ball's state between frames; each filament's ``aim`` is how far towards the camera its far end is (-1 to 1:
 * those at the back are dimmer). */
export interface Ball { charge: number; grown: number; born: boolean; pulse: number; flare: number; level: number;
  fils: Stream[]; turn: number }

/** A normal deviate from ``rand`` (Box-Muller). */
function gauss(rand: () => number): number {
  return Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-12))) * Math.cos(2 * Math.PI * rand());
}

/** How much of a body a ball has (0-1): none until its charge (and the voice) pass half BALL_FORMED. */
export function dense(level: number, charge: number): number {
  return Math.max(0, Math.min(1, (2 * (0.3 * level + 0.7 * charge)) / BALL_FORMED - 1));
}

/** One frame of the ball for a voice ``level`` (0-1), or null when it is gone: its charge follows the voice, its body
 * forms at its least size and grows at most BALL_RISE a frame towards the square root of its charge, a hit swells it
 * and flares its middle; its filaments (BALL_FILAMENTS, from its middle to past its skin while it is young) are struck
 * and re-struck as channels are, turning slowly with it. */
export function ballStep(b: Ball | null, level: number, hit: boolean, frame: number): Ball | null {
  const prev = b?.charge ?? 0;
  const charge = prev + (level - prev) * (1 - Math.exp(-1 / (level > prev ? BALL_GROW : BALL_SHRINK)));
  if (Math.max(level, charge) < BALL_ON[b ? 1 : 0]) return null;
  const born = (b?.born ?? false) || dense(level, charge) > 0;
  const grown = b?.born ? Math.min(Math.sqrt(charge), b.grown + BALL_RISE) : 0;
  const count = Math.round(BALL_FILAMENTS[0] + (BALL_FILAMENTS[1] - BALL_FILAMENTS[0]) * charge);
  const reach = 1 + FIL_REACH * (1 - grown);
  const fils: Stream[] = [];
  for (let i = 0; i < count; i++) {
    const old = b?.fils[i];
    const rand = rng((i + 1) * 104729 + frame);
    if (old?.ch && old.left > 0 && !hit) {
      const on = rand() < RESTROKE[0] + RESTROKE[1] * level;
      fils.push({ ...old, left: old.left - 1, flash: on ? 0.55 + 0.35 * rand() : 0.1 * rand() });
      continue;
    }
    let d = [0, 0, 1];
    for (let tries = 0; tries < 6 && Math.hypot(d[0], d[1]) < BALL_FLAT; tries++) {
      const v = [gauss(rand), gauss(rand), BALL_DEPTH * gauss(rand)];
      const l = Math.hypot(v[0], v[1], v[2]) || 1;
      d = v.map((x) => x / l);
    }
    const far = reach * (0.92 + 0.13 * rand());
    const from: P2 = [0.05 * (rand() - 0.5), 0.05 * (rand() - 0.5)];
    fils.push({ ch: channel(from, [d[0] * far, d[1] * far], rand, 1, 0.22, 5, 0.25, { clear: 0.02 }),
      left: life(rand) - 1, aim: d[2], seed: i + 1, flash: 1 });
  }
  return { charge, grown, born, pulse: hit ? BALL_PULSE : (b?.pulse ?? 0) * 0.6, flare: hit ? 1 : 0, level, fils,
    turn: (b?.turn ?? 0) + BALL_TURN };
}

/** The ball's radius in px for a hand ``scale`` px across. */
export function ballRadius(b: Ball, scale: number): number {
  const g = b.grown;
  return scale * (BALL_R[0] + (BALL_R[1] - BALL_R[0]) * g * g * (3 - 2 * g)) * (1 + b.pulse);
}

/** The body's colour from its middle out, as gradient stops [radii, r, g, b, alpha]: the shot's ball as shown
 * (cuesync.lightning, the median of each ring at full charge: test_lightning's BALL_PROFILE), white-hot in its middle
 * and a pale cyan-blue round that, the lit palm showing through towards its edge; dense in its middle, light over the
 * palm towards its edge (so the palm shows through it there), and gone a little past its radius with no outline. */
export const BODY_STOPS: [number, number, number, number, number][] = [
  [0, 253, 254, 255, 1], [0.15, 248, 251, 255, 1], [0.3, 230, 242, 255, 1], [0.45, 211, 234, 255, 1],
  [0.6, 210, 233, 255, 0.97], [0.75, 224, 234, 255, 0.88], [0.9, 244, 234, 252, 0.65], [1, 240, 228, 250, 0.35],
  [1.15, 236, 222, 248, 0],
];

/** Draws the ball at ``c`` for a hand ``scale`` px across: its body (once formed, BODY_STOPS), the faint blue-violet
 * haze of its light round it (HALO), its filaments white over it (dim inside, brightest at its skin, those at the back
 * dimmer) and its white-hot middle. ``px``: a pixel at 1080 lines. */
export function drawBall(ctx: CanvasRenderingContext2D, b: Ball, c: P2, scale: number, px: number) {
  const R = ballRadius(b, scale);
  const body = b.born ? dense(b.level, b.charge) : 0;
  ctx.save();
  if (body > 0) {
    const reach = BODY_STOPS[BODY_STOPS.length - 1][0] * R;
    const g = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach);
    for (const [at, r, gg, bb, a] of BODY_STOPS) {
      g.addColorStop((at * R) / reach, `rgba(${r}, ${gg}, ${bb}, ${a * body})`);
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(c[0], c[1], reach, 0, 2 * Math.PI);
    ctx.fill();
    const [k, sigma, rgb] = HALO;
    const far = 3.5 * sigma * R;
    const haze = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], far);
    for (let i = 0; i <= 8; i++) { // (a Gaussian, in nine stops out to 3.5 sigma, from its edge out: added over its
      const d = (3.5 * sigma * i) / 8; // body in the picture as shown, it washed it white)
      haze.addColorStop(i / 8, `rgba(${rgb}, ${d < 0.85 ? 0 : k * body * Math.exp((-0.5 * d * d) / (sigma * sigma))})`);
    }
    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = haze;
    ctx.fillRect(c[0] - far, c[1] - far, 2 * far, 2 * far);
  }
  const turn = b.turn, cs = Math.cos(turn), sn = Math.sin(turn);
  const place = (p: P2): P2 => [c[0] + R * (p[0] * cs - p[1] * sn), c[1] + R * (p[0] * sn + p[1] * cs)];
  const lines: [Channel, number][] = [];
  const width = Math.max(0.8, widthK(scale, px) * px * (0.8 + 0.6 * b.level));
  for (const f of b.fils) {
    if (!f.ch || (f.flash ?? 1) <= 0.02) continue;
    const pts = f.ch.pts.map(place);
    const j = Math.floor(0.2 * pts.length), k = Math.floor(0.45 * pts.length); // (lost in the white-hot middle, dim
    // round it, brightest at its skin: drawn from its middle at one brightness, they met there as a star's rays)
    const gain = (f.flash ?? 1) * (0.4 + 0.6 * b.level) * (0.55 + 0.45 * Math.max(-1, Math.min(1, f.aim)));
    lines.push([{ pts: pts.slice(j, k + 1), width: 0.7 * width, inten: 1, forks: [], at: 0 }, 0.3 * gain]);
    const forks = f.ch.forks.map((o) => ({ ...o, pts: o.pts.map(place), width: 0.6 * width }));
    lines.push([{ pts: pts.slice(k), width, inten: 1, forks, at: 0 }, gain]);
  }
  drawChannels(ctx, lines, px, 0.6, 0.3 * R);
  ctx.restore();
  drawSpark(ctx, c, 0.16 * R + 3 * px, (body ? 0.9 : 0.5) * (1 + b.flare) * (0.5 + 0.5 * b.level)); // (its body is
  // white-hot round it already: wider, it was a white star the filaments ran into like rays)
}

/** The ball's light on what is close round it (NEAR_LIGHT), screened under it: the cool near-white of its plasma on
 * the room and on the hand (its 21 landmarks ``lm``, px, or null), which brightens skin and cools it a little, keeping
 * its lines; strongest at its edge and falling off as the softened inverse square of the distance, so the fingertips
 * far from it are barely lit; as strong as its body is dense. */
export function drawBallLight(ctx: CanvasRenderingContext2D, b: Ball, c: P2, scale: number, lm: P2[] | null,
  w: number, h: number) {
  const body = b.born ? dense(b.level, b.charge) : 0;
  if (body <= 0) return;
  const R = ballRadius(b, scale);
  const [k, hover, reach] = NEAR_LIGHT;
  const falloff = (fill: CanvasGradient, rgb: string, gain: number) => {
    for (let i = 0; i <= 8; i++) { // (from its edge out to ``reach`` radii, faded out over the last one)
      const d = 1 + ((reach - 1) * i) / 8, fade = Math.min(1, reach - d);
      const a = (gain * k * body * (1 + hover * hover)) / (d * d + hover * hover);
      fill.addColorStop(i / 8, `rgba(${rgb}, ${a * fade * fade * (3 - 2 * fade)})`);
    }
  };
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  const room = ctx.createRadialGradient(c[0], c[1], R, c[0], c[1], reach * R);
  falloff(room, BALL_ROOM_RGB, 0.4);
  ctx.fillStyle = room;
  ctx.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
  const mask = lm ? handLayer(ctx, "ball-hand", lm, scale) : null;
  if (mask) {
    const hand = mask.createRadialGradient(c[0], c[1], R, c[0], c[1], reach * R);
    falloff(hand, BALL_HAND_RGB, 0.7);
    mask.fillStyle = hand;
    mask.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
    ctx.drawImage(mask.canvas, 0, 0, w, h);
  }
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
