/** Live lightning for Try it live: arcs struck from a fingertip, or a ball of light over an open palm, drawn on the
 * camera's canvas every frame next to the hand tracker, so everything here is a few hundred points and a few
 * gradients a frame. As the shot's own (cuesync.lightning), as far as Canvas 2D goes:
 * - a finger strikes one dominant main channel, its first stretch along the finger, its root welded to the fingertip;
 *   beside it at most two dim, thin streamers at most half as long, keeping well round from it, and dim forks towards
 *   its far end, every branch fading and thinning along its length and none crossing the main channel or another
 *   branch, nor a streamer running beside it (each is struck again until it keeps clear, or dropped);
 * - a new main channel strikes an edge of the camera's picture in its way (a floor's line, a door's jamb, a table's
 *   edge: a clear line, well inside the frame, within a cone round where the finger points), one of the best few each
 *   time, with a contact flash there (a white-hot point, a pool of its light along that surface, a splash of forks on
 *   its first frames); with nothing to strike it ends in the air in a brush of dim forks, never in a bead;
 * - a channel is struck, re-struck down the same path for 1-3 frames with its root riding the finger and its far end
 *   where it struck, then struck anew near its last aim;
 * - the light it throws, a cool blue on the room and a cool white on the hand, holds while the strokes strike and
 *   goes with the last of them (a tenth for one frame);
 * - over an open palm, a ball that is a light hovering off it, its body the shot's own by radius (white-hot in its
 *   middle, cyan round that, a saturated blue to its limb, its outline feathered), moving plasma in it (filaments
 *   turning through it), a faint blue haze of its light round it, a shell of jagged arcs crawling over its skin in 3D
 *   (brightest along its outline, which they draw) and arcs bowing through the air from it to the fingertips; its
 *   light on the skin round it a deep blue and on the wall a blue-white, growing from a crackle over about 5 frames,
 *   the camera exposing down for it. */

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

// ---- where it strikes (cuesync.lightning's TARGET_*, EDGE_*, IMPACT, POOL, SPLASH, BRUSH)
export const TARGET_CONE = [25, 40, 70]; // degrees either side of the finger's aim: a burst's first strike an edge
// within the first, else in the second; struck again, anywhere within the third, never where it struck last
export const TARGET_AT = [0.22, 0.4]; // the canvas's diagonals: how far out it strikes, quietest voice to loudest,
// preferring what is nearest that
const TARGET_BAND = [0.5, 1.5]; // shares of that: how far out what it strikes may be
export const TARGET_BORDER = 0.12; // share of the picture's short side: never this close to its edge
export const TARGET_TOP = 3, TARGET_APART = 150; // one of the best this many, px at 1080 apart and from where it
// struck last: a burst's strikes go back and forth between two places at least
export const PRONG = [0.25, 90, 240, 0.72, 0.85, 0.8]; // (chance; px at 1080 apart, its range; share along, its
// range; against its channel): one main channel in four that strikes an edge ends in two prongs, landing on two points
const EDGE_Q = 8; // the picture is read at 1 / EDGE_Q of the canvas
const EDGE_MIN = 40; // the gradient (Sobel, code values, at that size, softened) a line must reach to be struck: a step
// of about 15 levels (a blank wall's grain and the camera's noise stay under it)
const EDGE_NEAR = 3; // ...and at least this many more edge px round it (in its 5 x 5): a line, not a speck
export const IMPACT_MIN = 0.6; // a stroke at least this bright strikes with a contact flash (a weak one with none)
const IMPACT_PX = 2.5; // px at 1080: the white-hot point where it touches
const POOL = [150, 0.33, 0.6]; // (px at 1080 along the surface, its width across against that, alpha): its light
// pooled along the surface it strikes (at 110 px and 0.45 it barely lit the floor)
export const SPLASH = [3, 6, 30, 80, 0.55, 0.8, 0.45, 2]; // (count, its range; length px at 1080, its range;
// intensity against its stroke, its range; width against its root's; frames): forks splashing out where it strikes
export const BRUSH = [2, 3, 0.06, 0.14, 0.3, 0.5, 0.26, 0.7]; // (count, its range; length in its own, its range;
// intensity, its range; radians turned, its range): a channel that strikes nothing ends in the air in a brush of dim
// forks fanning on from its end

/** A channel: its points, width (px), brightness against its bolt's main channel (1), forks, the parent's point it
 * forks from (``at``), and (a main channel) ``strikes``: which way the surface its far end strikes runs, or none. */
export interface Channel { pts: P2[]; width: number; inten: number; forks: Channel[]; at: number; strikes?: P2 }

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

// ---- the picture it strikes into
/** The camera's picture, small: its luminance (0-255) ``w`` x ``h``, each of its px ``k`` canvas px. */
export interface Plate { w: number; h: number; k: number; lum: Float32Array }

/** The picture the lightning was last drawn over (drawChannels reads it), where a new bolt looks for what to strike:
 * a frame old, as a strike is struck before it is drawn. */
let seen: Plate | null = null;
const small = new WeakMap<object, Ctx2D>();

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
export interface Hit { at: P2; along: P2 }

/** Where a new main channel from ``root`` aimed ``aim`` (radians) strikes, preferring ``want`` px out, and the way
 * the surface runs there, or null; with ``prong``, a second point it lands on too (``two``). Of the ``edges``
 * (surfaces) TARGET_BAND of ``want`` out, TARGET_BORDER inside the ``size`` (w, h) frame and within TARGET_CONE of
 * the aim (a burst's first strike within its first cone, else its second; struck again, after one that struck at
 * ``last``, within its third and TARGET_APART from there): the strongest, nearest ``want`` out and nearest its
 * ``heading``, with a little chance; then one of the best TARGET_TOP, TARGET_APART from each other. The second point:
 * the best of those PRONG's px from the first. ``px``: a pixel at 1080 lines. */
export function pickTarget(edges: number[][], root: P2, aim: number, heading: number, want: number, size: P2,
  rand: () => number, px = 1, last: P2 | null = null, prong = false): (Hit & { two?: Hit }) | null {
  const [w, h] = size;
  const border = TARGET_BORDER * Math.min(w, h), apart = TARGET_APART * px;
  const offOf = (d: number) => ((d + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
  const cones = last ? TARGET_CONE.slice(2) : TARGET_CONE.slice(0, 2);
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
 * along the aim and its heading never more than ARC_OFF off it, at an edge of the ``plate`` in its way about ``want``
 * px out (pickTarget: a contact flash and a splash there, somewhere new each time, one in four in two prongs) or, with
 * none, ``reach`` px long, ending in the air in a brush. ``level``: the voice, which sets how often a re-stroke
 * flashes; ``px``: a pixel at 1080 lines. */
export function streamStep(s: Stream, root: P2, aim: number, reach: number, width: number, strike: boolean,
  frame: number, level = 1, plate: Plate | null = null, px = 1, want = 0.7 * reach): Stream {
  const rand = rng(s.seed * 7919 + frame);
  if (s.ch && s.left > 0 && !strike) {
    const on = rand() < RESTROKE[0] + RESTROKE[1] * level;
    return { ...s, ch: ride(s.ch, root), left: s.left - 1, flash: on ? 0.55 + 0.35 * rand() : 0.1 * rand(),
      age: (s.age ?? 0) + 1 };
  }
  const off = s.ch ? s.aim - aim + (rand() - 0.5) * 2 * ARC_JITTER : (rand() - 0.5) * 2 * ARC_JITTER; // (near its last)
  const a = aim + Math.max(-ARC_OFF, Math.min(ARC_OFF, off * 0.6));
  const l = reach * (0.8 + 0.25 * rand());
  const hit = plate ? pickTarget(surfaces(plate), root, aim, a, want * (0.9 + 0.2 * rand()),
    [plate.w * plate.k, plate.h * plate.k], rand, px, s.hitAt ?? null, rand() < PRONG[0]) : null;
  const far: P2 = hit ? hit.at : [root[0] + Math.cos(a) * l, root[1] + Math.sin(a) * l];
  const span = Math.hypot(far[0] - root[0], far[1] - root[1]);
  const lead: P2 = [root[0] + Math.cos(aim) * ARC_LEAD * span, root[1] + Math.sin(aim) * ARC_LEAD * span];
  const ch = channel(root, far, rand, width, 0.26, 6, 0.1 + 0.1 * level, { lead });
  if (hit) ch.strikes = hit.along;
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
  return { ch, left: life(rand) - 1, aim: a, seed: s.seed, flash: 1, splash: hit ? splash(ch, hit.along, rand, px) : [],
    age: 0, hitAt: hit ? hit.at : s.hitAt };
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
 * voice ``level`` (0-1); ``px``: a pixel at 1080 lines; ``plate``: the picture they strike into (by default the one
 * the lightning was last drawn over). The main channel is welded to the tip; its streamers (one over SIDE_ON's first
 * level, two over its second) leave its root SIDE_GAP off the main channel's way out on either side, at most
 * SIDE_REACH of its reach, struck anew with it or when their own time is up, and struck again (REDRAW times, then
 * dropped) until they cross neither it nor each other, run beside it no further than ALONG lets them and keep
 * SIDE_APART round from it. */
export function arcStep(arcs: Arcs, tip: P2, aim: number, scale: number, level: number, strike: boolean,
  frame: number, px = 1, plate: Plate | null = seen): Arcs {
  const kw = widthK(scale, px) * px;
  const width = kw * (ARC_WIDTH[0] + (ARC_WIDTH[1] - ARC_WIDTH[0]) * level);
  const diag = plate ? Math.hypot(plate.w * plate.k, plate.h * plate.k) : 1.6 * 1080 * px; // (the canvas's, as the
  // shot's are: before the picture has been read, a 4:3 one)
  const reach = diag * (ARC_REACH[0] + (ARC_REACH[1] - ARC_REACH[0]) * level);
  const want = diag * (TARGET_AT[0] + (TARGET_AT[1] - TARGET_AT[0]) * level);
  const main = streamStep(arcs.main, tip, aim, reach, width, strike, frame, level, plate, px, want);
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

/** Every channel of the arcs as drawn this frame, with how bright: [channel, gain] (the main channel's splash on its
 * first SPLASH frames). */
export function arcChannels(arcs: Arcs, level: number): [Channel, number][] {
  const g = 0.55 + 0.45 * level;
  const out: [Channel, number][] = [];
  const m = arcs.main;
  if (m.ch) out.push([m.ch, (m.flash ?? 1) * g]);
  if (m.ch && (m.age ?? 0) < SPLASH[7]) for (const s of m.splash ?? []) out.push([s, (m.flash ?? 1) * g]);
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
/** How channels are drawn: their glows ((blur in px of the small canvas at 1080 lines, alpha, rgb), the rim round
 * their core, and how much bluer a dim branch's core goes (red, green taken off). */
export interface Look { glows: [number, number, string][]; rim: string; branch: [number, number] }
export const ARC_LOOK: Look = { // the arcs: the air round a channel glows blue, violet further out (a paler glow over a
  glows: [[0.8, 0.55, "110, 150, 255"], [2.5, 0.3, "70, 110, 255"], [7, 0.16, "95, 85, 255"]], // lit wall went white)
  rim: "120, 140, 255", branch: [90, 70],
};
export const BALL_LOOK: Look = { // the ball's arcs: white-hot in a cyan-blue glow, as the shot's BALL_GLOW_RGB (in the
  glows: [[0.8, 0.5, "120, 205, 255"], [2.5, 0.28, "80, 170, 255"], [7, 0.14, "60, 140, 255"]], // arcs' violet, over
  rim: "150, 215, 255", branch: [40, 20], // skin they were bright lavender)
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

/** The contact flash where a channel strikes a surface running ``along`` at ``at``, ``amp`` bright, added: its light
 * pooled along the surface (an ellipse falling off from where it strikes) and the white-hot point where it touches. */
function drawImpact(ctx: CanvasRenderingContext2D, at: P2, along: P2, amp: number, px: number) {
  const [len, across, alpha] = POOL;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.translate(at[0], at[1]);
  ctx.rotate(Math.atan2(along[1], along[0]));
  ctx.scale(1, across);
  const r = len * px;
  const pool = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
  pool.addColorStop(0, `rgba(170, 195, 255, ${alpha * amp})`);
  pool.addColorStop(0.15, `rgba(120, 160, 255, ${0.45 * alpha * amp})`);
  pool.addColorStop(0.45, `rgba(90, 130, 255, ${0.15 * alpha * amp})`);
  pool.addColorStop(1, "rgba(80, 120, 255, 0)");
  ctx.fillStyle = pool;
  ctx.fillRect(-r, -r, 2 * r, 2 * r);
  ctx.restore();
  drawSpark(ctx, at, IMPACT_PX * 2 * px, amp);
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
    for (const [sigma, amp, rgb] of look_.glows) {
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
    ctx.strokeStyle = `rgba(${look_.rim}, ${Math.min(1, 0.3 * a)})`;
    ctx.lineWidth = Math.max(2.5 * px, 2.5 * w);
    stroke(ctx, pts);
  }
  for (const [pts, w, a, branch] of lit) { // the core
    const [r, g] = [238 - look_.branch[0] * branch, 243 - look_.branch[1] * branch];
    ctx.strokeStyle = `rgba(${r}, ${g}, 255, ${Math.min(1, a * (1.2 - 0.4 * branch))})`;
    ctx.lineWidth = Math.max(0.9 * px, w);
    stroke(ctx, pts);
  }
  ctx.restore();
  const impacts = (c: Channel, g: number) => { // (a channel's, and a prong's of it)
    if (c.strikes && c.inten * g >= IMPACT_MIN) drawImpact(ctx, c.pts[c.pts.length - 1], c.strikes, c.inten * g, px);
    c.forks.forEach((f) => impacts(f, g));
  };
  for (const [c, g] of list) impacts(c, g);
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

export const ROOM_RGB = "120, 150, 255"; // the arcs' light on the room: a cool blue (in the hand's violet, a tinted
export const HAND_RGB = "175, 195, 255"; // wall went mauve); on the hand a cool white (in a lavender, the fingertips
// went hot pink)
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
const BALL_R = [0.12, 0.5]; // hand sizes: its radius, from a point to one held long and loud
export const BALL_RISE = 0.2; // its growth (0-1) rises at most this much a frame: from a crackle to full over ~5 frames
const BALL_GROW = 6, BALL_SHRINK = 3; // frames: its charge follows the voice up this slowly, down this fast
const BALL_ON = [0.03, 0.015]; // it shows once the voice or its charge passes the first, goes once both are under
const BALL_FORMED = 0.08; // its charge at which its body is dense: under half of it, only its shell's crackle
const FIL_REACH = 0.7; // radii: a young ball's shell crackles this much further out than its body
const BALL_PULSE = 0.12; // a hit swells it by this share, gone over 3 frames
const BALL_TURN = 0.02; // radians a frame: it turns slowly (the shot's 0.05 at 24 frames a second, on a 60 Hz screen)
const BALL_LIFT = 150; // px at 1080: it hovers this far up the picture off the palm (on it, it was a sticker on skin)
export const SHELL = [10, 18]; // arcs crawling over its skin, small to full grown: the shell of filaments that draws
// its outline (filaments from its middle out to its skin were a radial starburst: a lens flare)
const SHELL_OUTLINE = 0.5, SHELL_SPAN = [0.87, 2.27]; // how often a shell arc runs along its outline; radians of its
// great circle it runs over
const SHELL_INNER = [0.3, 0.5, 0.85]; // share, radii: of them, this many crawl over a sphere this far in, seen through
// its body (dimmer, SHELL_INNER_INTEN)
const SHELL_INNER_INTEN = 0.55, SHELL_BACK = 0.4, SHELL_FACE = 0.35; // an inner arc's brightness; a stretch behind its
// middle; one facing the camera against one along its outline, where the shell is seen edge on and shows densest
export const SHELL_SKIN = 0.05; // radii: a shell arc is held this close to its sphere, however jagged (thrown past it,
// its arcs lay on the palm like glowing veins)
export const GROUND = [4, 6]; // arcs jumping from its skin to the fingertips, quiet to loud at its full size, struck
// every frame or two: two or three a side (at 6-10, each with its dot, it read as a spider on white legs), as many fewer
// as it is smaller, none once it has shrunk under GROUND_GONE of the largest it grew to
export const GROUND_GONE = 0.3;
const TIPS = [4, 8, 12, 16, 20]; // the landmarks they go to: the fingertips (to the joints and bases too, they ran down
// the skin to a dot mid-finger)
export const GROUND_BOW = [30, 60, 110]; // px at 1080: each bows through the air, its middle this far off its straight
// line either way, the first two its range, out to the third once those all lie on the hand
const GROUND_LEAVE = 1.05; // radians: it leaves the ball's skin at most this far round from the way to its fingertip
export const GROUND_FREE = 1.2, GROUND_ON = 0.04; // radii, share: out to the first from its middle an arc may pass over
// the hand; past it at most the second share of it lies on the hand but for where it lands, or it is drawn again, at
// most GROUND_TRIES times (bowed up the picture whatever lay there, they lay on the skin like drawn veins)
const GROUND_TRIES = 20;
export const LAND_IN = 12, LAND_NEAR = 25; // px at 1080: it comes down onto the finger from the air past its tip,
// landing this far inside the finger, at most LAND_NEAR from the fingertip's landmark
const LAND = [2.5, 9]; // px at 1080: where it lands, a white-hot point this wide in a blue glow this wide (a white spot
// over red skin read as a pink dot)
const SHELL_GONE = [0.45, 0.3]; // shares of the largest it grew to: shrinking, its shell fades out between these, and
// with its body once it has been a ball (its crackle crawled over the fingers with no source)
export const BALL_LIGHT = [0.75, 0.25, 40, 1, 3, 0.9]; // its light round it (cuesync.lightning's BALL_LIGHT): (alpha on
// the hand by its edge, on the room; px at 1080 its middle is over the palm, radii the wall is behind it; radii it is
// gone by; radii of its dense middle, which hides what it lights there): from its middle as from a ball of its
// radius, R^2 / (d^2 + h^2) at most 1, a cool white on the hand and a blue-white on the room (spread over the frame, a
// deep blue, its light tinted the hand lavender and graded the room)
const BALL_LIGHT_RGB = ["225, 240, 255", "150, 190, 255"]; // (on the hand, on the room)
const HALO: [number, number, string] = [0.22, 0.8, "20, 90, 255"]; // the faint blue haze of its light round it, added
// over skin and wall alike: at most this, falling off as a Gaussian of this sigma (radii; cuesync.lightning's HALO)
export const PLASMA = [0.45, 1.5]; // (contrast, turns a second): its body is moving plasma, filaments of brighter and
// darker light turning through it (cuesync.lightning's PLASMA), not an airbrushed marble
const PLASMA_LAYERS = [[1, 1, 0.2, 1, 0.3], [-0.75, 0.6, -0.5, 1, 0.2]]; // (rate, share, axis x, y, z): its two
// layers of filaments, each turning about its own axis through its middle; seen on the sphere, front and (fainter)
// back, so they wrap it (drawn as a flat tile turned in the picture, they read as camouflage on a disc)
const PLASMA_PX = 72; // the plasma is worked out this many px across, and drawn up to the ball's size

/** An arc of the ball's shell: its points on the unit sphere (3D, z towards the camera), turned with the ball, and
 * how bright (an inner one dimmer); struck and re-struck as a channel is. */
export interface Shell { pts: P3[]; inten: number; left: number; flash: number }

/** An arc jumping from the ball to a fingertip: its shape from (0, 0) to (1, 0), the landmark it goes to, how far it
 * bows (px at 1080, to its chord's left, or its right under 0) and how far round the ball's skin it leaves (radians),
 * struck and re-struck as a channel is; ``seed``: where it looks for another way when this one lies on the hand. */
export interface Jump { ch: Channel; to: number; lift: number; leave: number; left: number; flash: number; seed: number }

/** The ball's state between frames: ``top``, its largest radius (hand sizes) and ``most``, how grown it has been at
 * most, while it shows; ``fade``: how much of its shell shows as it dies. */
export interface Ball { charge: number; grown: number; born: boolean; pulse: number; flare: number; level: number;
  shell: Shell[]; jumps: Jump[]; turn: number; frame: number; top: number; most: number; fade: number }

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

/** How much of a body a ball has (0-1): none until its charge (and the voice) pass half BALL_FORMED. */
export function dense(level: number, charge: number): number {
  return Math.max(0, Math.min(1, (2 * (0.3 * level + 0.7 * charge)) / BALL_FORMED - 1));
}

/** A shell arc (cuesync.lightning's _shell): a jagged stretch of a great circle SHELL_SPAN long, SHELL_OUTLINE of them
 * lying across the picture along its outline, SHELL_INNER's share on a sphere inside it, every point held within
 * SHELL_SKIN of its sphere. */
function shellArc(rand: () => number): P3[] {
  const outline = rand() < SHELL_OUTLINE;
  const inner = rand() < SHELL_INNER[0];
  const r = inner ? SHELL_INNER[1] + (SHELL_INNER[2] - SHELL_INNER[1]) * rand() : 1;
  const nrm = outline ? unit3([0.25 * gauss(rand), 0.25 * gauss(rand), 1])
    : unit3([gauss(rand), gauss(rand), gauss(rand)]);
  const e1 = unit3(cross3(nrm, [gauss(rand), gauss(rand), gauss(rand)]));
  const e2 = cross3(nrm, e1);
  const start = 2 * Math.PI * rand(), span = SHELL_SPAN[0] + (SHELL_SPAN[1] - SHELL_SPAN[0]) * rand();
  const n = 33;
  const pts: P3[] = [];
  let wob = 0, vel = 0;
  for (let i = 0; i < n; i++) {
    const phi = start + (span * i) / (n - 1);
    vel = 0.6 * vel + 0.09 * gauss(rand); // (a jagged wander off the circle, across it)
    wob = Math.max(-0.35, Math.min(0.35, wob + vel));
    const p: P3 = [0, 1, 2].map((k) => (Math.cos(phi) * e1[k] + Math.sin(phi) * e2[k]) * r + wob * nrm[k] * r) as P3;
    const l = Math.hypot(...p) || 1;
    const held = Math.max(r - SHELL_SKIN, Math.min(r + SHELL_SKIN, l)) / l;
    pts.push([p[0] * held, p[1] * held, p[2] * held]);
  }
  return pts;
}

/** One frame of the ball for a voice ``level`` (0-1), or null when it is gone: its charge follows the voice, its body
 * forms at its least size and grows at most BALL_RISE a frame towards the square root of its charge, a hit swells it
 * and flares its middle; its shell (SHELL arcs) and, once it has a body, its arcs to the fingertips (GROUND, as many
 * as it is large) are struck and re-struck as channels are, every frame or two, turning slowly with it; dying, its
 * shell fades out with it (SHELL_GONE). */
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
    shell.push({ pts, inten: Math.hypot(...pts[16]) < 0.9 ? SHELL_INNER_INTEN : 1, left: life(rand) - 1, flash: 1 });
  }
  const jumps: Jump[] = [];
  const count = body > 0 && size >= GROUND_GONE * top
    ? Math.round((GROUND[0] + (GROUND[1] - GROUND[0]) * level) * Math.min(1, size / BALL_R[1])) : 0;
  for (let i = 0; i < count; i++) {
    const old = b?.jumps[i], rand = rng((i + 1) * 15485863 + frame);
    if (old && old.left > 0 && !hit) { jumps.push({ ...old, left: old.left - 1, flash: restroke(rand) }); continue; }
    jumps.push({ ch: channel([0, 0], [1, 0], rand, 1, 0.2, 5, 0), to: TIPS[(i + Math.floor(rand() * 5)) % TIPS.length],
      lift: (rand() < 0.5 ? -1 : 1) * (GROUND_BOW[0] + (GROUND_BOW[1] - GROUND_BOW[0]) * rand()),
      leave: (2 * rand() - 1) * GROUND_LEAVE, left: Math.min(2, life(rand)) - 1, flash: 1,
      seed: Math.floor(rand() * 2 ** 31) });
  }
  return { charge, grown, born, pulse, flare: hit ? 1 : 0, level, shell, jumps, turn: (b?.turn ?? 0) + BALL_TURN, frame,
    top, most, fade };
}

/** A ball's radius in hand sizes, ``grown`` (0-1) and swollen by ``pulse``. */
function sized(grown: number, pulse: number): number {
  return (BALL_R[0] + (BALL_R[1] - BALL_R[0]) * grown * grown * (3 - 2 * grown)) * (1 + pulse);
}

/** The ball's radius in px for a hand ``scale`` px across. */
export function ballRadius(b: Ball, scale: number): number {
  return scale * sized(b.grown, b.pulse);
}

/** Where the ball hovers for a palm at ``palm``: BALL_LIFT up the picture off it (``px``: a pixel at 1080 lines). */
export function ballAt(palm: P2, px: number): P2 {
  return [palm[0], palm[1] - BALL_LIFT * px];
}

/** The body's colour from its middle out, as gradient stops [radii, r, g, b, alpha]: the shot's ball as shown
 * (cuesync.lightning, the median of each ring at full charge: test_lightning.py's BALL_PROFILE, which
 * lightning.test.ts reads to hold these to), white-hot in its middle, cyan round that and a saturated blue to its
 * limb, the palm it lights showing through it there; dense in its middle, light over the palm towards its edge, and
 * gone a little past its radius with no outline. */
export const BODY_STOPS: [number, number, number, number, number][] = [
  [0, 250, 252, 255, 1], [0.15, 231, 241, 255, 1], [0.3, 166, 206, 255, 1], [0.45, 148, 199, 254, 1],
  [0.6, 166, 203, 254, 0.97], [0.75, 192, 210, 252, 0.88], [0.9, 234, 232, 253, 0.65], [1, 236, 234, 252, 0.35],
  [1.15, 236, 234, 250, 0],
];

const plasmaCanvas = new WeakMap<object, Ctx2D>();
let plasmaField: Float32Array | null = null;
const FIELD_W = 128, FIELD_H = 64; // the ridged noise over longitude and latitude

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
  for (let i = 0; i < field.length; i++) field[i] = 2 * (1 - Math.min(1, Math.abs(field[i]) * 3)) ** 2 - 0.6;
  return (plasmaField = field);
}

/** PLASMA_LAYERS' turns at ``seconds``: each one's share, axis (unit) and the cosine and sine it is turned by. */
export function plasmaTurns(seconds: number): { share: number; k: P3; c: number; sn: number }[] {
  return PLASMA_LAYERS.map(([rate, share, ax, ay, az]) => {
    const l = Math.hypot(ax, ay, az), a = 2 * Math.PI * PLASMA[1] * rate * seconds;
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
  return Math.max(-1, Math.min(1, v / 2)) * Math.min(1, (1.02 - Math.sqrt(r2)) / 0.12);
}

/** The ball's plasma, PLASMA_PX across, at ``seconds`` (plasmaAt): filaments a pale blue over its body and the rest a
 * deeper blue, ``amount`` strong; null where there is no canvas to make it on. */
function plasmaLayer(ctx: CanvasRenderingContext2D, seconds: number, amount: number): Ctx2D | null {
  const main = ctx.canvas as HTMLCanvasElement | undefined;
  if (!main?.width || typeof ctx.drawImage !== "function") return null;
  let t = plasmaCanvas.get(main);
  if (!t) {
    const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(PLASMA_PX, PLASMA_PX)
      : Object.assign(document.createElement("canvas"), { width: PLASMA_PX, height: PLASMA_PX });
    const made = c.getContext("2d") as Ctx2D | null;
    if (!made) return null;
    plasmaCanvas.set(main, (t = made));
  }
  const n = PLASMA_PX, img = t.createImageData(n, n), turns = plasmaTurns(seconds);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const s = plasmaAt(((i + 0.5) / n) * 2 - 1, ((j + 0.5) / n) * 2 - 1, turns) * amount;
    const o = 4 * (j * n + i);
    if (s > 0) img.data.set([140, 195, 255, Math.round(255 * Math.min(1, 0.8 * s))], o); // (a filament: brighter)
    else if (s < 0) img.data.set([30, 80, 210, Math.round(255 * Math.min(1, -0.7 * s))], o); // (between: deeper)
  }
  t.putImageData(img, 0, 0);
  return t;
}

/** Draws the ball over a palm at ``palm`` (hovering off it: ballAt) for a hand ``scale`` px across: its body (once
 * formed, BODY_STOPS) with its plasma turning through it on the sphere (plasmaLayer, where there is a canvas to
 * make it on), the faint blue haze of its light round it (HALO), its shell's arcs white over it (brightest along its outline, those behind
 * it and inside it dimmer) and its white-hot middle. ``px``: a pixel at 1080 lines. */
export function drawBall(ctx: CanvasRenderingContext2D, b: Ball, palm: P2, scale: number, px: number) {
  const c = ballAt(palm, px);
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
    const plasma = plasmaLayer(ctx, b.frame / 60, PLASMA[0] * body * 1.4); // (a 60 Hz screen)
    if (plasma) ctx.drawImage(plasma.canvas, c[0] - R, c[1] - R, 2 * R, 2 * R); // (its plasma, on the sphere)
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
  const reach = 1 + FIL_REACH * (1 - b.most) ** 2; // (a young ball's shell crackles past where its skin will be; a
  // dying one's goes in with it)
  const cs = Math.cos(b.turn), sn = Math.sin(b.turn), tilt = 0.35, ct = Math.cos(tilt), st = Math.sin(tilt);
  const lines: [Channel, number][] = [];
  const width = Math.max(0.7, widthK(scale, px) * px * (0.5 + 0.4 * b.level));
  for (const s of b.shell) {
    if (s.flash <= 0.02) continue;
    const seen3 = s.pts.map(([x, y, z]): P3 => { // (turned about its upright, then tipped towards the camera)
      const x1 = x * cs + z * sn, z1 = -x * sn + z * cs;
      return [x1, y * ct - z1 * st, y * st + z1 * ct];
    });
    const front = seen3.filter(([, , z]) => z >= -0.1).length / seen3.length;
    const limb = seen3.reduce((sum, p) => sum + 1 - Math.abs(p[2]) / (Math.hypot(...p) || 1), 0) / seen3.length;
    const gain = 0.7 * s.flash * (0.4 + 0.6 * b.level) * s.inten * (SHELL_FACE + (1 - SHELL_FACE) * limb) *
      (SHELL_BACK + (1 - SHELL_BACK) * front) * b.fade;
    lines.push([{ pts: seen3.map(([x, y]): P2 => [c[0] + R * reach * x, c[1] + R * reach * y]), width, inten: 1,
      forks: [], at: 0 }, gain]);
  }
  drawChannels(ctx, lines, px, 0.4, 0.15 * R, false, BALL_LOOK);
  ctx.restore();
  drawSpark(ctx, c, 0.16 * R + 3 * px, (body ? 0.9 : 0.5) * (1 + b.flare) * (0.5 + 0.5 * b.level)); // (its body is
  // white-hot round it already: wider, it was a white star the arcs ran into like rays)
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

/** An arc's points from the ball (middle ``c``, radius ``R``) to a fingertip ``tip``: ``ch`` (drawn from (0, 0) to
 * (1, 0)) carried from its skin, ``leave`` radians round from the way to the tip, onto a bow ``lift`` px at 1080 off
 * its chord (to its left as it runs), coming down onto the finger past its tip, LAND_IN inside it; or null where it
 * cannot (no edge of the hand within LAND_NEAR of the tip the way it comes in, past GROUND_FREE of the ball more than
 * GROUND_ON of it on the hand, or back over the ball). ``hand``: whether a point is on the hand, or null (no hand: it
 * lands on the tip). */
export function jumpPath(ch: Channel, c: P2, R: number, tip: P2, leave: number, lift: number, px: number,
  hand: ((p: P2) => boolean) | null): P2[] | null {
  const phi = Math.atan2(tip[1] - c[1], tip[0] - c[0]) + leave;
  const a: P2 = [c[0] + R * Math.cos(phi), c[1] + R * Math.sin(phi)];
  const span = Math.hypot(tip[0] - a[0], tip[1] - a[1]);
  if (span < 1e-6) return null;
  const side: P2 = [((tip[1] - a[1]) / span) * lift * px, (-(tip[0] - a[0]) / span) * lift * px];
  let land = tip;
  if (hand) {
    const ctl: P2 = [(a[0] + tip[0]) / 2 + 2 * side[0], (a[1] + tip[1]) / 2 + 2 * side[1]];
    const cl = Math.hypot(tip[0] - ctl[0], tip[1] - ctl[1]) || 1, come: P2 = [(tip[0] - ctl[0]) / cl, (tip[1] - ctl[1]) / cl];
    let edge = -1;
    for (let e = 0; e < (LAND_NEAR + LAND_IN) * px; e += px) {
      if (!hand([tip[0] - come[0] * e, tip[1] - come[1] * e])) { edge = e; break; }
    }
    if (edge < 0) return null;
    const back = Math.max(0, edge - LAND_IN * px);
    land = [tip[0] - come[0] * back, tip[1] - come[1] * back];
  }
  const pts = bowed(ch, a, land, side, 1).pts;
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = cum[cum.length - 1];
  let on = 0, free = 0;
  for (let i = 1; i < pts.length; i++) {
    const r = Math.hypot(pts[i][0] - c[0], pts[i][1] - c[1]);
    if (cum[i] > 0.15 * total && r < 0.95 * R) return null; // (back over the ball)
    if (!hand || r <= GROUND_FREE * R || cum[i] >= total - (LAND_IN + 3) * px) continue;
    const l = cum[i] - cum[i - 1];
    free += l;
    if (hand(pts[i])) on += l;
  }
  return on > GROUND_ON * Math.max(free, 1e-6) ? null : pts;
}

/** The ball's light round it (BALL_LIGHT), screened: a cool white on the hand (its 21 landmarks ``lm``, px, or null)
 * and a blue-white on the room, from its middle as from a ball of its radius, R^2 / (d^2 + h^2) at most 1, gone by
 * three radii, little of it on what its dense middle hides, as strong as its body is dense. And, with the hand, the
 * arcs it throws to the fingertips (GROUND), bowing through the air off the hand and coming down onto each past its
 * tip (jumpPath; one that lies on the hand looks for another way, GROUND_TRIES times, or is not drawn), white-hot in a
 * cyan-blue glow, with a white-hot point in a blue glow where each lands. ``palm``: the palm it hovers over (ballAt);
 * ``px``: a pixel at 1080 lines. */
export function drawBallLight(ctx: CanvasRenderingContext2D, b: Ball, palm: P2, scale: number, lm: P2[] | null,
  w: number, h: number, px = h / 1080) {
  const c = ballAt(palm, px);
  const R = ballRadius(b, scale);
  const body = b.born ? dense(b.level, b.charge) : 0;
  if (body > 0) {
    const [onHandA, onRoomA, hover, behind, reach, shade] = BALL_LIGHT;
    const lit = (0.5 + 0.5 * b.level) * body;
    const falloff = (fill: CanvasGradient, rgb: string, alpha: number, hh: number) => {
      for (let i = 0; i <= 12; i++) { // (from its middle out to its reach)
        const d = (reach * i) / 12, gone = Math.max(0, 1 - (d / reach) ** 2) ** 2;
        const hide = 1 - Math.exp(-((d / shade) ** 8));
        fill.addColorStop(i / 12, `rgba(${rgb}, ${alpha * lit * Math.min(1, 1 / (d * d + hh * hh)) * gone * hide})`);
      }
    };
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    const room = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach * R);
    falloff(room, BALL_LIGHT_RGB[1], onRoomA, behind);
    ctx.fillStyle = room;
    ctx.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
    const mask = lm ? handLayer(ctx, "ball-hand", lm, scale) : null;
    if (mask) {
      const hand = mask.createRadialGradient(c[0], c[1], 0, c[0], c[1], reach * R);
      falloff(hand, BALL_LIGHT_RGB[0], onHandA, (hover * px) / Math.max(R, 1e-6));
      mask.fillStyle = hand;
      mask.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
      ctx.drawImage(mask.canvas, 0, 0, w, h);
    }
    ctx.restore();
  }
  if (!lm || !b.born) return;
  const hand = (p: P2) => onHand(p, lm, scale);
  const lines: [Channel, number][] = [];
  const lands: [P2, number][] = [];
  const used = new Map<number, number>();
  for (const j of b.jumps) {
    const free = TIPS.filter((t) => Math.hypot(lm[t][0] - c[0], lm[t][1] - c[1]) >= 1.15 * R);
    if (!free.length) break;
    let pts = free.includes(j.to) ? jumpPath(j.ch, c, R, lm[j.to], j.leave, j.lift, px, hand) : null;
    let to = j.to;
    if (!pts) { // (another way: the fingertips with the fewest arcs first, a short bow and then a long one each)
      const rand = rng(j.seed + b.frame);
      const order = [...free].sort((p, q) => (used.get(p) ?? 0) - (used.get(q) ?? 0));
      for (let k = 0; k < GROUND_TRIES && !pts; k++) {
        to = order[Math.floor(k / 4) % order.length];
        const lift = (rand() < 0.5 ? -1 : 1) * (k % 4 < 2 ? GROUND_BOW[0] + (GROUND_BOW[1] - GROUND_BOW[0]) * rand()
          : GROUND_BOW[1] + (GROUND_BOW[2] - GROUND_BOW[1]) * rand());
        pts = jumpPath(j.ch, c, R, lm[to], (2 * rand() - 1) * GROUND_LEAVE, lift, px, hand);
      }
    }
    if (!pts) continue;
    used.set(to, (used.get(to) ?? 0) + 1);
    if (j.flash <= 0.05) continue;
    lines.push([{ pts, width: Math.max(0.9, widthK(scale, px) * px * 1.4), inten: 1, forks: [], at: 0 }, 0.85 * j.flash]);
    lands.push([pts[pts.length - 1], j.flash]);
  }
  drawChannels(ctx, lines, px, 0.5, 0.1 * scale, false, BALL_LOOK);
  for (const [at, amp] of lands) { // (where each lands: a white-hot point in a blue glow)
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(at[0], at[1], 0, at[0], at[1], LAND[1] * px);
    g.addColorStop(0, `rgba(240, 248, 255, ${Math.min(1, amp)})`);
    g.addColorStop(LAND[0] / LAND[1] / 2, `rgba(200, 230, 255, ${0.8 * Math.min(1, amp)})`);
    g.addColorStop(0.45, `rgba(60, 140, 255, ${0.35 * Math.min(1, amp)})`);
    g.addColorStop(1, "rgba(40, 110, 255, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(at[0] - LAND[1] * px, at[1] - LAND[1] * px, 2 * LAND[1] * px, 2 * LAND[1] * px);
    ctx.restore();
  }
}

/** ``c`` (drawn from (0, 0) to (1, 0)) carried onto a bow from ``a`` to ``b`` whose middle is ``side`` (px) off the
 * straight line: along a quadratic curve, each point's second coordinate across it, as long as the chord; ``width``
 * px wide. */
export function bowed(c: Channel, a: P2, b: P2, side: P2, width: number): Channel {
  const chord = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
  const ctl: P2 = [(a[0] + b[0]) / 2 + 2 * side[0], (a[1] + b[1]) / 2 + 2 * side[1]];
  const pts = c.pts.map(([u, v]): P2 => {
    const x = (1 - u) ** 2 * a[0] + 2 * u * (1 - u) * ctl[0] + u * u * b[0];
    const y = (1 - u) ** 2 * a[1] + 2 * u * (1 - u) * ctl[1] + u * u * b[1];
    const tx = 2 * (1 - u) * (ctl[0] - a[0]) + 2 * u * (b[0] - ctl[0]);
    const ty = 2 * (1 - u) * (ctl[1] - a[1]) + 2 * u * (b[1] - ctl[1]);
    const tl = Math.hypot(tx, ty) || 1;
    return [x - (ty / tl) * v * chord, y + (tx / tl) * v * chord];
  });
  return { pts, width, inten: 1, forks: [], at: 0 };
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
