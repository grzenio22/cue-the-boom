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
const ARC_REACH = [1.4, 5.0]; // hand sizes: a main channel's reach, quietest voice to loudest
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
export const TARGET_CONE = [25, 40]; // degrees either side of the finger's aim: an edge within the first, else in
// the second
const TARGET_BAND = [0.4, 1.1]; // shares of its reach: how far out what it strikes may be
export const TARGET_BORDER = 0.12; // share of the picture's short side: never this close to its edge
export const TARGET_TOP = 3, TARGET_APART = 40; // one of the best this many, px at 1080 apart and from the last
const EDGE_Q = 8; // the picture is read at 1 / EDGE_Q of the canvas
const EDGE_MIN = 40; // the gradient (Sobel, code values, at that size, softened) a line must reach to be struck: a step
// of about 15 levels (a blank wall's grain and the camera's noise stay under it)
const EDGE_NEAR = 3; // ...and at least this many more edge px round it (in its 5 x 5): a line, not a speck
export const IMPACT_MIN = 0.6; // a stroke at least this bright strikes with a contact flash (a weak one with none)
const IMPACT_PX = 2.5; // px at 1080: the white-hot point where it touches
const POOL = [110, 0.3, 0.45]; // (px at 1080 along the surface, its width across against that, alpha): its light
// pooled along the surface it strikes
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

/** Where a new main channel from ``root`` aimed ``aim`` (radians) ``reach`` px out strikes, and the way the surface
 * runs there, or null: of the ``edges`` (surfaces) within TARGET_CONE of the aim (its first, else its second),
 * TARGET_BAND of the reach out and TARGET_BORDER inside the ``size`` (w, h) frame, the strongest, nearest its
 * ``heading`` and furthest out, with a little chance; then one of the best TARGET_TOP, TARGET_APART from each other
 * and from where it struck ``last``. ``px``: a pixel at 1080 lines. */
export function pickTarget(edges: number[][], root: P2, aim: number, heading: number, reach: number, size: P2,
  rand: () => number, px = 1, last: P2 | null = null): { at: P2; along: P2 } | null {
  const [w, h] = size;
  const border = TARGET_BORDER * Math.min(w, h), apart = TARGET_APART * px;
  const offOf = (d: number) => ((d + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
  const found: [number, number[], number][] = []; // (cone, row, off)
  for (const row of edges) {
    const [x, y] = row;
    if (x < border || x > w - 1 - border || y < border || y > h - 1 - border) continue;
    const dist = Math.hypot(x - root[0], y - root[1]);
    if (dist < TARGET_BAND[0] * reach || dist > TARGET_BAND[1] * reach) continue;
    const off = Math.abs(offOf(Math.atan2(y - root[1], x - root[0]) - aim)) * 180 / Math.PI;
    const cone = TARGET_CONE.findIndex((c) => off <= c);
    if (cone >= 0) found.push([cone, row, offOf(Math.atan2(y - root[1], x - root[0]) - heading)]);
  }
  const first = Math.min(...found.map(([c]) => c));
  const scored = found.filter(([c]) => c === first).map(([c, row, off]): [number, number[]] => [
    Math.min(1, row[2] / (2 * EDGE_MIN)) + 0.5 * Math.hypot(row[0] - root[0], row[1] - root[1]) / reach -
    0.5 * Math.abs(off) / (TARGET_CONE[c] * Math.PI / 180) + 0.3 * rand(), row]);
  scored.sort((p, q) => q[0] - p[0]);
  const best: number[][] = [];
  for (const [, row] of scored) {
    const far = (p: P2 | number[]) => Math.hypot(row[0] - p[0], row[1] - p[1]) >= apart;
    if ((last && !far(last)) || !best.every(far)) continue;
    best.push(row);
    if (best.length === TARGET_TOP) break;
  }
  if (!best.length) return null;
  const pick = best[Math.min(best.length - 1, Math.floor(rand() * best.length))];
  return { at: [pick[0], pick[1]], along: [pick[3], pick[4]] };
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
 * strikes a surface (``splash``, shown SPLASH's frames) and how many frames since it was struck (``age``). */
export interface Stream { ch: Channel | null; left: number; aim: number; seed: number; flash?: number;
  splash?: Channel[]; age?: number }

function life(rand: () => number): number {
  const r = rand();
  return r < LIFE_P[0] ? LIFE[0] : r < LIFE_P[0] + LIFE_P[1] ? LIFE[1] : LIFE[2];
}

/** One frame of a main channel from ``root`` (welded to it): ridden to the hand and re-struck, or struck anew towards
 * ``aim`` (radians) ``reach`` px long when its time is up (or on ``strike``: a hit strikes every stream at once), its
 * first ARC_LEAD along the aim and its heading never more than ARC_OFF off it, at an edge of the ``plate`` in its way
 * (pickTarget: a contact flash and a splash there) or, with none, ending in the air in a brush. ``level``: the voice,
 * which sets how often a re-stroke flashes; ``px``: a pixel at 1080 lines. */
export function streamStep(s: Stream, root: P2, aim: number, reach: number, width: number, strike: boolean,
  frame: number, level = 1, plate: Plate | null = null, px = 1): Stream {
  const rand = rng(s.seed * 7919 + frame);
  if (s.ch && s.left > 0 && !strike) {
    const on = rand() < RESTROKE[0] + RESTROKE[1] * level;
    return { ...s, ch: ride(s.ch, root), left: s.left - 1, flash: on ? 0.55 + 0.35 * rand() : 0.1 * rand(),
      age: (s.age ?? 0) + 1 };
  }
  const off = s.ch ? s.aim - aim + (rand() - 0.5) * 2 * ARC_JITTER : (rand() - 0.5) * 2 * ARC_JITTER; // (near its last)
  const a = aim + Math.max(-ARC_OFF, Math.min(ARC_OFF, off * 0.6));
  const l = reach * (0.8 + 0.25 * rand());
  const last = s.ch?.strikes ? s.ch.pts[s.ch.pts.length - 1] : null;
  const hit = plate ? pickTarget(surfaces(plate), root, aim, a, l, [plate.w * plate.k, plate.h * plate.k], rand, px,
    last) : null;
  const far: P2 = hit ? hit.at : [root[0] + Math.cos(a) * l, root[1] + Math.sin(a) * l];
  const span = Math.hypot(far[0] - root[0], far[1] - root[1]);
  const lead: P2 = [root[0] + Math.cos(aim) * ARC_LEAD * span, root[1] + Math.sin(aim) * ARC_LEAD * span];
  const ch = channel(root, far, rand, width, 0.26, 6, 0.1 + 0.1 * level, { lead });
  if (hit) ch.strikes = hit.along;
  else ch.forks.push(...brush(ch, rand));
  return { ch, left: life(rand) - 1, aim: a, seed: s.seed, flash: 1, splash: hit ? splash(ch, hit.along, rand, px) : [],
    age: 0 };
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
  const reach = scale * (ARC_REACH[0] + (ARC_REACH[1] - ARC_REACH[0]) * level);
  const main = streamStep(arcs.main, tip, aim, reach, width, strike, frame, level, plate, px);
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
 * falloff in Canvas 2D at a sixteenth of the pixels; without a canvas for it, wide faint strokes), a violet-blue rim
 * and the core, white for a main channel (far over white: a phone clips it white with a blue rim) and bluer the
 * dimmer a branch, each channel in its stretches (thinning, a branch fading along it), and a contact flash where a
 * bright one strikes a surface (IMPACT_MIN). First (``read``) it reads the picture it is drawn over, for the next
 * strike to look for what to hit in. ``px``: a pixel at 1080 lines; ``most``: the widest glow, in px (a small ball's
 * arcs glow no wider than it is). */
export function drawChannels(ctx: CanvasRenderingContext2D, channels: Channel[] | [Channel, number][], px: number,
  glow = 1, most = Infinity, read = true) {
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
  for (const [c, g] of list) {
    if (c.strikes && c.inten * g >= IMPACT_MIN) drawImpact(ctx, c.pts[c.pts.length - 1], c.strikes, c.inten * g, px);
  }
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

export const DIP = [0.35, 4, 24]; // stops, frames rising, frames falling: the camera exposes down for the ball (as a
// phone's does for a bright light in the picture), so its body is the brightest thing over the palm and skin at the
// top of red comes off it to take its blue, settling for a phrase rather than breathing with the voice

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
export const BALL_ROOM_RGB = "70, 105, 160"; // the ball's on the room, a blue-white, and on the hand a deep blue at
export const BALL_HAND_RGB = "20, 45, 130"; // a share of it, next to no red or green in it, so skin already at the top
// of red takes it as blue and keeps its colour (cuesync.lightning's LIGHT_RGB; in its near-white the palm round it went
// chalky)

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
export const GROUND = [6, 10]; // arcs jumping from its skin to the fingers, quiet to loud, struck every frame or two
const TARGETS = [4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 6, 10, 14, 18]; // the landmarks they go to: tips, bases, joints
export const GROUND_BOW = [30, 60]; // px at 1080: each bows through the air, its middle this far off its straight line,
// up the picture (out from the palm's middle where it runs up or down it), coming down onto the finger
const NEAR_LIGHT = [0.6, 0.73, 3.5]; // its light on what is close round it: this strong at its edge, falling off
// with distance d (radii) as (1 + h^2) / (d^2 + h^2), h how high it hovers over the palm (radii), the softened inverse
// square of cuesync.lightning's, gone by the last: the skin next to it lit, the fingertips far from it barely
const HALO: [number, number, string] = [0.22, 0.8, "20, 90, 255"]; // the faint blue haze of its light round it, added
// over skin and wall alike: at most this, falling off as a Gaussian of this sigma (radii; cuesync.lightning's HALO)
export const PLASMA = [0.45, 1.5]; // (contrast, turns a second): its body is moving plasma, filaments of brighter and
// darker light turning through it (cuesync.lightning's PLASMA), not an airbrushed marble
const PLASMA_LAYERS: [number, number, string][] = [[1, 1, "multiply"], [-0.75, 0.5, "lighter"]]; // (rate, share, how):
// its two turning layers, one deepening its blue along its filaments, the other brightening them (added alone, a pale
// cyan, they washed it white)

/** An arc of the ball's shell: its points on the unit sphere (3D, z towards the camera), turned with the ball, and
 * how bright (an inner one dimmer); struck and re-struck as a channel is. */
export interface Shell { pts: P3[]; inten: number; left: number; flash: number }

/** An arc jumping from the ball to a finger: its shape from (0, 0) to (1, 0), the landmark it goes to, how far it bows
 * (px at 1080), struck and re-struck as a channel is. */
export interface Jump { ch: Channel; to: number; lift: number; left: number; flash: number }

/** The ball's state between frames. */
export interface Ball { charge: number; grown: number; born: boolean; pulse: number; flare: number; level: number;
  shell: Shell[]; jumps: Jump[]; turn: number; frame: number }

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
 * and flares its middle; its shell (SHELL arcs) and its arcs to the fingers (GROUND) are struck and re-struck as
 * channels are, every frame or two, turning slowly with it. */
export function ballStep(b: Ball | null, level: number, hit: boolean, frame: number): Ball | null {
  const prev = b?.charge ?? 0;
  const charge = prev + (level - prev) * (1 - Math.exp(-1 / (level > prev ? BALL_GROW : BALL_SHRINK)));
  if (Math.max(level, charge) < BALL_ON[b ? 1 : 0]) return null;
  const born = (b?.born ?? false) || dense(level, charge) > 0;
  const grown = b?.born ? Math.min(Math.sqrt(charge), b.grown + BALL_RISE) : 0;
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
  for (let i = 0; i < Math.round(GROUND[0] + (GROUND[1] - GROUND[0]) * level); i++) {
    const old = b?.jumps[i], rand = rng((i + 1) * 15485863 + frame);
    if (old && old.left > 0 && !hit) { jumps.push({ ...old, left: old.left - 1, flash: restroke(rand) }); continue; }
    jumps.push({ ch: channel([0, 0], [1, 0], rand, 1, 0.2, 5, 0),
      to: TARGETS[(i + Math.floor(rand() * 4)) % TARGETS.length],
      lift: GROUND_BOW[0] + (GROUND_BOW[1] - GROUND_BOW[0]) * rand(), left: Math.min(2, life(rand)) - 1, flash: 1 });
  }
  return { charge, grown, born, pulse: hit ? BALL_PULSE : (b?.pulse ?? 0) * 0.6, flare: hit ? 1 : 0, level, shell,
    jumps, turn: (b?.turn ?? 0) + BALL_TURN, frame };
}

/** The ball's radius in px for a hand ``scale`` px across. */
export function ballRadius(b: Ball, scale: number): number {
  const g = b.grown;
  return scale * (BALL_R[0] + (BALL_R[1] - BALL_R[0]) * g * g * (3 - 2 * g)) * (1 + b.pulse);
}

/** Where the ball hovers for a palm at ``palm``: BALL_LIFT up the picture off it (``px``: a pixel at 1080 lines). */
export function ballAt(palm: P2, px: number): P2 {
  return [palm[0], palm[1] - BALL_LIFT * px];
}

/** The body's colour from its middle out, as gradient stops [radii, r, g, b, alpha]: the shot's ball as shown
 * (cuesync.lightning, the median of each ring at full charge: test_lightning.py's BALL_PROFILE, which
 * lightning.test.ts reads to hold these to), white-hot in its middle, cyan round that and a saturated blue to its
 * limb, the palm lit blue showing through it there; dense in its middle, light over the palm towards its edge, and
 * gone a little past its radius with no outline. */
export const BODY_STOPS: [number, number, number, number, number][] = [
  [0, 250, 252, 255, 1], [0.15, 230, 241, 255, 1], [0.3, 161, 204, 255, 1], [0.45, 140, 195, 254, 1],
  [0.6, 152, 197, 254, 0.97], [0.75, 172, 202, 253, 0.88], [0.9, 196, 212, 254, 0.65], [1, 200, 214, 252, 0.35],
  [1.15, 205, 216, 250, 0],
];

const plasmaMaps = new WeakMap<object, Ctx2D>();

/** A tile of ridged noise (a blue, its alpha the noise: filaments), made once per canvas; null where there is none
 * to make it on. */
function plasmaTile(ctx: CanvasRenderingContext2D): Ctx2D | null {
  const main = ctx.canvas as HTMLCanvasElement | undefined;
  if (!main?.width) return null;
  const got = plasmaMaps.get(main);
  if (got) return got;
  const n = 128;
  const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(n, n)
    : Object.assign(document.createElement("canvas"), { width: n, height: n });
  const t = c.getContext("2d") as Ctx2D | null;
  if (!t) return null;
  const rand = rng(17);
  const img = t.createImageData(n, n);
  const field = new Float32Array(n * n);
  for (const [cells, weight] of [[4, 0.6], [9, 0.4]]) { // (two octaves of smooth noise, wrapping round the tile)
    const g = Array.from({ length: cells * cells }, () => rand() * 2 - 1);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const fx = (x / n) * cells, fy = (y / n) * cells, x0 = Math.floor(fx), y0 = Math.floor(fy);
      const u = fx - x0, v = fy - y0, s = (q: number) => q * q * (3 - 2 * q);
      const at = (i: number, j: number) => g[((j % cells) * cells) + (i % cells)];
      const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * s(u);
      const bot = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * s(u);
      field[y * n + x] += weight * (top + (bot - top) * s(v));
    }
  }
  for (let i = 0; i < n * n; i++) { // (ridged: its zero crossings bright, filaments)
    const ridge = (1 - Math.min(1, Math.abs(field[i]) * 3)) ** 2;
    img.data.set([90, 160, 255, Math.round(255 * ridge)], 4 * i);
  }
  t.putImageData(img, 0, 0);
  plasmaMaps.set(main, t);
  return t;
}

/** Draws the ball over a palm at ``palm`` (hovering off it: ballAt) for a hand ``scale`` px across: its body (once
 * formed, BODY_STOPS) with its plasma turning through it (PLASMA, where there is a canvas to make it on), the faint
 * blue haze of its light round it (HALO), its shell's arcs white over it (brightest along its outline, those behind
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
    const tile = plasmaTile(ctx);
    if (tile) { // (its plasma: the tile turned two ways at once inside its body, brighter filaments, added)
      ctx.save();
      ctx.beginPath();
      ctx.arc(c[0], c[1], 0.95 * R, 0, 2 * Math.PI);
      ctx.clip();
      for (const [rate, share, how] of PLASMA_LAYERS) {
        ctx.save();
        ctx.globalCompositeOperation = how as GlobalCompositeOperation;
        ctx.translate(c[0], c[1]);
        ctx.rotate((2 * Math.PI * PLASMA[1] * rate * b.frame) / 60); // (a 60 Hz screen)
        ctx.globalAlpha = PLASMA[0] * share * body;
        ctx.drawImage(tile.canvas, -1.2 * R, -1.2 * R, 2.4 * R, 2.4 * R);
        ctx.restore();
      }
      ctx.restore();
    }
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
  const reach = 1 + FIL_REACH * (1 - b.grown) ** 2; // (a young ball's shell crackles past where its skin will be)
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
      (SHELL_BACK + (1 - SHELL_BACK) * front);
    lines.push([{ pts: seen3.map(([x, y]): P2 => [c[0] + R * reach * x, c[1] + R * reach * y]), width, inten: 1,
      forks: [], at: 0 }, gain]);
  }
  drawChannels(ctx, lines, px, 0.4, 0.15 * R, false);
  ctx.restore();
  drawSpark(ctx, c, 0.16 * R + 3 * px, (body ? 0.9 : 0.5) * (1 + b.flare) * (0.5 + 0.5 * b.level)); // (its body is
  // white-hot round it already: wider, it was a white star the arcs ran into like rays)
}

/** The ball's light on what is close round it (NEAR_LIGHT), screened under it: a blue-white on the room and a deep
 * blue on the hand (its 21 landmarks ``lm``, px, or null), which skin takes as blue keeping its colour; strongest at
 * its edge and falling off as the softened inverse square of the distance, so the fingertips far from it are barely
 * lit; as strong as its body is dense. And, with the hand, the arcs it throws to the fingers (GROUND), bowing up
 * through the air from its skin and coming down onto each finger, white-hot with a spark where they land. ``palm``:
 * the palm it hovers over (ballAt); ``px``: a pixel at 1080 lines. */
export function drawBallLight(ctx: CanvasRenderingContext2D, b: Ball, palm: P2, scale: number, lm: P2[] | null,
  w: number, h: number, px = h / 1080) {
  const c = ballAt(palm, px);
  const R = ballRadius(b, scale);
  const body = b.born ? dense(b.level, b.charge) : 0;
  if (body > 0) {
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
    falloff(room, BALL_ROOM_RGB, 0.5);
    ctx.fillStyle = room;
    ctx.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
    const mask = lm ? handLayer(ctx, "ball-hand", lm, scale) : null;
    if (mask) {
      const hand = mask.createRadialGradient(c[0], c[1], R, c[0], c[1], reach * R);
      falloff(hand, BALL_HAND_RGB, 0.6);
      mask.fillStyle = hand;
      mask.fillRect(c[0] - reach * R, c[1] - reach * R, 2 * reach * R, 2 * reach * R);
      ctx.drawImage(mask.canvas, 0, 0, w, h);
    }
    ctx.restore();
  }
  if (!lm || !b.born) return;
  const middle: P2 = [[0, 5, 9, 13, 17].reduce((s, i) => s + lm[i][0], 0) / 5,
    [0, 5, 9, 13, 17].reduce((s, i) => s + lm[i][1], 0) / 5];
  const lines: [Channel, number][] = [];
  const sparks: [P2, number][] = [];
  for (const j of b.jumps) {
    const to = lm[j.to];
    if (j.flash <= 0.05 || Math.hypot(to[0] - c[0], to[1] - c[1]) < 1.15 * R) continue; // (not to a knuckle under it)
    const d = Math.hypot(to[0] - c[0], to[1] - c[1]);
    const a: P2 = [c[0] + ((to[0] - c[0]) / d) * R, c[1] + ((to[1] - c[1]) / d) * R];
    lines.push([bowed(j.ch, a, to, j.lift * px, middle, Math.max(0.9, widthK(scale, px) * px * 1.4)), 0.85 * j.flash]);
    sparks.push([to, 0.4 * j.flash]);
  }
  drawChannels(ctx, lines, px, 0.5, 0.1 * scale, false);
  for (const [at, amp] of sparks) drawSpark(ctx, at, 0.04 * scale, amp);
}

/** ``c`` (drawn from (0, 0) to (1, 0)) carried onto a bow from ``a`` to ``b``, its middle ``lift`` px off the straight
 * line across it, up the picture, or out from ``middle`` (the palm's) where it runs up or down the picture; ``width``
 * px wide. */
export function bowed(c: Channel, a: P2, b: P2, lift: number, middle: P2, width: number): Channel {
  const dx = b[0] - a[0], dy = b[1] - a[1], chord = Math.hypot(dx, dy) || 1;
  let n: P2 = [-dy / chord, dx / chord];
  const mid: P2 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  if (Math.abs(n[1]) < 0.35) {
    if (n[0] * (mid[0] - middle[0]) + n[1] * (mid[1] - middle[1]) < 0) n = [-n[0], -n[1]];
  } else if (n[1] > 0) n = [-n[0], -n[1]];
  const ctl: P2 = [mid[0] + 2 * lift * n[0], mid[1] + 2 * lift * n[1]];
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
