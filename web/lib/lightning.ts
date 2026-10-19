/** Live lightning for Try it live: jagged channels struck from a fingertip (arc) or crackling over a palm (ball),
 * drawn on the camera's canvas. As the shot's own (cuesync.lightning): a channel is a midpoint-displaced line with
 * forks that leave it more towards its far end; it is struck, held for a couple of frames with its root riding the
 * hand and its far end where it struck, then struck anew near its last aim; the channel is far over white with a
 * blue, then violet, glow round it. */

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

export interface Channel { pts: P2[]; width: number; forks: Channel[]; at: number } // at: the parent's point it forks from

/** A channel from ``a`` to ``b``: each midpoint pushed across its segment by up to ``rough`` of its length (gently at
 * the coarse levels so it runs on, sharply at the fine ones so it kinks), forks off it now and then, more towards
 * its far end, each thinner and shorter. */
export function channel(a: P2, b: P2, rand: () => number, width = 2, rough = 0.28, depth = 6, forkRate = 0.12, at = 0): Channel {
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
  const forks: Channel[] = [];
  if (width > 0.8) {
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let i = 2; i < pts.length - 2; i++) {
      const along = i / pts.length;
      if (rand() < forkRate * along * 0.25) {
        const turn = (rand() - 0.5) * 1.6;
        const [px, py] = pts[i], [qx, qy] = pts[i + 1];
        const ang = Math.atan2(qy - py, qx - px) + turn;
        const l = len * (0.15 + 0.3 * rand()) * (1 - along * 0.5);
        forks.push(channel(pts[i], [px + Math.cos(ang) * l, py + Math.sin(ang) * l], rand, width * 0.5, rough, depth - 2,
          forkRate * 0.5, i));
      }
    }
  }
  return { pts, width, forks, at };
}

/** ``c`` moved so its root is at ``root`` and its far end stays: each point by the root's move times how near the
 * root it is (the channel rides the hand at its root, holds where it struck at its end). */
export function ride(c: Channel, root: P2): Channel {
  const [dx, dy] = [root[0] - c.pts[0][0], root[1] - c.pts[0][1]];
  const n = c.pts.length - 1;
  const pts = c.pts.map(([x, y], i): P2 => [x + dx * (1 - i / n), y + dy * (1 - i / n)]);
  return { ...c, pts, forks: c.forks.map((f) => ride(f, pts[f.at])) };
}

/** Draws the channels, added: a wide blue glow, a violet-white body and a white core, on ``ctx`` (its scale: ``px``
 * is a pixel at 1080 lines). */
export function drawChannels(ctx: CanvasRenderingContext2D, channels: Channel[], px: number, glow = 1) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const passes: [number, string, number][] = [ // width (x the channel's), colour, blur: added over a lit wall, a pale
    // glow went white, the bolt a thick white rope
    [6, `rgba(40, 70, 255, ${0.10 * glow})`, 16],
    [2.2, `rgba(110, 100, 255, ${0.32 * glow})`, 5],
    [0.6, "rgba(255, 250, 255, 0.95)", 0],
  ];
  const stroke = (c: Channel, w: number) => {
    ctx.lineWidth = Math.max(0.6, c.width * w * px);
    ctx.beginPath();
    c.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    c.forks.forEach((f) => stroke(f, w));
  };
  for (const [w, colour, blur] of passes) {
    ctx.strokeStyle = colour;
    ctx.shadowColor = colour;
    ctx.shadowBlur = blur * px;
    channels.forEach((c) => stroke(c, w));
  }
  ctx.restore();
}

/** A stream of lightning: its channel, struck anew every 1-3 frames near its last aim (which sways slowly). */
export interface Stream { ch: Channel | null; left: number; aim: number; seed: number }

/** One frame of a stream from ``root``: its channel ridden to the hand, or struck anew towards ``aim`` (radians)
 * ``reach`` px long when its time is up (or ``strike``: a hit strikes every stream at once). */
export function streamStep(s: Stream, root: P2, aim: number, reach: number, width: number, strike: boolean, frame: number): Stream {
  const rand = rng(s.seed * 7919 + frame);
  if (s.ch && s.left > 0 && !strike) return { ...s, ch: ride(s.ch, root), left: s.left - 1 };
  const sway = s.ch ? s.aim + (rand() - 0.5) * 0.5 : aim + (rand() - 0.5) * 0.6;
  const a = 0.6 * sway + 0.4 * aim; // (near its last aim, pulled back to the hand's)
  const l = reach * (0.7 + 0.5 * rand());
  const ch = channel(root, [root[0] + Math.cos(a) * l, root[1] + Math.sin(a) * l], rand, width);
  return { ch, left: 1 + Math.floor(rand() * 3), aim: a, seed: s.seed };
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
