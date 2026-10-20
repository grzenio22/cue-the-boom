import { describe, expect, it } from "vitest";
import {
  apart, type Arcs, arcChannels, arcStep, BALL_FILAMENTS, type Ball, ballRadius, ballStep, beside, BODY_STOPS, BRANCH_W,
  BUZZ_HF, buzzOf, channel, crosses, dense, drawBall, drawBallLight, drawChannels, type Light, lightStep, newArcs,
  type P2, ride, rng, SIDE_APART, streamStep, stretches, TAPER,
} from "./lightning";

describe("channel", () => {
  it("runs from where it is struck to where it strikes, jagged, the same for the same seed", () => {
    const c = channel([0, 0], [100, 0], rng(3));
    expect(c.pts[0]).toEqual([0, 0]);
    expect(c.pts[c.pts.length - 1]).toEqual([100, 0]);
    expect(c.pts.length).toBe(65); // six levels of midpoints
    expect(Math.max(...c.pts.map(([, y]) => Math.abs(y)))).toBeGreaterThan(2); // it kinks
    expect(channel([0, 0], [100, 0], rng(3))).toEqual(c);
    expect(channel([0, 0], [100, 0], rng(4))).not.toEqual(c);
  });

  it("rides the hand at its root and holds where it struck", () => {
    const c = channel([0, 0], [100, 0], rng(5));
    const r = ride(c, [10, 20]);
    expect(r.pts[0]).toEqual([10, 20]);
    expect(r.pts[r.pts.length - 1]).toEqual([100, 0]);
    r.forks.forEach((f) => expect(f.pts[0]).toEqual(r.pts[f.at]));
  });

  it("has dim, thin forks that never cross it or each other", () => {
    let seen = 0;
    for (let seed = 1; seed <= 30; seed++) {
      const c = channel([0, 0], [0, -400], rng(seed), 4, 0.26, 6, 0.3);
      c.forks.forEach((f, i) => {
        seen++;
        expect(f.inten).toBeLessThanOrEqual(0.35);
        expect(f.width).toBeLessThanOrEqual(0.45 * 4 + 1e-9);
        expect(crosses(f.pts, c.pts, 0, 2)).toBe(false);
        c.forks.slice(i + 1).forEach((o) => expect(crosses(f.pts, o.pts)).toBe(false));
      });
    }
    expect(seen).toBeGreaterThan(10);
  });
});

describe("streamStep", () => {
  it("strikes anew when its time is up or on a hit, and rides the hand between", () => {
    let s = streamStep({ ch: null, left: 0, aim: 0, seed: 1 }, [0, 0], 0, 100, 2, false, 1);
    const first = s.ch!;
    expect(s.left).toBeGreaterThanOrEqual(1);
    s = { ...s, left: 2 };
    const held = streamStep(s, [5, 0], 0, 100, 2, false, 2);
    expect(held.ch!.pts[held.ch!.pts.length - 1]).toEqual(first.pts[first.pts.length - 1]); // far end held
    expect(streamStep(s, [5, 0], 0, 100, 2, true, 2).ch).not.toEqual(held.ch); // a hit strikes it anew
  });

  it("leaves along the finger: its first stretch on the aim, its heading at most 12 degrees off it", () => {
    for (let frame = 1; frame <= 40; frame++) {
      const s = streamStep({ ch: null, left: 0, aim: 0, seed: 3 }, [50, 50], -Math.PI / 2, 300, 3, false, frame);
      const pts = s.ch!.pts, end = pts[pts.length - 1];
      const l = Math.hypot(end[0] - 50, end[1] - 50);
      const lead: P2 = [50, 50 - 0.2 * l];
      expect(Math.min(...pts.map(([x, y]) => Math.hypot(x - lead[0], y - lead[1])))).toBeLessThan(1e-9);
      expect(Math.abs(Math.atan2(end[0] - 50, 50 - end[1]))).toBeLessThan(0.21 + 1e-9);
    }
  });
});

const chans = (a: Arcs) => [a.main, ...a.sides].map((s) => s.ch!);

describe("arcStep", () => {
  it("strikes one dominant main channel welded to the fingertip, its streamers dim, thin, short, clear of it and not beside it", () => {
    let arcs = newArcs();
    let sides = 0;
    for (let frame = 1; frame <= 60; frame++) {
      const tip: P2 = [300 + 2 * frame, 600];
      arcs = arcStep(arcs, tip, -Math.PI / 2, 150, 1, frame % 20 === 0, frame);
      const [main, ...rest] = chans(arcs);
      expect(main.pts[0]).toEqual(tip); // (welded)
      expect(rest.length).toBeLessThanOrEqual(2);
      const mainReach = Math.hypot(main.pts[main.pts.length - 1][0] - tip[0], main.pts[main.pts.length - 1][1] - tip[1]);
      rest.forEach((c, i) => {
        sides++;
        expect(c.pts[0]).toEqual(tip);
        expect(c.inten).toBeLessThanOrEqual(0.35);
        expect(c.width).toBeLessThanOrEqual(0.4 * main.width + 1e-9);
        const reach = Math.hypot(c.pts[c.pts.length - 1][0] - tip[0], c.pts[c.pts.length - 1][1] - tip[1]);
        expect(reach).toBeLessThan(0.6 * mainReach); // (half its reach when struck; the hand moves on a little)
        expect(crosses(c.pts, main.pts, 0, 3)).toBe(false);
        expect(beside(c.pts, main.pts, 0.12 * 150)).toBeLessThan(0.35); // (not a second wire beside it)
        const end = c.pts[c.pts.length - 1], far = Math.hypot(end[0] - tip[0], end[1] - tip[1]);
        const there = main.pts.find(([x, y]) => Math.hypot(x - tip[0], y - tip[1]) >= far) ?? main.pts[main.pts.length - 1];
        const off = Math.atan2(end[1] - tip[1], end[0] - tip[0]) - Math.atan2(there[1] - tip[1], there[0] - tip[0]);
        expect(Math.abs(((off + 3 * Math.PI) % (2 * Math.PI)) - Math.PI)).toBeGreaterThan(SIDE_APART[0] - 0.05); // (well
        // round from the main channel where that is as far out: leaving 18 degrees off, two ran beside it as hairlines)
        rest.slice(i + 1).forEach((o) => expect(crosses(c.pts, o.pts, 0, 3)).toBe(false));
      });
    }
    expect(sides).toBeGreaterThan(40); // (two at the loudest, mostly)
  });

  it("tells a streamer that keeps well round from its main channel from one that runs beside it", () => {
    const main: P2[] = Array.from({ length: 40 }, (_, i) => [0, -i * 10]); // (straight up, 390 px)
    const at = (deg: number, bend = 0): P2[] => Array.from({ length: 20 }, (_, i) => {
      const a = (deg - bend * (i / 19)) * Math.PI / 180, r = i * 8;
      return [r * Math.sin(a), -r * Math.cos(a)];
    });
    expect(apart(main, [0, 0], at(30))).toBe(true);
    expect(apart(main, [0, 0], at(12))).toBe(false); // (12 degrees off all along: a twin beside it)
    expect(apart(main, [0, 0], at(-30, -25))).toBe(false); // (leaving 30 degrees off, then bending back towards it)
  });

  it("strikes no streamer for a quiet voice and one for a middling one", () => {
    let quiet = newArcs(), mid = newArcs();
    for (let frame = 1; frame <= 20; frame++) {
      quiet = arcStep(quiet, [0, 0], 0, 150, 0.2, false, frame);
      mid = arcStep(mid, [0, 0], 0, 150, 0.6, false, frame);
      expect(quiet.sides.length).toBe(0);
      expect(mid.sides.length).toBeLessThanOrEqual(1);
    }
  });

  it("is the same for the same frames", () => {
    let a = newArcs(), b = newArcs();
    for (let frame = 1; frame <= 10; frame++) {
      a = arcStep(a, [100, 100], 1, 150, 0.9, false, frame);
      b = arcStep(b, [100, 100], 1, 150, 0.9, false, frame);
    }
    expect(a).toEqual(b);
  });
});

describe("drawing", () => {
  it("fades a branch along its length and keeps the main channel's brightness, thinning both, a branch to a wedge", () => {
    const pts: P2[] = Array.from({ length: 65 }, (_, i) => [0, -i * 5]);
    const main = stretches({ pts, width: 4, inten: 1, forks: [], at: 0 }, 1);
    const branch = stretches({ pts, width: 1.6, inten: 0.35, forks: [], at: 0 }, 1);
    expect(main.every(([, , a]) => a === 1)).toBe(true);
    expect(branch[branch.length - 1][2]).toBeLessThan(0.25 * branch[0][2]); // (a streamer fades nearly out)
    expect(branch[branch.length - 1][2] / branch[0][2]).toBeGreaterThan(1 - TAPER[0] - 0.01);
    for (const s of [main, branch]) for (let k = 1; k < s.length; k++) expect(s[k][1]).toBeLessThan(s[k - 1][1]);
    expect(branch[0][1]).toBeGreaterThan(1.3 * 1.6); // (thicker where it leaves than its width...)
    expect(branch[branch.length - 1][1]).toBeLessThan(0.5 * 1.6); // (...and under half of it at its end)
    expect(Math.min(...branch.map(([, w]) => w))).toBeGreaterThanOrEqual(BRANCH_W[2]);
  });

  it("stays cheap: a frame of the loudest arcs is a few hundred strokes", () => {
    const ctx = fakeCtx();
    let arcs = newArcs();
    for (let frame = 1; frame <= 30; frame++) {
      arcs = arcStep(arcs, [300, 600], -1.4, 150, 1, false, frame);
      drawChannels(ctx as unknown as CanvasRenderingContext2D, arcChannels(arcs, 1), 0.66);
    }
    expect(ctx.strokes / 30).toBeLessThan(400);
  });
});

describe("the light", () => {
  it("holds while the strokes strike, flickering, and dies within a few frames of the last", () => {
    let l: Light | null = null;
    const held: number[] = [];
    for (const p of [1, 0.05, 0.9, 0, 1, 0.1, 0.8, 0.05, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) {
      l = lightStep(l, p);
      held.push(l.held);
    }
    expect(Math.min(...held.slice(0, 10))).toBeGreaterThan(0.75); // (not strobing with them)
    expect(held[15]).toBeLessThan(0.1);
    expect(held[19]).toBe(0);
  });
});

describe("the ball", () => {
  const voice = [0, 0.05, 0.1, ...Array(24).fill(0.9), ...Array(30).fill(0)];
  const run = () => {
    let b: Ball | null = null;
    return voice.map((v, frame) => (b = ballStep(b, v, false, frame + 1)));
  };

  it("grows from a crackle over about 5 frames, and shrinks and goes when the voice stops", () => {
    const balls = run();
    expect(balls[0]).toBeNull();
    expect(balls[1]!.born).toBe(false); // (barely charged: its filaments' crackle, no body yet)
    const first = balls.findIndex((b) => b?.born);
    expect(ballRadius(balls[first]!, 100)).toBeLessThanOrEqual(0.35 * ballRadius(balls[first + 6]!, 100));
    for (let t = first + 1; t < 27; t++) expect(ballRadius(balls[t]!, 100)).toBeGreaterThanOrEqual(ballRadius(balls[t - 1]!, 100));
    expect(dense(balls[first + 2]!.level, balls[first + 2]!.charge)).toBe(1);
    expect(ballRadius(balls[32]!, 100)).toBeLessThan(0.75 * ballRadius(balls[26]!, 100));
    expect(balls[balls.length - 1]).toBeNull();
    const young = balls[1]!; // (its crackle reaches past where its skin will be)
    expect(Math.max(...young.fils.map((f) => Math.hypot(...f.ch!.pts[f.ch!.pts.length - 1])))).toBeGreaterThan(1.2);
  });

  it("is the shot's ball by radius: white-hot in its middle, a saturated blue to a cyan limb, its outline feathered", () => {
    const at = (r: number) => { // (the body's colour at r radii, as its gradient draws it)
      const k = BODY_STOPS.findIndex(([o]) => o >= r);
      const [o1, ...c1] = BODY_STOPS[k], [o0, ...c0] = BODY_STOPS[Math.max(0, k - 1)];
      const u = o1 > o0 ? (r - o0) / (o1 - o0) : 0;
      return c0.map((v, i) => v + u * (c1[i] - v));
    };
    BALL_PROFILE.forEach(([r, ...rgb]) => at(r).slice(0, 3).forEach((v, i) => expect(Math.abs(v - rgb[i])).toBeLessThanOrEqual(8)));
    const sat = ([r, g, b]: number[]) => (Math.max(r, g, b) - Math.min(r, g, b)) / Math.max(r, g, b);
    expect(sat(at(0))).toBeLessThan(0.1); // (white in its middle)
    expect(sat(at(0.75))).toBeGreaterThanOrEqual(0.35); // (a saturated blue towards its limb, not a pale glass sphere)
    expect(Math.min(...BODY_STOPS.filter(([o]) => o <= 0.95).map(([, , , , a]) => a))).toBeGreaterThan(0.95);
    expect(BODY_STOPS[BODY_STOPS.length - 1][4]).toBe(0); // (feathered, not a hard-edged disc)
    const ctx = fakeCtx(); // (and it draws its filaments over the body, the brightest at its skin)
    const balls = run();
    drawBall(ctx as unknown as CanvasRenderingContext2D, balls[20]!, [200, 200], 100, 0.66);
    expect(ctx.fills).toBeGreaterThan(0);
    expect(ctx.strokes).toBeGreaterThan(10);
  });

  it("lights what is close round it once it has a body, and nothing before", () => {
    const balls = run();
    const lit = fakeCtx(), crackle = fakeCtx();
    drawBallLight(lit as unknown as CanvasRenderingContext2D, balls[20]!, [200, 200], 100, null, 400, 400);
    drawBallLight(crackle as unknown as CanvasRenderingContext2D, balls[1]!, [200, 200], 100, null, 400, 400);
    expect(lit.fills).toBeGreaterThan(0);
    expect(crackle.fills).toBe(0);
  });

  it("strikes 5-9 filaments at angles in 3D, reaching its skin at different lengths as seen, not sun-ray spikes", () => {
    const balls = run();
    for (const b of balls.slice(10, 27)) {
      expect(b!.fils.length).toBeGreaterThanOrEqual(BALL_FILAMENTS[0]);
      expect(b!.fils.length).toBeLessThanOrEqual(BALL_FILAMENTS[1]);
    }
    const reach = balls.slice(10, 27).flatMap((b) => b!.fils.map((f) => Math.hypot(...f.ch!.pts[f.ch!.pts.length - 1])));
    expect(Math.min(...reach)).toBeGreaterThan(0.4);
    expect(Math.max(...reach) - Math.min(...reach)).toBeGreaterThan(0.3);
  });
});

/** The shot's ball by radius, as shown: [radii, r, g, b], the median colour of each thin ring of cuesync.lightning's
 * ball over the synthetic palm at full charge at 1080x1920 (test_lightning.py's BALL_PROFILE, which holds the shot to
 * the same numbers): the live preview's body is held within 8 levels of it. */
const BALL_PROFILE: [number, number, number, number][] = [
  [0, 246, 252, 255], [0.15, 224, 242, 255], [0.3, 191, 225, 255], [0.45, 177, 208, 255], [0.6, 172, 202, 255],
  [0.75, 156, 194, 255], [0.9, 172, 214, 255],
];

/** A canvas context that only counts what is drawn (jsdom has no canvas). */
function fakeCtx() {
  const gradient = { addColorStop() {} };
  return {
    strokes: 0, fills: 0, globalCompositeOperation: "", strokeStyle: "", fillStyle: "" as unknown, lineWidth: 1,
    lineJoin: "", lineCap: "",
    save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {}, arc() {}, clip() {},
    stroke() { this.strokes++; }, fill() { this.fills++; }, fillRect() { this.fills++; },
    createRadialGradient: () => gradient,
  };
}

describe("buzzOf", () => {
  const sr = 48000, n = 2048, bins = 1024;
  const spectrumOf = (x: Float32Array) => { // a crude dB spectrum, as the analyser gives
    const out = new Float32Array(bins);
    for (let k = 0; k < bins; k++) {
      let re = 0, im = 0;
      for (let i = 0; i < n; i += 2) { re += x[i] * Math.cos((2 * Math.PI * k * i) / n); im -= x[i] * Math.sin((2 * Math.PI * k * i) / n); }
      out[k] = 10 * Math.log10(re * re + im * im + 1e-12);
    }
    return out;
  };
  const noise = (seed: number, hi: boolean) => { // white, or only high (a first difference, twice)
    const r = rng(seed), x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = r() - 0.5;
    if (hi) for (let pass = 0; pass < 2; pass++) for (let i = n - 1; i > 0; i--) x[i] -= x[i - 1];
    return x;
  };

  it("hears a pitch with a high hiss on it as a buzz, a hiss alone as no pitch", () => {
    const hiss = noise(1, true);
    const buzz = new Float32Array(n);
    for (let i = 0; i < n; i++) buzz[i] = 0.3 * Math.sin((2 * Math.PI * 140 * i) / sr) + 0.5 * hiss[i];
    const b = buzzOf(spectrumOf(buzz), buzz, sr);
    expect(b.voiced).toBe(true);
    expect(b.hf).toBeGreaterThan(BUZZ_HF);
    expect(buzzOf(spectrumOf(hiss), hiss, sr).voiced).toBe(false); // an "fffff": no pitch
    const tone = new Float32Array(n).map((_, i) => 0.3 * Math.sin((2 * Math.PI * 140 * i) / sr));
    expect(buzzOf(spectrumOf(tone), tone, sr).hf).toBeLessThan(BUZZ_HF); // a hum: a pitch, no hiss
  });
});
