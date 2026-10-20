import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  apart, type Arcs, arcChannels, arcStep, type Ball, ballAt, ballRadius, ballStep, beside, BODY_STOPS, bowed, BRANCH_W,
  BUZZ_HF, buzzOf, channel, crosses, dense, drawBall, drawBallLight, drawChannels, GROUND, GROUND_BOW,
  IMPACT_MIN, type Light, LIGHT_TAIL, lightStep, newArcs, type P2, pickTarget, type Plate, ride, rng, SHELL, SHELL_SKIN,
  SIDE_APART, SPLASH, streamStep, stretches, surfaces, TAPER, TARGET_APART, TARGET_BORDER, TARGET_CONE,
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

/** A camera's picture, small (1 / 8 of a 1080 x 1920 canvas): a wall (150) over a floor (60) meeting it along a line
 * ``line`` canvas px down, and a speck of dirt on the wall. */
function plateOf(line: number): Plate {
  const w = 135, h = 240, k = 8;
  const lum = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) lum[y * w + x] = y * k < line ? 150 : 60;
  lum[40 * w + 30] = 20; // (a speck: no surface)
  return { w, h, k, lum };
}

describe("where a bolt strikes", () => {
  const tip: P2 = [540, 1500];
  const up = -Math.PI / 2;

  it("finds the lines of the picture, not its specks, and which way they run", () => {
    const rows = surfaces(plateOf(900));
    expect(rows.length).toBeGreaterThan(50);
    for (const [x, y, , ax, ay] of rows) {
      expect(Math.abs(y - 900)).toBeLessThan(24); // (the floor's line)
      expect(Math.abs(ay)).toBeLessThan(0.2); // (running across the picture)
      expect(Math.hypot(x - 244, y - 324)).toBeGreaterThan(30); // (not the speck)
      expect(Math.hypot(ax, ay)).toBeCloseTo(1, 5);
    }
  });

  it("strikes an edge within the cone round the aim, inside the frame, somewhere new each time", () => {
    const edges = surfaces(plateOf(900));
    const rand = rng(9);
    let last: P2 | null = null;
    for (let i = 0; i < 20; i++) {
      const hit: { at: P2; along: P2 } = pickTarget(edges, tip, up, up, 800, [1080, 1920], rand, 1, last)!;
      expect(hit).not.toBeNull();
      const [x, y] = hit.at;
      expect(Math.abs(y - 900)).toBeLessThan(24);
      const off = Math.abs(Math.atan2(x - tip[0], tip[1] - y)) * 180 / Math.PI;
      expect(off).toBeLessThanOrEqual(TARGET_CONE[1] + 1e-9);
      expect(Math.min(x, 1079 - x, y, 1919 - y)).toBeGreaterThanOrEqual(TARGET_BORDER * 1080 - 1e-9);
      if (last) expect(Math.hypot(x - last[0], y - last[1])).toBeGreaterThanOrEqual(TARGET_APART);
      last = hit.at;
    }
    expect(pickTarget(edges, tip, up, up, 300, [1080, 1920], rand)).toBeNull(); // (out of its reach: none)
    expect(pickTarget(edges, tip, 0, 0, 800, [1080, 1920], rand)).toBeNull(); // (aimed along it, not at it: none)
  });

  it("strikes it with a contact flash and a splash for its first frames; with nothing there, ends in a brush", () => {
    const plate = plateOf(900);
    let s = streamStep({ ch: null, left: 0, aim: up, seed: 4 }, tip, up, 800, 4, false, 1, 1, plate);
    const end = s.ch!.pts[s.ch!.pts.length - 1];
    expect(Math.abs(end[1] - 900)).toBeLessThan(24);
    expect(s.ch!.strikes).toBeDefined();
    expect(s.splash!.length).toBeGreaterThanOrEqual(SPLASH[0]);
    for (const f of s.splash!) {
      expect(f.pts[0]).toEqual(end);
      expect(f.inten).toBeGreaterThanOrEqual(SPLASH[4]); // (at least half as bright as its stroke)
      const reach = Math.max(...f.pts.map(([x, y]) => Math.hypot(x - end[0], y - end[1])));
      expect(reach).toBeLessThanOrEqual(1.3 * SPLASH[3]);
    }
    const arcs: Arcs = { main: s, sides: [] };
    expect(arcChannels(arcs, 1).length).toBe(1 + s.splash!.length);
    s = { ...s, left: 3 };
    for (let f = 2; f <= 3; f++) s = streamStep(s, tip, up, 800, 4, false, f, 1, plate);
    expect(arcChannels({ main: s, sides: [] }, 1).length).toBe(1); // (its splash gone after SPLASH's frames)
    const blank = streamStep({ ch: null, left: 0, aim: up, seed: 4 }, tip, up, 800, 4, false, 1, 1,
      { w: 135, h: 240, k: 8, lum: new Float32Array(135 * 240).fill(150) });
    expect(blank.ch!.strikes).toBeUndefined();
    expect(blank.splash).toEqual([]);
    const tail = blank.ch!.pts.length * 0.8;
    expect(blank.ch!.forks.filter((f) => f.at >= tail).length).toBeGreaterThanOrEqual(2); // (a brush at its end)
  });

  it("draws the contact flash only for a bright stroke", () => {
    const s = streamStep({ ch: null, left: 0, aim: up, seed: 4 }, tip, up, 800, 4, false, 1, 1, plateOf(900));
    const bright = fakeCtx(), dim = fakeCtx();
    drawChannels(bright as unknown as CanvasRenderingContext2D, [[s.ch!, 1]], 1);
    drawChannels(dim as unknown as CanvasRenderingContext2D, [[s.ch!, 0.5 * IMPACT_MIN]], 1);
    expect(bright.fills).toBeGreaterThanOrEqual(2); // (its pool and its white-hot point)
    expect(dim.fills).toBe(0);
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
  it("holds while the strokes strike, flickering, and goes with the last of them", () => {
    let l: Light | null = null;
    const held: number[] = [];
    for (const p of [1, 0.05, 0.9, 0.1, 1, 0.1, 0.8, 0.05, 1, 0, 0, 0, 0.9, 0.9, 0]) {
      l = lightStep(l, p);
      held.push(l.held);
    }
    expect(Math.min(...held.slice(0, 9))).toBeGreaterThan(0.75); // (not strobing with them)
    expect(held[9]).toBeCloseTo(LIGHT_TAIL * held[8], 9); // (a tenth of it for a frame...)
    expect(held[10]).toBe(0); // (...then none: no lit room with no bolt in it)
    expect(held[11]).toBe(0);
    expect(held[12]).toBeCloseTo(0.9, 9); // (back with the next stroke)
  });
});

/** The shot's ball by radius, as shown: [radii, r, g, b], read from test_lightning.py's BALL_PROFILE (the median
 * colour of each thin ring of cuesync.lightning's ball over a synthetic palm at full charge at 1080x1920, which holds
 * the shot to it): the live preview's body is held within 8 levels of it, so the two cannot drift apart. */
const BALL_PROFILE: [number, number, number, number][] = (() => {
  const src = readFileSync(join(import.meta.dirname, "../../cuesync/tests/test_lightning.py"), "utf8");
  const block = src.slice(src.indexOf("BALL_PROFILE = ("), src.indexOf("# (radii, r, g, b)", src.indexOf("BALL_PROFILE = (")));
  return [...block.matchAll(/\(([\d.]+), (\d+), (\d+), (\d+)\)/g)]
    .map((m) => [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4])] as [number, number, number, number]);
})();

describe("the ball", () => {
  const voice = [0, 0.05, 0.1, ...Array(24).fill(0.9), ...Array(30).fill(0)];
  const run = () => {
    let b: Ball | null = null;
    return voice.map((v, frame) => (b = ballStep(b, v, false, frame + 1)));
  };

  it("grows from a crackle over about 5 frames, and shrinks and goes when the voice stops", () => {
    const balls = run();
    expect(balls[0]).toBeNull();
    expect(balls[1]!.born).toBe(false); // (barely charged: its shell's crackle, no body yet)
    expect(balls[1]!.shell.length).toBeGreaterThanOrEqual(SHELL[0]);
    const first = balls.findIndex((b) => b?.born);
    expect(ballRadius(balls[first]!, 100)).toBeLessThanOrEqual(0.35 * ballRadius(balls[first + 6]!, 100));
    for (let t = first + 1; t < 27; t++) expect(ballRadius(balls[t]!, 100)).toBeGreaterThanOrEqual(ballRadius(balls[t - 1]!, 100));
    expect(dense(balls[first + 2]!.level, balls[first + 2]!.charge)).toBe(1);
    expect(ballRadius(balls[32]!, 100)).toBeLessThan(0.75 * ballRadius(balls[26]!, 100));
    expect(balls[balls.length - 1]).toBeNull();
  });

  it("is the shot's ball by radius: the profile test_lightning.py holds the shot to", () => {
    expect(BALL_PROFILE.length).toBe(7);
    const at = (r: number) => { // (the body's colour at r radii, as its gradient draws it)
      const k = BODY_STOPS.findIndex(([o]) => o >= r);
      const [o1, ...c1] = BODY_STOPS[k], [o0, ...c0] = BODY_STOPS[Math.max(0, k - 1)];
      const u = o1 > o0 ? (r - o0) / (o1 - o0) : 0;
      return c0.map((v, i) => v + u * (c1[i] - v));
    };
    BALL_PROFILE.forEach(([r, ...rgb]) => at(r).slice(0, 3).forEach((v, i) => expect(Math.abs(v - rgb[i])).toBeLessThanOrEqual(8)));
    const sat = ([r, g, b]: number[]) => (Math.max(r, g, b) - Math.min(r, g, b)) / Math.max(r, g, b);
    expect(sat(at(0))).toBeLessThan(0.1); // (white in its middle)
    expect(sat(at(0.5))).toBeGreaterThanOrEqual(0.25); // (a clear cyan-blue round it, not blown out)
    expect(sat(at(0.5))).toBeLessThanOrEqual(0.6);
    const alpha = (r: number) => at(r)[3];
    expect(Math.min(...[0, 0.15, 0.3, 0.45, 0.6].map(alpha))).toBeGreaterThan(0.95); // (dense in its middle,
    expect(alpha(1)).toBeLessThanOrEqual(0.5); // light over the palm at its edge,
    const [last] = BODY_STOPS[BODY_STOPS.length - 1];
    expect(last).toBeGreaterThan(1.05); // and gone a little past its radius: no outline)
    expect(BODY_STOPS[BODY_STOPS.length - 1][4]).toBe(0);
    const ctx = fakeCtx(); // (and it draws its shell over the body)
    const balls = run();
    drawBall(ctx as unknown as CanvasRenderingContext2D, balls[20]!, [200, 400], 100, 0.66);
    expect(ctx.fills).toBeGreaterThan(0);
    expect(ctx.strokes).toBeGreaterThan(10);
  });

  it("hovers off the palm, up the picture", () => {
    expect(ballAt([200, 400], 1)[0]).toBe(200);
    expect(400 - ballAt([200, 400], 1)[1]).toBeGreaterThanOrEqual(GROUND_BOW[0]);
  });

  it("lights what is close round it once it has a body, and nothing before", () => {
    const balls = run();
    const lit = fakeCtx(), crackle = fakeCtx();
    drawBallLight(lit as unknown as CanvasRenderingContext2D, balls[20]!, [200, 200], 100, null, 400, 400);
    drawBallLight(crackle as unknown as CanvasRenderingContext2D, balls[1]!, [200, 200], 100, null, 400, 400);
    expect(lit.fills).toBeGreaterThan(0);
    expect(crackle.fills).toBe(0);
  });

  it("crawls a shell of arcs over its skin, held to it, never a starburst from its middle", () => {
    const balls = run();
    for (const b of balls.slice(10, 27)) {
      expect(b!.shell.length).toBeGreaterThanOrEqual(SHELL[0]);
      expect(b!.shell.length).toBeLessThanOrEqual(SHELL[1]);
      for (const s of b!.shell) {
        const r = s.pts.map((p) => Math.hypot(...p));
        const sphere = s.inten < 1 ? r[16] : 1;
        expect(Math.max(...r.map((v) => Math.abs(v - sphere)))).toBeLessThanOrEqual(SHELL_SKIN + 0.12);
        expect(Math.min(...r)).toBeGreaterThan(0.35); // (none from its middle)
      }
    }
  });

  it("throws 6-10 arcs to the fingers, each bowing up through the air from its skin onto a finger", () => {
    const b = run()[20]!;
    expect(b.jumps.length).toBeGreaterThanOrEqual(GROUND[0]);
    expect(b.jumps.length).toBeLessThanOrEqual(GROUND[1]);
    const unit = channel([0, 0], [1, 0], rng(2), 1, 0, 5, 0); // (straight: the bow alone)
    const arc = bowed(unit, [100, 300], [300, 300], 40, [200, 400], 2);
    const mid = arc.pts[(arc.pts.length - 1) / 2];
    expect(arc.pts[0]).toEqual([100, 300]);
    expect(arc.pts[arc.pts.length - 1][0]).toBeCloseTo(300, 9);
    expect(300 - mid[1]).toBeCloseTo(40, 6); // (up the picture, as far as it bows)
    const side = bowed(unit, [200, 100], [200, 300], 40, [100, 200], 2); // (running down the picture, right of the palm)
    expect(side.pts[(side.pts.length - 1) / 2][0] - 200).toBeCloseTo(40, 6); // (out from the palm's middle)
    const lm: P2[] = Array.from({ length: 21 }, (_, i) => [100 + 10 * i, 600 + (i % 4) * 30]);
    const ctx = fakeCtx();
    drawBallLight(ctx as unknown as CanvasRenderingContext2D, b, [200, 700], 100, lm, 400, 800);
    expect(ctx.strokes).toBeGreaterThan(b.jumps.length);
  });
});

/** A canvas context that only counts what is drawn (jsdom has no canvas). */
function fakeCtx() {
  const gradient = { addColorStop() {} };
  return {
    strokes: 0, fills: 0, globalCompositeOperation: "", strokeStyle: "", fillStyle: "" as unknown, lineWidth: 1,
    lineJoin: "", lineCap: "",
    save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {}, arc() {}, clip() {}, translate() {}, rotate() {},
    scale() {},
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
    expect(buzzOf(spectrumOf(tone), tone, sr).hf).toBeLessThan(BUZZ_HF);
  });
});

