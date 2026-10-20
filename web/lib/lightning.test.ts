import { describe, expect, it } from "vitest";
import {
  type Arcs, arcChannels, arcStep, type Ball, BALL_HAND_RGB, BALL_LIGHT, BALL_ROOM_RGB, ballAt, ballRadius,
  ballStep, BODY_STOPS, BRANCH_W, BUZZ_HF, buzzOf, channel, crosses, dense, DIP, drawBall, drawBallLight,
  drawChannels, drawLand, drawLight, GROUND, GROUND_BOW, GROUND_FREE, GROUND_GONE, GROUND_ON, IMPACT_MIN, jumpPath,
  LAND_NEAR, leader, LEADER, type Light, LIGHT_TAIL, lightStep, newArcs, onHand, type P2, pickTarget, type Plate,
  plasmaAt, plasmaTurns, PRONG, ride, rng, SHELL, SHELL_SKIN, SPLASH, streamStep, stretches, surfaces,
  swing, TAPER, TARGET_APART, TARGET_AT, TARGET_BORDER, TARGET_CONE,
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
 * ``line`` canvas px down (and, with ``also``, a dark band of the wall from there down to ``line``: a second line),
 * and a speck of dirt on the wall. */
function plateOf(line: number, also = 0): Plate {
  const w = 135, h = 240, k = 8;
  const lum = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) lum[y * w + x] = y * k < line ? (also && y * k >= also ? 100 : 150) : 60;
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
      expect(off).toBeLessThanOrEqual(TARGET_CONE[last ? 2 : 1] + 1e-9); // (struck again: a wider cone)
      expect(Math.min(x, 1079 - x, y, 1919 - y)).toBeGreaterThanOrEqual(TARGET_BORDER * 1080 - 1e-9);
      if (last) expect(Math.hypot(x - last[0], y - last[1])).toBeGreaterThanOrEqual(TARGET_APART);
      last = hit.at;
    }
    expect(pickTarget(edges, tip, up, up, 300, [1080, 1920], rand)).toBeNull(); // (out of its reach: none)
    expect(pickTarget(edges, tip, 0, 0, 800, [1080, 1920], rand)).toBeNull(); // (aimed along it, not at it: none)
    const askew = -10 * Math.PI / 180; // (aimed 46 degrees and more off every point of it: only looking wider)
    expect(pickTarget(edges, tip, askew, askew, 800, [1080, 1920], rand)).toBeNull();
    const wide = pickTarget(edges, tip, askew, askew, 800, [1080, 1920], rand, 1, null, false, true)!;
    expect(Math.abs(wide.at[1] - 900)).toBeLessThan(24);
  });

  it("strikes as far out as it is asked, and now and then in two prongs", () => {
    const edges = surfaces(plateOf(1100, 500)); // (lines 400 and 1000 px over the fingertip)
    const rand = rng(5);
    for (let i = 0; i < 10; i++) {
      expect(Math.abs(pickTarget(edges, tip, up, up, 450, [1080, 1920], rand)!.at[1] - 1100)).toBeLessThan(24);
      expect(Math.abs(pickTarget(edges, tip, up, up, 1000, [1080, 1920], rand)!.at[1] - 500)).toBeLessThan(24);
    }
    const two = pickTarget(edges, tip, up, up, 1000, [1080, 1920], rand, 1, null, true)!;
    const gap = Math.hypot(two.two!.at[0] - two.at[0], two.two!.at[1] - two.at[1]);
    expect(gap).toBeGreaterThanOrEqual(PRONG[1]);
    expect(gap).toBeLessThanOrEqual(PRONG[2]);
    let prongs = 0, struck = 0;
    for (let seed = 1; seed <= 80; seed++) {
      const s = streamStep({ ch: null, left: 0, aim: up, seed }, tip, up, 1300, 4, false, 1, 1, plateOf(1100, 500), 1,
        1000);
      if (!s.ch!.strikes) continue;
      struck++;
      const p = s.ch!.forks.filter((f) => f.strikes);
      prongs += p.length;
      for (const f of p) expect(f.inten).toBeGreaterThanOrEqual(0.7); // (a prong: as bright as a channel)
    }
    expect(prongs / struck).toBeGreaterThan(0.08);
    expect(prongs / struck).toBeLessThan(0.45);
  });

  it("reaches and strikes as far as the canvas is big and the voice loud, whatever the hand's size", () => {
    const plate = plateOf(1100, 500);
    const ends = (scale: number, level: number) => {
      let arcs = newArcs(7), far = 0;
      for (let frame = 1; frame <= 12; frame++) {
        arcs = arcStep(arcs, tip, up, scale, level, frame === 1, frame, 1, plate);
        const p = arcs.main.ch!.pts;
        far = Math.max(far, Math.hypot(p[p.length - 1][0] - tip[0], p[p.length - 1][1] - tip[1]));
      }
      return far;
    };
    const diag = Math.hypot(1080, 1920);
    expect(Math.abs(ends(100, 1) - ends(300, 1))).toBeLessThan(30); // (not the hand's size)
    expect(ends(150, 1)).toBeGreaterThan(0.75 * TARGET_AT[1] * diag); // (loud: the far line)
    expect(ends(150, 0.1)).toBeLessThan(0.75 * TARGET_AT[1] * diag); // (quiet: the near one)
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
    const [ex, ey] = blank.ch!.pts[blank.ch!.pts.length - 1]; // (in the air inside the picture, not off it)
    expect(Math.min(ex, 1079 - ex, ey, 1919 - ey)).toBeGreaterThanOrEqual(TARGET_BORDER * 1080 - 1e-6);
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

describe("arcStep", () => {
  it("never invents a second emitter from the side of a single tracked fingertip", () => {
    let arcs = newArcs();
    for (let frame = 1; frame <= 12; frame++) {
      arcs = arcStep(arcs, [540, 1500], -Math.PI / 2, 220, 1, frame === 1, frame, 1, plateOf(900));
      expect(arcs.main.ch!.strikes).toBeDefined();
      expect(arcs.sides).toEqual([]);
    }
  });

  it("strikes a broad low-texture wall patch when no contrast edge faces the finger", () => {
    const plate: Plate = { w: 135, h: 240, k: 8, lum: new Float32Array(135 * 240).fill(150) };
    const arcs = arcStep(newArcs(), [540, 1500], -Math.PI / 2, 220, 1, true, 1, 1, plate);
    expect(arcChannels(arcs, 1).length).toBeGreaterThan(0);
    expect(arcs.main.ch!.diffuse).toBe(true);
    const end = arcs.main.ch!.pts.at(-1)!;
    expect(end[1]).toBeLessThan(1200);
    expect(Math.min(end[0], 1079 - end[0], end[1], 1919 - end[1])).toBeGreaterThan(120);
    expect(arcs.main.splash).toEqual([]);
  });

  it("prefers a real side surface up to 120 degrees off the finger before a featureless wall patch", () => {
    const plate: Plate = { w: 135, h: 240, k: 8, lum: new Float32Array(135 * 240).fill(150) };
    for (let y = 194; y < 220; y++) for (let x = 10; x < 22; x++) plate.lum[y * plate.w + x] = 60;
    const arcs = arcStep(newArcs(), [540, 1500], -Math.PI / 2, 220, 0.6, true, 1, 1, plate);
    const main = arcs.main.ch!, end = main.pts.at(-1)!;
    expect(main.strikes).toBeDefined();
    expect(main.diffuse).not.toBe(true);
    expect(end[0]).toBeLessThan(190);
    expect(end[1]).toBeGreaterThan(1500);
    const off = Math.abs(Math.atan2(end[0] - 540, 1500 - end[1])) * 180 / Math.PI;
    expect(off).toBeGreaterThan(70);
    expect(off).toBeLessThanOrEqual(120);
    const span = Math.hypot(end[0] - 540, end[1] - 1500);
    expect(main.pts).toContainEqual([540, 1500 - 0.07 * span]);
  });

  it("drops a held channel when the camera picture becomes unreadable", () => {
    let arcs = arcStep(newArcs(), [540, 1500], -Math.PI / 2, 220, 1, true, 1, 1, plateOf(900));
    arcs.main.left = 2;
    arcs = arcStep(arcs, [540, 1500], -Math.PI / 2, 220, 1, false, 2, 1, null);
    expect(arcChannels(arcs, 1)).toEqual([]);
  });

  it("keeps the main channel attached to the current fingertip while its surface contact stays put", () => {
    const plate = plateOf(900);
    let arcs = arcStep(newArcs(), [540, 1500], -Math.PI / 2, 220, 1, true, 1, 1, plate);
    const end = arcs.main.ch!.pts.at(-1);
    arcs.main.left = 2;
    arcs = arcStep(arcs, [550, 1490], -Math.PI / 2, 220, 1, false, 2, 1, plate);
    expect(arcs.main.ch!.pts[0]).toEqual([550, 1490]);
    expect(arcs.main.ch!.pts.at(-1)).toEqual(end);
    expect(arcs.main.ch!.rootSoft).toBeGreaterThan(1);
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
  it("breaks the main filament into uneven thickness and brightness without flattening dim branches", () => {
    const c = channel([540, 1500], [540, 900], rng(3), 4);
    const main = stretches(c, 1);
    expect(Math.max(...main.map(([, , a]) => a)) - Math.min(...main.map(([, , a]) => a))).toBeGreaterThan(0.2);
    expect(main.some(([, w], i) => i > 0 && w > main[i - 1][1])).toBe(true);
  });

  it("fades and tapers a branch into a dim fine wedge below its main filament", () => {
    const pts: P2[] = Array.from({ length: 65 }, (_, i) => [0, -i * 5]);
    const main = stretches({ pts, width: 4, inten: 1, forks: [], at: 0 }, 1);
    const branch = stretches({ pts, width: 1.6, inten: 0.35, forks: [], at: 0 }, 1);
    expect(Math.max(...branch.map(([, , a]) => a))).toBeLessThan(Math.min(...main.map(([, , a]) => a)));
    expect(branch[branch.length - 1][2]).toBeLessThan(0.25 * branch[0][2]); // (a streamer fades nearly out)
    expect(branch[branch.length - 1][2] / branch[0][2]).toBeGreaterThan(1 - TAPER[0] - 0.01);
    for (let k = 1; k < branch.length; k++) expect(branch[k][1]).toBeLessThan(branch[k - 1][1]);
    expect(branch[0][1]).toBeGreaterThan(1.3 * 1.6); // (thicker where it leaves than its width...)
    expect(branch[branch.length - 1][1]).toBeLessThan(0.5 * 1.6); // (...and under half of it at its end)
    expect(Math.min(...branch.map(([, w]) => w))).toBeGreaterThanOrEqual(BRANCH_W[2]);
  });

  it("stays cheap: a frame of the loudest arcs is a few hundred strokes", () => {
    const ctx = fakeCtx();
    let arcs = newArcs();
    for (let frame = 1; frame <= 30; frame++) {
      arcs = arcStep(arcs, [540, 1500], -1.4, 150, 1, false, frame, 1, plateOf(900));
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

  it("has a translucent core with no opaque white dot or bright circular rim", () => {
    expect(Math.max(...BODY_STOPS.map(([, , , , a]) => a))).toBeLessThan(0.8);
    const middle = BODY_STOPS.filter(([r]) => r < 0.65);
    const edge = BODY_STOPS.filter(([r]) => r >= 0.85);
    expect(Math.max(...edge.map(([, , , , a]) => a))).toBeLessThan(0.3 * Math.max(...middle.map(([, , , , a]) => a)));
    const ctx = fakeCtx();
    drawBall(ctx as unknown as CanvasRenderingContext2D, run()[20]!, [200, 400], 100, 0.66);
    expect(ctx.strokes).toBeGreaterThan(10);
    expect(ctx.arcs).toEqual([]);
  });

  it("uses the tracked fingers to hide rear filaments while leaving front filaments visible", () => {
    const lm = handUp(200, 600, 100), palm: P2 = [200, 610];
    const b = { ...run()[20]!, grown: 1, most: 1, pulse: 0, turn: 0, shell: [{
      pts: Array.from({ length: 17 }, (_, i): [number, number, number] =>
        [-0.52 + i * 0.005, (0.9 - 0.5 * Math.sin(0.35)) / Math.cos(0.35), -0.5]),
      inten: 1, left: 1, flash: 1,
    }] };
    const open = fakeCtx(), hidden = fakeCtx();
    drawBall(open as unknown as CanvasRenderingContext2D, b, palm, 200, 1);
    drawBallLight(hidden as unknown as CanvasRenderingContext2D, { ...b, jumps: [] }, palm, 200, lm, 400, 800, 1);
    const before = hidden.strokes;
    drawBall(hidden as unknown as CanvasRenderingContext2D, b, palm, 200, 1);
    expect(hidden.strokes - before).toBeLessThan(open.strokes);
    const front = { ...b, shell: b.shell.map((s) => ({ ...s, pts: s.pts.map(([x, y, z]): [number, number, number] => [x, y, -z]) })) };
    const frontCtx = fakeCtx();
    drawBallLight(frontCtx as unknown as CanvasRenderingContext2D, { ...front, jumps: [] }, palm, 200, lm, 400, 800, 1);
    const frontBefore = frontCtx.strokes;
    drawBall(frontCtx as unknown as CanvasRenderingContext2D, front, palm, 200, 1);
    expect(frontCtx.strokes - frontBefore).toBeGreaterThan(hidden.strokes - before);
  });

  it("keeps rear filaments visible over the receiving palm below the distal fingers", () => {
    const lm = handUp(200, 600, 100), palm: P2 = [200, 760];
    const b = { ...run()[20]!, grown: 1, most: 1, pulse: 0, turn: 0, jumps: [], shell: [{
      pts: Array.from({ length: 17 }, (_, i): [number, number, number] =>
        [-0.1 + i * 0.005, (0.3 - 0.5 * Math.sin(0.35)) / Math.cos(0.35), -0.5]),
      inten: 1, left: 1, flash: 1,
    }] };
    const open = fakeCtx(), palmCovered = fakeCtx();
    drawBall(open as unknown as CanvasRenderingContext2D, b, palm, 200, 1);
    drawBallLight(palmCovered as unknown as CanvasRenderingContext2D, b, palm, 200, lm, 400, 800, 1);
    const before = palmCovered.strokes;
    drawBall(palmCovered as unknown as CanvasRenderingContext2D, b, palm, 200, 1);
    expect(palmCovered.strokes - before).toBe(open.strokes);
    expect(open.strokes).toBeGreaterThan(0);
  });

  it("hovers off the palm, up the picture", () => {
    expect(ballAt([200, 400], 1)[0]).toBe(200);
    expect(400 - ballAt([200, 400], 1)[1]).toBeGreaterThanOrEqual(100);
  });

  it("lights what is close round it once it has a body, gone by three radii, and nothing before", () => {
    const balls = run();
    const lit = fakeCtx(), crackle = fakeCtx();
    drawBallLight(lit as unknown as CanvasRenderingContext2D, balls[20]!, [200, 200], 100, null, 400, 400);
    drawBallLight(crackle as unknown as CanvasRenderingContext2D, balls[1]!, [200, 200], 100, null, 400, 400);
    expect(lit.fills).toBeGreaterThan(0);
    expect(crackle.fills).toBe(0);
    const R = ballRadius(balls[20]!, 100);
    for (const g of lit.gradients) {
      expect(g.r1).toBeCloseTo(BALL_LIGHT[4] * R, 6); // (out to its reach...)
      const alpha = g.stops.map(([, c]) => Number(c.match(/([\d.]+)\)$/)![1]));
      expect(alpha[alpha.length - 1]).toBe(0); // (...and gone there)
      expect(alpha[0]).toBeLessThan(0.05); // (what its dense middle hides, barely)
      expect(Math.max(...alpha)).toBeGreaterThan(0.05);
    }
    const wash = fakeCtx(); // (and none of the broad wash the arcs throw: a colour grade over the room)
    drawLight(wash as unknown as CanvasRenderingContext2D, [200, 200], 100, 1, null, BALL_ROOM_RGB, BALL_HAND_RGB, 400,
      400);
    expect(wash.fills).toBe(0);
    expect(DIP[0]).toBeLessThanOrEqual(0.15); // (an eighth of a stop)
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

  it("strikes its arcs a side at a time, now one, now three, now none, and now and then one to a finger's pad", () => {
    const voice_ = [0, 0.05, 0.1, ...Array(60).fill(0.9)];
    let b: Ball | null = null, both = 0, oneSide = 0, pads = 0;
    const struck: number[][] = [];
    voice_.forEach((v, frame) => {
      const prev: Ball | null = b;
      b = ballStep(b, v, frame === 30, frame + 1);
      if (!b || frame < 12) return;
      const fresh = b.jumps.filter((j) => j.flash === 1 && !j.pad && !(prev?.jumps ?? []).some((o) => o.seed === j.seed));
      const sides = new Set(fresh.map((j) => j.side));
      if (frame !== 30) expect(sides.size).toBeLessThanOrEqual(1); // (a side strikes a frame, but on a hit)
      else expect(sides.size).toBe(2);
      for (const s of [-1, 1]) expect(b.jumps.filter((j) => j.side === s).length).toBeLessThanOrEqual(3);
      const on = [-1, 1].map((s) => b!.jumps.some((j) => j.side === s));
      both += Number(on[0] && on[1]);
      oneSide += Number(on[0] !== on[1]);
      pads += b.jumps.filter((j) => j.pad).length;
      struck.push(fresh.map((j) => j.side));
    });
    expect(oneSide).toBeGreaterThan(10); // (uneven: not a pair of wings every frame)
    expect(both).toBeLessThan(0.7 * struck.length);
    expect(pads).toBeGreaterThan(3);
    expect(GROUND.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 9);
  });

  it("draws each arc in straight steps kinked 20-50 degrees, its course never turning back on itself", () => {
    const rand = rng(4);
    const kinks: number[] = [];
    for (let i = 0; i < 40; i++) {
      const q = leader(300, 0.25, rand, 1);
      expect(q[0]).toEqual([0, 0]);
      expect(q[q.length - 1]).toEqual([1, 0]);
      expect(swing(q)).toBeLessThan(180);
      const d = q.slice(1).map((p, k) => [(p[0] - q[k][0]) * 300, (p[1] - q[k][1]) * 300]);
      d.slice(0, -1).forEach(([x, y]) => {
        expect(Math.hypot(x, y)).toBeGreaterThanOrEqual(LEADER[0] - 1e-6);
        expect(Math.hypot(x, y)).toBeLessThanOrEqual(LEADER[1] + 1e-6);
      });
      for (let k = 1; k < d.length - 1; k++) {
        const t = Math.atan2(d[k][1], d[k][0]) - Math.atan2(d[k - 1][1], d[k - 1][0]);
        kinks.push((Math.abs(((t + 3 * Math.PI) % (2 * Math.PI)) - Math.PI) * 180) / Math.PI);
      }
    }
    expect(kinks.filter((k) => k >= 15 && k <= 60).length).toBeGreaterThan(0.6 * kinks.length);
    expect(swing(Array.from({ length: 40 }, (_, i): P2 => [Math.cos(i / 5), Math.sin(i / 5)]))).toBeGreaterThan(300);
  });

  it("throws each arc from its skin facing its finger, through the air, landing on the finger by its tip", () => {
    const b = run()[20]!;
    expect(b.jumps.length).toBeLessThanOrEqual(7);
    const lm = handUp(200, 600, 100); // (a hand held up, the palm to the camera, the ball over its knuckles)
    const hand = (p: P2) => onHand(p, lm, 100);
    const c = ballAt([200, 610], 1), R = 0.4 * 100;
    let found = 0;
    for (let k = 0; k < 40; k++) {
      const rand = rng(k + 1);
      const tip = [4, 8, 12, 16, 20][k % 5];
      const shape = leader(Math.hypot(lm[tip][0] - c[0], lm[tip][1] - c[1]) - R, (rand() < 0.5 ? -1 : 1) * GROUND_BOW[k % 2],
        rand, 1);
      const pts = jumpPath(shape, c, R, lm[tip], 0.2 * (2 * rand() - 1), 1, hand, 100);
      if (!pts) continue;
      found++;
      const end = pts[pts.length - 1];
      expect(Math.hypot(pts[0][0] - c[0], pts[0][1] - c[1])).toBeCloseTo(R, 6); // (from its skin)
      expect(Math.hypot(end[0] - lm[tip][0], end[1] - lm[tip][1])).toBeLessThanOrEqual(LAND_NEAR + 1e-9);
      expect(hand(end)).toBe(true); // (it lands on the finger...)
      let over = 0, free = 0;
      const cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      for (let i = 1; i < pts.length; i++) {
        if (Math.hypot(pts[i][0] - c[0], pts[i][1] - c[1]) <= GROUND_FREE * R || cum[i] >= cum[cum.length - 1] - LAND_NEAR) continue;
        free += cum[i] - cum[i - 1];
        if (hand(pts[i])) over += cum[i] - cum[i - 1];
      }
      expect(over).toBeLessThanOrEqual(GROUND_ON[0] * free + 1e-9); // (...and is in the air till then)
    }
    expect(found).toBeGreaterThan(5);
    const gap: P2 = [(lm[8][0] + lm[12][0]) / 2, lm[8][1] - 40]; // (a landmark off the hand, between two fingers)
    expect(jumpPath(leader(150, 0.2, rng(1), 1), c, R, gap, 0, 1, hand, 100)).toBeNull();
    const ctx = fakeCtx();
    drawBallLight(ctx as unknown as CanvasRenderingContext2D, b, [200, 610], 100, lm, 400, 800);
    expect(ctx.strokes).toBeGreaterThan(0);
  });

  it("lands its arcs as a white-hot point in a blue glow, the skin under it taken towards cyan, not a pink halo", () => {
    const ctx = fakeCtx();
    drawLand(ctx as unknown as CanvasRenderingContext2D, [100, 100], 1, 1);
    for (const g of ctx.gradients) {
      for (const [, col] of g.stops) {
        const [r, , b_] = col.match(/[\d.]+/g)!.map(Number);
        expect(r).toBeLessThanOrEqual(b_); // (no warm tint anywhere)
      }
    }
    expect(ctx.ops).toContain("multiply");
  });

  it("thins out its arcs as it shrinks, none under GROUND_GONE of its largest, and its shell goes with its body", () => {
    const balls = run();
    const top = Math.max(...balls.map((b) => (b ? ballRadius(b, 1) : 0)));
    let seen = 0;
    for (const b of balls.slice(27)) {
      if (!b) continue;
      if (ballRadius(b, 1) < GROUND_GONE * top) expect(b.jumps.length).toBe(0);
      if (b.born && b.most >= 0.5 && dense(b.level, b.charge) === 0) {
        seen++;
        expect(b.fade).toBe(0); // (its body gone: its crackle with it)
      }
    }
    expect(balls.slice(15, 27).some((b) => b!.jumps.length > 0)).toBe(true);
    expect(balls.slice(27).filter(Boolean).at(-1)!.jumps.length).toBe(0);
    expect(seen + balls.slice(27).filter((b) => !b).length).toBeGreaterThan(0);
  });

  it("has moving plasma on its sphere, not a flat tile turned in the picture", () => {
    const now = plasmaTurns(0), later = plasmaTurns(0.1);
    const ring = (turns: ReturnType<typeof plasmaTurns>, r: number) =>
      Array.from({ length: 64 }, (_, i) => plasmaAt(r * Math.cos((i * Math.PI) / 32), r * Math.sin((i * Math.PI) / 32), turns));
    const a = ring(now, 0.5), b = ring(later, 0.5);
    expect(Math.max(...a) - Math.min(...a)).toBeGreaterThan(0.3); // (filaments)
    expect(a.some((v, i) => Math.abs(v - b[i]) > 0.1)).toBe(true); // (moving)
    expect(plasmaAt(0.99, 0.1, now)).toBeCloseTo(plasmaAt(0.99, 0.1, now), 12);
    expect(Math.abs(plasmaAt(0.995, 0, now))).toBeLessThan(0.25); // (faded out over its limb)
    expect(plasmaAt(1.01, 0, now)).toBe(0);
    // turned as a flat tile, a point on its outline would keep the value a point at its middle's height had: on the
    // sphere, the outline's filaments pass edge on, not as the middle's
    const tile = (r: number, t: number) => plasmaAt(r * Math.cos(t), r * Math.sin(t), now);
    expect(Array.from({ length: 16 }, (_, i) => Math.abs(tile(0.3, i) - tile(0.9, i))).some((d) => d > 0.05)).toBe(true);
  });
});

/** A canvas context that only counts what is drawn (jsdom has no canvas), and keeps its radial gradients (their outer
 * radius and colour stops). */
function fakeCtx() {
  const gradients: { r1: number; stops: [number, string][] }[] = [];
  const ops: string[] = []; // (the ways it was told to lay what it draws over the picture)
  return {
    strokes: 0, fills: 0, strokeStyle: "", fillStyle: "" as unknown, lineWidth: 1, filter: "",
    lineJoin: "", lineCap: "", gradients, ops, arcs: [] as number[][],
    get globalCompositeOperation() { return ops[ops.length - 1] ?? ""; },
    set globalCompositeOperation(v: string) { ops.push(v); },
    save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {}, arc(...args: number[]) { this.arcs.push(args); }, clip() {}, translate() {}, rotate() {},
    scale() {},
    stroke() { this.strokes++; }, fill() { this.fills++; }, fillRect() { this.fills++; },
    createRadialGradient: (_x0: number, _y0: number, _r0: number, _x1: number, _y1: number, r1: number) => {
      const g = { r1, stops: [] as [number, string][], addColorStop(at: number, c: string) { g.stops.push([at, c]); } };
      gradients.push(g);
      return g;
    },
  };
}

/** A right hand held up, the palm to the camera, its 21 landmarks (px) for a hand ``s`` px across (wrist to middle
 * knuckle), the middle knuckle at (x, y): the fingers spread up from it, the thumb out to the left. */
function handUp(x: number, y: number, s: number): P2[] {
  const lm: P2[] = Array.from({ length: 21 }, () => [x, y]);
  lm[0] = [x, y + s];
  [[1, -0.35, 0.75], [2, -0.55, 0.55], [3, -0.72, 0.38], [4, -0.86, 0.22]].forEach(([i, dx, dy]) => (lm[i] = [x + dx * s, y + dy * s]));
  [-0.42, -0.12, 0.16, 0.42].forEach((dx, f) => {
    const b = 5 + 4 * f, lean = dx * 0.5;
    [0.05, -0.35, -0.6, -0.82].forEach((dy, j) => (lm[b + j] = [x + (dx + lean * j * 0.3) * s, y + dy * s]));
  });
  return lm;
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

