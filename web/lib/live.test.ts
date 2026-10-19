import { describe, expect, it } from "vitest";
import { liveHand, onsetStep, type OnsetState, playLook } from "./live";

const still: OnsetState = { floor: -60, top: -20, prev: -60, since: null, lastHit: -1, peak: -90 };

function run(levels: number[], fps = 60) {
  let s = still;
  const hits: number[] = [];
  const holding: boolean[] = [];
  levels.forEach((db, i) => {
    const r = onsetStep(s, db, i / fps);
    s = r.state;
    if (r.hit) hits.push(i);
    holding.push(r.holding);
  });
  return { hits, holding };
}

describe("the live preview's microphone", () => {
  it("hears a BOOM as one hit, not every frame of it", () => {
    const { hits, holding } = run([...Array(30).fill(-60), ...Array(10).fill(-15), ...Array(30).fill(-60)]);
    expect(hits).toEqual([30]);
    expect(holding.some(Boolean)).toBe(false); // a sixth of a second: a burst, not a held flame
  });
  it("hears a held fffff as held from a third of a second on, and let go when it stops", () => {
    const { hits, holding } = run([...Array(30).fill(-60), ...Array(60).fill(-25), ...Array(20).fill(-60)]);
    expect(hits).toEqual([30]);
    expect(holding[30 + 18]).toBe(false);
    expect(holding[30 + 25]).toBe(true);
    expect(holding[95]).toBe(false);
  });
  it("hears a gently started ffff as held, with no burst", () => {
    const ramp = Array.from({ length: 30 }, (_, i) => -60 + (35 * i) / 29); // half a second up to -25 dB
    const { hits, holding } = run([...Array(30).fill(-60), ...ramp, ...Array(60).fill(-25), ...Array(20).fill(-60)]);
    expect(hits).toEqual([]);
    expect(holding.slice(60, 120).some(Boolean)).toBe(true);
    expect(holding[135]).toBe(false); // and let go
  });
  it("hears two pf's apart as two hits", () => {
    const pf = [...Array(5).fill(-30), ...Array(25).fill(-60)];
    expect(run([...Array(30).fill(-60), ...pf, ...pf]).hits).toEqual([30, 60]);
  });
});

describe("a live hit's look", () => {
  it("is as big as it is loud against the loudest, as the shot plays it", () => {
    expect(playLook(-20, -20)).toEqual({ gain: 1, size: 1 });
    const pf = playLook(-32, -20);
    expect(pf.size).toBeCloseTo(0.5, 1);
    expect(pf.gain).toBeGreaterThan(pf.size);
  });
});

describe("the hand, live", () => {
  it("finds the palm, the fingertip, the way the finger points and whether it points", () => {
    const lm = Array.from({ length: 21 }, () => ({ x: 100, y: 200 }));
    lm[0] = { x: 100, y: 300 }; // wrist
    [5, 9, 13, 17].forEach((i, k) => { lm[i] = { x: 80 + 15 * k, y: 220 }; });
    lm[7] = { x: 90, y: 160 }; lm[8] = { x: 90, y: 140 }; // the index straight up
    lm[12] = { x: 95, y: 230 }; lm[16] = { x: 110, y: 230 }; // the others curled
    const h = liveHand(lm);
    expect(h.pointing).toBe(true);
    expect(h.dir[1]).toBeLessThan(-0.9);
    expect(h.tip[1]).toBeLessThan(140);
    expect(h.scale).toBeCloseTo(Math.hypot(100 - 95, 300 - 220), 0);
  });
});
