import { describe, expect, it } from "vitest";
import { dragDirection, hitAnchor, type Receipt, suggestEffect } from "./shot";

const SR = 48000;

/** 0.5 s of silence, then `dur` s of either a voiced tone (low zero-crossing rate) or noise (high). */
function noise(dur: number, kind: "voiced" | "noisy"): Float32Array {
  const pcm = new Float32Array(Math.round((dur + 1) * SR));
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
  for (let i = 0; i < dur * SR; i++) {
    const t = i / SR;
    pcm[Math.round(0.5 * SR) + i] = 0.5 * (kind === "voiced" ? Math.sin(2 * Math.PI * 140 * t) : rnd());
  }
  return pcm;
}

describe("suggestEffect", () => {
  it("hears a short hit as a fireball", () => expect(suggestEffect(noise(0.15, "noisy"), SR, 0.5)).toBe("fireball"));
  it("hears a breathy pfffff as a smoke puff", () => expect(suggestEffect(noise(0.5, "noisy"), SR, 0.5)).toBe("smoke_puff"));
  it("hears a long hiss as steam", () => expect(suggestEffect(noise(1.2, "noisy"), SR, 0.5)).toBe("steam_jet"));
  it("hears a long voiced fwoooom as a flame plume", () => expect(suggestEffect(noise(1.2, "voiced"), SR, 0.5)).toBe("flame_plume"));
  it("falls back to a fireball without an onset", () => expect(suggestEffect(new Float32Array(10), SR, null)).toBe("fireball"));
});

describe("dragDirection", () => {
  it("is clockwise from straight up", () => {
    expect(dragDirection(0, -10)).toBe(0);
    expect(dragDirection(10, 0)).toBe(90);
    expect(dragDirection(-10, 0)).toBe(-90);
    expect(Math.abs(dragDirection(0, 10))).toBe(180);
  });
  it("rounds to 5 degrees", () => expect(dragDirection(10, -9)).toBe(50));
});

describe("hitAnchor", () => {
  const guide = { length: 96, tail: "fade", keyframes: [0, 1, 9], anchor_frame: 7 };
  it("puts a scene shot's handle on its hit, not on the render's first frame", () =>
    expect(hitAnchor({ mode: "scene", guide, alignment: { hit_frame: 24 } } as Receipt)).toBe(24));
  it("keeps an element shot's guide anchor", () => expect(hitAnchor({ guide } as Receipt)).toBe(7));
  it("has nothing to anchor without a receipt", () => expect(hitAnchor(null)).toBe(0));
});
