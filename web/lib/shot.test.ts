import { describe, expect, it } from "vitest";
import { dragDirection, failure, hitAnchor, madeAhead, type Receipt, sameStrokes, sceneEffect, suggestEffect } from "./shot";

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

describe("madeAhead", () => {
  const candle = { sample: true, point: [0.53, 0.52] as [number, number], scene: { "pf.wav": "blow_out", hit: "blow_out" }, sound: true };
  const top = [{ points: [[0.36, 0.53], [0.5, 0.4]] as [number, number][], width: 0.09 }];
  const pumpkin = { ...candle, drawn: [{ title: "drawn", voice: "fwoooom.wav", effect: "flame_plume", strokes: top }] };
  it("plays a listed sample noise at the marked spot", () =>
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.53, 0.52], [])).toBe(true));
  it("plays the clip's own sound", () => expect(madeAhead(candle, "clip", null, [0.53, 0.52], [])).toBe(true));
  it("not once the spot moves, for a noise it doesn't list, a drawing, or your own clip", () => {
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.4, 0.52], [])).toBe(false);
    expect(madeAhead(candle, "voice", { sample: "hissss.wav" }, [0.53, 0.52], [])).toBe(false);
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.53, 0.52], top)).toBe(false);
    expect(madeAhead({ ...candle, sample: false }, "clip", null, [0.53, 0.52], [])).toBe(false);
  });
  it("not once it is aimed another way: the server keys an aimed shot by its direction", () => {
    const dino = { ...candle, direction: -90 };
    expect(madeAhead(dino, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], -90)).toBe(true);
    expect(madeAhead(dino, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], null)).toBe(true);
    expect(madeAhead(dino, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], 45)).toBe(false);
    expect(madeAhead(candle, "voice", { sample: "pf.wav" }, [0.53, 0.52], [], 45)).toBe(false);
  });
  it("plays the sample's own drawing with the noise it was made with", () => {
    expect(madeAhead(pumpkin, "voice", { sample: "fwoooom.wav" }, [0.36, 0.53], top)).toBe(true);
    const nudged = [{ points: [[0.362, 0.531], [0.5, 0.4]] as [number, number][], width: 0.09 }];
    expect(sameStrokes(top, nudged)).toBe(true);
    expect(madeAhead(pumpkin, "voice", { sample: "ka-boom.wav" }, [0.36, 0.53], top)).toBe(false);
    expect(madeAhead(pumpkin, "voice", { sample: "fwoooom.wav" }, [0.36, 0.53], [...top, ...top])).toBe(false);
  });
  it("says what a noise does to this plate", () => {
    expect(sceneEffect(candle.scene, { name: "pf.wav", url: "" })).toBe("blow_out");
    expect(sceneEffect(candle.scene, { name: "ka-boom.wav", url: "", gesture: "hit" })).toBe("blow_out");
    expect(sceneEffect(undefined, { name: "pf.wav", url: "" })).toBeUndefined();
  });
});

describe("failure", () => {
  it("shows the server's own words, not its JSON", async () => {
    expect((await failure(new Response(JSON.stringify({ detail: "the GPU is offline" }), { status: 503 }))).message).toBe("the GPU is offline");
    expect((await failure(new Response("Internal Server Error", { status: 500 }))).message).toBe("Internal Server Error");
    expect((await failure(new Response("", { status: 502 }))).message).toBe("request failed (502)");
  });
});
