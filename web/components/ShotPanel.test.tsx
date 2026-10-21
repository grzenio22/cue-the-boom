import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ShotPanel from "./ShotPanel";
import type { EffectInfo } from "@/lib/shot";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Props = ComponentProps<typeof ShotPanel>;
let host: HTMLDivElement, root: Root;
let sent: FormData[];

const effectInfo = (id: string, element = true, scene = false): EffectInfo => ({
  id, element, scene, fluid: "fire", behaviour: "burst", stops_with_voice: false,
});
const effects = [
  ...["fire_breath", "fire_burst", "flame_plume", "steam_jet"].map((id) => effectInfo(id)),
  ...["explosion", "lightning_arc", "lightning_ball", "water_orb", "water_whip", "air_ball", "air_blast"]
    .map((id) => effectInfo(id, false)),
  effectInfo("smoke_puff"), effectInfo("wick_flare", false, true),
];
const plate: Props["plate"] = {
  source: { sample: "hand_open.mp4" }, url: "/hand_open.mp4", title: "Your hand: open palm, lit room",
  point: [0.532, 0.707], direction: null, subject: null, sample: true, sound: false, hand: { frame: 30 },
};
const performance: Props["performance"] = {
  cue: "voice", voice: { sample: "shhhhh-sploosh.wav" }, timing: null, face: null, faceExt: "webm",
  suggested: "water_orb", onsetS: null, warnings: [], label: "shhhhh", previewUrl: "",
};

beforeEach(() => {
  sent = [];
  // Keep submitShot and its FormData real; stop at the network boundary before a worker is needed.
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (url !== "/api/jobs" || !(init?.body instanceof FormData)) throw new Error(`Unexpected request: ${url}`);
    sent.push(init.body);
    return { ok: false, status: 503, text: async () => "Test submission captured; no worker started" };
  }));
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

const show = async (overrides: Partial<Props> = {}) => {
  await act(async () => {
    root.render(<ShotPanel plate={plate} point={[0.532, 0.707]} direction={null} performance={performance}
      effects={effects} modes={["instant"]} strokes={[]} hand={{ frame: 30, prep: "prepared-hand" }}
      onDone={() => undefined} {...overrides} />);
  });
};
const autoPerformance = { ...performance, suggested: "fireball" };
const radios = () => [...host.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Effect"] [role="radio"]')];
const radio = (label: string) => {
  const found = radios().find((b) => b.textContent?.startsWith(label));
  expect(found, `The effect "${label}" should be visible`).toBeDefined();
  return found!;
};
const button = (label: string) => {
  const found = [...host.querySelectorAll<HTMLButtonElement>("button")].find((b) => b.textContent === label);
  expect(found, `The button "${label}" should be visible`).toBeDefined();
  return found!;
};
const tap = async (el: HTMLButtonElement) => { await act(async () => { el.click(); }); };
const offlineCopy = () => [...host.querySelectorAll("p")].find((p) => p.textContent?.startsWith("The GPU is offline:"))?.textContent;

describe("ShotPanel offline sample hand", () => {
  it("lets an automatic cue reveal the available made effects without offering rendered hand effects", async () => {
    await show({ performance: autoPerformance });
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("true");
    await tap(button("Choose it yourself"));

    for (const label of ["Explosion", "Lightning", "Ball lightning", "Water orb", "Water whip", "Air ball", "Air blast"]) {
      expect(radio(label).disabled).toBe(false);
    }
    expect(radios()).toHaveLength(8);
    expect(radios().some((b) => /^(Flamethrower|Fireball|Flame in the palm|Steam|Smoke puff|Flare-up)/.test(b.textContent ?? ""))).toBe(false);
  });

  it("offers only made effects advertised by this server", async () => {
    await show({ performance: autoPerformance, effects: [effectInfo("water_orb", false), effectInfo("fire_burst")] });
    await tap(button("Choose it yourself"));
    expect(radio("Water orb").disabled).toBe(false);
    expect(radios()).toHaveLength(2);
    expect(radios().some((b) => b.textContent?.startsWith("Air ball"))).toBe(false);
  });

  it("shows the suggested water selection and submits it as a hand shot", async () => {
    await show();
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("true");
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("false");
    await tap(button("Make the shot"));

    expect(sent).toHaveLength(1);
    expect(sent[0].get("effect")).toBe("water_orb");
    expect(sent[0].get("mode")).toBe("bending");
    expect(sent[0].get("plate_sample")).toBe("hand_open.mp4");
    expect(sent[0].get("frame")).toBe("30");
    expect(sent[0].get("reuse")).toBe("prepared-hand");
  });

  it("can submit the cue's choice and then reselect and submit water on the same take", async () => {
    await show();
    await tap(radio("Your cue decides"));
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("false");
    await tap(button("Make the shot"));
    await tap(radio("Water orb"));
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("true");
    expect(radio("Your cue decides").getAttribute("aria-checked")).toBe("false");
    await tap(button("Make the shot"));

    expect(sent.map((form) => form.get("effect"))).toEqual(["auto", "water_orb"]);
    expect(sent.map((form) => form.get("mode"))).toEqual(["bending", "bending"]);
  });

  it("submits a manually chosen air effect in place of the suggested water", async () => {
    await show();
    await tap(radio("Air ball"));
    expect(radio("Air ball").getAttribute("aria-checked")).toBe("true");
    expect(radio("Water orb").getAttribute("aria-checked")).toBe("false");
    expect(offlineCopy()).toContain("Air ball");
    expect(offlineCopy()).not.toContain("Water orb");
    await tap(button("Make the shot"));
    expect(sent.map((form) => form.get("effect"))).toEqual(["air_ball"]);
  });

  it.each([
    ["explosion", "Explosion"], ["lightning_arc", "Lightning"], ["lightning_ball", "Ball lightning"],
    ["water_orb", "Water orb"], ["water_whip", "Water whip"], ["air_ball", "Air ball"], ["air_blast", "Air blast"],
  ])("describes selected %s as made locally with no new render", async (id, label) => {
    await show({ performance: { ...performance, suggested: id } });
    expect(offlineCopy()).toContain(label);
    expect(offlineCopy()).toContain("made on this computer");
    expect(offlineCopy()).toContain(". No new render");
    expect(offlineCopy()).not.toContain("fireball");
  });

  it("explains both local effects and the fireball made ahead when the cue decides", async () => {
    await show();
    await tap(radio("Your cue decides"));
    expect(offlineCopy()).toContain("Your cue chooses");
    expect(offlineCopy()).toContain("made on this computer");
    expect(offlineCopy()).toContain("fireball, made ahead of time");
    expect(offlineCopy()).toContain("No new render");
  });
});

