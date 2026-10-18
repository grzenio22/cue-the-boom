import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import EffectBadge, { type Detected, reading } from "./EffectBadge";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); });

const FOUR = ["fire_breath", "fire_burst", "smoke_puff", "blow_out"];
const flamethrower: Detected = { effect: "fire_breath", kind: "hiss", gesture: "Victory", confidence: 0.82, mismatch: false };

function show(detected: Detected | null, choices = FOUR, onPick = vi.fn<(effect: string) => void>()) {
  act(() => root.render(<EffectBadge detected={detected} choices={choices} onPick={onPick} />));
  return { onPick, badge: () => host.querySelector<HTMLButtonElement>("button")! };
}

const tap = (el: Element) => act(() => { (el as HTMLElement).click(); });
const radio = (name: string) => [...host.querySelectorAll("[role=radio]")].find((b) => b.textContent === name)!;

describe("reading", () => {
  it("says the cue and the gesture in the performer's words", () => {
    expect(reading(flamethrower)).toBe("hiss + two-finger");
    expect(reading({ ...flamethrower, kind: "boom", gesture: "Closed_Fist" })).toBe("boom + fist");
    expect(reading({ ...flamethrower, kind: "pop", gesture: "None" })).toBe("pop");
    expect(reading({ ...flamethrower, kind: "snap", gesture: null })).toBe("snap");
  });
});

describe("EffectBadge", () => {
  it("says what it detected and from what", () => {
    const { badge } = show(flamethrower);
    expect(badge().textContent).toBe("Detected: Fire breath (hiss + two-finger)tap to change");
    expect(badge().title).toBe("82 % sure, from the hiss + two-finger");
  });

  it("calls an unsure pick a guess", () => {
    const { badge } = show({ ...flamethrower, confidence: 0.3 });
    expect(badge().textContent).toContain("(hiss + two-finger, a guess)");
  });

  it("cycles the four effects on each tap and comes back round to the detected one", () => {
    const { badge, onPick } = show(flamethrower);
    for (let i = 0; i < 3; i++) tap(badge());
    expect(onPick.mock.calls.map((c) => c[0])).toEqual(["fire_burst", "smoke_puff", "blow_out"]);
    expect(badge().textContent).toContain("Your pick: Blow it out");
    expect(host.textContent).toContain("detected Fire breath");
    tap(badge());
    expect(onPick).toHaveBeenLastCalledWith("fire_breath");
    expect(badge().textContent).toContain("Detected: Fire breath");
  });

  it("takes steam into the cycle only when it is a choice", () => {
    const { badge, onPick } = show({ ...flamethrower, effect: "blow_out", kind: "snap", gesture: null }, [...FOUR, "steam_jet"]);
    tap(badge());
    expect(onPick).toHaveBeenLastCalledWith("steam_jet");
    expect(badge().textContent).toContain("Your pick: Steam");
    tap(badge());
    expect(onPick).toHaveBeenLastCalledWith("fire_breath");
  });

  it("reaches any effect in two taps through the list", () => {
    const { onPick } = show(flamethrower);
    const all = host.querySelector('[aria-label="All effects"]')!;
    tap(all);
    expect(all.getAttribute("aria-expanded")).toBe("true");
    expect(radio("Fire breath").getAttribute("aria-checked")).toBe("true");
    tap(radio("Blow it out"));
    expect(onPick.mock.calls).toEqual([["blow_out"]]);
    expect(host.querySelector("[role=radiogroup]")).toBeNull(); // closed again
  });

  it("turns yellow and says why on a gesture mismatch", () => {
    const { badge } = show({ ...flamethrower, kind: "boom", gesture: "Open_Palm", effect: "smoke_puff", mismatch: true });
    expect(badge().style.boxShadow).toContain("var(--warn)");
    expect(badge().style.color).toBe("var(--warn)");
    expect(host.querySelector("[role=status]")!.textContent).toBe("your hand moved off the beat: it fires on your sound");
  });

  it("is plain when the gesture and the cue agree", () => {
    const { badge } = show(flamethrower);
    expect(badge().style.boxShadow).toBe("");
    expect(host.querySelector("[role=status]")).toBeNull();
  });

  it("asks for a pick when nothing was detected", () => {
    const { badge, onPick } = show(null);
    expect(badge().textContent).toBe("Pick an effecttap to change");
    tap(badge());
    expect(onPick).toHaveBeenLastCalledWith("fire_breath");
    expect(badge().textContent).toContain("Your pick: Fire breath");
  });

  it("starts again from a new take's own reading", () => {
    const { badge, onPick } = show(flamethrower);
    tap(badge());
    expect(badge().textContent).toContain("Your pick: Fire burst");
    show({ ...flamethrower, effect: "smoke_puff", kind: "puff", gesture: "Open_Palm" }, FOUR, onPick);
    expect(badge().textContent).toContain("Detected: Smoke puff (puff + open palm)");
  });
});
