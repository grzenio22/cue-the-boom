"use client";

import { useState } from "react";
import { EFFECT_LABELS } from "@/lib/shot";

/** What `--effect auto` made of a take: the effect, the cue's kind (boom, hiss, pop, ...), the hand's gesture, how
 * sure it is, and whether the hand moved off the cue (gesture_mismatch: the launch then stays on the sound). */
export interface Detected { effect: string; kind: string; gesture: string | null; confidence: number; mismatch: boolean }

interface Props {
  detected: Detected | null;
  choices: string[]; // what a tap goes through, in order: fire_breath, fire_burst, smoke_puff, blow_out (+ steam_jet)
  onPick: (effect: string) => void;
  names?: Record<string, string>; // the effects' names here (a hand's: Fireball, Flamethrower), over the library's
}

// MediaPipe's gestures and the tracker's emitters, in the performer's words
const GESTURES: Record<string, string> = {
  Victory: "two-finger", two_finger: "two-finger", Pointing_Up: "pointing", index_tip: "pointing",
  Open_Palm: "open palm", palm: "open palm", Closed_Fist: "fist", fist: "fist",
  Thumb_Up: "thumb up", Thumb_Down: "thumb down", ILoveYou: "rock-on hand",
};

const SURE = 0.5; // below this the badge says the pick is a guess

const libraryLabel = (id: string) => EFFECT_LABELS[id]?.name ?? id;

/** What the pick was read from: "hiss + two-finger", or the cue's kind alone when no gesture was seen. */
export function reading(d: Detected): string {
  const g = d.gesture && d.gesture !== "None" ? (GESTURES[d.gesture] ?? d.gesture.replace(/_/g, " ").toLowerCase()) : null;
  const kind = d.kind.replace(/_/g, " ");
  return g ? `${kind} + ${g}` : kind;
}

/** The auto pick, said out loud, with a one-tap override: a tap on the badge moves on to the next effect, the arrow
 * beside it lists them all (any effect in two taps). Yellow when the hand moved off the cue. Never silent: until
 * the performer picks, it says what was detected and why. */
export default function EffectBadge({ detected, choices, onPick, names }: Props) {
  const label = (id: string) => names?.[id] ?? libraryLabel(id);
  // the performer's own pick holds for the take it was made on; a new reading starts from its own
  const take = detected ? `${detected.effect}|${detected.kind}|${detected.gesture}` : "";
  const [own, setOwn] = useState<{ take: string; effect: string } | null>(null);
  const [open, setOpen] = useState(false);
  const current = (own?.take === take ? own.effect : null) ?? detected?.effect ?? null;
  const picked = current !== null && current !== detected?.effect;

  const choose = (effect: string) => { setOwn({ take, effect }); setOpen(false); onPick(effect); };
  const next = () => { if (choices.length) choose(choices[(choices.indexOf(current ?? "") + 1) % choices.length]); };
  const warn = !!detected?.mismatch;

  return (
    <>
      <div className="row">
        <button className="secondary" disabled={!choices.length} onClick={next}
          title={detected ? `${Math.round(detected.confidence * 100)} % sure, from the ${reading(detected)}` : undefined}
          style={warn ? { boxShadow: "inset 0 0 0 2px var(--warn)", color: "var(--warn)" } : undefined}>
          {detected && !picked
            ? <>Detected: {label(detected.effect)} <span className="muted">({reading(detected)}{detected.confidence < SURE ? ", a guess" : ""})</span></>
            : current === null ? "Pick an effect" : <>Your pick: {label(current)}</>}
          <span className="muted" style={{ marginLeft: 8 }}>tap to change</span>
        </button>
        <button className="secondary" aria-label="All effects" aria-expanded={open} disabled={!choices.length}
          onClick={() => setOpen(!open)}>{open ? "▴" : "▾"}</button>
        {picked && detected && <span className="muted">detected {label(detected.effect)}</span>}
        {warn && <span className="chip warn" role="status">your hand moved off the beat: it fires on your sound</span>}
      </div>
      {open && (
        <div className="row" role="radiogroup" aria-label="Effect">
          {choices.map((id) => (
            <button key={id} role="radio" aria-checked={id === current} className={id === current ? "" : "secondary"}
              onClick={() => choose(id)}>{label(id)}</button>
          ))}
        </div>
      )}
    </>
  );
}
