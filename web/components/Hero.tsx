"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import Seal, { ELEMENT_COLOR, type Element } from "@/components/Seal";

/** The reel's beats (public/hero/reel.mp4: three of the sample shots made ahead of time, cut to their hits). */
const BEATS: { until: number; element: Element; word: string; caption: string }[] = [
  { until: 1.9, element: "fire", word: "BOOM", caption: "Fire · one stored LTX-2.5 render, cued by a voice" },
  { until: 4.1, element: "air", word: "hoooo", caption: "Air · the stored LTX-2.5 whirlwind, held by a breath" },
  { until: Infinity, element: "explosion", word: "BOOOOM", caption: "Explosion · the stored fireball, thrown off a fist" },
];
const ELEMENTS: { element: Element; name: string; cue: string }[] = [
  { element: "fire", name: "Fire", cue: "BOOM" },
  { element: "water", name: "Water", cue: "shhh" },
  { element: "air", name: "Air", cue: "hoooo" },
  { element: "explosion", name: "Explosion", cue: "BOOOOM" },
  { element: "lightning", name: "Lightning", cue: "zzzt" },
];
const BRUSH = "M18 128 C110 70 290 46 512 72 C700 94 858 58 984 88 C990 104 988 120 978 136 C846 118 704 150 516 142 C318 132 160 164 34 172 C22 160 16 144 18 128 Z";

/** The first screen: the name, the promise, the five elements, and real shots made with it, looping. */
export default function Hero() {
  const [beat, setBeat] = useState(0);
  const reel = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    // a viewer who asks for less motion gets the first frame and the controls instead of a loop
    const v = reel.current;
    if (v && window.matchMedia("(prefers-reduced-motion: reduce)").matches) { v.pause(); v.controls = true; }
  }, []);
  const b = BEATS[beat];

  return (
    <header className="hero" style={{ "--el": ELEMENT_COLOR[b.element] } as CSSProperties}>
      <div className="hero-inner">
        <div className="hero-copy">
          <p className="kicker">Open Source Higgsfield · open weights · LTX-2.5 + ComfyUI</p>
          <h1 className="hero-title">Cue the <span className="boom">Boom</span></h1>
          <p className="hero-line">Render it once. <span>Perform it forever.</span></p>
          <p className="lede"><strong>No regeneration:</strong> your voice plays a stored LTX-2.5 render on your own real footage.
            Fire, water, air, explosion and lightning.</p>
          <ul className="seal-row" aria-label="The elements">
            {ELEMENTS.map((e) => (
              <li key={e.element} style={{ "--c": ELEMENT_COLOR[e.element] } as CSSProperties}>
                <Seal element={e.element} size={48} />
                <span className="seal-name">{e.name}</span>
                <span className="seal-cue" aria-hidden="true">{e.cue}</span>
              </li>
            ))}
          </ul>
          <div className="row hero-actions">
            <a className="btn hero-cta" href="#studio">Perform it with your voice</a>
            <a className="btn secondary" href="https://github.com/grzenio22/cue-the-boom" target="_blank" rel="noreferrer">Open source on GitHub</a>
          </div>
        </div>
        <div className="hero-stage">
          <div className="stage-glow" aria-hidden="true" />
          <Seal key={`ring-${b.element}`} element={b.element} size={520} ring={1.4} stroke={2.2} className="stage-seal" />
          <figure className="reel">
            <video ref={reel} src="/hero/reel.mp4" poster="/hero/reel.jpg" autoPlay muted loop playsInline preload="auto"
              aria-label="Shots made with it: fire, air and an explosion on a real hand, each cued by a voice"
              onTimeUpdate={(e) => {
                const t = e.currentTarget.currentTime;
                const i = BEATS.findIndex((x) => t < x.until);
                if (i !== beat) setBeat(i);
              }} />
            <div className="brush" aria-hidden="true">
              <svg viewBox="0 0 1000 200" preserveAspectRatio="none"><path d={BRUSH} /></svg>
              <span key={`word-${beat}`} className="brush-word">{b.word}</span>
            </div>
            <figcaption key={`cap-${beat}`}>{b.caption}</figcaption>
          </figure>
        </div>
      </div>
      <ol className="hero-steps">
        <li><strong>Render once</strong><span>LTX-2.5 and the ainvfx-fluid IC-LoRA in ComfyUI. Stored, never regenerated.</span></li>
        <li><strong>Make the noise</strong><span>BOOM, shhh or hoooo: your voice picks when it hits and how big it plays.</span></li>
        <li><strong>Get the shot</strong><span>Your hand tracked, matted and lit by the element on a CPU. 0 new renders.</span></li>
      </ol>
    </header>
  );
}
