"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { JobStatus } from "@/lib/shot";

const CUES = ["BOOM", "pf, pf, BOOM", "pf pf pf PF"];

/** Three performances of the same render, with one clock for their muted comparison. */
export default function Again() {
  const [shots, setShots] = useState<JobStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [ready, setReady] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const [listening, setListening] = useState<number | null>(null);
  const [buffering, setBuffering] = useState(false);
  const [mediaError, setMediaError] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const players = useRef<(HTMLVideoElement | null)[]>([]);
  const wantedPlaying = useRef(false);
  const listenIndex = useRef<number | null>(null);
  const waiting = useRef(false);
  const generation = useRef(0);
  const autoStarted = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    fetch("/api/gallery", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : [])
      .then((payload: unknown) => {
        if (!active || !Array.isArray(payload)) return;
        const ordered = [0, 1, 2].map((index) => payload.find((shot: JobStatus | null) =>
          shot && typeof shot.again === "number" && shot.again === index && shot.files?.["final.mp4"]));
        if (ordered.every((shot): shot is JobStatus => !!shot)) setShots(ordered);
      })
      .catch(() => { /* The unavailable state keeps the rest of the studio usable. */ })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; controller.abort(); };
  }, []);

  const pauseMedia = useCallback(() => {
    players.current.forEach((video) => video?.pause());
  }, []);

  const pause = useCallback(() => {
    generation.current += 1;
    wantedPlaying.current = false;
    waiting.current = false;
    pauseMedia();
    setPlaying(false);
    setBuffering(false);
  }, [pauseMedia]);

  const begin = useCallback((index: number | null, restart = false) => {
    const media = players.current.filter((video): video is HTMLVideoElement => video !== null);
    if (media.length !== 3) return;
    autoStarted.current = true;
    const token = ++generation.current;
    wantedPlaying.current = true;
    waiting.current = false;
    listenIndex.current = index;
    pauseMedia();
    const time = restart ? 0 : media[index ?? 0].currentTime;
    media.forEach((video, i) => {
      video.muted = index === null || index !== i;
      if (index === null || i === index) video.currentTime = time;
    });
    setListening(index);
    setPlaying(true);
    setBuffering(true);
    setMessage(null);
    // Call play in the button's gesture, including the first audible performance.
    const targets = index === null ? media : [media[index]];
    void Promise.all(targets.map((video) => video.play())).then(() => {
      if (generation.current === token) setBuffering(false);
    }).catch(() => {
      if (generation.current !== token) return;
      generation.current += 1;
      wantedPlaying.current = false;
      pauseMedia();
      setPlaying(false);
      setBuffering(false);
      setMessage(index === null ? "Playback paused. Press Play to start the three performances."
        : `Playback paused. Press Play to hear performance ${index + 1}.`);
    });
  }, [pauseMedia]);

  const stopListening = useCallback(() => {
    pause();
    listenIndex.current = null;
    setListening(null);
    setMessage(null);
    players.current.forEach((video) => {
      if (video) { video.muted = true; video.currentTime = 0; }
    });
  }, [pause]);

  const allReady = ready.length === 3;
  useEffect(() => {
    if (!allReady || mediaError) return;
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!autoStarted.current) {
      autoStarted.current = true;
      if (!preference.matches) begin(null, true);
    }
    const change = (event: MediaQueryListEvent) => { if (event.matches) pause(); };
    preference.addEventListener("change", change);
    return () => preference.removeEventListener("change", change);
  }, [allReady, mediaError, begin, pause]);

  useEffect(() => {
    if (!playing || listening !== null || buffering) return;
    let frame = 0;
    const sync = () => {
      const leader = players.current[0];
      if (leader) players.current.slice(1).forEach((video) => {
        if (video && Math.abs(video.currentTime - leader.currentTime) > 0.1) video.currentTime = leader.currentTime;
      });
      frame = requestAnimationFrame(sync);
    };
    frame = requestAnimationFrame(sync);
    return () => cancelAnimationFrame(frame);
  }, [playing, listening, buffering]);

  useEffect(() => {
    const media = players.current.filter((video): video is HTMLVideoElement => video !== null);
    return () => {
      generation.current += 1;
      wantedPlaying.current = false;
      media.forEach((video) => { video.pause(); video.muted = true; });
    };
  }, [shots]);

  function canPlay(index: number) {
    setReady((current) => current.includes(index) ? current : [...current, index]);
    if (!waiting.current || !wantedPlaying.current) return;
    const required = listenIndex.current === null ? players.current : [players.current[listenIndex.current]];
    if (required.every((video) => video && video.readyState >= 3)) begin(listenIndex.current);
  }

  function waitForMedia(index: number) {
    if (!wantedPlaying.current || (listenIndex.current !== null && listenIndex.current !== index)) return;
    generation.current += 1;
    waiting.current = true;
    pauseMedia();
    setBuffering(true);
  }

  function ended(index: number) {
    if (!wantedPlaying.current) return;
    if (listenIndex.current === null) begin(null, true);
    else if (listenIndex.current === index) {
      if (index < 2) begin(index + 1, true);
      else {
        stopListening();
        setMessage("All three performances heard. Press Play to compare them together again.");
      }
    }
  }

  const status = message ?? (buffering || (!allReady && !playing) ? "Waiting for the performances to load…"
    : listening !== null ? `${playing ? "Hearing" : "Paused at"} performance ${listening + 1} of 3.`
    : playing ? "Playing together, muted." : "Paused. Play together, or hear each performance in turn.");

  return (
    <section className="again" aria-labelledby="again-heading">
      <h2 id="again-heading">One render. Three performances.</h2>
      <p>The same LTX-2.5 render on the author&apos;s hand, timed three times by three takes of their voice: one generation, three performances, no new render, no GPU.</p>
      <p className="again-fact"><strong>1 generation, 3 performances, 0 new renders</strong></p>
      {loading ? <p className="muted" role="status">Loading performances…</p>
        : shots.length !== 3 ? <p className="muted" role="status">The three performances are unavailable right now. Please try again later.</p>
        : <>
          <div className="again-row">
            {shots.map((shot, index) => (
              <figure key={shot.id} className={listening === index ? "again-current" : undefined}>
                <video ref={(video) => { players.current[index] = video; }} src={shot.files["final.mp4"]}
                  poster={shot.files["poster.jpg"]} muted={listening !== index} playsInline preload="auto"
                  aria-label={`Performance ${index + 1}: ${CUES[index]}`}
                  onCanPlay={() => canPlay(index)} onWaiting={() => waitForMedia(index)} onEnded={() => ended(index)}
                  onError={() => { pause(); setMediaError(true); setMessage("A performance could not load. Reload this page to try again."); }} />
                <figcaption>{CUES[index]}</figcaption>
              </figure>
            ))}
          </div>
          <div className="row">
            <button type="button" disabled={mediaError}
              aria-label={playing ? "Pause performances" : listening !== null ? "Play: continue listening" : "Play performances together"}
              onClick={() => playing ? pause() : begin(listenIndex.current)}>{playing ? "Pause" : "Play"}</button>
            <button type="button" className="secondary" disabled={mediaError}
              onClick={() => listening === null ? begin(0, true) : stopListening()}>
              {listening === null ? "Hear them one by one" : "Stop listening"}
            </button>
          </div>
          <p className="muted again-status" role="status">{status}</p>
        </>}
      <p className="muted"><a href="#studio">Perform it again yourself</a>: below, pick <strong>Your hand: open palm</strong>, then <strong>Use my microphone</strong>.</p>
    </section>
  );
}
