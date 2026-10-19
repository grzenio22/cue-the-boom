"use client";

import { useEffect, useRef, useState } from "react";
import { contentRect, type Rect } from "@/lib/geometry";

/** The parts of hand_track.json the aim draws (docs/bending_contracts.md): px in a W x H frame, one entry a frame. */
export interface HandTrack { xy: [number, number][]; dir: [number, number][]; src: string[]; W: number; H: number; scale?: number[] }

/** The track of where you tapped (x, y as fractions, on frame k), carried with the hand: kept as an offset in the
 * hand's own frame (along its direction and across it, in hand sizes), so it moves, turns and scales with the hand;
 * with ``aim`` (degrees clockwise from up, on frame k) its direction turned by the same angle on every frame, as the
 * shot carries it (cuesync.bending.anchored). */
export function carried(track: HandTrack, k: number, tap: [number, number], aim: number | null = null): HandTrack {
  const n = track.xy.length;
  if (!n) return track;
  const at = (t: number) => {
    const i = Math.min(Math.max(0, Math.round(t)), n - 1);
    const [x, y] = track.xy[i], [ux, uy] = track.dir[i] ?? [0, -1], s = track.scale?.[i] ?? 1;
    return { x, y, ux, uy, s: Math.max(s, 1e-6) };
  };
  const p = at(k), dx = tap[0] * track.W - p.x, dy = tap[1] * track.H - p.y;
  const a = (dx * p.ux + dy * p.uy) / p.s, b = (-dx * p.uy + dy * p.ux) / p.s; // along, across (y down)
  const turn = aim === null ? 0 : (wrap(aim - (Math.atan2(p.ux, -p.uy) * 180) / Math.PI) * Math.PI) / 180;
  const c = Math.cos(turn), sn = Math.sin(turn);
  return {
    ...track,
    xy: track.xy.map((_, t) => { const q = at(t); return [q.x + q.s * (a * q.ux - b * q.uy), q.y + q.s * (a * q.uy + b * q.ux)]; }),
    dir: track.xy.map((_, t) => { const q = at(t); return [q.ux * c - q.uy * sn, q.ux * sn + q.uy * c]; }), // clockwise (y down)
  };
}

/** The share of the clip's frames the tracker lost the hand on (0 to 1). */
export function lostShare(track: HandTrack): number {
  return track.src.length ? track.src.filter((s) => s === "lost").length / track.src.length : 0;
}

/** Where the element leaves the hand and which way it goes, on the hit frame: x and y as fractions of the frame,
 * angle in degrees clockwise from up (TapPoint's direction). */
export interface AimValue { frame: number; x: number; y: number; angle: number }

interface Props {
  src: string;
  frame: number; // the hit frame (24 fps): the clip shows it, paused
  track?: HandTrack;
  value: AimValue | null; // null: the tracked emitter and direction on the hit frame
  onChange: (v: AimValue) => void;
}

const FPS = 24; // a plate is 24 fps once load_plate has it
const GRAB_PX = 24; // a handle takes a press this close: a 48 px target for a finger
const MIN_PX = 12; // a drag nearer the seed than this says nothing about the way
const NUDGE = 0.01; // Shift+arrow moves the seed 1 % of the frame
const LOST = "#ff5d5d";

const r4 = (v: number) => Math.round(v * 10000) / 10000;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
/** Degrees into (-180, 180]: 0 up, 90 right, -90 left, 180 down. */
const wrap = (deg: number) => { const d = ((deg % 360) + 360) % 360; return d > 180 ? d - 360 : d; };

/** The way a drag from the seed points (dx right, dy down): the nearest of the 8 directions, or with `free` to the
 * degree. */
export function aimAngle(dx: number, dy: number, free = false): number {
  const deg = (Math.atan2(dx, -dy) * 180) / Math.PI;
  return wrap(free ? Math.round(deg) : Math.round(deg / 45) * 45);
}

/** An arrow key: on to the next of the 8 directions, clockwise (+1) or back (-1), from any angle. */
export function turnAngle(deg: number, turn: 1 | -1): number {
  const k = turn > 0 ? Math.floor(deg / 45 + 1e-6) + 1 : Math.ceil(deg / 45 - 1e-6) - 1;
  return wrap(k * 45);
}

/** The tracked emitter on a frame as an aim: its point over the frame size, its direction as an angle. */
export function trackedAim(track: HandTrack, frame: number): AimValue | null {
  if (!track.xy.length) return null;
  const i = Math.min(Math.max(0, Math.round(frame)), track.xy.length - 1);
  const [x, y] = track.xy[i], [ux, uy] = track.dir[i] ?? [0, -1];
  return { frame, x: r4(clamp01(x / track.W)), y: r4(clamp01(y / track.H)), angle: wrap(Math.round((Math.atan2(ux, -uy) * 180) / Math.PI)) };
}

type Kind = "seen" | "filled" | "lost";
const RANK: Record<Kind, number> = { seen: 0, filled: 1, lost: 2 };
// mp, rtm, flow and tapir saw the hand; interp and hold filled a gap; lost kept the last good point
const kindOf = (s: string | undefined): Kind => (s === "lost" ? "lost" : s === "interp" || s === "hold" ? "filled" : "seen");

/** Equal neighbours as runs, [from, to] inclusive. */
function runsOf(kinds: Kind[]): { kind: Kind; from: number; to: number }[] {
  const runs: { kind: Kind; from: number; to: number }[] = [];
  kinds.forEach((kind, i) => {
    const last = runs[runs.length - 1];
    if (last && last.kind === kind) last.to = i;
    else runs.push({ kind, from: i, to: i });
  });
  return runs;
}

/** The frames the tracker lost, as runs [first, last]. */
export function lostRuns(src: string[]): [number, number][] {
  return runsOf(src.map(kindOf)).filter((r) => r.kind === "lost").map((r) => [r.from, r.to]);
}

type At = (x: number, y: number) => { left: number; top: number };

/** The track across the whole clip, faint, so a failure shows before anything renders: solid where the tracker saw
 * the hand, dashed where it filled a gap, red where it lost it (a lost frame holds the last point, so the red is the
 * jump when it finds the hand again, and a ring where it held). Along the bottom, the clip's frames the same way. */
function Trail({ track, frame, at, r }: { track: HandTrack; frame: number; at: At; r: Rect }) {
  const T = track.xy.length;
  const kinds = track.xy.map((_, i) => kindOf(track.src[i]));
  const pts = track.xy.map(([x, y]) => at(x / track.W, y / track.H));
  // a segment is as bad as its worse end
  const segs = runsOf(kinds.slice(1).map((k, i) => (RANK[k] > RANK[kinds[i]] ? k : kinds[i])));
  const strip = { y: r.y + r.height - 9, h: 5 };
  return (
    <g>
      {segs.map((s) => (
        <polyline key={s.from} points={pts.slice(s.from, s.to + 2).map((p) => `${p.left},${p.top}`).join(" ")} fill="none"
          stroke={s.kind === "lost" ? LOST : "#ffb020"} strokeWidth={s.kind === "lost" ? 2.5 : 1.5}
          strokeDasharray={s.kind === "filled" ? "3 4" : undefined} opacity={s.kind === "lost" ? 0.95 : 0.45} strokeLinejoin="round" />
      ))}
      {lostRuns(track.src).map(([a, b]) => (
        <circle key={a} cx={pts[a].left} cy={pts[a].top} r={8} fill={`${LOST}33`} stroke={LOST} strokeWidth={2}>
          <title>{`lost the hand on frames ${a}-${b}`}</title>
        </circle>
      ))}
      <rect x={r.x} y={strip.y} width={r.width} height={strip.h} fill="#00000099" />
      {runsOf(kinds).map((s) => (
        <rect key={s.from} x={r.x + (s.from / T) * r.width} y={strip.y} width={((s.to - s.from + 1) / T) * r.width} height={strip.h}
          fill={s.kind === "lost" ? LOST : "#ffb020"} opacity={s.kind === "lost" ? 1 : s.kind === "filled" ? 0.3 : 0.6} />
      ))}
      <rect x={r.x + ((frame + 0.5) / T) * r.width - 1} y={strip.y - 4} width={2} height={strip.h + 8} fill="#f4f1ea" />
    </g>
  );
}

/** The aim as TapPoint draws it (a cone from the seed, an arrow down its middle), with a ring to drag the seed by
 * and a grip on the head; while aiming, the 8 directions it snaps to. */
function Arrow({ base, deg, L, ticks }: { base: { left: number; top: number }; deg: number; L: number; ticks: boolean }) {
  const spread = 0.2; // radians either side
  const a = ((deg - 90) * Math.PI) / 180; // clockwise from up -> screen angle
  const pt = (r: number, t: number, from = a) => [base.left + r * Math.cos(from + t), base.top + r * Math.sin(from + t)];
  const [tx, ty] = pt(L, 0);
  const ux = Math.cos(a), uy = Math.sin(a);
  const head = [[tx, ty], [tx - 14 * ux - 7 * uy, ty - 14 * uy + 7 * ux], [tx - 14 * ux + 7 * uy, ty - 14 * uy - 7 * ux]];
  return (
    <g>
      {ticks && [0, 45, 90, 135, 180, 225, 270, 315].map((d) => {
        const [x, y] = pt(L, 0, ((d - 90) * Math.PI) / 180);
        return <circle key={d} cx={x} cy={y} r={3} fill={wrap(d) === wrap(deg) ? "#ffb020" : "#f4f1ea80"} />;
      })}
      <polygon points={[[base.left, base.top], pt(L, -spread), pt(L, spread)].map((q) => q.join(",")).join(" ")}
        fill="#ffb02030" stroke="#ffb020" strokeWidth={1.5} strokeDasharray="5 4" />
      <line x1={base.left} y1={base.top} x2={tx - 10 * ux} y2={ty - 10 * uy} stroke="#ffb020" strokeWidth={2.5} />
      <circle cx={tx - 6 * ux} cy={ty - 6 * uy} r={15} fill="#ffb02026" stroke="#ffb02080" strokeWidth={1} />
      <polygon points={head.map((q) => q.join(",")).join(" ")} fill="#ffb020" />
      <circle cx={base.left} cy={base.top} r={13} fill="#0000004d" stroke="#ffb020" strokeWidth={2} style={{ filter: "drop-shadow(0 0 6px #ff8c00)" }} />
    </g>
  );
}

const WAYS = ["up", "up and right", "right", "down and right", "down", "down and left", "left", "up and left"];
const way = (deg: number) => {
  const near = WAYS[((Math.round(deg / 45) % 8) + 8) % 8];
  return deg % 45 === 0 ? near : `about ${near} (${deg}°)`;
};

const KEYS: Record<string, { turn: 1 | -1; dx: number; dy: number }> = {
  ArrowRight: { turn: 1, dx: 1, dy: 0 }, ArrowUp: { turn: 1, dx: 0, dy: -1 },
  ArrowLeft: { turn: -1, dx: -1, dy: 0 }, ArrowDown: { turn: -1, dx: 0, dy: 1 },
};

interface Drag { mode: "head" | "base"; from: AimValue; x: number; y: number; last: AimValue }

/** The hit frame of a bending clip, paused, with the aim on it: an arrow from where the element leaves the hand
 * (the tracked emitter, until the performer moves it) along the way it goes. Drag the head, or tap anywhere, to aim
 * it (8 directions; Alt for any angle); drag the ring to move where it starts. Keys: arrows turn it 45 degrees,
 * Shift+arrows move it, Home puts it back on the track. */
export default function AimOverlay({ src, frame, track, value, onChange }: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [box, setBox] = useState<{ w: number; h: number } | null>(null);
  const [video, setVideo] = useState<{ w: number; h: number } | null>(null);
  const [ticks, setTicks] = useState(false); // the 8 directions, shown while aiming
  const drag = useRef<Drag | null>(null);

  // Track the rendered box size so the arrow stays on the hand when the layout changes.
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setBox({ w: entry.contentRect.width, h: entry.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // A new hit frame: show it (the middle of the frame, so no browser rounds onto the one before).
  useEffect(() => {
    const v = videoRef.current;
    if (v && v.readyState >= 1) v.currentTime = (frame + 0.5) / FPS;
  }, [frame]);

  const onLoaded = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    v.currentTime = (frame + 0.5) / FPS;
    setVideo({ w: v.videoWidth, h: v.videoHeight });
  };

  // the clip's own size once it loads; until then the track's (the same frame)
  const clip = video && video.w ? video : track ? { w: track.W, h: track.H } : null;
  const r = clip && box && box.w ? contentRect({ width: box.w, height: box.h }, clip.w, clip.h) : null;
  const at: At = (x, y) => (r ? { left: r.x + x * r.width, top: r.y + y * r.height } : { left: 0, top: 0 });
  const aim = value ?? (track ? trackedAim(track, frame) : null);
  const L = r ? Math.min(110, Math.max(56, 0.3 * Math.min(r.width, r.height))) : 0; // px, the arrow's length

  const move = (px: number, py: number, free: boolean) => {
    const d = drag.current;
    if (!d || !r) return;
    let next: AimValue;
    if (d.mode === "base") {
      next = { ...d.from, frame, x: r4(clamp01(d.from.x + (px - d.x) / r.width)), y: r4(clamp01(d.from.y + (py - d.y) / r.height)) };
    } else {
      const s = at(d.from.x, d.from.y), dx = px - s.left, dy = py - s.top;
      if (Math.hypot(dx, dy) < MIN_PX) return;
      next = { ...d.from, frame, angle: aimAngle(dx, dy, free) };
    }
    if (next.x === d.last.x && next.y === d.last.y && next.angle === d.last.angle) return;
    d.last = next;
    onChange(next);
  };

  const onDown = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (!r) return;
    const b = ev.currentTarget.getBoundingClientRect();
    const px = ev.clientX - b.left, py = ev.clientY - b.top;
    let from = aim, mode: Drag["mode"] = "head", onHead = false;
    if (from) {
      const s = at(from.x, from.y), a = (from.angle * Math.PI) / 180;
      onHead = Math.hypot(px - s.left - L * Math.sin(a), py - s.top + L * Math.cos(a)) <= GRAB_PX;
      if (!onHead && Math.hypot(px - s.left, py - s.top) <= GRAB_PX) mode = "base";
    } else {
      // nothing to aim from yet: the press is the seed and a drag from it the way, as on TapPoint
      const x = (px - r.x) / r.width, y = (py - r.y) / r.height;
      if (x < 0 || x > 1 || y < 0 || y > 1) return;
      from = { frame, x: r4(x), y: r4(y), angle: 0 };
      onHead = true;
      onChange(from);
    }
    drag.current = { mode, from, x: px, y: py, last: from };
    ev.currentTarget.setPointerCapture(ev.pointerId);
    setTicks(mode === "head");
    if (!onHead && mode === "head") move(px, py, ev.altKey); // a tap away from the arrow aims it there
  };

  const onMove = (ev: React.PointerEvent<HTMLDivElement>) => {
    const b = ev.currentTarget.getBoundingClientRect();
    move(ev.clientX - b.left, ev.clientY - b.top, ev.altKey);
  };

  const onUp = () => { drag.current = null; setTicks(false); };

  const onKey = (ev: React.KeyboardEvent<HTMLDivElement>) => {
    if (!aim) return;
    if (ev.key === "Home" && track) {
      const t = trackedAim(track, frame);
      if (t) { ev.preventDefault(); onChange(t); }
      return;
    }
    const k = KEYS[ev.key];
    if (!k) return;
    ev.preventDefault();
    if (ev.shiftKey) onChange({ ...aim, frame, x: r4(clamp01(aim.x + k.dx * NUDGE)), y: r4(clamp01(aim.y + k.dy * NUDGE)) });
    else { onChange({ ...aim, frame, angle: turnAngle(aim.angle, k.turn) }); setTicks(true); }
  };

  const lost = track ? lostRuns(track.src) : [];
  const lostFrames = lost.reduce((n, [a, b]) => n + b - a + 1, 0);
  const hitLost = !!track && lost.some(([a, b]) => frame >= a && frame <= b);
  const landscape = !!clip && clip.w > clip.h;

  return (
    <div style={{ width: "100%", maxWidth: landscape ? 640 : 360 }}>
      <div ref={boxRef} role="slider" tabIndex={0} aria-label="Aim: where the effect shoots"
        aria-valuemin={-180} aria-valuemax={180} aria-valuenow={aim?.angle}
        aria-valuetext={aim ? `shoots ${way(aim.angle)}, from ${Math.round(aim.x * 100)}% across and ${Math.round(aim.y * 100)}% down`
          : "not aimed yet: tap where the hand is"}
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} onKeyDown={onKey} onBlur={() => setTicks(false)}
        style={{ position: "relative", width: "100%", aspectRatio: clip ? `${clip.w} / ${clip.h}` : "9 / 16", background: "#000",
          cursor: "crosshair", touchAction: "none", userSelect: "none" }}>
        <video ref={videoRef} src={src} muted playsInline preload="auto" onLoadedData={onLoaded}
          style={{ width: "100%", height: "100%", objectFit: "contain", pointerEvents: "none", display: "block" }} />
        {r && (
          <svg aria-hidden style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", overflow: "visible" }}>
            {track && track.xy.length > 1 && <Trail track={track} frame={frame} at={at} r={r} />}
            {aim && <Arrow base={at(aim.x, aim.y)} deg={aim.angle} L={L} ticks={ticks} />}
          </svg>
        )}
      </div>
      {lostFrames > 0 && (
        <p style={{ color: "var(--bad)", fontSize: "0.85rem", margin: "8px 0 0" }}>{hitLost ? "Lost the hand on the hit frame: drag the ring onto it."
          : `Lost the hand on ${lostFrames} frame${lostFrames === 1 ? "" : "s"} (red): there the effect holds where it last saw it.`}</p>
      )}
      <p className="muted" style={{ fontSize: "0.85rem", margin: "6px 0" }}>
        {aim ? `${value ? "Your aim" : "Aimed from the hand's track"}. Drag the arrowhead (or tap) where it should shoot, the ring to move where it starts; Alt for any angle.`
          : "Tap where the hand is, then drag the way it should shoot."}
      </p>
    </div>
  );
}
