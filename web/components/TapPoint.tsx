"use client";

import { useEffect, useRef, useState } from "react";
import { contentRect, normalizedPoint } from "@/lib/geometry";
import { dragDirection } from "@/lib/shot";

interface Props {
  src: string;
  value: [number, number] | null;
  onChange: (p: [number, number]) => void;
  direction?: number | null; // degrees clockwise from up; drawn as an arrow from the point
  onDirection?: (deg: number) => void; // set by dragging away from the point
  more?: [number, number][]; // call and response: the sources of the later hits, in order
  adding?: boolean; // a tap adds the next hit's source instead of moving the first
  onMore?: (points: [number, number][]) => void;
}

export const MAX_SOURCES = 4;

const DRAG_PX = 14;

interface Dims { boxW: number; boxH: number; videoW: number; videoH: number }

/** Paused first frame of the plate; tap where the effect should come from, drag to aim it. With ``adding``, each
 * tap is the source of the next hit of a rhythm (numbered, joined in order). */
export default function TapPoint({ src, value, onChange, direction, onDirection, more = [], adding = false, onMore }: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState<Dims | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  // Track the rendered box size so the marker stays put when the layout changes.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const ro = new ResizeObserver(([entry]) => {
      setDims((d) => (d ? { ...d, boxW: entry.contentRect.width, boxH: entry.contentRect.height } : d));
    });
    ro.observe(box);
    return () => ro.disconnect();
  }, []);

  const onLoaded = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    v.currentTime = 0;
    const box = boxRef.current?.getBoundingClientRect();
    setDims({ boxW: box?.width ?? 0, boxH: box?.height ?? 0, videoW: v.videoWidth, videoH: v.videoHeight });
  };

  const onDown = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (!dims || !dims.videoW) return;
    const box = ev.currentTarget.getBoundingClientRect();
    const p = normalizedPoint(ev.clientX - box.left, ev.clientY - box.top, { width: box.width, height: box.height }, dims.videoW, dims.videoH);
    if (!p) return;
    if (adding && value && onMore) {
      if (more.length < MAX_SOURCES - 1) onMore([...more, p]);
      return;
    }
    onChange(p);
    drag.current = { x: ev.clientX, y: ev.clientY };
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };

  const onMove = (ev: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || !onDirection) return;
    const dx = ev.clientX - d.x, dy = ev.clientY - d.y;
    if (Math.hypot(dx, dy) >= DRAG_PX) onDirection(dragDirection(dx, dy));
  };

  const onUp = () => { drag.current = null; };

  let marker: React.CSSProperties | null = null;
  let at: ((p: [number, number]) => { left: number; top: number }) | null = null;
  if (value && dims && dims.videoW && dims.boxW) {
    const r = contentRect({ width: dims.boxW, height: dims.boxH }, dims.videoW, dims.videoH);
    at = (p) => ({ left: r.x + p[0] * r.width, top: r.y + p[1] * r.height });
    marker = at(value);
  }
  const chain = value && at && more.length ? [value, ...more].map(at) : [];

  return (
    <div ref={boxRef} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
      style={{ position: "relative", width: "100%", maxWidth: 360, aspectRatio: "9 / 16", background: "#000", cursor: "crosshair", touchAction: "none" }}>
      <video
        src={src}
        muted
        playsInline
        preload="auto"
        onLoadedData={onLoaded}
        style={{ width: "100%", height: "100%", objectFit: "contain", pointerEvents: "none" }}
      />
      {chain.length > 1 && (
        <svg aria-hidden style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
          <polyline points={chain.map((c) => `${c.left},${c.top}`).join(" ")} fill="none" stroke="#ffb020" strokeWidth={2} strokeDasharray="6 5" opacity={0.8} />
        </svg>
      )}
      {marker && (
        <div aria-label={more.length ? "source of hit 1" : "emission point"} style={{ position: "absolute", ...marker, width: 28, height: 28, marginLeft: -14, marginTop: -14,
          border: "2px solid #ffb020", borderRadius: "50%", boxShadow: "0 0 12px #ff8c00", pointerEvents: "none",
          display: "grid", placeItems: "center", color: "#ffb020", font: "700 13px/1 system-ui", textShadow: "0 0 3px #000" }}>
          {more.length ? 1 : null}
        </div>
      )}
      {chain.slice(1).map((c, i) => (
        <div key={i} aria-label={`source of hit ${i + 2}`} style={{ position: "absolute", ...c, width: 28, height: 28, marginLeft: -14, marginTop: -14,
          border: "2px solid #ffb020", borderRadius: "50%", boxShadow: "0 0 12px #ff8c00", pointerEvents: "none",
          display: "grid", placeItems: "center", color: "#ffb020", font: "700 13px/1 system-ui", textShadow: "0 0 3px #000" }}>
          {i + 2}
        </div>
      ))}
      {marker && direction != null && (
        <div aria-label={`aimed ${direction} degrees`} style={{ position: "absolute", ...marker, width: 72, height: 3, marginTop: -1.5,
          background: "linear-gradient(90deg, #ffb020, #ff8c0000)", transformOrigin: "0 50%",
          transform: `rotate(${direction - 90}deg)`, pointerEvents: "none" }} />
      )}
    </div>
  );
}
