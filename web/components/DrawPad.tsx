"use client";

import { useEffect, useRef, useState } from "react";
import { contentRect, normalizedPoint } from "@/lib/geometry";
import type { Stroke } from "@/lib/shot";

const BRUSHES: { label: string; width: number }[] = [
  { label: "thin", width: 0.03 },
  { label: "medium", width: 0.06 },
  { label: "wide", width: 0.12 },
];

interface Dims { boxW: number; boxH: number; videoW: number; videoH: number }

/**
 * Draw where the effect goes, on the plate's first frame: each stroke is a polyline in normalised video
 * coordinates plus a brush width (a share of the frame width). Where you start drawing is where it comes from.
 */
export default function DrawPad({ src, strokes, onChange }: {
  src: string;
  strokes: Stroke[];
  onChange: (s: Stroke[]) => void;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dims, setDims] = useState<Dims | null>(null);
  const [brush, setBrush] = useState(BRUSHES[1].width);
  const live = useRef<Stroke | null>(null);

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const ro = new ResizeObserver(([entry]) => {
      setDims((d) => (d ? { ...d, boxW: entry.contentRect.width, boxH: entry.contentRect.height } : d));
    });
    ro.observe(box);
    return () => ro.disconnect();
  }, []);

  // Redraw every stroke (and the one being drawn) whenever anything changes.
  const paint = () => {
    const c = canvasRef.current;
    if (!c || !dims || !dims.videoW) return;
    c.width = dims.boxW;
    c.height = dims.boxH;
    const g = c.getContext("2d");
    if (!g) return;
    g.clearRect(0, 0, c.width, c.height);
    const r = contentRect({ width: dims.boxW, height: dims.boxH }, dims.videoW, dims.videoH);
    const all = live.current ? [...strokes, live.current] : strokes;
    for (const s of all) {
      g.strokeStyle = "rgba(255, 140, 0, 0.62)";
      g.fillStyle = "rgba(255, 140, 0, 0.62)";
      g.lineCap = "round";
      g.lineJoin = "round";
      g.lineWidth = Math.max(2, s.width * r.width);
      const pts = s.points.map(([x, y]) => [r.x + x * r.width, r.y + y * r.height]);
      if (pts.length === 1) {
        g.beginPath();
        g.arc(pts[0][0], pts[0][1], g.lineWidth / 2, 0, Math.PI * 2);
        g.fill();
      } else {
        g.beginPath();
        g.moveTo(pts[0][0], pts[0][1]);
        for (const [x, y] of pts.slice(1)) g.lineTo(x, y);
        g.stroke();
      }
    }
    if (strokes.length > 0) { // where it comes from: the first stroke's first point
      const [x, y] = strokes[0].points[0];
      g.strokeStyle = "#fff";
      g.lineWidth = 2;
      g.beginPath();
      g.arc(r.x + x * r.width, r.y + y * r.height, 9, 0, Math.PI * 2);
      g.stroke();
    }
  };
  useEffect(paint);

  const onLoaded = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    v.currentTime = 0;
    const box = boxRef.current?.getBoundingClientRect();
    setDims({ boxW: box?.width ?? 0, boxH: box?.height ?? 0, videoW: v.videoWidth, videoH: v.videoHeight });
  };

  const at = (ev: React.PointerEvent<HTMLDivElement>): [number, number] | null => {
    if (!dims || !dims.videoW) return null;
    const box = ev.currentTarget.getBoundingClientRect();
    return normalizedPoint(ev.clientX - box.left, ev.clientY - box.top, { width: box.width, height: box.height }, dims.videoW, dims.videoH);
  };

  const onDown = (ev: React.PointerEvent<HTMLDivElement>) => {
    const p = at(ev);
    if (!p) return;
    live.current = { points: [p], width: brush };
    ev.currentTarget.setPointerCapture(ev.pointerId);
    paint();
  };

  const onMove = (ev: React.PointerEvent<HTMLDivElement>) => {
    const s = live.current;
    const p = s ? at(ev) : null;
    if (!s || !p) return;
    const [lx, ly] = s.points[s.points.length - 1];
    if (Math.hypot(p[0] - lx, p[1] - ly) < 0.008) return; // keep strokes light: a point every ~1 % of the frame
    s.points.push(p);
    paint();
  };

  const onUp = () => {
    const s = live.current;
    live.current = null;
    if (s) onChange([...strokes, s]);
  };

  return (
    <div>
      <div ref={boxRef} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
        style={{ position: "relative", width: "100%", maxWidth: 360, aspectRatio: "9 / 16", background: "#000",
          cursor: "crosshair", touchAction: "none" }}>
        <video src={src} muted playsInline preload="auto" onLoadedData={onLoaded}
          style={{ width: "100%", height: "100%", objectFit: "contain", pointerEvents: "none" }} />
        <canvas ref={canvasRef} aria-label="your drawing" style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />
      </div>
      <div className="row" role="radiogroup" aria-label="Brush">
        {BRUSHES.map((b) => (
          <button key={b.label} role="radio" aria-checked={brush === b.width} className={brush === b.width ? "" : "secondary"}
            onClick={() => setBrush(b.width)}>{b.label}</button>
        ))}
        <button className="secondary" disabled={strokes.length === 0} onClick={() => onChange(strokes.slice(0, -1))}>Undo</button>
        <button className="secondary" disabled={strokes.length === 0} onClick={() => onChange([])}>Clear</button>
      </div>
    </div>
  );
}
