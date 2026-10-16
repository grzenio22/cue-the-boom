"use client";

import { useEffect, useRef, useState } from "react";
import { contentRect, normalizedPoint } from "@/lib/geometry";

interface Props {
  src: string;
  value: [number, number] | null;
  onChange: (p: [number, number]) => void;
}

interface Dims { boxW: number; boxH: number; videoW: number; videoH: number }

/** Paused first frame of the plate; tap where the effect should come from. */
export default function TapPoint({ src, value, onChange }: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState<Dims | null>(null);

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

  const onClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    if (!dims || !dims.videoW) return;
    const box = ev.currentTarget.getBoundingClientRect();
    const p = normalizedPoint(ev.clientX - box.left, ev.clientY - box.top, { width: box.width, height: box.height }, dims.videoW, dims.videoH);
    if (p) onChange(p);
  };

  let marker: React.CSSProperties | null = null;
  if (value && dims && dims.videoW && dims.boxW) {
    const r = contentRect({ width: dims.boxW, height: dims.boxH }, dims.videoW, dims.videoH);
    marker = { left: r.x + value[0] * r.width, top: r.y + value[1] * r.height };
  }

  return (
    <div ref={boxRef} onClick={onClick} style={{ position: "relative", width: "100%", maxWidth: 360, aspectRatio: "9 / 16", background: "#000", cursor: "crosshair" }}>
      <video
        src={src}
        muted
        playsInline
        preload="auto"
        onLoadedData={onLoaded}
        style={{ width: "100%", height: "100%", objectFit: "contain", pointerEvents: "none" }}
      />
      {marker && (
        <div aria-label="emission point" style={{ position: "absolute", ...marker, width: 28, height: 28, marginLeft: -14, marginTop: -14,
          border: "2px solid #ffb020", borderRadius: "50%", boxShadow: "0 0 12px #ff8c00", pointerEvents: "none" }} />
      )}
    </div>
  );
}
