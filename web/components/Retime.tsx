"use client";

import { useEffect, useRef, useState } from "react";
import { followJob, type JobEvent, type JobStatus } from "@/lib/shot";

const W = 600, H = 90, FPS = 24;

/**
 * The voice ribbon of a finished shot with the hit (and, for a flame or a jet of steam, the stop) as handles:
 * drag them and the shot is composited again from the same rendered elements, in seconds, no new render.
 */
export default function Retime({ job, stopsWithVoice, onRetimed }: {
  job: JobStatus;
  stopsWithVoice: boolean;
  onRetimed: (job: JobStatus) => void;
}) {
  const [env, setEnv] = useState<number[]>([]);
  const anchor = job.receipt?.guide.anchor_frame ?? 0;
  const voiceEnd = job.receipt?.voice.voice_end_frame ?? null;
  const [hit, setHit] = useState(anchor);
  const [stop, setStop] = useState<number | null>(voiceEnd);
  const [drag, setDrag] = useState<"hit" | "stop" | null>(null);
  const [busy, setBusy] = useState(false);
  const [event, setEvent] = useState<JobEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const url = job.files["cue.json"];
    if (url) fetch(url).then((r) => r.json()).then((c: { envelope: number[] }) => setEnv(c.envelope), () => setEnv([]));
  }, [job]);

  if (env.length === 0) return null;
  const n = env.length;
  const x = (f: number) => (f / Math.max(1, n - 1)) * W;
  const frameAt = (clientX: number) => {
    const r = svg.current!.getBoundingClientRect();
    return Math.round(Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * (n - 1));
  };
  const moved = hit !== anchor || (stopsWithVoice && stop !== voiceEnd);

  const apply = async () => {
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("shift", String(hit - anchor));
      if (stopsWithVoice && stop !== null && stop !== voiceEnd) form.append("stop", String(stop));
      const r = await fetch(`/api/jobs/${job.id}/retime`, { method: "POST", body: form });
      if (!r.ok) throw new Error((await r.text()) || `re-time failed (${r.status})`);
      const queued: JobStatus = await r.json();
      const done = await followJob(queued.id, setEvent);
      if (done.status === "done") onRetimed(done);
      else setError(done.error ?? "re-time failed");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="retime">
      <p className="muted">Re-time it: drag the <b>hit</b>{stopsWithVoice ? <> or the <b>stop</b></> : null} along your voice.
        It is composited again from the same element, no new render.</p>
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} className="ribbon" role="slider" aria-label="hit and stop on your voice"
        aria-valuenow={hit} onPointerMove={(e) => {
          if (!drag) return;
          const f = frameAt(e.clientX);
          if (drag === "hit") setHit(Math.min(Math.max(f, anchor - 48), anchor + 48));
          else setStop(Math.max(f, hit + 2));
        }} onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
        {env.map((v, i) => (
          <rect key={i} x={x(i)} y={H - 6 - v * (H - 24)} width={Math.max(1, W / n - 1)} height={v * (H - 24)} fill="#ff8c00" opacity={0.75} />
        ))}
        <g onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setDrag("hit"); }} style={{ cursor: "ew-resize" }}>
          <line x1={x(hit)} x2={x(hit)} y1={0} y2={H} stroke="#fff" strokeWidth={3} />
          <rect x={x(hit) - 18} y={0} width={36} height={16} rx={4} fill="#fff" />
          <text x={x(hit)} y={12} textAnchor="middle" fontSize={11} fill="#000">hit</text>
        </g>
        {stopsWithVoice && stop !== null && (
          <g onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setDrag("stop"); }} style={{ cursor: "ew-resize" }}>
            <line x1={x(stop)} x2={x(stop)} y1={0} y2={H} stroke="#5ad1ff" strokeWidth={3} />
            <rect x={x(stop) - 20} y={H - 16} width={40} height={16} rx={4} fill="#5ad1ff" />
            <text x={x(stop)} y={H - 4} textAnchor="middle" fontSize={11} fill="#000">stop</text>
          </g>
        )}
      </svg>
      <div className="row">
        <button disabled={!moved || busy} onClick={() => void apply()}>{busy ? (event?.message ?? "Re-timing") + "…" : "Re-time it"}</button>
        {moved && <span className="muted">hit {hit - anchor >= 0 ? "+" : ""}{Math.round(((hit - anchor) / FPS) * 1000)} ms
          {stopsWithVoice && stop !== null && stop !== voiceEnd ? `, stop at ${(stop / FPS).toFixed(2)} s` : ""}</span>}
        {moved && !busy && <button className="secondary" onClick={() => { setHit(anchor); setStop(voiceEnd); }}>Reset</button>}
      </div>
      {error && <p className="chip bad">{error}</p>}
    </div>
  );
}
