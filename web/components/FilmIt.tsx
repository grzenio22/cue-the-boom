"use client";

import { useEffect, useRef, useState } from "react";
import { AUDIO_CONSTRAINTS } from "@/lib/capture/constraints";
import { pickMime } from "@/lib/capture/pickMime";

const MAX_S = 8; // a take is a strike or two: the shot uses up to 10 s

/** Film your hand here (a laptop's webcam, a phone's back camera), its sound with it: the take is your clip, and
 * the noise you made is in it, ready to lock to. Needs a secure page (localhost, or HTTPS). */
export default function FilmIt({ onTake }: { onTake: (file: File, url: string) => void }) {
  const [open, setOpen] = useState(false);
  const [ready, setReady] = useState(false); // the camera is on
  const [facing, setFacing] = useState<"user" | "environment">("environment");
  const [error, setError] = useState<string | null>(null);
  const [recording, setRecording] = useState<number | null>(null); // seconds into the take
  const live = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const rec = useRef<MediaRecorder | null>(null);

  const stop = () => { stream.current?.getTracks().forEach((t) => t.stop()); stream.current = null; };
  useEffect(() => stop, []);
  useEffect(() => {
    if (!open) return;
    let gone = false;
    // 16:9 where the camera has it (left to itself a webcam gives 4:3, and the shot crops that to 16:9)
    navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, frameRate: { ideal: 30 }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: AUDIO_CONSTRAINTS })
      .then((s) => {
        if (gone) { s.getTracks().forEach((t) => t.stop()); return; }
        stop();
        stream.current = s;
        if (live.current) live.current.srcObject = s;
        setReady(true);
      }, (e: Error) => {
        if (!gone) setError(e.name === "NotAllowedError" ? "The camera or the microphone was not allowed." : e.message);
      });
    return () => { gone = true; };
  }, [open, facing]);
  useEffect(() => { // the take's length, and its end at MAX_S
    if (recording === null) return;
    if (recording >= MAX_S) { rec.current?.stop(); return; }
    const t = setTimeout(() => setRecording((r) => (r === null ? null : r + 0.1)), 100);
    return () => clearTimeout(t);
  }, [recording]);

  const begin = () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) { setError("No camera here: the page needs localhost or HTTPS."); return; }
    setReady(false);
    setOpen(true);
  };
  const close = () => { stop(); setReady(false); setOpen(false); };
  const record = () => {
    const s = stream.current;
    if (!s) return;
    const mime = pickMime((t) => MediaRecorder.isTypeSupported(t));
    const r = new MediaRecorder(s, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    r.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    r.onstop = () => {
      setRecording(null);
      const type = r.mimeType || mime || "video/webm";
      const file = new File(chunks, `my take.${type.includes("mp4") ? "mp4" : "webm"}`, { type });
      close();
      onTake(file, URL.createObjectURL(file));
    };
    rec.current = r;
    r.start(250);
    setRecording(0);
  };

  if (!open) {
    return (
      <>
        <button className="secondary" onClick={begin}>Film it here</button>
        {error && <span className="chip bad">{error}</span>}
      </>
    );
  }
  return (
    <div className="film">
      <video ref={live} autoPlay muted playsInline className={facing === "user" ? "mirror" : undefined} />
      <p className="muted">Hold your hand up in the light, the camera still; make the noise as you strike (BOOM, pf, a long
        fffff). Up to {MAX_S} s.</p>
      <div className="row">
        {recording === null ? (
          <>
            <button disabled={!ready} onClick={record}>Record</button>
            <button className="secondary" onClick={() => { setReady(false); setFacing(facing === "user" ? "environment" : "user"); }}>Switch camera</button>
            <button className="secondary" onClick={close}>Cancel</button>
          </>
        ) : (
          <>
            <button onClick={() => rec.current?.stop()}>Stop</button>
            <span className="chip">recording {recording.toFixed(1)} s</span>
          </>
        )}
      </div>
      {error && <p className="chip bad">{error}</p>}
    </div>
  );
}
