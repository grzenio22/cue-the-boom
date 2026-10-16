import Link from "next/link";

export default function Home() {
  return (
    <main>
      <p className="muted">Open Source Higgsfield · built on Comfy + LTX-2.5</p>
      <h1>Cue the Boom</h1>
      <p style={{ fontSize: "1.25rem" }}>Make the noise. Get the shot.</p>
      <p className="muted">
        Film a pumpkin, a mug or a toy, then perform the noise: <em>pf.</em>, <em>fwoooOOOM</em>, <em>KA-BOOM</em>.
        Your voice picks the effect and shapes it, and it lands on your exact frame in your own footage.
      </p>
      <div className="row">
        <Link className="btn" href="/dub">Dub Mode: perform over a clip</Link>
        <span className="btn secondary" aria-disabled="true">One-take (coming soon)</span>
      </div>
    </main>
  );
}
