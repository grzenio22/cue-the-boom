import Studio from "@/components/Studio";

export default function Home() {
  return (
    <main>
      <p className="muted">Open Source Higgsfield · open weights · LTX-2.5 + ComfyUI</p>
      <h1>Cue the Boom</h1>
      <p className="lede">Make the noise. Get the shot.</p>
      <p className="muted">
        Pick a shot, tap where it happens, then perform the noise: <em>pf.</em>, <em>fwoooOOOM</em>, <em>pfffff</em>,{" "}
        <em>hissss</em>. Your voice picks fire, smoke or steam and shapes it: when it erupts, how big it gets and when it
        stops. Your footage is never regenerated; the effect is rendered on its own and composited onto it.
      </p>
      <Studio />
    </main>
  );
}
