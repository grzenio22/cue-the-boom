import Gallery from "@/components/Gallery";
import Studio from "@/components/Studio";

export default function Home() {
  return (
    <main>
      <p className="muted">Open Source Higgsfield · open weights · LTX-2.5 + ComfyUI</p>
      <h1>Cue the Boom</h1>
      <p className="lede">Draw where. Cue it with a sound. The scene reacts.</p>
      <p className="muted">
        Put fire, smoke or steam into your own shot exactly where it should come from (tap it, or draw it) and exactly
        when it should happen: say it (<em>pf.</em>, <em>KA-BOOM</em>, <em>fwoooOOOM</em>, <em>hissss</em>), lock it to a sound
        already in your clip (a snap, a click, a slam), or press and hold. How you cue it shapes it: when it erupts, how
        big it gets, when it stops. LTX-2.5 renders it into the scene, so the room lights up and the scene changes; a
        real video keeps its own pixels and gets only what changed.
      </p>
      <Studio />
      <Gallery />
    </main>
  );
}
