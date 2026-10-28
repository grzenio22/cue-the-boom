import Gallery from "@/components/Gallery";
import Hook from "@/components/Hook";
import Again from "@/components/Again";
import Studio from "@/components/Studio";

export default function Home() {
  return (
    <main>
      <p className="muted">Open Source Higgsfield · open weights · LTX-2.5 + ComfyUI</p>
      <h1>Cue the <span className="boom">Boom</span></h1>
      <p className="lede">Your voice. Fire, air, explosion and lightning.</p>
      <div id="studio" className="studio-anchor"><Studio /></div>
      <Again />
      <Hook />
      <Gallery />
      <p className="muted" style={{ marginTop: 32 }}>
        Fire, smoke and steam by LTX-2.5 (Lightricks) and the ainvfx-fluid IC-LoRA (AInVFX), on ComfyUI through the Comfy
        SDK; lightning and explosion made from your clip; hands followed by MediaPipe and RTMPose and matted
        by SAM 2.1. Open weights only.
      </p>
    </main>
  );
}
