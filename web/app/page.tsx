import Gallery from "@/components/Gallery";
import Hook from "@/components/Hook";
import Again from "@/components/Again";
import Studio from "@/components/Studio";
import Hero from "@/components/Hero";
import Embers from "@/components/Embers";

export default function Home() {
  return (
    <>
      <Embers />
      <Hero />
      <main>
        <div className="studio-head">
          <p className="kicker">The studio · no GPU, no camera needed</p>
          <h2>Choose an element. Make the noise.</h2>
        </div>
        <div id="studio" className="studio-anchor"><Studio /></div>
        <Again />
        <Hook />
        <Gallery />
        <p className="muted credits">
          Fire, smoke and steam by LTX-2.5 (Lightricks) and the ainvfx-fluid IC-LoRA (AInVFX), on ComfyUI through the Comfy
          SDK; water, lightning and explosion made from your clip; hands followed by MediaPipe and RTMPose and matted
          by SAM 2.1. Open weights only.
        </p>
      </main>
    </>
  );
}
