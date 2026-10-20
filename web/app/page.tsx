import Gallery from "@/components/Gallery";
import Hook from "@/components/Hook";
import Again from "@/components/Again";
import Studio from "@/components/Studio";

export default function Home() {
  return (
    <main>
      <p className="muted">Open Source Higgsfield · open weights · LTX-2.5 + ComfyUI</p>
      <h1>Cue the Boom</h1>
      <p className="lede">Film your hand. Make the noise. Throw fire, lightning, water and air.</p>
      <Hook />
      <Again />
      <p className="muted">
        Tap your palm or your fingertips and make the noise: hum to charge an ember, <em>BOOM</em> to throw the fireball,
        a long <em>fffff</em> from pointed fingers for a flamethrower that whips through the air when you swing. The fire is
        rendered once and carried on your hand; <strong>Perform it again</strong> gives it a new rhythm with no new render.
        A buzzing <em>zzzzz</em> strikes lightning from your fingertips, a long <em>BOOOOM</em> with a punch sets off an
        explosion, and water and air bend in your open palm: these are made on this computer from your hand and your
        voice, no GPU (<strong>Choose it yourself</strong> under Make the shot).
        Or tap or draw anything in your shot (<em>pf.</em>, <em>KA-BOOM</em>, <em>fwoooOOOM</em>, <em>hissss</em>, a snap in
        the clip, or press and hold) and it is rendered into the scene, timed and sized by your cue.
      </p>
      <div id="studio" className="studio-anchor"><Studio /></div>
      <Gallery />
      <p className="muted" style={{ marginTop: 32 }}>
        Fire, smoke and steam by LTX-2.5 (Lightricks) and the ainvfx-fluid IC-LoRA (AInVFX), on ComfyUI through the Comfy
        SDK; lightning, water and air made procedurally from your clip; hands followed by MediaPipe and RTMPose and matted
        by SAM 2.1. Open weights only.
      </p>
    </main>
  );
}
