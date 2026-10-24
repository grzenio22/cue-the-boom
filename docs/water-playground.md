# Water playground

Water is a deterministic CPU effect driven by the recorded voice, measured hand
track and hand matte. It does not require a GPU. The webcam preview currently
supports fire and lightning; these water forms appear in the finished shot.

## Playing

Choose **Water**, then a form. The sample clip is a fast way to try the voice
timing. Upload your own hand clip to control the gestures as well. Keep the camera
steady, leave space above your hand and hold a soft **shhh** while moving.

| Form | Held shape | Voice / hand action |
| --- | --- | --- |
| Orb | Rolling open current over the palm | Release the sound to spill it |
| Whip | Rising, flexible stream | Sploosh lashes the stream |
| Surge | Broad breaking curl | Sweep to cast a water blade |
| Ribbon | Spiralling water sheet | Sweep for a blade; sploosh for a pulse jet |
| Tidal shield | Curved water curtain | Lift for a rising wave; drop for a splash crown |
| Jet | Narrow rising streams | Repeated accents launch separate directed jets |

All forms respond to sustained, measured sweeps, lifts and downward strokes while
the voice holds. A gesture detector compensates for camera motion and rejects
tracking jumps, low-confidence positions and reacquisition. It measures 2D hand
motion, not real-world collisions or a reconstructed floor. Released sheets and
spray inherit their birth-frame position and continue independently of the hand.

## Material and rendering

`water_sheets.py` builds continuous triangle surfaces with moving folds, local
refraction, room reflections, advected froth and free-edge breakup. Small spray
drops supplement the sheets. The existing water renderer retains the whip,
released reservoirs, wet skin and palm runoff. A stationary hold keeps flowing.

The hand matte excludes skin from the light estimate before filtering. Surface
brightness follows the recorded room; there is no constant bright-light floor.
The hand's depth hides water that passes behind it. Refraction remains close to
the covered pixels because a single view does not reveal the background geometry.

This is a stylized water-bending effect, not a physically complete liquid
simulation. Broad transparent sheets can still look synthetic in isolated frames.
Extremely dark footage contains little reflected-light information. The clearest
results come from some side light and a visible hand against a darker background.

## Verification

`test_water_sheets.py` covers lighting isolation, moving held geometry, distinct
gesture responses, voice pulses, finite lifetimes, camera-relative casts,
occlusion, determinism and untouched background pixels. `test_water.py` covers
the existing optics and hand interaction. Render the production bright/dim suite:

```powershell
.\cuesync\.venv\Scripts\python.exe tools/render_element_suite.py `
  --out runs/my-water-review --width 1080 `
  --effects water_orb water_whip water_surge water_ribbon water_shield water_jet
```

The tool records inputs, source hashes, timings and complete video decode checks.
Recorded-motion demonstrations that use a scripted hold must be labelled as such;
they are not live microphone demonstrations.
