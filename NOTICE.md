# Licences and media provenance

The repository is licensed under the MIT License, with these exceptions and
third-party materials:

- `cuesync/` is licensed under the Apache License 2.0 (cuesync/LICENSE).
- `workflows/` is derived from the AInVFX ainvfx-fluid sample workflow and is
  licensed under the LTX-2.x Community License Agreement
  (workflows/LICENSE-LTX-2.x).

`samples/` and `media/` contain a mixture of original recordings, third-party
source media, generated outputs, procedural effects, screenshots and diagrams:

- The author's original hand footage and voice recordings are provided under
  the repository's licence. The two author-voice rhythm presets are edited from
  those recordings; most other preset noises and the added Foley are synthetic.
- LTX-2.5 with ainvfx-fluid generated the stored fire elements, the explosion's
  fireball, and the effects rendered into scene shots. Some sample plates are
  also LTX-2.5-generated stand-ins. See the LTX-2.x Community License for the
  terms applying to those model outputs.
- Lightning, water and experimental air effects are procedural. The explosion
  combines a stored LTX fireball with procedural blast effects. Compositing,
  lighting and other processing do not make the original recordings or these
  procedural components LTX-generated source media.
- Third-party stock clips, photos and voice recordings retain their own
  licences (Pexels License, Pixabay Content License, CC0 or public domain, as
  listed per source). Processed shots can contain both this source material and
  generated effects; `plates/hand_cupped.mp4` is the credited Pexels source clip.
- `media/` includes screenshots, explanatory graphics and frames from these
  mixed-source shots; it is not a collection consisting entirely of generated
  images.

See [samples/CREDITS.md](samples/CREDITS.md) for source attribution and audio
provenance. These notices distinguish the components; they do not replace the
applicable source or model licences.

Model weights are not part of this repository and carry their own licences;
see [build/model-files.md](build/model-files.md).
