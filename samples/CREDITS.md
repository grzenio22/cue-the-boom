# Sample and media provenance

The samples combine original phone footage and voice recordings, licensed stock
sources, LTX-generated elements and scene renders, and procedural effects.
The screenshots and explanatory graphics in `../media/` show those assets and
the app; they are not all model-generated images. See [NOTICE.md](../NOTICE.md)
for the repository's licence exceptions.

## The author's footage

`plates/hand_palm.mp4`, `hand_point.mp4`, `hand_walk.mp4`, `hand_punch.mp4`,
`hand_fist.mp4` and `hand_open.mp4` are the author's own hand footage, filmed on
a phone for this project, with the author's voice where present. These original
recordings are supplied under the repository's licence. Their finished shots
are in the matching `made_ahead/bending/hand_*` directories.

The fire in those finished shots, and the explosion's fireball, comes from
LTX-2.5 with ainvfx-fluid. Lightning and water are procedural CPU effects; the
explosion combines the stored fireball with procedural blast effects. The
procedural lightning examples use synthetic cue sounds over the author's footage.

## The author's voice and the three performances

**One render. Three performances.** uses the same palm footage and the same
generated fire element for all three performances:

| Performance | Audio source | Finished shot |
|---|---|---|
| BOOM | The original recorded BOOM in the author's hand clip | `made_ahead/bending/hand_palm__boom__fire_burst` |
| pf, pf, BOOM | `voices/pf-pf-boom-author.wav`: edited from the author's recorded pf and BOOM sounds, with room tone from the hand take | `made_ahead/bending/hand_palm__pfpfboom__again` |
| pf pf pf PF | `voices/pf-x4-author.wav`: edited from the same recorded sounds and room tone into a different rhythm | `made_ahead/bending/hand_palm__pfx4__again` |

The two rhythm presets are assembled performances, rather than separate
continuous voice takes. Their `receipt.json` files point back to the BOOM shot,
record `no_new_render: true` and `no_gpu: true`, and name the same stored
LTX-2.5 + ainvfx-fluid element. A new performance re-composites that element on
the CPU; it never invokes the model or uses a GPU.
The audio notes in [samples.json](samples.json) record the edits too.

The other presets in `voices/` are synthetic. `tools/make_hand_noises.py` makes
the five presets for lightning, water, air and the explosion. Added Foley is
procedurally synthesized by `cuesync.sfx_synth`. The demo's narration is a
separate synthetic voice made with Kokoro-82M; it is not the author's cue voice.

## Generated stand-ins and elements

`plates/pumpkin_kitchen.mp4`, `mug_counter.mp4`, `toy_desk.mp4`,
`candle_shelf.mp4` and `jackolantern_kitchen.mp4` are LTX-2.5-generated stand-ins.
`plates/candle_snap.mp4` uses the candle stand-in with a synthetic snap in its
soundtrack. The element library in `library/` contains stored LTX renders;
generated outputs have the model terms described in [NOTICE.md](../NOTICE.md).

## Fire elements

The gallery's hand-fire shots and `library/bending` play stored LTX-2.5 22B (int8)
+ ainvfx-fluid renders; each shot's `receipt.json` names its element.
[source_assets/b-turbulent-v2](source_assets/b-turbulent-v2/README.md) preserves a
later generation (seed `2026102222`) with its request, ComfyUI history and guide,
and [derive_fire_element.py](../tools/derive_fire_element.py) reproduces its two
CPU derivatives from it. They were compared with the shipped elements and are kept
as preserved experiments; the gallery and the library do not use them. Hand fire
is a 2.5D approximation with tracked placement, hand segmentation and local
lighting; these samples do not establish a full 3D fluid or scene-lighting
simulation.

## Stock footage and photos

The `made_ahead/scene/real_*` shots and the renders in `timing_set/real/` were made on real stock video and
photos. Each one is processed (the effect rendered into it by LTX-2.5 with ainvfx-fluid); those source clips
themselves are not in this repository. The one exception is the sample hand: `plates/hand_cupped.mp4` is the
Pexels clip itself (at 720p), so a hand of your own can be tried with no camera.

| Shots | Source | Author | Licence |
|---|---|---|---|
| `made_ahead/scene/real_trex_photo__fwoooom` | [Toy T-rex on a wooden table (photo)](https://pixabay.com/photos/dino-dinosaur-toy-t-rex-648717/) | Efraimstochter | Pixabay Content License |
| `made_ahead/scene/real_lanterns__kaboom` | [Lit jack-o'-lanterns and a ghost figure on a table (4K)](https://pixabay.com/videos/halloween-pumpkin-thanksgiving-53930/) | omisido | Pixabay Content License |
| `made_ahead/scene/real_candle_stool__pf` | [A Burning Candle on a Wooden Stool](https://www.pexels.com/video/a-burning-candle-on-a-wooden-stool-6343575/) | KATRIN BOLOVTSOVA | Pexels License |
| `made_ahead/scene/real_firepit__fwoooom` | [Firewood Inside the Fire Pit](https://www.pexels.com/video/firewood-inside-the-fire-pit-7147124/) | Kindel Media | Pexels License |
| `made_ahead/scene/real_jack_photo__kaboom`, `timing_set/real/jack_photo__pf-pf-ka-boom` | [Jack-O-Lantern at night](https://commons.wikimedia.org/wiki/File:Jack-O-Lantern_at_night.jpg) | Alabama Extension | CC0 1.0 |
| `made_ahead/scene/real_dino_books__fwoooom` | [A Toy Dinosaur on a Stack of Books](https://www.pexels.com/video/a-toy-dinosaur-on-a-stack-of-books-8014678/) | Cup of Couple | Pexels License |
| `made_ahead/scene/real_cake_candle__pf` | [Close Up Video of Candle on a Cake](https://www.pexels.com/video/close-up-video-of-candle-on-a-cake-8015404/) | Cup of Couple | Pexels License |
| `made_ahead/scene/real_candles_glasses__fwoooom` | [Still Video of Candles in Glasses](https://www.pexels.com/video/still-video-of-candles-in-glasses-6343626/) | KATRIN BOLOVTSOVA | Pexels License |
| `plates/hand_cupped.mp4`, `made_ahead/bending/hand_cupped__kaboom__fire_burst` | [Close-Up Shot of Open Hands](https://www.pexels.com/video/close-up-shot-of-open-hands-7298370/) | Kindel Media | Pexels License |
| the demo's pointing-hand flamethrower (not shipped) | [Close up on Man in Suit Making Gesture](https://www.pexels.com/video/close-up-on-man-in-suit-making-gesture-9862181/) | Pexels | Pexels License |
| `made_ahead/scene/real_kuri_photo__fwoooom` | [Red kuri pumpkin centred on a counter (photo)](https://pixabay.com/photos/pumpkin-autumn-red-kuri-squash-1583767/) | sparparim | Pixabay Content License |
| `made_ahead/scene/real_pumpkins_tap__fwoooom` | [Two Pumpkins Placed Near a Glass Window](https://www.pexels.com/video/two-pumpkins-placed-near-a-glass-window-5490678/) | olia danilevich | Pexels License |
| `made_ahead/scene/real_jack_stacked__kaboom` | [Video of a Jack-O'-Lantern](https://www.pexels.com/video/video-of-a-jack-o-lantern-5604115/) | Yaroslav Shuraev | Pexels License |
| `timing_set/real/kettle__hissss` | [Steam Rising from Boiling Kettle](https://www.pexels.com/video/steam-rising-from-boiling-kettle-32001622/) | Dominik ZÃ­tka | Pexels License |
| `timing_set/real/table_pumpkin__fwoooom` | [Laying Down A Pumpkin Over The Wooden Table](https://www.pexels.com/video/laying-down-a-pumpkin-over-the-wooden-table-5692173/) | Monstera Production | Pexels License |
| `timing_set/real/cacti_pumpkin__fwoooom` | [A Pumpkin next to Potted Cacti and Succulents](https://www.pexels.com/video/a-pumpkin-next-to-potted-cacti-and-succulents-9894140/) | Nathan J Hilton | Pexels License |
| `timing_set/real/hot_drink__hissss` | [Close Up Video of a Hot Drink](https://www.pexels.com/video/close-up-video-of-a-hot-drink-6938926/) | cottonbro studio | Pexels License |
| `timing_set/real/candle_photo__pf` | [Candle Burning](https://commons.wikimedia.org/wiki/File:Candle_Burning.JPG) | Nanda93 | Public domain |
| `timing_set/real/mug_photo__hissss` | [Coffee steaming in a mug](https://commons.wikimedia.org/wiki/File:Coffee_steaming_in_a_mug.jpg) | PedanticNarrativeController | CC0 1.0 |
| `timing_set/real/dragon__fwoooom` | [A Dragon Figurine](https://www.pexels.com/video/a-dragon-figurine-8710886/) | RDNE Stock project | Pexels License |

## Real voices

The `made_ahead/scene/real_voice_*` shots are cued by recordings of real people's voices, cut and placed as
a performance (each shot's receipt lists its hits). The recordings are CC0 on Freesound; the plates are
credited above.

| Recording | Author | Licence | Shots |
|---|---|---|---|
| [Powf.wav](https://freesound.org/people/huw2k8/sounds/459936/) | huw2k8 | CC0 1.0 | `made_ahead/scene/real_voice_jack_huw2k8` |
| [MouthBooms.wav](https://freesound.org/people/Jedimichael/sounds/491216/) | Jedimichael | CC0 1.0 | `made_ahead/scene/real_voice_lanterns_call`, `made_ahead/scene/real_voice_jack_jedimichael` |
| [blowing out candle.wav](https://freesound.org/people/Reitanna/sounds/242867/) | Reitanna | CC0 1.0 | `made_ahead/scene/real_voice_candles_call` |
| [Various short blows/puffs of air from male mouth](https://freesound.org/people/rykenn/sounds/432170/) | rykenn | CC0 1.0 | `made_ahead/scene/real_voice_lanterns_call`, `made_ahead/scene/real_voice_candles_call` |
| [Boom (vocal)](https://freesound.org/people/Sadiquecat/sounds/802505/) | Sadiquecat | CC0 1.0 | `made_ahead/scene/real_voice_jack_sadiquecat` |
| ["Ka-Boom!" (a vocal; unfa's own recording)](https://freesound.org/people/unfa/sounds/245604/) | unfa | CC0 1.0 | `made_ahead/scene/real_voice_pumpkins_kaboom`, `made_ahead/scene/real_voice_jack_unfa`, `made_ahead/bending/hand_cupped__kaboom__fire_burst` |

The cupped-hand shot's original cue recording also ships as
`made_ahead/bending/hand_cupped__kaboom__fire_burst/source_voice.wav`; it is the
credited unfa recording, not a synthetic preset.


## Curated lightning shots

`made_ahead/bending/elements_water_air_20261024__dark__lightning_arc` and
`made_ahead/bending/elements_water_air_20261024__dark__lightning_ball` use the
author's prepared dim-room `hand_palm.mp4`, the synthetic `zzzzz-tsst.wav` cue
and procedural Foley. These are procedural CPU composites, without a new model
generation or outside stock element. Each shot records input and source hashes.

## Retained procedural renders and comparison media

Water, air and earth are experimental CPU effects under the collapsed **Experimental** disclosure; their materials and motion can still look synthetic.

Earlier procedural renders and the retained comparison media use the author's
bright `hand_open.mp4`, dim `hand_palm.mp4` and original phone recordings. Their
synthetic cues are `hoooo.wav`, `hoo-hoo-HOO.wav`, `shhhhh-sploosh.wav` and
`zzzzz-tsst.wav`. Earth shatter and projectile demonstrations use the credited
`pf-pf-boom-author.wav` edit; held stone uses the synthetic sustained cue.
Procedural Foley is mixed with the cues or original phone audio as labelled.

`media/stone-refinement.mp4` compares the earlier stone, CPU stone and a bounded
LTX-2.5 + ainvfx-fluid detail pass at the original 24 fps on both prepared phone
plates. `media/five-elements-original-takes.mp4` retains the original IMG_6668 and
IMG_6658 sound plus Foley. Its bright take starts one second into the recording;
its dim take is complete. Fire reuses the credited LTX asset, earth uses the
October 24 LTX generation with CPU detail transfer, and air, water and lightning
are procedural. These are recorded demonstrations; playback makes no new render.

`media/water-before-after.mp4` uses the full original IMG_6668 and IMG_6658 takes
with phone sound and Foley. `media/air-dark-before-after.mp4` uses 3.5 seconds per
form from the prepared dim take, with synthetic `hoooo.wav` and `hoo-hoo-HOO.wav`
cues and Foley. Both comparisons retain matching framing, tracking, cues and
seeds at 24 fps; their baseline air/water renderer is commit `22412d1`.

`media/water-evolution.mp4` compares the full original IMG_6668 with matching
tracking, cue and seed at 24 fps, using phone sound and the surge's Foley.
`media/earth-performance.mp4` starts with matched IMG_6668 before/after renders
(baseline `3121daf`), then shows IMG_6641 sweeps, IMG_6645 downward motion and
IMG_6663 drop/lift with a labelled scripted hold. Its final projectile comparison
uses the credited author rhythm. These are procedural CPU demonstrations.

`media/water-materials.mp4` and `media/earth-materials.mp4` use the full original
bright IMG_6668 with phone sound and Foley, or prepared dim `hand_palm.mp4` with
matched synthetic cues and Foley. The earth projectile examples use the author
rhythm. The retained `media/water-play-before-after.mp4`, `media/water-play-forms.mp4`
and `media/water-play-movements.mp4` are procedural CPU comparisons; the movement
reel labels its scripted hold. These videos play at normal speed and are not live
microphone tests.

Stone material sources remain credited because the code, earlier renders and
comparison media retain them: Poly Haven's CC0 **Rock Boulder Dry**, photographed
by Dimitrios Savva and processed by Rico Cilliers, and CC0 **Rock 07** by Jenelle
van Heerden. The material-pass examples use Rock 07 for exteriors and Rock Boulder
Dry for fracture interiors. Source links, bundled maps and checksums are recorded in [NOTICE.md](../NOTICE.md)
and the stone provenance JSON files in `cuesync/assets/stone/`.
