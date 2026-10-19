# tools

Measurement and demo tools. They read sample shots (in `samples/`) or the shots you make (in `runs/`, which is
not part of the repository).

| Tool | What it does |
|---|---|
| `measure_timing.py` | Where each hit shows on the raw render and in the finished shot, against its cue, from the onsets marked by eye in `timing_annotations.json` (a detector for unmarked shots). `python tools/measure_timing.py samples/made_ahead/scene samples/timing_set/before_fixes samples/timing_set/other_seeds samples/timing_set/dropped samples/timing_set/real` |
| `timing_annotations.json` | The eye marks: the first frame each hit is clearly there on the raw render, per shot. |
| `check_take_timing.py` | Replaces the studio's microphone with a synthetic one that makes a noise at a known frame, and checks the take lands there. Needs the studio and the render API running, and Playwright with a Chromium (`pip install playwright`). |
| `pack_made_ahead.py` | Packs shots made with `cuesync precompute` small enough to ship in `samples/made_ahead`. |
| `make_readme_images.py` | `media/hero.jpg` and `media/how-it-was-made.jpg` from the sample shots. |
| `demo/build_demo.py`, `demo/storyboard.json` | The demo video from a storyboard of clips, stills, titles and raw-render-vs-shot comparisons, with each shot's sound drawn under it. Its inputs are shots and screen recordings in `runs/`. |
| `demo/make_storyboard.py`, `demo/*_card.py` | The submitted cut's storyboard, and its cards (how it's made, the ComfyUI graph, the README's hands). Its footage is the author's takes, kept outside the repository. |
| `demo/record_studio.py` | Records the studio in use (a fake microphone, drawing, re-timing, your own clip with its hand followed; `--hd` for sharp frames) for the demo. Needs Playwright. |
