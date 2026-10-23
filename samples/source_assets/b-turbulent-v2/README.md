# B-turbulent-v2 source evidence

The original/ directory contains unchanged evidence from the earlier LTX-2.5 generation. Adoption and compositing use CPU only. Original generation time is historical.

compact/ is a temporal and radiance derivative. crown/ additionally applies the documented soft mask. Neither is a new model generation. The original and derivative hashes are separate in their recipes. Source playback adapters live with each gallery shot as source-adapter.json; current source_profile and motion are authored playback choices, not generation settings.

Reproduce from the repository root with the public helper:

```sh
python tools/derive_fire_element.py --source samples/source_assets/b-turbulent-v2/original/element.mp4 --recipe samples/source_assets/b-turbulent-v2/compact/derivative-recipe.json --output /your/output/compact.mp4
python tools/derive_fire_element.py --source samples/source_assets/b-turbulent-v2/original/element.mp4 --recipe samples/source_assets/b-turbulent-v2/crown/derivative-recipe.json --output /your/output/crown.mp4
```

The public helper reproduced both supplied elements byte-for-byte using PyAV/libx264 CRF12, preset fast, explicit limited-range BT.709. Encoder versions may change encoded bytes; retain and compare the decoded frames and the emitted reproduction report in that case.
