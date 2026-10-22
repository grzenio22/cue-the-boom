"""Refresh the existing contribution recap from production composites, without changing speech or edit timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from build_demo import file_sha256
from v11_cards import frame, portrait
from v13_contribution import PANEL_HEIGHT, PANEL_LEFTS, PANEL_TOP, PANEL_WIDTH
from v14_contribution import recap


def refresh(root: Path, run: Path) -> None:
    cards, clips = run / "cards", run / "clips"
    cards.mkdir(parents=True, exist_ok=True)
    prior_card = root / "runs/demo_v15/cards/pipeline.png"
    timing_path = root / "runs/demo_v15/edit_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    # Keep the approved type, source plate, tracking/matte panel and caption area.
    # Replace every fire picture; the final panel now shows the production look.
    image = Image.open(prior_card).convert("RGB")
    raw, result = clips / "how_raw.mp4", clips / "proof_boom/composite.mp4"
    image.paste(portrait(frame(raw, 70), .52, .42, PANEL_WIDTH, PANEL_HEIGHT, 1.3),
                (PANEL_LEFTS[2], PANEL_TOP))
    image.paste(portrait(frame(result, 70), .5, .48, PANEL_WIDTH, PANEL_HEIGHT),
                (PANEL_LEFTS[3], PANEL_TOP))
    card = cards / "pipeline.png"
    image.save(card)
    movie = recap(root, run, stages=timing["stages"], duration=timing["recap_duration"],
                  source_card=card, source_clip=raw)
    sources = [prior_card, timing_path, raw, result, Path(__file__).resolve(),
               Path(__file__).with_name("v14_contribution.py"), Path(__file__).with_name("v11_cards.py")]
    receipt = dict(method="Existing recap layout and timing; production fire pictures replaced; no narration generation",
                   no_new_render=True, no_gpu=True,
                   sources_sha256={str(p.resolve()): file_sha256(p) for p in sources},
                   outputs_sha256={str(p.resolve()): file_sha256(p) for p in [card, movie]})
    (cards / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(movie)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    refresh(args.root, args.run or args.root / "runs/demo_v16")
