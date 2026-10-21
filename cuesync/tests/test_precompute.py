import json
from pathlib import Path

from cuesync.precompute import cache_key, copy_shot, sample_pairs


def test_a_copied_hand_shot_keeps_its_replay_inputs_and_poster(tmp_path):
    src, dst = tmp_path / "made", tmp_path / "job"
    src.mkdir()
    files = {"final.mp4": b"video", "receipt.json": b"{}", "hand_track.json": b"track",
             "hand_mask.npz": b"mask", "poster.jpg": b"poster"}
    for name, contents in files.items():
        (src / name).write_bytes(contents)

    copy_shot(src, dst)

    assert {p.name: p.read_bytes() for p in dst.iterdir()} == files


def test_every_shipped_index_key_still_matches_its_sample_pair():
    root = Path(__file__).resolve().parents[2] / "samples"
    meta = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    pairs = {(p["plate"], p["voice"], p["effect"]): p for p in sample_pairs(root, "scene")}
    stale = []
    for index_path in sorted((root / "made_ahead").glob("*/index.json")):
        for key, entry in json.loads(index_path.read_text(encoding="utf-8")).items():
            mode, plate, voice, effect = (entry[k] for k in ("mode", "plate", "voice", "effect"))
            pm = meta["plates"].get(plate, {})
            pair = pairs.get((plate, voice, effect), {}) if mode == "scene" else {}
            receipt = json.loads((index_path.parent / entry["dir"] / "receipt.json").read_text(encoding="utf-8"))
            point = pair.get("point", pm.get("point", receipt["inputs"]["point"]))
            fields = key.split("|")
            direction = pair.get("direction") if pair else (float(fields[6]) if len(fields) > 6 else None)
            subject = pm.get("subject", receipt["inputs"].get("subject"))
            seed = int(entry.get("seed", 1)) if mode == "bending" else 1
            expected = cache_key(mode, plate, voice, effect, tuple(point), seed, direction, subject)
            if expected != key:
                stale.append((key, expected))
    assert not stale


def test_every_scene_sample_pair_is_available_in_the_shipped_index():
    root = Path(__file__).resolve().parents[2] / "samples"
    index = json.loads((root / "made_ahead" / "scene" / "index.json").read_text(encoding="utf-8"))
    keys = [cache_key("scene", p["plate"], p["voice"], p["effect"], p["point"], 1,
                      p.get("direction"), p.get("subject")) for p in sample_pairs(root, "scene")]
    assert keys and all(key in index for key in keys)
