"""Install a verified render_element_suite batch into the local sample gallery.

This only changes checkout files. It does not publish, deploy or submit anything.
Requires a complete 1080p bright/dim suite and preserves the previous index beside
the private render evidence. Source hashes and every MP4 are checked before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess

from cuesync.precompute import cache_key
from render_element_suite import CASES, VOICES

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"air_ball": "Air vortex", "air_blast": "Air blast", "water_orb": "Water orb",
         "water_ribbon": "Water ribbon", "water_shield": "Tidal shield", "water_jet": "Water jet",
         "water_whip": "Water whip", "lightning_arc": "Lightning arcs", "lightning_ball": "Ball lightning",
         "earth_rock": "Living stone", "earth_shatter": "Stone shatter", "water_surge": "Water surge",
         "earth_volley": "Shard volley", "earth_lance": "Stone lance"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batch-name", default="elements_20261023")
    parser.add_argument("--effects",nargs="+",choices=NAMES,help="Explicit complete bright/dim subset to update")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.batch_name):
        raise SystemExit("batch-name must contain only letters, numbers, underscores or hyphens")
    source = args.source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    selected=args.effects or list(NAMES)
    wanted = {(case, effect) for case in ("bright", "dark") for effect in selected}
    actual = {(c["case"], c["effect"]) for c in manifest["cases"]}
    if manifest["width"] != 1080 or actual != wanted or len(manifest["cases"]) != len(wanted):
        raise SystemExit("A complete, unique 1080p bright/dim batch is required")
    if manifest.get("renderer_baseline_ref"):
        raise SystemExit("An old-code comparison must not replace the release gallery")
    for file, digest in manifest["source_sha256"].items():
        if sha(ROOT / file) != digest:
            raise SystemExit(f"Changed source: {file}; render again before installation")
    for item in manifest["cases"]:
        final = source / f'{item["case"]}-{item["effect"]}' / "final.mp4"
        if sha(final) != item["sha256"] or item["decode"] != "passed":
            raise SystemExit(f"Unverified render: {final}")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"], check=True)
    gallery = ROOT / "samples/made_ahead/bending"
    index_file = gallery / "index.json"
    backup = source / "gallery-index-before.json"
    if backup.exists():
        raise SystemExit("This evidence batch was already installed; use its saved index to review")
    shutil.copy2(index_file, backup)
    meta = json.loads((ROOT / "samples/samples.json").read_text(encoding="utf-8"))["plates"]
    index = json.loads(index_file.read_text(encoding="utf-8"))
    index = {k:v for k,v in index.items() if not (v["effect"] in selected and v["plate"] in ("hand_open.mp4", "hand_palm.mp4"))}
    installed = []
    for item in manifest["cases"]:
        case,effect = item["case"],item["effect"]
        name = f"{args.batch_name}__{case}__{effect}"
        target = gallery / name
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing gallery evidence: {target}")
        target.mkdir()
        shot = source / f"{case}-{effect}"
        for filename in ("composite.mp4", "final.mp4", "poster.jpg"):
            shutil.copy2(shot / filename, target / filename)
        if (shot / "effect_mask.npz").exists():
            shutil.copy2(shot / "effect_mask.npz", target / "effect_mask.npz")
        for filename in ("hand_track.json", "hand_mask.npz"):
            shutil.copy2(source / "inputs" / case / filename, target / filename)
        receipt = json.loads((shot / "receipt.json").read_text())
        for field,filename in (("track","hand_track.json"),("mask","hand_mask.npz")):
            receipt["bending"][field] = (target / filename).relative_to(ROOT).as_posix()
        (target / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        provenance = {"renderer": "procedural CPU", "cloud_spend_usd": 0, "seed": manifest["seed"],
                      "plate": f"samples/plates/{CASES[case][0]}", "plate_sha256": item["plate_sha256"],
                      "voice": f"samples/voices/{VOICES[effect]}", "voice_sha256": item["voice_sha256"],
                      "voice_note": manifest["voice_note"], "offset_s": item["offset_s"],
                      "source_sha256": manifest["source_sha256"],
                      "files": {f.name: sha(f) for f in target.iterdir() if f.is_file()}}
        (target / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        plate = CASES[case][0]
        key = cache_key("bending", plate, VOICES[effect], effect, tuple(meta[plate]["point"]), manifest["seed"])
        primary = case == ("dark" if effect.startswith("lightning") else "bright")
        index[key] = {"dir": name, "title": f'{NAMES[effect]} · {"bright room" if case == "bright" else "dim room"}',
                      "plate": plate, "voice": VOICES[effect], "effect": effect, "mode": "bending", "seed": manifest["seed"],
                      "tap_frame": meta[plate]["hand"]["frame"], "real": True,
                      "rank": -90 + list(NAMES).index(effect) + (0 if primary else 20)}
        installed.append(name)
    pending = index_file.with_suffix(".json.pending")
    pending.write_text(json.dumps(index, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    pending.replace(index_file)
    (source / "gallery-install.json").write_text(json.dumps({"installed": installed, "index_sha256": sha(index_file)},indent=2))
    print(f"Installed {len(installed)} verified shots into the local gallery; nothing published")


if __name__ == "__main__":
    main()
