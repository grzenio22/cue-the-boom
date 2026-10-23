"""Reproduce a documented CPU fire derivative from its original LTX render.

    python tools/derive_fire_element.py --source original-element.mp4 \
        --recipe derivative-recipe.json --output reproduced-element.mp4

Install ./cuesync first. No model, network or GPU is used. The recipe selects
zero-based source frames and scales radiance in gamma-2.2 linear light. The
optional supported crown mask removes the lower stem without moving pixels.
RGB is encoded as limited-range BT.709 YUV420P using the recipe's libx264 CRF
and preset. A JSON sidecar records hashes, runtime versions and actual tags.

Byte identity is checked against the recipe, not assumed: encoder/library and
CPU threading differences can change compressed bytes across machines. A hash
mismatch leaves the output and report for inspection and returns exit code 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cuesync/src"))
from cuesync.media import read_video, write_video  # noqa: E402
from cuesync.transfer import gam, lin  # noqa: E402

CROWN_FORMULA = "(1 - smoothstep(edge, edge+140, y))^2; edge=340-.0025*(x-646)^2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crown_mask(height: int, width: int, spec: dict | None) -> np.ndarray | None:
    if spec is None:
        return None
    if not isinstance(spec, dict) or spec.get("formula") != CROWN_FORMULA:
        raise ValueError("Unsupported mask formula; only the documented soft crown is supported")
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    edge = 340 - .0025 * (xx - 646) ** 2
    # Keep the historical float32 operation order, including this denominator.
    t = np.clip((yy - edge) / ((edge + 140) - edge), 0, 1)
    return ((1 - t * t * (3 - 2 * t)) ** 2).astype(np.float32)


def derive_frames(source: np.ndarray, recipe: dict) -> np.ndarray:
    """Apply validated temporal mapping and optional feather; never mutate source."""
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8 or not len(source):
        raise ValueError("Source must be a nonempty uint8 N,H,W,3 array")
    mapping = recipe.get("time_mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("time_mapping must be nonempty")
    for key in ("frames", "derivative_frames"):
        if key in recipe and (type(recipe[key]) is not int or recipe[key] != len(mapping)):
            raise ValueError("Declared frame count disagrees with time_mapping")
    for index, item in enumerate(mapping):
        if not isinstance(item, dict) or type(item.get("frame")) is not int or item["frame"] != index:
            raise ValueError("Output frame indices must be contiguous and start at zero")
        source_index = item.get("source_frame")
        if type(source_index) is not int or not 0 <= source_index < len(source):
            raise ValueError(f"Invalid source_frame at output frame {index}")
        gain = item.get("linear_radiance_gain")
        if type(gain) not in (int, float) or not math.isfinite(gain) or gain < 0:
            raise ValueError(f"Invalid linear_radiance_gain at output frame {index}")
    mask = crown_mask(*source.shape[1:3], recipe.get("mask"))
    result = []
    for item in mapping:
        radiance = lin(source[item["source_frame"]])
        if mask is not None:
            radiance = radiance * mask[..., None]
        result.append(gam(radiance * item["linear_radiance_gain"]))
    return np.stack(result)


def reproduce(source_path: Path, recipe_path: Path, output_path: Path) -> dict:
    source_path, recipe_path, output_path = (Path(p).resolve() for p in (source_path, recipe_path, output_path))
    sidecar = output_path.with_suffix(".reproduction.json")
    mask_path = output_path.with_name(output_path.stem + "-mask.png")
    for destination in (output_path, sidecar, mask_path):
        for protected in (source_path, recipe_path):
            if destination == protected or (destination.exists() and destination.samefile(protected)):
                raise ValueError("Output must not overwrite the original source or recipe")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("--output must name an .mp4 file")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    source_hash = sha256(source_path)
    if source_hash != recipe.get("original_element_sha256"):
        raise ValueError("Original source hash does not match original_element_sha256 in the recipe")
    fps = recipe.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Recipe fps must be positive and finite")
    encoding = recipe.get("encoding", {})
    if encoding.get("codec") != "libx264":
        raise ValueError("Only libx264 encoding is supported")
    crf, preset = encoding.get("crf"), encoding.get("preset")
    if type(crf) is not int or not 0 <= crf <= 51:
        raise ValueError("Encoding CRF must be an integer from 0 to 51")
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
        raise ValueError("Unsupported x264 preset")
    source = read_video(source_path).frames
    derived = derive_frames(source, recipe)
    write_video(output_path, derived, fps=fps, crf=crf, preset=preset)
    with av.open(str(output_path)) as container:
        stream = container.streams.video[0]
        cc = stream.codec_context
        tags = {"colorspace": cc.colorspace, "color_primaries": cc.color_primaries,
                "color_trc": cc.color_trc, "color_range": cc.color_range}
        actual_fps = float(stream.average_rate)
        audio_streams = len(container.streams.audio)
    decoded = read_video(output_path).frames
    if decoded.shape != derived.shape or actual_fps != float(fps) or audio_streams or set(tags.values()) != {1}:
        raise ValueError("Encoded output failed frame, FPS, silent-stream or BT.709 limited-range validation")
    source_after = sha256(source_path)
    if source_after != source_hash:
        raise ValueError("Original source changed during reproduction")
    output_hash = sha256(output_path)
    expected_hash = recipe.get("derivative_sha256")
    byte_identical = output_hash == expected_hash if expected_hash else None
    report = {
        "kind": "CPU derivative reproduction; no new generation or spatial warping",
        "source": {"path": str(source_path), "sha256": source_hash, "unchanged": True},
        "recipe": {"path": str(recipe_path), "sha256": sha256(recipe_path)},
        "output": {"path": str(output_path), "sha256": output_hash,
                   "expected_sha256": expected_hash, "byte_identical_to_recipe": byte_identical,
                   "frames": len(decoded), "width": decoded.shape[2], "height": decoded.shape[1],
                   "fps": actual_fps, "audio_streams": audio_streams, "color_tags": tags,
                   "decoded_rgb_sha256": hashlib.sha256(decoded.tobytes()).hexdigest()},
        "encoding": {**encoding, "pixel_format": "yuv420p", "matrix": "BT.709", "range": "limited"},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pyav": av.__version__,
                    "opencv": cv2.__version__, "ffmpeg_libraries": av.library_versions},
        "implementation": {"script_sha256": sha256(Path(__file__)),
                           "media_sha256": sha256(ROOT / "cuesync/src/cuesync/media.py"),
                           "transfer_sha256": sha256(ROOT / "cuesync/src/cuesync/transfer.py")},
        "new_model_renders": 0, "gpu_used": False,
        "cross_machine_note": "x264/library builds and CPU-dependent encoder threading can change bytes. "
                              "A mismatch is not proof of pixel equality; compare decoded RGB against the reference.",
    }
    mask = crown_mask(*source.shape[1:3], recipe.get("mask"))
    if mask is not None:
        if not cv2.imwrite(str(mask_path), (mask * 255 + .5).astype(np.uint8)):
            raise ValueError("Could not write the diagnostic mask PNG")
        mask_hash = sha256(mask_path)
        report["mask"] = {"path": str(mask_path), "sha256": mask_hash,
                          "matches_recipe_png": mask_hash == recipe["mask"].get("sha256"),
                          "computation": "float32 formula; the quantized PNG is diagnostic only"}
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Untouched original LTX element MP4")
    parser.add_argument("--recipe", type=Path, required=True, help="Derivative recipe JSON with original source hash")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4; never the original source")
    args = parser.parse_args()
    try:
        report = reproduce(args.source, args.recipe, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 2 if report["output"]["byte_identical_to_recipe"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
